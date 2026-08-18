#!/usr/bin/env python3
"""NoLlama — OpenAI-compatible API server for Intel NPU / ARC GPU.

Auto-detects available devices (NPU, GPU, CPU) and model type (VLM/LLM).
NPU-first: works on any Intel Core Ultra laptop. ARC GPU optional.
Dual mode: NPU for chat + GPU for vision, simultaneously.

Usage:
    python nollama.py                                        # auto-detect device
    python nollama.py --device NPU                           # force NPU
    python nollama.py --device GPU                           # force GPU
    python nollama.py --gpu-model-dir gpu-model              # dual: NPU chat + GPU vision
    python nollama.py --model-dir ~/models/qwen3-14b-int4-ov --device GPU  # big LLM on GPU
    python nollama.py --whisper-dir whisper-model             # add speech-to-text
    python nollama.py --scan                                 # what models do I have?
"""

# Date-based, because semver invites arguments about whether a fix is "minor"
# and tells a user nothing about what code they actually have. A git checkout
# also reports its short commit (see _version_string below), so a bug report
# names the exact tree. Nothing parses this: the version VS Code's Ollama client
# validates is a separate faked one on /api/version.
__version_date__ = "2026-08-18"

import argparse
import base64
import hashlib
import io
import itertools
import json
import os
import re
import socket
import sys
import time


def _version_string():
    """"<date>" from a release ZIP, "<date>-<commit>" from a git checkout.

    Release ZIPs carry no .git, so they report the date alone. A checkout adds
    the short commit — including local commits ahead of the tag, which is the
    case where "0.10.0" would have been a lie.
    """
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=2)
        if r.returncode == 0 and r.stdout.strip():
            return f"{__version_date__}-{r.stdout.strip()}"
    except Exception:
        pass          # no git, no repo, or git is slow — the date alone is fine
    return __version_date__


__version__ = _version_string()
import threading
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from urllib.parse import unquote

# Silence OpenVINO's verbose property dump on model load (Model: OV Tokenizer
# + ~25 lines of NETWORK_NAME / NUM_STREAMS / INFERENCE_NUM_THREADS / ...).
# Must be set BEFORE openvino is imported.
os.environ.setdefault("OPENVINO_LOG_LEVEL", "0")

import numpy as np
import openvino as ov
import openvino_genai as ovg
from flask import Flask, Response, jsonify, request, render_template
from PIL import Image
from werkzeug.serving import ThreadedWSGIServer
try:
    import soundfile as sf
except ImportError:
    sf = None

# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def is_vlm(model_dir):
    """Detect if a model is a VLM.

    The definitive signal is structural: OpenVINO exports a VLM as a
    multi-component model with a separate vision encoder alongside the language
    model, whereas a text-only LLM is a single openvino_model.xml. Match *any*
    vision-encoder component file rather than one exact name — the filenames
    vary across generations (Qwen3.5 ships three:
    openvino_vision_embeddings_model.xml, ..._merger_model.xml, ..._pos_model.xml;
    LLaVA-style exports use image_encoder), and a single hard-coded name would
    miss a future variant. Architecture-name sniffing misses new generations
    too — e.g. Qwen3.5 reports Qwen3_5ForConditionalGeneration / qwen3_5,
    matching none of the keys below — so check the files first and fall back to
    the config keys.
    """
    try:
        for fn in os.listdir(model_dir):
            low = fn.lower()
            if low.endswith(".xml") and ("vision" in low or "image_encoder" in low):
                return True
    except OSError:
        pass
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        arch = cfg.get("architectures", [""])[0].lower()
        model_type = cfg.get("model_type", "").lower()
        return any(
            k in arch or k in model_type
            for k in ("vl", "vision", "llava", "qwen2vl", "internvl", "minicpm",
                      "multimodal", "image_text", "got_ocr")
        )
    except Exception:
        return False


# Architectures openvino_genai pipelines cannot run — nemotron_h is hybrid
# Mamba2+MoE with no GenAI path. Served through optimum-intel's python
# runtime (OptimumSlot) instead. muse_glimmer left this set on 2026-08-18:
# every Glimmer export in existence (Intel's official one and ours) is
# VLM-shaped, and VLMPipeline feeds its language model from embeddings by
# design — the "takes inputs_embeds" problem only ever ruled out
# LLMPipeline. So Glimmer is served as a plain VLM (verified on the 2026.4
# GenAI nightly, B60 2026-08-18, ~14 tok/s vs 8-11 on optimum). If an
# LLM-shaped Glimmer export ever appears, put it back; meanwhile
# --backend optimum still forces the python runtime for any model.
NEEDS_OPTIMUM = {"nemotron_h"}


def needs_optimum(model_dir):
    """True when this model must run on the optimum-intel python backend.

    Reads the *top-level* model_type deliberately (not _text_config): for a
    VLM-shaped export, the top-level type is the one that names the
    architecture GenAI would have to support.
    """
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            return json.load(f).get("model_type", "").lower() in NEEDS_OPTIMUM
    except Exception:
        return False


# Suffixes describing the *export* rather than the model, dropped from the
# display name (and so from the model ID clients configure).
_NAME_SUFFIXES = ("-ov", "-openvino", "-int8", "-int4")

# Directory names carrying no information. install.ps1 links model/ -> the
# real model directory, so one of these means "look through the link".
# Deliberately excludes "whisper-model": it's a real directory name people
# download into, and treating it as generic would rename that model's API ID
# from "whisper-model" to "whisper" for no gain.
_GENERIC_DIR_NAMES = ("model", "models", "gpu-model", "npu-model", "")


def _strip_name_suffixes(name):
    for suffix in _NAME_SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name


def resolve_display_name(model_dir):
    """Return (name, why) for the name NoLlama shows and clients request.

    The directory name as given wins. Only when it carries no information
    (install.ps1 links a generic model/ at the real directory) do we follow
    the link to find a real name. Resolving symlinks unconditionally — what
    this did before #19 — silently threw away a deliberate rename, so
    renaming a model folder appeared to have no effect at all. Renaming the
    directory is the one naming interface that needs no documentation, so it
    has to work.
    """
    given = _strip_name_suffixes(
        os.path.basename(os.path.normpath(os.path.abspath(model_dir))))
    if given.lower() not in _GENERIC_DIR_NAMES:
        return given, "directory name"

    target = _strip_name_suffixes(
        os.path.basename(os.path.normpath(os.path.realpath(model_dir))))
    if target.lower() not in _GENERIC_DIR_NAMES:
        return target, "link target (directory name is generic)"

    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            model_type = json.load(f).get("model_type")
        if model_type:
            return model_type, "config.json model_type (no usable directory name)"
    except Exception:
        pass
    return "unknown", "nothing identifiable found"


def model_display_name(model_dir):
    """Human-readable model name — see resolve_display_name()."""
    return resolve_display_name(model_dir)[0]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(url_or_data, max_dim):
    """Load an image from a base64 data URI or file:// URI."""
    if url_or_data.startswith("data:"):
        header, b64data = url_or_data.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif url_or_data.startswith("file:///"):
        path = unquote(url_or_data[8:])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        return Image.open(path).convert("RGB")
    else:
        raise ValueError(
            f"Unsupported image URL scheme. Use data:image/...;base64,... "
            f"or file:///path. Got: {url_or_data[:80]}"
        )


def pil_to_tensor(img, max_dim):
    """Convert PIL Image to OpenVINO Tensor (NHWC uint8)."""
    if max(img.width, img.height) > max_dim:
        ratio = max_dim / max(img.width, img.height)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS
        )
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None, ...])
    return ov.Tensor(arr)


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

# Reasoning we emitted to the client (<think> blocks) comes back verbatim in
# the next turn's history — clients echo assistant content as-is. It must not
# reach the model again: Muse Glimmer reasons on a separate ATEM channel, so
# <think> text inside an assistant-to-user turn reads as a corrupted transcript
# (observed: the model calls a clean follow-up question "garbled nonsense"),
# and for every model it burns prompt tokens on stale reasoning. Leading block
# only — that's where our stream filter puts it; anything later is quoted text.
_LEADING_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _strip_history_think(role, text):
    if role == "assistant":
        return _LEADING_THINK_RE.sub("", text, count=1)
    return text


def parse_messages(messages, max_dim):
    """Parse OpenAI messages. Returns (text_prompt, images, raw_messages)."""
    text_parts = []
    images = []
    raw_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        # `or ""`: OpenAI clients legally send "content": null (assistant
        # turns that carried only tool_calls — Zed does this); .get's default
        # covers a missing key, not an explicit null, and iterating None was
        # a 500 (issue #24 report, 'NoneType' surfaced in Zed).
        content = msg.get("content") or ""

        if isinstance(content, str):
            content = _strip_history_think(role, content)
            text_parts.append(content)
            raw_messages.append({"role": role, "content": content})
            continue

        msg_text = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                msg_text.append(block.get("text", ""))
            elif btype == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if url:
                    img = load_image(url, max_dim)
                    images.append(pil_to_tensor(img, max_dim))
                    # Anchor the image to ITS turn in the flattened prompt —
                    # without the tag, genai clusters all images before the
                    # prompt and the model answers about image #1 regardless
                    # of which turn asked the question.
                    msg_text.append(f"<ov_genai_image_{len(images) - 1}>")

        joined = _strip_history_think(role, " ".join(msg_text))
        text_parts.append(joined)
        raw_messages.append({"role": role, "content": joined})

    return "\n".join(text_parts), images, raw_messages


# ---------------------------------------------------------------------------
# Memory preflight — warn (never block) when a model won't fit its device
# ---------------------------------------------------------------------------

def _system_ram_bytes():
    """Total physical RAM in bytes. None if it can't be determined."""
    try:
        if os.name == "nt":
            import ctypes
            class _MemStatus(ctypes.Structure):
                _fields_ = ([("dwLength", ctypes.c_ulong),
                             ("dwMemoryLoad", ctypes.c_ulong)] +
                            [(n, ctypes.c_ulonglong) for n in (
                                "ullTotalPhys", "ullAvailPhys",
                                "ullTotalPageFile", "ullAvailPageFile",
                                "ullTotalVirtual", "ullAvailVirtual",
                                "ullAvailExtendedVirtual")])
            st = _MemStatus(dwLength=ctypes.sizeof(_MemStatus))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _device_mem_bytes(device_name, device_id):
    """Memory budget for a device, in bytes. None if unknown.

    GPU: ask the driver — on Windows iGPUs the reported figure already
    reflects the OS shared-memory policy (default ~half of RAM) and Intel's
    "Shared GPU Memory Override" driver setting, so it is the real budget,
    not a guess. CPU: total system RAM.
    """
    if device_name == "GPU":
        try:
            return int(ov.Core().get_property(device_id, "GPU_DEVICE_TOTAL_MEM_SIZE"))
        except Exception:
            return None
    if device_name == "CPU":
        return _system_ram_bytes()
    return None


# Compiled-model caches OpenVINO writes INSIDE the model dir on first GPU
# load (optimum-intel defaults CACHE_DIR to <model>/model_cache). The .blob
# mirrors the weights, so counting it doubles the apparent model size and
# turns the memory preflight into a false alarm on every run after the first.
_CACHE_DIR_NAMES = {"model_cache", ".cache"}


def _dir_size_bytes(model_dir):
    """Size of a model directory excluding cache subdirs (≈ weight bytes).
    None on failure."""
    try:
        total = 0
        for root, dirs, files in os.walk(model_dir):
            dirs[:] = [d for d in dirs if d not in _CACHE_DIR_NAMES]
            for fn in files:
                total += os.path.getsize(os.path.join(root, fn))
        return total or None
    except OSError:
        return None


def _verify_weights_integrity(model_dir):
    """Byte-exact completeness check. The IR .xml records each weight blob's
    offset+size into the .bin, so max(offset+size) is the exact minimum byte
    count the .bin must have — catches a download/copy that lost even the
    last 8 bytes (the IR carries no checksum, so corruption-in-place is out
    of scope; truncation is the realistic failure). Returns an error string,
    or None when intact / not checkable.

    Checks **every** IR .xml in the directory. Two reasons this isn't the two
    hardcoded names it used to be:

    - A modern VLM export is several IRs, not one: Qwen3.6-35B-A3B ships
      language_model + text_embeddings + vision_embeddings(+_merger/_pos),
      and any single one of them can arrive short.
    - A .bin that is missing *entirely* used to be skipped rather than
      reported, so it read as "weights complete". Found the hard way: a
      17 GB openvino_language_model.bin whose transfer died left a directory
      that passed the check with 4 GB of the 18 GB present.

    An .xml declaring no weight blobs needs no .bin, so absence is only an
    error when the graph actually references weights.
    """
    try:
        names = sorted(f for f in os.listdir(model_dir) if f.endswith(".xml"))
    except OSError:
        return None
    for name in names:
        base = name[:-4]
        binf = os.path.join(model_dir, base + ".bin")
        try:
            with open(os.path.join(model_dir, name), "rb") as f:
                data = f.read()
            need = max((int(m.group(1)) + int(m.group(2)) for m in
                        re.finditer(rb'offset="(\d+)" size="(\d+)"', data)),
                       default=0)
        except (OSError, ValueError):
            continue
        if not need:
            continue
        if not os.path.isfile(binf):
            return (f"{base}.bin is missing, but {name} references "
                    f"{need:,} bytes of weights. Incomplete download or copy "
                    f"— delete the model directory and re-fetch it "
                    f"(install.ps1 / download-model.ps1).")
        have = os.path.getsize(binf)
        if have < need:
            return (f"{base}.bin is truncated: {have:,} bytes on disk but "
                    f"{name} expects at least {need:,}. Incomplete "
                    f"download or copy — delete the model directory and "
                    f"re-fetch it (install.ps1 / download-model.ps1).")
    return None


def _text_config(model_dir):
    """config.json for the language model — VLMs nest it under text_config."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    nested = cfg.get("text_config")
    return nested if isinstance(nested, dict) else cfg


def _kv_bytes_per_token(model_dir):
    """KV-cache bytes per token from config.json geometry (K+V, fp16).

    E.g. Qwen2.5-Coder-7B (28 layers x 4 KV heads x 128 head-dim) ≈ 57 KB/tok;
    Qwen3-Coder-30B ≈ 96 KB/tok. None when the geometry can't be read.

    VLM configs nest the language model under "text_config" (the top level
    holds only the vision/text split), so read through that when present —
    otherwise every VLM silently skipped the KV half of the preflight.
    """
    try:
        cfg = _text_config(model_dir)
        layers = cfg["num_hidden_layers"]
        heads = cfg["num_attention_heads"]
        kv_heads = cfg.get("num_key_value_heads") or heads
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
        return 2 * layers * kv_heads * head_dim * 2
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model introspection (--scan)
# ---------------------------------------------------------------------------

def _flatten_rt_info(elem, prefix=""):
    """Flatten <a><b value="x"/></a> into {"a/b": "x"}."""
    out = {}
    for child in elem:
        key = prefix + child.tag
        if child.get("value") is not None:
            out[key] = child.get("value")
        out.update(_flatten_rt_info(child, key + "/"))
    return out


def read_ir_rt_info(model_dir):
    """Model-level <rt_info> from the IR .xml — the authoritative record of
    how a model was exported: nncf weight-compression mode and group size,
    plus the OpenVINO / optimum-intel / transformers versions that built it.
    Believe this over the directory name, which can say anything.

    Read from the tail of the file. The .xml holds the graph (tens of MB on a
    large model) and the model-level block is the last <rt_info> in it —
    per-node ones live inside <layers>, which precedes <edges> and the
    trailing block.
    """
    for base in ("openvino_model", "openvino_language_model"):
        xml = os.path.join(model_dir, base + ".xml")
        if not os.path.isfile(xml):
            continue
        try:
            with open(xml, "rb") as f:
                f.seek(max(0, os.path.getsize(xml) - 262144))
                tail = f.read()
            start = tail.rfind(b"<rt_info>")
            end = tail.rfind(b"</rt_info>")
            if start == -1 or end <= start:
                return {}
            fragment = tail[start:end + len(b"</rt_info>")]
            return _flatten_rt_info(ET.fromstring(fragment))
        except (OSError, ET.ParseError):
            return {}
    return {}


def weight_precision(model_dir, rt=None):
    """Human-readable weight precision, read from the IR's own nncf record.

    A directory called -int4-ov can contain anything; this is what the
    weights actually are.
    """
    rt = read_ir_rt_info(model_dir) if rt is None else rt
    mode = rt.get("nncf/weight_compression/mode")
    if not mode:
        try:
            with open(os.path.join(model_dir, "config.json")) as f:
                cfg = json.load(f)
            dtype = cfg.get("dtype") or cfg.get("torch_dtype")
            return f"{dtype} (weights not compressed)" if dtype else "unknown"
        except Exception:
            return "unknown"

    bits = ("INT4" if "int4" in mode else
            "INT8" if "int8" in mode else
            "FP8" if "f8" in mode or "fp8" in mode else mode)
    detail = []
    if mode.endswith("_sym"):
        detail.append("symmetric")
    elif mode.endswith("_asym"):
        detail.append("asymmetric")
    group_size = rt.get("nncf/weight_compression/group_size")
    if group_size == "-1":
        detail.append("channel-wise")
    elif group_size:
        detail.append(f"group size {group_size}")
    label = bits + (f" ({', '.join(detail)})" if detail else "")

    ratio = rt.get("nncf/weight_compression/ratio")
    try:
        # ratio < 1 means only that fraction of layers got the low-bit
        # treatment and the rest fell back to backup_mode — a mixed model
        # that a folder name would report as plain "int4".
        if ratio and float(ratio) < 1.0:
            backup = rt.get("nncf/weight_compression/backup_mode", "int8")
            label += f", {float(ratio) * 100:.0f}% of layers (rest {backup})"
    except ValueError:
        pass
    if rt.get("nncf/weight_compression/awq") == "True":
        label += " +AWQ"
    if rt.get("nncf/weight_compression/scale_estimation") == "True":
        label += " +scale-estimation"
    return label


def describe_model(model_dir):
    """Everything NoLlama can determine about a model directory from its own
    files, with no user input.

    The one thing the files do NOT record is the variant: config.json has no
    _name_or_path, and e.g. Qwen3-Coder-Next and Qwen3-Next-Instruct are
    identical in architecture and geometry. So the directory name stays the
    carrier of the variant — which is why resolve_display_name() must respect
    a rename (#19).
    """
    name, why = resolve_display_name(model_dir)
    rt = read_ir_rt_info(model_dir)
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    try:
        # Geometry lives under text_config on a VLM.
        geo = _text_config(model_dir)
    except Exception:
        geo = cfg

    if cfg.get("model_type") == "whisper":
        kind = "Whisper (speech-to-text)"
    elif is_vlm(model_dir):
        kind = "VLM (vision + text)"
    else:
        kind = "LLM (text)"
    if needs_optimum(model_dir):
        kind += ", served text-only via optimum backend"

    return {
        "path": os.path.abspath(model_dir),
        "name": name,
        "name_source": why,
        "kind": kind,
        "backend": "optimum-intel (python runtime)" if needs_optimum(model_dir)
                   else "openvino_genai",
        "architecture": (cfg.get("architectures") or [None])[0],
        "model_type": cfg.get("model_type"),
        "layers": geo.get("num_hidden_layers"),
        "context": geo.get("max_position_embeddings"),
        "experts": geo.get("num_experts") or geo.get("num_local_experts"),
        "experts_active": geo.get("num_experts_per_tok"),
        "precision": weight_precision(model_dir, rt),
        "size_bytes": _dir_size_bytes(model_dir),
        "kv_per_token": _kv_bytes_per_token(model_dir),
        "integrity": _verify_weights_integrity(model_dir),
        "openvino_version": rt.get("Runtime_version"),
        "optimum_intel_version": rt.get("optimum/optimum_intel_version"),
        "transformers_version": rt.get("optimum/transformers_version"),
    }


def _is_model_dir(path):
    return any(os.path.isfile(os.path.join(path, f)) for f in
               ("openvino_model.xml", "openvino_language_model.xml",
                "openvino_encoder_model.xml"))


def _model_dirs_under(path, depth):
    """Model directories at or below `path`, searching `depth` levels down."""
    if not os.path.isdir(path):
        return []
    if _is_model_dir(path):
        return [path]
    if depth <= 0:
        return []
    found = []
    try:
        for entry in sorted(os.listdir(path)):
            sub = os.path.join(path, entry)
            if os.path.isdir(sub) and not entry.startswith("."):
                found.extend(_model_dirs_under(sub, depth - 1))
    except OSError:
        pass
    return found


def scan_models(paths):
    """Print what NoLlama actually sees in each model directory.

    The alternative to this was a --model-name override flag, which needs
    knowledge a user shouldn't have to have (#19). This answers "what have I
    got, and what will it be called?" from the files on disk, so the answer
    comes from the machine rather than from documentation.
    """
    searched = [os.path.normpath(os.path.expanduser(p)) for p in
                (paths or [SCRIPT_DIR, "~/models"])]
    # One model reached by several paths (install.ps1 links model/ at a
    # directory in ~/models) is one model — report it once, listing the
    # aliases, rather than twice as if there were two copies.
    dirs, aliases = [], {}
    for path in searched:
        for d in _model_dirs_under(path, depth=2):
            real = os.path.realpath(d)
            if real in aliases:
                if d not in aliases[real]:
                    aliases[real].append(d)
                continue
            aliases[real] = []
            dirs.append(d)

    print("  NoLlama model scan\n")
    if not dirs:
        print("  No OpenVINO models found in:")
        for path in searched:
            print(f"    {path}")
        print("\n  A model directory is one holding openvino_model.xml + .bin.")
        print("  Fetch one with:  .\\download-model.ps1 <hf-repo-id>")
        return

    for directory in dirs:
        info = describe_model(directory)
        print(f"  {info['path']}")
        for alias in aliases.get(os.path.realpath(directory), []):
            print(f"    (also reachable as {alias})")
        print(f"    Name in API/UI : {info['name']}"
              f"      (from {info['name_source']})")
        print(f"    Kind           : {info['kind']}")
        arch = info["architecture"] or info["model_type"] or "unknown"
        if info["model_type"] and info["architecture"]:
            arch += f" / {info['model_type']}"
        print(f"    Architecture   : {arch}")
        if info["backend"] != "openvino_genai":
            note = ""
            import importlib.util
            if importlib.util.find_spec("optimum") is None:
                note = ("  — NOT importable in this python; serve from the "
                        "model-lab venv (see README)")
            print(f"    Backend        : {info['backend']}{note}")
        print(f"    Weights        : {info['precision']}", end="")
        if info["size_bytes"]:
            print(f"   {info['size_bytes'] / (1 << 30):,.1f} GB on disk")
        else:
            print()
        if info["experts"]:
            active = info["experts_active"] or "?"
            print(f"    MoE            : {info['experts']} experts, "
                  f"{active} active per token")
        geometry = []
        if info["layers"]:
            geometry.append(f"{info['layers']} layers")
        if info["context"]:
            geometry.append(f"{info['context']:,}-token context")
        if info["kv_per_token"]:
            geometry.append(f"{info['kv_per_token'] / 1024:,.0f} KB/token KV")
        if geometry:
            print(f"    Geometry       : {', '.join(geometry)}")
        built = [v for v in (
            f"OpenVINO {info['openvino_version']}" if info["openvino_version"] else None,
            f"optimum-intel {info['optimum_intel_version']}" if info["optimum_intel_version"] else None,
            f"transformers {info['transformers_version']}" if info["transformers_version"] else None,
        ) if v]
        if built:
            print(f"    Exported with  : {', '.join(built)}")
        if info["kind"].startswith("LLM"):
            print(f"    Agent mode     : tool calling on GPU/CPU; never on NPU "
                  f"(hard prompt cap)")
        if info["integrity"]:
            print(f"    PROBLEM        : {info['integrity']}")
        else:
            print(f"    Integrity      : weights complete")
        print()

    print("  To change the name shown in the UI and requested by clients,")
    print("  rename the model directory — that name is what NoLlama uses.")


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------

def extract_text(result):
    """Extract text from an openvino_genai generate result."""
    if isinstance(result, str):
        return result.strip()
    for attr in ("texts", "text", "output_text", "response"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if isinstance(val, (list, tuple)):
                return val[0].strip() if val else ""
            return val.strip()
    return str(result).strip()


def extract_perf(result):
    """Pull (ttft_ms, gen_ms) off a genai result. Returns (None, None) when the
    build or pipeline doesn't provide perf_metrics — never raises.
    """
    pm = getattr(result, "perf_metrics", None)
    if pm is None:
        return None, None
    try:
        return pm.get_ttft().mean, pm.get_generate_duration().mean
    except Exception:
        return None, None


def explain_genai_error(e, slot=None):
    """Map opaque OpenVINO GenAI runtime errors to actionable messages.
    Pass the serving slot when available — some hints cite its config."""
    msg = str(e)
    if "unfinished GenerationStatus" in msg:
        # Continuous-batching scheduler couldn't fit the sequence in the KV
        # pool (seen with 30B-class models + big agent prompts, issue #21).
        pool = getattr(slot, "kv_pool_gb", 0) or PROMPT_CACHE_GB
        cur = f" (currently {pool} GB)" if pool else ""
        return (f"{msg} — likely the KV-cache pool is too small for this "
                f"prompt: raise --cache-size-gb{cur}")
    if "dynamic shape" in msg and "[GPU]" in msg:
        # GPU plugin limitation, not a loading conflict: some optimum-exported
        # graphs (e.g. Muse Glimmer's inputs_embeds LM) keep shapes dynamic in
        # ways the GPU plugin can't execute. Observed on Xe-LPG (285K iGPU).
        return (f"{msg} — the OpenVINO GPU plugin cannot run this model's "
                f"dynamic-shape graph on this GPU; serve it with --device CPU")
    if "Compilation failed" in msg and ("NPU" in msg or "ZE_RESULT" in msg or "vpux" in msg):
        # NPU (vpux) compiler rejected the model — a model/driver-combination
        # problem, not a busy device (issue #20). Known trigger: an INT4
        # node-naming bug in older compilers (openvino#29823); also models
        # beyond the NPU envelope (>8B params).
        return (f"{msg} — the NPU compiler could not compile this model. "
                f"Usual causes: NPU driver too old for this model's INT4 "
                f"layout (update the Intel NPU driver; on Linux the "
                f"intel-npu-driver + compiler versions must match), or the "
                f"model is beyond the NPU envelope (proven NPU models are "
                f"INT4-CW, 8B params or less). Try an NPU model from the "
                f"install menu, or run this model on GPU/CPU instead.")
    if "Could not find a model in the directory" in msg:
        # read_model() found neither openvino_model.xml nor
        # openvino_language_model.xml — usually an interrupted download that
        # left the big .bin without its .xml descriptor (issue #17).
        return (f"{msg} — the directory has no openvino_model.xml / "
                f"openvino_language_model.xml. Incomplete download or "
                f"conversion? If the directory is a link, check the link "
                f"target's contents; re-run install.ps1 or download-model.ps1 "
                f"to repair.")
    return msg


# ---------------------------------------------------------------------------
# Tool calling (function calling) — Qwen3-Coder native format
#
# VS Code Copilot Chat sends tool definitions in the request `tools` array and
# expects structured `tool_calls` back. OpenVINO GenAI applies the model's chat
# template but gives us no hook to pass `tools` through it, so we render the
# specs into a system prompt ourselves (the way Qwen3-Coder was trained) and
# parse the model's emitted calls back into OpenAI shape.
#
# Qwen3-Coder emits calls as:
#   <tool_call>
#   <function=NAME>
#   <parameter=KEY>
#   VALUE
#   </parameter>
#   </function>
#   </tool_call>
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)
# Muse Glimmer ATEM protocol (see the model's chat_template.jinja)
_ATEM_BLOCK_RE = re.compile(
    r"<atem:function_calls>\s*(.*?)\s*</atem:function_calls>", re.DOTALL)
_ATEM_INVOKE_RE = re.compile(
    r'<atem:invoke name="([^"]+)"\s*>(.*?)</atem:invoke>', re.DOTALL)
_ATEM_PARAM_RE = re.compile(
    r'<atem:parameter name="([^"]+)"\s*>(.*?)</atem:parameter>', re.DOTALL)

# Other model families emit their own native tool-call syntax. A small model
# often ignores our Qwen3-Coder system prompt and falls back to whatever it was
# trained on, so we recognize those too:
#   Mistral   : [TOOL_CALLS][{"name": ..., "arguments": {...}}, ...]
#   Llama 3.x : <|python_tag|>{"name": ..., "parameters": {...}}  (';'-separated)
#   DeepSeek  : <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
#               ```json\n{...}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
_MISTRAL_RE = re.compile(r"\[TOOL_CALLS\]")
_PYTHON_TAG_RE = re.compile(r"<\|python_tag\|>")
_DS_BEGIN = "<｜tool▁calls▁begin｜>"
_DS_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>\s*\w+\s*<｜tool▁sep｜>\s*([^\n`]+?)\s*"
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def _tool_param_types(tools):
    """Map {tool_name: {param_name: json_schema_type}} for value coercion."""
    types = {}
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        types[name] = {k: (v or {}).get("type", "string") for k, v in props.items()}
    return types


def _coerce_value(raw, json_type):
    """Best-effort coerce an XML <parameter> string to its schema type."""
    raw = raw.strip()
    try:
        if json_type in ("number", "integer", "object", "array"):
            return json.loads(raw)
        if json_type == "boolean":
            return raw.strip().lower() == "true"
    except (ValueError, TypeError):
        pass
    return raw


def _extract_name_args(obj):
    """Pull (name, args dict) from a tool-call dict across naming conventions.

    Handles name vs function.name vs function (string), and arguments vs
    parameters vs function.arguments, with args possibly JSON-encoded as a
    string. Returns (None, {}) if obj isn't a usable call.
    """
    if not isinstance(obj, dict):
        return None, {}
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else None
    name = obj.get("name") or (fn or {}).get("name")
    if isinstance(obj.get("function"), str):
        name = obj["function"]
    args = obj.get("arguments")
    if args is None and fn:
        args = fn.get("arguments")
    if args is None:
        args = obj.get("parameters")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _iter_json_objects(s):
    """Yield top-level {...} substrings from s, respecting string escaping.

    Lets us pull several JSON objects out of one blob (e.g. Llama's
    ';'-separated parallel calls) without a brittle balanced-brace regex.
    """
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield s[start:i + 1]
                start = None


def _json_calls(blob):
    """Parse one-or-more tool-call dicts from a blob.

    Accepts a JSON array, a single JSON object, or several objects run together
    / separated by ';' or whitespace (the variants Mistral and Llama emit).
    """
    blob = blob.strip()
    if not blob:
        return []
    try:
        obj = json.loads(blob)
        if isinstance(obj, list):
            return [o for o in obj if isinstance(o, dict)]
        if isinstance(obj, dict):
            return [obj]
    except (ValueError, TypeError):
        pass
    calls = []
    for chunk in _iter_json_objects(blob):
        try:
            o = json.loads(chunk)
        except (ValueError, TypeError):
            continue
        if isinstance(o, dict):
            calls.append(o)
    return calls


def render_tools_prompt(tools):
    """Build a system-prompt block describing tools in Qwen3-Coder format."""
    specs = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if fn.get("name"):
            specs.append(json.dumps(fn, ensure_ascii=False))
    if not specs:
        return ""
    return (
        "You have access to the following tools. When you need to call one, "
        "emit it exactly in this format (one <tool_call> block per call, and "
        "nothing else in that turn):\n"
        "<tool_call>\n<function=TOOL_NAME>\n<parameter=ARG_NAME>\nARG_VALUE\n"
        "</parameter>\n</function>\n</tool_call>\n\n"
        "Available tools (JSON schema):\n<tools>\n"
        + "\n".join(specs)
        + "\n</tools>"
    )


def _tool_calls_to_text(tool_calls):
    """Render assistant tool_calls (from history) back into Qwen3-Coder XML."""
    blocks = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        params = "".join(
            f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
            for k, v in args.items()
        )
        blocks.append(f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>")
    return "\n".join(blocks)


def prepare_messages_for_tools(messages, tools):
    """Normalize OpenAI messages for a tool-enabled turn.

    - Injects/extends a system message describing the tools.
    - Renders prior assistant tool_calls back into the model's XML so the
      conversation stays coherent across turns.
    - Folds `tool` result messages into tagged user content (works regardless
      of whether the chat template knows the `tool` role).
    """
    tool_prompt = render_tools_prompt(tools)
    out = []
    injected = False
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "system":
            sys_text = content if isinstance(content, str) else (content or "")
            if tool_prompt and not injected:
                sys_text = (sys_text + "\n\n" + tool_prompt).strip()
                injected = True
            out.append({"role": "system", "content": sys_text})
            continue

        if role == "assistant" and msg.get("tool_calls"):
            rendered = _tool_calls_to_text(msg["tool_calls"])
            base = content if isinstance(content, str) and content else ""
            out.append({"role": "assistant", "content": (base + "\n" + rendered).strip()})
            continue

        if role == "tool":
            name = msg.get("name", "")
            result = content if isinstance(content, str) else json.dumps(content)
            tag = f' name="{name}"' if name else ""
            out.append({"role": "user",
                        "content": f"<tool_response{tag}>\n{result}\n</tool_response>"})
            continue

        # plain user/assistant (content may be None on a tool-call-only turn)
        out.append({"role": role, "content": content if content is not None else ""})

    if tool_prompt and not injected:
        out.insert(0, {"role": "system", "content": tool_prompt})
    return out


def parse_tool_calls(text, tools):
    """Extract tool calls from generated text.

    Returns (content_text, tool_calls) where tool_calls is a list of OpenAI
    tool_call dicts (empty if none found). Handles Qwen3-Coder XML, Hermes-style
    JSON-in-<tool_call>, Muse Glimmer ATEM blocks, Mistral [TOOL_CALLS], Llama
    <|python_tag|>, DeepSeek's <｜tool▁calls▁begin｜> blocks, and a bare JSON
    fallback.
    """
    param_types = _tool_param_types(tools)
    known = set(param_types)
    tool_calls = []

    def add(name, args):
        tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })

    blocks = _TOOL_CALL_RE.findall(text)
    if blocks:
        content = _TOOL_CALL_RE.sub("", text).strip()
        for block in blocks:
            block = block.strip()
            m = _FUNCTION_RE.search(block)
            if m:
                name = m.group(1).strip()
                args = {}
                for k, v in _PARAM_RE.findall(m.group(2)):
                    k = k.strip()
                    args[k] = _coerce_value(v, param_types.get(name, {}).get(k, "string"))
                add(name, args)
            else:
                # JSON inside <tool_call> (Hermes / Qwen2.5 style)
                try:
                    obj = json.loads(block)
                except (ValueError, TypeError):
                    obj = None
                name, args = _extract_name_args(obj)
                if name:
                    add(name, args)
        return content, tool_calls

    # Muse Glimmer native: ATEM <atem:function_calls> blocks (the model's own
    # chat template renders tool calls this way, so this is what it emits no
    # matter what format our rendered prompt asks for).
    atem_blocks = _ATEM_BLOCK_RE.findall(text)
    if atem_blocks:
        content = _ATEM_BLOCK_RE.sub("", text).strip()
        for block in atem_blocks:
            for name, params in _ATEM_INVOKE_RE.findall(block):
                name = name.strip()
                args = {}
                for k, v in _ATEM_PARAM_RE.findall(params):
                    k = k.strip()
                    args[k] = _coerce_value(v, param_types.get(name, {}).get(k, "string"))
                add(name, args)
        if tool_calls:
            return content, tool_calls

    # Qwen2.5-Coder native: bare <function=NAME>...</function> with NO
    # surrounding <tool_call> wrapper. The Qwen2.5-Coder models emit this form
    # regardless of the prompt we render, so the wrapped path above never fires
    # for them. _FUNCTION_RE only matches the literal <function=...> syntax, so
    # this won't trip on ordinary prose.
    fn_blocks = list(_FUNCTION_RE.finditer(text))
    if fn_blocks:
        content = _FUNCTION_RE.sub("", text).strip()
        for m in fn_blocks:
            name = m.group(1).strip()
            args = {}
            for k, v in _PARAM_RE.findall(m.group(2)):
                k = k.strip()
                args[k] = _coerce_value(v, param_types.get(name, {}).get(k, "string"))
            add(name, args)
        if tool_calls:
            return content, tool_calls

    # Mistral: [TOOL_CALLS] followed by a JSON array of {name, arguments}.
    m = _MISTRAL_RE.search(text)
    if m:
        content = text[:m.start()].strip()
        for obj in _json_calls(text[m.end():]):
            name, args = _extract_name_args(obj)
            if name:
                add(name, args)
        if tool_calls:
            return content, tool_calls

    # Llama 3.x: <|python_tag|> then JSON object(s) (';'-separated for parallel).
    m = _PYTHON_TAG_RE.search(text)
    if m:
        content = text[:m.start()].strip()
        for obj in _json_calls(text[m.end():]):
            name, args = _extract_name_args(obj)
            if name:
                add(name, args)
        if tool_calls:
            return content, tool_calls

    # DeepSeek: function name + ```json args``` inside <｜tool▁call▁begin｜> blocks.
    if _DS_BEGIN in text:
        content = text.split(_DS_BEGIN, 1)[0].strip()
        for name, blob in _DS_CALL_RE.findall(text):
            try:
                args = json.loads(blob)
            except (ValueError, TypeError):
                args = {}
            if name.strip():
                add(name.strip(), args if isinstance(args, dict) else {})
        if tool_calls:
            return content, tool_calls

    # Fallback: bare JSON (object or array) where every call names a known tool
    # — the failure mode when the model isn't told the proper format. We only
    # treat it as tool calls if all names match, so normal JSON answers pass
    # through untouched.
    stripped = text.strip()
    if known and stripped[:1] in ("{", "["):
        named = []
        for obj in _json_calls(stripped):
            name, args = _extract_name_args(obj)
            if name is None:
                # Some models emit {"function": "x", "k": v, ...} with no
                # arguments wrapper — treat leftover keys as the arguments.
                continue
            if not args:
                args = {k: v for k, v in obj.items()
                        if k not in ("name", "function", "type",
                                     "arguments", "parameters")}
            named.append((name, args))
        if named and all(n in known for n, _ in named):
            for n, a in named:
                add(n, a)
            return "", tool_calls

    return text, tool_calls


# ---------------------------------------------------------------------------
# Device slot — holds one pipeline + its metadata
# ---------------------------------------------------------------------------

class DeviceSlot:
    """One loaded model on one device."""

    backend = "genai"  # which runtime serves this slot (surfaced in /health)

    def __init__(self, device_name, device_id=None):
        self.device_name = device_name   # canonical "NPU", "GPU", "CPU" (display + routing)
        self.device_id = device_id or device_name  # OpenVINO id (may be "GPU.1" on multi-GPU)
        self.device_full = ""            # "Intel(R) AI Boost"
        self.pipe = None
        self.model_name = ""
        self.model_type = ""             # "vlm" or "llm"
        self.status = "not_configured"   # not_configured -> loading -> warming_up -> ready / error / idle_unloaded
        self.lock = threading.Lock()
        self._cancel = threading.Event()  # signal to stop generation
        self._stream_error = None        # last stream_tokens backend exception (single consumer per lock)
        self.last_used = time.time()     # for idle-unload watchdog
        self.model_dir = None            # remembered so we can reload after unload
        self.last_ttft_ms = None         # last request's time-to-first-token (prefix-cache hit ≈ low)
        self.prewarmed = False           # did _prewarm_slot succeed for this load
        self.supports_prefix_cache = True  # GenAI CB backend can prefix-cache; OptimumSlot can't
        self.kv_pool_gb = 0              # resolved KV pool for THIS slot (_resolve_kv_pool)
        self._atem = False               # Muse Glimmer channel stream needs translating

    def load(self, model_dir):
        """Load model, auto-detecting VLM vs LLM."""
        self.status = "loading"
        self.model_dir = model_dir
        self.model_name = model_display_name(model_dir)
        vlm = is_vlm(model_dir)
        self.model_type = "vlm" if vlm else "llm"
        # Muse Glimmer emits harmony-style channel routing; the GenAI pipeline
        # strips the special tokens, so what reaches the streamer is plain-text
        # routing words — _AtemPlainFilter translates them.
        try:
            with open(os.path.join(model_dir, "config.json")) as f:
                self._atem = json.load(f).get("model_type") == "muse_glimmer"
        except Exception:
            self._atem = False

        print(f"  [{self.device_name}] Detected: {self.model_type.upper()} ({self.model_name})")
        integrity_err = _verify_weights_integrity(model_dir)
        if integrity_err:
            raise RuntimeError(integrity_err)
        self._resolve_kv_pool(vlm)
        self._preflight_memory(vlm)
        print(f"  [{self.device_name}] Loading...", flush=True)

        # MoE disk offload (--offload-ratio): GPU-only plugin property, and it
        # only does anything on XMX hardware (Arc dGPU, Lunar Lake 140V+) —
        # verified: Qwen3-30B-A3B int4 runs in 2.35 GB resident at ratio 90 on
        # a 140V, while non-XMX iGPUs silently ignore it (see TODONT.md).
        offload = {}
        if OFFLOAD_RATIO > 0 and self.device_name == "GPU":
            offload = {"OFFLOAD_RATIO": OFFLOAD_RATIO}
            print(f"  [{self.device_name}] MoE disk offload on "
                  f"({OFFLOAD_RATIO}% of expert weights streamed)", flush=True)

        if vlm:
            VLMPipe = getattr(ovg, "VLMPipeline", None)
            if VLMPipe is None:
                raise RuntimeError("No VLMPipeline in this openvino_genai build.")
            if PROMPT_CACHE and self.kv_pool_gb and self.device_name in ("GPU", "CPU"):
                # VLMPipeline honors scheduler_config on current runtimes —
                # measured 2026-08-18: 2026.3 release (140V, ~9k-token prefix
                # 21.7s -> 3.9s TTFT) and 2026.4 nightly (B60, 33k-token
                # prefix 54.5s -> 1.3s). The old "CB backend is LLM-only"
                # belief was stale. A runtime that rejects the property falls
                # back to the plain pipeline, same pattern as the LLM branch.
                try:
                    sc = ovg.SchedulerConfig()
                    sc.enable_prefix_caching = True
                    sc.cache_size = self.kv_pool_gb
                    self.pipe = VLMPipe(str(model_dir), device=self.device_id,
                                        scheduler_config=sc, **offload)
                    src = "" if PROMPT_CACHE_GB is not None else \
                        ", auto-sized — pin with --cache-size-gb"
                    print(f"  [{self.device_name}] prefix caching on "
                          f"({self.kv_pool_gb} GB KV pool{src})", flush=True)
                except Exception as e:
                    print(f"  [{self.device_name}] prefix caching unavailable "
                          f"({e}); using plain pipeline", flush=True)
                    self.pipe = VLMPipe(str(model_dir), device=self.device_id,
                                        **offload)
            else:
                self.pipe = VLMPipe(str(model_dir), device=self.device_id,
                                    **offload)
        else:
            # NPU has a default prompt limit of 1024 tokens — raise it
            if self.device_name == "NPU":
                self.pipe = ovg.LLMPipeline(
                    str(model_dir), device=self.device_id,
                    MAX_PROMPT_LEN=4096,
                )
            elif PROMPT_CACHE:
                # GPU/CPU: enable prefix (KV) caching via the continuous-batching
                # backend so a repeated prompt prefix — e.g. an agent's fixed
                # system prompt + tool schemas, identical every turn — is
                # prefilled once instead of every turn. Auto-invalidated by any
                # prefix change (no staleness). Opt out with --no-prompt-cache.
                try:
                    sc = ovg.SchedulerConfig()
                    sc.enable_prefix_caching = True
                    sc.cache_size = self.kv_pool_gb
                    self.pipe = ovg.LLMPipeline(
                        str(model_dir), device=self.device_id, scheduler_config=sc,
                        **offload,
                    )
                    src = "" if PROMPT_CACHE_GB is not None else \
                        ", auto-sized — pin with --cache-size-gb"
                    print(f"  [{self.device_name}] prefix caching on "
                          f"({self.kv_pool_gb} GB KV pool{src})", flush=True)
                except Exception as e:
                    print(f"  [{self.device_name}] prefix caching unavailable "
                          f"({e}); using plain pipeline", flush=True)
                    self.pipe = ovg.LLMPipeline(str(model_dir), device=self.device_id,
                                                **offload)
            else:
                self.pipe = ovg.LLMPipeline(str(model_dir), device=self.device_id,
                                            **offload)

    def _resolve_kv_pool(self, vlm):
        """Set self.kv_pool_gb — the KV-cache pool for this slot's CB backend.

        An explicit --cache-size-gb pins it. Otherwise auto-size from the
        device budget: a third of what's left after weights, floored at the
        old 2 GB default and capped at AUTO_KV_TOKENS of this model's KV
        geometry. Sized from the *total* budget, not free memory, so the
        result is stable across restarts and idle-unload reloads. The CB
        backend grows into the pool rather than allocating it upfront, but
        with prefix caching on, blocks are never released — a long agent
        session eventually owns the whole pool, hence the headroom fraction
        (see AUTO_KV_HEADROOM_SHARE).
        """
        # VLM slots size a pool too: openvino_genai honors scheduler_config
        # on VLMPipeline (verified on 2026.3 release and the 2026.4 nightly;
        # a runtime that rejects it falls back to the plain pipeline at load,
        # leaving the pool unused).
        if not (PROMPT_CACHE and self.supports_prefix_cache
                and self.device_name in ("GPU", "CPU")):
            self.kv_pool_gb = 0
            return
        if PROMPT_CACHE_GB is not None:
            self.kv_pool_gb = PROMPT_CACHE_GB
            return
        gib = 2 ** 30
        mem = _device_mem_bytes(self.device_name, self.device_id)
        weights = _dir_size_bytes(self.model_dir)
        if not mem or not weights:
            self.kv_pool_gb = AUTO_KV_MIN_GB  # can't estimate — old default
            return
        headroom = mem - weights * 1.1  # same overhead margin as the preflight
        pool = headroom / AUTO_KV_HEADROOM_SHARE / gib
        per_tok = _kv_bytes_per_token(self.model_dir)
        if per_tok:
            pool = min(pool, AUTO_KV_TOKENS * per_tok / gib)
        self.kv_pool_gb = max(AUTO_KV_MIN_GB, int(pool))

    def _preflight_memory(self, vlm):
        """Sanity-check model weights + KV pool against the device's memory
        budget before loading. Warns and keeps going — the numbers are
        estimates, and on 16 GB cards OpenVINO's silent CPU fallback (or a
        'Got unfinished GenerationStatus' abort mid-request) is far worse
        than a false-positive warning here.
        """
        gib = 2 ** 30
        mem = _device_mem_bytes(self.device_name, self.device_id)
        weights = _dir_size_bytes(self.model_dir)
        if not mem or not weights:
            return  # can't estimate — stay quiet rather than guess
        kv_pool = self.kv_pool_gb * gib  # 0 when this slot can't prefix-cache
        need = (weights + kv_pool) * 1.1  # ~10% runtime/activation overhead
        if need > mem:
            if OFFLOAD_RATIO and self.device_name == "GPU":
                # MoE disk offload keeps only part of the expert weights
                # resident; the estimate above ignores that (expert share
                # isn't knowable from config geometry alone). Inform, don't
                # cry wolf — a 15.2 GB MoE serves fine on a 16 GB iGPU at
                # --offload-ratio 30 (measured).
                print(f"  [{self.device_name}] model (~{weights / gib:.1f} GB)"
                      f"{f' + KV pool ({kv_pool // gib} GB)' if kv_pool else ''} "
                      f"exceeds the {mem / gib:.1f} GB device budget, but "
                      f"--offload-ratio {OFFLOAD_RATIO} keeps only part of it "
                      f"resident — MoE models will likely fit; dense models "
                      f"will not (offload only covers MoE experts)", flush=True)
            else:
                hint = ("use a smaller quant or lower --cache-size-gb"
                        if self.device_name == "CPU" else
                        "raise the iGPU budget (Intel Graphics Software -> Shared GPU "
                        "Memory Override), use a smaller quant, or lower --cache-size-gb")
                print(f"  [{self.device_name}] WARNING: model (~{weights / gib:.1f} GB)"
                      f"{f' + KV pool ({kv_pool // gib} GB)' if kv_pool else ''} needs "
                      f"~{need / gib:.1f} GB but the device budget is {mem / gib:.1f} GB "
                      f"— this will likely NOT work ({hint})", flush=True)
        if kv_pool:
            per_tok = _kv_bytes_per_token(self.model_dir)
            if per_tok:
                capacity = kv_pool // per_tok
                line = (f"  [{self.device_name}] KV pool {self.kv_pool_gb} GB ~ "
                        f"{capacity // 1000}k tokens for this model "
                        f"({per_tok // 1024} KB/token)")
                if capacity < 32768:
                    line += (" — agent prompts (20k+ tokens) will exhaust it; "
                             "raise --cache-size-gb")
                print(line, flush=True)

    def warmup(self):
        self.status = "warming_up"
        print(f"  [{self.device_name}] Warmup...", end="", flush=True)
        t0 = time.perf_counter()
        gen = ovg.GenerationConfig()
        gen.max_new_tokens = 5
        gen.do_sample = False
        gen.top_k = 1
        try:
            if self.model_type == "vlm":
                self.pipe.generate(prompt="Hello", generation_config=gen)
            else:
                history = ovg.ChatHistory()
                history.append({"role": "user", "content": "Hi"})
                self.pipe.generate(history, gen)
            elapsed = time.perf_counter() - t0
            print(f" done ({elapsed:.1f}s)", flush=True)
            self.status = "ready"
        except Exception as e:
            print(f" failed: {e}", flush=True)
            self.status = "error"

    def unload(self):
        """Release the loaded pipeline. Caller must hold self.lock."""
        if self.pipe is None:
            return
        print(f"  [{self.device_name}] Idle — unloading {self.model_name}", flush=True)
        self.pipe = None
        self.status = "idle_unloaded"
        import gc
        gc.collect()

    def ensure_loaded(self):
        """Reload pipeline if it was unloaded. Blocks until ready."""
        if self.pipe is not None and self.status == "ready":
            return
        with self.lock:
            if self.pipe is not None and self.status == "ready":
                return  # someone else loaded it while we waited
            if self.model_dir is None:
                raise RuntimeError(f"Slot {self.device_name} has no model_dir")
            print(f"  [{self.device_name}] Reloading {self.model_name}...", flush=True)
            self.load(self.model_dir)
            self.warmup()

    def generate_vlm(self, text_prompt, images, gen):
        """VLM generate — images optional."""
        with self.lock:
            if images:
                imgs = images[0] if len(images) == 1 else images
                result = self.pipe.generate(
                    prompt=text_prompt, images=imgs, generation_config=gen,
                )
            else:
                result = self.pipe.generate(
                    prompt=text_prompt, generation_config=gen,
                )
            self.last_used = time.time()
        text = extract_text(result)
        if self._atem:
            atem = _AtemPlainFilter()
            text = atem.feed(text) + atem.close()
        return text

    def generate_llm(self, raw_messages, gen):
        """LLM generate — non-streaming."""
        history = ovg.ChatHistory()
        for msg in raw_messages:
            history.append({"role": msg["role"], "content": msg["content"]})
        with self.lock:
            result = self.pipe.generate(history, gen)
            self.last_used = time.time()
        ttft_ms, _ = extract_perf(result)
        if ttft_ms is not None:
            self.last_ttft_ms = ttft_ms
        return extract_text(result)

    def cancel(self):
        """Signal the current generation to stop."""
        self._cancel.set()

    def stream_tokens(self, raw_messages, gen, heartbeat, tag=""):
        """Low-level token stream — the seam between backend and protocol.

        Yields str (a decoded text chunk) as soon as one exists, or None when
        `heartbeat` seconds passed with no chunk and the generation worker is
        still alive (the caller decides: SSE pings, ndjson aborts). Terminates
        when generation completes, was cancelled, or the worker died.

        Owns: the worker thread, self.lock around generate, clearing _cancel
        INSIDE the lock (racing the previous request's finally otherwise),
        last_used, and recording any backend exception in self._stream_error
        (None on success) before terminating.

        Does NOT touch last_ttft_ms, set _cancel on exit, or emit protocol
        frames — those belong to the consumers. In particular the
        client-disconnect safety net (finally: _cancel.set()) MUST stay in the
        consumers: a generator's finally also runs on normal exhaustion, which
        would make the consumer's `was_cancelled = _cancel.is_set()` read True
        on every completed stream.
        """
        history = ovg.ChatHistory()
        for msg in raw_messages:
            history.append({"role": msg["role"], "content": msg["content"]})

        token_queue = Queue()
        self._stream_error = None

        def streamer_callback(token):
            if self._cancel.is_set():
                return True  # stop generation
            token_queue.put(token)
            return False

        def _generate():
            try:
                with self.lock:
                    self._cancel.clear()
                    self.pipe.generate(history, gen, streamer_callback)
                    self.last_used = time.time()
            except Exception as e:
                self._stream_error = e
                print(f"{datetime.now():%H:%M:%S} !! [{self.device_name}] "
                      f"{tag}generate error: {explain_genai_error(e, self)}", flush=True)
            finally:
                token_queue.put(None)

        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        while True:
            try:
                token = token_queue.get(timeout=heartbeat)
            except Empty:
                if not t.is_alive():
                    return  # worker died without the sentinel — defensive
                yield None  # heartbeat marker
                continue
            if token is None:
                return
            yield token

    def stream_vlm(self, text_prompt, images, gen, completion_id, created, t0):
        """VLM generate — SSE streaming. openvino-genai 2026.1+."""
        token_queue = Queue()
        token_count = 0
        gen_error = [None]

        def streamer_callback(token):
            if self._cancel.is_set():
                return True
            token_queue.put(token)
            return False

        def _generate():
            try:
                with self.lock:
                    self._cancel.clear()
                    if images:
                        imgs = images[0] if len(images) == 1 else images
                        self.pipe.generate(
                            prompt=text_prompt, images=imgs,
                            generation_config=gen, streamer=streamer_callback,
                        )
                    else:
                        self.pipe.generate(
                            prompt=text_prompt, generation_config=gen,
                            streamer=streamer_callback,
                        )
                    self.last_used = time.time()
            except Exception as e:
                gen_error[0] = e
                print(f"{datetime.now():%H:%M:%S} !! [{self.device_name}] "
                      f"VLM generate error: {e}", flush=True)
            finally:
                token_queue.put(None)

        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        atem = _AtemPlainFilter() if self._atem else None

        try:
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": self.model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            while True:
                try:
                    token = token_queue.get(timeout=180)
                except Empty:
                    break
                if token is None:
                    break
                token_count += 1
                if atem:
                    token = atem.feed(token)
                    if not token:
                        continue  # swallowed into a channel header / held tail
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            if atem:
                tail = atem.close()
                if tail:
                    chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": self.model_name,
                        "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            was_cancelled = self._cancel.is_set()
            if gen_error[0] is not None:
                err_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {
                        "content": f"\n[error: {gen_error[0]}]"
                    }, "finish_reason": "error"}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
            else:
                finish_reason = "cancelled" if was_cancelled else "stop"
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            self._cancel.set()

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        tag = " (cancelled)" if was_cancelled else (" (error)" if gen_error[0] else "")
        print(f"{datetime.now():%H:%M:%S} -> [{self.device_name}] "
              f"VLM {token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s){tag}",
              flush=True)

    def stream_llm(self, raw_messages, gen, completion_id, created, t0):
        """LLM generate — SSE streaming. Protocol layer over stream_tokens()."""
        token_count = 0

        try:
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": self.model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            for token in self.stream_tokens(raw_messages, gen, heartbeat=HEARTBEAT_SECS):
                if token is None:
                    # No token yet — likely a long prefill on a big prompt.
                    # Emit an empty-content delta to keep the client's idle
                    # watchdog from aborting (a real chunk resets content-based
                    # watchdogs, not just byte-based ones; empty string is a
                    # no-op for assembly). The background thread delivers tokens
                    # or the sentinel when ready.
                    ka = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": self.model_name,
                        "choices": [{"index": 0, "delta": {"content": ""},
                                     "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(ka)}\n\n"
                    continue
                if token_count == 0:
                    # Wall-clock TTFT: prefill is over when the first token lands.
                    self.last_ttft_ms = (time.perf_counter() - t0) * 1000
                token_count += 1
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Capture state BEFORE the finally-block safety-net sets _cancel
            was_cancelled = self._cancel.is_set()
            gen_error = [self._stream_error]
            if gen_error[0] is not None:
                finish_reason = "error"
                err_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {
                        "content": f"\n[error: {explain_genai_error(gen_error[0], self)}]"
                    }, "finish_reason": "error"}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
            else:
                finish_reason = "cancelled" if was_cancelled else "stop"
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Safety net: if client disconnects, stop generation
            self._cancel.set()

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        tag = " (cancelled)" if was_cancelled else (" (error)" if gen_error[0] else "")
        ttft = (f", TTFT {self.last_ttft_ms:.0f}ms" if token_count and
                self.last_ttft_ms is not None else "")
        print(f"{datetime.now():%H:%M:%S} -> [{self.device_name}] "
              f"{token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s{ttft}){tag}",
              flush=True)

    @property
    def info(self):
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            "device": self.device_full,
            "device_name": self.device_name,   # canonical NPU/GPU/CPU (routing/checks)
            "backend": self.backend,            # genai or optimum runtime
            "tools": _tools_supported(self),    # can this slot drive an agent loop?
            "prewarmed": self.prewarmed,        # did the --prewarm prefill succeed
            "kv_pool_gb": self.kv_pool_gb or None,  # resolved KV pool (null: no prefix cache)
            "last_ttft_ms": (round(self.last_ttft_ms)
                             if self.last_ttft_ms is not None else None),
        }


# ---------------------------------------------------------------------------
# Optimum-intel slot — python-runtime backend for architectures GenAI can't run
# ---------------------------------------------------------------------------

class _AtemStreamFilter:
    """Incremental translator for Muse Glimmer's ATEM channel stream.

    The model emits harmony-style channels after the generation prompt
    ('<|start|>assistant'): ' to=self<|message|>reasoning<|eom|>' then
    '<|start|>assistant to=user<|message|>answer<|eot|>'. NoLlama's surface
    for reasoning is <think> blocks (the web UI collapses them, agents skip
    them), so: to=self content -> <think>...</think>, to=user -> plain text,
    tool channels (to=<name>) -> passthrough (the ATEM XML flows to
    parse_tool_calls). Feed decoded text WITH special tokens; markers may
    split across chunks, so a possible marker prefix is held back.
    """

    _MARKERS = ("<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "<|return|>")
    _END = ("<|eom|>", "<|eot|>", "<|return|>")

    def __init__(self):
        self._buf = ""
        self._in_header = True   # generation prompt ends mid-header
        self._thinking = False
        self._done = False

    def _held_tail(self):
        """Length of the buffer tail that could be the start of a marker."""
        for ln in range(min(len(self._buf), 11), 0, -1):
            tail = self._buf[-ln:]
            if any(m.startswith(tail) for m in self._MARKERS):
                return ln
        return 0

    def feed(self, text):
        if self._done:
            return ""
        self._buf += text
        out = []
        while True:
            if self._in_header:
                i = self._buf.find("<|message|>")
                if i < 0:
                    break  # header still streaming — swallow, wait
                header = self._buf[:i]
                self._buf = self._buf[i + len("<|message|>"):]
                self._in_header = False
                if "to=self" in header:
                    self._thinking = True
                    out.append("<think>")
                continue
            hits = [(self._buf.find(m), m) for m in self._MARKERS]
            hits = [(p, m) for p, m in hits if p >= 0]
            if not hits:
                held = self._held_tail()
                emit = self._buf[:-held] if held else self._buf
                self._buf = self._buf[-held:] if held else ""
                if emit:
                    out.append(emit)
                break
            pos, marker = min(hits)
            if self._buf[:pos]:
                out.append(self._buf[:pos])
            self._buf = self._buf[pos + len(marker):]
            if self._thinking and marker in self._END:
                out.append("</think>\n")
                self._thinking = False
            if marker in ("<|eot|>", "<|return|>"):
                self._done = True  # end of turn — drop any trailing decode
                self._buf = ""
                break
            if marker in ("<|eom|>", "<|start|>"):
                self._in_header = True  # next channel header follows
        return "".join(out)

    def close(self):
        """Flush whatever remains (worker ended mid-channel)."""
        rest = "" if (self._in_header or self._done) else self._buf
        self._buf = ""
        if self._thinking:
            self._thinking = False
            return rest + "</think>\n"
        return rest


class _AtemPlainFilter:
    """ATEM channel translator for the GenAI path, where the markers are gone.

    VLMPipeline strips special tokens when decoding (GenerationConfig has no
    skip_special_tokens; only Tokenizer.decode takes one, and the pipeline
    doesn't expose it), so unlike _AtemStreamFilter there is no <|start|> or
    <|message|> to key on. What survives of Muse Glimmer's channel framing is
    the routing the model emits as ordinary text: generation begins mid-header
    (the prompt ends with '<|start|>assistant'), so the stream opens with
    'to=self' glued straight onto the reasoning, and the channel switch
    arrives as 'assistant to=user' glued onto the answer — the <|eom|><|start|>
    pair between them detokenizes to nothing. Shape verified against real
    VLMPipeline output (B60, 2026-08-18).

    Same surface as _AtemStreamFilter: to=self content -> <think>...</think>,
    to=user -> plain text; ATEM XML tool calls are ordinary text and pass
    through to parse_tool_calls. A tool turn ends in a to=<name> channel
    rather than to=user, so the think block also closes at the first
    <atem:function_calls> — any 'assistant to=<name>' header glued in front
    of it stays hidden inside the think block (channel names are glued to
    their content; there is no delimiter to split them out reliably).
    Necessarily more fragile than the marker filter — 'assistant to=user'
    could legitimately occur inside reasoning content — but the alternative
    is raw channel soup in every client.
    """

    _OPENERS = ("to=self", "to=user")
    _SWITCH = "assistant to=user"
    _TOOL_XML = "<atem:function_calls>"

    def __init__(self):
        self._buf = ""
        self._state = "header"  # header -> think | answer

    def _held_tail(self):
        """Length of the buffer tail that could be the start of a marker."""
        best = 0
        for mk in (self._SWITCH, self._TOOL_XML):
            for ln in range(min(len(self._buf), len(mk) - 1), 0, -1):
                if mk.startswith(self._buf[-ln:]):
                    best = max(best, ln)
                    break
        return best

    def feed(self, text):
        self._buf += text
        out = []
        while self._buf:
            if self._state == "header":
                lead = self._buf.lstrip()
                opener = next((o for o in self._OPENERS if lead.startswith(o)), None)
                if opener:
                    self._buf = lead[len(opener):]
                    if opener == "to=self":
                        out.append("<think>")
                        self._state = "think"
                    else:
                        self._state = "answer"
                    continue
                # Tool channel straight from the header (no reasoning first):
                # the name ends where its ATEM XML starts — '<' delimits.
                m = re.match(r"to=[\w.]+(?=<)", lead)
                if m:
                    self._buf = lead[m.end():]
                    self._state = "answer"
                    continue
                if (not lead or any(o.startswith(lead) for o in self._OPENERS)
                        or re.fullmatch(r"to=[\w.]*", lead)):
                    break  # could still grow into a channel header — hold
                self._state = "answer"  # no header at all — pass through
                continue
            if self._state == "think":
                i = self._buf.find(self._SWITCH)
                j = self._buf.find(self._TOOL_XML)
                if i >= 0 and (j < 0 or i < j):
                    out.append(self._buf[:i] + "</think>\n")
                    self._buf = self._buf[i + len(self._SWITCH):]
                    self._state = "answer"
                    continue
                if j >= 0:
                    # Tool call: close the think block and let the ATEM XML
                    # flow through to parse_tool_calls (marker kept).
                    out.append(self._buf[:j] + "</think>\n")
                    self._buf = self._buf[j:]
                    self._state = "answer"
                    continue
                held = self._held_tail()
                emit = self._buf[:-held] if held else self._buf
                self._buf = self._buf[-held:] if held else ""
                if emit:
                    out.append(emit)
                break
            out.append(self._buf)  # answer: pass through
            self._buf = ""
        return "".join(out)

    def close(self):
        """Flush whatever remains (stream ended mid-channel)."""
        rest, self._buf = self._buf, ""
        if self._state == "think":
            self._state = "answer"
            return rest + "</think>\n"
        return rest


_HF_PENALTY_NOTED = [False]


def _hf_gen_kwargs(gen):
    """Translate the routes' ovg.GenerationConfig into transformers generate
    kwargs. The routes keep building GenAI configs (no churn there); the
    optimum backend translates at the edge."""
    kw = dict(max_new_tokens=gen.max_new_tokens, do_sample=bool(gen.do_sample),
              repetition_penalty=gen.repetition_penalty)
    if gen.do_sample:
        # Greedy is do_sample=False alone — the routes' top_k=1 is a GenAI-ism.
        kw.update(temperature=gen.temperature, top_p=gen.top_p, top_k=int(gen.top_k))
    if any(getattr(gen, a, 0) for a in ("frequency_penalty", "presence_penalty")):
        if not _HF_PENALTY_NOTED[0]:
            print("  [optimum] frequency/presence penalties are not supported by "
                  "transformers generate — ignored (repetition_penalty still applies)",
                  flush=True)
            _HF_PENALTY_NOTED[0] = True
    return kw


class OptimumSlot(DeviceSlot):
    """One model served through optimum-intel's python runtime instead of an
    openvino_genai pipeline — for the NEEDS_OPTIMUM architectures GenAI can't
    run (nemotron_h: hybrid Mamba2+MoE), and for anything forced here with
    --backend optimum. Muse Glimmer lived here until Intel's VLM-shaped
    export moved it to the GenAI path (2026-08-18); this slot remains its
    fallback on release runtimes whose VLMPipeline lacks the arch.

    TEXT-ONLY for now: model_type reports "llm" even for a VLM-shaped export
    like Muse Glimmer, so the routes reject images with a clean 400 and hand
    us role-structured messages (the VLM paths flatten roles away). No prefix
    cache (that's a GenAI CB-scheduler feature) — prewarm and the KV-pool
    preflight are gated off via supports_prefix_cache.
    """

    backend = "optimum"

    def __init__(self, device_name, device_id=None):
        super().__init__(device_name, device_id)
        self.supports_prefix_cache = False
        self.model = None
        self.tokenizer = None

    def _lazy_import(self):
        """Import the heavy stack at load time (soundfile pattern): plain
        installs don't carry optimum/transformers, and GenAI-only serving
        must not pay the torch import."""
        try:
            import transformers  # noqa: F401
            import optimum.intel  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                f"{self.model_name} needs the optimum-intel python backend, "
                f"but it is not installed in this python ({e}). Serve from the "
                f"model-lab venv instead (see README: scripts/glimmer-export "
                f"builds it; one-time 'pip install flask openvino-genai')."
            )
        # transformers 5.16-dev materializes checkpoint shards in a 4-worker
        # thread pool with no env knob; parallel mmap copies die with a native
        # access violation (0xC0000005) on 32 GB machines. Serialize — same
        # workaround as scripts/glimmer-export/run-export.py.
        import transformers.core_model_loading as _cml
        _cml.GLOBAL_WORKERS = 1

    def load(self, model_dir):
        self.status = "loading"
        self.model_dir = model_dir
        self.model_name = model_display_name(model_dir)
        self.model_type = "llm"  # text-only phase, even when the export is VLM-shaped
        print(f"  [{self.device_name}] Detected: LLM ({self.model_name}, "
              f"optimum backend)")
        integrity_err = _verify_weights_integrity(model_dir)
        if integrity_err:
            raise RuntimeError(integrity_err)
        self._preflight_memory(vlm=False)
        self._lazy_import()
        if self.device_name == "GPU":
            # The corruption is OpenVINO-version-dependent, so key the warning
            # on the runtime rather than the device. On 2026.3 every GPU tested
            # was wrong: Xe-LPG died at warmup on a dynamic-shape op; Xe2 (140V,
            # Windows), Xe3 (B390, Linux — issue #24) and discrete Battlemage
            # (Arc Pro B60) warmed up fine and SILENTLY computed garbage — the
            # model half-perceives the prompt and greedy-loops, while the same
            # IR is correct on CPU. Verified fixed on 2026.4.0.dev20260814
            # (B60, 2026-08-15): the issue's own repro now quotes the prompt
            # verbatim on GPU. Warn before the compile, which costs minutes.
            try:
                import openvino as _ov
                _ov_ver = _ov.__version__
            except Exception:
                _ov_ver = ""
            # 2026.4+ (including its nightlies) carries the fix; anything older
            # is a known-bad combination for this backend. Compare numerically,
            # not as strings: "2026.10" < "2026.4" lexically but not in fact.
            _m = re.match(r"(\d+)\.(\d+)", _ov_ver or "")
            _fixed = bool(_m) and (int(_m.group(1)), int(_m.group(2))) >= (2026, 4)
            if _fixed:
                print(f"  [GPU] NOTE: the optimum backend on GPU corrupted "
                      f"output on OpenVINO <=2026.3 (issue #37419); this "
                      f"runtime ({_ov_ver}) is past the fix. Sanity-check the "
                      f"first reply anyway — the failure is silent.",
                      flush=True)
            else:
                print(f"  [GPU] WARNING: the optimum backend on GPU produces "
                      f"CORRUPTED OUTPUT on OpenVINO <=2026.3 — every device "
                      f"tested (Xe-LPG/Xe2/Xe3 iGPUs and the discrete Arc Pro "
                      f"B60) fails, silently (openvino issue #37419). This "
                      f"runtime is {_ov_ver or 'unknown'}. Use --device CPU, "
                      f"or 2026.4+ where it is verified fixed.",
                      flush=True)
        print(f"  [{self.device_name}] Loading (optimum-intel runtime)...",
              flush=True)
        from optimum.intel import OVModelForCausalLM, OVModelForVisualCausalLM
        from transformers import AutoTokenizer
        # Structural pick: a VLM-shaped export (Glimmer) must go through the
        # visual class even for text-only generate; nemotron_h is a plain LM.
        cls = OVModelForVisualCausalLM if is_vlm(model_dir) else OVModelForCausalLM
        try:
            self.model = cls.from_pretrained(model_dir, device=self.device_id)
        except KeyError as e:
            raise RuntimeError(
                f"transformers {__import__('transformers').__version__} does not "
                f"know this architecture ({e}) — the model-lab venv carries "
                f"transformers from git main, which does.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # We always pass max_new_tokens; a max_length in the model's
        # generation_config.json makes transformers warn about the pair on
        # every request. Drop it once here instead.
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_length = None
        # Muse Glimmer streams ATEM channel markers (reasoning/user/tool
        # routing) — those need translating; other architectures stream plain
        # text and get special tokens stripped at the streamer instead.
        try:
            with open(os.path.join(model_dir, "config.json")) as f:
                self._atem = json.load(f).get("model_type") == "muse_glimmer"
        except Exception:
            self._atem = False
        self.pipe = self.model  # inherited ensure_loaded/unload key on .pipe

    def _stopping_criteria(self):
        from transformers import StoppingCriteria, StoppingCriteriaList

        slot = self

        class _CancelStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return slot._cancel.is_set()

        return StoppingCriteriaList([_CancelStop()])

    def _generate_unlocked(self, raw_messages, hf_kwargs, streamer=None):
        """The bare generate call, no locking — exists because ensure_loaded()
        calls warmup() while HOLDING self.lock (non-reentrant), mirroring how
        DeviceSlot.warmup talks to the pipeline directly."""
        inputs = self.tokenizer.apply_chat_template(
            raw_messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        if debug:
            # The request dump above shows messages as the CLIENT sent them
            # (pre-sanitization); this is the prompt the model actually gets.
            print(f"{datetime.now():%H:%M:%S} [DEBUG/{self.device_name}] "
                  f"rendered prompt:\n"
                  f"{self.tokenizer.decode(inputs['input_ids'][0])}",
                  flush=True)
        self.model.generate(
            **inputs, streamer=streamer,
            stopping_criteria=self._stopping_criteria(),
            pad_token_id=self.tokenizer.eos_token_id, **hf_kwargs)

    def warmup(self):
        self.status = "warming_up"
        print(f"  [{self.device_name}] Warmup...", flush=True)
        t0 = time.perf_counter()
        self._generate_unlocked([{"role": "user", "content": "Hi"}],
                                dict(max_new_tokens=5, do_sample=False))
        print(f"  [{self.device_name}] Warmup done "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        self.status = "ready"
        self.last_used = time.time()

    def stream_tokens(self, raw_messages, gen, heartbeat, tag=""):
        """Same contract as DeviceSlot.stream_tokens, over TextIteratorStreamer."""
        from transformers import TextIteratorStreamer

        # ATEM models need the channel markers (special tokens) visible to the
        # filter; plain models get them stripped by the streamer directly.
        atem = _AtemStreamFilter() if getattr(self, "_atem", False) else None
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True,
            skip_special_tokens=atem is None,
            timeout=heartbeat)  # None → block forever (non-stream join)
        self._stream_error = None

        def _generate():
            try:
                with self.lock:
                    self._cancel.clear()
                    self._generate_unlocked(raw_messages, _hf_gen_kwargs(gen),
                                            streamer=streamer)
                    self.last_used = time.time()
            except Exception as e:
                self._stream_error = e
                print(f"{datetime.now():%H:%M:%S} !! [{self.device_name}] "
                      f"{tag}generate error: {e}", flush=True)
                streamer.end()  # unblock the consumer even on pre-generate failure

        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        it = iter(streamer)
        while True:
            try:
                chunk = next(it)
            except Empty:  # TextIteratorStreamer raises queue.Empty on timeout
                if not t.is_alive():
                    return
                yield None  # heartbeat marker
                continue
            except StopIteration:
                tail = atem.close() if atem else ""
                if tail:
                    yield tail
                return
            if atem is not None:
                chunk = atem.feed(chunk)
            if chunk:
                yield chunk

    def generate_llm(self, raw_messages, gen):
        """Non-streaming = joined stream: one code path, wall-clock TTFT
        (extract_perf has nothing to read on this backend)."""
        t0 = time.perf_counter()
        chunks = []
        for chunk in self.stream_tokens(raw_messages, gen, heartbeat=None):
            if not chunks:
                self.last_ttft_ms = (time.perf_counter() - t0) * 1000
            chunks.append(chunk)
        if self._stream_error is not None:
            raise self._stream_error
        return "".join(chunks).strip()

    def generate_vlm(self, text_prompt, images, gen):
        raise RuntimeError("vision path not enabled on the optimum backend yet")

    def stream_vlm(self, text_prompt, images, gen, completion_id, created, t0):
        raise RuntimeError("vision path not enabled on the optimum backend yet")

    def unload(self):
        if self.pipe is None:
            return
        self.model = None
        self.tokenizer = None
        super().unload()


# ---------------------------------------------------------------------------
# Whisper (speech-to-text) slot
# ---------------------------------------------------------------------------

def _load_audio(file_storage):
    """Read uploaded audio file to float32 numpy array at 16 kHz."""
    if sf is None:
        raise RuntimeError("soundfile not installed. pip install soundfile")
    audio, sr = sf.read(io.BytesIO(file_storage.read()), dtype="float32")
    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Resample to 16 kHz if needed
    if sr != 16000:
        target_len = int(len(audio) * 16000 / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


class WhisperSlot:
    """Holds a WhisperPipeline for speech-to-text."""

    def __init__(self, device_name, device_id=None):
        self.device_name = device_name
        self.device_id = device_id or device_name
        self.device_full = ""
        self.pipe = None
        self.model_name = ""
        self.model_type = "stt"
        self.status = "not_configured"
        self.lock = threading.Lock()

    def load(self, model_dir):
        self.status = "loading"
        self.model_name = model_display_name(model_dir)
        print(f"  [{self.device_name}] Loading Whisper ({self.model_name})...",
              flush=True)
        WhisperPipe = getattr(ovg, "WhisperPipeline", None)
        if WhisperPipe is None:
            raise RuntimeError(
                "No WhisperPipeline in this openvino_genai build. "
                "Upgrade to openvino-genai >= 2025.1."
            )
        self.pipe = WhisperPipe(str(model_dir), self.device_id)

    def warmup(self):
        self.status = "ready"
        print(f"  [{self.device_name}] Whisper ready", flush=True)

    def transcribe(self, audio_samples, language=None):
        """Transcribe float32 audio at 16 kHz. Returns text."""
        kwargs = {}
        if language:
            kwargs["language"] = f"<|{language}|>"
            kwargs["task"] = "transcribe"
        with self.lock:
            result = self.pipe.generate(audio_samples, **kwargs)
        if hasattr(result, "texts") and result.texts:
            return result.texts[0].strip()
        return str(result).strip()

    @property
    def info(self):
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            "device": self.device_full,
        }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_REQUEST_BYTES = 50 * 1024 * 1024  # 50 MB — enough for large base64 images
HEARTBEAT_SECS = 15  # SSE keep-alive cadence during long prefill (big prompts / tool turns)

# Default repetition penalty, overridable in nollama.ini ([generation] section)
# next to this file. Ollama ships 1.1, which breaks thinking-loops faster but
# is known to hurt code generation; 1.05 is the compromise default. Clients
# that send their own penalties (OpenAI frequency/presence_penalty, Ollama
# repeat_penalty) override this per-request — see apply_penalties().
import configparser as _configparser
_ini = _configparser.ConfigParser()
_ini.read(Path(__file__).parent / "nollama.ini")
REPETITION_PENALTY = _ini.getfloat("generation", "repetition_penalty", fallback=1.05)


def apply_penalties(gen, repetition=None, frequency=None, presence=None):
    """Set gen's penalties: per-request client values win over the
    nollama.ini/default repetition penalty. frequency/presence map through
    when this openvino-genai build supports them (CB pipelines do; the
    static NPU pipeline may ignore them)."""
    gen.repetition_penalty = repetition if repetition is not None else REPETITION_PENALTY
    for attr, val in (("frequency_penalty", frequency), ("presence_penalty", presence)):
        if val is not None:
            try:
                setattr(gen, attr, float(val))
            except Exception:
                pass
PROMPT_CACHE = True   # prefix-KV caching on GPU/CPU LLM slots (set False via --no-prompt-cache)
PROMPT_CACHE_GB = None  # KV pool GB pinned via --cache-size-gb; None = auto-size per slot at load
AUTO_KV_MIN_GB = 2    # auto-size floor — the old fixed default; also the fallback when the
                      # device budget or model geometry can't be read
AUTO_KV_TOKENS = 65536  # auto-size cap, in tokens of the model's KV geometry: covers an
                        # agent's ~21k-token system prompt plus a long session; beyond that
                        # the pool is waste
AUTO_KV_HEADROOM_SHARE = 3  # auto takes at most 1/(this) of what's left after weights —
                            # iGPU "device memory" and the CPU pool are the same RAM the
                            # agent's own compilers and tests run in
OFFLOAD_RATIO = 0     # % of MoE expert weights streamed from disk on GPU (--offload-ratio).
                      # Needs an XMX-capable GPU (Arc/Lunar Lake+) — silent no-op without.
                      # Measured on Arc 140V: 30B-A3B int4 runs in 2.35 GB resident at 90.
PREWARM_FILE = None   # path (--prewarm) to a saved prompt: prefilled at startup, auto-captured while serving
PREWARM_MIN_CHARS = 4000  # only pre-warm/capture big (agent-sized) system prompts, not plain chat
_prewarm_hash = None  # debounce: only re-capture when the system prompt changes

app = Flask("NoLlama",
            template_folder=os.path.join(SCRIPT_DIR, "templates"),
            static_folder=os.path.join(SCRIPT_DIR, "static"))
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

# Device slots — filled in main()
primary = None        # main model (NPU, GPU, or CPU)
secondary = None      # optional second model (GPU, for vision or bigger LLM)
whisper_slot = None   # optional Whisper STT model
max_dim = 768
debug = False
vscode_compat = False  # report a real Ollama version so VS Code accepts us

# Ollama version reported under --vscode-compat. VS Code's Ollama client
# validates this field and rejects anything non-numeric, so it has to look like
# a real Ollama release; it is a shim for their check, not a claim about what we
# are. Override with --vscode-ollama-version if VS Code raises its floor. We
# deliberately do NOT look the current number up over the network: a local-first
# server should not need the internet to start, and "high enough" is all this
# has to be.
VSCODE_OLLAMA_VERSION = "0.18.3"
_request_counter = itertools.count(1)  # thread-safe id generator


def make_id():
    return f"arc-{next(_request_counter):04d}"


def overall_status():
    """Ready when all configured devices are ready."""
    slots = [s for s in (primary, secondary) if s and s.status != "not_configured"]
    if not slots:
        return "not_configured"
    # If ANY slot is ready or idle_unloaded (will reload on demand), we can
    # serve requests. A dead secondary shouldn't kill the primary.
    if any(s.status in ("ready", "idle_unloaded") for s in slots):
        return "ready"
    if all(s.status == "error" for s in slots):
        return "error"
    return "loading"


def openai_error(message, error_type="invalid_request_error", status=400):
    return jsonify({"error": {"message": message, "type": error_type}}), status


def _sse_replay(completion_id, created, model, message, finish_reason):
    """Emit a buffered chat result as an OpenAI streaming SSE sequence.

    Used for tool-enabled turns, where we must buffer the full generation
    before we can hand back a structured tool_calls delta.
    """
    def chunk(delta, finish=None):
        return "data: " + json.dumps({
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }) + "\n\n"

    yield chunk({"role": "assistant"})
    if message.get("content"):
        yield chunk({"content": message["content"]})
    for i, tc in enumerate(message.get("tool_calls") or []):
        yield chunk({"tool_calls": [{
            "index": i, "id": tc["id"], "type": "function",
            "function": tc["function"],
        }]})
    yield chunk({}, finish_reason)
    yield "data: [DONE]\n\n"


def _sse_tool_stream(slot, raw_messages, gen, tools, completion_id, created, t0,
                     vlm=None):
    """Buffered tool turn, streamed with keep-alive frames.

    A tool turn must be fully generated before we can emit a structured
    tool_calls delta — but prefilling a big agent prompt (e.g. an OpenClaw
    system prompt) on a small device can take minutes, longer than a client's
    idle watchdog. So run generation in a background thread and emit SSE pings
    until it finishes, then replay the parsed result. Without this the client
    sees nothing during prefill and aborts (and OpenVINO can't cancel a blocked
    prefill, so the abandoned generation keeps churning).

    vlm: (text_prompt, images) when the slot is a VLM — the same buffered
    turn, generated through the VLM pipeline instead of ChatHistory.
    """
    result = {}

    def _run():
        try:
            if vlm is not None:
                result["text"] = slot.generate_vlm(vlm[0], vlm[1], gen)
            else:
                result["text"] = slot.generate_llm(raw_messages, gen)
        except Exception as e:  # noqa: BLE001 — surfaced to the client below
            result["error"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    # Immediate role frame so the client sees activity at once, then a ping
    # every HEARTBEAT_SECS while generation runs (no tokens exist yet).
    yield ("data: " + json.dumps({
        "id": completion_id, "object": "chat.completion.chunk",
        "created": created, "model": slot.model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"},
                     "finish_reason": None}],
    }) + "\n\n")
    while th.is_alive():
        th.join(timeout=HEARTBEAT_SECS)
        if th.is_alive():
            # Empty-content keep-alive (resets content- and byte-based client
            # watchdogs alike; empty string is a no-op for message assembly).
            yield ("data: " + json.dumps({
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": slot.model_name,
                "choices": [{"index": 0, "delta": {"content": ""},
                             "finish_reason": None}],
            }) + "\n\n")

    elapsed = time.perf_counter() - t0
    if result.get("error") is not None:
        err = explain_genai_error(result["error"], slot)
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] "
              f"{'VLM' if vlm is not None else 'LLM'} error: {err}", flush=True)
        yield ("data: " + json.dumps({
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": slot.model_name,
            "choices": [{"index": 0, "delta": {"content": f"\n[error: {err}]"},
                         "finish_reason": "error"}],
        }) + "\n\n")
        yield "data: [DONE]\n\n"
        return

    text = result.get("text", "")
    n_words = len(text.split())
    ttft = (f", TTFT {slot.last_ttft_ms:.0f}ms"
            if slot.last_ttft_ms is not None else "")
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"~{n_words} tokens in {elapsed:.1f}s "
          f"({n_words / max(elapsed, 1e-6):.1f} tok/s{ttft})", flush=True)

    text, tool_calls = parse_tool_calls(text, tools)
    if tool_calls:
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
              f"{len(tool_calls)} tool call(s): "
              f"{', '.join(tc['function']['name'] for tc in tool_calls)}", flush=True)
        message = {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish_reason = "stop"

    # Reuse the replay emitter (it re-sends a role frame — harmless, clients
    # just set role twice) for the content/tool_calls/finish/[DONE] tail.
    for frame in _sse_replay(completion_id, created, slot.model_name, message, finish_reason):
        yield frame


def _slot_serviceable(slot):
    """A slot can serve requests if loaded or just idle-unloaded (will reload)."""
    return slot and slot.status in ("ready", "idle_unloaded")


def _route_request(has_images, requested_model):
    """Pick which DeviceSlot handles this request."""
    # Explicit model@device selection overrides routing
    if requested_model:
        for slot in (primary, secondary):
            if not _slot_serviceable(slot):
                continue
            # Match "model@DEVICE" or just "model"
            slot_full = f"{slot.model_name}@{slot.device_name}"
            if requested_model in (slot_full, slot.model_name):
                return slot

    # Dual mode routing
    if _slot_serviceable(secondary):
        if has_images:
            # Images → whichever slot is a VLM
            for slot in (secondary, primary):
                if _slot_serviceable(slot) and slot.model_type == "vlm":
                    return slot
            return None  # no VLM loaded
        else:
            # Text → prefer the better/primary model
            # If GPU has a big LLM, use GPU. Otherwise use primary (NPU).
            if secondary.model_type == "llm":
                return secondary  # GPU has a big LLM — use it
            return primary  # GPU has VLM, text goes to NPU

    # Single mode — everything goes to primary
    return primary if _slot_serviceable(primary) else None


def _tools_supported(slot):
    """Tool calling runs on GPU/iGPU and CPU, but not the NPU.

    The NPU has a hard prompt cap (MAX_PROMPT_LEN) and small NPU-class models
    can't reliably drive multi-step agent loops, so we never honor `tools` there
    — that request is answered as plain chat. A capable coder LLM on the GPU, or
    on a strong desktop CPU (e.g. Core Ultra 9 with many cores), drives tool
    loops fine; tool turns are buffered with SSE keep-alive (see
    _sse_tool_stream) so a slow prefill doesn't trip the client's watchdog.
    """
    return bool(slot) and slot.device_name in ("GPU", "CPU")


def _maybe_capture_prewarm(raw_messages):
    """Save a big (agent) prompt to PREWARM_FILE so the next startup can warm
    the prefix cache. Debounced on the system prompt — written once per distinct
    system prompt, not every turn. Stale-safe: if the agent's prompt later
    changes, a fresh request overwrites the file; a mismatch only costs a cold
    first turn, never a wrong answer.
    """
    if not PREWARM_FILE or not raw_messages:
        return
    sys_text = "".join(str(m.get("content", "")) for m in raw_messages
                       if m.get("role") == "system")
    if len(sys_text) < PREWARM_MIN_CHARS:
        return
    global _prewarm_hash
    # sha256, not builtin hash(): hash() is PYTHONHASHSEED-salted per process,
    # which forced one redundant rewrite after every restart.
    h = hashlib.sha256(sys_text.encode("utf-8")).hexdigest()
    if h == _prewarm_hash:
        return
    # Save system + the first user turn — enough to cache the shared prefix.
    to_save = []
    for m in raw_messages:
        to_save.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        if m.get("role") == "user":
            break
    try:
        with open(PREWARM_FILE, "w", encoding="utf-8") as f:
            json.dump(to_save, f)
        _prewarm_hash = h
    except OSError:
        pass


def _prewarm_slot(slot):
    """Prefill the saved prompt at startup so the first real (cold) turn is a
    cache hit. Only meaningful on a GPU/CPU LLM slot with prefix caching on.
    """
    if not (PREWARM_FILE and PROMPT_CACHE and slot.model_type == "llm"
            and slot.device_name in ("GPU", "CPU")
            and getattr(slot, "supports_prefix_cache", False)):
        # OptimumSlot has no prefix cache — a prewarm prefill would burn a
        # full 30B-scale prefill at startup for nothing.
        return
    if not os.path.isfile(PREWARM_FILE):
        return
    try:
        with open(PREWARM_FILE, encoding="utf-8") as f:
            raw_messages = json.load(f)
    except (OSError, ValueError):
        return
    if not raw_messages:
        return
    try:
        gen = ovg.GenerationConfig()
        gen.max_new_tokens = 1
        gen.do_sample = False
        t0 = time.perf_counter()
        slot.generate_llm(raw_messages, gen)  # prefills -> populates prefix cache
        slot.prewarmed = True
        print(f"  [{slot.device_name}] pre-warmed prompt cache from "
              f"{os.path.basename(PREWARM_FILE)} ({time.perf_counter() - t0:.1f}s)",
              flush=True)
    except Exception as e:
        slot.prewarmed = False
        print(f"  [{slot.device_name}] pre-warm failed: {explain_genai_error(e, slot)}", flush=True)


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

def _log_request(api_label):
    if not debug:
        return
    body_raw = request.get_data(as_text=True)
    try:
        body_str = json.dumps(json.loads(body_raw), indent=2) if body_raw else ""
    except Exception:
        body_str = body_raw
    ua = request.headers.get("User-Agent", "")
    print(f"{datetime.now():%H:%M:%S} [DEBUG/{api_label}] {request.method} {request.path}"
          f"  UA={ua!r}", flush=True)
    if body_str:
        for line in body_str.splitlines():
            print(f"  {line}", flush=True)


@app.before_request
def _debug_openai():
    _log_request("OpenAI")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/")
def gui():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    devices = {}
    if primary and primary.status != "not_configured":
        devices[primary.device_name.lower()] = primary.info
    if secondary and secondary.status != "not_configured":
        devices[secondary.device_name.lower()] = secondary.info
    # prompt_cache stays a bare bool — start-openclaw.ps1's health check
    # truth-tests it; the details live in prompt_cache_info (per-slot TTFT
    # and prewarm state are in each device's info block).
    result = {"status": overall_status(), "version": __version__,
              "devices": devices,
              "prompt_cache": PROMPT_CACHE,
              "prompt_cache_info": {
                  "enabled": PROMPT_CACHE,
                  # pinned --cache-size-gb value; null = auto-sized (the
                  # resolved size is per-slot: devices[*].kv_pool_gb)
                  "pool_gb": PROMPT_CACHE_GB,
                  "auto": PROMPT_CACHE_GB is None,
                  "prewarm_file": PREWARM_FILE,
              }}
    if whisper_slot and whisper_slot.status != "not_configured":
        result["whisper"] = whisper_slot.info
    return jsonify(result)


@app.route("/v1/models", methods=["GET"])
def list_models():
    data = []
    for slot in (primary, secondary):
        if slot and slot.status == "ready":
            data.append({
                "id": f"{slot.model_name}@{slot.device_name}",
                "object": "model",
                "created": 0,
                "owned_by": f"local-{slot.device_name.lower()}",
            })
    if whisper_slot and whisper_slot.status == "ready":
        data.append({
            "id": f"whisper@{whisper_slot.device_name}",
            "object": "model",
            "created": 0,
            "owned_by": f"local-{whisper_slot.device_name.lower()}",
        })
    return jsonify({"object": "list", "data": data})


@app.route("/v1/cancel", methods=["POST"])
def cancel_generation():
    """Stop any in-progress generation. Returns immediately."""
    for slot in (primary, secondary):
        if slot:
            slot.cancel()
    return jsonify({"status": "ok"})


@app.route("/v1/audio/transcriptions", methods=["POST"])
def audio_transcriptions():
    """OpenAI-compatible speech-to-text. Accepts multipart form with audio file."""
    if not whisper_slot or whisper_slot.status != "ready":
        return openai_error(
            "No speech-to-text model loaded. Use --whisper-dir.", "server_error", 503,
        )

    if "file" not in request.files:
        return openai_error("'file' is required (multipart form upload)")

    audio_file = request.files["file"]
    language = request.form.get("language")
    response_format = request.form.get("response_format", "json")

    try:
        audio_samples = _load_audio(audio_file)
    except Exception as e:
        return openai_error(f"Failed to read audio: {e}")

    duration = len(audio_samples) / 16000
    lang_tag = f", lang={language}" if language else ""
    print(f"\n{datetime.now():%H:%M:%S} <- [{whisper_slot.device_name}] "
          f"Whisper {duration:.1f}s audio{lang_tag}", flush=True)

    t0 = time.perf_counter()
    try:
        text = whisper_slot.transcribe(audio_samples, language=language)
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{whisper_slot.device_name}] "
              f"Whisper error: {e}", flush=True)
        return openai_error(f"Transcription failed: {e}", "server_error", 500)

    elapsed = time.perf_counter() - t0
    print(f"{datetime.now():%H:%M:%S} -> [{whisper_slot.device_name}] "
          f"Whisper {len(text)} chars in {elapsed:.1f}s", flush=True)

    if response_format == "text":
        return Response(text, mimetype="text/plain")

    return jsonify({"text": text})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    if overall_status() != "ready":
        return openai_error(
            f"Server not ready (status: {overall_status()}). "
            "Check GET /health.", "server_error", 503,
        )

    body = request.get_json(silent=True)
    if not body:
        return openai_error("Request body must be JSON")
    messages = body.get("messages")
    if not messages:
        return openai_error("'messages' is required")

    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 0.0)
    top_p = body.get("top_p", 1.0)
    stream = body.get("stream", False)
    requested_model = body.get("model", "")
    tools = body.get("tools") or []

    # Parse messages
    try:
        text_prompt, images, raw_messages = parse_messages(messages, max_dim)
    except FileNotFoundError as e:
        return openai_error(str(e))
    except ValueError as e:
        return openai_error(str(e))
    except Exception as e:
        return openai_error(f"Failed to parse request: {e}")

    # Route to device
    slot = _route_request(bool(images), requested_model)
    if slot is None:
        if images:
            return openai_error("No vision model loaded. Send text only, or load a VLM.")
        return openai_error("No model ready to handle this request.", "server_error", 503)

    # Tool calling is GPU/iGPU-only. Only when a GPU slot serves the turn do we
    # render tool specs into the prompt and (later) parse calls back out; on
    # NPU/CPU the request is answered as a plain chat turn.
    tools_active = bool(tools) and _tools_supported(slot)
    if tools_active:
        try:
            text_prompt, images, raw_messages = parse_messages(
                prepare_messages_for_tools(messages, tools), max_dim)
        except Exception as e:
            return openai_error(f"Failed to parse request: {e}")
    elif tools:
        print(f"{datetime.now():%H:%M:%S} -- [{slot.device_name}] "
              f"tools ignored (GPU-only feature)", flush=True)

    # Reject images on LLM
    if images and slot.model_type == "llm":
        return openai_error(
            f"Model '{slot.model_name}' on {slot.device_name} does not support images. "
            "Remove image content or load a VLM."
        )

    # Reload if the slot was idle-unloaded (blocks until ready)
    try:
        slot.ensure_loaded()
    except Exception as e:
        return openai_error(f"Failed to reload model: {e}", "server_error", 500)

    # Build generation config
    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_tokens
    if temperature and temperature > 0.01:
        gen.do_sample = True
        gen.temperature = temperature
        gen.top_p = top_p
    else:
        gen.do_sample = False
        gen.top_k = 1
    apply_penalties(gen,
                    repetition=body.get("repetition_penalty"),
                    frequency=body.get("frequency_penalty"),
                    presence=body.get("presence_penalty"))

    completion_id = make_id()
    created = int(time.time())
    n_images = len(images)
    tag = f"{n_images} image{'s' if n_images != 1 else ''}, " if n_images else ""
    stream_tag = " (stream)" if stream else ""
    print(f"\n{datetime.now():%H:%M:%S} <- [{slot.device_name}] {tag}"
          f"{len(text_prompt)} chars, max_tokens={max_tokens}{stream_tag}",
          flush=True)

    t0 = time.perf_counter()

    # --- VLM path ---
    if slot.model_type == "vlm":
        # Tool turns are buffered like the LLM path's (structured tool_calls
        # need the whole generation); images alongside tools are allowed — a
        # screenshot plus tool specs is a legitimate agent turn.
        if stream and tools_active:
            return Response(
                _sse_tool_stream(slot, raw_messages, gen, tools, completion_id,
                                 created, t0, vlm=(text_prompt, images)),
                mimetype="text/event-stream",
                headers={"X-Device": slot.device_name, "X-Model": slot.model_name},
            )
        if stream:
            return Response(
                slot.stream_vlm(text_prompt, images, gen, completion_id, created, t0),
                mimetype="text/event-stream",
                headers={"X-Device": slot.device_name, "X-Model": slot.model_name},
            )

        try:
            text = slot.generate_vlm(text_prompt, images, gen)
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] VLM error: {e}", flush=True)
            return openai_error(f"Inference failed: {e}", "server_error", 500)

        elapsed = time.perf_counter() - t0
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
              f"{len(text)} chars in {elapsed:.1f}s", flush=True)

        tool_calls = []
        if tools_active:
            text, tool_calls = parse_tool_calls(text, tools)
            if tool_calls:
                print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
                      f"{len(tool_calls)} tool call(s): "
                      f"{', '.join(tc['function']['name'] for tc in tool_calls)}",
                      flush=True)
        if tool_calls:
            message = {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": text}
            finish_reason = "stop"

        resp = jsonify({
            "id": completion_id, "object": "chat.completion",
            "created": created, "model": slot.model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
        })
        resp.headers["X-Device"] = slot.device_name
        resp.headers["X-Model"] = slot.model_name
        return resp

    # Capture a big agent prompt so the next startup can pre-warm it (no-op
    # unless --prewarm is set).
    _maybe_capture_prewarm(raw_messages)

    # --- LLM path ---
    if stream and not tools_active:
        return Response(
            slot.stream_llm(raw_messages, gen, completion_id, created, t0),
            mimetype="text/event-stream",
            headers={"X-Device": slot.device_name, "X-Model": slot.model_name},
        )

    # Tool turns must be buffered (we need the whole generation before emitting a
    # structured tool_calls delta). When streaming, buffer in a background thread
    # and emit SSE keep-alive frames so a long prefill on a big agent prompt
    # doesn't trip the client's idle watchdog (see _sse_tool_stream).
    if stream and tools_active:
        return Response(
            _sse_tool_stream(slot, raw_messages, gen, tools, completion_id, created, t0),
            mimetype="text/event-stream",
            headers={"X-Device": slot.device_name, "X-Model": slot.model_name},
        )

    # Non-streaming (with or without tools): one blocking generate + JSON reply.
    try:
        text = slot.generate_llm(raw_messages, gen)
    except Exception as e:
        err = explain_genai_error(e, slot)
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] LLM error: {err}", flush=True)
        return openai_error(f"Inference failed: {err}", "server_error", 500)

    elapsed = time.perf_counter() - t0
    n_words = len(text.split())
    ttft = (f", TTFT {slot.last_ttft_ms:.0f}ms"
            if slot.last_ttft_ms is not None else "")
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"~{n_words} tokens in {elapsed:.1f}s ({n_words / max(elapsed, 1e-6):.1f} tok/s{ttft})",
          flush=True)

    tool_calls = []
    if tools_active:
        text, tool_calls = parse_tool_calls(text, tools)
        if tool_calls:
            print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
                  f"{len(tool_calls)} tool call(s): "
                  f"{', '.join(tc['function']['name'] for tc in tool_calls)}",
                  flush=True)

    if tool_calls:
        message = {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish_reason = "stop"

    resp = jsonify({
        "id": completion_id, "object": "chat.completion",
        "created": created, "model": slot.model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    })
    resp.headers["X-Device"] = slot.device_name
    resp.headers["X-Model"] = slot.model_name
    return resp


# ---------------------------------------------------------------------------
# Ollama-compatible API (port 11434)
# ---------------------------------------------------------------------------

ollama_app = Flask("NoLlama-Ollama")
ollama_app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

OLLAMA_PORT = 11434


@ollama_app.before_request
def _debug_ollama():
    _log_request("Ollama")


@ollama_app.route("/")
def ollama_health():
    return "Ollama is running"


@ollama_app.route("/api/version", methods=["GET"])
def ollama_version():
    # Two callers, two answers. --vscode-compat gets a plausible Ollama version
    # because VS Code validates the field (see VSCODE_OLLAMA_VERSION). Everyone
    # else gets our real version, prefixed so no client mistakes us for Ollama.
    # It used to report a hardcoded "nollama-0.1.0" here, which stopped being
    # true a long time ago.
    version = VSCODE_OLLAMA_VERSION if vscode_compat else f"nollama-{__version__}"
    return jsonify({"version": version})


@ollama_app.route("/api/tags", methods=["GET"])
def ollama_tags():
    models = []
    for slot in (primary, secondary):
        if slot and slot.status == "ready":
            models.append({
                "name": slot.model_name,
                "model": slot.model_name,
                "size": slot.info.get("size", 0),
                "details": {
                    "family": slot.model_name.split("-")[0],
                    "parameter_size": "",
                    "quantization_level": "int4",
                },
            })
    return jsonify({"models": models})


@ollama_app.route("/api/show", methods=["POST"])
def ollama_show():
    body = request.get_json(silent=True) or {}
    model_name = body.get("model", "")

    model_info = {}
    # Copilot Chat uses `general.basename` to label models in the picker, but
    # it prefers the name returned by /api/tags. Echo back what we returned there
    # so the picker shows the same name the user sees in /api/tags.
    if request.headers.get("User-Agent", "").startswith("GitHubCopilotChat/"):
        model_info["general.basename"] = model_name

    for slot in (primary, secondary):
        if slot and slot.model_name == model_name:
            # Only advertise `tools` for GPU/iGPU slots — tool calling is a
            # GPU-only feature, so NPU/CPU models show as completion-only and
            # Copilot won't offer them for agent (tool) mode.
            caps = ["completion"] + (["tools"] if _tools_supported(slot) else [])
            return jsonify({
                "model": model_name,
                "details": {
                    "family": model_name.split("-")[0],
                    "parameter_size": "",
                    "quantization_level": "int4",
                },
                "model_info": model_info,
                "capabilities": caps,
            })
    # Unknown model — we can't confirm it's on a GPU, so don't claim tools.
    return jsonify({"model": model_name, "details": {}, "model_info": model_info,
                    "capabilities": ["completion"]})


@ollama_app.route("/api/chat", methods=["POST"])
def ollama_chat():
    if overall_status() != "ready":
        return jsonify({"error": "model not ready"}), 503

    body = request.get_json(silent=True) or {}
    ollama_messages = body.get("messages", [])
    stream = body.get("stream", True)  # Ollama defaults to streaming
    requested_model = body.get("model", "")

    max_tokens = body.get("options", {}).get("num_predict", 2048)
    temperature = body.get("options", {}).get("temperature", 0.0)
    tools = body.get("tools") or []

    # Translate Ollama messages to internal format
    has_images = False
    internal_messages = []
    for msg in ollama_messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""  # explicit null is legal, see parse_messages
        msg_images = msg.get("images", [])

        if msg_images:
            has_images = True
            blocks = [{"type": "text", "text": content}]
            for img_b64 in msg_images:
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
            internal_messages.append({"role": role, "content": blocks})
        else:
            internal_messages.append({"role": role, "content": content})

    # Route to device
    slot = _route_request(has_images, requested_model)
    if slot is None:
        return jsonify({"error": "no model ready"}), 503

    # Tool calling is GPU/iGPU-only (see chat_completions). Only on a GPU slot do
    # we render tool specs into the prompt; on NPU/CPU we ignore `tools`.
    tools_active = bool(tools) and _tools_supported(slot)
    if tools_active:
        # Render tool specs + prior calls into the prompt. Images ride along
        # untouched — a screenshot plus tool specs is a legitimate agent turn
        # on a VLM slot.
        internal_messages = prepare_messages_for_tools(ollama_messages, tools)
    elif tools:
        print(f"{datetime.now():%H:%M:%S} -- [{slot.device_name}] [Ollama] "
              f"tools ignored (GPU-only feature)", flush=True)

    # Parse through same pipeline as OpenAI
    try:
        text_prompt, images, raw_messages = parse_messages(internal_messages, max_dim)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if has_images and slot.model_type == "llm":
        return jsonify({"error": f"model '{slot.model_name}' does not support images"}), 400

    try:
        slot.ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"Failed to reload model: {e}"}), 500

    # Capture a big agent prompt so the next startup can pre-warm it (no-op
    # unless --prewarm is set). Mirrors the OpenAI path at chat_completions —
    # clients on the plain Ollama API (pre-0.53 Copilot, Open WebUI, ...)
    # reach us through this handler, not /v1.
    if slot.model_type == "llm":
        _maybe_capture_prewarm(raw_messages)

    # Build generation config
    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_tokens
    if temperature and temperature > 0.01:
        gen.do_sample = True
        gen.temperature = temperature
    else:
        gen.do_sample = False
        gen.top_k = 1
    _opts = body.get("options", {})
    apply_penalties(gen,
                    repetition=_opts.get("repeat_penalty"),
                    frequency=_opts.get("frequency_penalty"),
                    presence=_opts.get("presence_penalty"))

    print(f"\n{datetime.now():%H:%M:%S} <- [{slot.device_name}] [Ollama] "
          f"{'image, ' if has_images else ''}{len(text_prompt)} chars"
          f"{' (stream)' if stream else ''}", flush=True)

    t0 = time.perf_counter()

    # VLM path (no streaming)
    if slot.model_type == "vlm":
        try:
            text = slot.generate_vlm(text_prompt, images, gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        elapsed = time.perf_counter() - t0
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
              f"{len(text)} chars in {elapsed:.1f}s", flush=True)

        message = {"role": "assistant", "content": text}
        if tools_active:
            text, tool_calls = parse_tool_calls(text, tools)
            message["content"] = text
            if tool_calls:
                # Ollama shape: arguments are an object, not a JSON string.
                message["tool_calls"] = [{
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.loads(tc["function"]["arguments"]),
                    }
                } for tc in tool_calls]
                print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
                      f"{len(tool_calls)} tool call(s): "
                      f"{', '.join(tc['function']['name'] for tc in tool_calls)}",
                      flush=True)

        return jsonify({
            "model": slot.model_name,
            "message": message,
            "done": True,
            "total_duration": int(elapsed * 1e9),
        })

    # LLM path. Tool turns are buffered (see chat_completions for why).
    if stream and not tools_active:
        return Response(
            _ollama_stream_chat(slot, raw_messages, gen, t0),
            mimetype="application/x-ndjson",
        )

    try:
        text = slot.generate_llm(raw_messages, gen)
    except Exception as e:
        return jsonify({"error": explain_genai_error(e, slot)}), 500

    elapsed = time.perf_counter() - t0
    ttft = (f", TTFT {slot.last_ttft_ms:.0f}ms"
            if slot.last_ttft_ms is not None else "")
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
          f"~{len(text.split())} tokens in {elapsed:.1f}s{ttft}", flush=True)

    message = {"role": "assistant", "content": text}
    if tools_active:
        text, tool_calls = parse_tool_calls(text, tools)
        message["content"] = text
        if tool_calls:
            # Ollama shape: arguments are an object, not a JSON string.
            message["tool_calls"] = [{
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]),
                }
            } for tc in tool_calls]
            print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
                  f"{len(tool_calls)} tool call(s): "
                  f"{', '.join(tc['function']['name'] for tc in tool_calls)}",
                  flush=True)

    final = {
        "model": slot.model_name,
        "message": message,
        "done": True,
        "total_duration": int(elapsed * 1e9),
    }
    if stream:
        # tools + stream: emit the buffered result as a single ndjson line.
        return Response(json.dumps(final) + "\n", mimetype="application/x-ndjson")
    return jsonify(final)


def _ollama_stream_chat(slot, raw_messages, gen, t0):
    """Ollama streaming: newline-delimited JSON (not SSE). Protocol layer over
    slot.stream_tokens() — a None marker after 120 s of silence aborts the
    stream, preserving this path's historical no-token timeout."""
    token_count = 0

    try:
        for token in slot.stream_tokens(raw_messages, gen, heartbeat=120,
                                        tag="[Ollama] "):
            if token is None:
                break
            if token_count == 0:
                # Wall-clock TTFT: prefill is over when the first token lands.
                slot.last_ttft_ms = (time.perf_counter() - t0) * 1000
            token_count += 1
            yield json.dumps({
                "model": slot.model_name,
                "message": {"role": "assistant", "content": token},
                "done": False,
            }) + "\n"

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0

        yield json.dumps({
            "model": slot.model_name,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "total_duration": int(elapsed * 1e9),
            "eval_count": token_count,
        }) + "\n"
    finally:
        slot._cancel.set()

    ttft = (f", TTFT {slot.last_ttft_ms:.0f}ms" if token_count and
            slot.last_ttft_ms is not None else "")
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
          f"{token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s{ttft})", flush=True)


@ollama_app.route("/api/generate", methods=["POST"])
def ollama_generate():
    """Single-turn completion (no chat history)."""
    if overall_status() != "ready":
        return jsonify({"error": "model not ready"}), 503

    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "")
    stream = body.get("stream", True)
    requested_model = body.get("model", "")
    max_tokens = body.get("options", {}).get("num_predict", 2048)
    temperature = body.get("options", {}).get("temperature", 0.0)

    # Images in generate endpoint
    images_b64 = body.get("images", [])
    has_images = bool(images_b64)

    slot = _route_request(has_images, requested_model)
    if slot is None:
        return jsonify({"error": "no model ready"}), 503

    try:
        slot.ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"Failed to reload model: {e}"}), 500

    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_tokens
    if temperature and temperature > 0.01:
        gen.do_sample = True
        gen.temperature = temperature
    else:
        gen.do_sample = False
        gen.top_k = 1
    _opts = body.get("options", {})
    apply_penalties(gen,
                    repetition=_opts.get("repeat_penalty"),
                    frequency=_opts.get("frequency_penalty"),
                    presence=_opts.get("presence_penalty"))

    t0 = time.perf_counter()

    # VLM with images
    if has_images and slot.model_type == "vlm":
        img_tensors = []
        for b64 in images_b64:
            img = load_image(f"data:image/jpeg;base64,{b64}", max_dim)
            img_tensors.append(pil_to_tensor(img, max_dim))
        try:
            text = slot.generate_vlm(prompt, img_tensors, gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        elapsed = time.perf_counter() - t0
        return jsonify({
            "model": slot.model_name,
            "response": text,
            "done": True,
            "total_duration": int(elapsed * 1e9),
        })

    if has_images and slot.model_type == "llm":
        return jsonify({"error": "model does not support images"}), 400

    # Text-only generate → wrap as single-turn chat
    raw_messages = [{"role": "user", "content": prompt}]

    if stream and slot.model_type == "llm":
        return Response(
            _ollama_stream_generate(slot, raw_messages, gen, t0),
            mimetype="application/x-ndjson",
        )

    # Non-streaming
    if slot.model_type == "vlm":
        try:
            text = slot.generate_vlm(prompt, [], gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        try:
            text = slot.generate_llm(raw_messages, gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elapsed = time.perf_counter() - t0
    return jsonify({
        "model": slot.model_name,
        "response": text,
        "done": True,
        "total_duration": int(elapsed * 1e9),
    })


def _ollama_stream_generate(slot, raw_messages, gen, t0):
    """Ollama /api/generate streaming. Protocol layer over slot.stream_tokens();
    None after 120 s of silence aborts (historical behavior, no TTFT here)."""
    token_count = 0

    try:
        for token in slot.stream_tokens(raw_messages, gen, heartbeat=120,
                                        tag="[Ollama] "):
            if token is None:
                break
            token_count += 1
            yield json.dumps({
                "model": slot.model_name,
                "response": token,
                "done": False,
            }) + "\n"

        elapsed = time.perf_counter() - t0
        yield json.dumps({
            "model": slot.model_name,
            "response": "",
            "done": True,
            "total_duration": int(elapsed * 1e9),
            "eval_count": token_count,
        }) + "\n"
    finally:
        slot._cancel.set()


# Stubs — clients expect these to exist
@ollama_app.route("/api/pull", methods=["POST"])
def ollama_pull():
    return jsonify({"status": "success"})


@ollama_app.route("/api/delete", methods=["DELETE"])
def ollama_delete():
    return "", 200


@ollama_app.route("/api/copy", methods=["POST"])
def ollama_copy():
    return "", 200


# Copilot Chat 0.53+ sends actual chat via /v1/chat/completions on the Ollama
# port rather than /api/chat — delegate to the same handler.
@ollama_app.route("/v1/chat/completions", methods=["POST"])
def ollama_v1_chat_completions():
    return chat_completions()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

class _ExclusiveThreadedWSGIServer(ThreadedWSGIServer):
    """Exclusive on the port, and listening on IPv4 *and* IPv6."""

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # Werkzeug enables SO_REUSEADDR by default. On Windows that permits
            # a later, more-specific bind to split traffic on the same port.
            self.allow_reuse_address = False
            self.socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
            )
        if self.socket.family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
            # Serve both families from one socket. This is not a nicety: on
            # Windows getaddrinfo("localhost") returns ::1 FIRST, so against an
            # IPv4-only listener every localhost client attempts IPv6, waits for
            # the connect to fail, and only then falls back to 127.0.0.1.
            # Measured on Windows 11 + Python urllib (2026-08-18, Arc B60 box):
            # 2.05 s per request via localhost vs 0.015 s via 127.0.0.1 — a
            # fixed ~2 s tax that looks exactly like slow prefill and hides in
            # TTFT. start-openclaw.ps1 and the startup banner both hand out
            # localhost URLs, so the agent path paid it on every turn.
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def _serve_app(flask_app, port):
    # "::" with V6ONLY off accepts IPv4 and IPv6 alike. Fall back to IPv4-only
    # where IPv6 is disabled outright, rather than failing to start.
    try:
        server = _ExclusiveThreadedWSGIServer("::", port, flask_app)
    except OSError as e:
        print(f"  [net] IPv6 bind failed ({e}); IPv4 only — clients using "
              f"'localhost' may see a ~2s per-request delay, use 127.0.0.1",
              flush=True)
        server = _ExclusiveThreadedWSGIServer("0.0.0.0", port, flask_app)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def check_port(port):
    """True if nothing is serving on the port.

    Connect-test, NOT bind-test. A bind probe lies on Windows: a specific-
    address binding (Ollama's default is 127.0.0.1:11434) and a 0.0.0.0
    wildcard bind are treated as distinct, so bind("0.0.0.0", 11434)
    SUCCEEDS while real Ollama is running — and so does Flask's own bind
    right after. The result is two servers on one port: localhost clients
    reach Ollama (most-specific binding wins), LAN clients reach NoLlama,
    and which one answers depends on the caller's route. Asking "does
    anything accept a connection on any local address?" is the question we
    actually mean.
    """
    addresses = {"127.0.0.1"}
    try:
        addresses.update(
            info[4][0]
            for info in socket.getaddrinfo(
                socket.gethostname(), port, socket.AF_INET, socket.SOCK_STREAM
            )
            if info[4][0] != "0.0.0.0"
        )
    except socket.gaierror:
        pass  # loopback still catches wildcard and loopback-only listeners

    for address in addresses:
        try:
            with socket.create_connection((address, port), timeout=0.25):
                return False  # somebody answered
        except OSError:
            continue
    return True


def _identify_ollama(port):
    """Best-effort: is the process on <port> actually Ollama? Its root
    endpoint answers the plain-text 'Ollama is running'."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as r:
            return b"ollama" in r.read(200).lower()
    except OSError:
        return False


def detect_devices():
    """Return {kind: {"id": ov_id, "name": full_name}} of usable devices.

    kind is the canonical category ("NPU", "GPU", "CPU"). For "GPU", "id"
    may be "GPU.0", "GPU.1", etc. when OpenVINO enumerates multiple GPUs;
    callers must pass "id" to OpenVINO, not "kind".

    Non-Intel GPUs are filtered out: OpenVINO's intel_gpu plugin enumerates
    any OpenCL-capable GPU (NVIDIA, AMD), but its kernels only run on Intel
    hardware. Selecting a non-Intel GPU produces hundreds of compile errors
    and crashes at warmup with CL_INVALID_VALUE — better not to offer it.
    """
    devices = {}
    core = ov.Core()
    for d in core.get_available_devices():
        try:
            full_name = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            full_name = d
        if d.startswith("GPU"):
            if "intel" not in full_name.lower():
                continue
            if "GPU" not in devices:  # first Intel GPU wins
                devices["GPU"] = {"id": d, "name": full_name}
        elif d in ("NPU", "CPU"):
            devices[d] = {"id": d, "name": full_name}
    return devices


def _idle_watchdog(slots, idle_timeout, check_interval=30):
    """Background thread: unload slots that have been idle too long."""
    while True:
        time.sleep(check_interval)
        now = time.time()
        for slot in slots:
            if not slot or slot.status != "ready":
                continue
            # WhisperSlot has no last_used/unload — never idle-unloaded
            if getattr(slot, "last_used", None) is None:
                continue
            if now - slot.last_used < idle_timeout:
                continue
            # Try non-blocking lock acquire — skip if a request is in progress
            if not slot.lock.acquire(blocking=False):
                continue
            try:
                slot.unload()
            finally:
                slot.lock.release()


_banner_lock = threading.Lock()
_banner_printed = False


def _load_in_background(slot, model_dir, devices, port, ollama_port, banner_slots):
    """Background thread: load model + warmup on one device."""
    global _banner_printed
    try:
        slot.device_full = devices.get(slot.device_name, {}).get("name", slot.device_name)
        slot.load(model_dir)
        slot.warmup()
        _prewarm_slot(slot)
    except Exception as e:
        slot.status = "error"
        print(f"\n  [{slot.device_name}] ERROR: Failed to load model: {explain_genai_error(e, slot)}")
        if not any(s in str(e) for s in ("Could not find a model", "is truncated",
                                         "Compilation failed", "dynamic shape")):
            # Device-contention hint only where it's plausible — for a
            # missing/truncated model or a compiler failure it sends people
            # chasing ghosts (#17, #20 — the latter is literally titled
            # after this hint).
            print(f"  Is another process using the {slot.device_name}?", flush=True)

    # Print banner when all slots are done — only one thread wins
    with _banner_lock:
        if _banner_printed:
            return
        all_done = all(
            s.status in ("ready", "error", "not_configured")
            for s in banner_slots
        )
        if not all_done:
            return
        _banner_printed = True

    if any(s.status == "ready" for s in banner_slots):
        lines = []
        for s in banner_slots:
            if s.status == "ready":
                lines.append(f"    {s.device_name:5s}: {s.model_name} ({s.model_type.upper()}) "
                             f"-- {s.device_full}")
        url = f"http://localhost:{port}"
        api_lines = [f"    API  : {url}  (OpenAI)"]
        if ollama_port:
            api_lines.append(f"    API  : http://localhost:{ollama_port}  (Ollama)")
        print(f"""
================================================
  NoLlama ready
{chr(10).join(lines)}
{chr(10).join(api_lines)}
================================================
""", flush=True)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version",
                   version=f"NoLlama {__version__}")
    default_model = str(Path(__file__).parent / "model")
    p.add_argument("--model-dir", default=default_model,
                   help="Primary model directory (default: model/)")
    p.add_argument("--device", default="auto",
                   help="Device for primary model: NPU, GPU, CPU, or auto (default: auto)")
    p.add_argument("--gpu-model-dir", default=None,
                   help="Secondary GPU model (enables dual mode: NPU chat + GPU vision/LLM)")
    p.add_argument("--port", type=int, default=8000,
                   help="OpenAI API port (default: 8000)")
    # Note: the CLI flag is --ollama-port (dash), read back as args.ollama_port
    # (underscore) — argparse converts dashes to underscores in attribute names
    # by design, per Python convention. Same for every --two-word flag here.
    p.add_argument("--ollama-port", type=int, default=11434,
                   help="Ollama API port (default: 11434, 0 to disable)")
    p.add_argument("--max-dim", type=int, default=768,
                   help="Max image dimension before resize (default: 768)")
    p.add_argument("--whisper-dir", default=None,
                   help="Whisper model directory for speech-to-text (enables /v1/audio/transcriptions)")
    p.add_argument("--whisper-device", default="CPU",
                   help="Device for Whisper: CPU or GPU (default: CPU)")
    p.add_argument("--idle-timeout", type=int, default=1800,
                   help="Change idle-unload timeout in seconds "
                        "(default: 1800 = 30 min). Use 0 to disable unloading "
                        "(recommended for agent use; also auto-enables --prewarm).")
    p.add_argument("--debug", action="store_true",
                   help="Log every inbound API request (method, path, User-Agent, body)")
    p.add_argument("--vscode-compat", action="store_true",
                   help=f"Report an Ollama-shaped version ({VSCODE_OLLAMA_VERSION}) on "
                        f"/api/version so VS Code's Ollama client accepts the server. "
                        f"Its check rejects anything non-numeric, so our own dated "
                        f"version will not pass it")
    p.add_argument("--vscode-ollama-version", default=VSCODE_OLLAMA_VERSION,
                   metavar="X.Y.Z",
                   help=f"Version to report under --vscode-compat (default "
                        f"{VSCODE_OLLAMA_VERSION}). Raise it if VS Code starts "
                        f"demanding a newer Ollama")
    p.add_argument("--no-prompt-cache", action="store_true",
                   help="Disable prefix (KV) caching on GPU/CPU LLM slots. Caching is "
                        "ON by default — it prefills a repeated prompt prefix (e.g. an "
                        "agent's fixed system prompt) once instead of every turn.")
    p.add_argument("--cache-size-gb", type=int, default=None,
                   help=f"KV-cache pool size in GB when prefix caching is on. "
                        f"Default: auto — sized per device from its memory "
                        f"budget and the model's KV geometry (a third of what "
                        f"weights leave free, floor {AUTO_KV_MIN_GB} GB, cap "
                        f"~{AUTO_KV_TOKENS // 1024}k tokens). Pass a number "
                        f"to pin it.")
    p.add_argument("--prewarm", default=None, metavar="FILE",
                   help="Prefill a saved agent prompt at startup so the first turn is a "
                        "cache hit (no cold-prefill stall). The file auto-populates from the "
                        "first big prompt served, so: run once, then restart with --prewarm. "
                        "Auto-enabled (as prewarm-<port>.json) when --idle-timeout is 0.")
    p.add_argument("--no-prewarm", action="store_true",
                   help="Disable the automatic prewarm that --idle-timeout 0 turns on.")
    p.add_argument("--offload-ratio", type=int, default=0, metavar="PCT",
                   help="Stream PCT%% of MoE expert weights from disk instead of "
                        "keeping them GPU-resident (OpenVINO 2026.3+ disk offload). "
                        "Lets 30B-class MoE models run on 16 GB-class GPUs at the "
                        "cost of decode speed. Requires an XMX-capable Intel GPU "
                        "(Arc, Lunar Lake) — silently does nothing without one. "
                        "GPU slots only; 1-99.")
    p.add_argument("--scan", nargs="*", default=None, metavar="DIR",
                   help="Report what each model directory actually contains "
                        "(name, precision, architecture, integrity) and exit. "
                        "Searches the NoLlama directory and ~/models by "
                        "default; pass directories to search those instead.")
    p.add_argument("--backend", choices=("auto", "genai", "optimum"),
                   default="auto",
                   help="Runtime for LLM/VLM slots: openvino_genai pipelines "
                        "(genai), optimum-intel python runtime (optimum), or "
                        "pick per model (auto — optimum only for architectures "
                        "GenAI can't run: "
                        f"{', '.join(sorted(NEEDS_OPTIMUM))}).")
    return p.parse_args()


def main():
    global primary, secondary, whisper_slot, max_dim, debug, vscode_compat
    global PROMPT_CACHE, PROMPT_CACHE_GB, PREWARM_FILE, OFFLOAD_RATIO
    global VSCODE_OLLAMA_VERSION

    args = parse_args()

    # --scan is a report, not a server: no ports, no devices, no model load.
    if args.scan is not None:
        print(flush=True)
        scan_models(args.scan)
        return

    model_dir = os.path.expanduser(args.model_dir)
    max_dim = args.max_dim
    debug = args.debug
    vscode_compat = args.vscode_compat
    VSCODE_OLLAMA_VERSION = args.vscode_ollama_version
    PROMPT_CACHE = not args.no_prompt_cache
    PROMPT_CACHE_GB = args.cache_size_gb
    OFFLOAD_RATIO = max(0, min(99, args.offload_ratio))
    if OFFLOAD_RATIO:
        # Offload is a silent no-op without XMX — say so up front instead of
        # letting the user believe their model got smaller (see TODONT.md).
        try:
            caps = ov.Core().get_property("GPU", "OPTIMIZATION_CAPABILITIES")
            if "GPU_HW_MATMUL" not in caps:
                print("WARNING: --offload-ratio set, but this GPU has no XMX "
                      "(OPTIMIZATION_CAPABILITIES lacks GPU_HW_MATMUL). MoE disk "
                      "offload will silently do nothing — the model must fit in "
                      "GPU memory.", flush=True)
        except Exception:
            pass
    if args.prewarm:
        PREWARM_FILE = os.path.expanduser(args.prewarm)
    elif args.idle_timeout == 0 and not args.no_prewarm and PROMPT_CACHE:
        # Agent/server mode (--idle-timeout 0 keeps models loaded forever) is
        # exactly where prewarm pays off — and the idle unload that would
        # throw the warmed cache away can't happen, so turn it on. Port-scoped
        # filename so two instances on one install don't overwrite each other.
        PREWARM_FILE = os.path.join(SCRIPT_DIR, f"prewarm-{args.port}.json")
    else:
        PREWARM_FILE = None

    # Quiet Flask/Werkzeug startup noise: kills the "Serving Flask app" /
    # "Debug mode: off" / "Running on http://..." / "Press CTRL+C to quit"
    # block (printed twice — once per Flask app), the dev-server warning,
    # and per-request access logs that would otherwise flood the console
    # in normal use. nollama.py has its own app-level request logging.
    import logging
    import flask.cli
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    flask.cli.show_server_banner = lambda *a, **k: None

    print(flush=True)

    # 1. Check ports
    if not check_port(args.port):
        print(f"ERROR: Port {args.port} is already in use.")
        print(f"Use --port <number> to pick another port.")
        sys.exit(1)
    if args.ollama_port and not check_port(args.ollama_port):
        who = ("real Ollama is running there"
               if _identify_ollama(args.ollama_port)
               else "another process is using it")
        print(f"WARNING: Ollama port {args.ollama_port} is taken — {who}. "
              f"NoLlama's Ollama emulation disabled.")
        args.ollama_port = 0

    # 2. Detect devices
    devices = detect_devices()
    print("  Devices:", flush=True)
    for kind, info in devices.items():
        suffix = f" [{info['id']}]" if info['id'] != kind else ""
        print(f"    {kind}{suffix}: {info['name']}")
    print()

    def _id_of(kind):
        return devices.get(kind, {}).get("id", kind)

    # 3. Resolve primary device
    device = args.device.upper()
    if device == "AUTO":
        if args.gpu_model_dir:
            # Dual mode: primary goes on NPU (or CPU if no NPU)
            device = "NPU" if "NPU" in devices else "CPU"
        elif "NPU" in devices:
            device = "NPU"
        elif "GPU" in devices:
            device = "GPU"
        else:
            device = "CPU"

    if device not in devices and device != "CPU":
        print(f"ERROR: Device {device} not available. Found: {list(devices.keys())}")
        sys.exit(1)

    # 4. Verify model directories
    if not os.path.isdir(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)
    if args.gpu_model_dir and not os.path.isdir(args.gpu_model_dir):
        print(f"ERROR: GPU model directory not found: {args.gpu_model_dir}")
        sys.exit(1)
    if args.whisper_dir and not os.path.isdir(args.whisper_dir):
        print(f"ERROR: Whisper model directory not found: {args.whisper_dir}")
        sys.exit(1)

    # 4b. Pick the serving backend per model. GenAI pipelines are the default;
    # NEEDS_OPTIMUM architectures (or --backend optimum) go through the
    # optimum-intel python runtime instead.
    def _slot_class(mdir):
        if args.backend == "genai":
            return DeviceSlot
        if args.backend == "optimum":
            return OptimumSlot
        return OptimumSlot if needs_optimum(mdir) else DeviceSlot

    primary_cls = _slot_class(model_dir)
    secondary_cls = _slot_class(args.gpu_model_dir) if args.gpu_model_dir else None

    if OptimumSlot in (primary_cls, secondary_cls):
        # Fast wrong-venv check (find_spec only — no heavy imports): the
        # common failure is running from a plain install, and the load-time
        # error would otherwise arrive minutes later, mid-startup.
        import importlib.util
        missing = [m for m in ("optimum", "transformers")
                   if importlib.util.find_spec(m) is None]
        if missing:
            print(f"ERROR: this model needs the optimum-intel backend, but "
                  f"{'/'.join(missing)} is not installed in this python.")
            print("Serve from the model-lab venv instead — see README "
                  "(scripts\\glimmer-export builds it; one-time: "
                  "pip install flask openvino-genai).")
            sys.exit(1)

    if primary_cls is OptimumSlot and device == "NPU":
        if args.device.upper() == "NPU":
            print("ERROR: this model runs on the optimum-intel backend, which "
                  "has no NPU path. Use --device GPU or CPU.")
            sys.exit(1)
        # AUTO resolved to NPU — re-route to a device the backend can drive.
        device = "GPU" if "GPU" in devices else "CPU"
        print(f"  [auto] optimum-backend model — using {device} (NPU has no "
              f"optimum path)", flush=True)

    # 5. Create device slots
    primary = primary_cls(device, _id_of(device))
    all_slots = [primary]

    if args.gpu_model_dir:
        if "GPU" not in devices:
            print("WARNING: --gpu-model-dir given but no GPU detected. Ignoring.")
        else:
            secondary = secondary_cls("GPU", _id_of("GPU"))
            all_slots.append(secondary)

    if args.whisper_dir:
        whisper_device = args.whisper_device.upper()
        if whisper_device not in devices and whisper_device != "CPU":
            print(f"WARNING: Whisper device {whisper_device} not available, falling back to CPU.")
            whisper_device = "CPU"
        whisper_slot = WhisperSlot(whisper_device, _id_of(whisper_device))
        all_slots.append(whisper_slot)

    # 6. Start Flask, load models in background
    ports_msg = f"port {args.port}"
    if args.ollama_port:
        ports_msg += f" + Ollama on {args.ollama_port}"
    print(f"  NoLlama {__version__} starting on {ports_msg}...", flush=True)
    # The same port serves both — people wiring up an agent read the API
    # lines, people who just installed need to be told the chat UI exists.
    print(f"  Web UI (chat):  http://localhost:{args.port}/", flush=True)
    print(f"  OpenAI API:     http://localhost:{args.port}/v1", flush=True)
    if args.ollama_port:
        print(f"  Ollama API:     http://localhost:{args.ollama_port}", flush=True)
    else:
        # Off for one of two reasons: --ollama-port 0, or the port was busy
        # (real Ollama running — the earlier WARNING says so). Print the state
        # either way: an omitted line is not information.
        print("  Ollama API:     disabled", flush=True)

    threads = []
    t = threading.Thread(
        target=_load_in_background,
        args=(primary, model_dir, devices, args.port, args.ollama_port, all_slots),
        daemon=True,
    )
    threads.append(t)

    if secondary:
        t2 = threading.Thread(
            target=_load_in_background,
            args=(secondary, args.gpu_model_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(t2)

    if whisper_slot:
        tw = threading.Thread(
            target=_load_in_background,
            args=(whisper_slot, args.whisper_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(tw)

    for t in threads:
        t.start()

    # Idle watchdog — unload models after inactivity
    if args.idle_timeout > 0:
        print(f"  Idle unload after {args.idle_timeout}s of inactivity", flush=True)
        if PREWARM_FILE:
            # The prefix cache lives in the pipeline; unload drops both and the
            # reload path doesn't re-warm (a synchronous re-warm would stall
            # the triggering request for the whole prefill).
            print(f"  WARNING: --prewarm + idle unload: the warmed cache is lost "
                  f"when the model idle-unloads and not rebuilt until restart. "
                  f"Use --idle-timeout 0 to keep it.", flush=True)
        watchdog = threading.Thread(
            target=_idle_watchdog,
            args=(all_slots, args.idle_timeout),
            daemon=True,
        )
        watchdog.start()
    elif PREWARM_FILE and not args.prewarm:
        if primary_cls is OptimumSlot and secondary_cls is not DeviceSlot:
            # No GenAI LLM slot to warm — prompts are still captured to the
            # file (useful if the model later moves to a GenAI export), but
            # nothing prefills at startup. Don't promise otherwise.
            print(f"  Prewarm: inert on the optimum backend (no prefix cache); "
                  f"prompts still captured to "
                  f"{os.path.basename(PREWARM_FILE)}", flush=True)
        else:
            print(f"  Prewarm auto-enabled (--idle-timeout 0): "
                  f"{os.path.basename(PREWARM_FILE)} (--no-prewarm to disable)",
                  flush=True)

    # Suppress Flask's default "Serving Flask app" banner — we have our own
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    # Start Ollama API on separate port in background thread
    if args.ollama_port:
        print(f"  Ollama API on port {args.ollama_port}", flush=True)
        def _run_ollama():
            try:
                _serve_app(ollama_app, args.ollama_port)
            except SystemExit:
                print(f"  WARNING: Ollama API failed to claim port "
                      f"{args.ollama_port}. NoLlama's Ollama emulation disabled.",
                      flush=True)
            except Exception as e:
                print(f"  WARNING: Ollama API failed to start: {e}", flush=True)
        ollama_thread = threading.Thread(target=_run_ollama, daemon=True)
        ollama_thread.start()

    # OpenAI API on main thread
    print(f"  OpenAI API on port {args.port}", flush=True)
    _serve_app(app, args.port)


if __name__ == "__main__":
    main()
