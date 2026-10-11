"""Bare openvino_genai WhisperPipeline over the test set, one model x one device.

    python scripts/whisper-probe.py <model-dir> <device> <set-dir> <out.jsonl>

The set comes from scripts/whisper-build-set.py. Run one process per model
and device: a failing NPU (ZE_RESULT_ERROR_DEVICE_LOST on whisper-medium)
takes the process down, and the next run should not inherit that.

Writes one JSON line per clip plus a header line (versions, compile time)
and a footer (per-group WER, total audio s / wall s). No NoLlama involved.
"""
import json
import re
import sys
import time
import unicodedata

import numpy as np
import openvino as ov
import openvino_genai as og
import soundfile as sf

model_dir, device, set_dir, out_path = sys.argv[1:5]
LANG_TOKEN = {"en": "<|en|>", "no": "<|no|>"}


def norm(s):
    """Normalise text to a word list for WER.

    Why: Whisper punctuates and capitalises freely; counting a comma as a
    word error would swamp the comparison.

    In: any string. Out: list of lowercased words, punctuation removed.
    Digits stay digits, so '47' against 'førtisju' still counts as wrong.
    """
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()


def wer(ref, hyp):
    """Word error rate by Levenshtein distance over normalised words.

    Why: stdlib-only, so the probe runs in the bare venv with no jiwer.

    In: reference and hypothesis strings. Out: float, can exceed 1.0 when
    the hypothesis is longer than the reference; None when the reference is
    empty (the silence clip), where any output is a hallucination.
    """
    r, h = norm(ref), norm(hyp)
    if not r:
        return None
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


def main():
    """Compile the pipeline, transcribe every clip, write header/records/footer.

    Why: the standing order is bare runtime first; this is that, for Whisper.

    In: argv as in the module docstring. Out: JSON lines in out.jsonl. A
    compile failure writes the header with compile_error and returns; a
    per-clip exception is recorded and the run continues, but a native
    crash ends the process with only the lines already flushed.
    """
    out = open(out_path, "w", encoding="utf-8")
    head = {"model": model_dir, "device": device, "openvino": ov.__version__,
            "genai": og.__version__}
    try:
        head["device_name"] = ov.Core().get_property(device, "FULL_DEVICE_NAME")
        if device == "NPU":
            head["npu_driver"] = ov.Core().get_property("NPU", "NPU_DRIVER_VERSION")
    except Exception as e:
        head["device_prop_error"] = repr(e)
    t = time.time()
    try:
        pipe = og.WhisperPipeline(model_dir, device)
    except Exception as e:
        head["compile_error"] = repr(e)[:2000]
        out.write(json.dumps(head) + "\n")
        print(json.dumps(head))
        return
    head["compile_s"] = round(time.time() - t, 2)
    out.write(json.dumps(head) + "\n")
    out.flush()
    print(json.dumps(head), flush=True)

    clips = [json.loads(l) for l in open(f"{set_dir}/manifest.jsonl", encoding="utf-8")]
    groups = {}
    for c in clips:
        a, sr = sf.read(f"{set_dir}/{c['path']}", dtype="float32")
        cfg = pipe.get_generation_config()
        cfg.task = "transcribe"
        cfg.max_new_tokens = 440
        if c["lang"] in LANG_TOKEN:
            cfg.language = LANG_TOKEN[c["lang"]]
        t = time.time()
        try:
            text = str(pipe.generate(a.tolist(), cfg)).strip()
            err = None
        except Exception as e:
            text, err = "", repr(e)[:500]
        dt = time.time() - t
        w = wer(c["ref"], text) if err is None else None
        rec = {"path": c["path"], "group": c["group"], "seconds": c["seconds"],
               "wall_s": round(dt, 3), "rtf": round(dt / max(c["seconds"], 0.01), 4),
               "wer": None if w is None else round(w, 4), "hyp": text, "error": err}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        g = groups.setdefault(c["group"], {"n": 0, "audio_s": 0, "wall_s": 0, "wers": [], "errors": 0})
        g["n"] += 1
        g["audio_s"] += c["seconds"]
        g["wall_s"] += dt
        if w is not None:
            g["wers"].append(w)
        if err:
            g["errors"] += 1
    foot = {"summary": {k: {"n": g["n"], "errors": g["errors"],
                            "rtf": round(g["wall_s"] / g["audio_s"], 4),
                            "mean_wer": round(float(np.mean(g["wers"])), 4) if g["wers"] else None}
                        for k, g in groups.items()}}
    out.write(json.dumps(foot) + "\n")
    print(json.dumps(foot), flush=True)


if __name__ == "__main__":
    main()
