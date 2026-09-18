r"""The NPU-generation guard: a model that is correct on one NPU and wrong on another.

The defect this guards is silent. LFM2 int4-cw compiles, decodes at full speed
and returns fluent word salad on NPU 4, while the same files on NPU 3 are
correct — so nothing in the log, the tok/s or the exit status says anything is
wrong. Everything a user could check looks healthy.

Two halves have to agree or the guard is worse than useless: `models.json`
keeps these models off the installer menu, and `npu_arch_warning` catches the
copy already on disk that never passed the installer. `test_registry_and_code_agree`
is the one that fails when someone updates one and not the other.

The family test reads a REAL export's `config.json` rather than a fixture,
because what is pinned is that these files declare `model_type: lfm2` — a
re-export that renamed the type would slip past the guard silently, and a
fixture would happily keep passing.

Run:

    venv\Scripts\python -m pytest tests\test_npu_arch_guard.py -q
    venv\Scripts\python tests\test_npu_arch_guard.py       # no pytest needed

Skips cleanly when the LFM2 export is not on this box: a missing model is not
a failing convention.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nollama import NPU_ARCH_SUSPECT, npu_arch_warning  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DEVICE_ARCHITECTURE values, not marketing names. '3720' is Meteor/Arrow Lake
# (NPU 3), '4000' is Lunar Lake (NPU 4) — confirmed on the 258V laptop.
NPU3, NPU4 = "3720", "4000"

LFM2_CANDIDATES = [
    "LFM2.5-1.2B-Instruct-int4-cw-ov",
    "LFM2.5-350M-int8-ov",
    "LFM2-1.2B-int4-cw-ov",
]
CLEAN_CANDIDATES = ["Qwen3-8B-int4-cw-ov", "SmolLM3-3B-int4-cw-ov",
                    "Phi-3.5-mini-instruct-int4-cw-ov"]


def _find(names):
    """First of these model directories present under ~/models, else None."""
    root = os.path.expanduser("~/models")
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return None


def test_registry_and_code_agree():
    """models.json gates the menu, nollama.py gates the load — one truth.

    No model needed. If these drift, the installer hides a model the server
    then loads without a word, or the reverse.
    """
    reg = json.load(open(os.path.join(REPO, "models.json"), encoding="utf-8"))
    denied = set()
    for entry in reg["npu"]:
        for arch in entry.get("npu_arch_deny", []):
            denied.add(arch)
    in_code = {a for arches in NPU_ARCH_SUSPECT.values() for a in arches}
    assert denied == in_code, f"models.json {denied} vs nollama.py {in_code}"


def test_real_lfm2_export_declares_the_type_the_guard_matches():
    """[OBSERVED 2026-09-18] Both LFM2.5 exports carry model_type 'lfm2'.

    The guard keys on that string. A re-export under another type would make
    it a no-op, and only reading a real config.json can catch that.
    """
    model = _find(LFM2_CANDIDATES)
    if not model:
        print("SKIP: no LFM2 export under ~/models")
        return
    cfg = json.load(open(os.path.join(model, "config.json"), encoding="utf-8"))
    assert cfg.get("model_type", "").lower() in NPU_ARCH_SUSPECT, cfg.get("model_type")


def test_lfm2_warns_on_npu4_and_stays_quiet_on_npu3():
    """The whole point: same files, same call, opposite answers by generation."""
    model = _find(LFM2_CANDIDATES)
    if not model:
        print("SKIP: no LFM2 export under ~/models")
        return
    warning = npu_arch_warning(model, NPU4)
    assert warning is not None
    assert "38100" in warning, warning          # the upstream issue, so it is chaseable
    assert npu_arch_warning(model, NPU3) is None


def test_a_healthy_model_is_never_flagged():
    """Over-warning is the failure mode that gets a guard ignored."""
    model = _find(CLEAN_CANDIDATES)
    if not model:
        print("SKIP: no non-LFM2 export under ~/models")
        return
    assert npu_arch_warning(model, NPU4) is None
    assert npu_arch_warning(model, NPU3) is None


def test_unknown_platform_never_warns():
    """An unreadable DEVICE_ARCHITECTURE must not produce a guess."""
    model = _find(LFM2_CANDIDATES) or REPO
    assert npu_arch_warning(model, "") is None
    assert npu_arch_warning(model, None) is None


def test_missing_config_never_blocks_a_load():
    """A directory with no config.json is 'not known bad', never an exception."""
    with tempfile.TemporaryDirectory() as d:
        assert npu_arch_warning(d, NPU4) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
