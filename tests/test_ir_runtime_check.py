r"""The IR-newer-than-runtime refusal (T-009).

The defect this guards has no error: a 2026.4-built IR under OpenVINO 2026.3.1
takes the process down in native code, GPU and CPU alike [OBSERVED 2026-09-23,
258V laptop, bare openvino_genai]. The check must fire before any pipeline is
constructed and must name the pip line.

The pure cases pin the comparison (major.minor, numeric). The real-data case
reads Intel's gemma-4 export on this box, the IR that first caused this, so
the test fails if Intel stops writing Runtime_version or changes its shape.
Skips cleanly when that model is not present.

Run:

    venv\Scripts\python -m pytest tests\test_ir_runtime_check.py -q
    venv\Scripts\python tests\test_ir_runtime_check.py      # no pytest needed
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nollama  # noqa: E402
from nollama import _runtime_pair, ir_runtime_mismatch, read_ir_rt_info  # noqa: E402

MODELS = os.path.expanduser("~/models")
GEMMA = os.path.join(MODELS, "gemma-4-E4B-it-int8-ov")


def test_runtime_pair_parses_both_shapes():
    assert _runtime_pair("2026.4.0-22959-99c81491cc3-releases/2026/4") == (2026, 4)
    assert _runtime_pair("2026.5.0-22942-597262ef181") == (2026, 5)
    assert _runtime_pair("2026.3.1") == (2026, 3)
    assert _runtime_pair(None) is None
    assert _runtime_pair("garbage") is None


def test_numeric_not_lexical():
    # "2026.10" < "2026.4" as strings; as a toolchain it is six releases later.
    assert _runtime_pair("2026.10.0") > _runtime_pair("2026.4.0")


def test_newer_ir_refused_with_pip_line(tmp_path=None):
    rt = {"Runtime_version": "2026.4.0-22959-99c81491cc3-releases/2026/4"}
    msg = ir_runtime_mismatch("unused", installed="2026.3.1-22476-x", rt=rt)
    assert msg and "pip install -U openvino openvino-genai openvino-tokenizers" in msg
    assert "2026.4.0-22959" in msg and "2026.3.1-22476" in msg
    assert "--ignore-ir-version" in msg


def test_same_or_older_ir_passes():
    rt = {"Runtime_version": "2026.4.0-22959-99c81491cc3-releases/2026/4"}
    assert ir_runtime_mismatch("unused", installed="2026.4.0-22959-x", rt=rt) is None
    assert ir_runtime_mismatch("unused", installed="2026.4.1-23000-x", rt=rt) is None
    assert ir_runtime_mismatch("unused", installed="2026.5.0-22942-nightly", rt=rt) is None
    old = {"Runtime_version": "2025.1.0-18503-6fec06580ab-releases/2025/1"}
    assert ir_runtime_mismatch("unused", installed="2026.4.0", rt=old) is None


def test_patch_level_does_not_refuse():
    # [INFERRED] patch releases share the IR format; only the generation gates.
    rt = {"Runtime_version": "2026.3.1-22476-759c5a6ab8c-releases/2026/3"}
    assert ir_runtime_mismatch("unused", installed="2026.3.0-22451-x", rt=rt) is None


def test_unreadable_passes():
    assert ir_runtime_mismatch("unused", installed="2026.4.0", rt={}) is None
    assert ir_runtime_mismatch("unused", installed="", rt={"Runtime_version": "2026.9.0"}) is None


def test_flag_bypasses():
    rt = {"Runtime_version": "2026.9.0-x"}
    nollama.IGNORE_IR_VERSION = True
    try:
        assert ir_runtime_mismatch("unused", installed="2026.4.0", rt=rt) is None
    finally:
        nollama.IGNORE_IR_VERSION = False
    assert ir_runtime_mismatch("unused", installed="2026.4.0", rt=rt)


def test_real_gemma4_export_is_a_2026_4_ir():
    if not os.path.isdir(GEMMA):
        print("  SKIP gemma-4: not present")
        return
    rt = read_ir_rt_info(GEMMA)
    assert _runtime_pair(rt.get("Runtime_version")) == (2026, 4), rt.get("Runtime_version")
    # The shipped failure: this IR under the 2026.3.1 runtime.
    assert ir_runtime_mismatch(GEMMA, installed="2026.3.1-22476-759c5a6ab8c-releases/2026/3", rt=rt)
    # And the installed runtime on this box, which is at or past the floor.
    assert ir_runtime_mismatch(GEMMA, rt=rt) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all passed")
