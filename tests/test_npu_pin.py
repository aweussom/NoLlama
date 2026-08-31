"""Regression test for the NPU_PLATFORM pin (PR #34).

Guards the ordering of ``explain_genai_error`` so the ``AUTO_DETECT``
branch cannot be silently displaced below the broader
"``Compilation failed``" matcher that would otherwise hand the user
the wrong advice. The real driver-4841 failure surface contains both
strings; the AUTO_DETECT / ``--npu-platform`` hint must win.

Runs anywhere with the NoLlama venv — no NPU hardware required, and no pytest
either: pytest is not a NoLlama dependency, so this file puts the repo root on
sys.path and carries its own __main__ runner, the same shape as
tests/test_stream_tools.py. Both work:

    venv\\Scripts\\python tests\\test_npu_pin.py
    venv\\Scripts\\python -m pytest tests\\test_npu_pin.py   # where pytest exists
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nollama import explain_genai_error  # noqa: E402  (needs the path above)


def _exc(msg: str) -> RuntimeError:
    return RuntimeError(msg)


def test_auto_detect_wins_over_compilation_failed():
    # The real failure surface on the driver-4841 path: the genai loader
    # wraps the AUTO_DETECT failure in "Compilation failed". The broader
    # NPU "driver too old" hint would otherwise win and tell the user
    # the opposite of what they need (update the driver vs. pass the
    # --npu-platform override).
    real = (
        "Exception from src/plugins/intel_npu/src/plugin/src/plugin.cpp:576:\n"
        "Exception from src/plugins/intel_npu/src/compiler_adapter/src/compiler_impl.cpp:280:\n"
        "Compilation failed. vclAllocatedExecutableCreate4 result: 0x78000004\n"
        "  - [NPU_VCL] Compiler returned msg: Unsupported platform: 'AUTO_DETECT'"
    )
    assert "--npu-platform" in explain_genai_error(_exc(real))


def test_auto_detect_alone():
    # Synthetic, no "Compilation failed" prefix.
    synth = "NPU: Unsupported platform: 'AUTO_DETECT'"
    assert "--npu-platform" in explain_genai_error(_exc(synth))


def test_compilation_failed_without_auto_detect_still_uses_old_hint():
    # A genuine NPU compile failure (e.g. INT4 layout / model envelope)
    # must NOT be misrouted to the AUTO_DETECT / --npu-platform hint —
    # that would be a regression of the ordering guard above.
    real = (
        "Exception from src/plugins/intel_npu/src/plugin/src/plugin.cpp:576:\n"
        "Compilation failed. vclAllocatedExecutableCreate4 result: 0x78000004\n"
        "  - [NPU_VCL] INT4 node-naming bug (openvino#29823)"
    )
    msg = explain_genai_error(_exc(real))
    assert "NPU driver too old" in msg
    assert "--npu-platform" not in msg


def test_auto_detect_on_a_gpu_slot_does_not_get_npu_advice():
    # "AUTO_DETECT" is an OpenVINO-wide token, so the branch is gated on the
    # slot: a GPU-side message mentioning it must fall through to the generic
    # handling rather than telling the user to pass --npu-platform.
    class _Slot:
        device_name = "GPU"
        kv_pool_gb = 0

    msg = "Exception ... [GPU] plugin mentions AUTO_DETECT somewhere"
    assert "--npu-platform" not in explain_genai_error(_exc(msg), _Slot())
    # No slot to consult (many call sites) keeps the NPU hint.
    assert "--npu-platform" in explain_genai_error(_exc(msg))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as e:
                failures += 1
                print("FAIL", name, "-", e)
    sys.exit(1 if failures else 0)
