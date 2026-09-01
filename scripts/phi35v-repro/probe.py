r"""Minimal repro: repetition_penalty breaks VLM prompts whose image
placeholder ids fall outside [0, vocab_size).

Why no NoLlama in the loop: the failure was first seen through NoLlama's
serving path, and for a full day that framing hid the cause. Every test ran
through the server, so the server was the one variable never varied. This
drives openvino_genai directly with a synthetic image, so the result is
about the runtime and the IR alone -- and it is what found the bug.

Observed on Phi-3.5-vision-instruct-int4-ov, Arc 140V, genai 2026.3.0.0-3277
and 2026.5.0.0-3402: identical prompt and image, greedy, differing only in
repetition_penalty.

    venv\Scripts\python scripts\phi35v-repro\probe.py
    venv-nightly\Scripts\python scripts\phi35v-repro\probe.py   # runtime axis

Out: stack versions, then a row per (penalty, with/without image). Exit
status is 0 whatever happens -- this reports, it does not assert.
"""
import os
import sys
import traceback

import numpy as np
import openvino as ov
import openvino_genai as ovg

MODEL = os.environ.get(
    "PHI35V_DIR",
    os.path.expanduser("~/models/Phi-3.5-vision-instruct-int4-ov"),
)
DEVICE = os.environ.get("PHI35V_DEVICE", "GPU")


def synthetic_image(w=336, h=336):
    """A plain two-tone RGB block as an NHWC uint8 ov.Tensor.

    Why synthetic: a repro needing an attached photo is one the reader will
    not run. Nothing here depends on image content -- the failure is in how
    placeholder tokens are formed, not in what is depicted. Verified against
    a real 807x552 screenshot and synthetic squares from 336 to 2048 px:
    all behave identically.
    """
    arr = np.zeros((1, h, w, 3), dtype=np.uint8)
    arr[:, : h // 2, :, 0] = 200          # red top half
    arr[:, h // 2:, :, 2] = 200           # blue bottom half
    return ov.Tensor(np.ascontiguousarray(arr))


def case(pipe, label, with_image, **cfg):
    """Run one generate and print a single result row, failure included."""
    gen = ovg.GenerationConfig()
    gen.max_new_tokens = 20
    for key, value in cfg.items():
        setattr(gen, key, value)
    kind = "image" if with_image else "text "
    try:
        if with_image:
            out = pipe.generate(prompt="Describe this image.",
                                images=synthetic_image(), generation_config=gen)
        else:
            out = pipe.generate(prompt="Say hello.", generation_config=gen)
        text = out if isinstance(out, str) else getattr(out, "texts", [str(out)])[0]
        print(f"  {label:26} {kind}  OK    {str(text)[:52]!r}")
    except Exception as e:
        first = str(e).strip().splitlines()[0]
        print(f"  {label:26} {kind}  FAIL  {first[:96]}")


def main():
    print("openvino      :", ov.__version__)
    print("openvino_genai:", ovg.__version__)
    print("device        :", DEVICE)
    print("model         :", MODEL)
    try:
        print("GPU           :", ov.Core().get_property(DEVICE, "FULL_DEVICE_NAME"))
    except Exception as e:
        print("GPU           : <unavailable>", e)

    if not os.path.isdir(MODEL):
        print("\nMissing model. Fetch it first:")
        print("  .\\download-model.ps1 OpenVINO/Phi-3.5-vision-instruct-int4-ov")
        return 0

    print("\nloading...")
    try:
        pipe = ovg.VLMPipeline(MODEL, DEVICE)
    except Exception:
        traceback.print_exc()
        return 0

    print("cases:")
    for label, cfg in (
        ("default (no penalty)", {}),
        ("repetition_penalty=1.0", {"repetition_penalty": 1.0}),
        ("repetition_penalty=1.05", {"repetition_penalty": 1.05}),
        ("presence_penalty=0.5", {"presence_penalty": 0.5}),
        ("frequency_penalty=0.5", {"frequency_penalty": 0.5}),
    ):
        for with_image in (False, True):
            case(pipe, label, with_image, **cfg)
    print("\nExpected: only repetition_penalty > 1.0 combined with an image fails.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
