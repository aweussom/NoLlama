r"""Exercise a model through bare openvino_genai, with NoLlama out of the loop.

Why: when a model misbehaves under the server, the server is the one thing a
server-side test cannot rule out. Phi-3.5-vision was written off as broken
after a day of NoLlama-mediated tests that each correctly eliminated
something else; driving VLMPipeline directly found the real cause (our own
repetition penalty) in minutes. CLAUDE.md makes running this first a
standing order for any new or suspect model.

It also sweeps the generation settings that have actually bitten us rather
than only the happy path, because "it works" from a bare pipeline with
default settings is a weaker claim than it looks: a default GenerationConfig
applies no penalties, and the penalties are where the bodies were buried.

    venv\Scripts\python scripts\bare-probe.py ~\models\some-model-int4-ov
    venv-nightly\Scripts\python scripts\bare-probe.py <dir>        # runtime axis
    venv\Scripts\python scripts\bare-probe.py <dir> --device CPU
    venv\Scripts\python scripts\bare-probe.py <dir> --image shot.png

Out: stack versions, model kind, then one row per case. Always exits 0 --
this reports, it does not assert, so a failing row is data rather than a
broken run.
"""
import argparse
import os
import sys
import traceback

import numpy as np
import openvino as ov
import openvino_genai as ovg

# The settings that have actually caused trouble, not a full sweep. Keep this
# list short enough that people run it: every entry costs a generate.
CASES = (
    ("default (no penalty)", {}),
    ("repetition_penalty=1.0", {"repetition_penalty": 1.0}),
    ("repetition_penalty=1.05", {"repetition_penalty": 1.05}),
    ("presence_penalty=0.5", {"presence_penalty": 0.5}),
    ("frequency_penalty=0.5", {"frequency_penalty": 0.5}),
    ("sampled (temp 0.7)", {"do_sample": True, "temperature": 0.7, "top_p": 0.9}),
)


def is_vlm(model_dir):
    """True when the directory holds a vision-language export.

    Why not reuse nollama.is_vlm: importing nollama would defeat the entire
    point of this script, which is to prove the runtime's behaviour with our
    code absent. The check is deliberately duplicated -- do NOT unify it.

    In: a directory path. Out: bool, from the presence of the split language
    -model IR that VLM exports carry and text-only ones do not.
    """
    return os.path.isfile(os.path.join(model_dir, "openvino_language_model.xml"))


def synthetic_image(width=336, height=336):
    """A plain two-tone RGB block as an NHWC uint8 ov.Tensor.

    Why synthetic by default: a repro that needs an attached photo is one
    nobody runs. Image *content* has never mattered to any failure we have
    chased here -- verified against a real 807x552 screenshot and synthetic
    squares from 336 to 2048 px on Phi-3.5-vision, all identical.

    In: pixel dimensions. Out: an ov.Tensor shaped (1, H, W, 3).
    """
    arr = np.zeros((1, height, width, 3), dtype=np.uint8)
    arr[:, : height // 2, :, 0] = 200          # red top half
    arr[:, height // 2:, :, 2] = 200           # blue bottom half
    return ov.Tensor(np.ascontiguousarray(arr))


def load_image(path, max_dim=1024):
    """Load a real image file as an NHWC uint8 ov.Tensor, downscaled to fit.

    Why the downscale: matches what the server does before handing pixels to
    the pipeline, so a --image run stays comparable with a served request.

    In: a path to anything PIL opens. Out: an ov.Tensor. Raises if Pillow is
    missing, which is a real answer -- the venv is then not the one that
    serves models.
    """
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if max(img.width, img.height) > max_dim:
        ratio = max_dim / max(img.width, img.height)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)),
                         Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)[None, ...]
    return ov.Tensor(np.ascontiguousarray(arr))


def run_case(pipe, label, image, vlm, **cfg):
    """Generate once and print a single result row, failures included.

    Why it swallows everything: one config blowing up is the interesting
    output, not a reason to abandon the remaining rows. The first line of
    the exception carries the genai assertion, which is the part worth
    pasting into an issue.

    Why the two call shapes: LLMPipeline.generate takes the prompt
    positionally and rejects `prompt=` outright ("incompatible function
    arguments"), while VLMPipeline needs the keyword form to accept
    `images`. Passing the VLM shape to an LLM fails every row identically,
    which reads as a broken model rather than a broken probe.

    In: a live pipeline, a label, an ov.Tensor or None, whether the pipeline
    is a VLM, and GenerationConfig attributes. Out: None -- prints.
    """
    gen = ovg.GenerationConfig()
    gen.max_new_tokens = 20
    for key, value in cfg.items():
        setattr(gen, key, value)
    kind = "image" if image is not None else "text "
    try:
        if image is not None:
            out = pipe.generate(prompt="Describe this image.", images=image,
                                generation_config=gen)
        elif vlm:
            out = pipe.generate(prompt="Say hello.", generation_config=gen)
        else:
            out = pipe.generate("Say hello.", gen)
        text = out if isinstance(out, str) else getattr(out, "texts", [str(out)])[0]
        print(f"  {label:26} {kind}  OK    {str(text)[:52]!r}")
    except Exception as e:
        print(f"  {label:26} {kind}  FAIL  {str(e).strip().splitlines()[0][:96]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", help="Model directory (an OpenVINO export)")
    ap.add_argument("--device", default="GPU", help="GPU, CPU or NPU (default GPU)")
    ap.add_argument("--image", help="Real image to use instead of the synthetic one")
    args = ap.parse_args()

    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    print("openvino      :", ov.__version__)
    print("openvino_genai:", ovg.__version__)
    print("device        :", args.device)
    print("model         :", model_dir)
    try:
        print("device name   :",
              ov.Core().get_property(args.device, "FULL_DEVICE_NAME"))
    except Exception as e:
        print("device name   : <unavailable>", e)

    if not os.path.isdir(model_dir):
        print("\nNo such model directory.")
        return 0

    vlm = is_vlm(model_dir)
    print("kind          :", "VLM (vision + text)" if vlm else "LLM (text only)")

    print("\nloading...")
    try:
        pipe = (ovg.VLMPipeline if vlm else ovg.LLMPipeline)(model_dir, args.device)
    except Exception:
        traceback.print_exc()
        return 0

    image = None
    if vlm:
        image = load_image(args.image) if args.image else synthetic_image()

    print("cases:")
    for label, cfg in CASES:
        run_case(pipe, label, None, vlm, **cfg)
        if vlm:
            run_case(pipe, label, image, vlm, **cfg)

    print("\nAny FAIL above happened with NoLlama absent, so it is the runtime "
          "or the IR.\nIf every row is OK but the server misbehaves, the bug "
          "is ours.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
