"""Minimal upstream repro: one WhisperPipeline, one clip, one device.

    python scripts/whisper-npu-repro.py <model-dir> <device> <clip.wav>

Prints the versions, the device and NPU driver/architecture, compile time,
then the transcript or the exception. Bare openvino_genai, no NoLlama, so the
output can be pasted into an upstream issue as is. Written for T-050:
OpenVINO/whisper-medium-int8-ov dies with ZE_RESULT_ERROR_DEVICE_LOST on
NPU 4 and runs on NPU 3. The clip must be 16 kHz mono (jfk.wav from
whisper.cpp's samples/ is).
"""
import sys
import time

import openvino as ov
import openvino_genai as og
import soundfile as sf


def main():
    """Compile, transcribe once, print everything an issue needs.

    Why: an upstream report needs the shortest script that fails on their
    published model, with versions and driver in the same output, so a reader
    cannot ask for any of it.

    In: argv as in the module docstring. Out: text on stdout. A native crash
    (the NPU 4 case) can kill the process after "generate ..." is printed;
    that missing last line is the result, so every line flushes.
    """
    model_dir, device, clip = sys.argv[1:4]
    core = ov.Core()
    print("openvino", ov.__version__, "| genai", og.__version__, flush=True)
    print("device", device, core.get_property(device, "FULL_DEVICE_NAME"), flush=True)
    if device == "NPU":
        print("NPU_DRIVER_VERSION", core.get_property("NPU", "NPU_DRIVER_VERSION"),
              "| DEVICE_ARCHITECTURE", core.get_property("NPU", "DEVICE_ARCHITECTURE"), flush=True)
    audio, sr = sf.read(clip, dtype="float32")
    assert sr == 16000, sr
    t = time.time()
    pipe = og.WhisperPipeline(model_dir, device)
    print(f"compiled in {time.time() - t:.1f} s", flush=True)
    cfg = pipe.get_generation_config()
    cfg.task = "transcribe"
    cfg.language = "<|en|>"
    print("generate ...", flush=True)
    t = time.time()
    try:
        text = str(pipe.generate(audio.tolist(), cfg)).strip()
        print(f"OK in {time.time() - t:.2f} s: {text}", flush=True)
    except Exception as e:
        print("FAILED:", repr(e), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
