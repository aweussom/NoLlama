# Device support

| Device | Examples | What it does | Streaming? |
|---|---|---|---|
| NPU (Intel AI Boost) | Core Ultra 7 258V, Core Ultra 9 285K | Text chat via LLMPipeline. Speech-to-text via WhisperPipeline (`--whisper-device NPU`; see [MODELS.md](MODELS.md#speech-to-text-whisper)). Low power, sustained workload sweet spot. | Yes |
| ARC iGPU | ARC 140V (Core Ultra) | Vision + text, or bigger LLM | Yes (VLM streams in 2026.1+) |
| ARC discrete | A770, B580 | Same as iGPU, more VRAM for larger models | Yes (VLM streams in 2026.1+) |
| CPU | Any x86-64 with AVX2 — Intel supported, **AMD works too** (measured: Ryzen 9 5950X on DDR4, 23 tok/s on a 3B) | Fallback for everything. On desktops with DDR5 and many cores, often *faster* than NPU — see benchmarks. Decode is memory-bandwidth-bound: DDR4 boxes should size models accordingly. | Yes |

### Intel only — by design

NoLlama is Intel-hardware-only and will stay that way. Non-Intel GPUs
(NVIDIA, AMD) are filtered out of device detection on purpose, even
though OpenVINO 2026 now ships an experimental NVIDIA plugin via
[`openvino-extensibility`](https://docs.openvino.ai/2026/documentation/openvino-extensibility/openvino-plugin-library/plugin.html).
That path drags CUDA/cuDNN into the stack — it's a developer-backend
extension, not a drop-in user feature, and it loses every reason
NoLlama exists in the first place (NPU-first, Intel-first, no CUDA).

If you have an NVIDIA GPU, **use Ollama**. Ollama will always do
Ollama better than NoLlama could, and that's the right tool for that
hardware. NoLlama's value is specifically the Intel NPU / ARC story
that Ollama doesn't tell.

One honest footnote: the **CPU path happens to run on any AVX2 x86**,
because OpenVINO's CPU plugin doesn't actually care whose name is on
the silicon. Measured on an AMD Ryzen 9 5950X with DDR4 (fresh
Windows 11, installed from the release ZIP): 23 tok/s, TTFT 360 ms, on
a small thinking model (SmolLM3-3B int4). Not supported, not tuned
for, works fine — but read the number honestly: decode is
memory-bandwidth-bound, DDR4 dual-channel is roughly half a DDR5
desktop's bandwidth, so a 3B at 23 tok/s is the expectation-setter,
not a promise about 30Bs. Non-Intel GPUs stay filtered out, but nobody
will stop your CPU.

## Requirements

- PowerShell 7+ (Windows PowerShell 5.1 is not supported; on Linux,
  see [Microsoft's install
  instructions](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-linux))
- Python 3.10+
- OpenVINO 2026.1+ with openvino-genai
- At least one of:
  - Intel Core Ultra (NPU + ARC iGPU)
  - Intel ARC discrete GPU (A770, B580, etc.)
  - Any Intel CPU (slower, but works)
- ~1-17 GB disk per model

`install.ps1` handles the venv, dependencies, and model download. It *is*
the installer on every platform — `install-windows.bat` and
`install-linux.sh` are thin entry-point shims that check the
prerequisites above (offering to install what's missing) and then hand
off to it. There is deliberately no separate Bash installer: on
Linux/macOS the shim runs `pwsh ./install.ps1` for you (flags like
`-HfToken` pass through); paths and link creation branch on
`$IsWindows`. Windows is the primary platform, but
**Linux is confirmed working** by user reports (Core Ultra 7 258V, NPU +
GPU detected — see [#6](https://github.com/aweussom/NoLlama/issues/6));
macOS is untested. On Linux, NPU and GPU detection needs the Intel
userspace drivers installed (`intel-npu-driver` for the NPU, the GPU
compute runtime for the iGPU) — without them only the CPU shows up. The
NPU Linux stack is less battle-tested than Windows.

## In a container, or through WSL

Not a supported configuration yet — the measurements behind this section
are in `DOCKER-INSTALL.md` (issue #31). Two things are worth knowing if you
try it anyway.

**Intel's own `openvino/ubuntu24_runtime` image may not see your GPU.** It
carries whatever compute-runtime it was built against, and a driver that
predates your card enumerates nothing at all — `CPU` only, no error worth
the name. Install the current release from
[intel/compute-runtime](https://github.com/intel/compute-runtime/releases)
into your image and check `available_devices` before believing anything
else.

**A GPU reached through WSL's `/dev/dxg` caps a single allocation at
1 GiB**, while still reporting the card's full memory as its total. Any
model with a tensor over that size — Gemma 4 E2B's per-layer embedding
table is 2.2 GB — fails at load with "Exceeded max size of memory object
allocation". NoLlama detects the mismatch (max-alloc below total budget)
and passes `GPU_ENABLE_LARGE_ALLOCATIONS`, printing a line when it does.
Models that need the hint load roughly 3x slower than native; models that
fit under the cap run at native speed. Native installs never see this —
they report the whole budget as allocatable and take no hint.

Correctness is not implied by either of those working. One model measured
so far — `gemma-4-26b-a4b-it-int4-ov` — loads cleanly on the container GPU
path and then generates deterministic gibberish, while being correct
natively, correct on CPU in the same container, and while every other model
tried on that path was correct. Check the output, not the device list.
