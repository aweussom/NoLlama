# NoLlama

**Local LLM server for the full Intel stack.** NPU, ARC iGPU, ARC discrete, CPU.
OpenAI + Ollama APIs. One server, every Intel device.

No NVIDIA required. No Ollama install. No llama.cpp. **No problem.**

It detects your hardware, picks the best device, and speaks both OpenAI and
Ollama APIs — so any client that talks to either just works. It drives coding
agents too: OpenCode, Goose and VS Code Copilot Chat run against it with local
**tool-calling** on your Intel GPU or CPU.

![NoLlama in action](docs/images/nollama-demo.gif)

## Quick start

**No git?** Grab the [latest release ZIP](https://github.com/aweussom/NoLlama/releases/latest),
unzip, then double-click **`install-windows.bat`** (Windows) or run
**`./install-linux.sh`**. Both check for PowerShell 7 and Python 3.10+ and offer
to install what's missing.

Already have those, or cloned the repo:

```powershell
.\install.ps1     # detects hardware, pick a model, writes start.ps1
.\start.ps1       # then open http://localhost:8000
```

Needs Intel hardware, Python 3.10+, PowerShell 7+. Windows and Linux both work.
Details in [docs/DEVICES.md](docs/DEVICES.md).

## Use it from anything

Point any OpenAI client at `http://localhost:8000/v1`:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json"   -d '{"model":"model","messages":[{"role":"user","content":"Hi"}]}'
```

Ollama clients work too, on port 11434. Images, streaming, Whisper
transcription, and every flag: [docs/API.md](docs/API.md).

## What to expect

**On a GPU or iGPU: usable now.** Qwen3-8B runs at conversational speed, a 30B
MoE fits a 24 GB card, and both drive agent loops with tool-calling. **CPU works
too**, slower — and on a strong desktop it beats a weak iGPU.

**On the NPU: real, but narrow.** Qwen3-8B does run there, at 10 tok/s;
SmolLM3-3B at 23 is genuinely pleasant for bounded work — summarise this,
extract these fields. Speed is not the binding limit though. The NPU caps
prompts at **4096 tokens** and has **no tool-calling and no prefix cache**, so
it cannot drive a coding agent at all. Treat it as a short-document and
side-task device, not a daily driver. That gap is closing fast, and the NPU is
the part of this that gets interesting.

Below ~3B, expect trouble. Asked for the capital of Norway, a 1.5B once answered
that "Norway is a small island". Fine for testing the plumbing.

**Check which NPU generation you have.** A model can be correct on one and
silently wrong on the next. The LFM2 1.2B builds decode at full speed on Lunar
Lake (**NPU 4**, Core Ultra 200V) and return word salad, while the same files on
Meteor/Arrow Lake (**NPU 3**) answer correctly — across two NPU drivers, three
OpenVINO versions and both compilers
([openvino#38100](https://github.com/openvinotoolkit/openvino/issues/38100)).
Nothing about the failure looks like a failure: it compiles, it is fast, and the
sentences are fluent. The installer reads your generation and keeps the models
that cannot work on it off the menu, and the server warns if you load one
anyway — but if you are picking models by hand,
[docs/MODELS.md](docs/MODELS.md) carries the per-generation list.

## Speed at a glance

Steady-state decode, tok/s, int4 weights, `count 1-100` test. Every number is a
real run on hardware named below.

| Model (int4) | NPU | iGPU | Arc dGPU | CPU (DDR5) | CPU (DDR4) |
|---|---|---|---|---|---|
| SmolLM3-3B (~2 GB) | 23.3 ᵃ | 29.7 ᵃ | **81.9** ᵈ | 37.5 ᵃ | 23.0 ᵇ |
| Qwen3-8B (~5 GB) | 10.0 ᵃ | 24.7 ᵉ / 21.7 ᶜ / 15.4 ᵃ | **65.9** ᵈ | 17.8 ᵃ | *wanted* |
| Qwen3-30B-A3B MoE (~17 GB) | n/a | **52.7** ᵉ / 25.3 ᶜ (offload 30) | **52.8** ᵈ | 23.7 ᵃ / 21.1 ᵉ | — |

ᵃ Core Ultra 9 **285K** desktop, DDR5-6400 · ᵇ Ryzen 9 **5950X**, DDR4 ·
ᶜ Core Ultra 7 **258V** laptop, Arc 140V on LPDDR5X · ᵈ **Arc Pro B60** 24 GB dGPU ·
ᵉ Core Ultra X7 **358H** (Panther Lake), Arc B390 iGPU, 64 GB LPDDR5X-8533, Linux
— community reports from a laptop and a mini-PC, issues #24 and #32

Decode ≈ memory bandwidth ÷ active weight bytes, so the memory column predicts
the table better than the device column. Treat cells as ±10%. The 30B fits
resident on the B60 but not the 140V, which had to stream experts to run at all.
On the B390's 64 GB of shared LPDDR5X it fits resident too — and an iGPU then
decodes it as fast as the discrete B60, which is the bandwidth rule again.

Full methodology, MoE disk offload, and the Ollama and RTX 5090 comparisons:
**[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

## Recommended models

| Use-case | Pick in the menu | HuggingFace | Size |
|---|---|---|---|
| NPU chat | Qwen3 8B (INT4-CW) | `OpenVINO/Qwen3-8B-int4-cw-ov` | ~5 GB |
| GPU vision | Qwen3-VL 8B (INT8) | `OpenVINO/Qwen3-VL-8B-Instruct-int8-ov` | ~9 GB |
| Coding agent | Qwen3 Coder 30B-A3B MoE (INT4) | `OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov` | ~17 GB |

All pre-exported — no conversion. `install.ps1` offers these; the menu adapts to
the devices it finds. The coding-agent pick needs a 24 GB GPU or a 32 GB laptop:
on a 16 GB machine **Qwen3-14B works, slowly** (3-17 minutes per small task on
a Lunar Lake iGPU, ~13 GB of GPU memory -- see docs/MODELS.md). More, plus how to convert anything from HuggingFace:
**[docs/MODELS.md](docs/MODELS.md)**.

## When to use NoLlama, and when to use Ollama

NoLlama isn't trying to replace Ollama. It covers the Intel devices Ollama
doesn't reach. Pick per device, not per project:

| Run on | Use | Why |
|---|---|---|
| **Intel NPU** | **NoLlama** | Ollama can't target it at all. This is the reason NoLlama exists — within the 4096-token, no-tool-calling limits above. |
| **Intel iGPU / ARC**, text | **NoLlama** | OpenVINO INT4 is ~1.6× faster on decode than Ollama's Vulkan on an Arc 140V. Ollama also needs `OLLAMA_IGPU_ENABLE=1` or it silently falls back to CPU. |
| **Intel iGPU / ARC**, images | **NoLlama** | Ollama has no Intel path for local vision models. |
| **CPU only** | **Ollama** | llama.cpp's CPU backend is more mature, and `ollama pull` beats model conversion. |
| **NVIDIA or AMD GPU** | **Ollama** | NoLlama is Intel-only by design. |

They coexist: Ollama keeps port 11434, NoLlama notices and disables its own
Ollama shim.

## Coding agents

OpenCode, Goose and VS Code Copilot Chat all work, with tool-calling on GPU or CPU
(never the NPU — it has a hard prompt cap). Prefix caching is on by default, so
an agent's fixed system prompt is prefilled once rather than every turn.

The installer writes an `opencode.json` for you. Two shapes, depending on the
use-case you pick:

- **"Coding agent"** → *one server*, the coder. OpenCode's side requests
  (session titles) go to the coder too: they cost under a second there, so a
  second server would only spend RAM.
- **"Chat + Coding agent"** → *dual mode*: **one** server, one port, two
  devices — the coder on the GPU for OpenCode, and a chat model on the NPU or
  CPU for everything else, addressed as `<model>@GPU` and `<model>@NPU`.

Either way: `.\launch-agent.ps1 -Path <your project>` starts the server if it
is down, waits for the model to load, and opens OpenCode against it. Nothing is
written to your machine — the config reaches OpenCode through `OPENCODE_CONFIG`,
which layers on top of your own settings instead of replacing them.

**On a discrete Intel GPU, leave `--gpu-keepalive` on** (the default). Windows
evicts an idle dGPU's VRAM to host RAM after ~80 seconds and copies it back on
the next request; measured on an Arc Pro B60, that left the model evicted 95% of
an interactive session. See [docs/dev/machines.md](docs/dev/machines.md).

Setup for all of them: **[docs/AGENTS.md](docs/AGENTS.md)**.

## Documentation

| | |
|---|---|
| [docs/MODELS.md](docs/MODELS.md) | Model registry, downloading, converting, `--scan`, brand-new architectures, the `-Nightly` runtime |
| [docs/API.md](docs/API.md) | OpenAI + Ollama endpoints, all CLI flags, prefix caching, the repetition penalty (`nollama.ini`), web UI |
| [docs/DEVICES.md](docs/DEVICES.md) | Per-device support and requirements |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Methodology, all numbers, MoE offload, vs Ollama and OVMS |
| [docs/AGENTS.md](docs/AGENTS.md) | OpenCode, Goose and Copilot setup — including the NPU as the agent's side-task model |
| [docs/INTERNALS.md](docs/INTERNALS.md) | Architecture, file layout, known limitations |
| [docs/DOCKER.md](docs/DOCKER.md) | Running in a container — GPU/CPU only, no NPU; what it costs and the two defects that bite |
| [docs/DIAGRAMS.md](docs/DIAGRAMS.md) | Logic diagrams of the core flows (request routing, slot lifecycle, token streaming, web UI rendering) — machine-checked against the code, so a diagram that's fallen behind gets flagged, not trusted |

## Known limitations

- NPU prompts cap at 4096 tokens; no vision, no tool-calling there.
- On the Ollama API, tool-enabled turns are still buffered (the OpenAI
  endpoint streams them, gated at the tool-call block).
- Big agent prompts prefill slowly on weak iGPUs. Use a smaller coder model, or
  CPU on a strong desktop.
- Thinking models can spend their whole token budget in `<think>`.
- Containers get GPU and CPU, never the NPU — it is not exposed to WSL 2 or
  to Linux containers on Windows ([docs/DOCKER.md](docs/DOCKER.md)).

The full list, with the reasoning: [docs/INTERNALS.md](docs/INTERNALS.md).

## License

MIT.

## Author

Tommy Leonhardsen ([@aweussom](https://github.com/aweussom))
