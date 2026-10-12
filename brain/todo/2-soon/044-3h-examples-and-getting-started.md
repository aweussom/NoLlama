# T-044 — `examples/` and a getting-started page

**Effort:** 3 h · **Produces:** `examples/*.py`, `docs/GETTING-STARTED.md`,
the transcription section `docs/API.md` lacks · **Area:** docs, clients

We document flags and endpoints, and we have benchmark scripts, but nothing a
newcomer can copy and run against their own server. OpenAI and Ollama APIs are
well known; that is not the same as "here is the five-line call for NoLlama,
and it works". Prompted 2026-10-11 by a Facebook thread on STM32N6 speech
recognition: someone asked why that isn't possible on a notebook. It is, and
there is nothing to point him at.

## The scripts

One task per file, under ~40 lines, docstring on top, `--base-url` flag,
defaults to `http://localhost:8000`:

| Script | Shows |
|---|---|
| `chat.py` | `openai` client, one turn, then streaming |
| `chat_stdlib.py` | the same with `urllib` only — no installs |
| `vision.py` | image + question, base64 |
| `transcribe.py` | WAV to text; `--mic` records 5 s first |
| `embed_rag.py` | embed 5 sentences, answer from the nearest one |
| `tools.py` | one tool-call round trip (GPU/CPU only, never NPU) |
| `ollama_client.py` | the same chat through the `ollama` package on 11434 |

No LangChain or other framework examples: they rot fastest, and anyone who
has the plain API working gets there in two minutes.

## Done when

**In progress 2026-10-12.** All seven scripts are written in `examples/`,
**none run yet**: the first test hit T-046 (idle reload stuck). Client deps
for testing: a scratch venv with `openai ollama sounddevice numpy`; the
server venv has none of them, so `chat_stdlib.py` is the only one it runs.
Watch on first run: `ollama_client.py` streams Qwen3's `<think>` into
`content` (the Ollama surface keeps tags inline) — decide whether to pass
`think=False` or strip it.

Server for the full run (vision + embed + whisper in one process):

```powershell
.\venv\Scripts\python nollama.py --model-dir $HOME\models\Qwen3-VL-8B-Instruct-int4-ov --device GPU --embed-model-dir $HOME\models\nomic-embed-text-v1.5-ov --whisper-dir $HOME\models\whisper-small-int8-ov --whisper-device NPU
```

- [ ] Every script has run against a live server, on the laptop, before
      commit. A `examples/check.ps1` runs them all, for after an upgrade.
- [ ] `docs/GETTING-STARTED.md`: install → start → first curl → first
      script, under five minutes, then a table pointing at `examples/`.
- [ ] README links it near the top, plus a speech paragraph: Whisper on the
      NPU, push-to-talk, with the 2026-10-12 numbers from `STATUS.md`.
- [ ] `transcribe.py --mic` demonstrated on the laptop NPU — needs Tommy's
      voice.
