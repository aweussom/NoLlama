# T-024 — Load GGUF directly; openvino-genai already can

**Effort:** 2 h · **Produces:** a verdict, and a `models.json` path if it holds ·
**Area:** model loading

`openvino_genai` can read a GGUF. That would let people bring a model they
already downloaded for Ollama or llama.cpp instead of converting, which is the
single biggest friction point in first-contact installs.

Unknowns to settle before this is worth building:

- [ ] does the NPU path accept it at all? The NPU export rule is channel-wise
      int4 with static shapes — a GGUF is neither.
- [ ] does prefix caching survive? If the CB backend refuses a GGUF-backed
      pipeline, GPU users lose the thing that makes agents usable.
- [ ] what does it cost at load, against a pre-exported IR?

Measure before promising. `docs/BENCHMARKS.md` now has the llama.cpp
comparison, which is the honest baseline for "why not just use their file".

## Done when

Either a documented "yes, with these limits" or a `TODONT.md` entry.
