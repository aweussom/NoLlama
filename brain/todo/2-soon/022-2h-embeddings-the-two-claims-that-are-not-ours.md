# T-022 — Embeddings: close the two claims that are not ours

**Effort:** 2 h · **Produces:** a measurement and a client run ·
**Area:** `EmbedSlot`, `docs/dev/models.md`

Embeddings landed and were closed by a real client run — LangChain over FAISS
indexed this repo's docs (25 files, 483 chunks) through `/api/embed`, then
answered questions with Qwen3-8B on the iGPU from the same process, 3 of 3
correct and grounded. What is left is what that run did not touch.

- [ ] **The no-reindex claim is MyrkoF's, not ours.** 8/8 same top-1 against a
      60,748-vector Ollama-built index is his measurement with his build of
      nomic. Ours is a different conversion (the repo's fp16 ONNX) and nobody
      has compared the two. **Do not repeat the number as ours** until someone
      has.
- [ ] **NOMAD has never been pointed at NoLlama.** LangChain exercises the same
      two wire shapes, but NOMAD's *Remote Connection* screen and its
      `/api/tags` name lookup are what issue #43 actually asks about.
- [ ] **Indexing is slow on CPU** — ~1.7 chunks/s for nomic here, so a
      book-sized corpus is hours. The iGPU was slower still on short encoder
      passes. Worth trying `--embed-device GPU` with a LARGE slice, the one
      case where dispatch overhead might amortise. Untested.
- [ ] **`--embed-threads` carries a +32% measured on MyrkoF's 285H**, not here.
- [ ] **`-Task` on `download-model.ps1` is new and lightly exercised.**

No installer entry, deliberately — same as Whisper.

## Done when

Every number in `docs/dev/models.md`'s embedding section was measured by us, or
is labelled with whose it is.
