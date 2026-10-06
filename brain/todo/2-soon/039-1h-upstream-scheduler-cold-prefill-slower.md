# T-039 — Ask upstream why the scheduler path prefills cold 1.5–3× slower

**Effort:** 1 h · **Produces:** an openvino.genai issue · **Area:** prefix cache

Two GPU classes now show it, so the condition `docs/dev/prefix-cache.md` set
for asking upstream is met:

- 285K Xe-LPG, no XMX, Qwen3-8B int4-cw, genai 2026.3.0: 72 s plain vs 216 s
  scheduler, same in bare genai. One CPU core at 100 %, GPU near idle.
- #33 reporter's 140T, Xe-LPG+ with XMX, Qwen3-Coder-30B-A3B int8, nightly
  2026.5.0: 185–194 s plain vs 287–469 s scheduler (`benchmark.py --long`).

Before filing: a bare-genai repro (`ContinuousBatchingPipeline` vs
`LLMPipeline`, same prompt) on a box we own, plus a run on Xe2 (140V or B60)
to say whether Xe2 is affected. If Xe2 is clean, say so; that's half the
answer. The laptop driver is one release behind (`docs/dev/machines.md`), so
update it before measuring.

## Done when

The issue is filed with the bare repro, and the link is in `prefix-cache.md`.
