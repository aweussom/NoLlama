# T-031 — Qwen3.8-27B on the B60 dies at the auto-sized 5 GB pool: headroom or leak?

**Effort:** 2 h · **Produces:** a decided cause and, if headroom, a dGPU-aware
reserve in `_resolve_kv_pool` · **Area:** `nollama.py`, `docs/dev/prefix-cache.md`

[OBSERVED 2026-09-18, B60, driver 32.0.101.8805, OpenVINO 2026.3.1,
`OpenVINO/Qwen3.8-27B-int4-ov` rev 2026.3.1, VLM slot] 15 GB of weights, the
auto-sizer chose a 5 GB pool, and the slot died with `CL_OUT_OF_RESOURCES` on
the **34th request** of a benchmark run — a 323-char prompt, `max_tokens`
512 — after a ~4.5k-token prefill and three 500-token generations had already
succeeded. `--cache-size-gb 3` ran the same sequence twice more without
incident. `docs/dev/machines.md`, B60 section, has the full note.

Two explanations fit and they call for different fixes:

- **Headroom.** A dGPU has no shared-memory spill; 15 GB + 5 GB + the vision
  tower + activations may simply exceed 24 GB on some request shape. Fix:
  `_resolve_kv_pool` reserves more on a dGPU with a VLM loaded.
- **Leak.** Something per request grows until it does not fit. Fix: upstream.

- [ ] rerun at 5 GB while polling the card's dedicated memory (Task Manager's
      graph is the only tell on Windows for Arc; `nvidia-smi` has no
      equivalent here) — flat until the crash says headroom, a climb says leak
- [ ] whichever it is, record it in `machines.md` and `prefix-cache.md`
- [ ] if headroom: the dGPU reserve, with the number that fixed it

## Done when

The B60 note says *why* 5 GB failed, and a fresh install of this model on a
24 GB card does not hit it.
