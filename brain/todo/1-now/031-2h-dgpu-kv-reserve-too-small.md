# T-031 — The discrete-GPU KV reserve is too small on a 24 GB card

**Effort:** 2 h · **Produces:** a corrected discrete branch in
`_resolve_kv_pool` and a transient term in `_preflight_memory` · **Area:**
`nollama.py`, `docs/dev/prefix-cache.md`

Two models on the B60, and the auto-sizer overshot both:

| model | weights | auto pool | what happened | pool that works |
|---|---|---|---|---|
| Qwen3.8-27B int4 (2026-09-18) | ~15 GB | 5 GB | `CL_OUT_OF_RESOURCES` on request 34 | 3 GB |
| Muse-Glimmer-30B int4 (2026-09-29) | 16.95 GiB | 3 GB | a 35k-token prefill ran at the budget ceiling, 2.2 GiB spilled to host memory, prefill 12x slower and decode 2x slower, **no error** | 2 GB |

The Glimmer run was measured with dedicated and shared memory sampled every
2s. The table, the method and the provenance are in `docs/dev/prefix-cache.md`
("The 2 GB discrete reserve is too small..."). The Qwen3.8 case is in
`docs/dev/machines.md`, B60 section.

**What the measurement says.** A cold prefill adds ~2.4 GiB on top of
Glimmer's idle footprint (19.7 → 22.1 GiB at a 2 GB pool). The discrete
reserve is 2 GiB, and `round()` then spends up to half a GiB more of it:
Glimmer's 2.68 rounded to 3. The preflight uses the same weights x 1.1 +
pool estimate, so it called the spilling configuration "fits".

**A leak is not ruled out for Qwen3.8**, which died on request 34 and not
request 1. Glimmer was flat across 35 requests at 2 GB, and its failure at 3 GB
was on request 1, which is headroom. The two may still have different
causes.

- [ ] discrete branch: `math.floor` instead of `round`, and/or
      `AUTO_KV_RESERVE_GB` 2 → 3. Either gives Glimmer 2 and Qwen3.8 3
- [ ] **re-run Qwen3-Coder-30B on the B60 before shipping**: it runs at 5 GB
      today (`prefix-cache.md`, resolved-values table) and would shrink.
      Use the same load and memory sampling as the Glimmer run, so the
      rule change is not paid for with a regression
- [ ] `_preflight_memory`: add a transient term, so a pairing that will spill
      is at least "tight"
- [ ] re-run Qwen3.8 at 5 GB with memory sampling: flat then crash means
      headroom, a climb means a leak and an upstream ticket
- [ ] update the resolved-values table in `prefix-cache.md`

Load harness: `scripts/agent-load-probe.py` (`--mode openai` for NoLlama,
`--mode ollama` for the cross-stack arm). Sample memory next to it with
`typeperf "\GPU Adapter Memory(<luid>)\Dedicated Usage" "...\Shared Usage" -si 2`.

## Done when

A fresh install of each of these three models on a 24 GB card auto-sizes a
pool that neither spills nor crashes under a 35-request agent-shaped load.
