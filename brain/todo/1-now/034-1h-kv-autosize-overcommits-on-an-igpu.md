# T-034 — The KV auto-sizer over-commits on a shared-memory iGPU

**Effort:** 1 h · **Produces:** a sizing rule that accounts for shared memory ·
**Area:** `nollama.py` (`AUTO_KV_MIN_GB` and the auto-size path)

[OBSERVED 2026-09-23, 258V laptop, 140V, OpenVINO 2026.4] Loading
`Qwen2.5-Coder-14B-int4` (7.9 GB) the auto-sizer chose a **12 GB** KV pool. On a
32 GB box with the user's own applications running, free RAM went to **zero** and
the machine was in thrash territory until the server was killed. `--cache-size-gb
4` left 7 GB free and served fine.

The sizer reads the GPU's reported budget. On a discrete card that budget is
VRAM and spending it is free. On an integrated GPU **the budget is system RAM**,
so every GB handed to the KV pool is a GB taken from the OS, the user's editor,
and the staging copy the loader itself needs.

- [ ] cap the auto-sized pool by *free system RAM* on an integrated GPU, not by
      the GPU budget
- [ ] leave a floor of headroom (the box must stay usable — this is a laptop
      someone is working on)
- [ ] keep `--cache-size-gb` as the override and keep the existing warning when
      the resulting pool is too small for agent prompts

Related but not the same: `T-010` is the dGPU memory-preflight false alarm — that
one refuses a load that would fit. This one accepts a load that should not.

## Done when

An 8 GB model on the 140V auto-sizes to a pool that leaves the box usable, and
the number it picks is explainable from free RAM rather than from the adapter's
advertised budget.
