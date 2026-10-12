# T-046 — Idle reload fails and the slot sits in `loading` forever

**Effort:** 2 h · **Produces:** a fix in `DeviceSlot.ensure_loaded` and the
reload path's KV sizing · **Area:** idle unload, prefix cache, memory

## What happened (2026-10-12, laptop, 258V / 140V iGPU)

Server started 09:29 with Qwen3-8B-int4-cw on the GPU, auto-sized 7 GB KV
pool, whisper-small on the NPU. GPU 32.0.101.8991, NPU 32.0.100.5540,
OpenVINO 2026.4.1. The GPU slot idle-unloaded after 1800 s. The next request
(an example script) triggered a reload, which printed:

```
[GPU] prefix caching unavailable (... [CL ext] Can not allocate 622329856 bytes
for USM Host. ptr: 0000000000000000, error: -6); using plain pipeline
[GPU] still loading... 15s / 30s / 60s ...
```

and never finished. `/health` stayed `"status":"loading"` for 5+ minutes;
every request got 503 (the first one 500). The python process held only
1.4 GB private, so the plain-pipeline fallback was not loading weights, it
was hung. At that point: 8.6 GB of 31.5 GB RAM free, `vmmemWSL` at 7.2 GB.
The startup load of the same model with the same pool took 11 s.

## Three separate defects

1. **`ensure_loaded()` has no `except`.** `load()` sets `status =
   "loading"` first; if it raises on a reload, nothing sets `error`.
   Startup has a failure path, reload does not. Small, fix first.
2. **The reload reuses the startup KV sizing** `[INFERRED]` — the pool was
   sized against the memory free at 09:29, and the box had less 30 min
   later (WSL had grown, or started). The preflight (`_preflight_memory`,
   `docs/dev/prefix-cache.md`) should run again on reload, or the pool
   should be re-sized. Confirm by reading the reload path first.
3. **The plain-pipeline fallback hung after the failed USM allocation.**
   Unknown whether the driver context is left poisoned by the failed
   allocation (cf. `test_poisoned_gpu.py`). Reproduce bare before blaming
   ourselves or the driver: load a pipeline, fail a big allocation, load
   again. Re-test on the new driver (installed 2026-10-12 after this).

## Done when

- [ ] A reload that raises leaves `status: error` with the reason in
      `/health`, and the next request retries the load instead of 503
      forever. Test with a forced exception.
- [ ] Reload re-checks memory; with too little free it shrinks the pool or
      says why it gave up.
- [ ] Defect 3 reproduced or not, bare, on the new driver; result here.
