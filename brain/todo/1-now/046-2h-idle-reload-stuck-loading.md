# T-046 — Idle reload fails and the slot sits in `loading` forever

**Effort:** 1 h left · **Produces:** a fix in `DeviceSlot.ensure_loaded` and the
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

## Defect 1: fixed 2026-10-27

A reload that raises now goes back to `idle_unloaded` with `error_reason`
in `/health`; the request gets 500 and the next one retries. A poisoned
context goes to `error` and is never retried. `load()` calls
`_LoadProgress.abort()` when `_build_pipeline` raises. Tests:
`tests/test_reload_failure.py`; end to end on the laptop iGPU with
Qwen3-1.7B, model dir renamed away after idle unload: 500, 500 (retried, not
503), dir back, 200, reason cleared [OBSERVED 2026-10-27].

## Defect 3 is probably defect 1

The "hang" was the ticker. It keys on status `loading`, nothing called
`abort()`, and a raising reload left the status there, so "still
loading..." printed forever over a load that had already failed. The first
request's **500** is what `ensure_loaded()` raising produces; a hung plain
pipeline would have hung that request instead, and 1.4 GB private fits a
failed load [INFERRED: the server log of 2026-10-12 was not kept]. So the
plain fallback most likely raised, and nothing hung. Re-open only if a
reload is seen to block with the fix in.

## Defect 2: open, and narrower than first written

The `[INFERRED]` "reload reuses the startup KV sizing" is wrong in the
letter: `load()` reruns `_resolve_kv_pool` and `_preflight_memory` on every
reload. Both use the *total* budget, deliberately (stable sizing,
`docs/dev/prefix-cache.md`). What nothing checks is *free* memory, and on an
iGPU that is the machine's RAM: 7 GB pool + ~5 GB weights against 8.6 GB free
on 2026-10-12. With defect 1 fixed the reload now says why it gave up, which
meets the letter of the box below. The remaining question is whether to
retry the CB pipeline at a smaller pool (`AUTO_KV_MIN_GB`) before falling
back to plain, so a reload into a fuller machine keeps prefix caching.

## Done when

- [x] A reload that raises leaves the reason in `/health`, and the next
      request retries the load instead of 503 forever.
- [ ] Defect 2: decide the smaller-pool retry (Tommy), or close as "says why".
- [x] Defect 3: explained by defect 1 [INFERRED]; no bare repro needed unless
      it recurs.
