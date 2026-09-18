# T-012 — Find the residency control `_gpu_keepalive` is standing in for

**Effort:** 4 h · **Produces:** either a working residency call or a filed ask
to Intel · **Area:** `nollama.py`, GPU plugin

`_gpu_keepalive` works and should stay until something better is proven, but it
is a workaround wearing the shape of a fix: we defeat an idle timer by
manufacturing non-idleness, once a minute, forever. It burns power, it takes
the slot lock, it costs a 1-token prefill, and it needed its own
`CL_OUT_OF_RESOURCES` handling because a ping that poisons the context,
swallowed, feeds a dead slot to the idle watchdog.

## What is already known

- The eviction is **not memory pressure**: 20.34 GB -> 0 at 80 s, 80 s and 79 s
  after the last request on an Arc Pro B60, with 21.5 GB of host RAM free
  [OBSERVED 2026-09-13, `docs/slot-lifecycle.mmd`]. Three for three at a
  consistent threshold. That is an **idle timer**, and an idle timer has a
  setting somewhere.
- **iGPUs are immune** — their VRAM is host RAM, nothing to evict to. Whatever
  is found only applies to the discrete path, same as today's gate.
- NoLlama is not involved: status stays `ready`, the allocation is never
  released from our side, committed memory does not change.

## Ruled out: the OpenVINO GPU plugin has no such property

[OBSERVED 2026-09-13, `SUPPORTED_PROPERTIES` on the 285K, OpenVINO 2026.3.0.]
The complete RW set is `PERF_COUNT`, `MODEL_PRIORITY`, `GPU_HOST_TASK_PRIORITY`,
`GPU_QUEUE_PRIORITY`, `GPU_QUEUE_THROTTLE`, `GPU_ENABLE_SDPA_OPTIMIZATION`,
`GPU_ENABLE_LORA_OPERATION`, `GPU_ENABLE_LARGE_ALLOCATIONS`,
`GPU_ENABLE_LOOP_UNROLLING`, `GPU_DISABLE_WINOGRAD_CONVOLUTION`, `CACHE_DIR`,
`CACHE_MODE`, `PERFORMANCE_HINT`, `EXECUTION_MODE_HINT`,
`COMPILATION_NUM_THREADS`, `NUM_STREAMS`, `PERFORMANCE_HINT_NUM_REQUESTS`,
`INFERENCE_PRECISION_HINT`, `ENABLE_CPU_PINNING`, `ENABLE_CPU_RESERVATION`,
`DEVICE_ID`, `DYNAMIC_QUANTIZATION_GROUP_SIZE`, `ACTIVATIONS_SCALE_FACTOR`,
`WEIGHTS_PATH`, `KV_CACHE_PRECISION`, `OFFLOAD_RATIO`, `CONFIG_FILE`.

Nothing resembling residency, eviction, reservation or keep-alive. So the lever
is **below** OpenVINO — which makes "ask Intel for one at the plugin layer" a
legitimate second outcome of this hunt. Caveat: that enumeration was taken on a
Xe-LPG iGPU + an RTX 5090, not on the B60. The list is plugin-level so it
should hold; re-run it on the B60 before treating the negative as final.

## Leads, in the order worth trying

- [ ] **`MODEL_PRIORITY` (`ov::hint::model_priority`).** One property, ten
      minutes, already exposed. `HIGH` may map onto something the driver
      consults when deciding what to page out. Weak prior — the trigger is an
      idle timer, not contention — but it is the cheapest test we have.
- [ ] **DXGI residency, the actual WDDM-level lever.**
      `IDXGIAdapter3::SetVideoMemoryReservation` tells Windows how much video
      memory this process wants kept resident, and `QueryVideoMemoryInfo`
      reports `CurrentReservation` / `CurrentUsage` / `Budget`, so the effect
      is **directly measurable** rather than inferred from TTFT. Reachable from
      Python by `ctypes` against `dxgi.dll` with no OpenVINO involvement, so it
      is testable as a standalone probe beside a running server. **Best
      candidate.**
- [ ] **Level Zero residency.** `zeContextMakeMemoryResident()` /
      `zeContextEvictMemory()` are the documented L0 API for exactly this. The
      obstacle: OpenVINO's GPU plugin on Windows is on **OpenCL**, not L0 — the
      #38 traceback names `ocl_common.hpp` — so we would be making a residency
      call against allocations we do not own. Understand it before dismissing;
      may be the thing to ask Intel to wire up.
- [ ] **Intel USM extensions.** `GPU_USM_MEMORY` is in the iGPU's
      `OPTIMIZATION_CAPABILITIES`, so `cl_intel_unified_shared_memory` is
      present. Check whether `clEnqueueMigrateMemINTEL` /
      `clEnqueueMemAdviseINTEL` carry an advice value meaning "keep resident".
      Same ownership problem as Level Zero.
- [ ] **Driver-side and Windows-side settings.** An 80-second idle threshold
      that fires with memory to spare smells like driver power management.
      Check Intel Graphics Software's power settings on the B60 and the Windows
      power mode. **Do not change a driver setting on the B60 without asking** —
      `docs/dev/machines.md` rules apply.

## Done when

Either a residency call that survives a 3-hour idle without a ping, or a
written negative naming what was tried, so the keepalive stops looking like
something nobody investigated.
