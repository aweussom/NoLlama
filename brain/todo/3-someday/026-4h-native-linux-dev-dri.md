# T-026 — Native Linux `/dev/dri`, which is what #31 actually asks for

**Effort:** 4 h · **Produces:** a GPU-in-container result on native Linux ·
**Area:** Docker, Linux

The container path works and was measured 2026-08-24: an Intel GPU is usable
from a container at native throughput (74-79 vs 76-78 tok/s; prefix cache
1.9s -> 0.3s vs 2.1s -> 0.2s native). All of that was **WSL**.

Neither limitation found under WSL predicts the native answer:

- the `/dev/dxg` **1 GiB allocation cap** (the same B60 reports 25,055,051,776
  bytes natively and exactly 1,073,741,824 through a container) is a WSL
  artifact by construction
- **NPU in a container is closed and the answer is no** — no `/dev/accel*` in
  either WSL channel, and `wslc run` exposes `--gpus` and no device flag, so
  there is no way to even ask. See `TODONT.md`.

A live-USB run book is ready at `docs/dev/linux-native-gpu-test.md`. A USB is
prepared for the B60 box, and the 285K can be made dual-boot if that turns out
to be the better host.

- [ ] boot the USB, run the book
- [ ] record the packaged `intel-opencl-icd` version and whether `clinfo` saw
      the card — that pair is the finding even if nothing else works

Also open from the container work, smaller:

- [ ] why models needing `GPU_ENABLE_LARGE_ALLOCATIONS` load ~3x slower (E2B
      11.5 s native vs 33-42.5 s in-container; SmolLM3, which needs no hint, is
      at parity)
- [ ] `_maybe_capture_prewarm` swallows `OSError`, so a read-only rootfs is a
      silent cold start forever. Wants a log line.

## Done when

#31 can be answered with a native-Linux number instead of a WSL one.
