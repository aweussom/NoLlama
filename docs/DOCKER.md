# Running NoLlama in a container

**Status: experimental, GPU and CPU only, no NPU.** The measurements behind
every claim here are in `DOCKER-INSTALL.md` (issue
[#31](https://github.com/aweussom/NoLlama/issues/31)). Read the two known
defects at the bottom before you deploy anything you care about.

## No NPU, and that is not going to change soon

The NPU is not exposed to WSL 2 or to Linux containers on Windows. Confirmed
on hardware that has a working one rather than inferred from forum threads:
no `/dev/accel*` device node exists inside WSL at all, only `/dev/dxg` for
the GPU ([WSL #40842](https://github.com/microsoft/WSL/issues/40842)).

If the NPU is why you are here, run NoLlama natively. Containers give you
the GPU and the CPU.

## Quick start

Models are **bind-mounted read-only** from the host and never baked into the
image or downloaded inside it. The image is model-free and about 1.3 GB.

Mount each model under its **real directory name** — that name is the model
id NoLlama serves. Mounting to a generic path makes clients ask for `model`.

### Native Linux

```bash
export MODELS=/srv/models MODEL=Qwen3-8B-int4-ov
export RENDER_GID=$(stat -c %g /dev/dri/renderD128)
docker compose up
```

**Untested by this project on current hardware.** One outside run (#31)
confirmed that `/dev/dri` passthrough enumerates an Intel iGPU on native
Linux, but on an old Gen 9.5 part in an older OpenVINO image. **This image
cannot see Gen 9 iGPUs at all**: the compute-runtime it installs dropped
them from version 24.35 onwards. Anything from Xe on is the target. If you run this, the result is
worth reporting on #31, whichever way it goes. The first thing to check is
that `GPU` appears at all:

```bash
docker compose run --rm --entrypoint python3 nollama -c \
  "import openvino as ov; c=ov.Core(); print([(d, c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices])"
```

Then check that it is *correct*, which is a different question — see the
defects below.

### WSL 2 on Windows

This is the configuration that was actually measured.

```bash
export MODELS='C:\Users\you\models' MODEL=Qwen3-8B-int4-ov
docker compose -f docker-compose.yml -f docker-compose.wsl.yml up
```

WSL exposes the GPU as `/dev/dxg` and puts the D3D12 shims in
`/usr/lib/wsl`; the override file handles both.

Before loading anything large, give the WSL VM room in
`%USERPROFILE%\.wslconfig` — model load stages through host RAM at roughly
model size, and WSL defaults to about half your RAM:

```ini
[wsl2]
memory=24GB
swap=16GB
autoMemoryReclaim=gradual
```

## Both ports, or Ollama clients break

NoLlama serves the OpenAI API on 8000 and the Ollama API on 11434. The
compose file publishes both. A deployment that publishes only 8000 works
perfectly right up until someone points an Ollama client at it.

## Prewarm needs somewhere to write

`--idle-timeout 0` (the agent setup) auto-enables prewarm, which writes
`prewarm-<port>.json`. The default location is the working directory, which
lives in the container's writable layer — `docker run --rm` throws it away,
and a read-only rootfs silently never persists it, with nothing in the log
to say why.

Point it at the state volume instead:

```yaml
command: [--model-dir, /models/${MODEL}, --device, GPU,
          --idle-timeout, "0", --prewarm, /state/prewarm.json]
```

## What it costs

Nothing measurable, for models that fit under the allocation cap below.
Container vs native on the same box and model: 74-79 vs 76-78 tok/s, prefix
cache 1.9s → 0.3s vs 2.1s → 0.2s, load 6.1s vs 4.4-6.6s.

Models that need the large-allocation workaround load about 3x slower
(11.5s native vs 33-42.5s in-container for Gemma 4 E2B). Throughput once
loaded is unaffected.

Bind-mounting models from `/mnt/c` reads at ~178 MB/s versus multiple GB/s
from a native volume — a 35x gap that makes **no** difference to load time,
because OpenVINO maps the weights rather than streaming them. Copying models
onto ext4 to speed up loading does not work. Don't bother.

## The two things that will bite you

### A GPU reached through WSL caps single allocations at 1 GiB

`/dev/dxg` reports `GPU_DEVICE_MAX_ALLOC_MEM_SIZE` as exactly 1 GiB while
reporting the card's full memory as total — on an Arc Pro B60, 1,073,741,824
versus 25,055,051,776 natively. Any model with a tensor over that size dies
at load with *"Exceeded max size of memory object allocation"*. Gemma 4
E2B's per-layer embedding table is 2.2 GB, so this is not an exotic case.

NoLlama detects the mismatch and passes `GPU_ENABLE_LARGE_ALLOCATIONS`,
logging a line when it does. You should not have to do anything. Native
installs report the whole budget as allocatable and take no hint.

### One model computes wrong, silently

`gemma-4-26b-a4b-it-int4-ov` loads cleanly on the container GPU path and
then generates deterministic gibberish. The same model is correct natively,
correct on CPU inside the same container, and every other model tried on
that path is correct — including a 16 GB MoE. Two driver generations behave
identically, so it is not a recent regression.

Cause unknown, and that is the point worth taking away: **enumeration is not
correctness**. A device list, a clean load and a healthy `/health` all say
this model is fine. Read the output before you trust a container GPU.

## Not available in the container

`--backend optimum`. The image installs serving dependencies only — no
optimum-intel, no transformers — because nothing is ever exported or
downloaded inside a container. The `NEEDS_OPTIMUM` architectures
(`nemotron_h`) therefore cannot be served from this image. Run those
natively (`docs/dev/runtime-stacks.md`).

## Building on a different Intel GPU generation

The Dockerfile pins the Intel compute-runtime, IGC and GMM versions as build
args, and they move as a set — NEO 25.44 refuses IGC ≥ 2.25, for instance:

```bash
docker build --build-arg NEO=25.44.36015.8 --build-arg GMM=22.8.2 \
             --build-arg IGC=2.24.8 --build-arg IGC_BUILD=20344 -t nollama:local .
```

Do **not** substitute your distro's packaged driver. Ubuntu 24.04 ships a
NEO that predates Battlemage, and an unrecognised device does not produce an
error — it produces a device list with no GPU in it, which reads exactly
like a broken passthrough.
