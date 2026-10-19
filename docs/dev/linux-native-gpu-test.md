# Handover: testing the Intel GPU on native Linux (live USB)

**You are here because `DOCKER-INSTALL.md` has exactly one unanswered
question, and it needs a Linux boot to answer.** Everything else about the
container path is measured. This document is self-contained: it assumes you
know nothing about the session that produced it.

Run this from a **live USB on the B60 box** (the Ryzen 9 5950X + Arc Pro B60
workstation — `docs/dev/machines.md`). No installation, no repartitioning.

---

## The question, and why it is worth a boot

On Windows, an Arc Pro B60 reports its whole 23.3 GB as allocatable. Reached
through a container over WSL's `/dev/dxg`, the *same card* caps a single
allocation at exactly 1 GiB:

| | `GPU_DEVICE_TOTAL_MEM_SIZE` | `GPU_DEVICE_MAX_ALLOC_MEM_SIZE` |
|---|---|---|
| native Windows | 25,055,051,776 | 25,055,051,776 |
| container over WSL `/dev/dxg` | 25,055,051,776 | **1,073,741,824** |

That cap breaks any model with a tensor over 1 GiB — Gemma 4 E2B has a 2.2 GB
per-layer embedding table — until NoLlama passes
`GPU_ENABLE_LARGE_ALLOCATIONS`, which costs ~3x on load time.

**The question is whether the cap clears the model, not whether it equals
total.** A cap below total is normal — OpenCL guarantees only a quarter of
global memory, a Xe-LPG iGPU reports 0.12 natively, and the B60's 1.00 is the
outlier. What matters is whether it clears E2B's biggest tensor,
2,348,810,240 bytes. If it does on native Linux `/dev/dri`, the workaround is
WSL-only for that card. That is the single most valuable number to bring
back.

(Until 2026-10-19 this section and Step 2's verdict line asked "does it equal
total", the framing #31 retracted on 2026-08-24. The first outside run
printed `CAPPED — not a WSL artefact` on an iGPU where that was never the
question.)

Second question, nearly as valuable: `gemma-4-26b-a4b-it-int4-ov` computes
**deterministic garbage** on the container GPU path while being correct
natively on Windows and correct on CPU inside the same container. If it is
correct on native Linux, the defect is pinned to WSL `/dev/dxg` and the
upstream report writes itself.

## What is already known — do not re-derive these

| Fact | Confidence |
|---|---|
| Container GPU works at native speed (74-79 vs 76-78 tok/s) | measured 2026-08-24, WSL |
| The 1 GiB cap is not a driver-version thing — NEO 25.44 and 26.31 both | measured |
| The 1 GiB cap is the `/dev/dxg` path itself, not the B60: an Arrow Lake-HX iGPU caps at 4,294,959,104 natively and exactly 1,073,741,824 through WSL | measured 2026-10-19 by JRCuster, #31 |
| `/dev/dri` passthrough enumerates on native Linux — a Gen 9.5 UHD 600 on a Synology NAS, cap 0.50 of total (1.6 GiB), in `openvino/ubuntu22_runtime:2024.x` | measured 2026-10-19 by JRCuster, #31. Old hardware, old images: says nothing about Battlemage |
| compute-runtime 24.35+ dropped Gen 9, so **our image cannot see a Gen 9 iGPU** — the reverse of the too-old-driver trap | reported in #31, not checked here |
| The NPU is absent from WSL entirely, both channels; `wslc` has no device flag | measured, closed in TODONT |
| `openvino/ubuntu24_runtime:latest` cannot see a B60 — NEO 24.48 predates it | measured, in TODONT |
| Ubuntu 24.04's packaged Intel driver is also too old for Battlemage | same root cause |
| 26B garbage reproduces through raw `openvino_genai`, so NoLlama is not the cause | measured |
| CPU inside the container is correct for every model | measured |

## Before you boot

1. **Full shutdown, not a restart, and disable Fast Startup.** Windows Fast
   Startup leaves NTFS in a hibernated state and Linux will refuse to mount
   it, or mount it unreliably. `powercfg /h off` is the blunt fix.
2. **Check whether `C:` is BitLocker-encrypted** (`manage-bde -status C:`).
   If it is, Linux cannot read the models directory without the key, and you
   should plan to copy a model to the USB or an unencrypted drive first.
3. Models live at `C:\Users\wossn\models`. The three that matter here:
   - `SmolLM3-3B-int4-cw-ov` (1.6 GB) — baseline, no tensor over 1 GiB
   - `gemma-4-E2B-it-int4-ov` (4.1 GB) — **the large-allocation test**, 2.2 GB tensor
   - `gemma-4-26b-a4b-it-int4-ov` (15 GB) — the defect test
4. A live session keeps its filesystem in RAM and usually has no swap. This
   box has 32 GB, which is fine for the first two models. The 26B stages
   ~15 GB through RAM on load — do it last, and close everything else.

## Step 0 — where am I

```bash
. /etc/os-release && echo "$PRETTY_NAME"; uname -r
lspci -nn | grep -i -E "vga|display"          # expect 8086:e211 = Arc Pro B60
ls -la /dev/dri/                              # expect renderD128 and card0/1
free -g
```

If `/dev/dri/renderD128` does not exist, stop: the kernel does not have the
card. Note the kernel version and that is the finding.

## Step 1 — driver userspace, and the trap

Battlemage needs a recent Intel compute-runtime. **Do not assume the distro's
package is new enough** — that exact assumption is what makes the official
OpenVINO image report CPU-only on this hardware.

```bash
sudo apt update && sudo apt install -y clinfo
apt-cache policy intel-opencl-icd     # note the version
clinfo -l                             # does the B60 appear at all?
```

If `clinfo -l` lists the card, the packaged driver is new enough — record the
version, that is itself a useful finding. If it does not, install the upstream
release exactly as the repo's `Dockerfile` does:

```bash
mkdir -p /tmp/neo && cd /tmp/neo
IGC=2.40.13; IGCB=22418; NEO=26.31.39395.13; GMM=22.10.0
B=https://github.com/intel/intel-graphics-compiler/releases/download/v$IGC
curl -fsSLO $B/intel-igc-core-2_${IGC}%2B${IGCB}_amd64.deb
curl -fsSLO $B/intel-igc-opencl-2_${IGC}%2B${IGCB}_amd64.deb
C=https://github.com/intel/compute-runtime/releases/download/$NEO
for f in intel-opencl-icd_${NEO}-0 libze-intel-gpu1_${NEO}-0 intel-ocloc_${NEO}-0; do
  curl -fsSLO $C/$f\_amd64.deb; done
curl -fsSLO $C/libigdgmm12_${GMM}_amd64.deb
sudo dpkg -i *.deb
```

Add yourself to the `render` group if `clinfo` needs it:
`sudo usermod -aG render $USER` then `newgrp render`.

## Step 2 — THE TWO NUMBERS (this is the point of the whole exercise)

No Docker needed. This is the cheapest path to the answer, so do it first.

```bash
python3 -m venv ~/ov && ~/ov/bin/pip install -q openvino
~/ov/bin/python -c "
import openvino as ov
c = ov.Core()
print('devices:', [(d, c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices])
t = c.get_property('GPU','GPU_DEVICE_TOTAL_MEM_SIZE')
m = c.get_property('GPU','GPU_DEVICE_MAX_ALLOC_MEM_SIZE')
print('total    :', t)
print('max_alloc:', m)
print('ratio    : %.3f' % (m / t))
print('VERDICT  :', f'cap {m/2**30:.2f} GiB',
      'CLEARS E2B (2,348,810,240 B) — no large-allocation hint needed' if m > 2348810240
      else 'does NOT clear E2B — the hint is needed on this device')
"
```

**Bring that verdict back even if you do nothing else.**

## Step 3 — enumeration is not correctness

A device that lists fine and computes garbage is not hypothetical here; it is
the open defect. Always check compute.

```bash
~/ov/bin/pip install -q numpy
~/ov/bin/python -c "
import numpy as np, openvino as ov, math
c = ov.Core(); rng = np.random.default_rng(0)
A = rng.standard_normal((256,512)).astype(np.float32)
B = rng.standard_normal((512,256)).astype(np.float32)
p = ov.opset13.parameter([256,512], ov.Type.f32)
mdl = ov.Model([ov.opset13.gelu(ov.opset13.matmul(p, ov.opset13.constant(B), False, False), 'erf')], [p])
mm = A@B
ref = mm*0.5*(1+np.array([math.erf(v) for v in (mm/np.sqrt(2)).ravel()]).reshape(mm.shape))
for dev,cfg in (('CPU',{}),('GPU',{}),('GPU',{'INFERENCE_PRECISION_HINT':'f32'})):
    o = c.compile_model(mdl,dev,cfg)(A)[0]
    print(dev, cfg, 'max|err| = %.3e' % np.abs(o-ref).max())
"
```

Expected, matching Windows: CPU ~9.2e-05, GPU default (fp16) ~2.9e-02, GPU
f32 ~9.2e-05. The fp16 gap is precision, not corruption. Anything wildly
larger means the GPU is computing wrong and everything below is moot.

## Step 4 — the container path, `/dev/dri`

This is the configuration `docker-compose.yml` ships and nobody has run.

```bash
sudo apt install -y docker.io && sudo systemctl start docker
git clone https://github.com/aweussom/NoLlama && cd NoLlama
sudo docker build -t nollama:local .

sudo docker run --rm --device /dev/dri --entrypoint python3 nollama:local -c \
 "import openvino as ov; c=ov.Core(); print([(d,c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices]); print(c.get_property('GPU','GPU_DEVICE_TOTAL_MEM_SIZE'), c.get_property('GPU','GPU_DEVICE_MAX_ALLOC_MEM_SIZE'))"
```

If the GPU appears in-container, run the real thing. Mount the model under
its **real directory name** — that name is the API model id:

```bash
sudo mkdir -p /mnt/win && sudo mount -o ro /dev/nvme0n1p3 /mnt/win   # find the right partition
M=/mnt/win/Users/wossn/models
sudo docker run --rm --device /dev/dri -p 8000:8000 -p 11434:11434 \
  -v $M/SmolLM3-3B-int4-cw-ov:/models/SmolLM3-3B-int4-cw-ov:ro \
  nollama:local --model-dir /models/SmolLM3-3B-int4-cw-ov --device GPU
```

Watch for the line `large allocations enabled`. It fires when a model's
biggest single tensor exceeds the device's per-allocation cap — so for
SmolLM3 (biggest tensor 0.24 GB) it should **not** appear on any device.
Seeing it there would mean the cap is under 0.24 GB, which would be
remarkable and worth reporting on its own.

Then, from another terminal:

```bash
curl -s localhost:8000/v1/models          # must say SmolLM3-3B-int4-cw, not "model"
curl -s localhost:11434/api/tags          # both ports, or Ollama clients break
curl -s -X POST localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"SmolLM3-3B-int4-cw","max_tokens":30,"temperature":0,
       "messages":[{"role":"user","content":"Capital of Norway? One word."}]}'
```

## Step 5 — the E2B large-allocation case

E2B is the test case because its per-layer embedding table is a **single
2.19 GB tensor** — 2,348,810,240 bytes, the exact request that fails under
WSL's 1 GiB cap. Any device whose cap exceeds that loads it without a hint.

Same command as above with `gemma-4-E2B-it-int4-ov`. Two things to record:

- whether `large allocations enabled` appears. If Step 2 reported a cap
  above 2.19 GB it must not, and that is the confirmation.
- load time. On Windows: 11.5s native, 33-42.5s in-container with the hint
  active. Whether that 3x is the hint's cost or WSL's is still unknown — a
  native Linux number for a model that needs **no** hint helps separate them.

Note the cap does not have to equal total memory to be fine. OpenCL only
guarantees a quarter of global memory, and a Xe-LPG iGPU measured 0.12 of
total — 4.29 GB, still comfortably above E2B's biggest tensor. "Capped" and
"broken" are different findings; report the number, not a verdict.

## Step 6 — the defect, if RAM allows

```bash
sudo docker run --rm --device /dev/dri -p 8000:8000 \
  -v $M/gemma-4-26b-a4b-it-int4-ov:/models/gemma-4-26b-a4b-it-int4-ov:ro \
  nollama:local --model-dir /models/gemma-4-26b-a4b-it-int4-ov --device GPU --ollama-port 0
```

Ask it: *"Name the capital of Norway, then add 17 + 25. Answer in one short
sentence."*

- **Correct** ("Oslo … 42") → the defect is WSL `/dev/dxg`-specific. That is
  a clean upstream report and a genuinely good result.
- **Garbage** (`- @__-'ls--る1-_--...`) → it is the Linux Intel GPU stack, not
  WSL, and the report goes to Intel with much wider impact.

Either answer is publishable. There is no wasted outcome here.

## Bring back

Paste raw output, not summaries — the exact byte counts are the evidence.

```
distro / kernel:
packaged intel-opencl-icd version, and whether clinfo saw the card:
driver actually used (packaged / upstream <version>):

TOTAL    :
MAX_ALLOC:
VERDICT  :

matmul errors  CPU / GPU fp16 / GPU f32:
container enumeration (device list + both numbers):
"large allocations enabled" line present?   yes / no
SmolLM3 in-container: model id, both ports, answer text:
E2B load time:
26B answer text (verbatim):
```

## Gotchas already paid for — do not rediscover them

- **A too-old driver does not error, it hides the GPU.** An unrecognised
  device id produces a device list with no GPU in it, indistinguishable from
  a broken passthrough. `zeInit` returns `ZE_RESULT_ERROR_UNINITIALIZED` and
  `clGetPlatformIDs` returns `-1001`. Check the driver version first.
- **Mount the models read-only.** Nothing in this test writes to Windows, and
  a live session writing to a hibernated NTFS volume is how you lose a
  Windows install.
- **Model directory name is the API model id.** Bind-mounting to a generic
  path makes clients ask for `model`.
- **`LD_LIBRARY_PATH` is a WSL-only concern** (`/usr/lib/wsl/lib`). On native
  Linux you should not need it at all; if you find yourself setting it, note
  that, because it means something else is wrong.
- **Publish both ports.** 8000 is OpenAI, 11434 is Ollama.

## Where this goes when you are done

- Numbers into `DOCKER-INSTALL.md`, under **What is still not answered** —
  and move the item out of that list.
- `docs/DOCKER.md` currently says the `/dev/dri` path is **untested**. If it
  works, that sentence changes and the WSL caveats get scoped to WSL.
- If the 1 GiB cap turns out WSL-only, say so in the `_gpu_large_alloc_props`
  docstring in `nollama.py` — it currently carries an `[OBSERVED 2026-08-24]`
  tag naming only the WSL measurement.
- A comment on [#31](https://github.com/aweussom/NoLlama/issues/31), where
  the requester has been asked for this same datapoint. If we produced it
  first, tell them so they do not duplicate the work.
- The full protocol and every measurement behind this document are in
  `DOCKER-INSTALL.md` (the `docker-support` branch merged to `main` on
  2026-08-30 and is gone).
