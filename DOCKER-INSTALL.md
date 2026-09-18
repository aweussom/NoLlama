# Testing NoLlama in Docker — protocol

For [issue #31](https://github.com/aweussom/NoLlama/issues/31). Answer it
with measurements. Docker/WSL install is assumed done.

**Out of scope, decided:** the NPU. It is not exposed to WSL2 or Docker
Desktop Linux containers ([WSL #40842](https://github.com/microsoft/WSL/issues/40842),
[Intel Community](https://community.intel.com/t5/Graphics/NPU-in-WSL/td-p/1581143));
Anything we publish says "no NPU" in plain words.

**Correction (2026-08-24): there is no "WSL 3".** Microsoft
[denied it](https://www.windowslatest.com/2026/06/28/microsoft-denies-wsl-3-exists-reveals-windows-11s-wsl-containers-ship-next-week/)
- what Build 2026 announced is **WSL Containers**, built on WSL 2, and much
of the press mislabelled it. It is in public preview via
`wsl --update --pre-release` (WSL 2.9.3+), GA targeted for autumn 2026
([devblog](https://devblogs.microsoft.com/commandline/wsl-container-is-now-available-for-public-preview/)).
Whether it changes the NPU answer is **unverified** - that claim came from
the same mislabelled reporting, so treat it as rumour until someone runs
`available_devices` inside it.

## Ground rules

**No model downloads into a container. Ever.** Pulling 4-20 GB into a
throwaway is wasteful and pointless when the host already has the weights.
This is not a preference, it is a design constraint on everything below:

- Models are **bind-mounted read-only** from the host's models directory.
- The image is **model-free**. Nothing baked in, nothing fetched at build or
  run time. (Also the licence-clean choice: baking Gemma weights into an
  image is a redistribution question under the Gemma Terms. Avoided entirely
  by never doing it.)
- If a test seems to need a model we do not have, the test is wrong.

That makes bind-mounting mandatory rather than incidental, which promotes
hazards **1.1** (mount path vs the naming rule) and **1.4** (integrity check
over drvfs) from "worth checking" to "on the critical path".

### What is already on disk, and what each is for

| Model | Size | Use in this protocol |
|---|---|---|
| `SmolLM3-3B-int4-cw-ov` | 1.6 GB | **Default.** Plumbing, ports, streaming, restarts - smallest thing that proves the path |
| `gemma-4-E2B-it-int4-ov` | 4.1 GB | VLM path + the probe set; prefix-cache cold-vs-repeat |
| `Qwen2.5-VL-3B-Instruct-int8-ov` | 3.9 GB | Second VLM opinion if a result looks odd |
| `gemma-4-E4B-it-int8-ours` | 7.8 GB | Parity against the 1.16 s steady-state TTFT already measured |
| `gemma-4-26b-a4b-it-int4-ov` | 15 GB | **Only** for hazard 1.3 - the WSL2 memory-cap question needs a big model to be a question at all |

Iterate on SmolLM3. Touch the 26B once, deliberately, for 1.3.

Copying **one** small model onto the WSL2 ext4 filesystem is allowed for the
1.4 drvfs-vs-native timing comparison - that is a local copy, not a download.

---

## How bleeding-edge are we willing to be?

Standing rule, unchanged: **test on nightly, ship on release.** That is what
`venv-nightly` and `install.ps1 -Nightly` exist for, and it is the same rule
that pulled Qwen3.8 27B out of `models.json` on 2026-08-21. Testing here on a
preview stack is fine; a *supported* container path is not announced until
every layer below it is on a release.

| Layer | Needed for | Status | Verdict |
|---|---|---|---|
| WSL 2 | GPU via `/dev/dxg` | shipped | fine |
| Intel compute-runtime | GPU inside the distro/container | current **release** on GitHub; Ubuntu's packaged build lags for Battlemage | fine — an upstream release, just not the distro's |
| OpenVINO | the runtime itself | 2026.3 release works natively; nightly image available if the container needs newer | test on nightly, ship on release |
| WSL Containers | nothing we need | public preview, GA autumn 2026 | **skip** - see below |

**We do not want WSL Containers.** It is a container runtime inside WSL 2 for
running Linux containers on Windows without Docker Desktop - and every
benefit it offers is already covered by running plain `docker-ce` inside the
WSL 2 distro, which also dodges the Docker Desktop licence.

The deciding argument is portability. #31 asks for **Linux** deployment: a
`Dockerfile` and `compose.yml` built against ordinary `docker-ce` run on the
requester's hardware. Anything WSL-Containers-specific is Windows-only and
does not serve the request that prompted this work.

The one thing that could change that is its rumoured NPU passthrough - but
that claim comes from the same mislabelled "WSL 3" reporting, and even if it
holds it would be a Windows-only NPU path: nice for local dev, irrelevant to
#31, and NPU is scoped out anyway. Re-check only if someone demonstrates
`available_devices` listing NPU inside it.

So the realistic target for this experiment is **GPU and CPU only**, which is
also exactly what #31 can honestly be offered.

---

Run the phases in order and stop at the first hard failure — each one gates
the next, and an early stop is a valid answer for #31.

---

> **Machines:** see `docs/dev/machines.md`. Short version — the B60 box is
> the clean Intel GPU test, the 285K is the only NPU box available, and the
> **258V laptop is off-limits: do not touch WSL, Docker or drivers on it.**

## Track B — NPU in a container (separate question, worth asking)

Keep this **separate from #31**, which is a Linux deployment request that an
NPU path cannot serve. But NoLlama is NPU-first and Windows-primary, so
"Windows-only" is not the disqualifier it would be for the issue - it is most
of this project's actual audience. Panther Lake makes it more interesting,
not less.

Cost is low-ish: `wsl --update --pre-release` swaps a WSL component and is
**not** an Insider-channel change.

**Correction (2026-08-24): `wsl --update --rollback` does not exist.** This
section previously claimed it as the undo, and that claim was never tested.
On 2.9.8 it returns `Wsl/UpdatePackage/E_INVALIDARG`, and `wsl --help` lists
no sub-options for `--update` at all. Reverting means installing an older MSI
by hand from [microsoft/WSL releases](https://github.com/microsoft/WSL/releases)
— 2.9.8 is flagged pre-release there; the current stable is **2.7.12**. So
the switch is one command and the undo is a manual install. Still cheap, but
not symmetric, and worth knowing before you do it to a machine you care
about.

### B1 result, measured 2026-08-24 — no NPU in WSL 2

Run on the **Core Ultra 9 285K** (WSL 2.6.3.0, kernel 6.6.87.2, Ubuntu),
which has a working `Intel(R) AI Boost` NPU showing status OK in Windows:

```
ls -la /dev/accel* /dev/dxg /dev/dri
  /dev/accel*  -> No such file or directory     <- no NPU
  /dev/dri     -> No such file or directory
  /dev/dxg     -> crw-rw-rw- 10, 125            <- GPU paravirt present
```

So the NPU is genuinely absent from WSL 2, confirmed on hardware that has
one rather than inferred from forum posts. `/usr/lib/wsl/lib` on that box
holds 24 libraries, **all NVIDIA plus the D3D12 shims, zero Intel** - though
that is expected and not itself a blocker: NVIDIA injects host libraries
there, whereas Intel's WSL path is `intel-opencl-icd` /
`intel-level-zero-gpu` installed **inside the distro**, talking to
`/dev/dxg` via `libd3d12core.so`.

Remaining question is therefore B2 only: does the WSL Containers preview
(2.9.3+, that box is on 2.6.3.0) create an NPU device node?

**Primary source checked 2026-08-24, and it does not support the rumour.**
Microsoft's own WSL Containers announcement mentions GPU exactly once — a
CUDA example, `wslc run --rm --gpus all …` — and says nothing about NPU, AI
accelerators or `/dev/accel` anywhere. Its only performance claim is a
virtiofs filesystem speedup. The NPU-passthrough story lives entirely in
secondary coverage ([TechTimes](https://www.techtimes.com/articles/317598/20260602/wsl-3-build-2026-near-native-gpu-npu-passthrough-brings-local-ai-windows.htm),
[byteiota](https://byteiota.com/wsl3-gpu-npu-passthrough-windows-ai-dev/),
[it-connect](https://www.it-connect.tech/microsoft-unveils-wsl-3-and-wsl-containers-for-windows/)),
all of it under the "WSL 3" name Microsoft denied, and several pieces name
Meteor Lake and Lunar Lake specifically — a detail with no primary source
behind it.

That does not make B2 pointless; a devblog announcing a container feature is
not obliged to enumerate every device node. It does mean the prior is low,
and that the only evidence that will settle it is `ls /dev/accel*` inside
the preview.

### B2 result, measured 2026-08-24 — WSL Containers does not expose the NPU

Run on the **285K** after `wsl --update --pre-release` took it from WSL
2.6.3.0 to **2.9.8.0** (kernel 6.18.40.1), well past the 2.9.3 that gates
the WSL Containers preview. `wslc.exe` present at `C:\Program Files\WSL\`.

Four checks, cheapest first:

| Check | Result |
|---|---|
| `/dev/accel*` in a normal distro on 2.9.8 | absent — only `/dev/dxg`, exactly as on 2.6.3.0 |
| `/dev` in `wslc run --rm alpine` | minimal set; **no `dxg`, no `accel`** |
| `/dev` in `wslc run --rm --gpus all alpine` | `dxg` appears — **and nothing else** |
| `wslc run --help` hardware flags | `--gpus` is the **only** one. No `--device`, no NPU flag, no privileged mode |

That last row is the decisive one and it is not a measurement that can drift:
the CLI has no way to ask for an NPU, so no amount of driver work inside the
container reaches one. GPU passthrough is real and goes through the same
`/dev/dxg` paravirtualization WSL 2 already had — not a new mechanism.

**Verdict: the "WSL 3 brings NPU passthrough" reporting is false for what
actually shipped.** Combined with the primary-source check above (Microsoft's
own announcement mentions GPU once and NPU never), the story is closed: every
claim traces to secondary coverage under a product name Microsoft denied.

Re-open only if a future WSL release adds a device-passthrough flag. The
check is `wslc run --help` and takes ten seconds.

The commands, kept for the record (B1 and B2 both ran; B3 was never
reached because no device node appeared):

```bash
# B1. does stock WSL2 expose anything NPU-shaped at all?
wsl -e ls -la /dev/accel /dev/accel0 2>/dev/null || echo "no accel device"

# B2. if not, try the WSL Containers preview
wsl --update --pre-release && wsl --shutdown
wsl -e ls -la /dev/accel* 2>/dev/null || echo "still nothing"

# B3. only if a device node appears: userspace + enumeration
#     needs intel-npu-driver matching the HOST driver version
wsl -e bash -c '~/ovtest/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"'
```

| Result | Meaning |
|---|---|
| `NPU` listed **and** generates coherently | Genuinely new. Worth writing up publicly - nobody has shown this working |
| device node but no `NPU` device | Userspace/kernel version mismatch; recoverable, needs matching `intel-npu-driver` |
| no `/dev/accel*` in either mode | The rumour is false. Record it in `TODONT.md` and stop |

If B3 succeeds, apply the same suspicion as everywhere else in this project:
**enumeration is not correctness**. Run the probe set and compare against the
NPU numbers already in `docs/dev/models.md` before believing it.

Rolling back is a manual MSI install, not a flag — see the correction above.
The 285K was left on 2.9.8 after B2: WSL is unused on that box, and the
re-check trigger in TODONT wants `wslc` present anyway. If that box ever
becomes a measurement machine, revert it to stable first — this project
measures on shipped stacks.

---

## Phase 0 — gating: is the GPU visible *and correct* inside a container?

Visibility is not usability. This project has already been burned by a device
that enumerated fine and then computed garbage (openvino#37419), so check
both.

```bash
docker run --rm --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  openvino/ubuntu24_runtime:latest \
  python3 -c "import openvino as ov; c=ov.Core(); print([(d, c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices])"
```

| Result | Next |
|---|---|
| lists `GPU` = "Arc Pro B60" | Phase 1 |
| `CPU` only | **Stop.** Close #31 with the evidence; Linux-native may still work, note it as untested-here |

On native Linux the equivalent is `--device /dev/dri/renderD128`; we cannot
test that here, so any claim about it stays explicitly untested.

---

## Phase 1 — the NoLlama-specific hazards

The interesting failures are not "does the server start". These are the
places where containerization collides with something this codebase actually
does. Each has a concrete test and a concrete expected result.

### 1.1 Model naming — the directory name is authoritative

`resolve_display_name` uses the directory name as the API/UI model id and
only follows a symlink when that name is generic (`model/`, `gpu-model/`).
A bind mount is a rename waiting to happen.

```bash
# right: name preserved
-v /c/Users/wossn/models/gemma-4-E2B-it-int4-ov:/models/gemma-4-E2B-it-int4-ov
# wrong: clients would have to request "model"
-v /c/Users/wossn/models/gemma-4-E2B-it-int4-ov:/models/model
```

**Test:** `--scan` in-container, and `GET /v1/models`. **Expect** the real
directory name, not `model`. If compose encourages the generic form, the
compose file is wrong, not the code.

### 1.2 KV pool sizing under a container memory limit

`_resolve_kv_pool` and `_preflight_memory` size from the *device budget* —
GPU via `GPU_DEVICE_TOTAL_MEM_SIZE`, CPU via total RAM. A container sees host
RAM unless it reads cgroup limits, so the preflight may promise memory the
container cannot have.

```bash
docker run --memory=8g ... nollama --scan          # and a real load
```

**Expect:** the logged KV pool and token capacity to reflect **8 GB**, not
host RAM. If it reports host RAM, that is a real bug worth fixing regardless
of whether we ship Docker support — issue #21's `Got unfinished
GenerationStatus` is what a too-small pool produces.

### 1.3 Model load stages through host RAM

Loading a big model peaks at roughly model-sized host RAM even on a discrete
card (`docs/MODELS.md`, "peak system memory during load"). WSL2 caps its VM
at ~50% of host RAM by
default — on this 32 GB box that is ~16 GB, and the 26B needs 14.3 GB.

A `.wslconfig` was created 2026-08-24 in the user profile setting
`memory=24GB`, `swap=16GB` and `autoMemoryReclaim=gradual` precisely for
this, so the test measures the model rather than the default ceiling.

**Test:** load `gemma-4-26b-a4b-it-int4-ov` in-container. **Expect** a clean
load. **Also record** what it needs, because anything we ship must state the
`.wslconfig` requirement - a 32 GB box on defaults would thrash.

### 1.4 Weight integrity check over a bind mount

`_verify_weights_integrity` reads offsets from the `.xml` and checks the
`.bin` size. Over `/mnt/c` (drvfs) this is slow.

**Test:** time `--scan` against a bind-mounted model vs the same model on the
WSL2 ext4 filesystem. **Expect** correctness either way; **record** the time
difference, because "models on /mnt/c are slow" is the kind of thing that
looks like a hang.

### 1.5 Both API ports

NoLlama serves OpenAI on 8000 **and** Ollama on 11434. A compose file that
publishes only 8000 silently breaks every Ollama client.

**Test:** `/v1/models` on 8000 and `/api/tags` on 11434 from the host.

### 1.6 Streaming and the SSE heartbeat

Tool turns are buffered and rely on SSE keep-alives every `HEARTBEAT_SECS` to
stop client watchdogs aborting. A proxy or port-publishing layer that buffers
would defeat exactly the mechanism that exists to prevent aborts.

**Test:** a streaming request and a tools request through the published port;
confirm tokens/pings arrive incrementally, not in one lump at the end.

### 1.7 Writable state

`--idle-timeout 0` auto-enables prewarm, which writes `prewarm-<port>.json`
to the working directory. A read-only rootfs or a non-root user without write
access turns that into a startup failure.

**Test:** run with `--idle-timeout 0`, restart, confirm the prewarm file is
written and reused (log shows the startup prefill).

---

## Phase 2 — parity against numbers we already have

Same models, same probes, so the comparison is apples to apples. Native
figures are in `docs/dev/models.md` and `docs/MODELS.md`.

| Measure | Native (B60, 2026.3) | Container |
|---|---|---|
| `gemma-4-E2B-it-int4-ov` load time | ~16 s | ? |
| prefix cache: cold vs repeat TTFT | 3.1 s → 0.5 s | ? |
| `aweussom/gemma-4-E4B-it-int8-ov` steady TTFT | 1.16 s | ? |
| probe set correctness | 7 cases, known answers | must match exactly |

Reuse the probe harness — identical inputs, so any divergence is the
container and not the prompt. **It currently lives in a session scratchpad
and will vanish**; move it into the repo alongside `qa_vlm.py` before relying
on this section.

**A tok/s gap is publishable either way.** "Containers cost X%" is useful;
"containers are free" is more useful.

---

---

# RESULTS — measured 2026-08-24 on the B60 box

Phases 0, 1 and 2 all ran. **Verdict: it works, with one hard limitation and
one unexplained model-specific defect.** Two NoLlama bugs were found and
fixed on the way; both were container-only in effect, neither is
container-specific in cause.

Environment: Windows 11 Pro 26200, WSL 2.7.12.0 (kernel 6.18.33.2-2),
Ubuntu 24.04.4, Docker Desktop 4.87.0 / engine 29.7.2, Arc Pro B60 24 GB,
OpenVINO + GenAI 2026.3 (identical wheel versions in and out of the
container — verified, so nothing below is a version difference).

## The one-paragraph answer

An Intel GPU **is** usable from a Docker container on this box, at native
speed: 74-79 tok/s in-container vs 76-78 native on the same model, prefix
cache cold-vs-repeat 1.9s → 0.3s vs 2.1s → 0.2s native. Two things stand
between that and a supported path. First, **the stock
`openvino/ubuntu24_runtime` image cannot see the GPU at all** — its driver
predates Battlemage; you must build your own image on the current upstream
compute-runtime. Second, **WSL's `/dev/dxg` caps a single GPU allocation at
1 GiB** regardless of the card's real 23.3 GB, which kills any model with a
tensor over that size until a plugin hint is set. Neither limitation applies
to native Linux with `/dev/dri`, which is what #31 actually asked about —
but neither could be verified here either.

## Phase 0 — GPU visible and correct: PASS, but not with the stock image

The protocol's own command returns **CPU only**:

```
openvino/ubuntu24_runtime:latest  ->  [('CPU', 'AMD Ryzen 9 5950X ...')]
```

Not a container problem. That image ships `intel-opencl-icd 24.48.31907.7`
(NEO 24.48, Dec 2024), which predates Arc Pro B60 (BMG-G31, device id
`0xe211`). The Level Zero driver *does* reach the card — with
`NEOReadDebugKeys=1 PrintDebugMessages=1` it prints `Created Wddm context.
Status: :0, engine: 4` — and then enumerates nothing: `zeInit` returns
`ZE_RESULT_ERROR_UNINITIALIZED`, `clGetPlatformIDs` returns `-1001`
(`CL_PLATFORM_NOT_FOUND_KHR`). A device the driver does not know reads
exactly like a device that is not there.

Rebuilt on the current upstream release (compute-runtime 26.31.39395.13 +
IGC 2.40.13, installed over the image's own packages), the same command
gives:

```
[('CPU', 'AMD Ryzen 9 5950X 16-Core Processor'),
 ('GPU', 'Intel(R) Graphics [0xe211] (dGPU)')]
```

Visibility is not usability, so correctness was checked too — a
matmul+GELU compared against numpy on both devices:

| | max abs error vs numpy |
|---|---|
| CPU | 9.155e-05 |
| GPU (default fp16 inference) | 2.946e-02 |
| GPU, `INFERENCE_PRECISION_HINT=f32` | 9.155e-05 — **identical to CPU** |

So the fp16 delta is precision, not corruption. The GPU computes correctly.

Cosmetic note: NEO 26.31 reports `Intel(R) Graphics [0xe211] (dGPU)` where
the Windows driver reports `Intel(R) Arc(TM) Pro B60 Graphics (dGPU)`. Same
card, no marketing name in the Linux driver's table.

## THE BLOCKER — WSL /dev/dxg caps a single allocation at 1 GiB

Loading `gemma-4-E2B-it-int4-ov` in the container failed outright:

```
Exceeded max size of memory object allocation: requested 2348810240 bytes,
but max alloc size supported by device is 1073741824 bytes.
```

That 2.2 GB request is the model's
`openvino_text_embeddings_per_layer_model.bin`. The cap is exactly 1 GiB,
and it is a property of the /dev/dxg path, not of the card — the same B60,
same OpenVINO build, queried both ways:

| | `GPU_DEVICE_TOTAL_MEM_SIZE` | `GPU_DEVICE_MAX_ALLOC_MEM_SIZE` |
|---|---|---|
| native Windows | 25,055,051,776 | **25,055,051,776** |
| container over WSL `/dev/dxg` | 25,055,051,776 | **1,073,741,824** |

The driver names its own workaround in the error text, and it works:
compiling that same sub-model with `GPU_ENABLE_LARGE_ALLOCATIONS=True`
succeeds in 17.0s where the default config fails in 0.1s.

**Fixed in `nollama.py`** (`_gpu_large_alloc_props`): a GPU slot that
reports a max-allocation smaller than its own total budget gets the hint,
and says so at load. Gated on the reported numbers rather than on "am I in a
container", because the cap belongs to how the device was reached. Native
installs are unaffected — verified: the same code loads the 26B natively
without printing the line.

After the fix E2B loads and answers correctly in the container.

## Phase 1 — the NoLlama-specific hazards

| | Hazard | Result |
|---|---|---|
| 1.1 | model naming over a bind mount | **PASS** — `--scan` and `/v1/models` both report `SmolLM3-3B-int4-cw`, from the directory name. Integrity check passes over drvfs. Compose must bind under the real name; that is a compose-file rule, not a code change |
| 1.2 | KV pool under a container memory limit | **FAIL, now fixed** — see below |
| 1.3 | big model stages through host RAM | **PASS** — `gemma-4-26b-a4b-it-int4-ov` (15 GB) loads clean in 133s with `.wslconfig memory=24GB`; host RAM settles back to ~2 GB once the weights are on the card. It then answers incorrectly, which is a different problem — see the defect below |
| 1.4 | integrity check / load over drvfs | **PASS, and the worry was misplaced** — see below |
| 1.5 | both API ports | **PASS** — `/v1/models` on 8000 and `/api/tags` on 11434, both from the host |
| 1.6 | streaming and the SSE heartbeat | **PASS** — 61 chunks over 0.81s, first at 0.143s. Nothing buffers |
| 1.7 | writable state / prewarm | **PASS, with a caveat** — see below |

### 1.2 — confirmed as a real bug, fixed

`_system_ram_bytes` read `/proc/meminfo`, which reports the **host** total
inside a container. Measured in a `--memory=4g` container: `MemTotal` 23.5 GB,
`/sys/fs/cgroup/memory.max` 4 GB, and NoLlama sized **a 4 GB KV pool** on top
of 1.6 GB of weights, with no warning. Exactly the shape of issue #21's
`Got unfinished GenerationStatus`.

Fixed (`_cgroup_mem_limit_bytes`, cgroup v2 and v1, `min()` with physical
RAM). Re-measured, same model:

| container limit | KV pool before | KV pool after |
|---|---|---|
| `--memory=4g` | 4 GB | **2 GB** + the "agent prompts will exhaust it" warning |
| `--memory=8g` | 4 GB | **2 GB** |
| `--memory=24g` | 4 GB | 4 GB |
| no limit (host 23 GB) | 4 GB | 4 GB |

### 1.4 — drvfs is 35x slower to read and it does not matter

Raw sequential read of the same 1.6 GB `.bin`, warm both ways:

| | throughput |
|---|---|
| bind mount from `/mnt/c` (drvfs) | **178 MB/s** |
| docker volume on the WSL2 ext4 filesystem | **6.4-13.7 GB/s** |

And yet load time is indistinguishable — 6.13s / 6.21s from drvfs vs 6.30s /
7.08s from the volume, GPU, two runs each. OpenVINO maps the weights rather
than streaming them, and kernel compilation dominates. **Do not copy models
onto ext4 to make loading faster; it does not.** (Bulk *copying* over drvfs
is slow — 48.5s for 1.6 GB — which is the operation that gives drvfs its
reputation.)

### 1.7 — works, but the state lives in the container's writable layer

`--idle-timeout 0` auto-enables prewarm, a >4000-char system prompt is
captured to `/app/prewarm-8000.json`, and after `docker restart` the log
shows `pre-warmed prompt cache from prewarm-8000.json (0.3s)` with
`/health` reporting `"prewarmed": true`. Working as designed.

Two things a compose file must get right, neither a code bug:

- `docker run --rm` throws the file away with the container. Mount a
  writable volume, or point `--prewarm` at one.
- `--read-only` does **not** fail at startup, as this protocol guessed. It
  starts fine and then silently never persists — `_maybe_capture_prewarm`
  ends in `except OSError: pass`, so every restart is cold with nothing in
  the log to say why. Worth a log line; not fixed here.

## Phase 2 — parity: the container costs nothing measurable

Same box, same models, same prompts, container vs native.

| Measure | Native | Container |
|---|---|---|
| SmolLM3-3B steady-state | 75.7 / 77.8 / 77.1 tok/s | 74.2 / 70.3 / 78.8 tok/s |
| SmolLM3-3B TTFT (short prompt) | 76 / 54 / 37 ms | 71 / 53 / 32 ms |
| E2B prefix cache, ~4k-token prompt, cold → repeat | 2.1s → 0.2s | 1.9s → 0.3s |
| SmolLM3-3B load → ready | 4.4 / 6.6 s | 6.1 / 6.2 s |
| E2B load → ready | 11.5 / 11.5 s | 33 / 42.5 s |

Throughput and prefix caching are free. **Load time is not, for models that
need the large-allocation hint**: SmolLM3 (no tensor over 1 GiB) loads at
native speed, E2B (2.2 GB tensor, hint active) takes ~3x longer. Plausibly
the hint falls back to a slower allocation strategy; not investigated
further.

## The unexplained defect — gemma-4-26b-a4b is garbage on this path

`gemma-4-26b-a4b-it-int4-ov` loads cleanly in the container and then emits
deterministic gibberish (`- @__-...ls--...1-_--s-_-.lyje- de ...`),
byte-identical across runs and across pipelines. Everything else was held
constant and varied one axis at a time:

| | result |
|---|---|
| native Windows, GPU, same NoLlama, same OpenVINO build | **correct** — "The capital of Norway is Oslo, and 17 + 25 equals 42." |
| container, **CPU** | **correct** — same sentence |
| container, GPU, prefix cache on | garbage |
| container, GPU, `--no-prompt-cache` (plain pipeline) | garbage, *identical bytes* |
| container, GPU, with the large-allocation hint | garbage, identical bytes |

So it is not the container, not NoLlama, not the CB backend, not the
allocation cap, and not the model. It is the container **GPU** path.

It is also not "big models" or "MoE models" — those were checked:

| model in the container, on GPU | size | result |
|---|---|---|
| `SmolLM3-3B-int4-cw-ov` (dense) | 1.6 GB | correct |
| `gemma-4-E2B-it-int4-ov` (dense VLM) | 4.1 GB | correct |
| `gemma-4-E4B-it-int8-ours` (dense VLM) | 7.8 GB | correct |
| `Qwen3-30B-A3B-int4-ov` (**MoE**) | 16 GB | **correct** |
| `gemma-4-26b-a4b-it-int4-ov` (MoE) | 15 GB | **garbage** |

One model, one path. Determinism says compute defect, not memory
corruption.

Two further axes were checked, and both came back negative:

- **Driver version.** Rebuilt on NEO **25.44.36015.8** with matching IGC
  2.24.8 — a full generation back. Byte-identical garbage. Not a recent
  regression. (`GPU_DEVICE_MAX_ALLOC_MEM_SIZE` is 1 GiB on both releases
  too, so the allocation cap is a property of `/dev/dxg`, not of a driver
  version.)
- **Inference precision.** The fp16-overflow hypothesis that the Gemma
  family invites does not survive: native inference is fp16 by default and
  is correct. Forcing `INFERENCE_PRECISION_HINT=f32` in the container does
  not answer it either — that configuration fails to compile at all, in the
  GPU plugin's layout handling (`_type: any (format: bfyx, data_type: f32)`,
  shape `[?,8]`), so f32 is untestable here rather than clean.

Also ruled out by construction: NoLlama itself. The garbage reproduces
through a raw `openvino_genai.VLMPipeline` with none of our serving code in
the path.

That leaves the WSL `/dev/dxg` paravirtualization layer and this one
model's kernels. Clean upstream report; not filed yet. What would sharpen it
is a second gemma-4 **MoE** export to compare against — the int8 of the same
model is ~26 GB, which will not fit the card, so that test is not available
here. Gemma-4 *dense* is already known good on this path (E2B, E4B), and a
non-Gemma MoE is too (Qwen3-30B-A3B), so the suspect is specifically
gemma-4-MoE-on-dxg.

This is why Phase 0's "enumeration is not correctness" rule earns its place:
every check short of reading the output says this model is fine.

## What is still not answered

- **Native Linux with `/dev/dri`** — untested, and it is what #31 asks for.
  Both limitations found here (the 1 GiB cap, the stock image's driver age)
  are artefacts of the WSL path or of one vendor image; neither predicts the
  native answer. No claim either way.
  → **Run book: `docs/dev/linux-native-gpu-test.md`.** Self-contained, for a
  live USB on the B60 box; no install, no repartitioning. Step 2 alone (two
  integers) settles the 1 GiB question.
- ~~Track B / B2~~ — **answered 2026-08-24, and the answer is no.** The
  285K is now on WSL 2.9.8.0 with the WSL Containers preview; there is no
  NPU device node in either channel and `wslc run` has no device-passthrough
  flag. Full chain in the B2 section above and in TODONT.md.
- The 26B defect's cause.
- Whether the large-allocation hint is what costs E2B its 3x load time.

## Artefacts

Both images were built in a session scratchpad and are **not in the repo
yet** — that is Phase 3's job:

- an `openvino/ubuntu24_runtime` derivative with the current NEO, used for
  Phase 0 only
- `nollama:test` — Ubuntu 24.04, current NEO, runtime deps only (no
  `optimum`/`transformers`: the container never exports), `nollama.py` +
  `templates/` + `static/`, model-free, **1.28 GB**

Run shape that worked, for whoever writes the compose file:

```bash
docker run --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v /c/Users/you/models/<real-dir-name>:/models/<real-dir-name>:ro \
  -p 8000:8000 -p 11434:11434 \
  nollama:test --model-dir /models/<real-dir-name> --device GPU
```

`LD_LIBRARY_PATH` must **append** to the image's own value, not replace it —
replacing it costs you `libopenvino.so` and looks like a broken install.

## Phase 3 — packaging — DONE 2026-08-24

- `Dockerfile` — **not** the OpenVINO runtime base (see TODONT): Ubuntu
  24.04 + upstream compute-runtime, NEO/IGC/GMM pinned as build args that
  move as a set. Model-free, 1.28 GB
- `requirements-container.txt` — serving deps only. Dropping the export half
  costs the `--backend optimum` path inside the image; documented
- `docker-compose.yml` — native Linux `/dev/dri`, **untested**, marked as
  such in the file itself. Publishes 8000 **and** 11434, binds models under
  their real directory names (1.1), `/state` volume for prewarm (1.7)
- `docker-compose.wsl.yml` — the measured configuration: `/dev/dxg` +
  `/usr/lib/wsl`. Verified end-to-end from the repo files, both ports
- `.dockerignore` — keeps the venvs and models out of the build context
- `docs/DOCKER.md` — user-facing: no NPU in plain words, the `.wslconfig`
  requirement, the overhead numbers, and both defects
- `.gitattributes` — no `export-ignore`: the container path is measured and
  documented, not vestigial

## Closing out

Record the verdict in `TODONT.md` **even if the answer is no** — a documented
"we tested this and here is why not" is worth more to the next person than
silence. Then reply to #31 with the numbers.
