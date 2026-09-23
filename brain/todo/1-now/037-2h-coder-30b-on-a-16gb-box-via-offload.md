# T-037 — Can the one passing agent model run on a 16 GB laptop via offload?

**Effort:** 2 h plus one reboot each way · **Produces:** the 16 GB answer
for agentic coding — a recipe or a measured "nothing qualifies" · **Area:**
`--offload-ratio`, `docs/MODELS.md`, installer scenario menu

Why this and not another small model: every small candidate has now failed
the agent probe in both tool dialects (tier-2 table, `docs/dev/models.md`).
Tool training is the constraint, and `Qwen3-Coder-30B-A3B` is the one model
that passes. It is 16.3 GB. `--offload-ratio` streams part of its experts
from disk, and a 16 GB Lunar Lake has the same XMX iGPU as our 140V.

What to size against [DOCUMENTED `docs/dev/machines.md`]: a stock iGPU gets
~half of RAM, so **~8 GB** on a 16 GB box, shared with the KV pool. Resident
weights have to land near 5-6 GB → start at **ratio 65**.

What argues it will be too slow — read before believing a fast result:

- prefill touches every expert, so each prefill chunk re-streams the
  offloaded set: 10.6k tokens at ratio 30 took **138 s TTFT** on this 140V
  [OBSERVED 2026-08-09, Qwen3-30B-A3B]. An OpenCode first turn is ~10k tokens,
  and the ratio here is twice that.
- that run was served **from the Windows file cache, SSD idle**. On a 32 GB
  box the whole 16 GB file fits in cache; on a 16 GB box it cannot. **A run
  on the laptop at full RAM measures RAM, not disk, and proves nothing.**
- greedy decode varied 87-2040 tokens at ratio 30 on the B60 —
  `brain/proposed/2026-09-12-offload-non-determinism-on-the-b60.md`, unread
  content. A pass here should be run twice.

## The run — laptop booted as a 16 GB machine

Reboot with the RAM cut, so the page cache and the iGPU's half-of-RAM
ceiling both shrink the way they would on a real 16 GB box (elevated):

```powershell
manage-bde -protectors -disable C: -RebootCount 1    # BCD edit trips BitLocker recovery otherwise
bcdedit /set "{current}" truncatememory 0x400000000   # ignore RAM above 16 GiB
# reboot. Shared GPU Memory Override back to default for the run, or the
# ceiling is not stock -- note which it was.
```

```powershell
.\venv\Scripts\python nollama.py --model-dir $HOME\models\Qwen3-Coder-30B-A3B-Instruct-int4-ov `
    --device GPU --port 8000 --ollama-port 0 --offload-ratio 65 --cache-size-gb 2 `
    --idle-timeout 0 --log-file bench\coder30b-offload65-16gb.log
.\scripts\agent-probe.ps1 -Url http://127.0.0.1:8000/v1 -TimeoutSec 2400
```

Undo: `manage-bde -protectors -disable C: -RebootCount 1`, then
`bcdedit /deletevalue "{current}" truncatememory`, reboot.

- [ ] record: RAM seen by Windows, iGPU ceiling (`GPU_DEVICE_TOTAL_MEM_SIZE`
      at load), resident GB, cold TTFT, per-turn TTFT, wall clock, verdict,
      driver version
- [ ] if the preflight refuses or it will not load: ratio 75, then 85
- [ ] a pass: run it again (non-determinism), then the installer's 16 GB
      scenario gets this recipe with its wall clock stated plainly
- [ ] a fail or a 40-minute task: `docs/MODELS.md` says nothing qualifies
      at 16 GB, and why, with these numbers

## Done when

`docs/MODELS.md` tells a 16 GB laptop owner what to install for agentic
coding, or that nothing qualifies — from a run on a machine that was 16 GB
at the time.
