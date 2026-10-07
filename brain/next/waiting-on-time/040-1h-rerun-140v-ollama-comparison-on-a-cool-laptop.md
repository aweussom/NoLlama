# N-040 — Re-run the 140V Ollama comparison on a cool laptop, then decide direction

**Cost:** 1 h · **Asks:** Tommy · **Waiting on:** the laptop being somewhere
it does not throttle (back from Spain, or a cooling pad and a cold room)

## What

2026-10-07: Ollama 0.40 Vulkan reached decode parity with NoLlama on the B60
(60.6 vs 68.4 on count, 63.8 vs 63.8 thinking; `docs/BENCHMARKS.md`, "NoLlama
vs Ollama (Vulkan)"). The same-day 140V re-run is withheld: NoLlama fell from
20 to 12 tok/s inside 2.5 minutes and Ollama from 17.5 to 14, so the laptop was
measuring the room. The published June 140V row (21.7 vs 13.4, Ollama 0.30.8)
is still the only iGPU number, and it is stale.

## Do

Same recipe as the B60 run, laptop idle and cool, NoLlama first then Ollama,
never both loaded:

```powershell
.\venv\Scripts\python nollama.py --port 8010 --ollama-port 0 --device GPU --model-dir C:\Users\tommyl\models\Qwen3-8B-int4-cw-ov --idle-timeout 0 --no-prewarm
.\venv\Scripts\python benchmark.py --url http://localhost:8010 --label laptop-nollama-ov --runs 3 --llm-only
# stop it, then
$env:OLLAMA_IGPU_ENABLE=1; ollama serve        # the tray instance sees CPU only
.\venv\Scripts\python benchmark.py --backend ollama --model qwen3:8b --label laptop-ollama-vulkan --runs 3 --llm-only
```

Watch the server-side tok/s across the five tests. If it drops run over run,
the box is throttling again and the result is not a result. The 2026-10-07
withheld JSONs are in `bench-results/ollama-cmp/` (ignored, laptop only).

## Then

Tommy said on 2026-10-07 the repo "changes direction somewhat" once this is in.
If the iGPU is parity too, the GPU/CPU pitch is prefix cache, vision, NPU and
agent plumbing everywhere, and the README's "Run on" table and the roadmap
note in `docs/BENCHMARKS.md` follow. If the iGPU still shows a NoLlama edge,
say so with the number and keep the row.
