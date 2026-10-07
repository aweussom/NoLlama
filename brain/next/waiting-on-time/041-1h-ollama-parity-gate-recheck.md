# N-041 — The Ollama parity gate: re-measure the 140V every Ollama Vulkan step

**Cost:** 1 h · **Asks:** Tommy · **Waiting on:** the next Ollama release that
touches its Vulkan or Intel path, or 2027-01 at the latest

## The gate, stated 2026-10-07

Tommy: "When Ollama has parity, I will stop competing and focus on NPU."
Parity means decode on the **iGPU**, not the dGPU; the B60 reached parity on
2026-10-07 and that changed nothing, because the users are on 16 GB laptops.

Where it stands: 140V, cool, Qwen3-8B 4-bit, `benchmark.py --runs 3
--llm-only`: NoLlama 20.3 vs Ollama 0.40.0 13.7 on count (1.5x), 19.0 vs
14.6 thinking (1.3x). June, Ollama 0.30.8: 1.6x. So the gap closed by a
tenth or two in four months. Not parity; we continue.

## Do

Same recipe as `docs/BENCHMARKS.md`, "NoLlama vs Ollama (Vulkan)": NoLlama
first, then `OLLAMA_IGPU_ENABLE=1` Ollama against `127.0.0.1`, laptop cool
and idle, watch that every test holds its rate across its three runs. Add a
row to that section with the Ollama version. If the count ratio is at or
under 1.1x, the gate has tripped: bring it to Tommy, do not decide here.
