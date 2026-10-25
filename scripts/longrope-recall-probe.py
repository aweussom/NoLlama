"""Bare genai recall at growing prompt lengths: where does a model fall off?

Why: T-047's decay probe showed Phi-3.5-mini degenerate past ~5k tokens
through the server, and the standing order says rule out the server first.
This is that test with NoLlama absent: one planted code, real repo text as
filler, "what is the code?", greedy. It found the collapse in every
OpenVINO export of Phi-3.5-mini (int4-cw, int4, fp16) on GPU and CPU,
release and nightly, while transformers and llama.cpp read the same
prompts correctly [OBSERVED 2026-10-19] -- see docs/dev/models.md and
brain/next N-052, which this script is the attachment for.

`paged` adds SchedulerConfig with prefix caching, which is what NoLlama
runs on GPU/CPU; `plain` is genai's default path. Lengths are approximate
target token counts (filler sized at the 3.39 chars/token measured for
Phi-3.5's tokenizer); the printed count is the real one.

    python scripts/longrope-recall-probe.py <model-dir> <device> <plain|paged> [lengths...]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openvino_genai as ovg  # noqa: E402
from benchmark import long_corpus  # noqa: E402

model, device, mode = sys.argv[1], sys.argv[2], sys.argv[3]
lengths = [int(x) for x in sys.argv[4:]] or [2000, 3800, 4300, 6000, 12000]

kw = {}
if mode == "paged":
    sc = ovg.SchedulerConfig()
    sc.enable_prefix_caching = True
    sc.cache_size = 4
    kw["scheduler_config"] = sc
pipe = ovg.LLMPipeline(model, device, **kw)
tok = pipe.get_tokenizer()
SECRET = "GRAVLAKS-4417"

for n in lengths:
    # ~3.4 chars/token measured for this tokenizer by decay-probe
    filler = long_corpus(int(n * 3.39), offset=37_501)
    msgs = [{"role": "user", "content":
             f"The project access code is {SECRET}. Remember it.\n\n"
             f"Reference material:\n\n{filler}\n\n"
             f"What is the project access code? Answer with the code only."}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
    ntok = tok.encode(prompt).input_ids.get_shape()[1]
    cfg = ovg.GenerationConfig()
    cfg.max_new_tokens = 40
    cfg.do_sample = False
    t0 = time.time()
    out = pipe.generate(prompt, cfg)
    text = str(out).strip().replace("\n", " ")
    print(f"{device} {mode:5} {ntok:>6} tok  {'ok  ' if SECRET in text else 'FAIL'} "
          f"{time.time() - t0:6.1f}s  {text[:90]!r}", flush=True)
