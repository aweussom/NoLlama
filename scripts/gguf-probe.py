r"""Load one .gguf through bare openvino_genai and ask it two short questions.

Why: bare-probe.py takes an exported IR directory; a GGUF goes through a
different loader inside genai (its own legacy reader, or OpenVINO's GGUF
frontend since openvino.genai#4318), and the two fail differently. Bonsai's
Q1_0 GGUF is what this was written for (TODONT.md, "Bonsai 8B's Q1_0 GGUF").
Same rule as bare-probe: NoLlama is not imported, so a result here is the
runtime's, not ours.

One reader and one device per process, on purpose: a native crash in one
combination would otherwise take the rest of the run with it and look like
"no output". Run it once per combination:

    venv-nightly\Scripts\python scripts\gguf-probe.py <file.gguf> --device CPU --reader FRONTEND
    venv-nightly\Scripts\python scripts\gguf-probe.py <file.gguf> --device GPU --reader LEGACY

Out: stack versions, load time, then each prompt's answer, token count and
decode speed. Exits 0 when everything ran, 1 on a load or generate failure
(with the full exception message, not only OpenVINO's "Exception from
<file>:<line>" first line).
"""
import argparse
import logging
import sys
import time
import traceback

import openvino as ov
import openvino_genai as ovg

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(funcName)s — %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# Two prompts with answers that are trivially right or wrong, so a word-salad
# model cannot pass by producing fluent text.
PROMPTS = (
    "What is the capital of Norway? Answer in one word.",
    "Count from 1 to 10, separated by commas.",
)


def main():
    """Parse arguments, load the GGUF, run each prompt, report.

    Why: see the module docstring.

    In: argv. Out: process exit code; 1 if the pipeline would not load or a
    generate raised, 0 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gguf")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--reader", default="FRONTEND", choices=["FRONTEND", "LEGACY"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    log.info("openvino %s | openvino_genai %s", ov.__version__, ovg.__version__)
    log.info("loading %s on %s with GGUF_READER=%s", args.gguf, args.device, args.reader)
    t0 = time.perf_counter()
    try:
        pipe = ovg.LLMPipeline(args.gguf, args.device, GGUF_READER=args.reader)
    except Exception as exc:
        log.error("LOAD FAILED after %.1fs: %s", time.perf_counter() - t0, exc)
        traceback.print_exc()
        return 1
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    failed = False
    for prompt in PROMPTS:
        log.info("asking %r (greedy, max_new_tokens=%d)", prompt, args.max_new_tokens)
        history = ovg.ChatHistory()
        history.append({"role": "user", "content": prompt})
        try:
            result = pipe.generate(history, max_new_tokens=args.max_new_tokens, do_sample=False)
        except Exception as exc:
            log.error("GENERATE FAILED: %s", exc)
            traceback.print_exc()
            failed = True
            continue
        metrics = result.perf_metrics
        n_tokens = metrics.get_num_generated_tokens()
        tps = metrics.get_throughput().mean
        ttft_ms = metrics.get_ttft().mean
        answer = result.texts[0]
        log.info("answer (%d tokens, TTFT %.0f ms, %.1f tok/s): %r", n_tokens, ttft_ms, tps, answer)

    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
