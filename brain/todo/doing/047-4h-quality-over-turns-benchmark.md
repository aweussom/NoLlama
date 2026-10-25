# T-047 — Nothing measures answer *quality* as context and turns grow

**Effort:** 4 h · **Produces:** `scripts/decay-probe.py` + a
`docs/BENCHMARKS.md` section · **Area:** benchmarks, `scripts/judge.py`

We measure that long context still **runs**, and how much **slower** it gets.
Nobody measures whether the answers get worse. The gap was noticed on
2026-10-14 while asking whether we could say anything about
`LFM2.5-24B-A2B` at its 131k context.

What exists, and what each actually proves:

| | |
|---|---|
| `benchmark.py --long` | a ~100k-char real-text prompt completes, 3 runs, pass/fail. Built for #33's allocation cap — it proves the prompt survives |
| `scripts/model-test.py` | RUNAWAY (never emits EOS) and EMPTY on real work. Single turn |
| `scripts/judge.py` | LLM-as-judge on **one** completion |
| `scripts/agent-loop-test.py` | two turns, round-trip correctness, pass/fail |
| `docs/BENCHMARKS.md` | the speed curve once context outgrows the KV pool: 82k chars → 58 s TTFT, 140k → 108 s, 199k → 175 s |

The one quality-over-turns failure we have on record was found by a human
reading output, not by a test: a Phi-3.5-mini run "answered the previous
turn's question using this turn's research" (`scripts/judge.py` header).
`judge.py` exists because that kept happening.

## The shape

Ask the **same** question at turn 1, 5, 10, 20, padding context between turns,
and judge each answer. The control is the same question asked cold with the
same **measured prompt token count**, so prompt size and turn count are
separated — otherwise a result cannot tell "long prompt" from "many turns",
and those are different bugs with different fixes.

`scripts/decay-probe.py` does this, built from parts that already existed:
`judge.py` as oracle, `benchmark.py`'s `long_corpus()` as filler,
`agent-loop-test.py`'s history rewrite as harness.

- [x] probe, turn schedule, padding, judge per turn
- [x] cold-at-length control arm, matched on measured `prompt_tokens`
      (chars/token measured on the model's own tokenizer; Qwen3-8B within
      0.4-2.7% at every checkpoint)
- [x] calibration: every probe asked cold 3x first, dropped unless 3/3,
      pass rate kept in the output. The probe set is per-model, not universal
- [x] T-048: `finish_reason` reports "length" at the cap, so truncation is
      no longer scored as a wrong answer
- [x] `_digits_only` strips every thousands separator that is legal somewhere
      (U+202F, nbsp, thin space, comma, apostrophe, …) — LFM2.5 wrote
      `14 400` with a narrow no-break space and scored 0/5 while right
- [x] **Qwen3-8B, B60**: no decay to turn 30 / 31.8k tokens, both arms,
      mechanical and judge agreeing on every row (`decay-qwen3-8b-b60-gpu-v3`,
      2026-10-19, driver 8805 in the provenance, OV 2026.4.1)
- [x] **Qwen3-Coder-30B-A3B, B60**: no decay to **turn 80 / 85-88k tokens**,
      both arms, every row passing (`decay-qwen3-coder-30b-b60-gpu-v2`,
      2026-10-19, driver 8805)
- [x] **Qwen3-Coder-30B-A3B, 285K CPU**: no decay to turn 40 / 42k tokens in
      the conversational arm. The cold control at 40 timed out (1800 s client
      timeout on a cold 42k prefill on CPU), so that cell is a missing
      number, not a failure (2026-10-20)
- [ ] **Phi-3.5-mini, B60: collapses into gibberish from turn 5 (~6k tokens)
      in BOTH arms**, so it is prompt length, not turn count. Turn 1 already
      misquotes the planted code. [INFERRED] the LongRoPE short→long factor
      switch past 4,096 tokens; confirm with `bare-probe.py` at 3k vs 6k
      before anyone blames the model or the server (`decay-phi35-mini-b60-gpu`,
      2026-10-19)
- [ ] **Qwen3.5-4B, B60: void.** Calibration hit the 2000-token cap on
      recall/reason (thinking), then the server went to `status: error`
      and answered 503. Re-run needs the cause of the error state first
- [ ] **LFM2.5-24B-A2B: still no calibrated run.** Every LFM file on the B60,
      including the 285K CPU one dated 2026-10-20, has no `calibration` block,
      so all are void; see below
- [ ] the `format` probe is unusable on every Qwen model (0/3 cold): it
      measures nothing there. Fix the probe or drop it from the write-up
- [ ] `docs/BENCHMARKS.md` section, device and driver recorded per the
      standing order. Qwen3-8B and Coder-30B already qualify. Raw JSON is
      only on the B60 (`bench/decay-*.json`), not in the repo

## Benchmark faults found before any model fault (2026-10-14)

Four faults in the first three days, all in the benchmark, none in a model.
Each one produces a confident wrong curve, and each was caught by a number
that looked wrong rather than by the protocol. Kept here as the checklist for
the next probe anyone writes:

1. **A probe the model fails cold measures nothing.** Its failures at turn 20
   are its failures at turn 1. Calibrate first.
2. **Control matched on turn index, not tokens.** 19,512 vs 31,869 prompt
   tokens at the same nominal "turn 20", because the conversational arm also
   accumulates the model's own replies.
3. **`finish_reason` lied** (T-048). Rows of ~7,530 chars, exactly the
   3,000-token cap, came back as `stop`.
4. **The checker was stricter about formatting than the model.** A correct
   number with a locale separator scored 0/5. At temperature 0 the same model
   used a comma and passed, so it looked like a regression.

Also found on the way, filed separately: T-049, sampling is deterministic
because `rng_seed` is never set, so five calls at temperature 0.7 were n=1.

## Open observation on LFM2.5-24B-A2B

From the void runs, one mechanism survives as worth re-checking: the model
loops inside an **unclosed `<think>`** until the cap, and `_assistant_message`'s
rescue path hands that back as content. The judge calls it "degenerate
repetition" where the substring check passes, because the planted token does
appear — inside the loop. That disagreement is the design working. Whether
the frequency rises with turns is the open question; no number until the
calibrated re-run.

## Done when

`docs/BENCHMARKS.md` can answer "does this model get worse after N turns" with
a number, for at least one model on at least one device, with the
prompt-length arm separated from the turn-count arm. The Qwen3-8B result
qualifies once it is written up with its driver.
