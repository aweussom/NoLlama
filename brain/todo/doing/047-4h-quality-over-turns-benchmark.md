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

Everything needed is already here, which is why this is 4 h and not a project:

- oracle — `scripts/judge.py`, already calibrated against the heuristics
- filler — `benchmark.py`'s `long_corpus(n_chars, offset)`, real repo text
- harness — `scripts/agent-loop-test.py`'s history rewrite, which is the part
  that is easy to get wrong

Ask the **same** question at turn 1, 5, 10, 20, padding context between turns,
and judge each answer. The control is the same question asked cold at turn 1
with the same prompt length, so prompt size and turn count are separated —
otherwise a result cannot tell "long prompt" from "many turns", and those are
different bugs with different fixes.

- [x] `scripts/decay-probe.py`: question, turn schedule, padding, judge per turn
- [x] cold-at-length control arm
- [x] first real runs: `LFM2.5-24B-A2B` on the B60 GPU and the 285K CPU
- [ ] **calibrate the probes** (see below) — blocking, the current set is unsound
- [ ] **match the control arm on tokens, not turn index** — blocking
- [x] T-048 done: `finish_reason` now reports "length" at the cap, so a
      truncated answer is no longer scored as a wrong one
- [ ] run it on `Qwen3-Coder-30B-A3B` (the one agent-capable model) on the B60
- [ ] `docs/BENCHMARKS.md` section, device and driver recorded per the standing order

## What the first runs taught (2026-10-14)

Three runs, three faults, none of them in the model. Recorded because each one
is the kind that produces a confident wrong curve.

**1. A probe the model fails COLD measures nothing.** The `reason` probe
(3 x 1200 x 4 = 14400) failed at turn 1 with a 152-token prompt — the model
answered 1200. A probe without headroom cannot decay, and its failures at
turn 20 look identical to its failures at turn 1. **Calibrate first**: run
every probe N times cold, keep only those that pass every time, and record the
pass rate in the output. A probe set is per-model, not universal.

**2. The control arm was not matched.** It pads with `turn - 1` chunks while
the conversational arm accumulates padding *and* the model's own replies, so
at "turn 20" the arms sat at 19,512 and 31,869 prompt tokens. Comparing them
compares two different prompt lengths and calls the difference turn count.
Match on measured `prompt_tokens`, adding filler to the control until it
reaches the conversational arm's figure.

**3. `finish_reason` lied, so truncation read as decay.** Fixed (T-048,
closed 2026-10-14). Rows of
~7,530 characters — exactly the 3,000-token cap — came back as `stop`. Two
runs were scored before that was spotted, and the char counts, not the
protocol, are what gave it away.

## Not yet a verdict on the model

With those three open, the only thing worth repeating from the runs so far is
a mechanism, not a number: the model loops inside an **unclosed `<think>`**
until the cap, and `_assistant_message`'s rescue path hands that back as
content. The judge names it ("degenerate repetition") where the mechanical
substring check passes, because the planted token does appear — inside the
loop. That disagreement is the design working. The frequency-versus-turns
question stays open until the probes are calibrated.

## Why it is `todo/` and not `next/`

No decision is pending. The method above is a starting position, not a
proposal needing a verdict — if the first run shows the curve is flat, that is
the finding and the script stops being interesting.

## Done when

`docs/BENCHMARKS.md` can answer "does this model get worse after N turns" with
a number, for at least one model on at least one device, with the
prompt-length arm separated from the turn-count arm.
