# Greedy decoding is non-deterministic under `--offload-ratio 30` on the B60

**Noticed:** 2026-09-12 · **Status:** proposed, not trusted

## Claim

At `--offload-ratio 30`, **greedy** decoding returned between 87 and 2040
tokens for the same prompt across five runs. Resident bytes were identical
every time (478).

Greedy should be deterministic. Varying length proves *something* varies.

## Evidence

Five runs, length only. **Nobody has looked at whether the content is wrong or
merely different** — which is the whole question, and it is one afternoon's
work to answer.

Detail in `TODONT.md`. Related but separate: T-027, the resident fraction not
tracking the ratio.

## Suggested home

`docs/dev/moe-offload.md` once someone reads the actual outputs. If the content
is wrong, this stops being a curiosity and becomes a correctness bug in the
offload path.
