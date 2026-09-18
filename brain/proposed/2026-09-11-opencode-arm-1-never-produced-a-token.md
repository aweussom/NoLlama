# OpenCode evaluation arm 1 failed for a reason that did not reproduce

**Noticed:** 2026-09-11 (16:37-18:02) · **Status:** proposed, not trusted

## Claim

Six identical 46.6k-char build requests, each retried by OpenCode ~10-15 min
apart, **never produced a token** — no text event on the client, no completion
line on the server (which, before `ac4fc66`, logged nothing for a disconnected
client).

The same body replayed after a full restart gave a first token at 216 s, as did
bare genai. OpenCode's 5-minute chunk timer is reset by our keep-alives,
verified from a container. So the obvious explanations are all excluded and
none of them is the answer.

## Evidence

The window followed two MoE GPU-load aborts and an hour of 25 GB MoE compiles
on the same iGPU while an NPU server was up — see the sibling entry on MoE
loads aborting beside an NPU server, which may be the same underlying state.

That is circumstantial. Nothing here is a mechanism.

## Suggested home

Nowhere yet. Recorded so the **next** occurrence is measured rather than
guessed at: the client-gone log line now exists (`ac4fc66`), so run it again
with a 600 s budget and read that line.
