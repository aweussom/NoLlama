# N-043 — Delete `aweussom/gemma-4-E4B-it-int8-ov` from the Hub

**Cost:** 15 min · **Asks:** Tommy (deleting a public repo is his call) ·
**Waiting on:** time. Not before 2026-01-11, not after 2026-04-11.

Our E4B re-export existed because Intel's IR had no fused SDPA op and got no
prefix caching. Intel's 2026-09-17 re-export fixed that, and on the B60 /
2026.4.1 it matches ours on every turn and loads faster, so `models.json`
went back to Intel's on 2026-10-11 (`4fc45fe`, TODONT "Gemma 4 E4B for agent
serving"). The model card carries a superseded notice since the same day
(HF commit `57956f2`). Kept up for now so installs made before the flip keep
working.

## When the date comes

1. Check the repo's monthly downloads on the Hub. Still meaningful traffic →
   push the date back once, then delete anyway.
2. Delete the repo on the Hub.
3. Same commit: drop the `aweussom/` row and links from `docs/MODELS.md` and
   `docs/dev/models.md`, the `aweussom/gemma-4-E4B-it-int8-ov` entry in
   `scripts/watched_revisions.json`, and say "deleted <date>" where
   `TODONT.md` names it. Leave the TODONT history otherwise intact.
4. `git rm` this item.

## Saying yes means

The Hub link 404s for anyone still on a pre-2026-10-11 `start.ps1`; a
re-run of `install.ps1` gets Intel's build.
