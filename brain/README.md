# brain/ — working state

One file per item. **The directory is the state**; `git mv` is the transaction.
`ls brain/next/1-now` is the view — nothing here renders a merged list.

- `next/` — needs Tommy. A decision, or someone chased.
- `todo/` — startable cold by anyone, no decision pending.
- `proposed/` — noticed, not acted on, **not trusted**. Read the tags.
- `done/` — finished *and* the reasoning outlives the work. Everything else
  finished is `git rm`'d; git holds it.

Buckets are `1-now 2-soon 3-someday`, plus `waiting/` and `waiting-on-time/`
under `next/` and `doing/` under `todo/`. `waiting/` is blocked on a **person**
— you chase them. `waiting-on-time/` is blocked on a **date or an event** — you
only check; the item's `**Waiting on:**` line names what and when.

Filename is the schema: `NNN-<cost>-<slug>.md`. IDs are shared across `next/`
and `todo/` and are never reused. Age is `git log -1 --format=%as -- <path>`,
never mtime. Commits name the item: `N-007: ...`, `T-014: ...`.

## Two deliberate deviations from ace-brain

- **`TODONT.md` stays at the repo root**, not in here. It is referenced from
  `CLAUDE.md` and a dozen docs by that path, and it is read by people who never
  open `brain/`. Moving it would buy conformance and cost every existing link.
- **No `memory/` directory.** This is a public repo and the memories name
  individuals, their hardware and internal Q-Free repos. They stay in the
  session store until someone does the scrub pass.

Where everything else lives:

| | |
|---|---|
| `STATUS.md` (repo root) | Where things stand right now. Read it first. |
| `TODONT.md` (repo root) | Rejected approaches and why. **Read before proposing anything structural.** |
| `CLAUDE.md` | Conventions and the per-area reading list. |
| `docs/dev/` | The deep notes. An item points at one; it does not duplicate it. |
