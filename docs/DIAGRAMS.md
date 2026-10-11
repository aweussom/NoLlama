# Logic diagrams

One diagram per flow, in `docs/*.mmd`. Indexed by the question each one
**answers** — open the one whose question you have. Conventions (node IDs are
real function names, dashed = failure/degraded, `%% covers:` headers) are in
CLAUDE.md's docs-toolkit block; `.\check-docs.ps1` flags a diagram when code
moves out from under it.

| Diagram | Answers |
| --- | --- |
| [request-flow.mmd](request-flow.mmd) | Which handler, slot, and generation path serves a given request? Why did my `tools` get ignored? How does a tool turn stream, and where is the call block held? |
| [slot-lifecycle.mmd](slot-lifecycle.mmd) | What happens between startup and "ready"? When does a slot get the CB backend vs the plain pipeline, how big is the KV pool, when does prewarm fire, and why doesn't a reload re-warm? |
| [stream-plumbing.mmd](stream-plumbing.mmd) | How do tokens travel from `pipe.generate` to the wire? Who owns cancel, why must the `finally: _cancel.set()` stay in the consumers, and what happens on a client disconnect or a silent backend? |
| [webui-render.mmd](webui-render.mmd) | How does the web UI render a streaming reply? Why does the think-block scroll survive redraws, what are the four think-tag states, and which layer stops model-output XSS? |
| [webui-ptt.mmd](webui-ptt.mmd) | What happens between pressing the mic button and text in the input box? Why is the audio converted in the browser, why is the button disabled over LAN http, and what stops silence from becoming invented text? |

Prose companions (not machine-checked): [INTERNALS.md](INTERNALS.md),
[API.md](API.md), [DEVICES.md](DEVICES.md).
