# Spinoff: the Claude Code CLI as an Ollama backend

**Noticed:** 2026-05-26 · **Status:** proposed, not trusted

## Claim

A shim that speaks the Ollama API and answers from the Claude Code CLI would
let any Ollama client reach a cloud model through a local socket.

## Evidence

None — this is an idea, not a finding.

**It is explicitly not NoLlama.** NoLlama is local Intel inference; this is the
opposite, cloud Anthropic through a local CLI. Captured here only so it does
not get lost, and so nobody proposes bolting it onto this server.

Practical concerns already noted when it was first written down: the CLI is
interactive and stateful, so the IPC shape is the hard part, not the API
translation; and there is no streaming token interface to map onto Ollama's.

## Suggested home

A separate repo, if ever. Not this one.
