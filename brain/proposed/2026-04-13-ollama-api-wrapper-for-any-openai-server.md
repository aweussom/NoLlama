# Spinoff: an Ollama-API wrapper for any OpenAI-compatible server

**Noticed:** 2026-04-13 · **Status:** proposed, not trusted

## Claim

The Ollama shim in `nollama.py` is generic enough to be extracted: point it at
any OpenAI-compatible server and it would give that server an Ollama surface.

## Evidence

Honest assessment from when it was written (2026-04-13): the valid case is
**narrow**. Most clients that speak Ollama also speak OpenAI, and the ones that
do not usually want Ollama's *model management* (`pull`, `list`, `show`) rather
than just `/api/chat` — which a wrapper cannot provide for a server that has no
model store.

What NoLlama actually needs from the shim is tied to its own slots and devices
(`<model>@GPU`, `/api/tags` from the loaded slots, the VS Code version
handshake), so extraction would mean generalising exactly the parts that earn
their keep.

## Suggested home

A separate repo, if ever. The narrow valid case is "an OpenAI server behind a
client that only speaks Ollama's chat endpoints", which is worth a weekend and
not more.
