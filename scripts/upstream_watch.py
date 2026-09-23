#!/usr/bin/env python3
"""Upstream watcher — notify when a blocking upstream PR/issue changes state.

Some things NoLlama is waiting on are not models and never show up in
`model_watch.py`: they are pull requests and issues in other people's repos.
This polls a short, hand-written list of them and opens a GitHub Issue when one
actually moves — merged, closed, reopened, or out of draft.

Only *state* transitions are reported. Comments, pushes and label churn are
deliberately ignored: a watcher that fires every time someone types in a busy
upstream thread gets muted within a month, and then it is worth nothing on the
week that matters.

Diffs against a committed snapshot (watched_upstream.json), same as the model
watcher. First run establishes a baseline silently.

No third-party deps (urllib only), so the GitHub Action needs no pip install.

Outputs (for GitHub Actions, via $GITHUB_OUTPUT):
  changed=true   snapshot content changed (commit it back)
  new=true       something moved and an issue should be opened
Writes the issue title/body to scripts/.upstream_title and
scripts/.upstream_body.md.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / "watched_upstream.json"
TITLE_FILE = HERE / ".upstream_title"
BODY_FILE = HERE / ".upstream_body.md"

API = "https://api.github.com"

# Upstream work we are blocked on. Key is "owner/repo#number", value is why —
# it goes verbatim into the issue, so write it for whoever reads that issue in
# three months with no memory of this week.
WATCH = {
    "openvinotoolkit/openvino#38026":
        "Intel's own PR adding Q1_0 (1-bit ternary) to the GGUF frontend. Q2_0 "
        "(2-bit ternary) already landed in openvinotoolkit/openvino#37380, so "
        "this is the other half. "
        "Bonsai 8B ships Q1_0 as its released quantization — when this merges, "
        "that model becomes convertible and is worth a real test.",
    "openvinotoolkit/openvino#38076":
        "the fix for a segfault in Concat::evaluate on sub-byte types "
        "(openvinotoolkit/openvino#38057), which crashes compile_model on GPU "
        "for ternary models. CPU is unaffected. Until this merges, anything "
        "ternary is a CPU-only story on our boxes — worth knowing before "
        "promising a user a GPU number.",
    "openvinotoolkit/openvino#37457":
        "the standing 'please support Bonsai' feature request. It is the place "
        "an Intel decision would be stated, so a close (either way) is the "
        "earliest real signal about the family. See NoLlama issue #46.",
}


def api_get(path):
    """GET one GitHub API path as parsed JSON, or None if it fails.

    Why: every caller here must treat a failed fetch as "no news". Returning
    None rather than raising is what keeps a flaky network or a rate limit
    from being reported as an upstream state change — a false "merged!" is
    worse than a silent week.

    In: a path below the API root, e.g. "/repos/o/r/issues/1". Out: the
    decoded object, or None on any network, HTTP or decode failure (the
    reason goes to stderr). Sends the token from GH_TOKEN/GITHUB_TOKEN when
    one is set; unauthenticated works too, at 60 requests an hour.
    """
    req = urllib.request.Request(
        f"{API}{path}",
        headers={"User-Agent": "nollama-upstream-watch",
                 "Accept": "application/vnd.github+json"})
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except (urllib.error.URLError, urllib.error.HTTPError,
            ValueError, TimeoutError) as e:
        print(f"WARN: failed to fetch {path}: {e}", file=sys.stderr)
        return None


def fetch_item(key):
    """Current state of one watched issue or PR.

    Why: the issues endpoint answers for both kinds but never says whether a
    PR was *merged* — a merged PR and an abandoned one are both "closed"
    there, and those are opposite news. So a PR costs a second call.

    In: "owner/repo#number". Out: {"title", "kind", "state", "merged",
    "draft"}, or None if either call fails. `merged`/`draft` are always False
    for a plain issue, so callers can compare snapshots without special-casing
    the kind.
    """
    repo, _, number = key.partition("#")
    item = api_get(f"/repos/{repo}/issues/{number}")
    if not item:
        return None

    state = {"title": item.get("title", ""), "kind": "issues",
             "state": item.get("state", ""), "merged": False, "draft": False}
    if "pull_request" not in item:
        return state
    state["kind"] = "pull"

    pr = api_get(f"/repos/{repo}/pulls/{number}")
    if not pr:
        return None
    state["merged"] = bool(pr.get("merged"))
    state["draft"] = bool(pr.get("draft"))
    return state


def describe(was, now):
    """One sentence naming what changed, or None when nothing did.

    Why: the snapshot holds four fields, but only some combinations are news
    a human should be woken for. A title edit is not. Ordering matters —
    merged is checked before closed, because a merge is also a close and
    reporting it as "closed without merging" would say the opposite of what
    happened.

    In: the previous and current state dicts. Out: a markdown fragment, or
    None. Callers must treat None as "do not report", not as an error.
    """
    if now["merged"] and not was["merged"]:
        return "**merged**"
    if now["state"] == "closed" and was["state"] == "open":
        return "**closed** without merging"
    if now["state"] == "open" and was["state"] == "closed":
        return "**reopened**"
    if was["draft"] and not now["draft"]:
        return "out of **draft** — review can start"
    if not was["draft"] and now["draft"]:
        return "back to **draft**"
    return None


def check():
    """Poll every watched item and report the ones that moved.

    Why: the snapshot is rewritten even when nothing is reportable, so that a
    title edit or a newly added WATCH entry is absorbed quietly instead of
    surfacing next week as fake news.

    In: nothing. Out: (lines, changed) — markdown bullets for the issue body
    (empty when nothing moved) and whether the snapshot file needs
    committing. Items that fail to fetch keep their old snapshot entry.
    """
    try:
        before = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        before = {}

    lines, after = [], {}
    for key, why in WATCH.items():
        now = fetch_item(key)
        if not now:
            if key in before:
                after[key] = before[key]
            continue
        after[key] = now

        was = before.get(key)
        if not was:
            continue  # new to the list; baseline it silently
        what = describe(was, now)
        if what:
            repo, _, number = key.partition("#")
            lines.append(
                f"- [{key}](https://github.com/{repo}/{now['kind']}/{number}) — "
                f"*{now['title']}* — {what}.\n  Watched because: {why}")

    changed = after != before
    if changed:
        STATE_FILE.write_text(json.dumps(after, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    return lines, changed


def set_output(key, value):
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def emit_issue(lines):
    """Write the issue title/body files and flag the Action to open it."""
    TITLE_FILE.write_text(f"Upstream watch: {len(lines)} change(s)",
                          encoding="utf-8")
    BODY_FILE.write_text(
        "Upstream work NoLlama is waiting on has moved.\n\n"
        + "\n".join(lines)
        + "\n\n_A merge upstream is not a working feature here: it still needs "
        "an OpenVINO release or nightly carrying it, and then a real test on "
        "our own hardware. Watched items live in `WATCH` in "
        "`scripts/upstream_watch.py`._\n",
        encoding="utf-8")
    set_output("new", "true")


def main():
    lines, changed = check()
    if changed:
        set_output("changed", "true")
    if not lines:
        print("No upstream state changes since last run.")
        return 0
    emit_issue(lines)
    for line in lines:
        print(f"  moved: {line.splitlines()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
