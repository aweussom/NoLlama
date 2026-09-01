#!/usr/bin/env python3
"""Model watcher — notify when new NoLlama-relevant models appear on Hugging Face.

Polls the Hugging Face Hub for these orgs:
  - OpenVINO    : Intel's pre-exported, ready-to-run models (what NoLlama loads).
  - Qwen        : upstream base models (early heads-up before Intel exports them),
                  filtered to the families NoLlama actually wants (Coder / VL / Omni).
  - nvidia      : Nemotron family (3.5 Lightning is a wanted agent model — blocked
                  on optimum-intel nemotron_h support, PR #1789).
  - meta-models : Muse Glimmer family (30B multimodal agent model, Apache 2.0).

Diffs the current relevant set against a committed snapshot (seen_models.json).
New ids are reported; the snapshot is updated so you're not re-pinged. On the
very first run the snapshot is empty, so it just establishes a baseline silently
(no issue) — only genuinely *new* models after that trigger a notification.

No third-party deps (urllib only), so the GitHub Action needs no pip install.

Outputs (for GitHub Actions, via $GITHUB_OUTPUT):
  changed=true   snapshot content changed (commit it back)
  new=true       there are new models worth an issue
Writes the issue title/body to scripts/.watch_title and scripts/.watch_body.md.
"""

import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEEN_FILE = HERE / "seen_models.json"
REVISIONS_FILE = HERE / "watched_revisions.json"

# Published models we are waiting on a *fix* for. Key is the repo id, value is
# why — it goes verbatim into the issue, so write it for whoever reads that
# issue in three months, not for today.
REVISION_WATCH = {
    "OpenVINO/gemma-4-E4B-it-int8-ov":
        "its IR has no fused SDPA op, so it silently gets no prefix caching. "
        "Intel confirmed the defect on 2026-08-31 (openvino.genai#4343) and "
        "said the stored IR needs re-exporting. When this fires, re-check with "
        "`--scan` and consider pointing models.json back at Intel's build "
        "(theirs is ~2.2x faster on a cold turn) — see TODONT.md.",
}
TITLE_FILE = HERE / ".watch_title"
BODY_FILE = HERE / ".watch_body.md"
MODELS_JSON = REPO / "models.json"

API = "https://huggingface.co/api/models"

# OpenVINO org carries lots of non-LLM assets (diffusion, detection, etc.).
# Keep only ids that look like a model family NoLlama serves.
OPENVINO_RELEVANT = re.compile(
    r"(qwen|coder|-vl|vl-|whisper|gemma|phi|deepseek|mistral|llama|internvl|granite|smol"
    r"|nemotron|glimmer)",
    re.I,
)
# Quant/format re-uploads that aren't the thing we'd export ourselves.
UPSTREAM_SKIP = re.compile(r"(gguf|awq|gptq|mlx|fp8|fp4|-base|autoround|eagle|dflash)", re.I)

# Things NoLlama can actually serve. Gated orgs (nvidia) brand everything
# "nemotron" — rewards, embedders, OCR — so name alone is too broad there.
SERVABLE_PIPELINES = {"text-generation", "image-text-to-text"}

# org -> (want, skip, gate_pipeline) over the repo name. skip=None keeps
# everything want hits; gate_pipeline=True additionally requires a
# SERVABLE_PIPELINES pipeline_tag.
WATCHES = {
    "OpenVINO": (OPENVINO_RELEVANT, None, False),
    "Qwen": (re.compile(r"(coder|-vl|vl-|omni)", re.I), UPSTREAM_SKIP, False),
    "nvidia": (re.compile(r"nemotron", re.I), UPSTREAM_SKIP, True),
    "meta-models": (re.compile(r"glimmer", re.I), UPSTREAM_SKIP, True),
}


def fetch_revision(mid):
    """Current commit sha and mtime of one Hugging Face repo.

    Why: the id diff below only ever sees *new* repos. When a published model
    is defective and upstream has agreed to re-upload it, the id never
    changes — only the commit does — so that model would go un-watched
    forever. This is the hook for "tell me when they actually fix it",
    without anyone having to remember to look.

    In: a full repo id. Out: {"sha", "lastModified"}, or None if the repo is
    unreachable — callers must treat None as "no news", never as a change,
    or a flaky network turns into a false fix report.
    """
    url = f"{API}/{mid}?blobs=false"
    req = urllib.request.Request(url, headers={"User-Agent": "nollama-model-watch"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
    except (urllib.error.URLError, urllib.error.HTTPError,
            ValueError, TimeoutError) as e:
        print(f"WARN: failed to fetch revision {mid}: {e}", file=sys.stderr)
        return None
    return {"sha": d.get("sha") or "", "lastModified": (d.get("lastModified") or "")[:10]}


def check_revisions():
    """Report watched repos whose commit changed since the last run.

    Why: separate snapshot from seen_models.json on purpose — that file is a
    flat list of ids with its own baseline-on-first-run rule, and overloading
    it would make an empty revision map look like a fresh baseline and
    swallow the first real change. Seeded with the known-bad sha at the time
    of writing, so the very next run reports a re-upload rather than quietly
    establishing a baseline.

    In: nothing. Out: a list of markdown lines (empty when nothing moved).
    Rewrites the snapshot as a side effect.
    """
    try:
        before = json.loads(REVISIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        before = {}

    lines, after = [], dict(before)
    for mid, why in REVISION_WATCH.items():
        now = fetch_revision(mid)
        if not now:
            continue
        after[mid] = now
        was = before.get(mid)
        if was and was.get("sha") and was["sha"] != now["sha"]:
            lines.append(
                f"- [`{mid}`](https://huggingface.co/{mid}) was re-uploaded "
                f"({was['sha'][:7]} → {now['sha'][:7]}, {now['lastModified']}). "
                f"Watched because: {why}")

    if after != before:
        REVISIONS_FILE.write_text(json.dumps(after, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return lines


def fetch_org(author, limit=1000):
    """Return [{id, createdAt, downloads, likes, pipeline_tag}] for an org."""
    url = (f"{API}?author={author}&limit={limit}"
           "&sort=createdAt&direction=-1&full=false")
    req = urllib.request.Request(url, headers={"User-Agent": "nollama-model-watch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def relevant(author, models):
    """Filter an org's model list to NoLlama-relevant ids."""
    want, skip, gate_pipeline = WATCHES[author]
    out = {}
    for m in models:
        mid = m.get("id") or m.get("modelId") or ""
        if not mid:
            continue
        repo = mid.split("/", 1)[-1]
        if not want.search(repo) or (skip and skip.search(repo)):
            continue
        if gate_pipeline and m.get("pipeline_tag") not in SERVABLE_PIPELINES:
            continue
        out[mid] = {
            "created": (m.get("createdAt") or "")[:10],
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "pipeline": m.get("pipeline_tag", ""),
            "source": author,
        }
    return out


def known_family_stems():
    """First two '-' tokens of each models.json repo name, e.g. 'qwen2.5-coder'.

    Used to tell '⬆ another size/rev of a family you already run' from a
    '✨ new family'. Purely advisory — quality still needs a human.
    """
    stems = set()
    try:
        data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return stems
    for entries in data.values():
        for e in entries:
            repo = (e.get("hf_id", "").split("/", 1)[-1]).lower()
            toks = repo.split("-")
            if len(toks) >= 2:
                stems.add("-".join(toks[:2]))
    return stems


def classify(mid, stems):
    repo = mid.split("/", 1)[-1].lower()
    toks = repo.split("-")
    stem = "-".join(toks[:2]) if len(toks) >= 2 else repo
    return "⬆ upgrade?" if stem in stems else "✨ new"


def set_output(key, value):
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def emit_issue(new_ids, current, revision_notes):
    """Write the issue title/body files and flag the Action to open it.

    Why: two independent triggers now share one issue — new models appearing,
    and a watched model being re-uploaded. Either alone must produce a
    coherent issue, so the sections are built independently and the title
    names whichever fired.

    In: the new ids (possibly empty) with their `current` metadata, and the
    re-upload note lines (possibly empty); at least one must be non-empty or
    the caller has nothing to report. Out: None — writes TITLE_FILE and
    BODY_FILE and sets the `new` output.
    """
    sections, headline = [], []

    if new_ids:
        headline.append(f"{len(new_ids)} new model(s)")
        stems = known_family_stems()
        rows = []
        for mid in sorted(new_ids, key=lambda m: current[m]["created"], reverse=True):
            info = current[mid]
            rows.append(
                f"| {classify(mid, stems)} | [`{mid}`](https://huggingface.co/{mid}) "
                f"| {info['source']} | {info['created']} "
                f"| {info['pipeline'] or '—'} | {info['downloads']} | {info['likes']} |")
        sections.append(
            f"**{len(new_ids)} new NoLlama-relevant model(s)** appeared on "
            "Hugging Face.\n\n"
            "`⬆ upgrade?` = another size/revision of a family you already list in "
            "`models.json`. `✨ new` = a family you don't track yet. "
            "Quality isn't judged here — verify before trusting.\n\n"
            "| | Model | Org | Created | Task | DLs/mo | ♥ |\n"
            "|---|---|---|---|---|---|---|\n" + "\n".join(rows))

    if revision_notes:
        headline.append(f"{len(revision_notes)} re-upload(s)")
        sections.append(
            "### Watched model re-uploaded\n\n"
            "A model we were waiting on a fix for has a new commit. Verify "
            "before believing it is fixed — re-download and check `--scan`, "
            "don't trust the commit alone.\n\n" + "\n".join(revision_notes))

    body = "\n\n---\n\n".join(sections) + (
        "\n\n_Watched orgs: OpenVINO (ready-to-run), Qwen (upstream Coder/VL/Omni), "
        "nvidia (Nemotron), meta-models (Muse Glimmer). "
        "To add one, drop it into the matching block of `models.json`. "
        "Watched revisions live in `REVISION_WATCH` in this script._")

    TITLE_FILE.write_text(f"Model watch: {' and '.join(headline)}", encoding="utf-8")
    BODY_FILE.write_text(body, encoding="utf-8")
    set_output("new", "true")


def main():
    current = {}
    for author in WATCHES:
        try:
            current.update(relevant(author, fetch_org(author)))
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as e:
            print(f"WARN: failed to fetch {author}: {e}", file=sys.stderr)

    # Independent of the org diff: a watched repo can be re-uploaded in a week
    # where no new model appears at all, and an org fetch failing must not
    # suppress it.
    revision_notes = check_revisions()

    if not current:
        print("No models fetched (network?); leaving snapshot untouched.")
        if revision_notes:
            emit_issue([], {}, revision_notes)
            # Without this the re-upload snapshot is never committed and the
            # same re-upload is reported again every week.
            set_output("changed", "true")
        return 0

    try:
        seen = set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        seen = set()

    baseline = not seen
    new_ids = sorted(set(current) - seen)

    # Persist the union so we never re-report and never lose history.
    merged = sorted(seen | set(current))
    SEEN_FILE.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    set_output("changed", "true")  # snapshot file changed → commit it

    if baseline:
        print(f"Baseline established: {len(current)} relevant models tracked. "
              "No issue (first run).")
        return 0

    if not new_ids and not revision_notes:
        print("No new models since last run.")
        return 0

    emit_issue(new_ids, current, revision_notes)
    for mid in new_ids:
        print(f"  new: {mid}")
    for mid in REVISION_WATCH:
        if any(mid in note for note in revision_notes):
            # ASCII only: this runs on a Windows console too, where the
            # note's own arrow glyph would raise UnicodeEncodeError.
            print(f"  re-upload: {mid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
