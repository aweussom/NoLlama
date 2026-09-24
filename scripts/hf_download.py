"""Download a Hugging Face repo without invoking the `hf` console-script shim.

Why: `hf download` runs `venv\\Scripts\\hf.exe`, a generated launcher .exe.
Windows application-control policies (WDAC / AppLocker / Smart App Control)
routinely block those while allowing `python.exe` itself, so on a managed
machine the download dies with "En programkontrollpolicy har blokkert denne
filen" / "This file is blocked by an application control policy" and the
model never lands [OBSERVED 2026-09-01, Windows 11 Pro 26200 workstation].
Calling the library from python keeps one code path that works on both a
locked-down box and a loose one.

Auth needs nothing here: huggingface_hub reads HF_TOKEN from the
environment, which download-model.ps1 sets from -HfToken. A token that is
*sent* but no longer valid is the one auth case handled here -- see
download_with_fallback.

    python scripts/hf_download.py <repo-id> <target-dir> [--revision BRANCH]

Out: exit 0 on success, 1 with a readable reason otherwise. Resumes into an
existing directory -- complete files are skipped, so re-running after an
interrupted download is the documented recovery.
"""
import argparse
import sys


def _is_auth_rejection(e):
    """True when the hub refused the request's credentials (HTTP 401).

    In: any exception. Out: bool; False for anything without an HTTP
    response, so network and disk errors are never retried.
    """
    response = getattr(e, "response", None)
    return getattr(response, "status_code", None) == 401


def download_with_fallback(snapshot_download, get_token, repo_id, target, revision):
    """snapshot_download, retried once without credentials if they were rejected.

    Why: a stored login that has expired makes even a PUBLIC download fail.
    The hub rejects the stale token instead of ignoring it and reports
    "Repository Not Found" (401) -- a browser/OAuth login on the B60 box
    lapsed and a public OpenVINO repo became undownloadable [OBSERVED
    2026-09-24, "OAuth token has expired: exp claim"]. Retrying anonymously
    makes a public repo work regardless of the login's state. A genuinely
    missing or gated repo fails the retry too, so it costs one request, never
    a wrong answer.

    The trigger is a 401 whatever the message: huggingface_hub raises
    RepositoryNotFoundError for a 401 on the repo API, and keeps the response
    on the exception [DOCUMENTED huggingface_hub 0.36.2, utils/_http.py
    hf_raise_for_status]. A MALFORMED token does not trigger it -- the hub
    ignored `hf_` plus garbage and served the public repo [OBSERVED
    2026-09-24, hf-internal-testing/tiny-random-gpt2] -- so only a real
    expired or revoked token exercises this path end to end; the unit tests
    cover the logic.

    In: the two huggingface_hub callables (passed in so a test can fake
    them), repo, target dir, revision. Out: the local path; raises the
    ORIGINAL error if there was no token to drop or the retry also fails.
    """
    try:
        return snapshot_download(repo_id=repo_id, local_dir=target, revision=revision)
    except Exception as e:
        if not (_is_auth_rejection(e) and get_token()):
            raise
        print("WARNING: Hugging Face rejected the stored token (expired or revoked);"
              " retrying without it. Public repos still download; refresh the login"
              " for gated ones.", file=sys.stderr)
        try:
            return snapshot_download(repo_id=repo_id, local_dir=target,
                                     revision=revision, token=False)
        except Exception:
            raise e


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_id")
    ap.add_argument("target")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    try:
        from huggingface_hub import get_token, snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed in this python.",
              file=sys.stderr)
        print("  Activate the NoLlama venv, or: pip install huggingface_hub",
              file=sys.stderr)
        return 1

    try:
        path = download_with_fallback(snapshot_download, get_token,
                                      args.repo_id, args.target, args.revision)
    except Exception as e:
        # Deliberately broad: hub errors are a wide family (auth, network,
        # missing revision, disk) and the message is what the user needs,
        # not the class. download-model.ps1 prints the 401/403 hint.
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
