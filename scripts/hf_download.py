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
environment, which download-model.ps1 sets from -HfToken.

    python scripts/hf_download.py <repo-id> <target-dir> [--revision BRANCH]

Out: exit 0 on success, 1 with a readable reason otherwise. Resumes into an
existing directory -- complete files are skipped, so re-running after an
interrupted download is the documented recovery.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_id")
    ap.add_argument("target")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed in this python.",
              file=sys.stderr)
        print("  Activate the NoLlama venv, or: pip install huggingface_hub",
              file=sys.stderr)
        return 1

    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.target,
            revision=args.revision,
        )
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
