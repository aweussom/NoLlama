"""Build the Whisper test set under ROOT/set from downloaded archives.

    python scripts/whisper-build-set.py <root>

Expects, under <root> (each in its own directory, they are untrusted data):
  smoke/jfk.wav                     whisper.cpp samples/jfk.wav
  smoke/how_are_you_doing_today.wav storage.openvinotoolkit.org/models_contrib/
                                    speech/2021.2/librispeech_s5/
  fleurs-dl/{nb_no,en_us}-test.tsv + -test.tar.gz   google/fleurs data/<lang>/
  npsc-dl/20170207.json + 20170207_16K_mp3.tar.gz   NbAiLab/NPSC data/test/

Every clip becomes 16 kHz mono float WAV plus a manifest.jsonl line with the
reference text. First N rows of each source, so the set is reproducible.
Groups: smoke 2, fleurs-no 20, fleurs-en 20, npsc 20, long 1 (~62 s of
FLEURS nb joined), silence 1 (10 s). Used for T-042.
"""
import json
import os
import sys
import tarfile

import numpy as np
import soundfile as sf

ROOT = sys.argv[1]
N = 20
OUT = os.path.join(ROOT, "set")
SR = 16000


def put(name, audio, ref, lang, group, manifest):
    """Write one clip as WAV and append its manifest record.

    Why: one place decides the on-disk layout the probe reads back.

    In: float audio at 16 kHz, reference text ('' for silence). Out: none;
    appends to manifest. Overwrites an existing clip of the same name.
    """
    path = os.path.join(OUT, group, name + ".wav")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio.astype(np.float32), SR)
    manifest.append({"path": os.path.relpath(path, OUT), "ref": ref, "lang": lang,
                     "group": group, "seconds": round(len(audio) / SR, 2)})


def load(f):
    """Read audio from a path or file object as mono float32.

    Why: every source here is already 16 kHz; asserting it beats silently
    resampling a clip whose rate we got wrong.

    In: path or tar member file object. Out: 1-D float32 array; raises
    AssertionError on any other rate.
    """
    a, sr = sf.read(f, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    assert sr == SR, sr
    return a


def main():
    """Build every group and write manifest.jsonl.

    Why: one command rebuilds the exact set the docs' numbers came from.

    In: archives as listed in the module docstring. Out: <root>/set/.
    Slow (a minute or two) because it scans gzip tars member by member.
    """
    manifest = []
    for name, ref in [("jfk", "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."),
                      ("how_are_you_doing_today", "How are you doing today?")]:
        put(name, load(os.path.join(ROOT, "smoke", name + ".wav")), ref, "en", "smoke", manifest)

    long_parts, long_refs = [], []
    for lang, code in (("no", "nb_no"), ("en", "en_us")):
        rows = {}
        for line in open(os.path.join(ROOT, "fleurs-dl", f"{code}-test.tsv"), encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            rows.setdefault(c[1], c[2])
        picked = list(rows)[:N]
        with tarfile.open(os.path.join(ROOT, "fleurs-dl", f"{code}-test.tar.gz")) as t:
            for m in t:
                base = os.path.basename(m.name)
                if m.isfile() and base in picked:
                    a = load(t.extractfile(m))
                    put(base[:-4], a, rows[base], lang, f"fleurs-{lang}", manifest)
                    if lang == "no" and sum(len(p) for p in long_parts) < 60 * SR:
                        long_parts += [a, np.zeros(SR // 2, np.float32)]
                        long_refs.append(rows[base])
    put("fleurs-no-concat", np.concatenate(long_parts), " ".join(long_refs), "no", "long", manifest)

    meta = [json.loads(l) for l in open(os.path.join(ROOT, "npsc-dl", "20170207.json"), encoding="utf-8")]
    meta = [m for m in meta if m["end_time"] - m["start_time"] >= 3000][:N]
    want = {os.path.basename(m["path"]) + ".mp3": m for m in meta}
    with tarfile.open(os.path.join(ROOT, "npsc-dl", "20170207_16K_mp3.tar.gz")) as t:
        for m in t:
            base = os.path.basename(m.name)
            if m.isfile() and base in want:
                put(base[:-4], load(t.extractfile(m)), want[base]["normsentence_text"], "no", "npsc", manifest)

    put("silence-10s", np.zeros(10 * SR, np.float32), "", "none", "silence", manifest)
    with open(os.path.join(OUT, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(r["group"] for r in manifest))


if __name__ == "__main__":
    main()
