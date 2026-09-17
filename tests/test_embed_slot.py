"""EmbedSlot: the two things that only break against a real client.

Both defects here were invisible to hand-written curl calls and only showed up
when a real RAG client (LangChain) was pointed at the server.

1. **The lone surrogate.** The claim under test is not "the sanitiser strips
   odd characters". It is that the *underlying pipeline rejects the entire
   call* when one input holds a lone UTF-16 surrogate, so a single bad
   character in a 40-chunk batch costs all 40. A fixture cannot prove that —
   only the real pybind11 binding can, so that test reads real data through a
   real exported model. If openvino_genai ever starts accepting surrogates,
   it fails and tells us the sanitiser's justification has expired.

2. **The unbounded batch.** A RAG client sends its whole corpus in one
   request, so embed() must slice it. Passing it straight through measured
   431s and 18.5 GB against 224s and ~5 GB sliced (2026-09-17). Those tests
   use a counting stand-in, because what needs pinning is the slicing and the
   ordering across slice boundaries, not the arithmetic of the model.

Run:

    venv\\Scripts\\python -m pytest tests\\test_embed_slot.py -q
    venv\\Scripts\\python tests\\test_embed_slot.py          # no pytest needed

Needs an embedding model. Set NOLLAMA_EMBED_MODEL, or it looks for the ones
docs/dev/models.md lists as verified under ~/models. Skips cleanly otherwise,
because a missing model is not a failing convention.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nollama import EmbedSlot  # noqa: E402

# A lone high surrogate: what PDF, DOCX and ZIM extraction leaves behind when
# it clips a glyph in half. Valid in a Python str, not encodable to UTF-8.
LONE_SURROGATE = "clipped emoji \ud83d here"

CANDIDATES = [
    "nomic-embed-text-v1.5-ov",
    "all-MiniLM-L6-v2-int8-ov",
    "bge-m3-int8-ov",
]


def _model_dir():
    """Locate an embedding model, or None to skip."""
    env = os.environ.get("NOLLAMA_EMBED_MODEL")
    if env and os.path.isdir(env):
        return env
    root = os.path.expanduser("~/models")
    for name in CANDIDATES:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return None


def test_sanitize_is_pure_string_work():
    """No model needed: the coercions the pipeline never sees."""
    assert EmbedSlot._sanitize("plain") == "plain"
    assert EmbedSlot._sanitize("") == " "          # never a zero-token input
    assert EmbedSlot._sanitize("   ") == " "
    assert EmbedSlot._sanitize(1234) == "1234"     # non-str coerced, not raised
    assert "\ud83d" not in EmbedSlot._sanitize(LONE_SURROGATE)
    # Legitimate non-ASCII must survive untouched — over-eager cleaning here
    # would quietly change the vectors for every non-English document.
    assert EmbedSlot._sanitize("café Ωmega 日本語") == "café Ωmega 日本語"


def test_raw_pipeline_still_rejects_the_whole_batch():
    """[OBSERVED 2026-09-17, openvino-genai 2026.3.1] One lone surrogate makes
    embed_documents raise for every input in the call, not just the bad one.
    If this ever stops failing, _sanitize's reason to exist has changed."""
    model = _model_dir()
    if not model:
        print("SKIP: no embedding model found (set NOLLAMA_EMBED_MODEL)")
        return
    import openvino_genai as ovg
    pipe_cls = ovg.TextEmbeddingPipeline
    cfg = pipe_cls.Config()
    cfg.pooling_type = pipe_cls.PoolingType.MEAN
    cfg.normalize = True
    pipe = pipe_cls(model, "CPU", cfg)

    pipe.embed_documents(["good text"])  # control: the batch shape is fine

    raised = False
    try:
        pipe.embed_documents(["good text", LONE_SURROGATE, "more good text"])
    except Exception:
        raised = True
    assert raised, ("openvino_genai now accepts lone surrogates — re-check "
                    "whether EmbedSlot._sanitize is still needed")


def test_slot_survives_what_the_raw_pipeline_refuses():
    """The same batch, through the slot, returns a vector for every input."""
    model = _model_dir()
    if not model:
        print("SKIP: no embedding model found (set NOLLAMA_EMBED_MODEL)")
        return
    slot = EmbedSlot("CPU", "CPU")
    slot.load(model)
    slot.warmup()

    texts = ["good text", LONE_SURROGATE, "", "more good text"]
    vectors = slot.embed(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == slot.dims for v in vectors)
    # jsonify would emit ints as ints; clients expect floats.
    assert all(isinstance(x, float) for x in vectors[0])
    # Normalized, because that is what an existing index was built with.
    norm = sum(x * x for x in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-3, norm
    # The clean inputs must not have been disturbed by their bad neighbour.
    alone = slot.embed(["good text"])[0]
    cosine = sum(a * b for a, b in zip(vectors[0], alone))
    assert cosine > 0.9999, cosine


class _CountingPipe:
    """Stand-in that records the slice sizes embed() hands the pipeline."""

    def __init__(self, dims=4):
        self.dims = dims
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(len(texts))
        # A distinct, recoverable vector per input, so order can be checked.
        return [[float(len(t))] * self.dims for t in texts]


def test_embed_slices_the_request_and_keeps_order():
    """A RAG client sends its whole corpus in one call; embed() must not pass
    that straight through. Order across slice boundaries is what a vector
    store depends on — it zips these back against its own metadata list."""
    slot = EmbedSlot("CPU", "CPU", batch_size=16)
    slot.pipe = _CountingPipe()
    slot.dims = 4

    texts = ["x" * (i + 1) for i in range(40)]
    vectors = slot.embed(texts)

    assert slot.pipe.calls == [16, 16, 8], slot.pipe.calls
    assert len(vectors) == 40
    # Each vector encodes its input's length, so this catches reordering.
    assert [int(v[0]) for v in vectors] == [i + 1 for i in range(40)]


def test_batch_size_zero_means_one_call():
    """0 is the documented escape hatch: hand the pipeline everything."""
    slot = EmbedSlot("CPU", "CPU", batch_size=0)
    slot.pipe = _CountingPipe()
    vectors = slot.embed(["a", "b", "c"])
    assert slot.pipe.calls == [3], slot.pipe.calls
    assert len(vectors) == 3


def test_empty_input_makes_no_pipeline_call():
    """An empty list must not reach the pipeline, which rejects one."""
    slot = EmbedSlot("CPU", "CPU", batch_size=16)
    slot.pipe = _CountingPipe()
    assert slot.embed([]) == []
    assert slot.pipe.calls == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
