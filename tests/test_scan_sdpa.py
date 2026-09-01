"""Regression test for the prefix-caching column in ``--scan``.

Guards ``_count_sdpa_ops``, which tells a user whether an IR can take the
continuous-batching backend *before* they spend a download. The failure it
detects is real and shipped: Intel's ``OpenVINO/gemma-4-E4B-it-int8-ov``
traced attention decomposed and silently gets no prefix caching
(openvino.genai#4343).

Reads the real IRs on this machine rather than fixtures, so it fails if the
counting logic drifts or an export convention changes under us. The one
synthetic case is the zero branch: no local model exhibits it, and the model
that does is a 7.8 GB download.

    venv\Scripts\python tests\test_scan_sdpa.py
    venv\Scripts\python -m pytest tests\test_scan_sdpa.py   # where pytest exists
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nollama import _count_sdpa_ops  # noqa: E402  (needs the path above)

MODELS = os.path.expanduser("~/models")


def _local(name):
    path = os.path.join(MODELS, name)
    return path if os.path.isdir(path) else None


def test_dense_model_has_one_sdpa_per_layer():
    # Qwen3 is plain full attention throughout: ops == num_hidden_layers.
    path = _local("Qwen3-1.7B-int4-ov")
    if not path:
        print("  SKIP dense: Qwen3-1.7B-int4-ov not present")
        return
    with open(os.path.join(path, "config.json")) as f:
        layers = json.load(f)["num_hidden_layers"]
    assert _count_sdpa_ops(path) == layers


def test_hybrid_model_counts_only_attention_layers():
    # The reason the predicate is "> 0" and not "== layers": Qwen3.5
    # interleaves linear attention, which emits no SDPA node. 32 layers,
    # full_attention_interval 4 -> 8 ops. A one-per-layer check would
    # wrongly condemn this healthy export.
    path = _local("Qwen3.5-4B-int4-ov")
    if not path:
        print("  SKIP hybrid: Qwen3.5-4B-int4-ov not present")
        return
    assert _count_sdpa_ops(path) == 8


def test_sliding_attention_still_fuses():
    # Muse Glimmer is 39 sliding_attention + 13 full_attention. Sliding
    # attention is still attention and fuses, so the count is all 52 --
    # which is why "expected = full_attention layers" was rejected as a
    # warning heuristic (it false-alarmed on this model).
    path = _local("Muse-Glimmer-30B-int4-ov")
    if not path:
        print("  SKIP sliding: Muse-Glimmer-30B-int4-ov not present")
        return
    assert _count_sdpa_ops(path) == 52


def test_vlm_reads_the_language_model_not_the_vision_tower():
    # A VLM has no openvino_model.xml; the count must come from
    # openvino_language_model.xml. E4B's defect is invisible if you read the
    # vision tower instead (0 in the language model, 32 in the tower).
    path = _local("Qwen2.5-VL-3B-Instruct-int8-ov")
    if not path:
        print("  SKIP vlm: Qwen2.5-VL-3B-Instruct-int8-ov not present")
        return
    assert not os.path.isfile(os.path.join(path, "openvino_model.xml"))
    assert _count_sdpa_ops(path) == 36


def test_decomposed_attention_counts_zero_not_none(tmp_path=None):
    # The gemma-4-E4B shape. Zero and None mean different things: zero is a
    # verdict ("cannot cache"), None is "could not read". Conflating them
    # would either hide the defect or invent it.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "openvino_model.xml"), "w") as f:
            f.write('<net name="m"><layers>'
                    '<layer id="0" type="MatMul"/>'
                    '<layer id="1" type="SoftMax"/>'
                    '</layers></net>')
        assert _count_sdpa_ops(d) == 0


def test_missing_ir_is_none():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert _count_sdpa_ops(d) is None


def test_match_split_across_a_read_boundary_is_counted_once():
    # The reader streams in 1 MiB chunks and carries needle-1 bytes across.
    # Pad so a node lands on the seam; it must be seen exactly once.
    import tempfile
    needle = '<layer id="7" type="ScaledDotProductAttention"/>'
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "openvino_model.xml"), "w") as f:
            f.write("<net>" + " " * ((1 << 20) - 20) + needle + "</net>")
        assert _count_sdpa_ops(d) == 1


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL {name}: {e}")
    print("\nall passed" if not failures else f"\n{failures} failed")
    sys.exit(1 if failures else 0)
