r"""T-049: a client's seed reaches GenerationConfig.rng_seed; no seed changes nothing.

GenAI defaults rng_seed to 0 and NoLlama never set it, so sampling returned
the same tokens on every call and a client varying `seed` got one sample.

    venv\Scripts\python -m pytest tests\test_seed.py -q
    venv\Scripts\python tests\test_seed.py       # no pytest needed
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openvino_genai as ovg  # noqa: E402

from nollama import apply_seed  # noqa: E402


def _gen():
    return ovg.GenerationConfig()


def test_no_seed_keeps_the_default():
    g = _gen()
    default = g.rng_seed
    apply_seed(g, None)
    assert g.rng_seed == default


def test_int_seed_is_set():
    g = _gen()
    apply_seed(g, 1234)
    assert g.rng_seed == 1234


def test_numeric_string_seed_is_set():
    g = _gen()
    apply_seed(g, "42")
    assert g.rng_seed == 42


def test_garbage_seed_is_ignored():
    g = _gen()
    default = g.rng_seed
    apply_seed(g, "bogus")
    assert g.rng_seed == default


def test_bool_is_not_a_seed():
    # json true would otherwise become seed 1.
    g = _gen()
    default = g.rng_seed
    apply_seed(g, True)
    assert g.rng_seed == default


def test_negative_seed_wraps_instead_of_raising():
    g = _gen()
    apply_seed(g, -1)
    assert g.rng_seed == 0xFFFFFFFFFFFFFFFF


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all passed")
