"""The memory preflight's verdict band — the numbers that cried wolf on the B60.

Pure function, no model. Run:

    venv\\Scripts\\python -m pytest tests\\test_preflight.py -q
    venv\\Scripts\\python tests\\test_preflight.py          # no pytest needed
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nollama import _preflight_verdict  # noqa: E402

GIB = 2 ** 30


def test_b60_case_is_tight_not_over():
    """[OBSERVED 2026-09-12] Arc Pro B60: 15.2 GB weights + 6 GB pool, budget
    23.3 GB. The old code warned "will likely NOT work"; the load ran fine."""
    weights, pool, budget = 15.2 * GIB, 6 * GIB, 23.3 * GIB
    need = (weights + pool) * 1.1
    assert need > budget  # the raw comparison still says over
    assert _preflight_verdict(need, budget) == "tight"


def test_equality_fits():
    assert _preflight_verdict(100, 100) == "fits"
    assert _preflight_verdict(99, 100) == "fits"


def test_clearly_over_is_over():
    # a 15.2 GB int4 MoE + 6 GB pool on a stock 16 GB iGPU budget
    assert _preflight_verdict((15.2 + 6) * 1.1 * GIB, 16 * GIB) == "over"


def test_slack_boundary():
    assert _preflight_verdict(105, 100) == "tight"
    assert _preflight_verdict(105.1, 100) == "over"
    assert _preflight_verdict(110, 100, slack=0.10) == "tight"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all passed")
