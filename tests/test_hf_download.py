"""scripts/hf_download.py: the anonymous retry after a rejected token.

Runs without network: the hub's snapshot_download and get_token are faked.

    venv\\Scripts\\python -m pytest tests\\test_hf_download.py -q
    venv\\Scripts\\python tests\\test_hf_download.py          # no pytest needed
"""
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location(
    "hf_download", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "hf_download.py"))
hf_download = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf_download)


class _Resp:
    def __init__(self, status):
        self.status_code = status


class _HubError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = _Resp(status)


def _fake(fail_with):
    """A snapshot_download that fails with each queued error, then succeeds."""
    calls = []

    def snapshot_download(**kw):
        calls.append(kw)
        if fail_with:
            raise fail_with.pop(0)
        return "/models/x"
    return snapshot_download, calls


def test_expired_token_retries_anonymously():
    dl, calls = _fake([_HubError(401)])
    assert hf_download.download_with_fallback(dl, lambda: "hf_old", "o/r", "d", None) == "/models/x"
    assert "token" not in calls[0] and calls[1]["token"] is False


def test_no_token_means_no_retry():
    dl, calls = _fake([_HubError(401)])
    try:
        hf_download.download_with_fallback(dl, lambda: None, "o/r", "d", None)
        assert False, "should have raised"
    except _HubError:
        pass
    assert len(calls) == 1


def test_other_errors_are_not_retried():
    for err in (_HubError(404), _HubError(403), OSError("disk full")):
        dl, calls = _fake([err])
        try:
            hf_download.download_with_fallback(dl, lambda: "hf_x", "o/r", "d", None)
            assert False, "should have raised"
        except type(err):
            pass
        assert len(calls) == 1, err


def test_failed_retry_reports_the_original_error():
    first = _HubError(401)
    dl, calls = _fake([first, _HubError(404)])
    try:
        hf_download.download_with_fallback(dl, lambda: "hf_x", "o/r", "d", None)
        assert False, "should have raised"
    except _HubError as e:
        assert e is first
    assert len(calls) == 2


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as e:
                failures += 1
                print("FAIL", name, "-", e)
    sys.exit(1 if failures else 0)
