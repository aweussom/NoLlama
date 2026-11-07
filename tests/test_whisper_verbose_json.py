r"""verbose_json shaping and request validation for /v1/audio/transcriptions (T-057).

The shaping test uses stand-ins for genai's WhisperDecodedResults, so it runs
without a model. The real-model check is the server run in T-057.

    venv\Scripts\python tests\test_whisper_verbose_json.py
    venv\Scripts\python -m pytest tests\test_whisper_verbose_json.py
"""
import io
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nollama  # noqa: E402
from nollama import whisper_verbose_json  # noqa: E402


def _result():
    return NS(texts=[" I found a mango in the backseat"],
              chunks=[NS(start_ts=0.0, end_ts=13.82, text=" I found a mango"),
                      NS(start_ts=13.82, end_ts=18.28, text=" in the backseat")],
              words=[NS(start_ts=0.0, end_ts=7.1, word=" I"),
                     NS(start_ts=10.14, end_ts=10.54, word=" mango")])


def test_segments_and_words_in_openai_shape():
    body = whisper_verbose_json(_result(), 60.0, "en", True, True)
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert body["duration"] == 60.0
    assert body["text"] == "I found a mango in the backseat"
    assert body["segments"][1] == {"id": 1, "start": 13.82, "end": 18.28,
                                   "text": "in the backseat"}
    assert body["words"][1] == {"word": "mango", "start": 10.14, "end": 10.54}


def test_only_the_asked_granularities_appear():
    body = whisper_verbose_json(_result(), 60.0, None, True, False)
    assert "segments" in body and "words" not in body
    assert body["language"] is None
    body = whisper_verbose_json(_result(), 60.0, None, False, True)
    assert "words" in body and "segments" not in body


def test_silence_skip_is_an_empty_transcript_not_an_error():
    body = whisper_verbose_json(None, 4.0, None, True, True)
    assert body["text"] == "" and body["segments"] == [] and body["words"] == []
    assert body["duration"] == 4.0


def _post(data):
    nollama.whisper_slot = NS(status="ready", device_name="CPU")
    client = nollama.app.test_client()
    data = dict(data, file=(io.BytesIO(b"not audio"), "a.wav"))
    return client.post("/v1/audio/transcriptions", data=data,
                       content_type="multipart/form-data")


def test_unsupported_format_and_granularity_are_rejected():
    saved = nollama.whisper_slot
    try:
        assert _post({"response_format": "srt"}).status_code == 400
        assert _post({"response_format": "verbose_json",
                      "timestamp_granularities[]": "char"}).status_code == 400
        assert _post({"timestamp_granularities[]": "word"}).status_code == 400
    finally:
        nollama.whisper_slot = saved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
