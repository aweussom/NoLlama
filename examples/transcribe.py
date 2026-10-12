"""Speech to text: a WAV file, or 5 seconds from the microphone.

    pip install openai                    # file mode
    pip install openai sounddevice numpy  # --mic
    python transcribe.py speech.wav [--language en]
    python transcribe.py --mic [--seconds 5] [--language no]

Needs a server started with --whisper-dir (and --whisper-device NPU to keep
the GPU free for chat). Set the language when you know it: without it,
Whisper guesses, and on short clips it guesses wrong.
"""
import argparse
import io
import wave

from openai import OpenAI

p = argparse.ArgumentParser()
p.add_argument("file", nargs="?")
p.add_argument("--mic", action="store_true")
p.add_argument("--seconds", type=float, default=5)
p.add_argument("--language", default="en")
p.add_argument("--base-url", default="http://localhost:8000")
args = p.parse_args()
if not (args.file or args.mic):
    p.error("give a WAV file or --mic")

if args.mic:
    import sounddevice as sd
    print(f"Recording {args.seconds:g} s, speak now...")
    pcm = sd.rec(int(args.seconds * 16000), samplerate=16000, channels=1, dtype="int16")
    sd.wait()
    audio = io.BytesIO()
    with wave.open(audio, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(pcm.tobytes())
    upload = ("mic.wav", audio.getvalue())
else:
    upload = open(args.file, "rb")

client = OpenAI(base_url=args.base_url + "/v1", api_key="unused")
text = client.audio.transcriptions.create(model="whisper", file=upload, language=args.language).text
print(text or "(nothing heard)")
