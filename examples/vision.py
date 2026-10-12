"""Ask a question about an image. Needs a vision model on the GPU.

    pip install openai
    python vision.py photo.jpg ["How many cars are in this picture?"]

The image travels as base64 in a data URL, which works from any machine.
Small vision models (3B) want one plain question; put the logic in your
code, not in the prompt.
"""
import argparse
import base64
import mimetypes

from openai import OpenAI

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("question", nargs="?", default="Describe this image in one sentence.")
p.add_argument("--base-url", default="http://localhost:8000")
args = p.parse_args()

mime = mimetypes.guess_type(args.image)[0] or "image/jpeg"
with open(args.image, "rb") as f:
    data_url = f"data:{mime};base64," + base64.b64encode(f.read()).decode()

client = OpenAI(base_url=args.base_url + "/v1", api_key="unused")
reply = client.chat.completions.create(model="local", messages=[{"role": "user", "content": [
    {"type": "text", "text": args.question},
    {"type": "image_url", "image_url": {"url": data_url}},
]}])
print(reply.choices[0].message.content.strip())
