#!/usr/bin/env python3
"""Bonsai-2 vs base-model bench: speed + a small correctness probe set, per arm.

One arm = one OpenAI-compatible (or Ollama-native) server serving one model.
Run it once per arm, then `--compare` the JSONs into a markdown table.

    # Bonsai 2 27B PQ2_0 on the PrismML llama-server (CUDA build):
    python scripts/bonsai-bench.py --url http://127.0.0.1:8081 --model bonsai \\
        --label bonsai2-pq2-cuda --nothink template_kwargs --note "RTX 5090 616.92"

    # Base model in Ollama (native /api/chat so `think:false` and exact counts work):
    python scripts/bonsai-bench.py --url http://127.0.0.1:11434 --model qwen3.8:27b \\
        --transport ollama --label qwen38-q4km-ollama

    # Base model, Intel int4 export, under NoLlama on the B60:
    python scripts/bonsai-bench.py --url http://127.0.0.1:8000 --model auto \\
        --label qwen38-int4ov-b60 --nothink nollama

    python scripts/bonsai-bench.py --compare bench-results/bonsai-*.json

Why this exists next to benchmark.py: that script measures NoLlama's own
workloads across devices. This one holds *the model* constant-in-spirit (a
ternary distillation vs its base) across three stacks that disagree on how
thinking is switched off and on how tokens are counted, and it adds a pass/
fail probe set so a 9x-smaller model can be judged on answers, not only tok/s.
"""

import argparse
import glob
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# Same text the web UI sends; nollama.py matches on the "Reasoning strength:
# minimal" fragment and forecloses the <think> channel via enable_thinking.
NOLLAMA_NO_THINK_PROMPT = ("Respond directly and concisely, with no internal "
                           "reasoning preamble. Reasoning strength: minimal.")


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _post_json(url, body, timeout):
    """Open a POST with a JSON body and return the raw response object.

    Why: both transports stream line-oriented bodies (SSE or NDJSON), so the
    caller reads lines itself; this only centralises headers and the error
    text, which urllib otherwise swallows into a bare HTTPError.

    In: full URL, dict body. Out: an open HTTPResponse; raises RuntimeError
    carrying the server's error body on a non-2xx.
    """
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read()[:400]!r}") from None


def _get_json(url, timeout=10):
    """GET a JSON document, or None on any failure.

    Why: provenance endpoints (/props, /health, /api/version) differ per stack
    and a missing one must not fail the run.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


class Arm:
    """One server+model under test, with the stack's own thinking switch.

    Why: the three stacks disagree on how to turn reasoning off — llama-server
    takes `chat_template_kwargs.enable_thinking`, Ollama takes `think` (only
    on its native endpoint), NoLlama sniffs a system-prompt marker — and on
    how they count tokens. Hiding that here keeps every test identical
    across arms, which is the whole point of the comparison.

    In: url, model id, transport ('openai' | 'ollama'), nothink mechanism
    ('template_kwargs' | 'nollama' | 'ollama' | 'none'). Out: chat() returns a
    uniform result dict; see chat().
    """

    def __init__(self, url, model, transport, nothink, num_ctx, num_gpu=None, extra_opts=None):
        self.url = url.rstrip("/")
        self.model = model
        self.transport = transport
        self.nothink = nothink
        self.num_ctx = num_ctx
        # Ollama only: 0 forces a CPU-resident load for the CPU arms.
        self.num_gpu = num_gpu
        # Ollama only: raw options passthrough, e.g. draft_num_predict=0 to
        # switch off its MTP speculative decoder for a like-for-like decode row.
        self.extra_opts = extra_opts or {}

    def server_info(self):
        """Collect whatever provenance the stack exposes.

        Why: a tok/s number without build, quant and context is not
        reproducible; llama-server's /props names the ftype and build, Ollama's
        /api/version the version, NoLlama's /health the device and model.
        """
        info = {}
        if self.transport == "ollama":
            info["version"] = _get_json(self.url + "/api/version")
            show = None
            try:
                with _post_json(self.url + "/api/show", {"model": self.model}, 20) as r:
                    show = json.loads(r.read().decode())
            except Exception:
                pass
            if show:
                info["details"] = show.get("details")
                mi = show.get("model_info") or {}
                info["model_info"] = {k: v for k, v in mi.items()
                                      if k.endswith(("context_length", "parameter_count",
                                                     "block_count", "file_type"))}
            return info
        props = _get_json(self.url + "/props")
        if props:
            info["llama_server"] = {k: props.get(k) for k in
                                    ("model_ftype", "model_path", "total_slots")}
            info["llama_server"]["n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
            info["llama_server"]["build"] = props.get("build_info")
        health = _get_json(self.url + "/health")
        if health and "devices" in health:
            info["nollama_health"] = health
        models = _get_json(self.url + "/v1/models")
        if models and models.get("data"):
            m = models["data"][0]
            info["v1_model"] = {"id": m.get("id"), "meta": m.get("meta")}
        return info

    def resolve_model(self):
        """Turn `--model auto` into the served model id (NoLlama's GPU slot).

        Why: NoLlama names its model `<dir>@<DEVICE>` and the caller should not
        have to know which device the slot landed on.
        """
        if self.model != "auto":
            return self.model
        health = _get_json(self.url + "/health") or {}
        for dev, d in (health.get("devices") or {}).items():
            if d.get("status") == "ready":
                self.model = f"{d['model']}@{dev.upper()}"
                return self.model
        models = _get_json(self.url + "/v1/models") or {}
        ids = [m["id"] for m in models.get("data", [])]
        if not ids:
            raise SystemExit("could not resolve --model auto: no /health slots and no /v1/models")
        self.model = ids[0]
        return self.model

    def chat(self, messages, think, max_tokens, tools=None, timeout=1800):
        """Run one greedy chat turn and return uniform timings + text.

        Why streaming: TTFT is only observable from the first delta, and on a
        dense 27B at single-digit tok/s the difference between "prefill" and
        "decode" is the difference between two conclusions. The tool probe is
        the exception — tool_calls are read from a non-streamed message
        because the three stacks fragment streamed tool_calls differently.

        In: messages, think (bool), max_tokens, optional tools. Out: dict with
        content, reasoning, prompt_tokens, completion_tokens (server usage when
        the stack reports it, else counted deltas — `token_source` says which),
        ttft, elapsed, decode_s, tool_calls, finish. temperature is pinned to 0
        because Ollama's default is 0.8 and NoLlama's is 0.0 (see BENCHMARKS.md).
        """
        if self.transport == "ollama":
            return self._chat_ollama(messages, think, max_tokens, tools, timeout)
        return self._chat_openai(messages, think, max_tokens, tools, timeout)

    def _chat_openai(self, messages, think, max_tokens, tools, timeout):
        """OpenAI-compatible transport: llama-server and NoLlama.

        In/Out: as chat(). The no-think switch is applied here: a
        `chat_template_kwargs` field for llama-server, a prepended system
        marker for NoLlama, nothing for 'none' (the arm then always thinks).
        """
        msgs = list(messages)
        body = {"model": self.model, "messages": msgs, "temperature": 0,
                "max_tokens": max_tokens, "stream": tools is None,
                "stream_options": {"include_usage": True}}
        if not think:
            if self.nothink == "template_kwargs":
                body["chat_template_kwargs"] = {"enable_thinking": False}
            elif self.nothink == "nollama":
                msgs.insert(0, {"role": "system", "content": NOLLAMA_NO_THINK_PROMPT})
        if tools is not None:
            body["tools"] = tools
            body.pop("stream_options")
        t0 = time.perf_counter()
        out = {"content": "", "reasoning": "", "tool_calls": [], "finish": None,
               "prompt_tokens": None, "completion_tokens": None, "token_source": "deltas"}
        ttft = None
        n_deltas = 0
        with _post_json(self.url + "/v1/chat/completions", body, timeout) as resp:
            if tools is not None:
                obj = json.loads(resp.read().decode())
                ch = obj.get("choices", [{}])[0]
                msg = ch.get("message", {})
                out["content"] = msg.get("content") or ""
                out["reasoning"] = msg.get("reasoning_content") or msg.get("reasoning") or ""
                out["tool_calls"] = msg.get("tool_calls") or []
                out["finish"] = ch.get("finish_reason")
                usage = obj.get("usage") or {}
                ttft = time.perf_counter() - t0
            else:
                usage = {}
                while True:
                    raw = resp.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        obj = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for ch in obj.get("choices") or []:
                        d = ch.get("delta") or {}
                        c = d.get("content") or ""
                        r = d.get("reasoning_content") or d.get("reasoning") or ""
                        if c or r:
                            n_deltas += 1
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                        out["content"] += c
                        out["reasoning"] += r
                        if ch.get("finish_reason"):
                            out["finish"] = ch["finish_reason"]
        elapsed = time.perf_counter() - t0
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(ct, int) and ct > 0:
            out["completion_tokens"], out["token_source"] = ct, "usage"
        else:
            out["completion_tokens"] = n_deltas
        out["prompt_tokens"] = pt if isinstance(pt, int) and pt > 0 else None
        out["ttft"] = ttft if ttft is not None else elapsed
        out["elapsed"] = elapsed
        out["decode_s"] = max(elapsed - out["ttft"], 1e-6)
        return out

    def _chat_ollama(self, messages, think, max_tokens, tools, timeout):
        """Ollama native /api/chat transport (NDJSON).

        Why native and not /v1: `think:false` is only honoured here, and the
        final frame carries exact eval_count / prompt_eval_count plus the
        server's own durations. `num_ctx` is set explicitly because Ollama's
        default context is 4096 and the prefill test is ~4.5k tokens — an
        unset value silently truncates the prompt and the needle disappears.
        """
        body = {"model": self.model, "messages": list(messages), "stream": tools is None,
                "think": bool(think),
                "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens,
                            "num_ctx": self.num_ctx}}
        if self.num_gpu is not None:
            body["options"]["num_gpu"] = self.num_gpu
        body["options"].update(self.extra_opts)
        if tools is not None:
            body["tools"] = tools
        t0 = time.perf_counter()
        out = {"content": "", "reasoning": "", "tool_calls": [], "finish": None,
               "prompt_tokens": None, "completion_tokens": None, "token_source": "usage"}
        ttft = None
        final = {}
        with _post_json(self.url + "/api/chat", body, timeout) as resp:
            while True:
                raw = resp.readline()
                if not raw:
                    break
                try:
                    obj = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message") or {}
                c = msg.get("content") or ""
                r = msg.get("thinking") or ""
                if (c or r) and ttft is None:
                    ttft = time.perf_counter() - t0
                out["content"] += c
                out["reasoning"] += r
                if msg.get("tool_calls"):
                    out["tool_calls"].extend(msg["tool_calls"])
                if obj.get("done"):
                    final = obj
        elapsed = time.perf_counter() - t0
        out["finish"] = final.get("done_reason")
        out["completion_tokens"] = final.get("eval_count")
        out["prompt_tokens"] = final.get("prompt_eval_count")
        out["ttft"] = ttft if ttft is not None else elapsed
        out["elapsed"] = elapsed
        # Server-side durations are the honest decode clock when present.
        if final.get("eval_duration"):
            out["decode_s"] = final["eval_duration"] / 1e9
            out["server_prompt_s"] = (final.get("prompt_eval_duration") or 0) / 1e9
            out["load_s"] = (final.get("load_duration") or 0) / 1e9
        else:
            out["decode_s"] = max(elapsed - out["ttft"], 1e-6)
        return out


# ---------------------------------------------------------------------------
# Speed tests
# ---------------------------------------------------------------------------

def make_prefill_doc(n_items=125, seed=42):
    """Build a deterministic ~4.5k-token document with a retrievable needle.

    Why: agent prompts are prefill-heavy, and PQ2_0's selling point over PTQ1_0
    is prompt processing, so a short-prompt bench misses the axis that
    matters. Synthetic and seeded so every arm sees byte-identical input and
    the answer is checkable ("what is the code word for item N").

    In: item count, seed. Out: (document text, needle item number, code word).
    [OBSERVED 2026-09-18] 320 items tokenized to 11,423 tokens on the Qwen3.8
    tokenizer (~36 per item: the nonsense code words and numbers fragment),
    which overflowed an 8k-context CPU server; 125 items keeps it near 4.5k.
    """
    rng = random.Random(seed)
    syll = ["ka", "ro", "mi", "tu", "ve", "lo", "sa", "ne", "pi", "do", "fa", "gu"]
    lines = []
    words = {}
    for i in range(1, n_items + 1):
        w = "".join(rng.choice(syll) for _ in range(3))
        words[i] = w
        qty = rng.randint(3, 900)
        lines.append(f"Item {i}: the code word is {w} and the quantity on hand is {qty} units, "
                     f"stored in bay {rng.randint(1, 40)} of the {rng.choice(['north','south','east','west'])} hall.")
    needle = min(77, n_items)
    return "\n".join(lines), needle, words[needle]


def prefill_case(run_idx):
    """Fresh document per run so the prompt is never a cache hit.

    Why: llama-server and Ollama both cache the previous prompt's KV; with one
    fixed document the warmup primes it and every timed run reports a
    70,000+ tok/s "prefill" that is a lookup. A different seed per run keeps
    the size and the needle position while changing every code word.

    In: run index (-1 for warmup). Out: (prompt, checker).
    """
    doc, needle, word = make_prefill_doc(seed=1000 + run_idx)
    prompt = (f"Inventory list:\n\n{doc}\n\nWhat is the code word for item {needle}? "
              "Answer with only the word.")
    return prompt, (lambda t, w=word: w in t.lower())


def _static(prompt, checker=None):
    """Wrap a fixed prompt in the per-run factory shape prefill_case uses."""
    return lambda run_idx: (prompt, checker)


SPEED_TESTS = [
    # name, factory(run_idx) -> (prompt, checker|None), max_tokens
    ("hello", _static("Say hello."), 64),
    # Maximally predictable output: the best case for any speculative /
    # multi-token-prediction decoder, so read it as an upper bound.
    ("count100", _static("List the integers from 1 to 100, separated by commas, on a single line.",
                         lambda t: all(str(n) in t for n in (1, 50, 100)) and t.count(",") >= 95), 1024),
    # Free text: what decode looks like when the next token is not guessable.
    ("story300", _static("Write a short story of about 300 words about a lighthouse keeper who finds "
                         "a message in a bottle. Prose only, no title.",
                         lambda t: len(t.split()) >= 150), 600),
    ("prefill4k", prefill_case, 32),
]


# ---------------------------------------------------------------------------
# Correctness probes
# ---------------------------------------------------------------------------

def _num_in(text, value):
    """True when `value` appears as a standalone number in the reply.

    Why: models write "10,063" or "10 063" or "**10063**"; strip separators
    and formatting before comparing so the checker judges arithmetic, not
    typography.
    """
    t = re.sub(r"[,\s*_`]", "", text)
    return re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", t) is not None


def _word(text, *words):
    """True when any of `words` appears case-insensitively as a whole word."""
    return any(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in words)


def _extract_code(text):
    """Pull the python source out of a reply: fenced block first, else from `def`.

    In: reply text. Out: source string, or '' when nothing code-shaped is there.
    """
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    i = text.find("def ")
    return text[i:] if i >= 0 else ""


def _run_code(src, test_src, timeout=15):
    """Execute model code + assertions in a subprocess; True iff it exits 0.

    Why a subprocess: model code can loop forever or recurse into a stack
    overflow, and neither must take the benchmark down. It is still local
    code execution — this harness runs against our own servers only.

    In: model source, assertion source. Out: bool.
    """
    if not src.strip():
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src + "\n\n" + test_src + "\n")
        path = f.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _json_obj(text):
    """Parse the first JSON value in a reply (fenced or bare), or None."""
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    cand = m.group(1) if m else text
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = cand.find(opener), cand.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(cand[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


def _check_json1(t):
    o = _json_obj(t)
    return isinstance(o, dict) and "kari" in str(o.get("name", "")).lower() \
        and str(o.get("age")) in ("34", "34.0") and "trondheim" in str(o.get("city", "")).lower()


def _check_json2(t):
    o = _json_obj(t)
    return o == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def _check_ifeval1(t):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]
    return len(sents) == 3 and "z" not in t.lower()


def _check_ifeval3(t):
    line = [l for l in t.strip().splitlines() if l.strip()]
    if len(line) != 1:
        return False
    items = [x.strip() for x in line[0].split(",")]
    return len(items) == 5 and items == sorted(items) and all(x == x.lower() and x.isalpha() for x in items)


def _check_regex1(t):
    pat = t.strip().strip("`").strip()
    pat = re.sub(r"^regex:?\s*", "", pat, flags=re.I).strip("`").strip()
    if pat.startswith("/") and pat.rstrip("gim").endswith("/"):
        pat = pat[1:pat.rstrip("gim").rfind("/")]
    try:
        rx = re.compile(pat)
    except re.error:
        return False
    return bool(rx.fullmatch("7030")) and not rx.fullmatch("703") and not rx.fullmatch("70300") \
        and not rx.fullmatch("7O30")


def _check_sql1(t):
    u = t.upper()
    return "SUM(" in u.replace(" ", "") and "GROUP BY" in u and "ORDER BY" in u and "DESC" in u


PROBES = [
    # id, category, prompt, checker
    ("math1", "math", "What is 347 * 29? Answer with just the number.", lambda t: _num_in(t, 10063)),
    ("math2", "math", "A train leaves at 09:40 and arrives at 13:15 the same day. How many minutes "
                      "does the trip take? Answer with just the number.", lambda t: _num_in(t, 215)),
    ("math3", "math", "What is 15% of 860? Answer with just the number.", lambda t: _num_in(t, 129)),
    ("math4", "math", "Anna has 3 boxes with 24 apples each. She gives away 17 apples and then buys "
                      "2 more boxes of 24. How many apples does she have? Just the number.",
     lambda t: _num_in(t, 103)),
    ("logic1", "reasoning", "Tom is taller than Anna. Anna is taller than Per. Who is the shortest? "
                            "Answer with one name.", lambda t: _word(t, "Per") and not _word(t, "Tom")),
    ("logic2", "reasoning", "If all bloops are razzies and all razzies are lazzies, are all bloops "
                            "definitely lazzies? Answer yes or no.", lambda t: _word(t, "yes")),
    ("know1", "knowledge", "Which chemical element has atomic number 26? One word.", lambda t: _word(t, "iron")),
    ("know2", "knowledge", "In what year did the Berlin Wall fall? Just the year.", lambda t: _num_in(t, 1989)),
    ("know3", "knowledge", "What is the capital of Australia? One word.",
     lambda t: _word(t, "Canberra") and not _word(t, "Sydney")),
    ("date1", "reasoning", "What day of the week was 2026-09-18? One word.", lambda t: _word(t, "Friday")),
    ("string1", "instruction", "Reverse the string 'benchmark'. Output only the reversed string.",
     lambda t: "kramhcneb" in t.lower()),
    ("ifeval1", "instruction", "Write exactly three sentences about Bergen. No sentence may contain "
                               "the letter z. Output only the sentences.", _check_ifeval1),
    ("ifeval2", "instruction", "Answer in ALL CAPS and nothing else: what is the capital of Norway?",
     lambda t: "OSLO" in t and t.strip() == t.strip().upper()),
    ("ifeval3", "instruction", "List five fruits on a single line, comma-separated, alphabetical order, "
                               "all lowercase, nothing else.", _check_ifeval3),
    ("norsk1", "language", "Translate to English: 'Toget til Oslo er forsinket med tjue minutter på grunn "
                           "av signalfeil.' Output only the translation.",
     lambda t: _word(t, "twenty", "20") and _word(t, "delayed", "late", "delay") and _word(t, "signal", "signalling", "signaling")),
    ("json1", "format", "Extract to JSON with keys name, age, city from: 'Kari Nordmann, 34, lives in "
                        "Trondheim.' Output only the JSON object.", _check_json1),
    ("json2", "format", "Return a JSON array of all prime numbers below 30, ascending. Output only the JSON.",
     _check_json2),
    ("code1", "coding", "Write a Python function is_palindrome(s) that returns True when s reads the same "
                        "forwards and backwards, ignoring case and any non-alphanumeric characters. "
                        "Output only the code in a ```python block.",
     lambda t: _run_code(_extract_code(t),
                         "assert is_palindrome('A man, a plan, a canal: Panama')\n"
                         "assert not is_palindrome('benchmark')\nassert is_palindrome('')\n"
                         "assert is_palindrome('No lemon, no melon!')")),
    ("code2", "coding", "Write a Python function fizzbuzz(n) returning a list of strings for 1..n: "
                        "'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, "
                        "otherwise the number as a string. Output only the code in a ```python block.",
     lambda t: _run_code(_extract_code(t),
                         "r = fizzbuzz(15)\nassert len(r) == 15\nassert r[0] == '1'\nassert r[2] == 'Fizz'\n"
                         "assert r[4] == 'Buzz'\nassert r[14] == 'FizzBuzz'\nassert r[13] == '14'")),
    ("code3", "coding", "Write a Python function top_k_words(text, k) that returns the k most common "
                        "words (lowercased, split on whitespace, punctuation stripped) as a list of "
                        "(word, count) tuples, most common first, ties broken alphabetically. "
                        "Output only the code in a ```python block.",
     lambda t: _run_code(_extract_code(t),
                         "r = top_k_words('the cat and the dog and the bird. The end!', 2)\n"
                         "assert r == [('the', 4), ('and', 2)], r\n"
                         "r = top_k_words('b a b a c', 3)\nassert r == [('a', 2), ('b', 2), ('c', 1)], r")),
    ("regex1", "coding", "Give one regular expression that matches a Norwegian postal code (exactly four "
                         "digits) as the whole string. Output only the regex, nothing else.", _check_regex1),
    ("sql1", "coding", "Table orders(id, customer_id, amount). Write one SQL query returning the total "
                       "amount per customer_id, largest total first. Output only the SQL.", _check_sql1),
]

WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "City name"}},
                       "required": ["city"]},
    },
}]


def check_tool_call(result):
    """True when the reply is a get_weather call naming Trondheim.

    Why: the agentic-and-tool-calling row is where PrismML reports Bonsai's
    largest drop (77.6 vs 79.7), and a distilled model that stops emitting the
    tool-call grammar is useless for the coding-agent use case regardless of
    its MMLU. Accepts Ollama's `{function:{name,arguments}}` and OpenAI's
    string-encoded arguments alike.
    """
    for tc in result.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if fn.get("name") != "get_weather":
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        if "trondheim" in json.dumps(args).lower():
            return True
    return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _median(xs):
    return statistics.median(xs) if xs else None


def run_speed(arm, runs, log, wanted=None):
    """Run the speed tests `runs` times each (after one warmup) — no-think.

    Why no-think: a thinking arm's token count is a property of its greedy
    trajectory, not of the prompt (BENCHMARKS.md), so decode tok/s is only
    comparable when the channel is closed on every arm.

    In: `wanted` — a set of test names, or None for all; lets a slow device
    skip the 4.4k-token prefill test rather than skip the whole speed pass.
    Out: {test: {runs: [...], median_decode_tps, median_ttft, median_prefill_tps, correct}}.
    """
    out = {}
    for name, factory, max_tokens in SPEED_TESTS:
        if wanted is not None and name not in wanted:
            continue
        log(f"  [{name}] warmup...")
        prompt, checker = factory(-1)
        try:
            arm.chat([{"role": "user", "content": prompt}], think=False, max_tokens=max_tokens)
        except Exception as e:
            log(f"    warmup FAILED: {e}")
            out[name] = {"error": str(e)}
            continue
        rows = []
        for i in range(runs):
            prompt, checker = factory(i)
            try:
                r = arm.chat([{"role": "user", "content": prompt}], think=False, max_tokens=max_tokens)
            except Exception as e:
                log(f"    run {i+1} FAILED: {e}")
                continue
            ct = r["completion_tokens"] or 0
            row = {"completion_tokens": ct, "prompt_tokens": r["prompt_tokens"],
                   "ttft": r["ttft"], "elapsed": r["elapsed"], "decode_s": r["decode_s"],
                   "decode_tps": ct / r["decode_s"] if ct else None,
                   "prefill_tps": (r["prompt_tokens"] / r["ttft"]) if r["prompt_tokens"] and r["ttft"] > 0 else None,
                   "token_source": r["token_source"], "finish": r["finish"],
                   "correct": checker(r["content"]) if checker else None,
                   "reasoning_chars": len(r["reasoning"])}
            if "server_prompt_s" in r:
                row["server_prompt_s"] = r["server_prompt_s"]
            rows.append(row)
            log(f"    run {i+1}/{runs}: {ct} tok, ttft={r['ttft']:.2f}s, "
                f"decode={row['decode_tps'] or 0:.1f} tok/s"
                + (f", prefill={row['prefill_tps']:.0f} tok/s" if row["prefill_tps"] else "")
                + (f", correct={row['correct']}" if checker else "")
                + (f"  [LEAKED {row['reasoning_chars']} reasoning chars]" if row["reasoning_chars"] else ""))
        out[name] = {
            "runs": rows,
            "median_decode_tps": _median([x["decode_tps"] for x in rows if x["decode_tps"]]),
            "median_ttft": _median([x["ttft"] for x in rows]),
            "median_prefill_tps": _median([x["prefill_tps"] for x in rows if x["prefill_tps"]]),
            "median_prompt_tokens": _median([x["prompt_tokens"] for x in rows if x["prompt_tokens"]]),
            "median_tokens": _median([x["completion_tokens"] for x in rows]),
            "correct": all(x["correct"] for x in rows) if checker and rows else None,
        }
    return out


def run_probes(arm, think, max_tokens, log):
    """Run every probe once, greedy, and score it.

    Why once: temperature 0 makes a rerun mostly a replay; the budget is
    better spent on the thinking arm, which can cost 20x the tokens per probe.

    In: arm, think flag, per-probe max_tokens. Out: {probe_id: {...}} plus
    a 'tool1' entry from the tool-calling probe.
    """
    out = {}
    for pid, cat, prompt, checker in PROBES:
        try:
            r = arm.chat([{"role": "user", "content": prompt}], think=think, max_tokens=max_tokens)
            ok = bool(checker(r["content"]))
            err = None
        except Exception as e:
            r = {"content": "", "reasoning": "", "completion_tokens": 0, "elapsed": 0, "finish": "error"}
            ok, err = False, str(e)
        out[pid] = {"category": cat, "pass": ok, "completion_tokens": r["completion_tokens"],
                    "reasoning_chars": len(r["reasoning"]), "elapsed": r["elapsed"],
                    "finish": r["finish"], "answer": r["content"][:300], "error": err}
        log(f"    {pid:<9} {cat:<12} {'PASS' if ok else 'FAIL':<4} {r['completion_tokens'] or 0:>5} tok "
            f"{r['elapsed']:>6.1f}s  {r['content'][:60]!r}")
    try:
        r = arm.chat([{"role": "user", "content": "What is the weather in Trondheim right now?"}],
                     think=think, max_tokens=max_tokens, tools=WEATHER_TOOL)
        ok, err = check_tool_call(r), None
    except Exception as e:
        r = {"content": "", "reasoning": "", "completion_tokens": 0, "elapsed": 0, "finish": "error", "tool_calls": []}
        ok, err = False, str(e)
    out["tool1"] = {"category": "tools", "pass": ok, "completion_tokens": r["completion_tokens"],
                    "reasoning_chars": len(r["reasoning"]), "elapsed": r["elapsed"], "finish": r["finish"],
                    "answer": json.dumps(r.get("tool_calls"))[:300] or r["content"][:300], "error": err}
    log(f"    {'tool1':<9} {'tools':<12} {'PASS' if ok else 'FAIL':<4} {r['completion_tokens'] or 0:>5} tok "
        f"{r['elapsed']:>6.1f}s  {out['tool1']['answer'][:60]!r}")
    return out


def probe_summary(probes):
    """Collapse probe rows into pass counts, per category and total."""
    cats = {}
    for pid, p in probes.items():
        c = cats.setdefault(p["category"], [0, 0])
        c[1] += 1
        c[0] += 1 if p["pass"] else 0
    total = sum(1 for p in probes.values() if p["pass"])
    toks = [p["completion_tokens"] or 0 for p in probes.values()]
    return {"passed": total, "of": len(probes), "by_category": cats,
            "median_tokens": _median(toks), "total_seconds": sum(p["elapsed"] for p in probes.values())}


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def compare(paths):
    """Print a markdown table across saved arm JSONs.

    Why: the deliverable is one table in BENCHMARKS.md, and hand-copying
    twelve numbers from three JSONs is how transcription errors get into
    published measurements.
    """
    rows = []
    for p in sorted(set(sum((glob.glob(x) for x in paths), []))):
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        print("no result files matched")
        return

    # ASCII only: a Windows console defaulting to cp1252 raised UnicodeEncodeError
    # on a check mark here, after the JSON was saved but before the table showed.
    def f(x, fmt):
        return fmt.format(x) if isinstance(x, (int, float)) else "-"

    print("| Arm | Decode tok/s, free text | Decode tok/s, count 1-100 | TTFT short | "
          "Prefill tok/s (prompt tokens) | Needle | Probes no-think | Probes think | Think tokens/probe (median) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        sp = r.get("speed", {})
        c100, hello, pre = sp.get("count100", {}), sp.get("hello", {}), sp.get("prefill4k", {})
        story = sp.get("story300", {})
        pn, pt = r.get("probes_nothink_summary"), r.get("probes_think_summary")
        print(f"| {r['label']} | {f(story.get('median_decode_tps'), '{:.1f}')} | "
              f"{f(c100.get('median_decode_tps'), '{:.1f}')} | "
              f"{f(hello.get('median_ttft'), '{:.2f} s')} | {f(pre.get('median_prefill_tps'), '{:.0f}')}"
              f"{(' (' + f(pre.get('median_prompt_tokens'), '{:.0f}') + ')') if pre.get('median_prompt_tokens') else ''} | "
              f"{'ok' if pre.get('correct') else ('-' if pre.get('correct') is None else 'MISS')} | "
              f"{(str(pn['passed']) + '/' + str(pn['of'])) if pn else '-'} | "
              f"{(str(pt['passed']) + '/' + str(pt['of'])) if pt else '-'} | "
              f"{f(pt['median_tokens'], '{:.0f}') if pt else '-'} |")
    print()
    for r in rows:
        print(f"- **{r['label']}**: {r.get('note') or ''} -- {r['url']} model `{r['model']}`, "
              f"{r['timestamp']}")
    # Per-probe grid: which probes fail where.
    ids = [pid for pid, *_ in PROBES] + ["tool1"]
    for key, title in (("probes_nothink", "no-think"), ("probes_think", "think")):
        if not any(r.get(key) for r in rows):
            continue
        print(f"\nPer-probe ({title}):\n")
        print("| probe | " + " | ".join(r["label"] for r in rows) + " |")
        print("|---|" + "---|" * len(rows))
        for pid in ids:
            cells = []
            for r in rows:
                p = (r.get(key) or {}).get(pid)
                cells.append("-" if p is None else ("ok" if p["pass"] else "FAIL"))
            print(f"| {pid} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="auto", help="model id; 'auto' resolves NoLlama's ready slot")
    ap.add_argument("--transport", choices=["openai", "ollama"], default="openai",
                    help="ollama = native /api/chat (needed for think:false and exact token counts)")
    ap.add_argument("--nothink", choices=["template_kwargs", "nollama", "ollama", "none"], default=None,
                    help="how to switch reasoning off; default: template_kwargs (openai) / ollama (ollama)")
    ap.add_argument("--label", required=False, help="arm name in the output file and table")
    ap.add_argument("--note", default="", help="provenance: GPU, driver, build, quant, context")
    ap.add_argument("--runs", type=int, default=3, help="repeats per speed test (default 3)")
    ap.add_argument("--think", choices=["off", "on", "both"], default="both",
                    help="which probe passes to run (default both)")
    ap.add_argument("--think-max-tokens", type=int, default=4096)
    ap.add_argument("--nothink-max-tokens", type=int, default=512)
    ap.add_argument("--num-ctx", type=int, default=16384, help="Ollama num_ctx (default 16384)")
    ap.add_argument("--ollama-num-gpu", type=int, default=None,
                    help="Ollama options.num_gpu; 0 = CPU-only load (needs an idle Ollama or it reuses the GPU copy)")
    ap.add_argument("--ollama-opt", action="append", default=[], metavar="KEY=VALUE",
                    help="extra Ollama option (repeatable), e.g. draft_num_predict=0; ints/floats parsed")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--tests", default=None, metavar="a,b,c",
                    help="subset of speed tests: hello,count100,story300,prefill4k "
                         "(e.g. drop prefill4k on a device that prefills at 3 tok/s — "
                         "the 4.4k-token document took 22 minutes per run on a Zen 3 CPU)")
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--output-dir", default="bench-results")
    ap.add_argument("--compare", nargs="+", metavar="JSON", help="print a markdown table for these result files")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return
    if not args.label:
        ap.error("--label is required for a run")
    if args.nothink is None:
        args.nothink = "ollama" if args.transport == "ollama" else "template_kwargs"

    extra = {}
    for kv in args.ollama_opt:
        k, _, v = kv.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        extra[k] = v
    arm = Arm(args.url, args.model, args.transport, args.nothink, args.num_ctx, args.ollama_num_gpu, extra)
    model = arm.resolve_model()
    info = arm.server_info()
    print(f"== {args.label}: {args.url} model={model} transport={args.transport} nothink={args.nothink}")
    print(f"   server: {json.dumps(info)[:400]}")
    log = lambda s: print(s, flush=True)

    result = {"label": args.label, "url": args.url, "model": model, "transport": args.transport,
              "nothink": args.nothink, "note": args.note, "runs": args.runs,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "server_info": info}

    if not args.skip_speed:
        log("\n-- speed (no-think, greedy) --")
        wanted = set(args.tests.split(",")) if args.tests else None
        result["speed"] = run_speed(arm, args.runs, log, wanted)

    if not args.skip_probes and args.think in ("off", "both"):
        log(f"\n-- probes, no-think, max_tokens={args.nothink_max_tokens} --")
        result["probes_nothink"] = run_probes(arm, False, args.nothink_max_tokens, log)
        result["probes_nothink_summary"] = probe_summary(result["probes_nothink"])
        s = result["probes_nothink_summary"]
        log(f"   => {s['passed']}/{s['of']} passed, {s['total_seconds']:.0f}s")

    if not args.skip_probes and args.think in ("on", "both"):
        log(f"\n-- probes, thinking, max_tokens={args.think_max_tokens} --")
        result["probes_think"] = run_probes(arm, True, args.think_max_tokens, log)
        result["probes_think_summary"] = probe_summary(result["probes_think"])
        s = result["probes_think_summary"]
        log(f"   => {s['passed']}/{s['of']} passed, median {s['median_tokens']:.0f} tok/probe, "
            f"{s['total_seconds']:.0f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", args.label)
    path = os.path.join(args.output_dir, f"bonsai-{safe}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    log(f"\nSaved: {path}")
    compare([path])


if __name__ == "__main__":
    main()
