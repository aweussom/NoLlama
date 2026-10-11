# T-034 — Token usage: the OpenAI endpoint is fixed, two ends are still open

**Effort:** 1 h · **Produces:** real counts on the Ollama surface, and a
yes/no on OpenCode compaction · **Area:** `nollama.py`

`/v1/chat/completions` returns the runtime's own counts on every path as of
2026-09-29. That covers non-streaming, streaming with
`stream_options.include_usage`, tool turns, and LLM and VLM slots, with
image tokens counted. Verified live on the B60 with Qwen3-8B and
Qwen2.5-VL-3B. The mechanism is in `extract_usage` and in
`docs/stream-plumbing.mmd`. What is left:

- [ ] **The Ollama surface: built 2026-10-02, not yet seen live.**
      `/api/chat` and `/api/generate` pass a usage dict through the same
      seams on every path and fill `prompt_eval_count` + `eval_count`
      (`ollama_counts`; an unknown count is omitted, not -1). Covered by
      fake-slot tests in `tests/test_stream_tools.py`. To close: one
      streamed and one non-streamed `/api/chat` against a real slot, and
      check `eval_count` matches the OpenAI endpoint's `completion_tokens`
      for the same prompt at temperature 0
- [ ] **Does OpenCode compact on reported usage?** [INFERRED] With -1 it
      would never see the context filling, and the OPENCODE-PLAN arm-0
      runaway (1.49 M chars sent against a 40k window) fits that picture. To
      confirm, run one OpenCode session against NoLlama before and after
      this fix and check whether `compaction.auto` fires before the
      declared limit. If it does, say so in OPENCODE-PLAN.md
- [ ] **NPU is unprobed.** perf_metrics was checked on CPU and the B60 only.
      Run the probe (`get_num_input_tokens` / `get_num_generated_tokens` on
      a plain `LLMPipeline`) on the 285K or the laptop NPU; `usage_block`
      returns -1 if it reports nothing, so this is a gap, not a crash

## Done when

Both API surfaces return real counts on every slot type, and the OpenCode
question has an answer in OPENCODE-PLAN.md.
