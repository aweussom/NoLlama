# T-055 — LFM3: can it be exported, and where does it run

**Effort:** 1 d · **Produces:** a verdict per LFM3 model and device in
`docs/dev/models.md`, installer entries for what passes · **Area:**
`models.json`, export, N-007 watch

Liquid AI released LFM3 on 2026-10-27 [DOCUMENTED, vendor and press]: six
open-weight models on one 1.7B backbone. LFM3-1.7B (text, 256K context),
LFM3-VL-2B (images and video), LFM3-Audio-2B (18 languages, tool calls),
LFM3-Omni-2B (image + audio in, speech out), and two "decision" models.
Liquid's benchmark claims (1.7B ahead of Qwen3.5-9B) are self-reported.

What we know about the export path, 2026-10-29:

- `LiquidAI/LFM3-1.7B` config: `Lfm3ForCausalLM`, `auto_map` to
  `modeling_lfm3` (remote code), `layer_types` mixes `conv`,
  `full_attention` and `sliding_attention` [OBSERVED 2026-10-29, config.json].
  LFM2 had no sliding layers.
- No OpenVINO export on Hugging Face yet, from Intel or anyone else; GGUF,
  MLX, LiteRT and CoreML exist [OBSERVED 2026-10-29, HF search].
- The sliding layers may need what genai#4734 asks for (sliding-window KV
  cache type) [INFERRED]; check before blaming the export.

LFM2 history matters here: LFM2 int4-cw was NPU-only and was word salad on
NPU 4 until OpenVINO 2026.4.1 (openvino#38100). Test NPU 3 and NPU 4 both.

## Done when

- [ ] optimum-intel can export `lfm3` (try release and git main); if not,
      file it and put it on the N-007 watch.
- [ ] LFM3-1.7B bare genai, then server, on CPU, iGPU, B60, NPU 3, NPU 4.
- [ ] LFM3-Audio-2B: is it a Whisper alternative for T-054 (Norwegian WER
      against Whisper small)?
- [ ] LFM3-VL-2B as a small vision entry, if it exports.
