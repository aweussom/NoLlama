# T-008 — `benchmark.py` JSON records no provenance

**Effort:** 1 h · **Produces:** a header block in every `bench-results/*.json` ·
**Area:** `benchmark.py`

Three community reporters in one week (#24, #32) and for none of them does the
JSON say which runtime produced the number. It records no OpenVINO/genai
version, no server flags (`--offload-ratio`, `--cache-size-gb`), no driver, no
OS. CLAUDE.md's standing order is to record the driver with every measurement;
the tool that produces the measurements does not.

- [ ] `openvino.__version__` and `openvino_genai.__version__`
- [ ] the server's `/health` payload — model, device, `kv_pool_gb`
- [ ] `platform.platform()`
- [ ] the GPU/NPU driver string, from the same `FULL_DEVICE_NAME` /
      `NPU_DRIVER_VERSION` path `detect_devices` already walks

## Done when

A `bench-results/*.json` from a fresh run is citable in an upstream issue
without asking the reporter a single follow-up question.
