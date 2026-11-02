r"""Does an int4 MoE expert weight stay compressed when OpenVINO compiles it?

    venv\Scripts\python scripts\moe-fold-probe.py A CPU     # batched 3D expert block
    venv\Scripts\python scripts\moe-fold-probe.py C CPU     # same bytes as one 2D weight

Why: Nemotron 3.5 Lightning's int4 IR from optimum-intel#2085 (15.9 GB) took
LLMPipeline past 100 GB during load on a 63 GB box, CPU and iGPU, before any
token [OBSERVED 2026-11-02, 285K, OpenVINO 2026.4.1]. Its non-gated experts
miss the MoE fusion, so each projection stays one u4 constant of shape
[128, 1856, 21, 128] feeding a batched MatMul. This builds one such block in
plain OpenVINO, no model needed, and measures what compile does to it.

Variants: A = u4 -> f16 -> zero point -> scale -> reshape [128,1856,2688] ->
Convert f32 -> batched MatMul (Lightning's exact chain). B = A without the
f32 Convert, f16 MatMul. C = the same bytes as one 2D weight [128*1856, 2688].
Compressed the block is 0.32 GB; folded to f32 it is 2.55 GB.

Out: one line, process private memory after building and after compile.
On CPU, 2026-11-02: A +2.92 GB, B +2.92 GB, C +0.38 GB [OBSERVED, laptop
258V, 2026.4.1]. On GPU all three read ~+0.85 GB of host memory, so the
synthetic graph does not show what the real iGPU load does.
"""
import gc
import sys

import numpy as np
import openvino as ov
import openvino.opset13 as op
import psutil

E, O, G, GS = 128, 1856, 21, 128     # Lightning: 128 experts, moe_intermediate 1856, hidden 2688 = 21 x 128
I = G * GS


def build(variant):
    """Build the decompression + MatMul graph for one variant.

    Why: the three variants separate the two suspects, the trailing f32
    Convert (A vs B) and the batched 3D weight itself (A vs C).

    In: "A", "B" or "C". Out: an ov.Model with one dynamic-length input.
    Weight values are uninitialised; only shapes and types matter here.
    """
    lead = [E * O] if variant == "C" else [E, O]
    w = op.constant(ov.Tensor(ov.Type.u4, ov.Shape(lead + [G, GS])))
    zp = op.convert(op.constant(ov.Tensor(ov.Type.u4, ov.Shape(lead + [G, 1]))), ov.Type.f16)
    sc = op.constant(np.full(lead + [G, 1], 0.01, np.float16))
    d = op.multiply(op.subtract(op.convert(w, ov.Type.f16), zp), sc)
    d = op.reshape(d, op.constant(np.array(lead + [I], np.int64)), False)
    if variant == "B":
        x = op.parameter([E, -1, I], ov.Type.f16)
        mm = op.matmul(x, d, False, True)
    elif variant == "C":
        x = op.parameter([-1, I], ov.Type.f32)
        mm = op.matmul(x, op.convert(d, ov.Type.f32), False, True)
    else:
        x = op.parameter([E, -1, I], ov.Type.f32)
        mm = op.matmul(x, op.convert(d, ov.Type.f32), False, True)
    return ov.Model([mm], [x], variant)


def main():
    """Build, compile, print the private-memory growth of each step."""
    variant, device = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "CPU")
    p = psutil.Process()
    gc.collect()
    base = p.memory_info().private / 1e9
    m = build(variant)
    built = p.memory_info().private / 1e9
    compiled = ov.Core().compile_model(m, device)   # keep the reference, or its memory is gone before we read it
    comp = p.memory_info().private / 1e9
    del compiled
    print(f"{variant} {device}: built +{built - base:.2f} GB, after compile +{comp - base:.2f} GB")


if __name__ == "__main__":
    main()
