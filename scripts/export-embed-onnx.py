"""Export an embedding model to OpenVINO IR by way of its published ONNX.

Why this exists: some embedding models ship a custom architecture that
optimum-intel refuses. nomic-embed-text-v1.5 is `nomic_bert`, which is not a
registered export target, so `optimum-cli export openvino` fails outright --
and it is the model an Ollama-shaped RAG client asks for by name. Its repo
does publish an official ONNX graph, and OpenVINO converts that directly, so
the route around optimum is a download and two conversions.

The tokenizer is converted separately and written into the same directory,
because TextEmbeddingPipeline loads `openvino_tokenizer.xml` from beside the
model and fails with a bare "Cannot find tokenizer" if it is absent.

Usage:
    venv\\Scripts\\python scripts\\export-embed-onnx.py nomic-ai/nomic-embed-text-v1.5
    venv\\Scripts\\python scripts\\export-embed-onnx.py <repo> --out DIR --onnx onnx/model.onnx

Then serve it:
    python nollama.py --embed-model-dir DIR --embed-name nomic-embed-text:v1.5

Pooling and normalization are NOT baked in here -- NoLlama's EmbedSlot applies
MEAN pooling and L2 normalization at serving time, which is what these models
were trained with and what an index built by Ollama already contains.
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    """CLI: the repo id, and the knobs that differ per model.

    Why --onnx is a flag and not a constant: repos disagree on the path.
    nomic publishes `onnx/model_fp16.onnx`, others `onnx/model.onnx` or
    `model.onnx` at the root, and guessing wrong wastes a multi-hundred-MB
    download before it fails.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", help="HuggingFace repo id, e.g. nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--onnx", default="onnx/model_fp16.onnx",
                   help="Path to the ONNX file inside the repo (default: onnx/model_fp16.onnx)")
    p.add_argument("--out", default=None,
                   help="Output directory (default: ~/models/<repo-name>-ov)")
    p.add_argument("--trust", action="store_true",
                   help="Pass trust_remote_code when loading the tokenizer (nomic needs it)")
    return p.parse_args()


def convert(repo, onnx_path, out_dir, trust):
    """Download the ONNX, convert graph + tokenizer, save both into out_dir.

    Why the tokenizer conversion is guarded separately: it is the step most
    likely to fail on a machine whose `transformers` is older or newer than
    the model expects, and when it does, the IR beside it is still good --
    so the message has to say that the graph survived and only the tokenizer
    needs re-running, rather than implying the whole export is lost.

    In: repo id, in-repo ONNX path, target dir, trust-remote-code flag.
    Out: writes openvino_model.{xml,bin} and openvino_tokenizer.{xml,bin};
    raises with an actionable message if the ONNX path is wrong.
    """
    import openvino as ov
    from huggingface_hub import hf_hub_download

    print(f"  Downloading {onnx_path} from {repo} ...", flush=True)
    try:
        local = hf_hub_download(repo_id=repo, filename=onnx_path)
    except Exception as e:
        raise SystemExit(
            f"ERROR: could not fetch '{onnx_path}' from {repo}: {e}\n"
            f"  Check the repo's file list and pass the real path with --onnx\n"
            f"  (common: onnx/model.onnx, onnx/model_fp16.onnx, model.onnx)."
        )

    print("  Converting the graph to OpenVINO IR ...", flush=True)
    model = ov.convert_model(local)
    out_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, out_dir / "openvino_model.xml")

    print("  Converting the tokenizer ...", flush=True)
    try:
        from openvino_tokenizers import convert_tokenizer
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust)
        ov_tok = convert_tokenizer(hf_tok)
        # convert_tokenizer returns the tokenizer alone, or (tokenizer,
        # detokenizer) when the model has one. An embedding model does not
        # need the detokenizer, but unpacking wrongly would crash here.
        if isinstance(ov_tok, tuple):
            ov_tok = ov_tok[0]
        ov.save_model(ov_tok, out_dir / "openvino_tokenizer.xml")
    except Exception as e:
        raise SystemExit(
            f"ERROR: the graph converted but the tokenizer did not: {e}\n"
            f"  The IR in {out_dir} is fine -- only the tokenizer step needs re-running.\n"
            f"  Try --trust, or a venv whose transformers matches the model's."
        )


def main():
    """Export, then verify by embedding one string through the real pipeline.

    Why it self-tests: a graph that converts cleanly can still be unloadable
    (missing tokenizer, wrong pooling shape), and finding that out here beats
    finding it out as a 500 from a RAG client mid-ingestion.
    """
    args = parse_args()
    name = args.repo.rstrip("/").split("/")[-1]
    out_dir = Path(os.path.expanduser(args.out)) if args.out else \
        Path(os.path.expanduser("~/models")) / f"{name}-ov"

    print(f"\n  Exporting {args.repo}\n  Target: {out_dir}\n", flush=True)
    convert(args.repo, args.onnx, out_dir, args.trust)

    print("  Verifying with TextEmbeddingPipeline on CPU ...", flush=True)
    import openvino_genai as ovg
    pipe_cls = ovg.TextEmbeddingPipeline
    cfg = pipe_cls.Config()
    cfg.pooling_type = pipe_cls.PoolingType.MEAN
    cfg.normalize = True
    vec = pipe_cls(str(out_dir), "CPU", cfg).embed_documents(["hello"])[0]

    print(f"\n[OK] {out_dir}  ({len(vec)} dims)")
    print("\n  Serve it with:")
    print(f'    python nollama.py --embed-model-dir "{out_dir}" '
          f'--embed-name {name.lower()}\n')
    return 0


if __name__ == "__main__":
    sys.exit(main())
