#!/usr/bin/env python3
"""
quantize-flux-4bit.py — real fp16 -> 4-bit self-quantizer for FLUX.1-schnell.

Produces a COMPLETE diffusers multi-folder bundle that
`FluxModelCore.loadWeights(from:)` reads on a memory-constrained (<=24 GB)
Apple Silicon machine. The memory-dominant components (transformer and the
T5-XXL text_encoder_2) are 4-bit-quantized with MLX; the small vae and CLIP
text_encoder are left fp16. Each quantized component carries a per-component
`config.json` with `{"quantization": {"group_size": 64, "bits": 4}}` — the
block `FluxModelCore.quantizationConfig(in:)` reads to take the pre-quantized
branch and skip the in-memory quantize pass.

Why this layout (GitHub issue #39)
----------------------------------
fp16 FLUX.1-schnell is ~33.7 GB resident, so it cannot load on 24 GB. A
complete pre-quantized 4-bit bundle (~6-7 GB resident) can. The popular
single-file `argmaxinc/mlx-FLUX.1-schnell-4bit-quantized` bundle is INCOMPLETE
for this loader (no T5, no tokenizers), so we assemble the full layout instead.

24 GB RAM constraint
--------------------
Quantizing the ~24 GB fp16 transformer in one shot OOMs. This script works
COMPONENT-BY-COMPONENT and, within a component, STREAMS one tensor at a time:
each fp16 shard is opened lazily (safetensors zero-copy / mmap), one tensor is
read, quantized, the fp16 copy dropped, and the 4-bit result written out before
moving to the next tensor. Peak resident stays a small multiple of the single
largest tensor (a few hundred MB), not the whole component.

Layout produced (matches FluxBundleLayout.validate == .complete)
----------------------------------------------------------------
  <out>/
    model_index.json                       (copied)
    transformer/  *.safetensors + config.json {"quantization":{...}}   4-bit
    text_encoder_2/ *.safetensors + config.json {"quantization":{...}} 4-bit (T5)
    vae/          diffusion_pytorch_model.safetensors                  fp16
    text_encoder/ model.safetensors                                    fp16 (CLIP)
    tokenizer/    ...                                                   (copied)
    tokenizer_2/  ...                                                   (copied)

Usage
-----
  # download + quantize FLUX.1-schnell into ./flux-schnell-4bit:
  python3 scripts/quantize-flux-4bit.py --out ./flux-schnell-4bit

  # use an already-downloaded fp16 diffusers snapshot:
  python3 scripts/quantize-flux-4bit.py --src /path/to/fp16 --out ./flux-4bit

  # smoke-test the streaming quantizer on the smallest component only (vae or
  # text_encoder), no 34 GB download required:
  python3 scripts/quantize-flux-4bit.py --src /path/to/fp16 --out ./out \\
      --only text_encoder

Requires: Apple Silicon + `pip install mlx safetensors huggingface_hub`.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

GROUP_SIZE = 64
BITS = 4

# Components we 4-bit-quantize (memory-dominant) vs. copy as fp16.
QUANTIZED_COMPONENTS = ["transformer", "text_encoder_2"]
FP16_COMPONENTS = ["vae", "text_encoder"]
COPY_FOLDERS = ["tokenizer", "tokenizer_2"]
COPY_FILES = ["model_index.json"]


def rss_gb() -> float:
    """Peak resident set size of this process in GB (macOS reports bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def is_quantizable(key: str, shape) -> bool:
    """
    A tensor is 4-bit-quantizable iff it is the 2-D `.weight` of a Linear or an
    Embedding whose last dimension is divisible by the group size. This mirrors
    MLX's `nn.quantize` default filter (Linear/Embedding weights only) and, on
    the load side, `FluxModelCore.applyPreQuantization`, which only converts a
    layer when a matching `<path>.scales` tensor is present.
    """
    if not key.endswith(".weight"):
        return False
    if len(shape) != 2:
        return False
    # mx.quantize requires the last (input) dim divisible by group_size.
    return shape[-1] % GROUP_SIZE == 0


def shard_files(component_dir: Path) -> list[Path]:
    files = sorted(component_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no *.safetensors under {component_dir}")
    return files


def quantize_component(src_dir: Path, dst_dir: Path) -> None:
    """
    Stream-quantize one component shard-by-shard, tensor-by-tensor. Peak extra
    memory is ~the largest single tensor plus its 4-bit outputs, never the whole
    component. Writes one output .safetensors per input shard plus config.json.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_quant = 0
    n_fp16 = 0
    for shard in shard_files(src_dir):
        out_tensors: dict[str, mx.array] = {}
        # safe_open mmaps the shard; reading a tensor materializes just that one.
        with safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                arr = mx.array(f.get_tensor(key))
                shape = arr.shape
                if is_quantizable(key, shape):
                    w_q, scales, biases = mx.quantize(arr, group_size=GROUP_SIZE, bits=BITS)
                    base = key[: -len(".weight")]
                    out_tensors[f"{base}.weight"] = w_q
                    out_tensors[f"{base}.scales"] = scales
                    out_tensors[f"{base}.biases"] = biases
                    n_quant += 1
                    # Force eval so fp16 `arr` can be freed before next tensor.
                    mx.eval(w_q, scales, biases)
                else:
                    # Keep non-quantizable tensors (norms, 1-D, embeddings whose
                    # dim isn't group-aligned) at fp16.
                    out_tensors[key] = arr.astype(mx.float16)
                    mx.eval(out_tensors[key])
                    n_fp16 += 1
                del arr
        out_path = dst_dir / shard.name
        mx.save_safetensors(str(out_path), out_tensors)
        del out_tensors
        print(f"    {shard.name}: wrote {out_path.name}  (peak RSS {rss_gb():.2f} GB)")
    # The quantization block the Swift loader keys off of.
    config = {"quantization": {"group_size": GROUP_SIZE, "bits": BITS}}
    # Preserve any existing config.json fields if present.
    existing = src_dir / "config.json"
    if existing.exists():
        try:
            config = {**json.loads(existing.read_text()), **config}
        except Exception:
            pass
    (dst_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"  {src_dir.name}: quantized {n_quant} layers, kept {n_fp16} fp16; "
          f"config.json -> group_size={GROUP_SIZE} bits={BITS}")


def copy_fp16_component(src_dir: Path, dst_dir: Path) -> None:
    """Copy a small component (vae / CLIP text_encoder) unchanged (stays fp16)."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    print(f"  {src_dir.name}: copied fp16 unchanged ({rss_gb():.2f} GB peak RSS)")


def copy_tree_or_file(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f"  copied {src.name}")


def download_source(out: Path) -> Path:
    from huggingface_hub import snapshot_download
    src = out.with_name(out.name + ".fp16")
    print(f"==> downloading black-forest-labs/FLUX.1-schnell -> {src}")
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-schnell",
        local_dir=str(src),
        allow_patterns=[
            "model_index.json",
            "transformer/*", "vae/*", "text_encoder/*", "text_encoder_2/*",
            "tokenizer/*", "tokenizer_2/*",
        ],
    )
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None,
                    help="fp16 diffusers FLUX.1-schnell snapshot (downloaded if omitted)")
    ap.add_argument("--out", type=Path, required=True, help="output 4-bit bundle dir")
    ap.add_argument("--only", default=None,
                    help="quantize/copy only this single component "
                         "(e.g. text_encoder, vae, transformer) for a smoke test")
    args = ap.parse_args()

    src = args.src if args.src is not None else download_source(args.out)
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"==> quantizing {src} -> {out}  (group_size={GROUP_SIZE}, bits={BITS})")

    components = QUANTIZED_COMPONENTS + FP16_COMPONENTS
    if args.only:
        if args.only in QUANTIZED_COMPONENTS:
            quantize_component(src / args.only, out / args.only)
        elif args.only in FP16_COMPONENTS:
            copy_fp16_component(src / args.only, out / args.only)
        else:
            raise SystemExit(f"--only must be one of {components}")
        print(f"==> done (single component '{args.only}'). Peak RSS {rss_gb():.2f} GB")
        return 0

    for comp in QUANTIZED_COMPONENTS:
        print(f"==> 4-bit quantizing {comp} (memory-dominant) ...")
        quantize_component(src / comp, out / comp)
    for comp in FP16_COMPONENTS:
        print(f"==> copying {comp} (small, kept fp16) ...")
        copy_fp16_component(src / comp, out / comp)
    for folder in COPY_FOLDERS:
        copy_tree_or_file(src / folder, out / folder)
    for f in COPY_FILES:
        copy_tree_or_file(src / f, out / f)

    print(f"==> COMPLETE 4-bit bundle assembled at {out}")
    print(f"==> peak RSS during quantization: {rss_gb():.2f} GB")
    print(f"    verify: scripts/assemble-flux-4bit-bundle.sh {out}")
    print(f"    run:    MANIFOLD_FLUX_MODEL={out} scripts/test-mlx-integration.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
