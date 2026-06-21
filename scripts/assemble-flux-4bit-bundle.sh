#!/usr/bin/env bash
#
# assemble-flux-4bit-bundle.sh — assemble a COMPLETE pre-quantized 4-bit
# FLUX.1-schnell MLX bundle in the diffusers multi-folder layout that
# `FluxSwift`'s `FluxModelCore.loadWeights(from:)` can consume on a
# memory-constrained Apple Silicon machine (≤24 GB), then point
# `MANIFOLD_FLUX_MODEL` at it to run `FluxDiffusionIntegrationTests`.
#
# Why this exists (GitHub issue #39)
# ----------------------------------
# fp16 FLUX.1-schnell is ~33.7 GB on disk and the loader reads the full fp16
# weights BEFORE quantizing, so peak resident memory ≥33.7 GB — it cannot load
# on a 24 GB machine at all. A complete pre-quantized 4-bit bundle is ~6–7 GB
# resident and fits comfortably; the loader detects the on-disk quantization
# (per-component `config.json` `quantization` block, or `.scales`/`.biases`
# tensors) and SKIPS the in-memory quantize pass.
#
# IMPORTANT: the popular single-file 4-bit bundle
#   argmaxinc/mlx-FLUX.1-schnell-4bit-quantized
# is INCOMPLETE for this loader: it ships ONLY a quantized transformer +
# autoencoder, with NO T5 text encoder (text_encoder_2) and NO tokenizers. The
# diffusers loader here needs all components, so that bundle WILL NOT load.
# This script assembles the complete layout instead.
#
# Required complete layout (diffusers multi-folder)
# -------------------------------------------------
#   <root>/
#     model_index.json
#     transformer/      *.safetensors  + config.json {"quantization":{"bits":4,"group_size":64}}
#     vae/              diffusion_pytorch_model.safetensors  [+ config.json]
#     text_encoder/     model.safetensors                    (CLIP; fp16 ok, ~0.25 GB)
#     text_encoder_2/   *.safetensors  + config.json {"quantization":{...}}   (T5-XXL)
#     tokenizer/        vocab.json, merges.txt, ...           (CLIP tokenizer)
#     tokenizer_2/      tokenizer.json, tokenizer_config.json, spiece.model, ...  (T5)
#
# The transformer and text_encoder_2 (T5-XXL) are the memory-dominant pieces and
# MUST be 4-bit. The VAE and CLIP text_encoder are small; either fp16 or 4-bit
# is fine. Tokenizers are plain JSON/vocab assets (no weights).
#
# This mirrors the layout documented in
#   Sources/FluxSwift/FluxConfiguration.swift  (flux1Schnell)
# and validated, file-by-file, by `FluxBundleLayout.validate(_:)`.
#
# Usage
# -----
#   # 1. Assemble a bundle. This drives the real quantizer
#   #    scripts/quantize-flux-4bit.py (needs Python + mlx + Apple Silicon).
#   #    Reuse an existing fp16 snapshot to skip the ~34 GB download:
#   FLUX_FP16_SRC=/path/to/fp16 scripts/assemble-flux-4bit-bundle.sh /path/to/flux-schnell-4bit
#   #    ...or omit FLUX_FP16_SRC to download black-forest-labs/FLUX.1-schnell.
#
#   # 2. Run the gated integration tests against it:
#   MANIFOLD_FLUX_MODEL=/path/to/flux-schnell-4bit scripts/test-mlx-integration.sh
#
# The quantization is performed by scripts/quantize-flux-4bit.py, which streams
# component-by-component / tensor-by-tensor so peak RSS stays a few hundred MB
# even for the ~24 GB fp16 transformer (24 GB-RAM safe). Producing real 4-bit
# weights requires Apple Silicon + MLX and ~34 GB transient disk for the fp16
# source (unless FLUX_FP16_SRC points at one already on disk). If you already
# have a complete 4-bit bundle, skip to step 2 and point MANIFOLD_FLUX_MODEL at
# it.
set -euo pipefail

DEST="${1:-}"
if [[ -z "$DEST" ]]; then
    echo "usage: $0 <destination-bundle-dir>" >&2
    exit 2
fi

# Components the complete bundle MUST contain (kept in sync with
# FluxBundleLayout.requiredComponents / requiredTokenizerFolders).
REQUIRED_FILES=(
    "model_index.json"
    "vae/diffusion_pytorch_model.safetensors"
    "text_encoder/model.safetensors"
)
REQUIRED_GLOB_DIRS=(
    "transformer"        # one or more *.safetensors
    "text_encoder_2"     # one or more *.safetensors (T5-XXL)
)
REQUIRED_TOKENIZER_DIRS=(
    "tokenizer"
    "tokenizer_2"
)

verify_bundle() {
    local root="$1"
    local ok=1
    for f in "${REQUIRED_FILES[@]}"; do
        if [[ ! -f "$root/$f" ]]; then echo "  MISSING file: $f" >&2; ok=0; fi
    done
    for d in "${REQUIRED_GLOB_DIRS[@]}"; do
        if ! ls "$root/$d"/*.safetensors >/dev/null 2>&1; then
            echo "  MISSING *.safetensors in: $d/" >&2; ok=0
        fi
    done
    for d in "${REQUIRED_TOKENIZER_DIRS[@]}"; do
        if [[ ! -d "$root/$d" ]] || [[ -z "$(ls -A "$root/$d" 2>/dev/null)" ]]; then
            echo "  MISSING/empty tokenizer dir: $d/" >&2; ok=0
        fi
    done
    return $((1 - ok))
}

if verify_bundle "$DEST" 2>/dev/null; then
    echo "==> $DEST is already a complete 4-bit FLUX bundle. Nothing to do."
    echo "    Run: MANIFOLD_FLUX_MODEL=$DEST scripts/test-mlx-integration.sh"
    exit 0
fi

# Real quantizer: scripts/quantize-flux-4bit.py does the fp16 -> 4-bit pass
# component-by-component / tensor-by-tensor so peak RSS stays a few hundred MB
# on a 24 GB box (see its header). Drive it from here so callers have one entry
# point. Set FLUX_FP16_SRC to an already-downloaded fp16 snapshot to skip the
# ~34 GB download. The per-component config.json `quantization` block it writes
# is what FluxModelCore.quantizationConfig(in:) reads to take the pre-quantized
# branch and skip the in-memory quantize pass.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANTIZER="$SCRIPT_DIR/quantize-flux-4bit.py"
PYTHON="${PYTHON:-python3}"
FP16_SRC="${FLUX_FP16_SRC:-}"

echo "==> $DEST is not (yet) a complete bundle; running the 4-bit quantizer." >&2

if [[ ! -f "$QUANTIZER" ]]; then
    echo "ERROR: quantizer not found at $QUANTIZER" >&2
    exit 1
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON' not found. Install Python 3 and \`pip install mlx" >&2
    echo "       safetensors huggingface_hub\`. Set PYTHON=... for a venv." >&2
    exit 1
fi
if ! "$PYTHON" -c "import mlx.core, safetensors, huggingface_hub" 2>/dev/null; then
    echo "ERROR: missing Python deps. Run:" >&2
    echo "         $PYTHON -m pip install mlx safetensors huggingface_hub" >&2
    echo "       (Apple Silicon required for mlx.)" >&2
    exit 1
fi

if [[ -n "$FP16_SRC" ]]; then
    echo "==> using fp16 source: $FP16_SRC (no download)" >&2
    "$PYTHON" "$QUANTIZER" --src "$FP16_SRC" --out "$DEST"
else
    echo "==> no FLUX_FP16_SRC set; the quantizer will download" >&2
    echo "    black-forest-labs/FLUX.1-schnell (~34 GB transient). Ctrl-C to" >&2
    echo "    abort and re-run with FLUX_FP16_SRC=<snapshot> to reuse a download." >&2
    "$PYTHON" "$QUANTIZER" --out "$DEST"
fi

echo "==> verifying assembled bundle layout ..."
if verify_bundle "$DEST"; then
    echo "==> $DEST is a complete 4-bit FLUX bundle."
    echo "    Run: MANIFOLD_FLUX_MODEL=$DEST scripts/test-mlx-integration.sh"
    exit 0
fi
echo "ERROR: quantizer ran but $DEST is still incomplete (see MISSING above)." >&2
exit 1
