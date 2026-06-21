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
#   # 1. Assemble a bundle (downloads + quantizes; needs Python + mlx + Apple Silicon):
#   scripts/assemble-flux-4bit-bundle.sh /path/to/flux-schnell-4bit
#
#   # 2. Run the gated integration tests against it:
#   MANIFOLD_FLUX_MODEL=/path/to/flux-schnell-4bit scripts/test-mlx-integration.sh
#
# The quantization step below is a REFERENCE recipe. Producing real 4-bit
# weights requires Apple Silicon + MLX and ~34 GB of transient disk for the fp16
# source download. If you already have a complete 4-bit bundle, just skip to
# step 2 and point MANIFOLD_FLUX_MODEL at it.
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

cat >&2 <<EOF
==> $DEST is not (yet) a complete bundle.

To assemble one you need: Apple Silicon, Python 3, and \`pip install mlx
mlx-lm huggingface_hub\`, plus ~34 GB transient disk for the fp16 source.

Reference recipe (run manually; this script intentionally does NOT auto-
download tens of GB):

  # a. Pull the fp16 diffusers snapshot (transformer/vae/text_encoder/
  #    text_encoder_2/tokenizer/tokenizer_2/model_index.json):
  huggingface-cli download black-forest-labs/FLUX.1-schnell \\
      --local-dir "$DEST.fp16"

  # b. Quantize the memory-dominant components to 4-bit (group_size 64) with
  #    MLX, writing each component's safetensors + a config.json carrying:
  #        {"quantization": {"group_size": 64, "bits": 4}}
  #    for: transformer/  and  text_encoder_2/   (T5-XXL).
  #    Keep vae/ and text_encoder/ (CLIP) as fp16 — they are small.
  #    See mzbac/flux.swift and the MLX quantize() docs for the exact API.

  # c. Copy the unmodified tokenizer/, tokenizer_2/, and model_index.json
  #    from "$DEST.fp16" into "$DEST".

  # d. Re-run this script to verify the assembled layout:
  #        $0 "$DEST"

The per-component config.json \`quantization\` block is what FluxModelCore reads
to take the pre-quantized branch and skip the in-memory quantize pass — see
FluxModelCore.quantizationConfig(in:).
EOF
exit 1
