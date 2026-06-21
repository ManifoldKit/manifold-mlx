#!/usr/bin/env bash
#
# tool-decoy-sweep.sh — sweep the manifold-tools-mlx tool-selection harness
# across one or more local MLX models × a ladder of decoy counts, and print a
# per-model "correct-tool selection vs. # advertised tools" table.
#
# For each model and each N in the decoy ladder, this advertises the scenario's
# required tool(s) plus N plausible-but-wrong decoys (--extra-tools N) and
# records the macro-F1 of correct-tool selection (ManifoldInference 0.58's
# ConfusionCounts/MacroAveragedMetrics): precision drops as the model grabs
# decoys, recall drops as it misses the required tool. The decoy count at which
# F1 starts falling is the model's practical "selects correctly up to ~K tools"
# ceiling.
#
# CI runners ship Bash 3.2 (no `declare -A`); this script stays 3.2-compatible.
#
# Usage:
#   scripts/tool-decoy-sweep.sh MODEL_DIR [MODEL_DIR ...]
#
# Env overrides:
#   DECOY_LADDER  space-separated N values   (default: "0 1 3 5 10 20")
#   SCENARIO      --scenario value           (default: all)
#   FIXTURES_ROOT --fixtures-root value      (default: bundled)
#   CONFIG        swift build -c value       (default: release)
#   MTOOLS_BIN    pre-built manifold-tools-mlx binary to use as-is
#
# IMPORTANT: MLX generation needs mlx-swift's Metal `metallib`, which only the
# Xcode/xcodebuild build compiles+bundles — a plain `swift build` binary aborts
# at model load with "Failed to load the default metallib". So to actually run
# the eval, build the executable via xcodebuild and pass it through MTOOLS_BIN:
#
#   xcodebuild -scheme manifold-tools-mlx -configuration Release \
#     -derivedDataPath .build/tools-mlx-derived \
#     -destination 'platform=macOS,arch=arm64' build
#   MTOOLS_BIN=.build/tools-mlx-derived/Build/Products/Release/manifold-tools-mlx \
#     scripts/tool-decoy-sweep.sh MODEL_DIR ...
#
# Without MTOOLS_BIN the script falls back to `swift build` (compiles fine, but
# can only run `--list`/`--help`; generation will fail).
#
# Exit: 0 if every run completed (regardless of pass/fail); 1 on a run error or
# bad arguments. Per-run pass/fail is captured in the table, not the exit code.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 MODEL_DIR [MODEL_DIR ...]" >&2
  exit 1
fi

LADDER="${DECOY_LADDER:-0 1 3 5 10 20}"
SCENARIO="${SCENARIO:-all}"
CONFIG="${CONFIG:-release}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -n "${MTOOLS_BIN:-}" ]; then
  BIN="$MTOOLS_BIN"
  [ -x "$BIN" ] || { echo "MTOOLS_BIN is not an executable: $BIN" >&2; exit 1; }
  echo "Using pre-built binary: $BIN" >&2
else
  echo "Building manifold-tools-mlx (-c $CONFIG) … (NOTE: swift build can't generate — see header)" >&2
  swift build -c "$CONFIG" --product manifold-tools-mlx >&2
  BIN="$(swift build -c "$CONFIG" --product manifold-tools-mlx --show-bin-path)/manifold-tools-mlx"
fi

OUT_DIR="tmp/tool-decoy-sweep"
mkdir -p "$OUT_DIR"

fixtures_args=()
if [ -n "${FIXTURES_ROOT:-}" ]; then
  fixtures_args=(--fixtures-root "$FIXTURES_ROOT")
fi

# Collected "model<TAB>N<TAB>f1<TAB>clean<TAB>passed" rows for the final table.
# Headline metric is macro-F1 of correct-tool selection (ManifoldInference 0.58).
rows_file="$(mktemp)"
trap 'rm -f "$rows_file"' EXIT

for model in "$@"; do
  model_name="$(basename "$model")"
  baseline_f1=""
  for n in $LADDER; do
    # Short-circuit: if the model can't select the required tool with ZERO decoys
    # (f1==0 at n==0), higher decoy counts only re-measure "can't tool-call" —
    # skip them. Keyed on F1, not assertion pass: a model that selects the tool
    # but fails a content assertion still yields meaningful decoy-pressure data.
    if [ "$n" != "0" ] && [ "$baseline_f1" = "0.000" ]; then
      printf '%s\t%s\t%s\t%s\t%s\n' "$model_name" "$n" "skip" "skip" "skip" >>"$rows_file"
      printf '%s\t%s\t(skipped — f1=0.000 at n=0)\n' "$model_name" "$n" >&2
      continue
    fi
    log="$OUT_DIR/${model_name}.n${n}.jsonl"
    echo "── $model_name × extra-tools=$n ──" >&2
    # Don't let a non-zero exit (assertion failure) abort the sweep.
    set +e
    summary="$("$BIN" --model "$model" --scenario "$SCENARIO" \
      --extra-tools "$n" --output "$log" "${fixtures_args[@]}" 2>&2 \
      | grep '^SUMMARY ')"
    set -e
    # SUMMARY extra_tools=N passed=X/Y clean=A/Y precision=P recall=R f1=F decoy_calls=K scored=S
    f1="$(printf '%s' "$summary" | sed -n 's/.*[^_]f1=\([0-9.]*\).*/\1/p')"
    clean="$(printf '%s' "$summary" | sed -n 's/.*clean=\([0-9]*\/[0-9]*\).*/\1/p')"
    passed="$(printf '%s' "$summary" | sed -n 's/.*passed=\([0-9]*\/[0-9]*\).*/\1/p')"
    if [ -z "$f1" ]; then f1="?"; clean="?"; passed="?"; fi
    if [ "$n" = "0" ]; then baseline_f1="$f1"; fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$model_name" "$n" "$f1" "$clean" "$passed" >>"$rows_file"
    printf '%s\t%s\t%s\n' "$model_name" "$n" "${summary:-<no SUMMARY line>}" >&2
  done
done

echo
echo "=== Macro-F1 of correct-tool selection by decoy count (higher = better) ==="
# Header: model + one column per ladder step.
printf 'model'
for n in $LADDER; do printf '\t+%s' "$n"; done
printf '\n'

for model in "$@"; do
  model_name="$(basename "$model")"
  printf '%s' "$model_name"
  for n in $LADDER; do
    f1="$(awk -F'\t' -v m="$model_name" -v nn="$n" '$1==m && $2==nn {print $3}' "$rows_file")"
    printf '\t%s' "${f1:-?}"
  done
  printf '\n'
done

echo
echo "(clean = passed AND no decoy called; see SUMMARY lines on stderr for precision/recall/decoy_calls)"
echo "Transcripts: $OUT_DIR/<model>.n<N>.jsonl"
