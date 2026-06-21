#!/usr/bin/env bash
#
# tool-decoy-sweep.sh — sweep the manifold-tools-mlx tool-selection harness
# across one or more local MLX models × a ladder of decoy counts, and print a
# per-model "correct-tool selection vs. # advertised tools" table.
#
# For each model and each N in the decoy ladder, this advertises the scenario's
# required tool(s) plus N plausible-but-wrong decoys (--extra-tools N) and
# records `clean_dispatch` — scenarios that passed AND never called a decoy.
# The point at which clean_dispatch starts dropping is the model's practical
# "calls correctly up to ~K tools" ceiling.
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

echo "Building manifold-tools-mlx (-c $CONFIG) …" >&2
swift build -c "$CONFIG" --product manifold-tools-mlx >&2
BIN="$(swift build -c "$CONFIG" --product manifold-tools-mlx --show-bin-path)/manifold-tools-mlx"

OUT_DIR="tmp/tool-decoy-sweep"
mkdir -p "$OUT_DIR"

fixtures_args=()
if [ -n "${FIXTURES_ROOT:-}" ]; then
  fixtures_args=(--fixtures-root "$FIXTURES_ROOT")
fi

# Collected "model<TAB>N<TAB>clean<TAB>passed<TAB>total" rows for the final table.
rows_file="$(mktemp)"
trap 'rm -f "$rows_file"' EXIT

for model in "$@"; do
  model_name="$(basename "$model")"
  for n in $LADDER; do
    log="$OUT_DIR/${model_name}.n${n}.jsonl"
    echo "── $model_name × extra-tools=$n ──" >&2
    # Don't let a non-zero exit (assertion failure) abort the sweep.
    set +e
    summary="$("$BIN" --model "$model" --scenario "$SCENARIO" \
      --extra-tools "$n" --output "$log" "${fixtures_args[@]}" 2>&2 \
      | grep '^SUMMARY ')"
    set -e
    # SUMMARY extra_tools=N passed=X/Y clean_dispatch=A/Y
    passed="$(printf '%s' "$summary" | sed -n 's/.*passed=\([0-9]*\/[0-9]*\).*/\1/p')"
    clean="$(printf '%s' "$summary" | sed -n 's/.*clean_dispatch=\([0-9]*\)\/.*/\1/p')"
    total="$(printf '%s' "$summary" | sed -n 's/.*clean_dispatch=[0-9]*\/\([0-9]*\).*/\1/p')"
    if [ -z "$clean" ]; then clean="?"; total="?"; passed="?"; fi
    printf '%s\t%s\t%s\t%s\n' "$model_name" "$n" "$clean" "$total" >>"$rows_file"
    printf '%s\t%s\t%s\n' "$model_name" "$n" "${summary:-<no SUMMARY line>}" >&2
  done
done

echo
echo "=== Clean-dispatch (passed AND no decoy called) by decoy count ==="
# Header: model + one column per ladder step.
printf 'model'
for n in $LADDER; do printf '\t+%s' "$n"; done
printf '\ttotal\n'

for model in "$@"; do
  model_name="$(basename "$model")"
  printf '%s' "$model_name"
  total_seen="?"
  for n in $LADDER; do
    clean="$(awk -F'\t' -v m="$model_name" -v nn="$n" '$1==m && $2==nn {print $3}' "$rows_file")"
    total_seen="$(awk -F'\t' -v m="$model_name" -v nn="$n" '$1==m && $2==nn {print $4}' "$rows_file")"
    printf '\t%s' "${clean:-?}"
  done
  printf '\t%s\n' "$total_seen"
done

echo
echo "Transcripts: $OUT_DIR/<model>.n<N>.jsonl"
