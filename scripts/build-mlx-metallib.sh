#!/bin/bash
# build-mlx-metallib.sh — compile mlx-swift's Metal kernels into an mlx.metallib.
#
# WHY THIS EXISTS
# ----------------
# mlx-swift's `Device` constructor unconditionally loads a precompiled metallib
# (see mlx device.cpp `load_default_library`, which searches, in order:
#   1. <binary dir>/mlx.metallib            (colocated)
#   2. <binary dir>/Resources/mlx.metallib
#   3. mlx-swift_Cmlx.bundle/default.metallib (SwiftPM bundle, Xcode-only)
#   4. <binary dir>/Resources/default.metallib
#   5. a fixed fallback path
# and throws if none is found). The bundle in #3 is only filled by an Xcode /
# `xcodebuild` Metal-shader compile phase; a plain `swift build` produces a
# binary that loads/routes MLX models but dies at GPU init with:
#
#     MLX error: Failed to load the default metallib. library not found ...
#
# Of the five locations, the only one a *dependency* package can populate is the
# colocated `mlx.metallib` (#1) — #3 is hardcoded to mlx-swift's own bundle
# name. This script reproduces what the Xcode metallib phase does using the
# standalone Metal toolchain (`xcrun metal` / `xcrun metallib`). It compiles the
# *generated* kernel entry points mlx-swift ships for its JIT build
# (Source/Cmlx/mlx-generated/metal — every other kernel is JIT-compiled at
# runtime from embedded source), which is the complete, correct metallib for the
# macOS configuration Package.swift selects.
#
# It is invoked two ways:
#   - by the MLXMetallibPlugin SwiftPM prebuild plugin (positional-arg form),
#     which runs automatically during `swift build` of any consumer;
#   - manually, for debugging or to stage the metallib next to an already-built
#     binary (legacy --config/--dest form).
#
# GRACEFUL DEGRADATION
# --------------------
# If the Metal toolchain is unavailable (e.g. the Metal Toolchain component is
# not installed, or on a non-macOS host), the script prints a warning and exits
# 0 WITHOUT producing output. This is deliberate: the prebuild plugin must not
# fail the whole `swift build` on a machine that only compiles (CI, Linux, a Mac
# without `xcodebuild -downloadComponent MetalToolchain`). Such builds keep
# today's behaviour — they compile fine and only abort at MLX GPU init, exactly
# as before this change. Shader errors from an available compiler are fatal.
#
# USAGE
#   Plugin form:
#     build-mlx-metallib.sh <generated-metal-dir> <output-dir>
#       Compiles every *.metal under <generated-metal-dir> and writes
#       <output-dir>/mlx.metallib.
#
#   Manual form:
#     build-mlx-metallib.sh [--config debug|release] [--dest <dir-or-file>] [-v]
#       --config   SwiftPM config whose bin dir to target (default: debug).
#       --dest     Where to write the metallib. A directory → "<dest>/mlx.metallib";
#                  a path ending in .metallib → used verbatim. Defaults to the
#                  SwiftPM bin dir for --config (colocated lookup #1).
#       -v         Verbose: echo each compile.
set -euo pipefail

warn() { echo "build-mlx-metallib.sh: $*" >&2; }

GEN=""
OUTDIR=""
CONFIG="debug"
DEST=""
VERBOSE=0

if [ $# -ge 1 ] && [ "${1#-}" = "$1" ]; then
  if [ $# -ne 2 ] || [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
    warn "plugin form requires: <generated-metal-dir> <output-dir>"
    exit 2
  fi
  GEN="$1"
  OUTDIR="$2"
else
  while [ $# -gt 0 ]; do
    case "$1" in
      --config|--dest)
        if [ $# -lt 2 ] || [ -z "$2" ]; then
          warn "$1 requires a value"; exit 2
        fi
        if [ "$1" = --config ]; then CONFIG="$2"; else DEST="$2"; fi
        shift 2 ;;
      -v|--verbose) VERBOSE=1; shift ;;
      -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
      *) warn "unknown arg '$1'"; exit 2 ;;
    esac
  done
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  (cd "$ROOT" && swift package resolve)
  GEN="$ROOT/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"
  if [ -z "$DEST" ]; then
    DEST="$(cd "$ROOT" && swift build -c "$CONFIG" --show-bin-path)"
  fi
fi

if [ -n "$OUTDIR" ]; then
  OUT="${OUTDIR%/}/mlx.metallib"
else
  case "$DEST" in
    *.metallib) OUT="$DEST" ;;
    *) OUT="${DEST%/}/mlx.metallib" ;;
  esac
fi
mkdir -p "$(dirname "$OUT")"
STAMP="$(dirname "$OUT")/.$(basename "$OUT").inputs.sha256"
TMP="$(mktemp -d "$(dirname "$OUT")/.mlx-air.XXXXXX")"
# A failed build must never leave a stale or partial library that a later build
# can package. Temporary output also keeps partially linked files invisible.
SUCCEEDED=0
cleanup() {
  rm -rf "$TMP"
  if [ "$SUCCEEDED" -ne 1 ]; then rm -f "$OUT" "$STAMP"; fi
}
trap cleanup EXIT

if [ ! -d "$GEN" ]; then
  warn "mlx-swift generated metal sources not found at: $GEN"
  exit 1
fi
find "$GEN" -type f -name '*.metal' -print | LC_ALL=C sort > "$TMP/kernels"
if [ ! -s "$TMP/kernels" ]; then
  warn "no .metal kernels found under $GEN"
  exit 1
fi

# Probe availability separately. Once the installed drivers report their
# versions, shader or linker errors are build failures, never missing-tool skips.
if ! command -v xcrun >/dev/null 2>&1; then
  warn "xcrun unavailable; compile-only build, no MLX GPU library produced."
  exit 0
fi
if ! COMPILER="$(xcrun --sdk macosx metal --version 2>&1)"; then
  warn "Metal compiler unavailable; compile-only build, no MLX GPU library produced."
  warn "$COMPILER"
  exit 0
fi
if ! LINKER="$(xcrun --sdk macosx metallib --version 2>&1)"; then
  warn "Metal linker unavailable; compile-only build, no MLX GPU library produced."
  warn "$LINKER"
  exit 0
fi
SDK_PATH="$(xcrun --sdk macosx --show-sdk-path)"
SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"

# Content hashing catches header-only patches and preserved checkout timestamps.
# The script itself covers flags; compiler/linker + SDK cover toolchain changes.
find "$GEN" -type f \( -name '*.metal' -o -name '*.h' \) -print | LC_ALL=C sort > "$TMP/inputs"
{
  shasum -a 256 "$0"
  printf '%s\n' "$COMPILER" "$LINKER" "$SDK_PATH" "$SDK_VERSION"
  while IFS= read -r source; do shasum -a 256 "$source"; done < "$TMP/inputs"
} > "$TMP/input-hashes"
shasum -a 256 "$TMP/input-hashes" | awk '{print $1}' > "$TMP/fingerprint"
if [ -s "$OUT" ] && [ -f "$STAMP" ] && cmp -s "$STAMP" "$TMP/fingerprint"; then
  SUCCEEDED=1
  if [ "$VERBOSE" -eq 1 ]; then warn "up to date: $OUT"; fi
  echo "$OUT"
  exit 0
fi

AIRS=()
while IFS= read -r source; do
  air="$TMP/${#AIRS[@]}.air"
  if [ "$VERBOSE" -eq 1 ]; then warn "metal -c ${source#"$GEN"/}"; fi
  # Stock language selection is intentional: the dependency must support the
  # selected Xcode compiler, including its native app-target Metal build phase.
  xcrun --sdk macosx metal -O2 -c "$source" -I "$GEN" -o "$air"
  AIRS+=("$air")
done < "$TMP/kernels"
xcrun --sdk macosx metallib "${AIRS[@]}" -o "$TMP/mlx.metallib"
if [ ! -s "$TMP/mlx.metallib" ]; then
  warn "Metal linker produced no library"
  exit 1
fi
mv "$TMP/mlx.metallib" "$OUT"
mv "$TMP/fingerprint" "$STAMP"
SUCCEEDED=1
warn "wrote $OUT (${#AIRS[@]} kernels)"
echo "$OUT"
