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
# as before this change.
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
set -uo pipefail

warn() { echo "build-mlx-metallib.sh: $*" >&2; }

# --- Metal toolchain availability (graceful) ---------------------------------
# We need both `metal` (compile) and `metallib` (link). `xcrun --find` only
# proves the wrapper exists; the actual driver still fails if the Metal Toolchain
# component is missing, so the real compile below is the authoritative check.
if ! command -v xcrun >/dev/null 2>&1; then
  warn "xcrun not found (not a macOS host with Xcode); skipping metallib build."
  exit 0
fi

# --- Argument parsing --------------------------------------------------------
GEN=""
OUTDIR=""
CONFIG="debug"
DEST=""
VERBOSE=0

# Positional plugin form: exactly two non-flag args.
if [ $# -ge 1 ] && [ "${1#-}" = "$1" ]; then
  GEN="${1:-}"
  OUTDIR="${2:-}"
  if [ -z "$GEN" ] || [ -z "$OUTDIR" ]; then
    warn "plugin form requires: <generated-metal-dir> <output-dir>"
    exit 2
  fi
else
  # Legacy manual flag form.
  while [ $# -gt 0 ]; do
    case "$1" in
      --config) CONFIG="$2"; shift 2 ;;
      --dest)   DEST="$2"; shift 2 ;;
      -v|--verbose) VERBOSE=1; shift ;;
      -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
      *) warn "unknown arg '$1'"; exit 2 ;;
    esac
  done
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  ( cd "$ROOT" && swift package resolve >/dev/null 2>&1 ) || true
  MLX_CHECKOUT="$ROOT/.build/checkouts/mlx-swift"
  if [ ! -d "$MLX_CHECKOUT" ]; then
    MLX_CHECKOUT="$(find "$ROOT/.build" -maxdepth 3 -type d -name mlx-swift -path '*checkouts*' 2>/dev/null | head -1)"
  fi
  GEN="$MLX_CHECKOUT/Source/Cmlx/mlx-generated/metal"
  if [ -z "$DEST" ]; then
    DEST="$(cd "$ROOT" && swift build -c "$CONFIG" --show-bin-path 2>/dev/null)"
  fi
fi

# --- Validate generated kernel dir -------------------------------------------
if [ ! -d "$GEN" ]; then
  warn "mlx-swift generated metal sources not found at: $GEN"
  warn "run 'swift package resolve' / 'swift build' first; skipping."
  exit 0
fi

# --- Resolve output path -----------------------------------------------------
if [ -n "${OUTDIR:-}" ]; then
  mkdir -p "$OUTDIR" || { warn "cannot create output dir $OUTDIR; skipping."; exit 0; }
  OUT="${OUTDIR%/}/mlx.metallib"
else
  case "$DEST" in
    *.metallib) OUT="$DEST" ;;
    *)          OUT="${DEST%/}/mlx.metallib" ;;
  esac
  mkdir -p "$(dirname "$OUT")" || { warn "cannot create dir for $OUT; skipping."; exit 0; }
fi

# --- Up-to-date check --------------------------------------------------------
# Skip recompilation if the metallib is newer than every kernel source. Keeps
# the prebuild command cheap on incremental builds (prebuild runs every build).
if [ -f "$OUT" ]; then
  NEWEST_SRC="$(find "$GEN" -name '*.metal' -newer "$OUT" -print -quit 2>/dev/null)"
  if [ -z "$NEWEST_SRC" ]; then
    [ "$VERBOSE" -eq 1 ] && warn "up to date: $OUT"
    echo "$OUT"
    exit 0
  fi
fi

# --- Compile + link ----------------------------------------------------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

AIRS=()
COMPILE_FAILED=0
while IFS= read -r f; do
  rel="${f#"$GEN"/}"
  name="$(echo "$rel" | tr '/' '_' | sed 's/\.metal$//')"
  air="$TMP/$name.air"
  [ "$VERBOSE" -eq 1 ] && warn "metal -c $rel"
  # -I "$GEN": kernels #include their flattened sibling headers from here.
  # No explicit -std: let the toolchain pick its default Metal language version;
  # pinning an older -std collides with the newer toolchain's stdlib.
  if ! xcrun --sdk macosx metal -O2 -c "$f" -I "$GEN" -o "$air" 2>"$TMP/metal.err"; then
    # First compile failure is almost always "missing Metal Toolchain" — treat
    # as graceful skip rather than failing the build.
    warn "metal compile unavailable or failed; skipping metallib build."
    warn "  (install with: xcodebuild -downloadComponent MetalToolchain)"
    [ -s "$TMP/metal.err" ] && sed 's/^/    /' "$TMP/metal.err" >&2
    COMPILE_FAILED=1
    break
  fi
  AIRS+=("$air")
done < <(find "$GEN" -name '*.metal' | sort)

if [ "$COMPILE_FAILED" -eq 1 ]; then
  exit 0
fi
if [ ${#AIRS[@]} -eq 0 ]; then
  warn "no .metal kernels found under $GEN; skipping."
  exit 0
fi

if ! xcrun --sdk macosx metallib "${AIRS[@]}" -o "$OUT" 2>"$TMP/lib.err"; then
  warn "metallib link unavailable or failed; skipping."
  [ -s "$TMP/lib.err" ] && sed 's/^/    /' "$TMP/lib.err" >&2
  rm -f "$OUT"
  exit 0
fi

SIZE="$(stat -f%z "$OUT" 2>/dev/null || echo '?')"
warn "wrote $OUT (${SIZE} bytes, ${#AIRS[@]} kernels)"
echo "$OUT"
