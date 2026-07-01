#!/usr/bin/env bash
# check-vendored-sync.sh — detect drift between this repo's vendored copies of
# ManifoldKit core's tool-calling fixtures and the upstream originals.
#
# WHY THIS EXISTS
# ----------------
# `Sources/manifold-tools-mlx/Fixtures/manifold-tools/` is a hand-copied vendored
# tree from core's `Tests/Fixtures/manifold-tools/` (see
# Sources/manifold-tools-mlx/README.md). Nothing re-syncs it automatically, so it
# silently drifts whenever core changes those fixtures without a matching update
# here. This script diffs the local vendored files against the core tag that
# matches the ManifoldKit version resolved in Package.resolved.
#
# NOTE: as of ManifoldKit 0.62 (#2042) `ScenarioLoader.loadBuiltIn()` resolves
# scenario JSON directly via `Bundle.module` from the published `ManifoldTools`
# product, so the scenario JSONs themselves are no longer vendored in this repo
# — only the fixture tree is. If a `Sources/manifold-tools-mlx/Scenarios/`
# directory reappears in the future, add it to VENDORED_PAIRS below.
#
# USAGE
# -----
#   scripts/check-vendored-sync.sh            # warn mode (default): always exit 0
#   scripts/check-vendored-sync.sh --warn      # same, explicit
#   scripts/check-vendored-sync.sh --strict    # exit 1 if any file has drifted
#
# Bash 3.2 compatible (macOS ships 3.2 as /bin/bash, and CI runners too) — no
# `declare -A`, no `mapfile`, no associative arrays.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODE="warn"
case "${1:-}" in
  --strict) MODE="strict" ;;
  --warn|"") MODE="warn" ;;
  *)
    echo "usage: $0 [--warn|--strict]" >&2
    exit 2
    ;;
esac

# --- Resolve the ManifoldKit tag to compare against -------------------------

if [ ! -f Package.resolved ]; then
  echo "check-vendored-sync: Package.resolved not found; run 'swift package resolve' first." >&2
  exit "$([ "$MODE" = "strict" ] && echo 1 || echo 0)"
fi

MK_VERSION="$(python3 -c '
import json, sys
try:
    with open("Package.resolved") as f:
        data = json.load(f)
except Exception:
    sys.exit(1)
for pin in data.get("pins", []):
    if pin.get("identity") == "manifoldkit":
        state = pin.get("state", {}) or {}
        version = state.get("version")
        if version:
            print(version)
            sys.exit(0)
sys.exit(1)
' 2>/dev/null || true)"

if [ -z "$MK_VERSION" ]; then
  echo "check-vendored-sync: could not read the resolved ManifoldKit version from Package.resolved." >&2
  exit "$([ "$MODE" = "strict" ] && echo 1 || echo 0)"
fi

MK_TAG="v${MK_VERSION}"
RAW_BASE="https://raw.githubusercontent.com/ManifoldKit/ManifoldKit/${MK_TAG}"

echo "check-vendored-sync: comparing against ManifoldKit ${MK_TAG}"
echo

# --- Vendored pairs: "<local-dir>|<core-path>" -------------------------------
# One entry per vendored tree. Add a Scenarios pair here if that vendored copy
# ever comes back (see NOTE above).

VENDORED_PAIRS="Sources/manifold-tools-mlx/Fixtures/manifold-tools|Tests/Fixtures/manifold-tools"

DRIFT_COUNT=0
MISSING_COUNT=0
OK_COUNT=0

SCRATCH_BODY="$(mktemp)"
trap 'rm -f "$SCRATCH_BODY"' EXIT

printf '%-58s %-10s\n' "FILE" "STATUS"
printf '%-58s %-10s\n' "----" "------"

for pair in $VENDORED_PAIRS; do
  local_dir="${pair%%|*}"
  core_path="${pair##*|}"

  if [ ! -d "$local_dir" ]; then
    echo "check-vendored-sync: local dir '$local_dir' not found; skipping." >&2
    continue
  fi

  # find every regular file under the local vendored dir, relative path.
  while IFS= read -r rel_path; do
    local_file="${local_dir}/${rel_path}"
    remote_url="${RAW_BASE}/${core_path}/${rel_path}"

    # No -f: a 404 still exits 0 from curl (just an empty/error body), so we
    # read the HTTP status explicitly instead of relying on curl's exit code
    # under `set -e pipefail` (a non-zero curl exit inside a `$(...)`
    # assignment would otherwise abort the whole script).
    http_code="$(curl -sL --max-time 20 -o "$SCRATCH_BODY" -w '%{http_code}' "$remote_url" || echo "000")"

    if [ "$http_code" != "200" ]; then
      printf '%-58s %-10s\n' "${core_path}/${rel_path}" "MISSING-UPSTREAM (HTTP ${http_code})"
      MISSING_COUNT=$((MISSING_COUNT + 1))
      continue
    fi

    remote_hash="$(shasum -a 256 "$SCRATCH_BODY" | awk '{print $1}')"
    local_hash="$(shasum -a 256 "$local_file" | awk '{print $1}')"

    if [ "$local_hash" = "$remote_hash" ]; then
      printf '%-58s %-10s\n' "${local_dir}/${rel_path}" "OK"
      OK_COUNT=$((OK_COUNT + 1))
    else
      printf '%-58s %-10s\n' "${local_dir}/${rel_path}" "DRIFT"
      DRIFT_COUNT=$((DRIFT_COUNT + 1))
    fi
  done < <(cd "$local_dir" && find . -type f | sed 's#^\./##' | sort)
done

echo
echo "check-vendored-sync: ${OK_COUNT} ok, ${DRIFT_COUNT} drifted, ${MISSING_COUNT} missing-upstream/unreachable."

if [ "$OK_COUNT" -eq 0 ] && [ "$DRIFT_COUNT" -eq 0 ] && [ "$MISSING_COUNT" -gt 0 ]; then
  echo "check-vendored-sync: no successful comparisons at all — this looks like a network failure, not real drift. Exiting 0 regardless of mode." >&2
  exit 0
fi

if [ "$MODE" = "strict" ] && [ "$DRIFT_COUNT" -gt 0 ]; then
  exit 1
fi

exit 0
