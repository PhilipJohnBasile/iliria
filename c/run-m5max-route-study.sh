#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 ]] || { echo "Usage: bash run-m5max-route-study.sh MODEL_DIR [PROMPT]" >&2; exit 2; }
MODEL_DIR="${1%/}"
PROMPT="${2:-Review a Godot 4.7 GDScript controller for bugs and allocations.}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[[ -d "$MODEL_DIR" ]] || { echo "error: model directory not found: $MODEL_DIR" >&2; exit 1; }
USAGE_FILE="${ILI_ROUTE_STUDY_USAGE:-$MODEL_DIR/.fa_usage}"
[[ -s "$USAGE_FILE" ]] || { echo "error: usage file not found or empty: $USAGE_FILE" >&2; exit 1; }

NGEN="${ILI_ROUTE_STUDY_NGEN:-112}"
OUT_DIR="${ILI_ROUTE_STUDY_DIR:-bench-m5max/route-study-$(date +%Y%m%d-%H%M%S)}"
TRACE="$OUT_DIR/routes.bin"
LOG="$OUT_DIR/run.log"
CSV="$OUT_DIR/cache-simulation.csv"
mkdir -p "$OUT_DIR"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/iliria-route-study.XXXXXX")"
usage_target="$MODEL_DIR/.fa_usage"
usage_had=0
if [[ -f "$usage_target" ]]; then cp -p "$usage_target" "$TMP_DIR/usage.backup"; usage_had=1; fi
restore_usage() {
  if [[ "$usage_had" == 1 ]]; then cp -p "$TMP_DIR/usage.backup" "$usage_target"
  else rm -f "$usage_target"; fi
}
cleanup() { restore_usage 2>/dev/null || true; rm -rf "$TMP_DIR"; }
trap cleanup EXIT INT TERM

ILI_M5_LAB_METAL4=0 bash ./build-m5max-lab.sh
restore_usage

env \
  TEMP=0 SEED=1 \
  ILI_ROUTE_TRACE="$TRACE" \
  ILI_IGNORE_PROFILE=1 ILI_RAM_GB="${ILI_ROUTE_STUDY_RAM_GB:-114}" \
  ILI_DRAFT=0 ILI_PIPE=1 ILI_IO_THREADS="${ILI_ROUTE_STUDY_IO_THREADS:-8}" \
  ILI_OMP_THREADS="${ILI_ROUTE_STUDY_OMP_THREADS:-6}" ILI_KMP_BLOCKTIME=200 \
  ILI_PILOT=0 ILI_PILOT_REAL=0 ILI_REPIN=0 \
  ILI_MMAP=0 ILI_DIRECT=1 ILI_METAL4_MOE=0 ILI_AUTOPIN=1 \
  ILI_NGEN="$NGEN" \
  bash ./run-m5max-fast.sh "$MODEL_DIR" run "$PROMPT" 2>&1 | tee "$LOG"

[[ -s "$TRACE" ]] || { echo "error: route trace was not produced: $TRACE" >&2; exit 1; }

# Trace v2 carries the exact expert byte size and live pin/LRU slot budget, so no
# rounded memory estimate is supplied here.
python3 tools/simulate_m5max_cache.py "$TRACE" \
  --usage "$USAGE_FILE" \
  --pin-gb 0 8 12 16 20 24 32 48 \
  --tokens "$NGEN" \
  --csv "$CSV" | tee "$OUT_DIR/cache-simulation.txt"

cat <<EOF

Route study complete.
  trace:      $TRACE
  run log:    $LOG
  simulation: $OUT_DIR/cache-simulation.txt
  csv:        $CSV
EOF
