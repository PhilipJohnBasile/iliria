#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 ]] || { echo "Usage: bash ab-m5max-k6-matrix.sh MODEL_DIR" >&2; exit 2; }
MODEL_DIR="${1%/}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[[ -d "$MODEL_DIR" ]] || { echo "model directory not found: $MODEL_DIR" >&2; exit 1; }
[[ "$(uname -s)" == Darwin ]] || { echo "this matrix targets macOS" >&2; exit 2; }

PASSES="${ILI_K6_PASSES:-3}"
NGEN="${ILI_K6_NGEN:-112}"
CACHE_STATES="${ILI_K6_CACHE_STATES:-warm cold}"
PROFILE="${ILI_K6_HOTSET_PROFILE:-coding}"
# Which candidate the matrix A/Bs against off:
#   pilot  -> off vs PILOT_REAL K6            (candidate label: k6)
#   metal4 -> off vs ILI_METAL4_MOE=1        (candidate label: m4; builds the METAL4=1 lab engine)
#   pstate -> Metal4-on both modes, off vs ILI_METAL_PERSISTENT_STATE=1 (label: pst; METAL4 build)
AXIS="${ILI_K6_AXIS:-pilot}"
case "$AXIS" in
  pilot)  CAND=k6 ;;
  metal4) CAND=m4 ;;
  pstate) CAND=pst ;;
  *) echo "invalid ILI_K6_AXIS: $AXIS (pilot|metal4)" >&2; exit 2 ;;
esac
OUT="${ILI_K6_RESULT_DIR:-bench-m5max/${CAND}-matrix-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT/logs" "$OUT/system"
MANIFEST_TSV="$OUT/manifest.tsv"
printf 'prompt\tcache_state\ttrial\tmode\tlog\n' > "$MANIFEST_TSV"

# The engine rewrites the persistent usage histogram (.fa_usage[.<profile>])
# after every run, and AUTOPIN derives the hot-set from it at startup. A
# drifting hot-set changes CPU-grouped vs Metal expert placement, whose
# kernel-family rounding difference can flip greedy tokens — so every run
# must boot from the same snapshot or the hash gate is unpassable.
USAGE_SNAP_DIR="$OUT/system/usage-snapshot"
mkdir -p "$USAGE_SNAP_DIR"
usage_files=()
# (Re-)populate usage_files/USAGE_SNAP_DIR from whatever .fa_usage[.<profile>]
# exists on disk RIGHT NOW. Returns 1 (nothing found) / 0 (snapshotted).
snapshot_usage() {
  local f found=1
  for f in "$MODEL_DIR/.fa_usage" "$MODEL_DIR/.fa_usage.$PROFILE"; do
    if [[ -f "$f" ]]; then
      cp "$f" "$USAGE_SNAP_DIR/$(basename "$f")"
      usage_files+=("$f")
      found=0
    fi
  done
  return "$found"
}
if ! snapshot_usage; then
  echo "[ab-m5max-k6] WARNING: no .fa_usage[.${PROFILE}] found under $MODEL_DIR at matrix " \
       "start (fresh model dir / AUTOPIN has never run here) -- restore_usage would otherwise " \
       "be a silent permanent no-op for the WHOLE matrix, defeating the frozen-state hash-gate " \
       "guarantee above. Falling back to a LAZY snapshot on the first run_case that finds one: " \
       "the run that FIRST creates .fa_usage establishes the baseline and is itself unprotected; " \
       "every run after that is frozen relative to it. For full protection, run the engine once " \
       "on this model dir first so .fa_usage[.${PROFILE}] pre-exists." >&2
fi
restore_usage() {
  local f
  # Empty usage_files means either (a) nothing has been snapshotted yet (the
  # fresh-dir case the WARNING above describes -- try the lazy snapshot now,
  # since a prior run_case in this same matrix may have just created
  # .fa_usage), or (b) truly nothing exists anywhere yet, in which case
  # snapshot_usage fails again and this is correctly still a no-op.
  (( ${#usage_files[@]} )) || snapshot_usage || return 0
  for f in "${usage_files[@]}"; do
    cp "$USAGE_SNAP_DIR/$(basename "$f")" "$f"
  done
}
trap restore_usage EXIT

prompts=(
  "Implement a cancellation-safe bounded MPSC queue in Rust using atomics. Explain the memory ordering and include tests."
  "A Godot 4.7 GDScript character controller allocates every physics frame and occasionally tunnels through ramps. Diagnose it and provide a corrected implementation."
  "Review a TypeScript WebSocket reconnection manager for race conditions, leaked timers, stale closures, and exponential-backoff errors. Return a patch and tests."
)
labels=(rust_queue godot_controller typescript_websocket)

if pgrep -x glm >/dev/null 2>&1; then
  echo "another glm process is running; benchmark isolation would be invalid" >&2
  exit 1
fi

swap_mb() {
  local raw
  raw="$(sysctl -n vm.swapusage 2>/dev/null || true)"
  python3 - "$raw" <<'PY'
import re,sys
m=re.search(r"used = ([0-9.]+)([MG])",sys.argv[1])
if not m: print("0"); raise SystemExit
v=float(m.group(1)); print(v*1024 if m.group(2)=="G" else v)
PY
}

snapshot_system() {
  local prefix="$1"
  memory_pressure > "$OUT/system/${prefix}-memory-pressure.txt" 2>&1 || true
  vm_stat > "$OUT/system/${prefix}-vm-stat.txt" 2>&1 || true
  sysctl vm.swapusage > "$OUT/system/${prefix}-swap.txt" 2>&1 || true
  ps -axo pid,ppid,etime,%cpu,%mem,rss,command > "$OUT/system/${prefix}-ps.txt" 2>&1 || true
}

purge_cache() {
  sync
  if sudo -n purge >/dev/null 2>&1; then
    sleep 3
    return 0
  fi
  return 1
}

run_case() {
  local label="$1" prompt="$2" cache="$3" trial="$4" mode="$5" measured="$6"
  local pilot=0 real=0 k=6 metal4=0 pstate=0
  [[ "$AXIS" == pstate ]] && metal4=1
  if [[ "$mode" != off ]]; then
    case "$AXIS" in
      pilot)  pilot=1; real=1 ;;
      metal4) metal4=1 ;;
      pstate) pstate=1 ;;
    esac
  fi
  local stem="${cache}-${label}-t${trial}-${mode}"
  local log="$OUT/logs/${stem}.log"

  if [[ "$cache" == cold ]]; then
    if ! purge_cache; then
      echo "cold-cache case requested but passwordless 'sudo purge' is unavailable" >&2
      return 3
    fi
  fi

  restore_usage
  local before rc
  before="$(swap_mb)"
  snapshot_system "${stem}-before"
  # Subshell (not a brace group): `exit` must end only this case's logging
  # scope, and `set +e` lets a failed run be recorded as MATRIX-EXIT.
  (
    set +e
    echo "MATRIX-META: prompt=$label cache=$cache trial=$trial mode=$mode measured=$measured"
    echo "MATRIX-SWAP-BEFORE-MB: $before"
    env \
      TEMP=0 SEED=1 \
      ILI_PILOT_METRICS=1 \
      ILI_HOTSET_PROFILE="$PROFILE" \
      ILI_IGNORE_PROFILE=1 \
      ILI_RAM_GB="${ILI_K6_RAM_GB:-114}" \
      ILI_WIRED_MB="${ILI_K6_WIRED_MB:-122000}" \
      ILI_DRAFT=0 ILI_PIPE=1 ILI_IO_THREADS="${ILI_K6_IO_THREADS:-8}" \
      ILI_OMP_THREADS="${ILI_K6_OMP_THREADS:-6}" ILI_KMP_BLOCKTIME=200 \
      ILI_MMAP=0 ILI_DIRECT=1 ILI_REPIN=0 ILI_AUTOPIN=1 \
      ILI_PILOT="$pilot" ILI_PILOT_REAL="$real" ILI_PILOT_K="$k" \
      ILI_METAL4_MOE="$metal4" ILI_METAL_PERSISTENT_STATE="$pstate" ILI_CPU_GROUPED_MOE=1 ILI_CPU_MOE_MISSES=0 \
      ILI_NGEN="$NGEN" \
      bash ./run-m5max-fast.sh "$MODEL_DIR" run "$prompt"
    rc=$?
    echo "MATRIX-SWAP-AFTER-MB: $(swap_mb)"
    echo "MATRIX-EXIT: $rc"
    exit "$rc"
  ) > "$log" 2>&1 || rc=$?
  rc="${rc:-0}"
  snapshot_system "${stem}-after"
  if (( rc != 0 )); then
    echo "run failed (exit $rc), matrix aborted: see $log" >&2
    return "$rc"
  fi

  if [[ "$measured" == 1 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$cache" "$trial" "$mode" "logs/${stem}.log" >> "$MANIFEST_TSV"
  fi
}

ILI_M5_LAB_METAL4="$([[ "$AXIS" == metal4 || "$AXIS" == pstate ]] && echo 1 || echo 0)" bash ./build-m5max-lab.sh

actual_states=()
for cache in $CACHE_STATES; do
  case "$cache" in
    warm) actual_states+=(warm) ;;
    cold)
      if sudo -n true >/dev/null 2>&1 && command -v purge >/dev/null 2>&1; then
        actual_states+=(cold)
      else
        echo "warning: skipping cold matrix because passwordless sudo/purge is unavailable" >&2
      fi ;;
    *) echo "invalid cache state: $cache" >&2; exit 2 ;;
  esac
done

for pi in "${!prompts[@]}"; do
  label="${labels[$pi]}"; prompt="${prompts[$pi]}"
  for cache in "${actual_states[@]}"; do
    if [[ "$cache" == warm ]]; then
      # Symmetric unmeasured warmups prevent the first measured mode from owning all page-cache warming.
      run_case "$label" "$prompt" warm 0 off 0
      run_case "$label" "$prompt" warm 0 "$CAND" 0
    fi
    for ((trial=1; trial<=PASSES; trial++)); do
      if (( trial % 2 )); then order=(off "$CAND"); else order=("$CAND" off); fi
      for mode in "${order[@]}"; do
        echo "[$(date '+%F %T')] $cache $label trial=$trial mode=$mode"
        run_case "$label" "$prompt" "$cache" "$trial" "$mode" 1
      done
    done
  done
done

python3 - "$MANIFEST_TSV" "$OUT/manifest.json" "$PASSES" "$NGEN" <<'PY'
import csv,json,sys
src,dst,passes,ngen=sys.argv[1:]
with open(src,newline='') as f: runs=list(csv.DictReader(f,delimiter='\t'))
for r in runs: r['trial']=int(r['trial'])
with open(dst,'w') as f: json.dump({'passes':int(passes),'ngen':int(ngen),'runs':runs},f,indent=2)
PY

python3 tools/summarize_m5max_k6_matrix.py "$OUT" | tee "$OUT/summary-console.txt"
echo "Result: $OUT/SUMMARY.md"
