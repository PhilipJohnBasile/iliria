#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 ]] || { echo "Usage: bash autotune-m5max.sh MODEL_DIR [PROMPT]" >&2; exit 2; }
MODEL_DIR="${1%/}"
PROMPT="${2:-Explain how an optimizing compiler lowers SSA into machine code.}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NGEN="${ILI_AUTOTUNE_NGEN:-96}"
OUT_DIR="${ILI_AUTOTUNE_DIR:-bench-m5max/autotune-$(date +%Y%m%d-%H%M%S)}"
PROFILE_FILE="${ILI_PROFILE_FILE:-$SCRIPT_DIR/.m5max-profile.env}"
MIN_GAIN_PCT="${ILI_AUTOTUNE_MIN_GAIN_PCT:-2}"
mkdir -p "$OUT_DIR"

baseline_hash=""

best_rate=0
best_ram=114
best_draft=0
best_io=8
best_omp=auto
best_kmp=200
best_mmap=0
best_direct=1
best_pilot=0
best_pilot_real=0
best_pilot_k=auto
best_untracked=0
best_unretained=0
best_spin=0

extract_rate() {
  python3 - "$1" <<'PY'
from pathlib import Path
import re, sys
text = Path(sys.argv[1]).read_text(errors="replace")
rates = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?) tok/s", text)]
print(rates[-1] if rates else 0.0)
PY
}

extract_hash() {
  sed -n 's/.*output token hash: \([0-9a-fA-F][0-9a-fA-F]*\).*/\1/p' "$1" | tail -n 1
}

is_better() {
  python3 - "$1" "$2" "$MIN_GAIN_PCT" <<'PY'
import sys
new, old, pct = map(float, sys.argv[1:])
print(1 if new > old * (1.0 + pct / 100.0) else 0)
PY
}

run_case() {
  local name="$1" ram="$2" draft="$3" io="$4" omp="$5" kmp="$6"
  local mmap="$7" direct="$8" pilot="$9" pilot_real="${10}" pilot_k="${11}"
  local untracked="${12}" unretained="${13}" spin="${14}"
  local log="$OUT_DIR/$name.log" status output_hash

  echo "=== $name: RAM=$ram DRAFT=$draft PIPE_WORKERS=$io OMP=$omp KMP=$kmp MMAP=$mmap DIRECT=$direct PILOT=$pilot/$pilot_real/$pilot_k UNTRACKED=$untracked UNRETAINED=$unretained SPIN=$spin ===" | tee "$log"
  set +e
  TEMP=0 SEED=1 ILI_IGNORE_PROFILE=1 ILI_AUTOPIN=0 ILI_REPIN=0 \
  ILI_RAM_GB="$ram" ILI_DRAFT="$draft" ILI_IO_THREADS="$io" \
  ILI_OMP_THREADS="$omp" ILI_KMP_BLOCKTIME="$kmp" \
  ILI_MMAP="$mmap" ILI_DIRECT="$direct" \
  ILI_PILOT="$pilot" ILI_PILOT_REAL="$pilot_real" ILI_PILOT_K="$pilot_k" \
  ILI_UNTRACKED="$untracked" ILI_UNRETAINED="$unretained" ILI_SPIN="$spin" ILI_NGEN="$NGEN" \
    bash ./run-m5max-fast.sh "$MODEL_DIR" run "$PROMPT" 2>&1 | tee -a "$log"
  status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "candidate failed with status $status; continuing" | tee -a "$log" >&2
    LAST_RATE=0
  else
    output_hash="$(extract_hash "$log")"
    if [[ -z "$output_hash" ]]; then
      echo "candidate missing output token hash; rejecting" | tee -a "$log" >&2
      LAST_RATE=0
    elif [[ -z "$baseline_hash" ]]; then
      baseline_hash="$output_hash"
      LAST_RATE="$(extract_rate "$log")"
    elif [[ "$output_hash" != "$baseline_hash" ]]; then
      echo "candidate output hash $output_hash differs from baseline $baseline_hash; rejecting" | tee -a "$log" >&2
      LAST_RATE=0
    else
      LAST_RATE="$(extract_rate "$log")"
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$LAST_RATE" "$status" "$ram" "$draft" "$io" "$omp" "$kmp" "$mmap" "$direct" \
    "$pilot" "$pilot_real" "$pilot_k" "$untracked" "$unretained" "$spin" "${output_hash:-}" | tee -a "$OUT_DIR/results.tsv"
}

printf 'case\ttok_s\tstatus\tram_gb\tdraft\tpipe_workers\tomp_threads\tkmp_blocktime\tmmap\tdirect\tpilot\tpilot_real\tpilot_k\tuntracked\tunretained\tspin\toutput_hash\n' > "$OUT_DIR/results.tsv"

# Stabilize runtime shader compilation and common filesystem metadata. AUTOPIN remains
# disabled during comparison so later candidates do not receive a larger learned hot set.
echo "Warming up Metal and the model..."
TEMP=0 SEED=1 ILI_IGNORE_PROFILE=1 ILI_AUTOPIN=0 ILI_RAM_GB=114 ILI_DRAFT=0 \
ILI_IO_THREADS=8 ILI_OMP_THREADS=auto ILI_KMP_BLOCKTIME=200 \
ILI_MMAP=0 ILI_DIRECT=1 ILI_PILOT=0 ILI_PILOT_REAL=0 \
ILI_UNTRACKED=0 ILI_UNRETAINED=0 ILI_SPIN=0 ILI_NGEN=32 \
  bash ./run-m5max-fast.sh "$MODEL_DIR" run "Warm up the inference engine." \
  > "$OUT_DIR/warmup.log" 2>&1 || true

# Phase 1: resident expert budget and MTP depth.
phase_rate=0
for ram in 110 114 118; do
  # Include plain autoregressive decode. On a memory-bound MoE, speculative
  # verification can lose despite good draft acceptance because it loads the
  # union of experts across the proposed tokens.
  for draft in 0 2 4 6; do
    run_case "core-r${ram}-d${draft}" "$ram" "$draft" 8 auto 200 0 1 0 0 auto 0 0 0
    if [[ "$(is_better "$LAST_RATE" "$phase_rate")" == 1 ]]; then
      phase_rate="$LAST_RATE"; best_ram="$ram"; best_draft="$draft"
    fi
  done
done
best_rate="$phase_rate"

# Phase 2: storage path. MMAP reads expert weights directly from file-backed unified
# memory after CPU pre-touch; DIRECT uses copied slabs. Measure both on this SSD/filesystem.
phase_rate=0
for combo in "0 0" "0 1" "1 0" "1 1"; do
  set -- $combo; mmap="$1"; direct="$2"
  run_case "storage-m${mmap}-d${direct}" "$best_ram" "$best_draft" 8 auto 200 "$mmap" "$direct" 0 0 auto 0 0 0
  if [[ "$(is_better "$LAST_RATE" "$phase_rate")" == 1 ]]; then
    phase_rate="$LAST_RATE"; best_mmap="$mmap"; best_direct="$direct"
  fi
done
[[ "$(is_better "$phase_rate" "$best_rate")" == 1 ]] && best_rate="$phase_rate"

# Phase 3: actual PIPE worker count. The engine reads PIPE_WORKERS, not IO_THREADS.
phase_rate=0
for io in 4 8 12 16; do
  run_case "pipe-${io}" "$best_ram" "$best_draft" "$io" auto 200 "$best_mmap" "$best_direct" 0 0 auto 0 0 0
  if [[ "$(is_better "$LAST_RATE" "$phase_rate")" == 1 ]]; then
    phase_rate="$LAST_RATE"; best_io="$io"
  fi
done
[[ "$(is_better "$phase_rate" "$best_rate")" == 1 ]] && best_rate="$phase_rate"

# Phase 4: router lookahead. Hint-only and real speculative loads can behave very
# differently depending on hit rate and SSD/compute balance.
phase_rate=0
for spec in "0 0 auto" "1 0 8" "1 1 4" "1 1 6" "1 1 8"; do
  set -- $spec; pilot="$1"; preal="$2"; pk="$3"
  run_case "pilot-${pilot}-${preal}-${pk}" "$best_ram" "$best_draft" "$best_io" auto 200 \
    "$best_mmap" "$best_direct" "$pilot" "$preal" "$pk" 0 0 0
  if [[ "$(is_better "$LAST_RATE" "$phase_rate")" == 1 ]]; then
    phase_rate="$LAST_RATE"; best_pilot="$pilot"; best_pilot_real="$preal"; best_pilot_k="$pk"
  fi
done
[[ "$(is_better "$phase_rate" "$best_rate")" == 1 ]] && best_rate="$phase_rate"

# Phase 5: Metal hazard tracking and keep-alive behavior.
phase_rate=0
for untracked in 0 1; do
  for spin in 0 1; do
    run_case "metal-u${untracked}-s${spin}" "$best_ram" "$best_draft" "$best_io" auto 200 \
      "$best_mmap" "$best_direct" "$best_pilot" "$best_pilot_real" "$best_pilot_k" "$untracked" 0 "$spin"
    if [[ "$(is_better "$LAST_RATE" "$phase_rate")" == 1 ]]; then
      phase_rate="$LAST_RATE"; best_untracked="$untracked"; best_spin="$spin"
    fi
  done
done
[[ "$(is_better "$phase_rate" "$best_rate")" == 1 ]] && best_rate="$phase_rate"

# Phase 5b: command-buffer ownership only. All resources are already strongly held by
# registered slabs or persistent scratch, but the generated path stays opt-in until measured.
phase_rate=0
for unretained in 0 1; do
  run_case "metal-unretained-${unretained}" "$best_ram" "$best_draft" "$best_io" auto 200 \
    "$best_mmap" "$best_direct" "$best_pilot" "$best_pilot_real" "$best_pilot_k" \
    "$best_untracked" "$unretained" "$best_spin"
  if [[ "$(is_better "$LAST_RATE" "$phase_rate")" == 1 ]]; then
    phase_rate="$LAST_RATE"; best_unretained="$unretained"
  fi
done
[[ "$(is_better "$phase_rate" "$best_rate")" == 1 ]] && best_rate="$phase_rate"

# Phase 6: Apple performance-core count and LLVM libomp worker parking.
perf_cores="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 12)"
all_cores="$(sysctl -n hw.physicalcpu 2>/dev/null || echo "$perf_cores")"
phase_rate=0
last_omp=""
for omp in "$perf_cores" "$all_cores"; do
  [[ "$omp" == "$last_omp" ]] && continue
  last_omp="$omp"
  for kmp in 0 50 200 infinite; do
    run_case "omp-${omp}-kmp-${kmp}" "$best_ram" "$best_draft" "$best_io" "$omp" "$kmp" \
      "$best_mmap" "$best_direct" "$best_pilot" "$best_pilot_real" "$best_pilot_k" \
      "$best_untracked" "$best_unretained" "$best_spin"
    if [[ "$(is_better "$LAST_RATE" "$phase_rate")" == 1 ]]; then
      phase_rate="$LAST_RATE"; best_omp="$omp"; best_kmp="$kmp"
    fi
  done
done
[[ "$(is_better "$phase_rate" "$best_rate")" == 1 ]] && best_rate="$phase_rate"

cat > "$PROFILE_FILE" <<EOF
# Generated by autotune-m5max.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Parameter expansion preserves any explicit environment override.
: "\${ILI_RAM_GB:=$best_ram}"
: "\${ILI_DRAFT:=$best_draft}"
: "\${ILI_IO_THREADS:=$best_io}"
: "\${ILI_OMP_THREADS:=$best_omp}"
: "\${ILI_KMP_BLOCKTIME:=$best_kmp}"
: "\${ILI_MMAP:=$best_mmap}"
: "\${ILI_DIRECT:=$best_direct}"
: "\${ILI_PILOT:=$best_pilot}"
: "\${ILI_PILOT_REAL:=$best_pilot_real}"
: "\${ILI_PILOT_K:=$best_pilot_k}"
: "\${ILI_UNTRACKED:=$best_untracked}"
: "\${ILI_UNRETAINED:=$best_unretained}"
: "\${ILI_SPIN:=$best_spin}"
EOF

cat <<EOF

Best observed configuration
  tok/s:          $best_rate
  RAM GB:         $best_ram
  MTP draft:      $best_draft
  PIPE workers:   $best_io
  OMP threads:    $best_omp
  KMP blocktime:  $best_kmp
  MMAP / DIRECT:  $best_mmap / $best_direct
  PILOT:          $best_pilot (real=$best_pilot_real, k=$best_pilot_k)
  untracked:      $best_untracked
  unretained:     $best_unretained
  Metal spin:     $best_spin

Saved profile: $PROFILE_FILE
Raw results:  $OUT_DIR/results.tsv
EOF
