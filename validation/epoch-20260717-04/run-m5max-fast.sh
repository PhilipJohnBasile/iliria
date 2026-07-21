#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run-m5max-fast.sh MODEL_DIR [run|chat] [PROMPT...]

Environment:
  ILI_RAM_GB=114
  ILI_WIRED_MB=122000
  ILI_DRAFT=0              # measured fastest on streamed M5 Max MoE; MTP remains opt-in
  ILI_IO_THREADS=8          # maps to the engine's PIPE_WORKERS
  ILI_PIPE=1                # measured +7.9% vs off on the frozen 112-token M5 Max A/B
  ILI_OMP_THREADS=auto
  ILI_KMP_BLOCKTIME=200
  ILI_MMAP=0                # file-backed zero-copy experts; autotune before enabling
  ILI_DIRECT=1
  ILI_PILOT=0               # router-lookahead readahead hints; opt-in
  ILI_PILOT_REAL=0          # real cross-layer expert loads. K6 raises hit rate +13 pts but LOSES
                             # -0.7..-7.6% throughput on the 3-prompt held-out matrix
                             # (bench-m5max/k6-matrix-20260714-090059: attention +5-6s, barrier +1.3-2.3s
                             # outweigh disk -5..-7.7s). REVERT DEFAULT per decision gate.
  ILI_PILOT_K=auto
  ILI_REPIN=0               # live hot-set refresh interval in emitted tokens
  ILI_AUTOPIN=1             # persistent .fa_usage hot-expert learning
  ILI_MLOCK=-1              # auto; macOS wires pinned experts by default
  ILI_PIN_FILE=""            # optional explicit STATS/PIN profile
  ILI_PIN_GB=48
  ILI_TOPP=""                # optional quality/speed trade-off; e.g. 0.7
  ILI_METAL_GEMM_MIN=16
  ILI_METAL_PREFILL=1       # 1 = S>4 prefill attention + projections on GPU (6-9x TTFT,
                             # the prefill-I/O study §7). User ruling 2026-07-14: kernel-family
                             # rounding variance is an acceptable faithful forward (greedy prefill
                             # continuations may fork vs CPU; decode kernels unchanged). Default
                             # stays 0 because the serve-gate turn-2 TTFT came in at 130.4s vs the
                             # <=120s promotion threshold; flip to 1 to take the 6-9x TTFT win.
  ILI_DSA=0                 # DSA lightning-indexer sparse attention (needs out-idx-* weights).
                             # OPT-IN: selection runs CPU-only and bypasses the Metal prefill
                             # kernel, so with ILI_METAL_PREFILL=1 it is ~4.9x SLOWER at 3.8K ctx
                             # (362s vs 1780s wall); crossover vs GPU dense est. ~36K ctx. It also
                             # costs ~-10% decode below 2048 ctx (indexer key upkeep). Only wins
                             # when prefill attention is CPU-bound: -42% wall at 3.8K ctx vs CPU
                             # dense. See docs/roadmap-daily-driver.md DSA campaign annotation.
  ILI_METAL4_MOE=0          # requires make mac-fast-metal4; opt-in until real-model A/B wins
  ILI_UNTRACKED=0
  ILI_UNRETAINED=0          # command buffers do not retain already-owned resources
  ILI_SPIN=0
  ILI_CPU_GROUPED_MOE=1       # grouped ragged CPU fallback; 0 restores per-expert calls
  ILI_CPU_MOE_ALL=0           # diagnostic: compute every routed expert on CPU
  ILI_CPU_MOE_MISSES=0        # 0=Metal, 1=CPU, auto=small tails while resident Metal runs
  ILI_CPU_MOE_MISS_MAX_EXPERTS=2
  ILI_CPU_MOE_MISS_MAX_ROWS=4
  ILI_CPU_MOE_STATS=0         # print grouped/heterogeneous timing counters on exit
  ILI_HIGH_PRIORITY=0
  ILI_NGEN=512
  ILI_PROFILE_FILE=./.m5max-profile.env
  ILI_IGNORE_PROFILE=0
  ILI_EXTRA_ARGS=""
EOF
}

[[ $# -ge 1 ]] || { usage; exit 2; }
MODEL_DIR="${1%/}"
MODE="${2:-chat}"
if [[ $# -ge 2 ]]; then shift 2; else shift 1; fi

case "$MODE" in run|chat) ;; *) echo "error: mode must be run or chat" >&2; exit 2 ;; esac
[[ -d "$MODEL_DIR" ]] || { echo "error: model directory not found: $MODEL_DIR" >&2; exit 1; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[[ -x ./ili ]] || { echo "error: ./ili not found or not executable" >&2; exit 1; }

PROFILE_FILE="${ILI_PROFILE_FILE:-$SCRIPT_DIR/.m5max-profile.env}"
if [[ "${ILI_IGNORE_PROFILE:-0}" != 1 && -f "$PROFILE_FILE" ]]; then
  # Generated profiles use := expansion, so explicit environment values win.
  # shellcheck disable=SC1090
  source "$PROFILE_FILE"
fi

# Silent legacy fallback (ILI_* > COLI_* > FA_*): adopt old spellings from the
# environment or from profiles generated before the rename.
for _legacy in $(compgen -v | grep -E '^(COLI|FA)_' || true); do
  _canon="ILI_${_legacy#*_}"
  if [[ -z "${!_canon+x}" ]]; then eval "$_canon=\${$_legacy}"; fi
done
unset -v _legacy _canon 2>/dev/null || true

DRAFT="${ILI_DRAFT:-0}"
(( DRAFT >= 0 )) || { echo "error: ILI_DRAFT must be >= 0" >&2; exit 2; }

expected_sorted=(1065950496 3527131672 5366238584)
mtp_files=()
while IFS= read -r f; do mtp_files+=("$f"); done < <(find "$MODEL_DIR" -maxdepth 1 -type f -name 'out-mtp-*' -print | sort)
mtp_ok=1
actual_sizes=()
if [[ ${#mtp_files[@]} -ne 3 ]]; then
  mtp_ok=0
else
  for f in "${mtp_files[@]}"; do
    size="$(stat -f '%z' "$f" 2>/dev/null || stat -c '%s' "$f")"
    actual_sizes+=("$size")
  done
  actual_sorted=()
  while IFS= read -r n; do actual_sorted+=("$n"); done < <(printf '%s\n' "${actual_sizes[@]}" | sort -n)
  for i in 0 1 2; do
    [[ "${actual_sorted[$i]}" == "${expected_sorted[$i]}" ]] || mtp_ok=0
  done
fi

if (( DRAFT > 0 )) && [[ "$mtp_ok" -ne 1 ]]; then
  cat >&2 <<EOF
error: correct int8 MTP tensors were not found.
Expected sizes: ${expected_sorted[*]}
Found sizes:    ${actual_sizes[*]:-none}
EOF
  exit 1
fi

RAM_GB="${ILI_RAM_GB:-114}"
WIRED_MB="${ILI_WIRED_MB:-122000}"
IO_THREADS="${ILI_IO_THREADS:-8}"
PIPE_MODE="${ILI_PIPE:-1}"
OMP_THREADS="${ILI_OMP_THREADS:-auto}"
KMP_BLOCKTIME="${ILI_KMP_BLOCKTIME:-200}"
MMAP_MODE="${ILI_MMAP:-0}"
DIRECT_MODE="${ILI_DIRECT:-1}"
PILOT_MODE="${ILI_PILOT:-0}"
PILOT_REAL_MODE="${ILI_PILOT_REAL:-0}"
PILOT_K_MODE="${ILI_PILOT_K:-auto}"
REPIN="${ILI_REPIN:-0}"
AUTOPIN="${ILI_AUTOPIN:-1}"
MLOCK_MODE="${ILI_MLOCK:--1}"
PIN_FILE="${ILI_PIN_FILE:-}"
PIN_GB_VALUE="${ILI_PIN_GB:-48}"
TOPP_VALUE="${ILI_TOPP:-}"
GEMM_MIN="${ILI_METAL_GEMM_MIN:-16}"
METAL_PREFILL="${ILI_METAL_PREFILL:-1}"
DSA_MODE="${ILI_DSA:-0}"
METAL4_MOE="${ILI_METAL4_MOE:-0}"
UNTRACKED="${ILI_UNTRACKED:-0}"
UNRETAINED="${ILI_UNRETAINED:-0}"
SPIN="${ILI_SPIN:-0}"
CPU_GROUPED="${ILI_CPU_GROUPED_MOE:-1}"
CPU_ALL="${ILI_CPU_MOE_ALL:-0}"
CPU_MISSES="${ILI_CPU_MOE_MISSES:-0}"
CPU_MISS_EXPERTS="${ILI_CPU_MOE_MISS_MAX_EXPERTS:-2}"
CPU_MISS_ROWS="${ILI_CPU_MOE_MISS_MAX_ROWS:-4}"
CPU_STATS="${ILI_CPU_MOE_STATS:-0}"
NGEN="${ILI_NGEN:-512}"

(( IO_THREADS >= 1 )) || { echo "error: ILI_IO_THREADS must be >= 1" >&2; exit 2; }
[[ "$PIPE_MODE" == 0 || "$PIPE_MODE" == 1 ]] || { echo "error: ILI_PIPE must be 0 or 1" >&2; exit 2; }
[[ "$METAL4_MOE" == 0 || "$METAL4_MOE" == 1 ]] || { echo "error: ILI_METAL4_MOE must be 0 or 1" >&2; exit 2; }
[[ "$DSA_MODE" == 0 || "$DSA_MODE" == 1 ]] || { echo "error: ILI_DSA must be 0 or 1" >&2; exit 2; }
[[ "$CPU_GROUPED" == 0 || "$CPU_GROUPED" == 1 ]] || { echo "error: ILI_CPU_GROUPED_MOE must be 0 or 1" >&2; exit 2; }
[[ "$CPU_ALL" == 0 || "$CPU_ALL" == 1 ]] || { echo "error: ILI_CPU_MOE_ALL must be 0 or 1" >&2; exit 2; }
[[ "$CPU_STATS" == 0 || "$CPU_STATS" == 1 ]] || { echo "error: ILI_CPU_MOE_STATS must be 0 or 1" >&2; exit 2; }
case "$CPU_MISSES" in 0|1|auto|-1|metal) ;; *) echo "error: ILI_CPU_MOE_MISSES must be 0, 1, or auto" >&2; exit 2 ;; esac
[[ "$CPU_MISS_EXPERTS" =~ ^[0-9]+$ ]] || { echo "error: ILI_CPU_MOE_MISS_MAX_EXPERTS must be non-negative" >&2; exit 2; }
[[ "$CPU_MISS_ROWS" =~ ^[0-9]+$ ]] || { echo "error: ILI_CPU_MOE_MISS_MAX_ROWS must be non-negative" >&2; exit 2; }
(( REPIN >= 0 )) || { echo "error: ILI_REPIN must be >= 0" >&2; exit 2; }
if [[ "$PILOT_K_MODE" != auto ]]; then
  (( PILOT_K_MODE >= 1 )) || { echo "error: ILI_PILOT_K must be auto or >= 1" >&2; exit 2; }
fi
if [[ -n "$PIN_FILE" && ! -f "$PIN_FILE" ]]; then
  echo "error: ILI_PIN_FILE not found: $PIN_FILE" >&2
  exit 1
fi

if [[ "$OMP_THREADS" == auto ]]; then
  OMP_THREADS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo 8)"
fi

current_wired="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || true)"
if [[ -n "$current_wired" && "$current_wired" -lt "$WIRED_MB" ]]; then
  sudo sysctl "iogpu.wired_limit_mb=$WIRED_MB"
fi

if [[ "${ILI_HIGH_PRIORITY:-0}" == 1 ]]; then
  sudo -n renice -n -10 -p $$ >/dev/null 2>&1 || \
    echo "warning: could not raise process priority without an interactive sudo prompt" >&2
fi

ulimit -n 65536 2>/dev/null || true

unset MTP GOMP_SPINCOUNT PIN PIN_GB TOPP PILOT_K
if (( DRAFT == 0 )); then
  # A true autoregressive baseline does not load or reserve the MTP sidecar.
  export MTP=0
fi
export ILI_MODEL="$MODEL_DIR"
export ILI_METAL=1
export ILI_METAL_UNTRACKED="$UNTRACKED"
export ILI_METAL_UNRETAINED="$UNRETAINED"
export ILI_METAL_SPIN="$SPIN"
export ILI_METAL_GEMM_MIN="$GEMM_MIN"
export ILI_METAL_PREFILL="$METAL_PREFILL"
export DSA="$DSA_MODE"
export ILI_METAL4_MOE="$METAL4_MOE"
export ILI_MMAP="$MMAP_MODE"
export DIRECT="$DIRECT_MODE"
export DRAFT="$DRAFT"
export IO_THREADS="$IO_THREADS"       # retained for wrapper compatibility
export PIPE_WORKERS="$IO_THREADS"     # actual C-engine control
export PIPE="$PIPE_MODE"
export PILOT="$PILOT_MODE"
export PILOT_REAL="$PILOT_REAL_MODE"
export REPIN="$REPIN"
export AUTOPIN="$AUTOPIN"
export MLOCK="$MLOCK_MODE"
export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=ACTIVE
export KMP_LIBRARY=throughput
export KMP_BLOCKTIME="$KMP_BLOCKTIME"
export ILI_CPU_GROUPED_MOE="$CPU_GROUPED"
export ILI_CPU_MOE_ALL="$CPU_ALL"
export ILI_CPU_MOE_MISSES="$CPU_MISSES"
export ILI_CPU_MOE_MISS_MAX_EXPERTS="$CPU_MISS_EXPERTS"
export ILI_CPU_MOE_MISS_MAX_ROWS="$CPU_MISS_ROWS"
export ILI_CPU_MOE_STATS="$CPU_STATS"

[[ "$PILOT_K_MODE" == auto ]] || export PILOT_K="$PILOT_K_MODE"
if [[ -n "$PIN_FILE" ]]; then
  export PIN="$PIN_FILE"
  export PIN_GB="$PIN_GB_VALUE"
fi
[[ -z "$TOPP_VALUE" ]] || export TOPP="$TOPP_VALUE"

printf '[m5max] RAM=%sGB DRAFT=%s PIPE=%s/%s OMP=%s KMP=%s MMAP=%s DIRECT=%s PILOT=%s/%s/%s REPIN=%s AUTOPIN=%s METAL4_MOE=%s CPU_MOE=%s/%s/%s(%s,%s) UNTRACKED=%s UNRETAINED=%s SPIN=%s\n' \
  "$RAM_GB" "$DRAFT" "$PIPE_MODE" "$IO_THREADS" "$OMP_THREADS" "$KMP_BLOCKTIME" "$MMAP_MODE" "$DIRECT_MODE" \
  "$PILOT_MODE" "$PILOT_REAL_MODE" "$PILOT_K_MODE" "$REPIN" "$AUTOPIN" "$METAL4_MOE" \
  "$CPU_GROUPED" "$CPU_ALL" "$CPU_MISSES" "$CPU_MISS_EXPERTS" "$CPU_MISS_ROWS" \
  "$UNTRACKED" "$UNRETAINED" "$SPIN" >&2

extra_args=()
read -r -a extra_args <<< "${ILI_EXTRA_ARGS:-}" || true
prefix=()
if command -v caffeinate >/dev/null 2>&1; then
  prefix=(caffeinate -dimsu)
fi

if [[ "$MODE" == chat ]]; then
  if ((${#extra_args[@]})); then
    exec "${prefix[@]}" ./ili chat --model "$MODEL_DIR" --ram "$RAM_GB" "${extra_args[@]}"
  fi
  exec "${prefix[@]}" ./ili chat --model "$MODEL_DIR" --ram "$RAM_GB"
fi

[[ $# -gt 0 ]] || { echo "error: run mode requires a prompt" >&2; exit 2; }
if ((${#extra_args[@]})); then
  exec "${prefix[@]}" ./ili run --model "$MODEL_DIR" --ram "$RAM_GB" --ngen "$NGEN" "$*" "${extra_args[@]}"
fi
exec "${prefix[@]}" ./ili run --model "$MODEL_DIR" --ram "$RAM_GB" --ngen "$NGEN" "$*"
