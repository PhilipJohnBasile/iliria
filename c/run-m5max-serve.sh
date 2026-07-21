#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run-m5max-serve.sh MODEL_DIR [-- extra ili serve args]

Long-lived `ili serve` (OpenAI-compatible API) with the measured M5 Max tuning
from run-m5max-fast.sh plus prefix-KV reuse for agent harnesses.

Agent-harness contract (Step 1 of docs/roadmap-daily-driver.md):
  * keep ONE serve process alive for the whole session;
  * send monotonic message history (append-only: never rewrite old turns);
  * the engine tokenizes each prompt, matches it against the slot history and
    prefills ONLY the delta ([API] KV slot S prefix P/N token, prefill K);
  * KV persists in <MODEL_DIR>/.ili_kv[.N] (KVSAVE=1): restarting the server
    resumes the conversation warm, with no re-prefill.

Response-length discipline (Step 2 of the roadmap):
  * THINK stays OFF (ILI_THINK=0): thinking triples decode time per turn.
  * Send per-request `max_tokens`: ~200 for tool-call/short-answer turns,
    ~400 for code edits, ~700 only for from-scratch generation. The server
    default is 256 when the client sends nothing.
  * System-prompt snippet that measurably shortens replies:
      "Reply with unified diffs or changed hunks only, never full files.
       No preamble, no recap, no restating the request."

Environment (defaults mirror run-m5max-fast.sh unless noted):
  ILI_RAM_GB=114            # engine subtracts the KV pool before sizing the expert cache
  ILI_WIRED_MB=122000
  ILI_DRAFT=0               # MTP measured 3x slower on streamed M5 Max MoE; stays off
  ILI_IO_THREADS=8          # maps to the engine's PIPE_WORKERS
  ILI_PIPE=1
  ILI_OMP_THREADS=auto
  ILI_KMP_BLOCKTIME=200
  ILI_MMAP=0
  ILI_DIRECT=1
  ILI_PILOT=0               # K6 decision gate: PILOT_REAL loses -0.7..-7.6% throughput
  ILI_PILOT_REAL=0
  ILI_REPIN=0
  ILI_AUTOPIN=1
  ILI_MLOCK=-1
  ILI_PIN_FILE=""
  ILI_PIN_GB=48
  ILI_METAL_GEMM_MIN=16
  ILI_METAL_PREFILL=1       # 1 = GPU prefill attention (measured 6-9x TTFT, prefill-io-study
                             # §7; user ruling 2026-07-14 accepts the rounding-variance greedy forks).
                             # Default stays 0: serve-gate turn-2 TTFT was 130.4s vs the <=120s
                             # promotion threshold. Flip to 1 for the TTFT win; 0 = byte-exact CPU.
  ILI_DSA=0                 # DSA sparse attention (out-idx-* weights). OPT-IN: CPU-only selection
                             # bypasses the Metal prefill kernel — ~4.9x slower TTFT at 3.8K ctx
                             # with ILI_METAL_PREFILL=1, ~-10% decode under 2048 ctx. Only wins on
                             # CPU-bound prefill (-42% wall at 3.8K). See roadmap DSA annotation.
  ILI_CPU_GROUPED_MOE=1
  --- serve-specific ---
  ILI_KV_SLOTS=2            # 1-2 ONLY: each 40K-token slot costs ~7.3 GB of expert cache.
                             # 4 slots would steal ~29 GB and crater the hit rate.
  ILI_CTX=40960             # per-slot context (tokens); ~182 KB/token of KV
  ILI_KVSAVE=1              # persist KV to <MODEL_DIR>/.ili_kv[.N] across restarts
  ILI_NGEN=1024             # server-side hard cap on any single response
  ILI_THINK=0               # keep thinking OFF unless a client asks per request
  ILI_HOST=127.0.0.1
  ILI_PORT=8000
  ILI_TEMP=""               # optional server default temperature (API requests override)
  ILI_EXTRA_ARGS=""
EOF
}

[[ $# -ge 1 ]] || { usage; exit 2; }
MODEL_DIR="${1%/}"; shift
if [[ "${1:-}" == "--" ]]; then shift; fi
[[ -d "$MODEL_DIR" ]] || { echo "error: model directory not found: $MODEL_DIR" >&2; exit 1; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[[ -x ./ili ]] || { echo "error: ./ili not found or not executable" >&2; exit 1; }

PROFILE_FILE="${ILI_PROFILE_FILE:-$SCRIPT_DIR/.m5max-profile.env}"
if [[ "${ILI_IGNORE_PROFILE:-0}" != 1 && -f "$PROFILE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$PROFILE_FILE"
fi

# Silent legacy fallback (ILI_* > COLI_* > FA_*), same as run-m5max-fast.sh.
for _legacy in $(compgen -v | grep -E '^(COLI|FA)_' || true); do
  _canon="ILI_${_legacy#*_}"
  if [[ -z "${!_canon+x}" ]]; then eval "$_canon=\${$_legacy}"; fi
done
unset -v _legacy _canon 2>/dev/null || true

RAM_GB="${ILI_RAM_GB:-114}"
WIRED_MB="${ILI_WIRED_MB:-122000}"
DRAFT="${ILI_DRAFT:-0}"
IO_THREADS="${ILI_IO_THREADS:-8}"
PIPE_MODE="${ILI_PIPE:-1}"
OMP_THREADS="${ILI_OMP_THREADS:-auto}"
KMP_BLOCKTIME="${ILI_KMP_BLOCKTIME:-200}"
MMAP_MODE="${ILI_MMAP:-0}"
DIRECT_MODE="${ILI_DIRECT:-1}"
PILOT_MODE="${ILI_PILOT:-0}"
PILOT_REAL_MODE="${ILI_PILOT_REAL:-0}"
REPIN="${ILI_REPIN:-0}"
AUTOPIN="${ILI_AUTOPIN:-1}"
MLOCK_MODE="${ILI_MLOCK:--1}"
PIN_FILE="${ILI_PIN_FILE:-}"
PIN_GB_VALUE="${ILI_PIN_GB:-48}"
GEMM_MIN="${ILI_METAL_GEMM_MIN:-16}"
METAL_PREFILL="${ILI_METAL_PREFILL:-1}"
DSA_MODE="${ILI_DSA:-0}"
CPU_GROUPED="${ILI_CPU_GROUPED_MOE:-1}"
KV_SLOTS="${ILI_KV_SLOTS:-2}"
CTX="${ILI_CTX:-40960}"
KVSAVE_MODE="${ILI_KVSAVE:-1}"
NGEN="${ILI_NGEN:-1024}"
THINK_MODE="${ILI_THINK:-0}"
HOST="${ILI_HOST:-127.0.0.1}"
PORT="${ILI_PORT:-8000}"
TEMP_VALUE="${ILI_TEMP:-}"

(( IO_THREADS >= 1 )) || { echo "error: ILI_IO_THREADS must be >= 1" >&2; exit 2; }
[[ "$PIPE_MODE" == 0 || "$PIPE_MODE" == 1 ]] || { echo "error: ILI_PIPE must be 0 or 1" >&2; exit 2; }
[[ "$KVSAVE_MODE" == 0 || "$KVSAVE_MODE" == 1 ]] || { echo "error: ILI_KVSAVE must be 0 or 1" >&2; exit 2; }
[[ "$DSA_MODE" == 0 || "$DSA_MODE" == 1 ]] || { echo "error: ILI_DSA must be 0 or 1" >&2; exit 2; }
[[ "$THINK_MODE" == 0 || "$THINK_MODE" == 1 ]] || { echo "error: ILI_THINK must be 0 or 1" >&2; exit 2; }
[[ "$KV_SLOTS" =~ ^[0-9]+$ ]] && (( KV_SLOTS >= 1 && KV_SLOTS <= 16 )) || {
  echo "error: ILI_KV_SLOTS must be an integer between 1 and 16" >&2; exit 2; }
if (( KV_SLOTS > 2 )); then
  echo "warning: ILI_KV_SLOTS=$KV_SLOTS — each ${CTX}-token slot reserves ~$(( CTX * 182 / 1000000 )) GB" >&2
  echo "warning: of RAM that would otherwise hold experts. 1-2 slots is the measured sweet spot." >&2
fi
[[ "$CTX" =~ ^[0-9]+$ ]] && (( CTX >= 1024 )) || { echo "error: ILI_CTX must be an integer >= 1024" >&2; exit 2; }
(( NGEN >= 1 )) || { echo "error: ILI_NGEN must be >= 1" >&2; exit 2; }
if [[ -n "$PIN_FILE" && ! -f "$PIN_FILE" ]]; then
  echo "error: ILI_PIN_FILE not found: $PIN_FILE" >&2; exit 1
fi

if [[ "$OMP_THREADS" == auto ]]; then
  OMP_THREADS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo 8)"
fi

current_wired="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || true)"
if [[ -n "$current_wired" && "$current_wired" -lt "$WIRED_MB" ]]; then
  sudo sysctl "iogpu.wired_limit_mb=$WIRED_MB"
fi

ulimit -n 65536 2>/dev/null || true

unset MTP GOMP_SPINCOUNT PIN PIN_GB TOPP PILOT_K
if (( DRAFT == 0 )); then
  export MTP=0            # a true autoregressive baseline never loads the MTP sidecar
fi
export ILI_MODEL="$MODEL_DIR"
export ILI_METAL=1
export ILI_METAL_GEMM_MIN="$GEMM_MIN"
export ILI_METAL_PREFILL="$METAL_PREFILL"
export DSA="$DSA_MODE"
export ILI_MMAP="$MMAP_MODE"
export DIRECT="$DIRECT_MODE"
export DRAFT="$DRAFT"
export IO_THREADS="$IO_THREADS"
export PIPE_WORKERS="$IO_THREADS"
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
export KVSAVE="$KVSAVE_MODE"
export ILI_THINK="$THINK_MODE"
if [[ -n "$PIN_FILE" ]]; then
  export PIN="$PIN_FILE"
  export PIN_GB="$PIN_GB_VALUE"
fi

printf '[m5max-serve] RAM=%sGB CTX=%s KV_SLOTS=%s KVSAVE=%s NGEN=%s THINK=%s DRAFT=%s PIPE=%s/%s OMP=%s MMAP=%s DIRECT=%s PILOT=%s/%s http://%s:%s/v1\n' \
  "$RAM_GB" "$CTX" "$KV_SLOTS" "$KVSAVE_MODE" "$NGEN" "$THINK_MODE" "$DRAFT" \
  "$PIPE_MODE" "$IO_THREADS" "$OMP_THREADS" "$MMAP_MODE" "$DIRECT_MODE" \
  "$PILOT_MODE" "$PILOT_REAL_MODE" "$HOST" "$PORT" >&2

extra_args=()
read -r -a extra_args <<< "${ILI_EXTRA_ARGS:-}" || true
prefix=()
if command -v caffeinate >/dev/null 2>&1; then
  prefix=(caffeinate -dimsu)
fi

serve_args=(serve --model "$MODEL_DIR" --ram "$RAM_GB" --ctx "$CTX"
            --kv-slots "$KV_SLOTS" --ngen "$NGEN" --host "$HOST" --port "$PORT")
[[ -z "$TEMP_VALUE" ]] || serve_args+=(--temp "$TEMP_VALUE")
if ((${#extra_args[@]})); then serve_args+=("${extra_args[@]}"); fi
if (($#)); then serve_args+=("$@"); fi
exec "${prefix[@]}" ./ili "${serve_args[@]}"
