#!/usr/bin/env bash
# mode15_failloud_check.sh -- pre-merge verification-gate check for the
# fail-loud hardening half of commit d1aaa7e ("mode15: fix cap-sizing OOM
# root cause + fail-loud hardening for the multi-turn silent-death bug").
#
# The original bug report's most alarming symptom was not the OOM itself but
# that it was SILENT: "no FATAL, no crash report, no OOM/jetsam trace, the
# process just stops". The fix adds two independent layers of defense:
#   1. glm.c: ili_install_fatal_handlers() -- a sigaction-based handler for
#      SIGSEGV/SIGBUS/SIGABRT/SIGILL/SIGFPE that writes ONE precomputed-length
#      diagnostic line before restoring the default disposition and
#      re-raising. Does NOT and CANNOT cover SIGKILL (uncatchable by any
#      userspace handler, by POSIX design) -- a real kernel OOM-kill is always
#      SIGKILL.
#   2. ili (the Python wrapper): _report_engine_death(), now called at BOTH
#      of cmd_chat's stdout-EOF sites (initial load AND every later turn),
#      prints the real exit status ("killed by signal N" / "exited with code
#      N") plus the tail of the engine's own captured stderr -- this is what
#      makes even a SIGKILL death visible, since it observes the child's exit
#      status rather than trying to catch a signal the child itself cannot.
#
# This script exercises BOTH layers safely, at a tiny/cheap scale, using two
# mechanisms deliberately chosen because the obvious one does not work here:
# `ulimit -v` (RLIMIT_AS) is NOT settable on this machine's macOS/zsh --
# confirmed empirically ("setrlimit failed: invalid argument") -- so instead:
#
#   check=sigkill  Launch a real chat session against the given container and
#                  supervise the child glm process's OWN rss via `ps`,
#                  in a loop; the INSTANT its RSS crosses a tiny threshold
#                  (default 2 GB -- trips within seconds, long before any
#                  real budget ceiling), send it SIGKILL directly. This is
#                  scoped to ONLY that one child PID (never touches any other
#                  process) and reproduces the EXACT signal a real kernel
#                  OOM-kill sends -- arguably more faithful than any
#                  alternative available on this platform. Expected result:
#                  ili prints "[engine terminated] (killed by signal 9)"
#                  (SIGSEGV/etc.'s handler cannot fire -- SIGKILL is
#                  uncatchable, exactly as documented -- so no "FATAL:..."
#                  line from the engine itself is expected here).
#
#   check=sigsegv  Launch the same way, but as soon as the child glm pid is
#                  found, send it an EXTERNAL SIGSEGV via `kill`. A
#                  sigaction-registered handler fires for a signal regardless
#                  of whether it was self-triggered (a real fault) or
#                  externally delivered -- only SIGKILL/SIGSTOP are exempt --
#                  so this safely exercises ili_install_fatal_handlers()
#                  without needing to engineer an actual memory fault.
#                  Expected result: the engine's OWN stderr gets one line
#                  ("FATAL: iliria engine crashed: SIGSEGV (segmentation
#                  fault)") before it dies, and ili reports
#                  "[engine terminated] (killed by signal 11)" with that line
#                  shown as the captured stderr tail -- i.e. the full
#                  defense-in-depth chain, not just one layer of it.
#
# Usage:
#   mode15_failloud_check.sh sigkill MODEL_DIR [OUTDIR] [KILL_THRESHOLD_KB]
#   mode15_failloud_check.sh sigsegv MODEL_DIR [OUTDIR]
set -uo pipefail
CDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CDIR"

MODE="${1:?usage: mode15_failloud_check.sh sigkill|sigsegv MODEL_DIR [OUTDIR] [KILL_THRESHOLD_KB]}"
MODEL_DIR="${2:?usage: mode15_failloud_check.sh sigkill|sigsegv MODEL_DIR [OUTDIR] [KILL_THRESHOLD_KB]}"
OUTDIR="${3:-$CDIR/bench-m5max/mode15-failloud-$MODE-$(date +%Y%m%d-%H%M%S)}"
KILL_THRESHOLD_KB="${4:-$((2*1024*1024))}"   # 2 GB default -- trips fast, on purpose
mkdir -p "$OUTDIR"
LOG="$OUTDIR/failloud_$MODE.log"

log() { echo "[mode15-failloud $(date '+%H:%M:%S')] $*" >&2; }

[[ -x ./glm ]] || { echo "error: ./glm not built. Run: make MODE15=1 glm" >&2; exit 1; }
if pgrep -f "$CDIR/glm" >/dev/null 2>&1; then
  echo "error: an iliria engine process is already running under this worktree -- refusing to start a second heavy run" >&2
  exit 1
fi

PROMPT_FIFO=$(mktemp -u)
mkfifo "$PROMPT_FIFO"
# Keep the FIFO's write end open past the single prompt so the wrapper
# doesn't see early EOF (and exit on its own) before we get a chance to act.
( printf 'What is 2+2? Answer with just the number.\n'; sleep 60 ) >"$PROMPT_FIFO" &
FEEDER_PID=$!

SEED=1 AUTOPIN=0 REPIN=0 DRAFT=0 MTP=0 DIRECT=1 \
  ./ili chat --model "$MODEL_DIR" --ram 40 --temp 0 --ngen 4 \
  <"$PROMPT_FIFO" >"$LOG" 2>&1 &
ILI_PID=$!
log "ili wrapper pid=$ILI_PID"

GLM_PID=""
for i in $(seq 1 300); do
  GLM_PID=$(pgrep -f "$CDIR/glm" | head -1)
  [[ -n "$GLM_PID" ]] && break
  kill -0 "$ILI_PID" 2>/dev/null || { log "ili wrapper exited before glm child appeared"; break; }
  sleep 0.2
done

if [[ -z "$GLM_PID" ]]; then
  echo "RESULT: never found glm child pid -- cannot exercise fail-loud path this way" >&2
  kill "$FEEDER_PID" 2>/dev/null; rm -f "$PROMPT_FIFO"
  wait "$ILI_PID" 2>/dev/null
  exit 1
fi

case "$MODE" in
  sigkill)
    log "found glm child pid=$GLM_PID, watching RSS (kill threshold ${KILL_THRESHOLD_KB} KB)..."
    while kill -0 "$GLM_PID" 2>/dev/null; do
      RSS_KB=$(ps -o rss= -p "$GLM_PID" 2>/dev/null | tr -d ' ')
      [[ -z "$RSS_KB" ]] && { log "glm child exited on its own before threshold was crossed"; break; }
      if (( RSS_KB > KILL_THRESHOLD_KB )); then
        log "RSS ${RSS_KB} KB crossed threshold ${KILL_THRESHOLD_KB} KB -- SIGKILL $GLM_PID now"
        kill -KILL "$GLM_PID"
        break
      fi
      sleep 0.1
    done
    ;;
  sigsegv)
    log "found glm child pid=$GLM_PID -- sending external SIGSEGV now"
    kill -SEGV "$GLM_PID"
    ;;
  *)
    echo "unknown mode: $MODE (want sigkill or sigsegv)" >&2
    kill "$FEEDER_PID" 2>/dev/null; rm -f "$PROMPT_FIFO"; wait "$ILI_PID" 2>/dev/null
    exit 2
    ;;
esac

kill "$FEEDER_PID" 2>/dev/null
rm -f "$PROMPT_FIFO"
wait "$ILI_PID" 2>/dev/null
ILI_RC=$?
log "ili wrapper exit code: $ILI_RC"

echo "" >&2
if grep -q "\[engine terminated\]" "$LOG"; then
  log "PASS: wrapper reported the death instead of going silent:"
  grep -A4 "\[engine terminated\]" "$LOG" | sed 's/^/  /' >&2
else
  log "FAIL: no '[engine terminated]' line found in $LOG -- death was NOT surfaced"
fi
log "raw log: $LOG"
