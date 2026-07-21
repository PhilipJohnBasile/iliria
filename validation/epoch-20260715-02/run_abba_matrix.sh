#!/usr/bin/env bash
# The same-commit serve-mode MP=0/1 ABBA gate (queued in bench-m5max/campaign-state.json
# as "same-commit serve MP off/on ABBA matrix", right after the container gates).
#
# Question: at the current commit, in `ili serve` (not the one-shot run-m5max-fast.sh
# path ab-m5max-k6-matrix.sh already covers), how much does ILI_METAL_PREFILL=1 change
# TTFT/prefill/decode over a realistic monotonic-history agent transcript, and does it
# reproduce run-m5max-serve.sh's documented "accepts the rounding-variance greedy forks"
# note (i.e. do MP=0 and MP=1 outputs actually hash-diverge, and is each mode internally
# repeatable)? ABBA order (MP=0, MP=1, MP=1, MP=0) controls for monotonic drift across the
# ~2.5h run the same way ab-m5max-k6-matrix.sh's off/candidate alternation does.
#
# Usage:
#   bash scripts/run_abba_matrix.sh MODEL_DIR
#   bash scripts/run_abba_matrix.sh --dry-run [MODEL_DIR]     # mock engine, no glm, see below
#
# Design (per bench-m5max/campaign-log.md 2026-07-15 11:59:34 prep-agent brief):
#   (a) refuses to start unless scripts/quiesce_check.sh passes (real mode: hard gate;
#       --dry-run: still invoked for real and recorded, but does not block the mock flow --
#       there is no engine at risk in a dry run, and this machine legitimately has the
#       quality bench's `glm` running most of the time, which would otherwise make the
#       dry run untestable on a normal working day).
#   (b) 4 arms in ABBA order (MP=0, MP=1, MP=1, MP=0). Each arm: a FRESH dedicated
#       `ili serve` via run-m5max-serve.sh, ILI_METAL_PREFILL forced per arm, a FIXED
#       ILI_PREFILL_CHUNK (glm.c's own compiled-in default, 2048 -- see PREFILL_CHUNK
#       below) so chunking never confounds the MP=0/1 comparison; .fa_usage[.profile] is
#       snapshotted before and restored after EACH arm (frozen-state convention, read from
#       ab-m5max-k6-matrix.sh); the scripted 3-turn transcript is driven over HTTP by
#       scripts/abba_transcript_driver.py (imports scripts/serve_gate.py's sse_request/
#       load_chunks directly); serve is killed FULLY between arms -- both the python
#       wrapper (ili/openai_server.py) AND the glm child, independently verified, not
#       just signaled once and hoped for (see kill_engine_processes() below and the
#       module docstring of abba_transcript_driver.py for the empirical reason: this
#       machine demonstrated on 2026-07-15 that SIGTERM to a `caffeinate`-wrapped process
#       does NOT propagate to its child, which is left orphaned and still holding the
#       port). Port freedom is verified with lsof before the next arm starts.
#   (c) Per arm: TTFT/prefill-tok-s/decode-tok-s/output-hash per turn (JSON from the
#       driver), quiesce_check output before AND after, all with `$ATTEMPT_ID` embedded
#       in every filename (bench-m5max/campaign-state.json convention: "every result
#       artifact should embed attempt_id").
#   (d) bench-m5max/abba-<date>/results.md: the paired ABBA table (arm1..arm4 side by
#       side per turn) + per-mode medians, written by abba_transcript_driver.py summarize.
#   (e) --dry-run: abba_transcript_driver.py mock-serve stands in for `ili serve`
#       (in-process stub HTTP server, no engine) and forks a decoy child literally named
#       `glm` so kill_engine_processes()'s dual-kill path is genuinely exercised, not
#       assumed. Run with --dry-run to prove the flow before tonight's real window.
#
# Estimated REAL runtime: ~2.5h total (model load + 3-turn transcript, x4 arms). Per-arm
# ETA is printed at matrix start and at each arm's start; override with
# ILI_ABBA_TOTAL_ETA_MIN (default 150).
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CDIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"     # c/ -- engine scripts (run-m5max-serve.sh) live here
cd "$CDIR"

log() { echo "[abba $(date '+%H:%M:%S')] $*"; }
die() { echo "[abba $(date '+%H:%M:%S')] FATAL: $*" >&2; exit 1; }

# timing-run exclusivity (scripts/timing_lock.sh) -- sourcing only defines functions.
source "$SCRIPT_DIR/timing_lock.sh"

# ---- args ----------------------------------------------------------------------------
DRY_RUN=0
MODEL_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --) shift; break ;;
    -*) die "unknown flag: $1" ;;
    *) MODEL_DIR="${1%/}"; shift ;;
  esac
done

[[ "$(uname -s)" == Darwin ]] || die "this matrix targets macOS"

# ---- config (env-tunable; defaults match run-m5max-serve.sh / glm.c) -------------------
HOST="${ILI_ABBA_HOST:-127.0.0.1}"
PORT="${ILI_ABBA_PORT:-8000}"
# glm.c: `int pfc = ili_env("PREFILL_CHUNK") ? atoi(...) : 2048;` -- fixed across every
# arm on purpose, so chunking can never confound the MP=0/1 comparison.
PREFILL_CHUNK="${ILI_ABBA_PREFILL_CHUNK:-2048}"
PROFILE="${ILI_ABBA_PROFILE:-coding}"
MAX_TOKENS="${ILI_ABBA_MAX_TOKENS:-300}"
TURNS=3
READY_TIMEOUT_S="${ILI_ABBA_READY_TIMEOUT_S:-600}"
TEARDOWN_TIMEOUT_S="${ILI_ABBA_TEARDOWN_TIMEOUT_S:-30}"
TOTAL_ETA_MIN="${ILI_ABBA_TOTAL_ETA_MIN:-150}"     # ~2.5h
ARM_ETA_MIN=$((TOTAL_ETA_MIN / 4))
ATTEMPT_ID="${ILI_ABBA_ATTEMPT_ID:-abba-$(date +%Y%m%d)-attempt1}"
OUT="${ILI_ABBA_RESULT_DIR:-$CDIR/bench-m5max/abba-$(date +%Y%m%d-%H%M%S)}"

if [[ "$DRY_RUN" == 1 ]]; then
  if [[ -z "$MODEL_DIR" ]]; then
    MODEL_DIR="$OUT/fake-model"
  fi
  mkdir -p "$MODEL_DIR"
  # Something real for the .fa_usage snapshot/restore machinery to operate on.
  [[ -f "$MODEL_DIR/.fa_usage" ]] || echo "dry-run-fixture" > "$MODEL_DIR/.fa_usage"
  [[ -f "$MODEL_DIR/.fa_usage.$PROFILE" ]] || echo "dry-run-fixture-profile" > "$MODEL_DIR/.fa_usage.$PROFILE"
  # --dry-run never execs a real ./glm (abba_transcript_driver.py mock-serve stands in, see
  # the arm loop below); ./ili is the real, checked-in file that stands in for "what would
  # have run" in the executable-provenance manifest (scripts/provenance.sh).
  PROV_BINARY="$CDIR/ili"
else
  [[ -n "$MODEL_DIR" ]] || die "MODEL_DIR is required (or pass --dry-run)"
  [[ -d "$MODEL_DIR" ]] || die "model directory not found: $MODEL_DIR"
  PROV_BINARY="$CDIR/glm"
fi

mkdir -p "$OUT/logs" "$OUT/artifacts" "$OUT/system"
MANIFEST_TSV="$OUT/manifest.tsv"
printf 'arm\tmode\tmetal_prefill\tjson\tquiesce_before\tquiesce_after\tserve_log\n' > "$MANIFEST_TSV"
log "attempt_id=$ATTEMPT_ID dry_run=$DRY_RUN out=$OUT model_dir=$MODEL_DIR prefill_chunk=$PREFILL_CHUNK port=$PORT"

# ---- provenance manifest (executable-provenance system, scripts/provenance.sh) -------------
# One manifest for the whole matrix (all 4 arms share the same on-disk binary/model dir within
# a single invocation of this script); provenance_compare.py --vary ILI_METAL_PREFILL is the
# gate that gets run ACROSS arms afterward (see tools/provenance_compare.py).
if ! bash "$CDIR/scripts/provenance.sh" --attempt-id "$ATTEMPT_ID" --binary "$PROV_BINARY" \
       --model-dir "$MODEL_DIR" --artifact-dir "$OUT" \
       --quiesce-bin "$SCRIPT_DIR/quiesce_check.sh" \
       --prompt-file "$SCRIPT_DIR/abba_transcript_driver.py"; then
  if [[ "$DRY_RUN" == 1 ]]; then
    log "provenance manifest emission FAILED -- continuing anyway (--dry-run has no engine at risk)"
  else
    die "provenance manifest emission FAILED -- refusing to start a ~2.5h ABBA matrix without a record of what will run"
  fi
fi

# ---- timing-run exclusivity lock (MECHANICAL, not procedural) -------------------------
# Must acquire before the quiesce gate: refuses to start outright (real or --dry-run, no
# soft carve-out) if a competing timing driver already holds this lock -- see
# scripts/timing_lock.sh's header. Released on exit via on_exit() further down; this early
# trap only covers the window until on_exit's own `trap on_exit ...` replaces it.
timing_lock_acquire "$ATTEMPT_ID" || die "another timing driver holds the exclusivity lock ($(timing_lock_path)) -- refusing to start"
# Signal traps MUST exit: bash resumes the script after a signal-trap handler returns, so a
# non-exiting handler here would release the lock in this early window and keep running. Exiting
# from the signal fires the EXIT trap so release happens once. (on_exit below already exits.)
trap 'timing_lock_release' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- quiesce gate (design item a) ------------------------------------------------------
run_quiesce() {
  local label="$1" outfile="$2" rc
  # NOTE: no `set -e` in this script (baseline is `set -uo pipefail`, matching
  # run_container_gates.sh), so there is nothing to suspend/restore here -- an earlier
  # draft copied ab-m5max-k6-matrix.sh's set-+e-then-set--e convention verbatim, which
  # for THAT script (baseline `set -euo pipefail`) correctly restores errexit, but here
  # it incorrectly turned errexit ON for the rest of the script, causing it to exit
  # silently the next time any checked command (e.g. `lsof` finding no listener, which
  # legitimately returns non-zero) failed -- caught by the --dry-run proof run.
  bash "$SCRIPT_DIR/quiesce_check.sh" > "$outfile" 2>&1
  rc=$?
  cat "$outfile"
  return "$rc"
}

QUIESCE_PRE="$OUT/system/${ATTEMPT_ID}-preflight-quiesce.txt"
if run_quiesce preflight "$QUIESCE_PRE"; then
  log "preflight quiesce: PASS"
else
  if [[ "$DRY_RUN" == 1 ]]; then
    log "preflight quiesce: NOT QUIESCED -- continuing anyway (--dry-run has no engine at " \
        "risk; this is expected whenever the quality bench's glm is running, see " \
        "campaign-state.json)"
  else
    die "preflight quiesce FAILED -- refusing to start (see $QUIESCE_PRE). This is " \
        "correct/expected while another engine phase owns glm/the SSD; re-run once " \
        "campaign-state.json's queued_after_gates item is actually up."
  fi
fi

# ---- process-tree helpers (design item b: kill BOTH wrapper and glm child) ------------
glm_pids() { pgrep -x glm 2>/dev/null | sort; }

port_listener_pid() { lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1; }

port_is_free() { [[ -z "$(port_listener_pid)" ]]; }

pid_alive() { kill -0 "$1" 2>/dev/null; }

# Kill everything associated with one arm's serve process, independently verifying the
# python wrapper AND the glm child are both gone (never trust a single signal to the
# top-level PID: this machine demonstrated on 2026-07-15 that SIGTERM to a caffeinate-
# wrapped process does not propagate to its child, which survives, reparented to
# launchd, and keeps the port bound). Hard-aborts the whole matrix if teardown cannot be
# verified within TEARDOWN_TIMEOUT_S -- an unkillable leftover engine process must stop
# the campaign, not silently roll into the next arm.
kill_engine_processes() {
  local wrapper_pid="$1" new_glm_pids="$2" deadline waited=0
  local listener
  listener="$(port_listener_pid)"
  log "teardown: wrapper_pid=$wrapper_pid listener_pid=${listener:-none} " \
      "tracked_glm_pids=[${new_glm_pids:-none}]"

  # Signal each distinct PID exactly once: listener and wrapper_pid are frequently the
  # SAME process (the wrapper is what actually binds the port), and a second SIGTERM
  # landing while a naive handler is still mid-cleanup can crash it outright -- observed
  # empirically against abba_transcript_driver.py's mock (a reentrant `print` inside the
  # signal handler), which orphaned its decoy child exactly the way this whole function
  # exists to guard against. One signal per PID, always.
  [[ -n "$listener" ]] && kill -TERM "$listener" 2>/dev/null
  if [[ "$wrapper_pid" != "$listener" ]] && pid_alive "$wrapper_pid"; then
    kill -TERM "$wrapper_pid" 2>/dev/null
  fi

  deadline=$((TEARDOWN_TIMEOUT_S / 2))
  while (( waited < deadline )); do
    port_is_free && ! { for p in $new_glm_pids; do pid_alive "$p" && echo alive; done | grep -q alive; } && break
    sleep 1; waited=$((waited + 1))
  done

  # Fallback: anything still alive gets TERM then KILL directly, no more waiting politely.
  for p in $new_glm_pids; do
    if pid_alive "$p"; then
      log "teardown: glm pid $p survived the graceful path, forcing it down"
      kill -TERM "$p" 2>/dev/null; sleep 1
      pid_alive "$p" && kill -KILL "$p" 2>/dev/null
    fi
  done
  listener="$(port_listener_pid)"
  if [[ -n "$listener" ]]; then
    log "teardown: port $PORT still held by pid $listener, forcing it down"
    kill -KILL "$listener" 2>/dev/null
  fi
  if [[ "$wrapper_pid" != "$listener" ]] && pid_alive "$wrapper_pid"; then
    kill -KILL "$wrapper_pid" 2>/dev/null
  fi

  sleep 1
  local survivors=()
  for p in $new_glm_pids; do pid_alive "$p" && survivors+=("$p"); done
  if ! port_is_free || (( ${#survivors[@]} )); then
    ps -axo pid,ppid,command | grep -E "glm|ili|caffeinate" | grep -v grep >&2 || true
    die "teardown could not verify a clean state (port_free=$(port_is_free && echo yes || echo no), " \
        "surviving tracked glm pids=[${survivors[*]:-none}]) -- aborting the matrix rather than " \
        "risk racing the next arm against a leaked engine process"
  fi
  log "teardown: verified clean (port $PORT free, no tracked glm pids alive)"
}

# ---- .fa_usage frozen-state snapshot/restore (per arm; convention read from
#      ab-m5max-k6-matrix.sh) -----------------------------------------------------------
USAGE_FILES=()
snapshot_usage() {
  local snap_dir="$1" f
  mkdir -p "$snap_dir"
  USAGE_FILES=()
  for f in "$MODEL_DIR/.fa_usage" "$MODEL_DIR/.fa_usage.$PROFILE"; do
    if [[ -f "$f" ]]; then
      cp "$f" "$snap_dir/$(basename "$f")"
      USAGE_FILES+=("$f")
    fi
  done
}
restore_usage() {
  local snap_dir="$1" f
  (( ${#USAGE_FILES[@]} )) || return 0
  for f in "${USAGE_FILES[@]}"; do
    cp "$snap_dir/$(basename "$f")" "$f"
  done
}

# Best-effort safety net: if this script dies mid-arm, do not leave a serve process or a
# mutated .fa_usage behind.
CURRENT_WRAPPER_PID=""
CURRENT_NEW_GLM_PIDS=""
CURRENT_SNAP_DIR=""
on_exit() {
  local rc=$?
  if [[ -n "$CURRENT_WRAPPER_PID" ]]; then
    log "exit trap: cleaning up in-flight arm (rc=$rc)"
    kill_engine_processes "$CURRENT_WRAPPER_PID" "$CURRENT_NEW_GLM_PIDS" || true
  fi
  [[ -n "$CURRENT_SNAP_DIR" ]] && restore_usage "$CURRENT_SNAP_DIR"
  timing_lock_release
  exit "$rc"
}
trap on_exit EXIT INT TERM

# ---- readiness poll --------------------------------------------------------------------
wait_ready() {
  local wrapper_pid="$1" deadline=$((SECONDS + READY_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if ! pid_alive "$wrapper_pid"; then
      return 1
    fi
    if curl -sf -m 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

eta_str() { date -r "$1" '+%H:%M:%S' 2>/dev/null || date '+%H:%M:%S'; }

MATRIX_START=$SECONDS
MATRIX_START_EPOCH=$(date +%s)
log "4 arms, ~${ARM_ETA_MIN}min/arm, estimated total ~${TOTAL_ETA_MIN}min " \
    "(matrix ETA ~$(eta_str $((MATRIX_START_EPOCH + TOTAL_ETA_MIN * 60))))"

# ABBA order: arm 1=MP0, 2=MP1, 3=MP1, 4=MP0
ARMS_MP=(0 1 1 0)

for i in 1 2 3 4; do
  mp="${ARMS_MP[$((i-1))]}"
  arm_label="mp${mp}"
  arm_start_epoch=$(date +%s)
  log "arm $i/4 mode=MP=$mp starting; arm ETA ~$(eta_str $((arm_start_epoch + ARM_ETA_MIN * 60))), " \
      "matrix ETA ~$(eta_str $((MATRIX_START_EPOCH + TOTAL_ETA_MIN * 60)))"

  QUIESCE_BEFORE="$OUT/system/${ATTEMPT_ID}-arm${i}-${arm_label}-quiesce-before.txt"
  if run_quiesce "arm$i-before" "$QUIESCE_BEFORE"; then
    log "arm $i quiesce (before): PASS"
  else
    if [[ "$DRY_RUN" == 1 ]]; then
      log "arm $i quiesce (before): NOT QUIESCED -- continuing (--dry-run)"
    else
      die "arm $i quiesce (before) FAILED -- aborting rather than start a timing arm on a non-quiet machine"
    fi
  fi

  SNAP_DIR="$OUT/system/${ATTEMPT_ID}-arm${i}-usage-snapshot"
  snapshot_usage "$SNAP_DIR"
  CURRENT_SNAP_DIR="$SNAP_DIR"

  BEFORE_GLM="$(glm_pids)"
  if [[ "$DRY_RUN" == 0 && -n "$BEFORE_GLM" ]]; then
    die "arm $i: a glm process is already running before this arm even started " \
        "(pids=[$BEFORE_GLM]) -- a previous arm's teardown must have leaked. Aborting."
  fi

  SERVE_LOG="$OUT/logs/${ATTEMPT_ID}-arm${i}-${arm_label}.serve.log"
  if [[ "$DRY_RUN" == 1 ]]; then
    # Build the command as an array that always has >=1 element (macOS ships bash 3.2,
    # where "${empty_array[@]}" throws "unbound variable" under `set -u` -- see
    # run-m5max-serve.sh's own prefix=()/extra_args guard convention -- rather than
    # conditionally appending an always-declared-but-sometimes-empty array).
    # --decoy-lifetime-s bounds how long a leaked decoy (this arm's teardown fails AND
    # this whole matrix script dies before the exit trap can clean up) can pollute
    # `pgrep -x glm` for every OTHER caller of that check on the machine (quiesce_check.sh,
    # other agents' engine-ownership gates): a few minutes, not the driver's own 15-min
    # default, since a --dry-run arm never legitimately needs the decoy this long.
    mock_cmd=(python3 "$SCRIPT_DIR/abba_transcript_driver.py" mock-serve
              --host "$HOST" --port "$PORT" --serve-log "$SERVE_LOG" --metal-prefill "$mp"
              --work-dir "$OUT/artifacts" --decoy-lifetime-s 180)
    # Fault-inject the orphan-wrapper risk on exactly one arm so the dry run PROVES the
    # independent glm-kill fallback path, not just the happy path where the wrapper
    # cleans up on its own.
    [[ "$i" == 2 ]] && mock_cmd+=(--orphan-on-term)
    "${mock_cmd[@]}" > "$OUT/logs/${ATTEMPT_ID}-arm${i}-${arm_label}.mock.out" 2>&1 &
    WRAPPER_PID=$!
  else
    ILI_METAL_PREFILL="$mp" ILI_PREFILL_CHUNK="$PREFILL_CHUNK" \
    ILI_HOST="$HOST" ILI_PORT="$PORT" \
      bash "$CDIR/run-m5max-serve.sh" "$MODEL_DIR" > "$SERVE_LOG" 2>&1 &
    WRAPPER_PID=$!
  fi
  CURRENT_WRAPPER_PID="$WRAPPER_PID"
  log "arm $i: wrapper pid=$WRAPPER_PID, waiting for http://$HOST:$PORT/health (timeout ${READY_TIMEOUT_S}s)"

  if ! wait_ready "$WRAPPER_PID"; then
    log "arm $i: FAILED to become ready" >&2
    tail -n 60 "$SERVE_LOG" >&2 || true
    kill_engine_processes "$WRAPPER_PID" "$(comm -13 <(echo "$BEFORE_GLM") <(glm_pids))"
    CURRENT_WRAPPER_PID=""; CURRENT_NEW_GLM_PIDS=""
    die "arm $i never became ready within ${READY_TIMEOUT_S}s -- see $SERVE_LOG"
  fi
  AFTER_GLM="$(glm_pids)"
  NEW_GLM="$(comm -13 <(echo "$BEFORE_GLM") <(echo "$AFTER_GLM"))"
  CURRENT_NEW_GLM_PIDS="$NEW_GLM"
  log "arm $i: ready. tracked glm pid(s)=[${NEW_GLM:-none}]"

  ARM_JSON="$OUT/artifacts/${ATTEMPT_ID}-arm${i}-${arm_label}.json"
  python3 "$SCRIPT_DIR/abba_transcript_driver.py" drive \
    --host "$HOST" --port "$PORT" --serve-log "$SERVE_LOG" --turns "$TURNS" \
    --max-tokens "$MAX_TOKENS" --out "$ARM_JSON"
  DRIVE_RC=$?
  if (( DRIVE_RC != 0 )); then
    kill_engine_processes "$WRAPPER_PID" "$NEW_GLM"
    CURRENT_WRAPPER_PID=""; CURRENT_NEW_GLM_PIDS=""
    die "arm $i: transcript driver failed (rc=$DRIVE_RC) -- see $SERVE_LOG"
  fi

  kill_engine_processes "$WRAPPER_PID" "$NEW_GLM"
  CURRENT_WRAPPER_PID=""
  CURRENT_NEW_GLM_PIDS=""

  QUIESCE_AFTER="$OUT/system/${ATTEMPT_ID}-arm${i}-${arm_label}-quiesce-after.txt"
  run_quiesce "arm$i-after" "$QUIESCE_AFTER" || log "arm $i quiesce (after): NOT QUIESCED (recorded, non-gating)"

  restore_usage "$SNAP_DIR"
  CURRENT_SNAP_DIR=""

  printf '%s\tMP=%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$mp" "$mp" \
    "artifacts/${ATTEMPT_ID}-arm${i}-${arm_label}.json" \
    "system/${ATTEMPT_ID}-arm${i}-${arm_label}-quiesce-before.txt" \
    "system/${ATTEMPT_ID}-arm${i}-${arm_label}-quiesce-after.txt" \
    "logs/${ATTEMPT_ID}-arm${i}-${arm_label}.serve.log" >> "$MANIFEST_TSV"
  log "arm $i/4 complete"
done

python3 - "$MANIFEST_TSV" "$OUT/manifest.json" "$ATTEMPT_ID" "$PREFILL_CHUNK" "$DRY_RUN" <<'PY'
import csv, json, sys
src, dst, attempt_id, prefill_chunk, dry_run = sys.argv[1:]
with open(src, newline='') as f:
    runs = list(csv.DictReader(f, delimiter='\t'))
for r in runs:
    r['arm'] = int(r['arm'])
    r['metal_prefill'] = int(r['metal_prefill'])
with open(dst, 'w') as f:
    json.dump({'attempt_id': attempt_id, 'prefill_chunk': int(prefill_chunk),
               'dry_run': bool(int(dry_run)), 'arms': runs}, f, indent=2)
PY

RESULTS_MD="$OUT/results.md"
python3 "$SCRIPT_DIR/abba_transcript_driver.py" summarize \
  --manifest "$OUT/manifest.json" --result-dir "$OUT" --out "$RESULTS_MD"

log "matrix complete: $RESULTS_MD"
