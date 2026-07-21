#!/usr/bin/env bash
# The unattended overnight driver for tonight's engine sequence. Watches the already-armed
# container gates chain (scripts/run_container_gates.sh, itself watching the n=100 int4
# reference bench), reads gate A's verdict, and branches:
#
#   gate A PASSED  -> B1 (the mixed-container n=100 bench, gate B) is already running inside
#                     that same chain process. Nothing else to do tonight; log why and exit.
#   gate A FAILED  -> registered prediction (bench-m5max/campaign-state.json's
#                     gate_b_comparator_correction + campaign-log.md 09:08:49 entry). Runs the
#                     recovery sequence: (a) FAST numerical CPU-vs-Metal-prefill capture grid
#                     (scripts/capture_layer_outputs.md/.patch, T<=4K subset) -> (b) quiesce
#                     retries -> (c) same-commit ABBA matrix (scripts/run_abba_matrix.sh) ->
#                     (d) launch B0, the int4 CPU-routed n=100 comparator bench, detached ->
#                     (e) write the shard-reader marker file.
#
# Fail-closed: any failure in (a)/(b)/(c) skips the REST of (a)/(b)/(c) but (d) and (e) always
# still run on the gate-A-failed path -- B0 is the durable same-backend comparator this
# container's quality verdict depends on and must not be held hostage by a numerical-
# validation or timing-gate hiccup. An indeterminate gate-A verdict (gates.log has neither a
# PASS nor a FAIL marker -- e.g. the chain aborted before gate A even ran) takes NEITHER
# branch: fail closed, log clearly, stop.
#
# Every step appends one factual, timestamped line to bench-m5max/campaign-log.md and
# atomically rewrites bench-m5max/evening-status.json ({step, status, started, finished,
# attempt_id}; status vocabulary matches campaign-state.json's own
# running/passed/failed/timed_out/cancelled).
#
# Usage: bash scripts/evening_orchestrator.sh [--dry-run]
#   --dry-run: never runs a real engine command (no `make`, no `ili run/serve/bench`, no
#   `glm`). The capture step's `git apply`/`git checkout` on glm.c still run for real (git-only,
#   reversible, no engine); the FAST capture grid runs scripts/run_layer_capture_grid.py
#   --dry-run (synthesizes capture files, no serve); the ABBA step passes --dry-run through to
#   run_abba_matrix.sh (which already mocks its own serve, see that script's header); B0 is
#   replaced by a harmless background command. This mirrors run_abba_matrix.sh's own
#   --dry-run convention exactly (same script, same idea, one level up the call chain).
#
# Nearly everything below is overridable via ILI_EVENING_* environment variables so
# c/tests/test_evening_orchestrator.py can drive every branch (both gate-A verdicts, an
# indeterminate verdict, a quiesce-retries-exhausted failure, an ABBA failure) deterministically
# and in seconds, without touching real system state, the real campaign-log.md/evening-
# status.json, or the real /tmp marker path. See the variable block below for the full list.
set -uo pipefail   # NOT -e: see run_abba_matrix.sh's own comment on why this codebase's
                   # engine-sequencing scripts manage control flow explicitly via return codes
                   # rather than errexit (a copied set-e/set+e pair once silently broke that
                   # script's dry-run after just one arm).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CDIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/evening_orchestrator.sh [--dry-run]

Environment (all optional; defaults target tonight's real run):
  ILI_EVENING_ATTEMPT_ID           default: evening-<timestamp>
  ILI_EVENING_CAMPAIGN_LOG         default: bench-m5max/campaign-log.md
  ILI_EVENING_STATUS_FILE          default: bench-m5max/evening-status.json
  ILI_EVENING_OUT                  default: bench-m5max/evening-<attempt_id> (this run's own
                                     scratch dir: quiesce attempt logs, capture grid output,
                                     ABBA output, build logs)
  ILI_EVENING_GATES_PID            default: auto-discovered via `pgrep -f run_container_gates.sh`
  ILI_EVENING_GATES_LOG            default: bench-m5max/container-20260715/gates.log (matches
                                     run_container_gates.sh's own hardcoded $OUT for tonight)
  ILI_EVENING_GATES_POLL_S         default: 120 (matches run_container_gates.sh's own poll cadence)
  ILI_EVENING_QUIESCE_BIN          default: scripts/quiesce_check.sh
  ILI_EVENING_QUIESCE_RETRIES      default: 6
  ILI_EVENING_QUIESCE_INTERVAL_S   default: 600 (10 min)
  ILI_EVENING_ABBA_BIN             default: scripts/run_abba_matrix.sh
  ILI_EVENING_ABBA_MODEL_DIR       default: the int4 reference container (ABBA and the capture
                                     grid both investigate general engine/kernel questions, NOT
                                     the mixed container gate A tested -- see capture_layer_
                                     outputs.md's own scope note -- and the int4 reference sidesteps
                                     the mixed container's known Metal-mixed-format MoE hazard
                                     entirely; override if a different target is intended)
  ILI_EVENING_CAPTURE_GRID_BIN     default: scripts/run_layer_capture_grid.py
  ILI_EVENING_CAPTURE_PATCH        default: scripts/capture_layer_outputs.patch
  ILI_EVENING_CAPTURE_MODEL_DIR    default: the int4 reference container (see above)
  ILI_EVENING_CAPTURE_LAYERS       default: 5,39,74
  ILI_EVENING_CAPTURE_S_VALUES     default: 1,4,16,64,256
  ILI_EVENING_CAPTURE_T_VALUES     default: 128,1024,4096  (the FAST, T<=4K subset)
  ILI_EVENING_CAPTURE_HOST/_PORT   default: 127.0.0.1 / 8000
  ILI_EVENING_STATE_JSON           default: bench-m5max/campaign-state.json (source of
                                     gate_b_comparator_correction.b0_command)
  ILI_EVENING_B0_COMMAND           override: use this exact command instead of reading it from
                                     ILI_EVENING_STATE_JSON
  ILI_EVENING_MOCK_B0_COMMAND      default: `nohup sleep 2 >/dev/null 2>&1 &` -- what --dry-run
                                     launches instead of the real (resolved either way, for real)
                                     B0 command
  ILI_EVENING_MARKER_FILE          default: /tmp/iliria-evening-marker-shard-reads-ok
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# ---- configuration (env-tunable; see usage() above) --------------------------------------
BENCH_DIR="$CDIR/bench-m5max"
ATTEMPT_ID="${ILI_EVENING_ATTEMPT_ID:-evening-$(date +%Y%m%d-%H%M%S)}"
CAMPAIGN_LOG="${ILI_EVENING_CAMPAIGN_LOG:-$BENCH_DIR/campaign-log.md}"
STATUS_FILE="${ILI_EVENING_STATUS_FILE:-$BENCH_DIR/evening-status.json}"
EVENING_OUT="${ILI_EVENING_OUT:-$BENCH_DIR/$ATTEMPT_ID}"

GATES_PID="${ILI_EVENING_GATES_PID:-}"
GATES_LOG="${ILI_EVENING_GATES_LOG:-$BENCH_DIR/container-20260715/gates.log}"
GATES_POLL_S="${ILI_EVENING_GATES_POLL_S:-120}"

QUIESCE_BIN="${ILI_EVENING_QUIESCE_BIN:-$SCRIPT_DIR/quiesce_check.sh}"
QUIESCE_RETRIES="${ILI_EVENING_QUIESCE_RETRIES:-6}"
QUIESCE_INTERVAL_S="${ILI_EVENING_QUIESCE_INTERVAL_S:-600}"
QUIESCE_OUT_DIR="$EVENING_OUT/quiesce"

ABBA_BIN="${ILI_EVENING_ABBA_BIN:-$SCRIPT_DIR/run_abba_matrix.sh}"
ABBA_MODEL_DIR="${ILI_EVENING_ABBA_MODEL_DIR:-$HOME/models/GLM-5.2-int4-with-int8-mtp}"
ABBA_OUT="$EVENING_OUT/abba"
ABBA_LOG="$EVENING_OUT/abba.log"

CAPTURE_GRID_BIN="${ILI_EVENING_CAPTURE_GRID_BIN:-$SCRIPT_DIR/run_layer_capture_grid.py}"
CAPTURE_PATCH="${ILI_EVENING_CAPTURE_PATCH:-$SCRIPT_DIR/capture_layer_outputs.patch}"
CAPTURE_MODEL_DIR="${ILI_EVENING_CAPTURE_MODEL_DIR:-$HOME/models/GLM-5.2-int4-with-int8-mtp}"
CAPTURE_LAYERS="${ILI_EVENING_CAPTURE_LAYERS:-5,39,74}"
CAPTURE_S_VALUES="${ILI_EVENING_CAPTURE_S_VALUES:-1,4,16,64,256}"
CAPTURE_T_VALUES="${ILI_EVENING_CAPTURE_T_VALUES:-128,1024,4096}"
CAPTURE_HOST="${ILI_EVENING_CAPTURE_HOST:-127.0.0.1}"
CAPTURE_PORT="${ILI_EVENING_CAPTURE_PORT:-8000}"
CAPTURE_READY_TIMEOUT_S="${ILI_EVENING_CAPTURE_READY_TIMEOUT_S:-600}"
CAPTURE_TEARDOWN_TIMEOUT_S="${ILI_EVENING_CAPTURE_TEARDOWN_TIMEOUT_S:-30}"
CAPTURE_OUT="$EVENING_OUT/layer-capture"

STATE_JSON="${ILI_EVENING_STATE_JSON:-$BENCH_DIR/campaign-state.json}"
B0_COMMAND_OVERRIDE="${ILI_EVENING_B0_COMMAND:-}"
MOCK_B0_COMMAND="${ILI_EVENING_MOCK_B0_COMMAND:-nohup sleep 2 >/dev/null 2>&1 &}"

MARKER_FILE="${ILI_EVENING_MARKER_FILE:-/tmp/iliria-evening-marker-shard-reads-ok}"

# --dry-run never execs a real ./glm (no `make`, no `ili run/serve/bench`, no `glm` -- see the
# header comment above); ./ili is the real, checked-in file that stands in for "what would
# have run" in the executable-provenance manifest (scripts/provenance.sh).
PROV_BINARY="$CDIR/glm"
(( DRY_RUN )) && PROV_BINARY="$CDIR/ili"

mkdir -p "$BENCH_DIR" "$EVENING_OUT"

GATE_VERDICT=""
PATCH_APPLIED=0
FINAL_STATUS_WRITTEN=0

# ---- small helpers -------------------------------------------------------------------------
now() { date '+%Y-%m-%dT%H:%M:%S'; }
log() { echo "[evening $(date '+%H:%M:%S')] $*"; }

campaign_log() {
  mkdir -p "$(dirname "$CAMPAIGN_LOG")"
  printf -- '- %s | evening_orchestrator(%s) | %s\n' "$(now)" "$ATTEMPT_ID" "$1" >> "$CAMPAIGN_LOG"
}

# Atomic status write (temp file + rename, same filesystem): a reader never observes a
# torn/partial evening-status.json. Fields exactly match the requested contract.
write_status() {
  local step="$1" status="$2" started="${3:-}" finished="${4:-}"
  python3 - "$STATUS_FILE" "$step" "$status" "$started" "$finished" "$ATTEMPT_ID" <<'PY'
import json, os, sys, tempfile

path, step, status, started, finished, attempt_id = sys.argv[1:7]
data = {"step": step, "status": status, "started": started or None,
        "finished": finished or None, "attempt_id": attempt_id}
d = os.path.dirname(path) or "."
os.makedirs(d, exist_ok=True)
fd, tmp = tempfile.mkstemp(dir=d, prefix=".evening-status.", suffix=".tmp")
try:
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
except Exception:
    try:
        os.unlink(tmp)
    except OSError:
        pass
    raise
PY
}

run_step() {
  local name="$1" fn="$2" started rc
  started="$(now)"
  write_status "$name" "running" "$started" ""
  campaign_log "step '$name' starting"
  "$fn"
  rc=$?
  if (( rc == 0 )); then
    campaign_log "step '$name' done (ok)"
    write_status "$name" "passed" "$started" "$(now)"
  else
    campaign_log "step '$name' FAILED (rc=$rc)"
    write_status "$name" "failed" "$started" "$(now)"
  fi
  return "$rc"
}

skip_step() {
  local name="$1" reason="$2"
  campaign_log "step '$name' SKIPPED: $reason"
  write_status "$name" "cancelled" "" "$(now)"
}

# Best-effort safety net: if this script itself dies mid-flight (signal, unexpected error),
# never leave glm.c patched, and never leave evening-status.json on a stale "running" step.
on_exit() {
  local rc=$?
  if (( PATCH_APPLIED )); then
    log "exit trap: capture patch was still applied -- reverting glm.c"
    ( cd "$CDIR" && git checkout -- glm.c ) 2>/dev/null || true
    PATCH_APPLIED=0
  fi
  if (( ! FINAL_STATUS_WRITTEN )); then
    write_status "aborted" "failed" "" "$(now)"
    campaign_log "evening_orchestrator ABORTED unexpectedly (rc=$rc)"
  fi
  exit "$rc"
}
trap on_exit EXIT INT TERM

# ---- step 1/2: wait for the armed gates chain, read gate A's verdict ---------------------
# The gates-chain process (run_container_gates.sh) is NOT a child of this script -- it was
# armed independently, hours earlier, by a different process tree -- so its real exit code is
# not retrievable here (only a real parent can `wait` for that); this matches
# run_container_gates.sh's own `while kill -0 "$WAIT_PID"` wait for the bench PID it watches.
# Process exit only tells us WHEN to read gates.log; the verdict itself comes entirely from
# the log's terminal marker lines (fail-closed: if neither marker is present -- e.g. the chain
# aborted before gate A ran at all -- the verdict is "indeterminate", never guessed).
wait_for_gates() {
  if [[ -z "$GATES_PID" ]]; then
    GATES_PID="$(pgrep -f 'run_container_gates\.sh' 2>/dev/null | head -1 || true)"
    [[ -n "$GATES_PID" ]] && log "auto-discovered gates-chain pid=$GATES_PID via pgrep"
  fi
  if [[ -z "$GATES_PID" ]]; then
    log "no gates-chain pid: none provided via ILI_EVENING_GATES_PID and pgrep found no run_container_gates.sh process"
    GATE_VERDICT="indeterminate"
    return 1
  fi

  log "watching gates-chain pid=$GATES_PID (log: $GATES_LOG, poll every ${GATES_POLL_S}s)"
  campaign_log "waiting for gates-chain pid=$GATES_PID to exit"
  while kill -0 "$GATES_PID" 2>/dev/null; do
    sleep "$GATES_POLL_S"
  done
  log "gates-chain pid=$GATES_PID has exited"

  if [[ ! -f "$GATES_LOG" ]]; then
    log "gates log not found at $GATES_LOG -- cannot determine gate A verdict"
    GATE_VERDICT="indeterminate"
    return 1
  fi

  # Fail-closed precedence: a FAIL marker always wins if somehow both are present (should
  # never happen given run_container_gates.sh's own control flow -- it exits immediately on
  # the FAIL path -- but preferring to run the recovery path is the safe direction to be wrong in).
  if grep -q "GATE A FAIL" "$GATES_LOG"; then
    GATE_VERDICT="failed"
    log "gate A verdict: FAILED (registered prediction) -- 'GATE A FAIL' found in $GATES_LOG"
  elif grep -q "gate A PASS" "$GATES_LOG"; then
    GATE_VERDICT="passed"
    log "gate A verdict: PASSED -- 'gate A PASS' found in $GATES_LOG"
  else
    log "gates log at $GATES_LOG has neither a 'GATE A FAIL' nor a 'gate A PASS' marker -- indeterminate"
    GATE_VERDICT="indeterminate"
    return 1
  fi
  return 0
}

# ---- process/port helpers for the capture step's own serve lifecycle ----------------------
# Duplicated from (not imported from) run_abba_matrix.sh's kill_engine_processes/wait_ready/
# port helpers: bash has no clean way to import a single function out of a standalone script
# without sourcing -- and running -- its whole top-level body. Same dual-kill rationale as
# that script's own comment: a caffeinate-wrapped serve process's glm child can survive a
# single SIGTERM to the wrapper and keep the port bound (empirically confirmed on this
# machine 2026-07-15, see run_abba_matrix.sh's module docstring).
capture_glm_pids() { pgrep -x glm 2>/dev/null | sort; }
capture_port_listener_pid() { lsof -tiTCP:"$CAPTURE_PORT" -sTCP:LISTEN 2>/dev/null | head -1; }
capture_port_is_free() { [[ -z "$(capture_port_listener_pid)" ]]; }
capture_pid_alive() { kill -0 "$1" 2>/dev/null; }

capture_wait_ready() {
  local wrapper_pid="$1" deadline=$((SECONDS + CAPTURE_READY_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    capture_pid_alive "$wrapper_pid" || return 1
    curl -sf -m 2 "http://$CAPTURE_HOST:$CAPTURE_PORT/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

capture_kill_engine() {
  local wrapper_pid="$1" glm_pids="$2" listener waited=0 deadline p survivors
  listener="$(capture_port_listener_pid)"
  log "capture teardown: wrapper_pid=$wrapper_pid listener=${listener:-none} glm_pids=[${glm_pids:-none}]"
  [[ -n "$listener" ]] && kill -TERM "$listener" 2>/dev/null
  if [[ "$wrapper_pid" != "$listener" ]] && capture_pid_alive "$wrapper_pid"; then
    kill -TERM "$wrapper_pid" 2>/dev/null
  fi

  deadline=$((CAPTURE_TEARDOWN_TIMEOUT_S / 2))
  while (( waited < deadline )); do
    if capture_port_is_free; then
      survivors=0
      for p in $glm_pids; do capture_pid_alive "$p" && survivors=1; done
      (( survivors == 0 )) && break
    fi
    sleep 1; waited=$((waited + 1))
  done

  for p in $glm_pids; do
    if capture_pid_alive "$p"; then
      log "capture teardown: glm pid $p survived the graceful path, forcing it down"
      kill -TERM "$p" 2>/dev/null; sleep 1
      capture_pid_alive "$p" && kill -KILL "$p" 2>/dev/null
    fi
  done
  listener="$(capture_port_listener_pid)"
  if [[ -n "$listener" ]]; then
    log "capture teardown: port $CAPTURE_PORT still held by pid $listener, forcing it down"
    kill -KILL "$listener" 2>/dev/null
  fi
  if [[ "$wrapper_pid" != "$listener" ]] && capture_pid_alive "$wrapper_pid"; then
    kill -KILL "$wrapper_pid" 2>/dev/null
  fi

  sleep 1
  if capture_port_is_free; then
    log "capture teardown: verified port $CAPTURE_PORT free"
  else
    log "WARNING: capture teardown could not verify port $CAPTURE_PORT is free (a leaked process may remain)"
  fi
}

# ---- step 3(a): FAST numerical CPU-vs-Metal-prefill capture grid --------------------------
run_capture_grid_both_arms() {
  local arm label dir rc_all=0
  for arm in 0 1; do
    if [[ "$arm" == 0 ]]; then label="cpu"; else label="metal"; fi
    dir="$CAPTURE_OUT/captures-$label"
    mkdir -p "$dir"

    if (( DRY_RUN )); then
      log "capture grid (dry-run) arm=$label"
      if ! python3 "$CAPTURE_GRID_BIN" --capture-dir "$dir" --metal-prefill "$arm" \
             --layers "$CAPTURE_LAYERS" --s-values "$CAPTURE_S_VALUES" \
             --t-values "$CAPTURE_T_VALUES" --dry-run >"$CAPTURE_OUT/grid-$label.log" 2>&1; then
        log "capture grid (dry-run arm=$label) FAILED -- see $CAPTURE_OUT/grid-$label.log"
        rc_all=1
      fi
      continue
    fi

    log "capture grid arm=$label: starting dedicated serve (ILI_METAL_PREFILL=$arm)"
    local serve_log="$CAPTURE_OUT/serve-$label.log" wrapper_pid before_glm new_glm
    # Snapshot BEFORE starting, so a new glm child is correctly identified by diff even on the
    # readiness-failure path below (a crash after spawning glm but before health responds must
    # still be tracked and killed by pid, not just via the port listener).
    before_glm="$(capture_glm_pids)"
    ILI_METAL_PREFILL="$arm" ILI_LAYER_CAPTURE_DIR="$dir" ILI_LAYER_CAPTURE_LAYERS="$CAPTURE_LAYERS" \
      ILI_METAL4_MOE=0 ILI_KVSAVE=0 ILI_HOST="$CAPTURE_HOST" ILI_PORT="$CAPTURE_PORT" \
      bash "$CDIR/run-m5max-serve.sh" "$CAPTURE_MODEL_DIR" >"$serve_log" 2>&1 &
    wrapper_pid=$!

    if ! capture_wait_ready "$wrapper_pid"; then
      log "capture serve arm=$label never became ready within ${CAPTURE_READY_TIMEOUT_S}s -- see $serve_log"
      tail -n 60 "$serve_log" >&2 || true
      new_glm="$(comm -13 <(echo "$before_glm") <(capture_glm_pids))"
      capture_kill_engine "$wrapper_pid" "$new_glm"
      rc_all=1
      continue
    fi
    new_glm="$(comm -13 <(echo "$before_glm") <(capture_glm_pids))"

    log "capture grid arm=$label: driving the T<=4K grid"
    if ! python3 "$CAPTURE_GRID_BIN" --capture-dir "$dir" --metal-prefill "$arm" \
           --host "$CAPTURE_HOST" --port "$CAPTURE_PORT" --layers "$CAPTURE_LAYERS" \
           --s-values "$CAPTURE_S_VALUES" --t-values "$CAPTURE_T_VALUES" \
           >"$CAPTURE_OUT/grid-$label.log" 2>&1; then
      log "capture grid drive arm=$label FAILED -- see $CAPTURE_OUT/grid-$label.log"
      rc_all=1
    fi

    capture_kill_engine "$wrapper_pid" "$new_glm"
  done
  return "$rc_all"
}

run_capture_step() {
  mkdir -p "$CAPTURE_OUT"

  log "capture step: checking $CAPTURE_PATCH still applies cleanly"
  if ! ( cd "$CDIR" && git apply --check "$CAPTURE_PATCH" ) >"$CAPTURE_OUT/patch-check.log" 2>&1; then
    log "ABORT: capture patch no longer applies cleanly against glm.c -- see $CAPTURE_OUT/patch-check.log"
    campaign_log "numerical_captures ABORTED: capture_layer_outputs.patch no longer applies (see $CAPTURE_OUT/patch-check.log)"
    return 1
  fi

  log "applying capture patch"
  if ! ( cd "$CDIR" && git apply "$CAPTURE_PATCH" ) >"$CAPTURE_OUT/patch-apply.log" 2>&1; then
    log "ABORT: git apply failed despite --check passing -- see $CAPTURE_OUT/patch-apply.log"
    return 1
  fi
  PATCH_APPLIED=1

  local build_ok=1
  if (( DRY_RUN )); then
    log "dry-run: skipping the real 'make mac-fast' rebuild of the patched engine"
  else
    log "building patched engine (make mac-fast)"
    if ! ( cd "$CDIR" && make mac-fast ) >"$CAPTURE_OUT/build-patched.log" 2>&1; then
      log "patched build FAILED -- see $CAPTURE_OUT/build-patched.log"
      build_ok=0
    fi
  fi

  local grid_ok=1
  if (( build_ok )); then
    run_capture_grid_both_arms || grid_ok=0
  else
    grid_ok=0
  fi

  if (( grid_ok )); then
    log "comparing CPU vs Metal-prefill captures"
    if python3 "$CDIR/tools/compare_layer_captures.py" \
         --dir-a "$CAPTURE_OUT/captures-cpu" --dir-b "$CAPTURE_OUT/captures-metal" \
         --out-csv "$CAPTURE_OUT/summary.csv" >"$CAPTURE_OUT/compare.log" 2>&1; then
      log "capture comparison written: $CAPTURE_OUT/summary.csv"
    else
      log "compare_layer_captures.py FAILED -- see $CAPTURE_OUT/compare.log"
      grid_ok=0
    fi
  fi

  # Revert + clean rebuild ALWAYS run, regardless of grid/compare outcome above: the tree
  # must never be left patched and ./glm must never be left out of sync with glm.c.
  log "reverting capture patch"
  ( cd "$CDIR" && git checkout -- glm.c )
  PATCH_APPLIED=0

  local rebuild_ok=1
  if (( DRY_RUN )); then
    log "dry-run: skipping the real clean rebuild"
  else
    log "rebuilding clean engine (make mac-fast)"
    if ! ( cd "$CDIR" && make mac-fast ) >"$CAPTURE_OUT/build-clean.log" 2>&1; then
      log "CLEAN REBUILD FAILED after revert -- ./glm may not match glm.c -- see $CAPTURE_OUT/build-clean.log"
      rebuild_ok=0
    fi
  fi

  (( build_ok && grid_ok && rebuild_ok )) || return 1
  return 0
}

# ---- step 3(b): quiesce, up to QUIESCE_RETRIES x QUIESCE_INTERVAL_S -----------------------
run_quiesce_with_retries() {
  mkdir -p "$QUIESCE_OUT_DIR"
  local attempt=1 rc out
  while (( attempt <= QUIESCE_RETRIES )); do
    out="$QUIESCE_OUT_DIR/attempt-$attempt.log"
    log "quiesce attempt $attempt/$QUIESCE_RETRIES ($QUIESCE_BIN)"
    ILI_DIRECT=1 bash "$QUIESCE_BIN" >"$out" 2>&1
    rc=$?
    if (( rc == 0 )); then
      log "quiesce PASSED on attempt $attempt/$QUIESCE_RETRIES -- see $out"
      campaign_log "quiesce PASSED on attempt $attempt/$QUIESCE_RETRIES"
      return 0
    fi
    log "quiesce FAILED on attempt $attempt/$QUIESCE_RETRIES (rc=$rc) -- see $out"
    if (( attempt < QUIESCE_RETRIES )); then
      log "waiting ${QUIESCE_INTERVAL_S}s before the next quiesce attempt"
      sleep "$QUIESCE_INTERVAL_S"
    fi
    attempt=$((attempt + 1))
  done
  log "quiesce did not pass after $QUIESCE_RETRIES attempts -- giving up"
  campaign_log "quiesce FAILED after $QUIESCE_RETRIES attempts (see $QUIESCE_OUT_DIR)"
  return 1
}

# ---- step 3(c): same-commit ABBA matrix ---------------------------------------------------
run_abba_step() {
  mkdir -p "$EVENING_OUT"
  if capture_port_is_free; then
    log "port $CAPTURE_PORT confirmed free before ABBA"
  else
    log "WARNING: port $CAPTURE_PORT is NOT free before starting ABBA -- a leftover process may be holding it"
  fi

  local abba_args rc
  abba_args=()
  (( DRY_RUN )) && abba_args+=(--dry-run)
  # Real mode always targets a real model dir; --dry-run omits it so run_abba_matrix.sh uses
  # its own synthetic fixture directory (never touch a real (huge) model dir from a dry run).
  (( ! DRY_RUN )) && abba_args+=("$ABBA_MODEL_DIR")

  log "running ABBA matrix: bash $ABBA_BIN ${abba_args[*]:-}"
  # "${abba_args[@]+"${abba_args[@]}"}" (not the bare "${abba_args[@]}"): macOS's bash 3.2
  # throws "unbound variable" expanding an empty array under `set -u` (confirmed on this
  # machine; see run_abba_matrix.sh's own mock_cmd comment on the same pitfall) -- abba_args
  # always has exactly one element today, but this stays correct (zero real arguments, not one
  # spurious empty-string argument) if that ever changes.
  if ILI_ABBA_RESULT_DIR="$ABBA_OUT" bash "$ABBA_BIN" "${abba_args[@]+"${abba_args[@]}"}" >"$ABBA_LOG" 2>&1; then
    log "ABBA matrix completed OK -- see $ABBA_OUT/results.md"
    rc=0
  else
    log "ABBA matrix FAILED -- see $ABBA_LOG"
    rc=1
  fi

  if capture_port_is_free; then
    log "port $CAPTURE_PORT confirmed free after ABBA"
  else
    log "WARNING: port $CAPTURE_PORT is still NOT free after ABBA -- its teardown may have leaked a process"
  fi
  return "$rc"
}

# ---- step 3(d): launch B0 detached ---------------------------------------------------------
b0_command_from_state() {
  python3 - "$STATE_JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    data = json.load(fh)
print(data["gate_b_comparator_correction"]["b0_command"])
PY
}

run_b0_launch() {
  local real_cmd cmd repo_root orig_dir bg_pid
  if [[ -n "$B0_COMMAND_OVERRIDE" ]]; then
    real_cmd="$B0_COMMAND_OVERRIDE"
    log "B0 command source: ILI_EVENING_B0_COMMAND override"
  else
    if ! real_cmd="$(b0_command_from_state 2>"$EVENING_OUT/b0-state-read.err")"; then
      log "ABORT: could not read gate_b_comparator_correction.b0_command from $STATE_JSON -- see $EVENING_OUT/b0-state-read.err"
      return 1
    fi
    log "B0 command source: $STATE_JSON"
  fi
  campaign_log "B0 command resolved: $real_cmd"

  if (( DRY_RUN )); then
    cmd="$MOCK_B0_COMMAND"
    log "dry-run: launching a MOCK background process instead of the real B0 command above"
  else
    cmd="$real_cmd"
  fi

  # The given b0_command starts with "cd c && ..." (written for a repo-root shell): run it
  # from the repo root (one level above $CDIR) so that embedded `cd c` resolves correctly,
  # matching how a human operator would paste this exact one-liner.
  repo_root="$(cd -- "$CDIR/.." && pwd)"
  orig_dir="$PWD"
  cd "$repo_root" || { log "ABORT: could not cd to repo root $repo_root"; return 1; }
  eval "$cmd"
  bg_pid=$!
  cd "$orig_dir" || true
  disown "$bg_pid" 2>/dev/null || true

  sleep 1
  if kill -0 "$bg_pid" 2>/dev/null; then
    log "B0 launched OK, pid=$bg_pid (detached)"
    campaign_log "B0 launched (pid=$bg_pid, dry_run=$DRY_RUN)"
    return 0
  else
    log "B0 process (pid=$bg_pid) not found ~1s after launch -- may have exited immediately or backgrounding failed"
    campaign_log "B0 launch VERIFICATION FAILED (pid=$bg_pid not alive after 1s, dry_run=$DRY_RUN)"
    return 1
  fi
}

# ---- step 3(e): marker file -----------------------------------------------------------------
write_marker_file() {
  mkdir -p "$(dirname "$MARKER_FILE")"
  : > "$MARKER_FILE" 2>"$EVENING_OUT/marker-write.err"
  if [[ -f "$MARKER_FILE" ]]; then
    log "marker file written: $MARKER_FILE (shard readers may now start)"
    campaign_log "marker file written: $MARKER_FILE"
    return 0
  fi
  log "FAILED to write marker file $MARKER_FILE -- see $EVENING_OUT/marker-write.err"
  return 1
}

# ---- provenance manifest (executable-provenance system, scripts/provenance.sh) -------------
# Non-fatal by design, unlike the equivalent hard gate in run_container_gates.sh /
# run_abba_matrix.sh / roofline_run.sh: this orchestrator's own established contract (see the
# header comment and every *FailurePathTest in tests/test_evening_orchestrator.py) is that
# B0 + the marker file ALWAYS run regardless of any earlier step's outcome, because B0 is the
# durable comparator the container's quality verdict depends on. A provenance hiccup here is
# recorded (loudly, in the campaign log) but must not join the set of things that can block
# B0/marker -- reuses $QUIESCE_BIN so a test's fixture quiesce script is honored here too,
# instead of paying a real tens-of-seconds telemetry sample.
record_provenance() {
  if bash "$SCRIPT_DIR/provenance.sh" --attempt-id "$ATTEMPT_ID" --binary "$PROV_BINARY" \
       --model-dir "$ABBA_MODEL_DIR" --artifact-dir "$EVENING_OUT" --quiesce-bin "$QUIESCE_BIN"; then
    return 0
  fi
  log "provenance manifest emission FAILED -- recorded, non-gating (B0/marker are never held hostage by this)"
  return 1
}

# ---- main -----------------------------------------------------------------------------------
main() {
  write_status "init" "running" "$(now)" ""
  campaign_log "evening_orchestrator starting (attempt_id=$ATTEMPT_ID dry_run=$DRY_RUN)"
  record_provenance || true

  run_step "wait_for_gates" wait_for_gates

  if [[ "$GATE_VERDICT" != "passed" && "$GATE_VERDICT" != "failed" ]]; then
    campaign_log "gate verdict INDETERMINATE -- stopping without taking either branch (fail-closed)"
    write_status "gate_verdict" "failed" "" "$(now)"
    FINAL_STATUS_WRITTEN=1
    exit 1
  fi

  if [[ "$GATE_VERDICT" == "passed" ]]; then
    campaign_log "gate A PASSED -- B1 (mixed-container n=100 bench) already running via the gates chain; skipping numerical captures and ABBA"
    write_status "gate_a_passed" "passed" "" "$(now)"
    FINAL_STATUS_WRITTEN=1
    log "gate A passed: nothing further for this orchestrator to do tonight. Exiting 0."
    exit 0
  fi

  campaign_log "gate A FAILED (registered prediction) -- running the recovery sequence: numerical captures -> quiesce -> ABBA -> B0 -> marker"

  local overall_ok=1 b0_rc

  if run_step "numerical_captures" run_capture_step; then :; else overall_ok=0; fi

  if (( overall_ok )); then
    if run_step "quiesce" run_quiesce_with_retries; then :; else overall_ok=0; fi
  else
    skip_step "quiesce" "numerical_captures step failed"
  fi

  if (( overall_ok )); then
    if run_step "abba_matrix" run_abba_step; then :; else overall_ok=0; fi
  else
    skip_step "abba_matrix" "an earlier step failed"
  fi

  # B0 launch and the marker file are ALWAYS attempted on the gate-A-failed path, independent
  # of capture/quiesce/ABBA outcomes (see the header comment's fail-closed policy).
  run_step "b0_launch" run_b0_launch
  b0_rc=$?
  run_step "marker_file" write_marker_file

  if (( overall_ok )) && (( b0_rc == 0 )); then
    write_status "complete" "passed" "" "$(now)"
    FINAL_STATUS_WRITTEN=1
    campaign_log "evening_orchestrator complete: recovery sequence finished OK"
    exit 0
  else
    write_status "complete" "failed" "" "$(now)"
    FINAL_STATUS_WRITTEN=1
    campaign_log "evening_orchestrator complete WITH FAILURES in the recovery sequence (see per-step status in $STATUS_FILE); B0 + marker were still executed per fail-closed policy"
    exit 1
  fi
}

main
