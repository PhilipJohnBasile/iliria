#!/usr/bin/env bash
# Dual-roofline harness driver (docs/performance-theory.json p2-m5-gpu-tensor-path-probe).
#
# Drives the fixed coding prompt-set (the same three prompts/labels as
# ab-m5max-k6-matrix.sh: rust_queue, godot_controller, typescript_websocket) through
# run-m5max-fast.sh in TWO phases:
#   cold   -- `sudo -n purge` then the prompt-set, cycling, for ILI_ROOFLINE_COLD_S wall-
#             clock seconds (default 300 = first 5 minutes; peak/burst before thermal
#             throttling engages).
#   steady -- the same prompt-set, cycling, for ILI_ROOFLINE_STEADY_S wall-clock seconds
#             (default 2400 = 40min, clamped to the required 30-60min band), no purge (stays
#             warm), meant to reach a power-constrained steady state.
# A background sampler records accessible telemetry (pmset -g therm throttle state; iostat
# disk MB/s and CPU idle %; vm_stat pageouts) every ILI_ROOFLINE_SAMPLE_S seconds (default
# 10) to a per-phase CSV. tools/roofline_report.py then turns the engine's own PROFILE/
# METAL-*/METAL-GEMM stderr counters plus these CSVs into the roofline report.
#
# powermetrics is NOT available (no passwordless sudo beyond /usr/sbin/purge) -- GPU
# frequency/power, memory-controller bandwidth, SSD temperature, and fan state are therefore
# NOT sampled; this is recorded as a stated method limit in the generated report, not
# silently assumed away.
#
# Usage:
#   bash scripts/roofline_run.sh MODEL_DIR [options]
#   bash scripts/roofline_run.sh --mock [options]      # synthetic engine, no glm/model/GPU
#
# Options:
#   --mock             use tools/roofline_report.py's mock-log generator instead of a real
#                       `run-m5max-fast.sh` invocation -- proves the harness plumbing
#                       (quiesce gate, two-phase loop, telemetry sampler, attempt_id
#                       stamping, manifest, report generation) end to end without real
#                       hardware or a model. Phase durations default much shorter under
#                       --mock (see env below) so the whole dry run takes seconds.
#   --cold-only        run only the cold phase
#   --steady-only       run only the steady phase
#   -h|--help          print this header and exit
#
# Env (all optional):
#   ILI_ROOFLINE_ATTEMPT_ID   default roofline-<date>-attempt1
#   ILI_ROOFLINE_RESULT_DIR   default <repo>/bench-m5max/roofline-<date-time>
#   ILI_ROOFLINE_COLD_S       cold-phase duration, seconds (default 300; --mock: 12)
#   ILI_ROOFLINE_STEADY_S     steady-phase duration, seconds (default 2400; clamped to
#                              [1800,3600] in real mode per the 30-60min requirement;
#                              --mock: 24, not clamped)
#   ILI_ROOFLINE_SAMPLE_S     telemetry sample interval, seconds (default 10; --mock: 2)
#   ILI_ROOFLINE_NGEN         tokens/run passed to run-m5max-fast.sh (default 128)
#
# Design (per the calling brief; mirrors scripts/run_abba_matrix.sh's own conventions):
#   (a) refuses to start unless scripts/quiesce_check.sh passes (real mode: hard gate --
#       aborts rather than start a timing campaign on a non-quiet machine; --mock: still
#       invoked and recorded, but does not block, same carve-out run_abba_matrix.sh uses,
#       since a mock run risks no real engine/hardware and this machine legitimately runs
#       other things most of the day).
#   (b) ILI_DIRECT=1 is exported explicitly (quiesce_check.sh's own gate 8 requires this to
#       be set, not merely defaulted).
#   (c) every artifact filename embeds $ATTEMPT_ID (bench-m5max/campaign-state.json
#       convention: "every result artifact should embed attempt_id", per
#       scripts/run_abba_matrix.sh's own comment).
#   (d) tools/roofline_report.py report is invoked at the end against the whole result dir.
#
# Unlike run_abba_matrix.sh / ab-m5max-k6-matrix.sh, this harness does NOT snapshot/restore
# .fa_usage between runs: those matrices need frozen state because they A/B two configs
# against each other, but this harness characterizes ONE configuration's own sustained
# behavior over time -- letting AUTOPIN's usage history evolve naturally across the phase is
# the intended, representative condition, not a gap.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CDIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$CDIR"

log() { echo "[roofline $(date '+%H:%M:%S')] $*"; }
die() { echo "[roofline $(date '+%H:%M:%S')] FATAL: $*" >&2; exit 1; }

# timing-run exclusivity (scripts/timing_lock.sh) + process-group ownership
# (scripts/exp_launch.sh) -- sourcing only defines functions, see each file's own header.
source "$SCRIPT_DIR/timing_lock.sh"
source "$SCRIPT_DIR/exp_launch.sh"

# ---- args --------------------------------------------------------------------------------
MOCK=0
COLD=1
STEADY=1
MODEL_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock) MOCK=1; shift ;;
    --cold-only) STEADY=0; shift ;;
    --steady-only) COLD=0; shift ;;
    -h|--help) sed -n '2,55p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --) shift; break ;;
    -*) die "unknown flag: $1" ;;
    *) MODEL_DIR="${1%/}"; shift ;;
  esac
done
[[ "$COLD" == 1 || "$STEADY" == 1 ]] || die "--cold-only and --steady-only are mutually exclusive"

[[ "$(uname -s)" == Darwin ]] || die "this harness targets macOS"

if [[ "$MOCK" == 1 ]]; then
  [[ -z "$MODEL_DIR" ]] || log "note: MODEL_DIR ($MODEL_DIR) is ignored under --mock"
  # --mock never execs a real engine at all (tools/roofline_report.py's mock-log generator
  # stands in, see run_one() below) -- ./ili is the real, checked-in file that stands in for
  # "what would have run" in the executable-provenance manifest (scripts/provenance.sh).
  PROV_BINARY="$CDIR/ili"
else
  [[ -n "$MODEL_DIR" ]] || die "MODEL_DIR is required (or pass --mock)"
  [[ -d "$MODEL_DIR" ]] || die "model directory not found: $MODEL_DIR"
  [[ -x "$CDIR/ili" ]] || die "$CDIR/ili not found or not executable"
  PROV_BINARY="$CDIR/glm"
fi

# ---- config --------------------------------------------------------------------------
ATTEMPT_ID="${ILI_ROOFLINE_ATTEMPT_ID:-roofline-$(date +%Y%m%d)-attempt1}"
OUT="${ILI_ROOFLINE_RESULT_DIR:-$CDIR/bench-m5max/roofline-$(date +%Y%m%d-%H%M%S)}"
NGEN="${ILI_ROOFLINE_NGEN:-128}"

if [[ "$MOCK" == 1 ]]; then
  COLD_S="${ILI_ROOFLINE_COLD_S:-12}"
  STEADY_S="${ILI_ROOFLINE_STEADY_S:-24}"
  SAMPLE_S="${ILI_ROOFLINE_SAMPLE_S:-2}"
else
  COLD_S="${ILI_ROOFLINE_COLD_S:-300}"
  STEADY_S="${ILI_ROOFLINE_STEADY_S:-2400}"
  # Enforce the required 30-60min steady-state band in real mode; --mock is exempt (it
  # would defeat the point of a fast dry run) and so is an explicit override left as-is
  # below the enforcement -- this only clamps the DEFAULT, not a user-supplied value.
  if [[ -z "${ILI_ROOFLINE_STEADY_S:-}" ]]; then
    (( STEADY_S < 1800 )) && STEADY_S=1800
    (( STEADY_S > 3600 )) && STEADY_S=3600
  fi
  SAMPLE_S="${ILI_ROOFLINE_SAMPLE_S:-10}"
fi

mkdir -p "$OUT/logs" "$OUT/artifacts" "$OUT/system" "$OUT/telemetry"
MANIFEST_TSV="$OUT/manifest.tsv"
printf 'phase\tlabel\ttrial\tlog\n' > "$MANIFEST_TSV"
log "attempt_id=$ATTEMPT_ID mock=$MOCK out=$OUT cold_s=$COLD_S steady_s=$STEADY_S sample_s=$SAMPLE_S ngen=$NGEN"

# ---- timing-run exclusivity lock (MECHANICAL, not procedural) -------------------------
# Must acquire before the quiesce gate: refuses to start outright if a competing timing
# driver already holds this lock, real or --mock (no soft carve-out here, unlike the
# quiesce gate below -- see scripts/timing_lock.sh's header for why this guarantee is
# unconditional). Released on exit via cleanup() below; this early trap only covers the
# narrow window between acquiring here and cleanup()'s own `trap cleanup ...` further
# down replacing it.
timing_lock_acquire "$ATTEMPT_ID" || die "another timing driver holds the exclusivity lock ($(timing_lock_path)) -- refusing to start"
# Signal traps MUST exit: bash RESUMES the script after a signal-trap handler returns, so a
# non-exiting INT/TERM handler would release the lock and then keep running unprotected. Exiting
# from the signal fires the EXIT trap, so release still happens exactly once.
trap 'timing_lock_release' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- quiesce gate (design item a) ------------------------------------------------------
run_quiesce() {
  local outfile="$1" rc
  bash "$SCRIPT_DIR/quiesce_check.sh" > "$outfile" 2>&1
  rc=$?
  cat "$outfile"
  return "$rc"
}

QUIESCE_PRE="$OUT/system/${ATTEMPT_ID}-preflight-quiesce.txt"
if run_quiesce "$QUIESCE_PRE"; then
  log "preflight quiesce: PASS"
else
  if [[ "$MOCK" == 1 ]]; then
    log "preflight quiesce: NOT QUIESCED -- continuing anyway (--mock has no engine/hardware " \
        "at risk; this is expected whenever another phase legitimately owns the machine, see " \
        "bench-m5max/campaign-state.json)"
  else
    die "preflight quiesce FAILED -- refusing to start (see $QUIESCE_PRE). Re-run once the " \
        "machine is quiet; a roofline campaign measured on a non-quiet machine is not trustworthy."
  fi
fi

# ILI_DIRECT=1 must be EXPLICITLY exported (quiesce_check.sh gate 8; run-m5max-fast.sh's own
# ILI_DIRECT default of 1 is not enough -- the calling brief requires this to be explicit).
export ILI_DIRECT=1
# run-m5max-fast.sh reads ILI_NGEN from the environment (default 512); without exporting it
# here, ILI_ROOFLINE_NGEN would be read into $NGEN above but never actually reach the engine.
export ILI_NGEN="$NGEN"

# ---- fixed prompt-set (same three prompts/labels as ab-m5max-k6-matrix.sh) -------------
prompts=(
  "Implement a cancellation-safe bounded MPSC queue in Rust using atomics. Explain the memory ordering and include tests."
  "A Godot 4.7 GDScript character controller allocates every physics frame and occasionally tunnels through ramps. Diagnose it and provide a corrected implementation."
  "Review a TypeScript WebSocket reconnection manager for race conditions, leaked timers, stale closures, and exponential-backoff errors. Return a patch and tests."
)
labels=(rust_queue godot_controller typescript_websocket)

if [[ "$MOCK" == 0 ]] && pgrep -x glm >/dev/null 2>&1; then
  die "a glm process is already running; benchmark isolation would be invalid"
fi

# ---- provenance manifest (executable-provenance system, scripts/provenance.sh) -------------
# Snapshotted right before the cold/steady phases begin -- the moment that matters for a
# multi-hour campaign is "what will actually run", not what git HEAD claims.
prov_args=(--attempt-id "$ATTEMPT_ID" --binary "$PROV_BINARY" --artifact-dir "$OUT"
           --quiesce-bin "$SCRIPT_DIR/quiesce_check.sh" --prompt "${prompts[*]}")
[[ -n "$MODEL_DIR" ]] && prov_args+=(--model-dir "$MODEL_DIR")
if ! bash "$CDIR/scripts/provenance.sh" "${prov_args[@]}"; then
  if [[ "$MOCK" == 1 ]]; then
    log "provenance manifest emission FAILED -- continuing anyway (--mock has no engine/hardware at risk)"
  else
    die "provenance manifest emission FAILED -- refusing to start a multi-hour roofline campaign without a record of what will run"
  fi
fi

# ---- telemetry sampler (design item: sample every 10s to CSV) --------------------------
# pmset -g therm / iostat / vm_stat are the accessible tools named in the calling brief;
# powermetrics is not available (see header). Each field mirrors an existing, already-
# working parse convention from scripts/quiesce_check.sh so it stays robust to iostat's
# header-driven column layout rather than a hardcoded index.
PHASE_FILE="$OUT/system/${ATTEMPT_ID}-current-phase"
printf 'idle' > "$PHASE_FILE"
TELEM_CSV_COLD="$OUT/telemetry/${ATTEMPT_ID}-cold.csv"
TELEM_CSV_STEADY="$OUT/telemetry/${ATTEMPT_ID}-steady.csv"
printf 'timestamp,phase,elapsed_s,thermal_speed_limit_pct,cpu_idle_pct,disk_mb_s,pageouts_cum\n' > "$TELEM_CSV_COLD"
printf 'timestamp,phase,elapsed_s,thermal_speed_limit_pct,cpu_idle_pct,disk_mb_s,pageouts_cum\n' > "$TELEM_CSV_STEADY"

telemetry_loop() {
  local start_epoch phase now elapsed therm_raw therm iostat_out idcol cpu_idle disk_mbs pageouts csv
  start_epoch=$(date +%s)
  while true; do
    phase="$(cat "$PHASE_FILE" 2>/dev/null || echo idle)"
    [[ "$phase" == STOP ]] && break
    if [[ "$phase" == cold || "$phase" == steady ]]; then
      now=$(date +%s); elapsed=$((now - start_epoch))
      therm_raw="$(pmset -g therm 2>/dev/null)"
      therm="$(echo "$therm_raw" | grep -i 'CPU_Speed_Limit' | grep -o '[0-9]*' | head -1)"
      iostat_out="$(iostat -c 2 -w 1 2>/dev/null)"
      idcol="$(echo "$iostat_out" | awk '/id/{for(i=1;i<=NF;i++) if($i=="id"){print i; exit}}' | head -1)"
      cpu_idle=""
      [[ -n "$idcol" ]] && cpu_idle="$(echo "$iostat_out" | tail -1 | awk -v c="$idcol" '{print $c}')"
      # -c 2 (not -c 1): iostat's single report is the since-boot average, not the live delta --
      # the same bug fixed in timing_watchdog.read_disk_mbs; parse the last (1s-delta) line.
      disk_mbs="$(iostat -d -w 1 -c 2 disk0 2>/dev/null | tail -1 | awk '{print $3}')"
      pageouts="$(vm_stat 2>/dev/null | awk '/Pageouts/{gsub("\\.","",$2); print $2}')"
      csv="$TELEM_CSV_COLD"; [[ "$phase" == steady ]] && csv="$TELEM_CSV_STEADY"
      printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$phase" "$elapsed" "${therm:-}" "${cpu_idle:-}" \
        "${disk_mbs:-}" "${pageouts:-}" >> "$csv"
    fi
    sleep "$SAMPLE_S"
  done
}
telemetry_loop &
TELEM_PID=$!

# ---- timing watchdog (mechanical mid-run contamination detector, tools/timing_watchdog.py) --
# A campaign-scope watchdog runs for the whole cold+steady window. Each trial's glm lives in its
# OWN process group (run_one, via exp_launch), so the watchdog cannot take one static pgid; it
# reads an allowlist FILE every sample and run_one adds the current trial's pgid while the trial
# runs. It runs --no-terminate: it only marks TIMING_INVALID.json and this driver owns orderly
# teardown (the watchdog itself lives in the driver's group, so it must not kill it). run_phase
# polls that sentinel and aborts. Disk MB/s is TOTAL disk -- dominated by the engine's own
# legitimate expert streaming -- so it is effectively disabled here (very high default); the
# real contamination signals are unexpected PROCESSES (a foreign compiler/test/analysis proc =
# HARD, needs no calibration), foreign CPU, thermal, AC-loss and pageouts. Thresholds are
# env-tunable; run `timing_watchdog.py calibrate` on this machine before an unattended campaign.
DRIVER_PGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]')"; [[ -n "$DRIVER_PGID" ]] || DRIVER_PGID=$$
ALLOWLIST_FILE="$OUT/system/${ATTEMPT_ID}-watchdog-allowlist.txt"
WATCHDOG_JSON="$OUT/system/${ATTEMPT_ID}-watchdog.json"
WATCHDOG_PID=""
ROOFLINE_INVALID=0
# Atomic allowlist update (write-temp-then-rename): a plain `> file` truncates in place, and the
# watchdog's read_pgid_file can catch that empty window and see the trial's own glm as foreign.
write_allowlist() { printf '%s\n' "$@" > "$ALLOWLIST_FILE.tmp" && mv -f "$ALLOWLIST_FILE.tmp" "$ALLOWLIST_FILE"; }
write_allowlist "$DRIVER_PGID"
# A TIMING_INVALID.json left in a REUSED result dir (ILI_ROOFLINE_RESULT_DIR) would make
# run_phase's first poll abort at trial 0. The default OUT is timestamped (fresh); clear a
# pre-existing sentinel loudly for the reused-dir case so this run starts clean.
if [[ -e "$OUT/TIMING_INVALID.json" ]]; then
  log "NOTE: clearing a stale TIMING_INVALID.json from a prior run in this result dir before starting"
  rm -f "$OUT/TIMING_INVALID.json"
fi

cleanup() {
  printf 'STOP' > "$PHASE_FILE" 2>/dev/null || true
  if [[ -n "${TELEM_PID:-}" ]] && kill -0 "$TELEM_PID" 2>/dev/null; then
    kill "$TELEM_PID" 2>/dev/null || true
    wait "$TELEM_PID" 2>/dev/null || true
  fi
  # Stop the campaign-scope timing watchdog (it runs --no-terminate, so it never killed anything
  # itself). Stop it before cancelling experiment groups so it stops sampling first and does not
  # observe our own orderly teardown as "contamination".
  if [[ -n "${WATCHDOG_PID:-}" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  # Process-group ownership (scripts/exp_launch.sh): if an experiment child was still in
  # flight when cleanup fires (external signal mid-run), cancel its WHOLE process group --
  # not just its leaf pid -- so a killed roofline_run.sh can never leave an orphaned engine
  # process (or a respawn-loop descendant) behind. See exp_launch.sh's header for the
  # motivating incident.
  if [[ -n "${CURRENT_EXP_PGID:-}" ]]; then
    log "cleanup: an experiment child (pgid=$CURRENT_EXP_PGID) was still in flight -- cancelling its process group"
    exp_launch_cancel "$CURRENT_EXP_PGID" "$OUT/system/${ATTEMPT_ID}-exp-cancel-state.txt" 10
    CURRENT_EXP_PGID=""
  fi
  timing_lock_release
}
# Signal traps MUST exit (see the early-trap note above): bash resumes the script after a signal
# handler returns, so without this a Ctrl-C mid-campaign would run cleanup() -- releasing the
# lock and killing the watchdog+telemetry -- and then KEEP launching engine trials lock-free,
# unwatched, and report success (rc=0). Exiting from the signal fires the EXIT trap, so cleanup()
# runs exactly once on every path.
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Start the watchdog now that cleanup() (which stops it) and its trap are installed. Real runs
# only: --mock has no engine/timing to protect and skips the quiesce gate above for the same
# reason. Backgrounded into THIS driver's process group (no set -m) so its own pgid==DRIVER_PGID
# is in the allowed set, and it excludes its own pid automatically.
#
# CONTAMINATION MODEL (2026-07-15 validation-run calibration): aggregate foreign CPU% and total
# disk MB/s are INFORMATIONAL only -- both are dominated by the engine's OWN kernel_task streaming
# I/O (~13 GB/s disk, 331% "foreign" CPU measured), so no fixed threshold can distinguish them
# from contamination. Instead the watchdog enforces a GENERIC per-user-process net: any USER-owned
# process outside the allowed groups that persists (--consecutive-required samples) or spikes
# (--foreign-proc-instant-cpu) above --foreign-proc-min-cpu invalidates the run -- catching agents,
# git, browsers, iPhone Mirroring, compressors, and tools whose names were never anticipated, while
# kernel_task (root -- the engine's own I/O) is excluded by uid. --benign-pattern skips the
# legitimate operator processes that must stay running (your terminal + editor processes).
# Named glm/compiler/analysis procs stay HARD, as do thermal/AC/pageout/telemetry-gap. Override via
# the ILI_WATCHDOG_* env vars. (foreign-cpu/disk keep their informational defaults, unenforced.)
if [[ "$MOCK" == 0 ]]; then
  python3 "$CDIR/tools/timing_watchdog.py" run \
    --pgid "$DRIVER_PGID" --allowlist-file "$ALLOWLIST_FILE" \
    --out-json "$WATCHDOG_JSON" --result-dir "$OUT" --attempt-id "$ATTEMPT_ID" --no-terminate \
    --exclude-pids "$$,${TELEM_PID}" \
    --interval-s "${ILI_WATCHDOG_INTERVAL_S:-7}" \
    --consecutive-required "${ILI_WATCHDOG_CONSEC:-2}" \
    --foreign-proc-min-cpu "${ILI_WATCHDOG_PROC_MIN_CPU:-3}" \
    --foreign-proc-instant-cpu "${ILI_WATCHDOG_PROC_INSTANT_CPU:-25}" \
    --benign-pattern "${ILI_WATCHDOG_BENIGN:-ghostty|node}" \
    > "$OUT/system/${ATTEMPT_ID}-watchdog.stderr" 2>&1 &
  WATCHDOG_PID=$!
  log "timing watchdog started (pid=$WATCHDOG_PID base-pgid=$DRIVER_PGID) -> $(basename "$WATCHDOG_JSON")"
fi

# ---- one prompt run, real or mock ------------------------------------------------------
RUN_SEED=0
CURRENT_EXP_PGID=""
run_one() {
  local phase="$1" label="$2" prompt="$3" trial="$4"
  local stem="${ATTEMPT_ID}-${phase}-${label}-t${trial}"
  local log_path="$OUT/logs/${stem}.log"
  RUN_SEED=$((RUN_SEED + 1))
  if [[ "$MOCK" == 1 ]]; then
    python3 "$CDIR/tools/roofline_report.py" mock-log --out "$log_path" --phase "$phase" \
      --seed "$RUN_SEED" --tokens "$NGEN" > /dev/null
    # A real run takes many seconds; without a floor here, --mock's tiny per-call cost would
    # pack hundreds of iterations into a 12-24s window before the phase's wall-clock budget
    # is spent, which proves nothing extra about the loop and just churns out noise files.
    # A fixed per-run floor keeps the mock iteration count small and inspectable while still
    # genuinely exercising the wall-clock phase-duration logic (not a fixed iteration count).
    sleep "${ILI_ROOFLINE_MOCK_RUN_S:-1}"
  else
    # Launched via exp_launch.sh into its OWN process group (not just backgrounded into
    # this script's group) so cleanup()'s exp_launch_cancel can reach a respawn-loop
    # descendant too if this driver is killed mid-run -- see exp_launch.sh's header.
    exp_launch_begin
    ( set +e
      bash "$CDIR/run-m5max-fast.sh" "$MODEL_DIR" run "$prompt"
      echo "ROOFLINE-EXIT: $?"
    ) > "$log_path" 2>&1 &
    local pid=$!
    # Assign the pgid IMMEDIATELY from $pid (pgid==pid for the leader under `set -m`), BEFORE any
    # ps round-trip: an INT arriving in that gap would otherwise leave CURRENT_EXP_PGID empty so
    # cleanup could not cancel the group -> orphaned engine (the exact incident exp_launch.sh
    # exists to prevent). Then restore monitor mode in THIS shell (not a subshell, so it sticks),
    # and add the trial's group to the watchdog allowlist for the life of the trial so the
    # watchdog does not flag this trial's own glm as a foreign process.
    CURRENT_EXP_PGID="$pid"
    exp_launch_restore_m
    write_allowlist "$DRIVER_PGID" "$pid"
    wait "$pid" 2>/dev/null || true
    write_allowlist "$DRIVER_PGID"
    CURRENT_EXP_PGID=""
  fi
  printf '%s\t%s\t%s\t%s\n' "$phase" "$label" "$trial" "logs/${stem}.log" >> "$MANIFEST_TSV"
}

run_phase() {
  local phase="$1" duration_s="$2" phase_start i=0 trial idx label prompt
  printf '%s' "$phase" > "$PHASE_FILE"
  phase_start=$(date +%s)
  log "phase=$phase starting, target duration=${duration_s}s"
  while (( $(date +%s) - phase_start < duration_s )); do
    if [[ -e "$OUT/TIMING_INVALID.json" ]]; then
      log "WATCHDOG CONTAMINATION ($OUT/TIMING_INVALID.json) -- aborting phase=$phase after $i run(s)"
      ROOFLINE_INVALID=1
      break
    fi
    idx=$(( i % ${#prompts[@]} ))
    trial=$(( i / ${#prompts[@]} + 1 ))
    label="${labels[$idx]}"; prompt="${prompts[$idx]}"
    log "phase=$phase run $((i+1)) label=$label trial=$trial (elapsed $(( $(date +%s) - phase_start ))/${duration_s}s)"
    run_one "$phase" "$label" "$prompt" "$trial"
    i=$((i + 1))
  done
  printf 'idle' > "$PHASE_FILE"
  log "phase=$phase complete: $i run(s) in $(( $(date +%s) - phase_start ))s"
}

# ---- cold phase: purge then drive the prompt-set --------------------------------------
if [[ "$COLD" == 1 ]]; then
  if [[ "$MOCK" == 1 ]]; then
    log "cold phase: skipping purge (--mock has no page cache/model resident to reset)"
  else
    sync
    if sudo -n purge >/dev/null 2>&1; then
      sleep 3
      log "cold phase: purge complete"
    else
      log "warning: passwordless 'sudo purge' unavailable -- cold phase will actually be warm " \
          "(this repo's own convention, e.g. ab-m5max-k6-matrix.sh, treats this as a soft " \
          "warning, not a hard failure)"
    fi
  fi
  run_phase cold "$COLD_S"
fi

# ---- steady-state phase: no purge, sustained loop -------------------------------------
if [[ "$STEADY" == 1 && "$ROOFLINE_INVALID" == 0 ]]; then
  run_phase steady "$STEADY_S"
elif [[ "$STEADY" == 1 ]]; then
  log "skipping steady phase: run already invalidated by the watchdog during the cold phase"
fi

cleanup
trap - EXIT INT TERM

QUIESCE_POST="$OUT/system/${ATTEMPT_ID}-postflight-quiesce.txt"
run_quiesce "$QUIESCE_POST" || log "postflight quiesce: NOT QUIESCED (recorded, non-gating)"

# ---- manifest.tsv -> manifest.json (mirrors scripts/run_abba_matrix.sh's own converter) --
MANIFEST_JSON="$OUT/manifest.json"
python3 - "$MANIFEST_TSV" "$MANIFEST_JSON" "$ATTEMPT_ID" "$MOCK" "$NGEN" "$COLD_S" "$STEADY_S" "$SAMPLE_S" \
         "$(basename "$TELEM_CSV_COLD")" "$(basename "$TELEM_CSV_STEADY")" "${MODEL_DIR:-}" <<'PY'
import csv, json, sys
(src, dst, attempt_id, mock, ngen, cold_s, steady_s, sample_s,
 telem_cold, telem_steady, model_dir) = sys.argv[1:]
with open(src, newline='') as f:
    runs = list(csv.DictReader(f, delimiter='\t'))
for r in runs:
    r['trial'] = int(r['trial'])
manifest = {
    'attempt_id': attempt_id,
    'mock': bool(int(mock)),
    'model_dir': model_dir,
    'ngen': int(ngen),
    'cold_s': int(cold_s),
    'steady_s': int(steady_s),
    'sample_s': int(sample_s),
    'runs': runs,
    'telemetry': {'cold': f'telemetry/{telem_cold}', 'steady': f'telemetry/{telem_steady}'},
}
with open(dst, 'w') as f:
    json.dump(manifest, f, indent=2)
PY

REPORT_MD="$OUT/report.md"
python3 "$CDIR/tools/roofline_report.py" report --result-dir "$OUT" --out "$REPORT_MD"

if [[ -e "$OUT/TIMING_INVALID.json" || "${ROOFLINE_INVALID:-0}" == 1 ]]; then
  log "ROOFLINE INVALIDATED by the timing watchdog -- see $OUT/TIMING_INVALID.json. The report" \
      "above was generated over PARTIAL data for forensics and MUST NOT be trusted as a clean" \
      "measurement. Re-run once the machine is genuinely exclusive."
  exit 4
fi

log "roofline run complete: $REPORT_MD"
