#!/usr/bin/env bash
# Shared frozen-state-matrix machinery, sourced by fight_card.sh (the Saturday
# combination-sweep + ablation FIGHT harness).
#
# EXTRACTION NOTE: the functions below are lifted, behavior-for-behavior, from
# ab-m5max-k6-matrix.sh (the house frozen-state matrix: swap_mb, snapshot_system,
# purge_cache, the .fa_usage snapshot/restore pair, and the 3 held-out coding
# prompts/labels). fight_card.sh needs a MORE GENERAL per-row runner than that
# script's own run_case() -- ab-m5max-k6-matrix.sh hardcodes exactly one
# candidate-vs-off axis (pilot|metal4|pstate), whereas a fight card row is an
# arbitrary N-lever combination (the STACK, an ablation, or a revival addition)
# -- so this file factors out the parts that ARE identical (per the design's own
# "reuse... invoke or adapt, do not fork-and-drift: if adaptation is needed,
# factor a shared helper") rather than pasting a second, silently-diverging
# copy inside fight_card.sh.
#
# ab-m5max-k6-matrix.sh itself is intentionally left untouched here: it has no
# --dry-run mode of its own (every invocation runs the real engine), so a
# refactor of it could not be verified end-to-end under tonight's no-real-
# engine-runs constraint. Converging it onto this shared file is a reasonable
# reversible follow-up, not done here to avoid an unverifiable change to a
# load-bearing script the same night this harness is first proven.
#
# Callers must `cd` to c/ (or otherwise ensure relative paths resolve) before
# sourcing, exactly like ab-m5max-k6-matrix.sh's own SCRIPT_DIR/cd convention.
set -uo pipefail

# ---- the 3 held-out coding prompts (verbatim from ab-m5max-k6-matrix.sh) ------------------
FIGHT_PROMPTS=(
  "Implement a cancellation-safe bounded MPSC queue in Rust using atomics. Explain the memory ordering and include tests."
  "A Godot 4.7 GDScript character controller allocates every physics frame and occasionally tunnels through ramps. Diagnose it and provide a corrected implementation."
  "Review a TypeScript WebSocket reconnection manager for race conditions, leaked timers, stale closures, and exponential-backoff errors. Return a patch and tests."
)
FIGHT_LABELS=(rust_queue godot_controller typescript_websocket)

# ---- system telemetry (verbatim logic from ab-m5max-k6-matrix.sh) -------------------------
fight_swap_mb() {
  local raw
  raw="$(sysctl -n vm.swapusage 2>/dev/null || true)"
  python3 - "$raw" <<'PY'
import re,sys
m=re.search(r"used = ([0-9.]+)([MG])",sys.argv[1])
if not m: print("0"); raise SystemExit
v=float(m.group(1)); print(v*1024 if m.group(2)=="G" else v)
PY
}

fight_snapshot_system() {
  local prefix="$1" outdir="$2"
  memory_pressure > "$outdir/${prefix}-memory-pressure.txt" 2>&1 || true
  vm_stat > "$outdir/${prefix}-vm-stat.txt" 2>&1 || true
  sysctl vm.swapusage > "$outdir/${prefix}-swap.txt" 2>&1 || true
  ps -axo pid,ppid,etime,%cpu,%mem,rss,command > "$outdir/${prefix}-ps.txt" 2>&1 || true
}

fight_purge_cache() {
  sync
  if sudo -n purge >/dev/null 2>&1; then
    sleep 3
    return 0
  fi
  return 1
}

# ---- .fa_usage frozen-state snapshot/restore (per row/arm; convention read from
#      ab-m5max-k6-matrix.sh: a drifting hot-set changes CPU-grouped vs Metal expert
#      placement, whose kernel-family rounding difference can flip greedy tokens, so
#      every arm must boot from the same snapshot or the hash gate is unpassable). ------
FIGHT_USAGE_FILES=()
fight_snapshot_usage() {
  local model_dir="$1" profile="$2" snap_dir="$3" f
  mkdir -p "$snap_dir"
  FIGHT_USAGE_FILES=()
  for f in "$model_dir/.fa_usage" "$model_dir/.fa_usage.$profile"; do
    if [[ -f "$f" ]]; then
      cp "$f" "$snap_dir/$(basename "$f")"
      FIGHT_USAGE_FILES+=("$f")
    fi
  done
}

fight_restore_usage() {
  local snap_dir="$1" f
  (( ${#FIGHT_USAGE_FILES[@]} )) || return 0
  for f in "${FIGHT_USAGE_FILES[@]}"; do
    cp "$snap_dir/$(basename "$f")" "$f"
  done
}

# ---- quiesce gate (dry-run-tolerant convention from run_abba_matrix.sh's run_quiesce():
#      real mode hard-gates on FAIL; --dry-run still runs the check for real (so the log
#      it writes is real and its own --dry-run proof is genuine) but never blocks, since a
#      dry run risks no real engine and this machine legitimately runs other things most of
#      the day). Caller passes the quiesce binary explicitly (ILI_FIGHT_QUIESCE_BIN in
#      fight_card.sh) so tests can substitute a fast fixture instead of the real,
#      tens-of-real-seconds quiesce_check.sh -- same convention as
#      c/tests/test_evening_orchestrator.py's ILI_EVENING_QUIESCE_BIN override. -------
# Returns 0 if quiesced OR (dry_run==1 and not quiesced -- logged, not fatal);
# returns 1 only in real mode when quiesce genuinely failed (caller must treat as fatal).
fight_quiesce_gate() {
  local quiesce_bin="$1" label="$2" outfile="$3" dry_run="$4" rc
  ILI_DIRECT=1 bash "$quiesce_bin" > "$outfile" 2>&1
  rc=$?
  if (( rc == 0 )); then
    echo "[fight] quiesce ($label): PASS"
    return 0
  fi
  if [[ "$dry_run" == 1 ]]; then
    echo "[fight] quiesce ($label): NOT QUIESCED -- continuing (--dry-run has no engine at risk)"
    return 0
  fi
  echo "[fight] quiesce ($label): FAILED -- see $outfile" >&2
  return 1
}

# ---- hash/rate extraction for bash-side gating decisions (Python does the authoritative
#      parsing via tools/m5max_log_parse.py; these are lightweight bash-side mirrors used
#      only for immediate per-arm console logging, never for the report itself). ----------
fight_log_hash() {
  grep -o 'output token hash: [0-9a-fA-F]*' "$1" 2>/dev/null | tail -1 | awk '{print $NF}'
}

fight_log_tok_s() {
  grep -oE '\([0-9.]+ tok/s\)' "$1" 2>/dev/null | tail -1 | tr -d '()' | awk '{print $1}'
}
