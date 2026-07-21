#!/bin/bash
# Container quality gates for the mixed int4/int2 container, chained after the
# n=100 int4 reference bench. Fail-fast order:
#   gate A: long-generation collapse test (3 prompts x ~1600 tokens, ~1h)
#           -- prior GLM-5.2 quantization work showed length-dependent collapse
#           is the failure mode uniform int2 exhibits first; catching it here
#           saves the 8h bench when the container is catastrophically broken.
#   gate B: n=100 quality bench on the mixed container (same yardstick as the
#           int4 reference run, ~8h).
# Speed A/B (the +21% check, frozen-state 3-prompt) runs separately in daylight.
#
# Usage: run_container_gates.sh WAIT_PID
#   Waits for WAIT_PID (the running int4 reference bench) to exit first;
#   pass 0 to start immediately (engine must be idle).
set -uo pipefail

MIXED="${ILI_GATES_MIXED_DIR:-$HOME/models/GLM-5.2-iliria-mixed-280}"
INT4="${ILI_GATES_INT4_DIR:-$HOME/models/GLM-5.2-int4-with-int8-mtp}"
CDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$CDIR/bench-m5max/container-20260715"
mkdir -p "$OUT"
ATTEMPT_ID="${ILI_GATES_ATTEMPT_ID:-gates-$(date +%Y%m%d-%H%M%S)}"

log() { echo "[gates $(date '+%H:%M:%S')] $*"; }

WAIT_PID="${1:?usage: run_container_gates.sh WAIT_PID (0 = start now)}"
if [[ "$WAIT_PID" != 0 ]]; then
  log "waiting for PID $WAIT_PID (int4 reference bench) to exit"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
  sleep 60
fi
if pgrep -x glm >/dev/null; then
  log "ABORT: engine still busy after wait"; exit 1
fi

# The mixed container is freshly built: no expert-usage history, so AUTOPIN
# would start cold and the runs would be unrepresentatively slow. Routing is
# IDENTICAL between containers (router weights untouched by requantization),
# so the int4 container's histogram transfers 1:1. Copy, never move.
for f in .fa_usage .fa_usage.coding; do
  if [[ -f "$INT4/$f" && ! -f "$MIXED/$f" ]]; then
    cp "$INT4/$f" "$MIXED/$f" && log "copied $f into mixed container"
  fi
done

# ---- provenance manifest (executable-provenance system, scripts/provenance.sh) -------------
# Hard gate: gate B alone is an ~8h bench, and the whole point of this system is that no
# result is comparable without knowing exactly which binary/generated-source/container ran --
# refusing to start without a manifest is the same fail-closed posture this script already
# takes on "engine still busy after wait" above.
log "attempt_id=$ATTEMPT_ID recording provenance manifest before gate A"
if ! bash "$CDIR/scripts/provenance.sh" --attempt-id "$ATTEMPT_ID" --binary "$CDIR/glm" \
       --model-dir "$MIXED" --artifact-dir "$OUT"; then
  log "ABORT: provenance manifest emission failed -- refusing to start gate A without a record of what will run"
  exit 1
fi

# ---- gate A: long-generation collapse ----------------------------------------
log "gate A: long-generation collapse test (3 x ~1600 tok)"
PROMPTS=(
  "Write a detailed technical essay on how operating systems schedule processes, covering priorities, preemption, and fairness. Be thorough and continue until you have covered timer interrupts, context switch costs, and multicore load balancing."
  "Explain, step by step and at length, how a compiler turns C source code into an executable: lexing, parsing, type checking, IR, optimization passes, code generation, and linking. Include concrete examples throughout."
  "Describe the complete lifecycle of an HTTP request from typing a URL to rendered page: DNS, TCP, TLS, request routing, caching layers, HTML parsing, layout, and paint. Go deep on each stage."
)
COLLAPSE=0
for i in 0 1 2; do
  f="$OUT/collapse-p$((i+1)).txt"
  log "gate A prompt $((i+1))/3"
  # ILI_CPU_MOE_ALL=1: code review found the Metal MoE block builder takes its
  # format from the FIRST expert only (glm.c mfmt) — a mixed int4/int2 layer
  # whose first selected expert is int4 would read int2 bytes as int4 nibbles
  # (silent corruption). Until per-format sub-blocks or an int2 routed kernel
  # exist, the mixed container must run routed experts on CPU to be
  # interpretable. Metal attention stays on (no expert weights involved).
  ILI_CPU_MOE_ALL=1 ILI_NGEN=1600 bash "$CDIR/run-m5max-fast.sh" "$MIXED" run "${PROMPTS[$i]}" >"$f" 2>"$f.err"
  # Newline-AGNOSTIC repetition detector (scripts/collapse_detector.py). The
  # original inline heuristic here (chars<2000 || any 60-char LINE repeated >5x)
  # was newline-DEPENDENT and false-PASSED the 20260715 defective-int2 collapse:
  # the degenerate loops ('</think></think>...' on p1, "the C source-compiler's,"
  # on p2) were emitted as ONE unbroken line, so the awk repeated-line counter
  # logged max-repeated-line x1 (see $OUT/collapse-semantic-verdict.md and
  # tests/test_collapse_detector.py, which pins that near-miss as a regression
  # test). The detector keeps the same 2000-char whole-file floor, normalizes
  # all whitespace, and trips on repeated word-8-grams OR repeated 60-char
  # shingles (>5x either way); ANY nonzero exit (including a detector crash)
  # counts as collapse -- fail-closed. Its stderr (metrics + quoted repeated
  # text) lands in this script's own log for the human reviewer.
  # CSV format is a superset of the old one: columns 1-3
  # (prompt,chars,line60_reps) keep their exact old meaning;
  # word8_reps,shingle60_reps,verdict are appended.
  metrics=$(python3 "$CDIR/scripts/collapse_detector.py" "$f")
  det_rc=$?
  [[ -n "$metrics" ]] || metrics="0,0,0,0,DETECTOR-ERROR"
  log "gate A p$((i+1)): chars,line60,word8,shingle60,verdict = $metrics (detector rc=$det_rc)"
  echo "$((i+1)),$metrics" >>"$OUT/collapse-summary.csv"
  if [[ "$det_rc" -ne 0 ]]; then COLLAPSE=1; fi
done
if [[ "$COLLAPSE" == 1 ]]; then
  log "GATE A FAIL: short output or heavy repetition detected -- SKIPPING gate B (bench) to save 8h. Human review required: $OUT/collapse-*.txt"
  exit 3
fi
log "gate A PASS (no obvious collapse; human should still skim outputs)"

# ---- gate B: n=100 quality bench on the mixed container ----------------------
log "gate B: n=100 bench on mixed container (est ~8h)"
cd "$CDIR"
# Same mixed-format Metal hazard as gate A: force routed experts to CPU.
ILI_CPU_MOE_ALL=1 ./ili bench --model "$MIXED" --limit 100 >"$OUT/quality-mixed-n100.log" 2>&1
rc=$?
log "gate B done (rc=$rc) -> $OUT/quality-mixed-n100.log"
exit $rc
