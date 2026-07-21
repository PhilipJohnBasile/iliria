#!/bin/bash
# Stage-2 v3 canary: in-process 8K prefill (delay OFF) -> 32 warmup under treatment -> 256 measured.
# NO external priming passes (v2 behavior removed per reviewer). Setup cap = validity-only 2400s;
# measured-decode cap = 600s evaluated FROM THE RECORD (prefill never counts against decode).
set -uo pipefail
cd $REPO/c/bench-m5max/stage0-overhead-rerun
S2=$REPO/c/bench-m5max/stage2-v3
BIN=$REPO/c/glm
SHA=$(cat $S2/binary.sha)
MODEL=$HOME/models/GLM-5.2-int4-with-int8-mtp
python3 engine_run.py verify-hash --binary $BIN --expected-sha256 $SHA || { echo HASH-REFUSED; exit 2; }
source $REPO/c/scripts/timing_lock.sh
timing_lock_acquire s2v3-canary $$ || { echo LOCK-BUSY; exit 1; }
trap 'timing_lock_release' EXIT; trap 'exit 143' TERM
SNAP=$MODEL python3 engine_run.py run-one --binary $BIN --snap $MODEL \
  --ref $REPO/c/bench-m5max/stage2-v2/canary_trace_8k.json \
  --arg 8 --arg 8 --arg 8 --timeout-s 2400 \
  --env AUTOPIN=0 --env RAM_GB=2 --env ILI_IO_DELAY_US=0 \
  --env ILI_IO_DELAY_DECODE_US=16000 --env ILI_REPLAY_WARMUP=32 --env ILI_REPLAY_MEASURE=256 \
  > $S2/canary_v3_run.json 2>>$S2/canary_v3.err
echo "CANARY-RC=$?"
echo CANARY-V3-DONE
