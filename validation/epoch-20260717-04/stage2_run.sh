#!/bin/bash
set -uo pipefail
cd $REPO/c/bench-m5max/stage0-overhead-rerun
S2=$REPO/c/bench-m5max/stage2-v3
BIN=$REPO/c/glm; SHA=$(cat $REPO/c/bench-m5max/stage2-v3/binary.sha)   # single source of truth
MODEL=$HOME/models/GLM-5.2-int4-with-int8-mtp
source $REPO/c/scripts/timing_lock.sh
timing_lock_acquire stage2-v3 $$ || { echo LOCK-BUSY; exit 1; }
trap 'timing_lock_release' EXIT; trap 'exit 143' TERM
python3 - <<PY > $S2/exec_plan.txt
import json
p=json.load(open('$S2/plan.json'))
for b in p['blocks']:
    for lvl in b['order']: print(b['block'], b['trace'], lvl)
PY
while read BLK TRACE LVL; do
  python3 engine_run.py verify-hash --binary $BIN --expected-sha256 $SHA || { echo "HASH REFUSED block=$BLK"; exit 2; }
  SNAP=$MODEL python3 engine_run.py run-one --binary $BIN --snap $MODEL --ref $S2/$TRACE \
    --arg 8 --arg 8 --arg 8 --timeout-s 2400 --env AUTOPIN=0 --env RAM_GB=2 --env ILI_IO_DELAY_US=0 --env ILI_IO_DELAY_DECODE_US=$LVL --env ILI_REPLAY_WARMUP=32 --env ILI_REPLAY_MEASURE=256 \
    >> $S2/stage2_runs.jsonl 2>>$S2/stage2.err
  echo "done block=$BLK trace=$TRACE level=$LVL rc=$?"
done < $S2/exec_plan.txt
echo STAGE2-DONE
