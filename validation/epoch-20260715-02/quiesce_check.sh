#!/bin/bash
# Executable quiesce precondition for timing gates (ABBA, roofline, overlap
# instrumentation). FAIL-CLOSED: missing or unparseable telemetry is a
# failure, never a pass. Exits 0 only when every condition affirmatively
# passes; prints all values for the gate log (run before AND after each arm).
#
# SSD temperature has no direct sensor tool here; proxies = sustained disk
# idle + thermal-pressure state. DIRECT=1 and the purge/warmup sequence are
# the calling matrix's job; this script verifies the environment flag only.
set -u

FAIL=0
note() { echo "[quiesce] $*"; }
bad()  { echo "[quiesce] FAIL: $*"; FAIL=1; }

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_num()  { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }

# 1. No auxiliary shard readers / analytics (known names) ...
if pgrep -f "measure_expert|entropy|build_mixed|simulate_bytes|quant_error|long_ctx_profile" >/dev/null; then
  bad "offline reader/analytics process running"
else note "no known analytics processes"; fi

# ... and no ARBITRARY process holding files under the model/container dirs.
for D in $HOME/models/GLM-5.2-int4-with-int8-mtp $HOME/models/GLM-5.2-iliria-mixed-280; do
  HOLDERS=$(lsof +D "$D" 2>/dev/null | awk 'NR>1{print $1, $2}' | sort -u)
  if [ -n "$HOLDERS" ]; then
    bad "open files under $D by: $(echo "$HOLDERS" | tr '\n' ' ')"
  fi
done
[ "$FAIL" -eq 0 ] && note "no process holds model/container files"

# 2. No competing engine process.
if pgrep -x glm >/dev/null; then bad "a glm engine process is already running"
else note "no competing engine"; fi

# 3. Disk throughput below threshold for N consecutive samples (fail-closed).
THRESH_MBS=${QUIESCE_DISK_MBS:-50}
SAMPLES=${QUIESCE_SAMPLES:-6}       # 6 x 5s = 30s quiet
QUIET=0
for i in $(seq 1 "$SAMPLES"); do
  MBS=$(iostat -d -w 5 -c 2 disk0 2>/dev/null | tail -1 | awk '{print $3}')
  if ! is_num "${MBS:-x}"; then bad "disk telemetry unavailable (iostat gave '${MBS:-}')"; break; fi
  awk "BEGIN{exit !($MBS < $THRESH_MBS)}" && QUIET=$((QUIET+1))
done
if [ "$QUIET" -lt "$SAMPLES" ]; then
  bad "disk not quiet: $QUIET/$SAMPLES samples under ${THRESH_MBS}MB/s"
else note "disk quiet ($SAMPLES/$SAMPLES under ${THRESH_MBS}MB/s)"; fi

# 4. Aggregate CPU usage near idle (fail-closed). The 'id' column position is
# derived from iostat's own header line, never hardcoded — a layout change
# fails closed instead of silently reading a different column.
IOSTAT_OUT=$(iostat -c 2 -w 3 2>/dev/null)
IDCOL=$(echo "$IOSTAT_OUT" | awk '/id/{for(i=1;i<=NF;i++) if($i=="id"){print i; exit}}' | head -1)
if ! is_uint "${IDCOL:-x}"; then bad "CPU telemetry unavailable (no 'id' column in iostat header)"
else
  CPUIDLE=$(echo "$IOSTAT_OUT" | tail -1 | awk -v c="$IDCOL" '{print $c}')
  if ! is_num "${CPUIDLE:-x}"; then bad "CPU telemetry unavailable (unparseable idle field)"
  elif awk "BEGIN{exit !($CPUIDLE < 70)}"; then bad "CPU not idle enough (idle=${CPUIDLE}%, need >=70%)"
  else note "CPU idle ${CPUIDLE}% (id col $IDCOL, header-verified)"; fi
fi

# 5. Thermal pressure nominal (fail-closed).
THERM_RAW=$(pmset -g therm 2>/dev/null)
THERM=$(echo "$THERM_RAW" | grep -i "CPU_Speed_Limit" | grep -o '[0-9]*' | head -1)
if [ -z "$THERM_RAW" ]; then bad "thermal telemetry unavailable (pmset -g therm empty)"
elif [ -n "$THERM" ] && is_uint "$THERM" && [ "$THERM" -lt 100 ]; then
  bad "thermal throttling active (CPU_Speed_Limit=$THERM)"
else note "thermal nominal"; fi

# 6. Memory pressure normal + swap sane (fail-closed).
MEMLVL=$(memory_pressure -Q 2>/dev/null | grep -o "System-wide memory free percentage: [0-9]*" | grep -o '[0-9]*$')
if ! is_uint "${MEMLVL:-x}"; then bad "memory-pressure telemetry unavailable"
elif [ "$MEMLVL" -lt 10 ]; then bad "memory free ${MEMLVL}% — under pressure"
else note "memory free ${MEMLVL}%"; fi
# Swap ACTIVITY, not absolute usage (macOS keeps swap allocated long after
# pressure ends; what contaminates timing is ongoing paging, not history).
PO1=$(vm_stat 2>/dev/null | awk '/Pageouts/{gsub("\\.","",$2); print $2}')
sleep 5
PO2=$(vm_stat 2>/dev/null | awk '/Pageouts/{gsub("\\.","",$2); print $2}')
if ! is_uint "${PO1:-x}" || ! is_uint "${PO2:-x}"; then bad "swap/paging telemetry unavailable"
elif [ $((PO2 - PO1)) -gt 100 ]; then bad "active paging: $((PO2-PO1)) pageouts in 5s"
else note "paging quiet ($((PO2-PO1)) pageouts in 5s); swap used: $(sysctl -n vm.swapusage 2>/dev/null | grep -o 'used = [0-9.]*M' || echo 'n/a') (recorded, not gated)"; fi

# 7. On AC power (fail-closed on unreadable state).
BATT_RAW=$(pmset -g batt 2>/dev/null)
if [ -z "$BATT_RAW" ]; then bad "power telemetry unavailable"
elif echo "$BATT_RAW" | grep -q "AC Power"; then note "on AC power"
else bad "on battery"; fi

# 8. Environment checks: the timing runner must EXPLICITLY export ILI_DIRECT=1.
# Unset is a failure (fail-closed), not an implied default.
if [ "${QUIESCE_EXPECT_DIRECT:-1}" = 1 ]; then
  if [ "${ILI_DIRECT+x}" != x ]; then bad "ILI_DIRECT is unset — timing runner must export it explicitly"
  elif [ "$ILI_DIRECT" != 1 ]; then bad "ILI_DIRECT=$ILI_DIRECT (must be 1)"
  else note "ILI_DIRECT=1 (explicit)"; fi
fi

# Ambient record for the gate log.
note "timestamp: $(date '+%Y-%m-%dT%H:%M:%S')"
note "battery: $(echo "${BATT_RAW:-}" | grep -o '[0-9]*%' | head -1)"
note "load: $(uptime | awk -F'load averages:' '{print $2}')"

if [ "$FAIL" -eq 1 ]; then
  echo "[quiesce] NOT QUIESCED — do not start the timing gate"; exit 1
fi
echo "[quiesce] ALL CONDITIONS PASS — timing gate may begin"
