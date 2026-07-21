#!/usr/bin/env bash
# Saturday's FIGHT harness -- the combination-sweep + ablation runner that decides shipped
# defaults. Config-driven from a fight-card JSON (default: fight_card.default.json, next to
# this script; see that file's own header for the STACK/ABLATIONS/REVIVAL schema).
#
# Per card row ("arm"), runs the SAME frozen-state 3-prompt matrix as ab-m5max-k6-matrix.sh
# (3 held-out coding prompts x warm/cold cache): quiesce-gated, attempt_id-stamped,
# .fa_usage snapshot/restore per arm (frozen-state convention), hash recorded per run.
# scripts/fight_common.sh factors out the parts of ab-m5max-k6-matrix.sh's own machinery
# that generalize unchanged (prompts, swap/system telemetry, usage snapshot/restore,
# quiesce gate) -- see that file's own header for why ab-m5max-k6-matrix.sh itself is left
# untouched. ab-m5max-k6-matrix.sh's run_case() itself does NOT generalize (it hardcodes a
# single off-vs-one-candidate axis); a fight-card row is an arbitrary N-lever combination,
# so this script's run_arm() is new logic, built ONLY on the shared low-level primitives.
#
# ABBA interleaving, generalized from 2 arms to N: ab-m5max-k6-matrix.sh alternates
# (off,candidate)/(candidate,off) across trials to spread monotonic drift (thermal, disk
# fragmentation, etc.) evenly across both conditions. With N arms (baseline, stack, one
# ablation per stack lever, any triggered revival rows) the same idea generalizes to
# ping-ponging the FULL arm order: odd trials run the arms forward, even trials run them
# reversed -- so no single arm systematically owns the "goes first" or "goes last" slot.
#
# Hash gating: fight_report.py (not this script) does the actual gating, and DELIBERATELY
# does not reuse summarize_m5max_k6_matrix.py's "exactly one hash per (cache,prompt) across
# every mode" rule -- that rule assumes every candidate axis is a byte-identical mechanism
# (true for PILOT/METAL4/persistent-state, per their own docs/performance-theory.json
# entries), but this card's STACK includes ILI_METAL_PREFILL, which the project's own
# kernel-family-rounding policy explicitly allows to fork the output hash vs a CPU baseline
# (README.md, docs/PERFORMANCE_THEORY.json s-row-projection-gemms-prefill-gate-1 /
# dsa-sparse-indexer). fight_report.py instead requires SAME-CONFIG self-consistency
# (every trial of the SAME arm at a given cache/prompt must hash-match) and separately
# RECORDS cross-arm hash forks, annotated against each arm's kernel-family fingerprint
# (docs/performance-theory.json's own "faithful forward of unmodified weights" ruling).
#
# Usage:
#   bash scripts/fight_card.sh [--dry-run] [--card PATH]
#                              [--set-measurement metric=value]... [--force-revival name[,name...]]
#   bash scripts/fight_card.sh -h|--help
#
#   --dry-run             never runs a real engine command (no `make`, no `build-m5max-lab.sh`,
#                          no `run-m5max-fast.sh`, no `glm`). scripts/fight_mock_engine.sh
#                          stands in for the engine (see that file's header): same log format,
#                          same downstream parser, so the WHOLE pipeline -- card resolution,
#                          quiesce gate, usage snapshot/restore, ABBA-generalized row
#                          sequencing, hash recording, report generation -- is proven without
#                          real hardware. quiesce_check.sh itself still runs for real (recorded,
#                          never blocking), same carve-out as run_abba_matrix.sh/
#                          evening_orchestrator.sh.
#   --card PATH            fight-card JSON (default: fight_card.default.json next to this script)
#   --set-measurement M=V  supply a measured value a "measured_metric" revival trigger needs
#                          (repeatable). Fail-closed: an unsupplied metric leaves that row SKIPPED.
#   --force-revival LIST   comma-separated revival row names to force-run regardless of their
#                          trigger (TEST/OVERRIDE ONLY -- never use for a real campaign; this is
#                          how --dry-run proves the "trigger met" code path even on a night both
#                          real graveyard triggers are unmet).
#
# Environment (all optional):
#   ILI_FIGHT_CARD            default: fight_card.default.json next to this script
#   ILI_FIGHT_ATTEMPT_ID       default: fight-<timestamp>
#   ILI_FIGHT_RESULT_DIR       default: bench-m5max/<attempt_id>
#   ILI_FIGHT_CAMPAIGN_LOG     default: bench-m5max/campaign-log.md (campaign-state.json
#                               convention: one factual timestamped line per step)
#   ILI_FIGHT_QUIESCE_BIN      default: scripts/quiesce_check.sh (override for tests --
#                               same convention as ILI_EVENING_QUIESCE_BIN)
#   ILI_FIGHT_RUNNER_BIN       default: scripts/fight_mock_engine.sh under --dry-run,
#                               run-m5max-fast.sh otherwise
#   ILI_FIGHT_LAB_BUILD_BIN    default: build-m5max-lab.sh
#   ILI_FIGHT_SKIP_LAB_BUILD   default: 0 (real mode only; skip iff a fresh lab binary is
#                               already known-built)
#   ILI_FIGHT_PERF_THEORY_JSON default: ../docs/performance-theory.json (repo root)
#   ILI_FIGHT_PASSES/_NGEN/_CACHE_STATES   override the card's own matrix.* values
#   ILI_FIGHT_QUIESCE_EVERY_ARM  default: 0; 1 forces quiesce before every single arm
#                               instead of once per (cache,prompt) cell (slower, stricter)
set -uo pipefail   # NOT -e: this codebase's engine-sequencing scripts manage control flow
                   # explicitly via return codes (see run_abba_matrix.sh's own comment).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CDIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$CDIR"
# shellcheck source=fight_common.sh
source "$SCRIPT_DIR/fight_common.sh"

log() { echo "[fight $(date '+%H:%M:%S')] $*"; }
die() { echo "[fight $(date '+%H:%M:%S')] FATAL: $*" >&2; exit 1; }

usage() { sed -n '2,73p' "${BASH_SOURCE[0]}"; }

# ---- args ----------------------------------------------------------------------------
DRY_RUN=0
CARD="${ILI_FIGHT_CARD:-$SCRIPT_DIR/fight_card.default.json}"
MEASUREMENTS=()
FORCE_REVIVAL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --card) CARD="$2"; shift 2 ;;
    --set-measurement) MEASUREMENTS+=("--set-measurement" "$2"); shift 2 ;;
    --force-revival) FORCE_REVIVAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -f "$CARD" ]] || die "fight card not found: $CARD"

# ---- configuration (env-tunable; see usage() above) -----------------------------------
ATTEMPT_ID="${ILI_FIGHT_ATTEMPT_ID:-fight-$(date +%Y%m%d-%H%M%S)}"
BENCH_DIR="$CDIR/bench-m5max"
OUT="${ILI_FIGHT_RESULT_DIR:-$BENCH_DIR/$ATTEMPT_ID}"
CAMPAIGN_LOG="${ILI_FIGHT_CAMPAIGN_LOG:-$BENCH_DIR/campaign-log.md}"
QUIESCE_BIN="${ILI_FIGHT_QUIESCE_BIN:-$SCRIPT_DIR/quiesce_check.sh}"
if [[ "$DRY_RUN" == 1 ]]; then
  RUNNER_BIN="${ILI_FIGHT_RUNNER_BIN:-$SCRIPT_DIR/fight_mock_engine.sh}"
else
  RUNNER_BIN="${ILI_FIGHT_RUNNER_BIN:-$CDIR/run-m5max-fast.sh}"
fi
LAB_BUILD_BIN="${ILI_FIGHT_LAB_BUILD_BIN:-$CDIR/build-m5max-lab.sh}"
SKIP_LAB_BUILD="${ILI_FIGHT_SKIP_LAB_BUILD:-0}"
PERF_THEORY_JSON="${ILI_FIGHT_PERF_THEORY_JSON:-$CDIR/../docs/performance-theory.json}"
QUIESCE_EVERY_ARM="${ILI_FIGHT_QUIESCE_EVERY_ARM:-0}"

mkdir -p "$OUT/logs" "$OUT/artifacts" "$OUT/system"
PLAN_JSON="$OUT/plan.json"
MANIFEST_TSV="$OUT/manifest.tsv"
printf 'arm\tcache_state\tprompt\ttrial\tlog\n' > "$MANIFEST_TSV"

campaign_log() {
  mkdir -p "$(dirname "$CAMPAIGN_LOG")"
  printf -- '- %s | fight_card(%s) | %s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$ATTEMPT_ID" "$1" >> "$CAMPAIGN_LOG"
}

log "attempt_id=$ATTEMPT_ID dry_run=$DRY_RUN card=$CARD out=$OUT"
campaign_log "fight_card starting (card=$CARD, dry_run=$DRY_RUN)"

# ---- resolve the card into a concrete plan (baseline + stack + ablations + triggered
#      revival rows); pure config step, identical in --dry-run and real mode. -------------
resolve_plan() {
  python3 "$CDIR/tools/fight_plan.py" "$CARD" \
    --perf-theory-json "$PERF_THEORY_JSON" \
    --force-revival "$FORCE_REVIVAL" \
    "${MEASUREMENTS[@]+"${MEASUREMENTS[@]}"}"
}
if ! resolve_plan > "$PLAN_JSON" 2>"$OUT/plan-resolve.err"; then
  die "fight_plan.py failed to resolve $CARD -- see $OUT/plan-resolve.err"
fi

plan_field() {  # $1 = dotted top-level key (no nesting needed here)
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$PLAN_JSON" "$1"
}
plan_matrix_field() {  # $1 = key under "matrix"
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['matrix'][sys.argv[2]])" "$PLAN_JSON" "$1"
}
plan_arm_field() {  # $1 = arm name, $2 = field
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['arms'][sys.argv[2]][sys.argv[3]])" "$PLAN_JSON" "$1" "$2"
}
plan_arm_env_kv() {  # $1 = arm name -> KEY=VALUE lines
  python3 -c "
import json, sys
env = json.load(open(sys.argv[1]))['arms'][sys.argv[2]]['env']
for k, v in env.items():
    print(f'{k}={v}')
" "$PLAN_JSON" "$1"
}
plan_list_lines() {  # $1 = top-level list field -> newline-separated items
  python3 -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))[sys.argv[2]]))" "$PLAN_JSON" "$1"
}

ARM_ORDER=()
while IFS= read -r line; do [[ -n "$line" ]] && ARM_ORDER+=("$line"); done < <(plan_list_lines arm_order)
(( ${#ARM_ORDER[@]} > 0 )) || die "resolved plan has zero arms -- check $CARD"

SKIPPED_COUNT="$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['skipped']))" "$PLAN_JSON")"
if (( SKIPPED_COUNT > 0 )); then
  log "revival rows skipped this run ($SKIPPED_COUNT):"
  python3 -c "
import json, sys
for s in json.load(open(sys.argv[1]))['skipped']:
    print(f\"  - {s['name']}: {s['reason']}\")
" "$PLAN_JSON"
fi
log "resolved arms (${#ARM_ORDER[@]}): ${ARM_ORDER[*]}"

NEEDS_LAB_BUILD="$(plan_field needs_lab_build)"
PASSES="${ILI_FIGHT_PASSES:-$(plan_matrix_field passes)}"
NGEN="${ILI_FIGHT_NGEN:-$(plan_matrix_field ngen)}"
HOTSET_PROFILE="$(plan_matrix_field hotset_profile)"
CACHE_STATES_RAW=()
while IFS= read -r line; do [[ -n "$line" ]] && CACHE_STATES_RAW+=("$line"); done < <(
  if [[ -n "${ILI_FIGHT_CACHE_STATES:-}" ]]; then printf '%s\n' $ILI_FIGHT_CACHE_STATES
  else python3 -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['matrix']['cache_states']))" "$PLAN_JSON"
  fi
)
QUIESCE_GRANULARITY="$(plan_field quiesce_granularity)"
[[ "$QUIESCE_EVERY_ARM" == 1 ]] && QUIESCE_GRANULARITY="per_arm"

# ---- lab build (always invoked in real mode, mirroring ab-m5max-k6-matrix.sh's own
#      unconditional build-m5max-lab.sh call -- ILI_PILOT_METRICS=1 in fixed_env alone
#      needs that build's patch_m5max_pilot_metrics.py patch, independent of whether this
#      plan's stack happens to need the METAL4-specific patches too). ---------------------
if [[ "$DRY_RUN" == 1 ]]; then
  log "dry-run: would run 'ILI_M5_LAB_METAL4=$([[ "$NEEDS_LAB_BUILD" == True ]] && echo 1 || echo 0) bash $LAB_BUILD_BIN' -- skipped"
elif [[ "$SKIP_LAB_BUILD" == 1 ]]; then
  log "ILI_FIGHT_SKIP_LAB_BUILD=1: skipping lab build (operator attests a fresh binary is already built)"
else
  metal4_flag=0; [[ "$NEEDS_LAB_BUILD" == True ]] && metal4_flag=1
  log "building lab engine (ILI_M5_LAB_METAL4=$metal4_flag)"
  if ! ILI_M5_LAB_METAL4="$metal4_flag" bash "$LAB_BUILD_BIN" > "$OUT/lab-build.log" 2>&1; then
    die "lab build FAILED -- see $OUT/lab-build.log"
  fi
fi

# ---- preflight quiesce (dry-run tolerant; hard gate in real mode) --------------------
QUIESCE_PRE="$OUT/system/${ATTEMPT_ID}-quiesce-preflight.txt"
if ! fight_quiesce_gate "$QUIESCE_BIN" preflight "$QUIESCE_PRE" "$DRY_RUN"; then
  die "preflight quiesce FAILED -- refusing to start (see $QUIESCE_PRE)"
fi

# ---- .fa_usage snapshot per distinct container actually used in this plan ------------
CONTAINER_NAMES=()
while IFS= read -r line; do [[ -n "$line" ]] && CONTAINER_NAMES+=("$line"); done < <(
  python3 -c "
import json, sys
p = json.load(open(sys.argv[1]))
print('\n'.join(sorted({a['container'] for a in p['arms'].values()})))
" "$PLAN_JSON"
)

declare -a CONTAINER_PATHS=()
for name in "${CONTAINER_NAMES[@]}"; do
  path="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['containers'][sys.argv[2]])" "$PLAN_JSON" "$name")"
  if [[ "$DRY_RUN" == 1 && ! -d "$path" ]]; then
    # Synthetic fixture container so the usage-snapshot/restore machinery has something
    # real to operate on -- same convention as run_abba_matrix.sh's own --dry-run fixture.
    path="$OUT/artifacts/fake-container-$name"
    mkdir -p "$path"
    [[ -f "$path/.fa_usage" ]] || echo "dry-run-fixture" > "$path/.fa_usage"
    [[ -f "$path/.fa_usage.$HOTSET_PROFILE" ]] || echo "dry-run-fixture-profile" > "$path/.fa_usage.$HOTSET_PROFILE"
    log "dry-run: container '$name' not found on disk -- using synthetic fixture at $path"
  else
    [[ -d "$path" ]] || die "container '$name' path not found: $path"
  fi
  CONTAINER_PATHS+=("$path")
done

snap_dir_for() {  # $1 = container name -> its snapshot dir (also (re)computes the path)
  echo "$OUT/system/${ATTEMPT_ID}-usage-snapshot-$1"
}
for i in "${!CONTAINER_NAMES[@]}"; do
  fight_snapshot_usage "${CONTAINER_PATHS[$i]}" "$HOTSET_PROFILE" "$(snap_dir_for "${CONTAINER_NAMES[$i]}")"
done
container_path_for() {  # $1 = container name
  local i
  for i in "${!CONTAINER_NAMES[@]}"; do
    [[ "${CONTAINER_NAMES[$i]}" == "$1" ]] && { echo "${CONTAINER_PATHS[$i]}"; return 0; }
  done
  die "internal error: unknown container name $1"
}
restore_usage_for() {  # $1 = container name
  local snap_dir path
  snap_dir="$(snap_dir_for "$1")"
  path="$(container_path_for "$1")"
  # fight_snapshot_usage sets the global FIGHT_USAGE_FILES array as a side effect the last
  # time it ran for THIS container; re-derive it fresh here so restoring container B never
  # accidentally restores container A's file list (matters once >1 container is in play).
  local -a files=()
  local f
  for f in "$path/.fa_usage" "$path/.fa_usage.$HOTSET_PROFILE"; do
    [[ -f "$snap_dir/$(basename "$f")" ]] && files+=("$f")
  done
  local saved=("${FIGHT_USAGE_FILES[@]+"${FIGHT_USAGE_FILES[@]}"}")
  FIGHT_USAGE_FILES=("${files[@]+"${files[@]}"}")
  fight_restore_usage "$snap_dir"
  FIGHT_USAGE_FILES=("${saved[@]+"${saved[@]}"}")
}

# Best-effort safety net: restore every snapshotted container's usage file if this script
# dies mid-arm (signal, unexpected error) -- never leave a mutated .fa_usage behind.
on_exit() {
  local rc=$?
  local name
  for name in "${CONTAINER_NAMES[@]+"${CONTAINER_NAMES[@]}"}"; do
    restore_usage_for "$name" 2>/dev/null || true
  done
  exit "$rc"
}
trap on_exit EXIT INT TERM

# ---- cold-cache availability (mirrors ab-m5max-k6-matrix.sh's own actual_states filter;
#      under --dry-run, cold is exercised unconditionally -- purge itself is never invoked
#      for real, so real passwordless-sudo availability on the dry-run machine is
#      irrelevant to proving the SEQUENCING). ------------------------------------------
CACHE_STATES=()
for cache in "${CACHE_STATES_RAW[@]}"; do
  case "$cache" in
    warm) CACHE_STATES+=(warm) ;;
    cold)
      if [[ "$DRY_RUN" == 1 ]]; then
        CACHE_STATES+=(cold)
      elif sudo -n true >/dev/null 2>&1 && command -v purge >/dev/null 2>&1; then
        CACHE_STATES+=(cold)
      else
        log "warning: skipping cold arms because passwordless sudo/purge is unavailable"
      fi ;;
    *) die "invalid cache state: $cache" ;;
  esac
done

# ---- per-arm run --------------------------------------------------------------------
run_arm() {
  local arm="$1" label="$2" prompt="$3" cache="$4" trial="$5" measured="$6"
  local container container_path stem log_path rc

  container="$(plan_arm_field "$arm" container)"
  container_path="$(container_path_for "$container")"
  stem="${cache}-${label}-t${trial}-${arm}"
  log_path="$OUT/logs/${ATTEMPT_ID}-${stem}.log"

  if [[ "$cache" == cold ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then
      log "dry-run: would purge cache before $stem"
    elif ! fight_purge_cache; then
      die "cold-cache arm '$arm' requested but passwordless 'sudo purge' is unavailable"
    fi
  fi

  if [[ "$QUIESCE_GRANULARITY" == per_arm ]]; then
    if ! fight_quiesce_gate "$QUIESCE_BIN" "$stem" "$OUT/system/${ATTEMPT_ID}-${stem}-quiesce.txt" "$DRY_RUN"; then
      die "quiesce FAILED before arm '$stem'"
    fi
  fi

  restore_usage_for "$container"

  local -a env_args=()
  while IFS= read -r kv; do env_args+=("$kv"); done < <(plan_arm_env_kv "$arm")
  env_args+=("ILI_NGEN=$NGEN" "ILI_HOTSET_PROFILE=$HOTSET_PROFILE" "FIGHT_MOCK_TRIAL_SALT=${cache}-${trial}")

  fight_snapshot_system "${stem}-before" "$OUT/system"
  local before_swap; before_swap="$(fight_swap_mb)"
  (
    set +e
    echo "FIGHT-META: attempt_id=$ATTEMPT_ID arm=$arm cache=$cache prompt=$label trial=$trial measured=$measured"
    echo "FIGHT-SWAP-BEFORE-MB: $before_swap"
    env "${env_args[@]}" bash "$RUNNER_BIN" "$container_path" run "$prompt"
    rc=$?
    echo "FIGHT-SWAP-AFTER-MB: $(fight_swap_mb)"
    echo "FIGHT-EXIT: $rc"
    exit "$rc"
  ) > "$log_path" 2>&1 || rc=$?
  rc="${rc:-0}"
  fight_snapshot_system "${stem}-after" "$OUT/system"

  if (( rc != 0 )); then
    die "arm '$stem' failed (exit $rc) -- see $log_path"
  fi

  if [[ "$measured" == 1 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$arm" "$cache" "$label" "$trial" "logs/${ATTEMPT_ID}-${stem}.log" >> "$MANIFEST_TSV"
  fi
  log "arm '$stem' done (rc=0, hash=$(fight_log_hash "$log_path"), tok/s=$(fight_log_tok_s "$log_path"))"
}

reversed_arms() {
  local i out=()
  for (( i=${#ARM_ORDER[@]}-1; i>=0; i-- )); do out+=("${ARM_ORDER[$i]}"); done
  printf '%s\n' "${out[@]+"${out[@]}"}"
}

# ---- main matrix loop: cache x prompt x (per-arm warmup, warm only) x trial x
#      ABBA-generalized arm rotation ----------------------------------------------------
for cache in "${CACHE_STATES[@]+"${CACHE_STATES[@]}"}"; do
  for pi in "${!FIGHT_LABELS[@]}"; do
    label="${FIGHT_LABELS[$pi]}"
    prompt="${FIGHT_PROMPTS[$pi]}"

    if [[ "$QUIESCE_GRANULARITY" == per_cell ]]; then
      if ! fight_quiesce_gate "$QUIESCE_BIN" "cell-${cache}-${label}" \
             "$OUT/system/${ATTEMPT_ID}-quiesce-${cache}-${label}.txt" "$DRY_RUN"; then
        die "quiesce FAILED before cell cache=$cache prompt=$label"
      fi
    fi

    if [[ "$cache" == warm ]]; then
      # One unmeasured warmup PER ARM: with N>2 arms, ANY arm getting the first crack at a
      # cold page cache would bias it high relative to the rest -- generalizes
      # ab-m5max-k6-matrix.sh's "symmetric unmeasured warmups" comment from 2 arms to N.
      for arm in "${ARM_ORDER[@]}"; do
        run_arm "$arm" "$label" "$prompt" "$cache" 0 0
      done
    fi

    for (( trial=1; trial<=PASSES; trial++ )); do
      if (( trial % 2 )); then
        order=("${ARM_ORDER[@]}")
      else
        order=()
        while IFS= read -r a; do order+=("$a"); done < <(reversed_arms)
      fi
      for arm in "${order[@]}"; do
        log "[$(date '+%F %T')] cache=$cache prompt=$label trial=$trial arm=$arm"
        run_arm "$arm" "$label" "$prompt" "$cache" "$trial" 1
      done
    done
  done
done

# ---- manifest + report ----------------------------------------------------------------
python3 - "$MANIFEST_TSV" "$OUT/manifest.json" "$ATTEMPT_ID" "$PASSES" "$NGEN" "$DRY_RUN" "$QUIESCE_GRANULARITY" <<'PY'
import csv, json, sys
tsv, dst, attempt_id, passes, ngen, dry_run, quiesce_granularity = sys.argv[1:8]
with open(tsv, newline='') as f:
    runs = list(csv.DictReader(f, delimiter='\t'))
for r in runs:
    r['trial'] = int(r['trial'])
with open(dst, 'w') as f:
    json.dump({
        'attempt_id': attempt_id, 'passes': int(passes), 'ngen': int(ngen),
        'dry_run': bool(int(dry_run)), 'quiesce_granularity': quiesce_granularity,
        'runs': runs,
    }, f, indent=2)
PY

log "generating fight report"
if ! python3 "$CDIR/tools/fight_report.py" "$OUT" > "$OUT/report-console.txt" 2>&1; then
  cat "$OUT/report-console.txt" >&2
  die "fight_report.py failed -- see $OUT/report-console.txt"
fi
cat "$OUT/report-console.txt"

campaign_log "fight_card complete: $OUT/FIGHT_CARD.md"
log "done: $OUT/FIGHT_CARD.md"
