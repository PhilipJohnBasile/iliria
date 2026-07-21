#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 ]] || { echo "Usage: bash train-m5max-hotset.sh MODEL_DIR [passes]" >&2; exit 2; }
MODEL_DIR="${1%/}"
PASSES="${2:-${ILI_HOTSET_PASSES:-${ILI_HOTSET_PASSES:-1}}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[[ -d "$MODEL_DIR" ]] || { echo "error: model directory not found: $MODEL_DIR" >&2; exit 1; }
(( PASSES >= 1 )) || { echo "error: passes must be >= 1" >&2; exit 2; }

NGEN="${ILI_HOTSET_NGEN:-${ILI_HOTSET_NGEN:-96}}"
PROFILE="${ILI_HOTSET_PROFILE:-${ILI_HOTSET_PROFILE:-}}"
LOG_DIR="${ILI_HOTSET_LOG_DIR:-${ILI_HOTSET_LOG_DIR:-bench-m5max/hotset-${PROFILE:+${PROFILE}.}$(date +%Y%m%d-%H%M%S)}}"
mkdir -p "$LOG_DIR"
PROMPTS="$LOG_DIR/prompts.txt"

cat > "$PROMPTS" <<'EOF'
Review a Godot 4.7 GDScript player controller for movement bugs, typed API mistakes, and avoidable allocations. Return a concise patch plan.
Design a Godot 4.7 architecture for a 2D hockey game with deterministic puck physics, goalie AI, line changes, penalties, and replayable simulation state.
Diagnose a Godot project where an agent needs to read editor errors, inspect the scene tree, modify GDScript, run tests, and iterate without losing context.
Write a Rust Axum endpoint with typed errors, PostgreSQL transactions, tracing, validation, and tests. Explain ownership and concurrency choices.
Review unsafe Rust FFI code that wraps a C inference engine. Identify lifetime, aliasing, alignment, thread-safety, and cleanup defects.
Optimize a Rust hot loop for Apple Silicon while preserving exact output. Discuss profiling, cache locality, SIMD, allocation reuse, and benchmark design.
Implement a Python agent loop with tool schemas, bounded retries, structured JSON output, cancellation, logging, and deterministic tests.
Review Python code for an evaluation harness that measures tool selection, factuality, latency, token use, and regression failures across local models.
Design a retrieval-augmented generation system with document ingestion, hybrid retrieval, reranking, citations, tenant isolation, and prompt-injection defenses.
Write a TypeScript SvelteKit server action with Zod validation, secure session handling, optimistic UI, database errors, and unit tests.
Review a Svelte component for unnecessary rerenders, stale reactive state, accessibility problems, and browser/server boundary mistakes.
Design a real-time analytics dashboard using TypeScript, SvelteKit, Rust, PostgreSQL, and server-sent events with resilient reconnect behavior.
Inspect a C inference kernel for memory-bandwidth bottlenecks, false sharing, branch overhead, repeated allocation, and vectorization opportunities.
Review Objective-C++ Metal code for command-buffer stalls, shared-memory hazards, excess resource binding, temporary allocations, and synchronization errors.
Propose a Metal optimization plan for quantized mixture-of-experts inference on an Apple M5 Max with 128 GB unified memory.
Analyze a quantized int4 by int8 matrix-vector kernel for Apple ARM dot-product instructions, packing layout, accumulator precision, and tail handling.
Review a Git diff as a principal engineer. Find correctness regressions, race conditions, security issues, performance risks, missing tests, and unclear abstractions.
Plan an agentic coding task: inspect repository structure, reproduce a bug, form hypotheses, edit minimal files, run targeted tests, and report evidence.
Return a valid JSON tool call that searches a repository, reads the relevant files, applies a patch, runs tests, and summarizes changed behavior.
Debug an OpenAI-compatible local inference server that intermittently loses tool calls under streaming. Identify protocol, parser, concurrency, and test cases.
Design an NL-to-SQL agent with schema discovery, read-only enforcement, query validation, cost limits, repair loops, and cited results.
Review PostgreSQL indexes and query plans for a multi-tenant analytics workload with time-range filters, joins, pagination, and materialized aggregates.
Explain how to preserve context across a long agentic coding session using repository summaries, task ledgers, compact checkpoints, and targeted retrieval.
Compare two local-model benchmark runs and determine whether the apparent speedup comes from warm filesystem cache, expert-cache hit rate, speculative acceptance, or real kernel improvement.
EOF

prompt_count="$(wc -l < "$PROMPTS" | tr -d ' ')"
profile_label="${PROFILE:+[$PROFILE] }"
echo "Training the persistent expert hotset ${profile_label}with $prompt_count coding/agent prompts x $PASSES pass(es)."
echo "This uses one model load. Responses are written to $LOG_DIR/session.log."
echo "AUTOPIN is disabled during collection; the next normal launch will pin from the completed history."

{
  pass=1
  while (( pass <= PASSES )); do
    cat "$PROMPTS"
    pass=$((pass + 1))
  done
  printf ':q\n'
} | ILI_IGNORE_PROFILE=0 ILI_AUTOPIN=0 ILI_REPIN=0 ILI_NGEN="$NGEN" \
    ILI_AUTOPIN=0 ILI_REPIN=0 ILI_NGEN="$NGEN" \
    ILI_HOTSET_PROFILE="$PROFILE" ILI_HOTSET_PROFILE="$PROFILE" \
    bash ./run-m5max-fast.sh "$MODEL_DIR" chat \
    > "$LOG_DIR/session.log" 2>&1

# Determine the usage file path based on profile
if [[ -n "$PROFILE" ]]; then
  usage_file="$MODEL_DIR/.fa_usage.$PROFILE"
else
  usage_file="$MODEL_DIR/.fa_usage"
fi

if [[ ! -s "$usage_file" ]]; then
  echo "error: no persistent usage file was produced at $usage_file" >&2
  exit 1
fi

selections="$(awk '{s+=$3} END {printf "%.0f",s}' "$usage_file")"
entries="$(wc -l < "$usage_file" | tr -d ' ')"
cat <<EOF
Hotset training complete.
  usage file:  $usage_file
  entries:     $entries
  selections:  $selections

The next normal M5 Max launch uses AUTOPIN=1 and will size the pinned hot store
from this workload history while retaining LRU space for the current session.
EOF
