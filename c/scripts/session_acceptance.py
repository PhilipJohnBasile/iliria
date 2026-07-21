#!/usr/bin/env python3
"""Session-level acceptance harness -- the campaign's headline metric, measured end to end
for the first time.

THE QUESTION THIS ANSWERS: `docs/roadmap-daily-driver.md`'s unit that matters is "wall-clock
minutes per 20-turn agent session (~22K ctx, ~500 out tok/turn)". Three numbers for that unit
already exist, but every one of them is either a pre-measurement guess or a per-turn
extrapolation -- none is an actual, driven, 20-turn, front-to-back stopwatch measurement:

  1. ~148 min       "naive" baseline (no reuse, no discipline). docs/roadmap-daily-driver.md's
                     own opening line: "Baseline today ~= 148 min/session, TTFT 100-200 s/turn."
  2. ~5-7 h          "reuse-only" (Step 1 wired, Step 2 discipline, but BEFORE the Metal-prefill
                     kernel fix). Also a projection, not a measurement: roadmap Step 1 gate
                     section: "Session projection at measured rates (20 turns, ~1.5K new
                     tok/turn, 300 out/turn): ... with reuse ~= 30K delta tokens ~= 5-7 h + ~1 h
                     decode" -- extrapolated from 3 real turns' prefill throughput (~1.0-1.8
                     tok/s), never actually run for 20 turns.
  3. ~1.5-2 h        "reuse + Metal-prefill kernel", PROJECTED. After the Metal-prefill fix
                     (commits 8d26555/cc96775/ee81bbb, ILI_METAL_PREFILL=1) raised measured
                     prefill throughput ~6-9x (1.6-1.8 -> ~9-11 tok/s, serve-gate rerun,
                     roadmap "Serve-gate update" section), applying that speedup to band (2)
                     projects ~1.5-2h -- again arithmetic on 3 turns, not a driven session.

This script is the first thing that actually drives 20 realistic turns against a live
`ili serve` and stops a clock, instead of multiplying a 3-turn sample by 20/3. It also
answers a question none of the three numbers above could: the context-diet analysis measured
45-68% prompt-byte savings on an isolated synthetic transcript, but never at whole-session
scale on a REALISTIC (not uniform-tool-dump) turn mix -- see `--context-diet` below.

TURN-MIX DESIGN (SESSION_PLAN_SPEC): a realistic coding-agent session is not 20 identical
turns. This script scripts three turn shapes, deterministically ordered to read like a real
session (clarifying questions up front, escalating to file reads and big pastes, tapering
back to short questions at the end):

  "ask"    (10 turns, ~200-400 new tok each): a short question with a small inline snippet --
           the "tool-call/short-answer" shape from the Step 2 discipline table. max_tokens=200.
  "review" (6 turns, ~800-1.5K new tok each): user asks about a file, assistant issues a
           `read_file` tool call, tool responds with a real slice of a repo source file --
           EXACTLY long_ctx_profile.build_transcript's per-turn shape (reused via the shared
           `_LineCycler`, not reimplemented). The "code edit" discipline bucket. max_tokens=400.
  "paste"  (4 turns, ~2-4K new tok each): user pastes a large diff/log/file directly into the
           message (serve_gate.py's transcript shape). The most demanding turns -- mapped to
           the discipline's "from-scratch generation" bucket. max_tokens=700.

10+6+4 = 20 turns; target new-content tokens sum to ~22,150 -- this independently reproduces
the roadmap's own "~22K ctx" reference session from the turn-mix design, not by tuning to hit
it. All three shapes pull deterministic, never-random real repo source lines from ONE shared
`_LineCycler` (imported from `long_ctx_profile`, never reinstantiated per turn), so content
never repeats across the 20 turns and the whole plan is byte-identical across runs (see
`build_session_turns` and its unit tests).

Only "review" turns route content through a `role: tool` message. This is deliberate, not an
oversight: the context-diet analysis's trim_messages() only ever touches `role: tool` messages --
the 4 "paste" turns' large content lives in the `user` message and is NEVER reachable by
context diet, no matter which mode this script runs in. That means the session-level savings
this script measures are honestly SMALLER than the isolated 45-68% figure, because only 6 of
20 turns' aged-out content is even eligible. Measuring exactly how much smaller, on a
realistic mixed-size session instead of a uniform synthetic one, is what `--context-diet on`
answers for the first time (see `_context_diet_savings` below).

DISCIPLINE (always on, not a toggle): every turn sends `"enable_thinking": false` and the
per-kind `max_tokens` above (the roadmap's 200/400/700 mix); the system prompt is
`serve_gate.DISCIPLINED_SYSTEM` (diffs-not-files, terse). This script has no "undisciplined"
arm -- Step 1+2 discipline is the whole premise of a daily-driver session, so the "148 min
naive" reference point is deliberately a citation of the pre-existing baseline, not something
this script re-measures.

MODES (the two independent variables this script actually toggles):
  --context-diet {on,off}   (default off) sets the `trim_tool_output_tokens` /
                            `trim_keep_last_turns` request fields to the context-diet analysis
                            suggested defaults (300 / 2) vs. leaving them unset (server default
                            off). This is a pure client-side request-field toggle -- no server
                            env var needs to change between runs.
  --cold / --warm start     (default warm) runs long_ctx_profile.cold_purge() once before turn
                            1 (best-effort `sudo -n purge`) to represent a session starting
                            against a cold expert-page cache, vs. skipping the purge.

Dependency-free (stdlib only) except for two sibling scripts and the server module, imported
(not copied) exactly like abba_transcript_driver.py imports serve_gate:
  serve_gate       -- DISCIPLINED_SYSTEM, sse_request() (the SSE/TTFT/decode client)
  long_ctx_profile -- _LineCycler (corpus machinery), CHARS_PER_TOKEN, DEFAULT_SOURCES,
                      refuse_if_bench_running(), cold_purge(), start_mock_server(), tail_log(),
                      parse_last_kv_line()
  openai_server    -- trim_messages(), _TrimCache, render_chat() (the REAL production context-
                      diet code, so the savings number is measured, not estimated)

OPERATIONAL SEQUENCING FOR TOMORROW (engine-gated: this script never spawns `glm`, it only
talks HTTP to an already-running `ili serve`, exactly like scripts/long_ctx_profile.py):

  1. Confirm nothing else is on the engine (this script's own preflight,
     `long_ctx_profile.refuse_if_bench_running()`, re-checks this at startup and refuses if it
     finds zero or more than one `glm` process):
       pgrep -x glm

  2. Start a DEDICATED serve process (from `c/`). run-m5max-serve.sh's own defaults already
     set ILI_METAL_PREFILL=1, ILI_THINK=0, ILI_CTX=40960 -- all comfortably cover this
     session's ~22K-token depth, so no overrides are required except pinning to one slot
     (this script only ever drives `--slot 0`, so a second slot would just be idle):
       ILI_KV_SLOTS=1 bash run-m5max-serve.sh /path/to/GLM-5.2-int4-with-int8-mtp \\
         2> serve.log &

  3. Run the headline comparison arm (context-diet off, warm start -- directly comparable to
     the three reference points in the module docstring above):
       python3 scripts/session_acceptance.py --serve-log serve.log \\
         --attempt-id session-$(date +%Y%m%d-%H%M%S) --context-diet off --warm \\
         --out-csv bench-m5max/session-acceptance-$(date +%Y%m%d)-off-warm.csv \\
         --out-md  bench-m5max/session-acceptance-$(date +%Y%m%d)-off-warm.md

  4. Optional additional arms, same serve process (monotonic history is per `--slot`, so each
     arm below needs its own `--slot` if run back-to-back against the SAME server without
     restarting it, or just rerun step 2 fresh between arms for a clean slot 0):
       --context-diet on   (measures the session-level trim savings for the first time)
       --cold              (once, before turn 1 of that arm)

TONIGHT (no engine, per the campaign's hard constraint): everything below `--mock` is
exercised end-to-end against an in-process stub HTTP server (reused directly from
long_ctx_profile.start_mock_server -- not reimplemented) that speaks just enough of the
`/v1/chat/completions` protocol (SSE + the `[API] KV slot ...` stderr line) to drive the full
20-turn plan, the log parser, the context-diet diagnostic, and CSV/markdown output -- with
fabricated, clearly-non-authoritative timings:

    python3 scripts/session_acceptance.py --mock \\
      --attempt-id session-mock-$(date +%Y%m%d-%H%M%S) \\
      --out-csv /tmp/session-mock.csv --out-md /tmp/session-mock.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent      # c/scripts
REPO_C_DIR = HERE.parent                    # c/

sys.path.insert(0, str(HERE))
import serve_gate          # noqa: E402  (DISCIPLINED_SYSTEM, sse_request -- path insert first)
import long_ctx_profile     # noqa: E402  (_LineCycler, mock server, engine gate, log parsing)

sys.path.insert(0, str(REPO_C_DIR))
import openai_server        # noqa: E402  (trim_messages/_TrimCache/render_chat -- real prod code)

# ---- headline reference points (docs/roadmap-daily-driver.md; see module docstring) --------

NAIVE_BASELINE_MIN = 148.0
# Step 1 gate session projection, "5-7 h" band as literally stated (roadmap also notes an
# additional "+~1h decode" on top of the prefill estimate below -- i.e. ~6-8h all-in; the
# 5-7h/300-420min band here is the prefill-only figure the campaign has been citing).
REUSE_ONLY_LOW_MIN = 5 * 60.0
REUSE_ONLY_HIGH_MIN = 7 * 60.0
# Metal-prefill kernel fix (commits 8d26555/cc96775/ee81bbb) applied to the band above.
# PROJECTED -- this script's real (non-mock) run is the first end-to-end check of this band.
REUSE_KERNEL_PROJECTED_LOW_MIN = 90.0
REUSE_KERNEL_PROJECTED_HIGH_MIN = 120.0

SLACK_TOKENS = 64   # same delta-vs-prefill tolerance as serve_gate.py's verdict()

# ---- the 20-turn plan: (turn, kind, target_new_tokens, max_tokens) --------------------------
# kind -> max_tokens mapping follows run-m5max-serve.sh's discipline table verbatim: "~200 for
# tool-call/short-answer turns, ~400 for code edits, ~700 only for from-scratch generation."
#   ask    -> 200  (short question   = tool-call/short-answer)
#   review -> 400  (file-read + comment = code edit)
#   paste  -> 700  (big diff/log paste, asks for a full pass = closest to from-scratch)
SESSION_PLAN_SPEC = [
    (1,  "ask",    220,  200),
    (2,  "ask",    260,  200),
    (3,  "review", 850,  400),
    (4,  "ask",    300,  200),
    (5,  "ask",    340,  200),
    (6,  "paste",  2000, 700),
    (7,  "review", 1000, 400),
    (8,  "ask",    380,  200),
    (9,  "review", 1150, 400),
    (10, "paste",  2600, 700),
    (11, "ask",    400,  200),
    (12, "review", 1300, 400),
    (13, "ask",    360,  200),
    (14, "paste",  3200, 700),
    (15, "review", 1450, 400),
    (16, "ask",    320,  200),
    (17, "review", 1500, 400),
    (18, "paste",  4000, 700),
    (19, "ask",    280,  200),
    (20, "ask",    240,  200),
]

# One preamble per "ask" occurrence (10) / task line per "paste" occurrence (4), in session
# order -- gives the scripted session a narrative arc (clarifying -> escalating -> wrap-up)
# instead of 10 copies of the same sentence. The attached snippet (not this text) supplies
# most of each turn's token budget.
ASK_PREAMBLES = [
    "Before we start on the container work -- quick sanity check on this snippet, does the "
    "fallback path look right to you?",
    "Following up on that -- I'm seeing this in the logs, is it expected or a real problem?",
    "Quick one: does this need a null check before the dereference?",
    "Is this the right retry pattern, or is there a cleaner way to write it?",
    "Small thing before we go further -- should this comparison be `>=` instead of `>`?",
    "Sanity check: is this cast safe across all our target platforms?",
    "One more -- does this look like it could leak under an early return?",
    "Quick question on this helper -- is the rounding here intentional?",
    "Almost done -- anything jump out as risky in this last bit?",
    "Wrapping up -- does this final version look ready to ship?",
]
PASTE_TASKS = [
    "Here's the full diff from my last commit -- can you review it end-to-end and flag "
    "anything risky?",
    "The build just failed with this dump -- can you find the root cause?",
    "Pasting the complete error output here, this one looks more serious than the others.",
    "Last big one before we wrap up -- full file paste, final sanity check before I ship it?",
]


def validate_session_plan(plan):
    """Defense-in-depth invariant check for SESSION_PLAN_SPEC (also directly unit-tested):
    exactly 20 turns, numbered 1..20, with the documented 10/6/4 kind split and a total target
    token count in the ballpark of the roadmap's own ~22K-token reference session."""
    turns = [p[0] for p in plan]
    if turns != list(range(1, len(plan) + 1)):
        raise ValueError(f"SESSION_PLAN_SPEC turn numbers must be 1..N sequential, got {turns}")
    kinds = [p[1] for p in plan]
    counts = {k: kinds.count(k) for k in ("ask", "review", "paste")}
    if counts != {"ask": 10, "review": 6, "paste": 4}:
        raise ValueError(f"expected 10 ask/6 review/4 paste turns, got {counts}")
    total_tokens = sum(p[2] for p in plan)
    if not 18_000 <= total_tokens <= 26_000:
        raise ValueError(f"session target token total {total_tokens} is far from the "
                         "roadmap's ~22K-token reference session")
    return True


# ---- per-turn message construction (reuses long_ctx_profile's _LineCycler machinery) --------

def _ask_messages(cycler, turn, target_tokens, preamble):
    budget_chars = target_tokens * long_ctx_profile.CHARS_PER_TOKEN
    remaining = max(budget_chars - len(preamble), 200)
    rel, start, end, chunk = cycler.next_chunk(remaining)
    content = (f"[turn {turn}] {preamble}\n(from {rel}, lines {start}-{end})\n"
              f"```\n{chunk}\n```")
    return [{"role": "user", "content": content}]


def _review_messages(cycler, turn, target_tokens):
    """Exactly long_ctx_profile.build_transcript's per-turn shape (user ask, assistant
    read_file tool call, tool response with a real repo-source slice) -- reused, not
    reimplemented, just parameterized with this session's own target size."""
    budget_chars = target_tokens * long_ctx_profile.CHARS_PER_TOKEN
    rel, start, end, chunk = cycler.next_chunk(budget_chars)
    call_id = f"call_{turn:05d}"
    user = {"role": "user",
            "content": f"[turn {turn}] Can you look at {rel} around lines {start}-{end}? "
                       "Not sure if anything there is a problem."}
    assistant_call = {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": "read_file",
                     "arguments": json.dumps({"path": rel, "start": start, "end": end})}}]}
    tool_response = {"role": "tool", "tool_call_id": call_id,
                     "content": f"$ sed -n '{start},{end}p' {rel}\n{chunk}"}
    return [user, assistant_call, tool_response]


def _paste_messages(cycler, turn, target_tokens, task_text):
    budget_chars = target_tokens * long_ctx_profile.CHARS_PER_TOKEN
    rel, start, end, chunk = cycler.next_chunk(budget_chars)
    content = f"[turn {turn}] {task_text}\n(from {rel}, lines {start}-{end})\n```\n{chunk}\n```"
    return [{"role": "user", "content": content}]


def build_session_turns(repo_root, sources=None):
    """Pure, deterministic, no-HTTP construction of the 20-turn plan's per-turn NEW messages
    (the content each turn appends to history BEFORE the live assistant reply -- the caller
    supplies that reply after actually driving the turn against a real or mock server). Same
    inputs -> byte-identical output, always: one shared `_LineCycler` walks the given repo's
    own source files in order, so content is real code, never repeats across turns, and never
    spans a file boundary -- the same guarantees long_ctx_profile.build_transcript documents.
    """
    validate_session_plan(SESSION_PLAN_SPEC)
    sources = sources or long_ctx_profile.DEFAULT_SOURCES
    cycler = long_ctx_profile._LineCycler(repo_root, sources)
    ask_i = paste_i = 0
    out = []
    for turn, kind, target_tokens, max_tokens in SESSION_PLAN_SPEC:
        if kind == "ask":
            new_messages = _ask_messages(cycler, turn, target_tokens, ASK_PREAMBLES[ask_i])
            ask_i += 1
        elif kind == "review":
            new_messages = _review_messages(cycler, turn, target_tokens)
        elif kind == "paste":
            new_messages = _paste_messages(cycler, turn, target_tokens, PASTE_TASKS[paste_i])
            paste_i += 1
        else:
            raise ValueError(f"unknown turn kind: {kind!r}")
        out.append({"turn": turn, "kind": kind, "target_tokens": target_tokens,
                    "max_tokens": max_tokens, "new_messages": new_messages})
    return out


# ---- context-diet session-level diagnostic (real production code, not a mock estimate) -----

def _context_diet_savings(messages, keep_last_turns, tool_output_tokens, cache):
    """Render `messages` through the REAL openai_server.render_chat()/trim_messages(), exactly
    the context-diet analysis's own reproduction recipe, and return (full_chars, trimmed_chars,
    savings_pct). This is a pure client-side computation -- identical whether driving a mock or
    a live server -- so it is a genuine measurement, not a mock fabrication, in every mode.
    Returns (None, None, None) if rendering fails (diagnostic only; never fails the session)."""
    try:
        trimmed = openai_server.trim_messages(
            messages, keep_last_turns=keep_last_turns,
            tool_output_tokens=tool_output_tokens, cache=cache)
        full_chars = len(openai_server.render_chat(messages, enable_thinking=False))
        trimmed_chars = len(openai_server.render_chat(trimmed, enable_thinking=False))
        savings_pct = (round(100.0 * (1 - trimmed_chars / full_chars), 2)
                      if full_chars else None)
        return full_chars, trimmed_chars, savings_pct
    except Exception as exc:   # pragma: no cover -- diagnostic only, never fatal
        print(f"warning: context-diet diagnostic failed: {exc}", file=sys.stderr)
        return None, None, None


# ---- session driver ------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def run_session(args, sources, start_mode, cold_purge_s):
    plan = build_session_turns(args.repo_root, sources)
    messages = [{"role": "system", "content": serve_gate.DISCIPLINED_SYSTEM}]
    trim_cache = openai_server._TrimCache() if args.context_diet == "on" else None
    log_offset = (os.path.getsize(args.serve_log)
                  if args.serve_log and os.path.exists(args.serve_log) else 0)

    turns = []
    prev_prompt_tokens = 0
    session_start = time.monotonic()
    for step in plan:
        turn, kind, target_tokens, max_tokens = (
            step["turn"], step["kind"], step["target_tokens"], step["max_tokens"])
        messages.extend(step["new_messages"])

        payload = {"model": args.model_id, "messages": messages, "stream": True,
                   "stream_options": {"include_usage": True}, "temperature": 0,
                   "max_tokens": max_tokens, "cache_slot": args.slot,
                   "enable_thinking": False}
        if args.context_diet == "on":
            payload["trim_tool_output_tokens"] = args.trim_tool_output_tokens
            payload["trim_keep_last_turns"] = args.trim_keep_last_turns

        turn_start = time.monotonic()
        result = serve_gate.sse_request(args.host, args.port, payload, args.timeout)
        turn_wall_s = time.monotonic() - turn_start

        log_text, log_offset = long_ctx_profile.tail_log(args.serve_log, log_offset)
        kv = long_ctx_profile.parse_last_kv_line(log_text)

        usage = result["usage"] or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens", 0)
        decode_s = result["decode_s"]
        decode_tps = (completion_tokens / decode_s) if decode_s and completion_tokens else None
        prefill_engine = (kv or {}).get("prefill_delta_tokens")
        expected_delta = (prompt_tokens - prev_prompt_tokens
                          if prompt_tokens is not None else None)

        if turn == 1:
            reuse_engaged = None    # first turn: full prefill is CORRECT, not a reuse check
        elif prefill_engine is None or expected_delta is None:
            reuse_engaged = None
        else:
            reuse_engaged = prefill_engine <= expected_delta + SLACK_TOKENS

        reply_text = (result["text"] or "").strip() or "(empty)"
        messages.append({"role": "assistant", "content": reply_text})

        trim_full_chars = trim_trimmed_chars = trim_savings_pct = None
        if args.context_diet == "on":
            trim_full_chars, trim_trimmed_chars, trim_savings_pct = _context_diet_savings(
                messages, args.trim_keep_last_turns, args.trim_tool_output_tokens, trim_cache)

        record = {
            "attempt_id": args.attempt_id, "turn": turn, "kind": kind,
            "context_diet": args.context_diet, "start_mode": start_mode, "mock": args.mock,
            "target_new_tokens": target_tokens, "max_tokens": max_tokens,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "ttft_s": round(result["ttft_s"], 3) if result["ttft_s"] else None,
            "decode_tok_s": round(decode_tps, 3) if decode_tps else None,
            "total_s": round(turn_wall_s, 3),
            "prefill_delta_engine": prefill_engine,
            "prefill_delta_expected": expected_delta,
            "reuse_engaged": reuse_engaged,
            "trim_full_chars": trim_full_chars,
            "trim_trimmed_chars": trim_trimmed_chars,
            "trim_savings_pct": trim_savings_pct,
            "finish_reason": result["finish_reason"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        turns.append(record)
        print(f"[turn {turn}/20 {kind}] prompt={prompt_tokens} out={completion_tokens} "
              f"ttft={record['ttft_s']}s decode={record['decode_tok_s']}tok/s "
              f"prefill(engine)={prefill_engine} reuse_engaged={reuse_engaged} "
              f"total={record['total_s']}s", flush=True)

        if prompt_tokens is not None:
            prev_prompt_tokens = prompt_tokens

    session_wall_s = time.monotonic() - session_start
    engine_time_s = sum(t["total_s"] for t in turns)
    reuse_failures = [t["turn"] for t in turns if t["reuse_engaged"] is False]

    summary = {
        "attempt_id": args.attempt_id, "mock": args.mock, "context_diet": args.context_diet,
        "start_mode": start_mode,
        "cold_purge_s": round(cold_purge_s, 3) if cold_purge_s is not None else None,
        "turns": len(turns),
        "session_wall_s": round(session_wall_s, 3),
        "engine_time_s": round(engine_time_s, 3),
        "engine_time_share_pct": (round(100.0 * engine_time_s / session_wall_s, 2)
                                 if session_wall_s else None),
        "measured_minutes": round(session_wall_s / 60.0, 3),
        "final_prompt_tokens": turns[-1]["prompt_tokens"] if turns else None,
        "sum_prefill_delta_engine": sum(t["prefill_delta_engine"] for t in turns
                                        if t["prefill_delta_engine"] is not None),
        "mean_ttft_s": _mean([t["ttft_s"] for t in turns]),
        "mean_decode_tok_s": _mean([t["decode_tok_s"] for t in turns]),
        "final_trim_savings_pct": turns[-1]["trim_savings_pct"] if turns else None,
        "reuse_failures": reuse_failures,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return summary, turns


# ---- headline comparison ---------------------------------------------------------------

def classify_measured_minutes(minutes):
    """Plain-English placement of a measured session length against the three reference
    bands -- a comparison, not a pass/fail gate (this script measures; it does not judge).

    NOTE the bands are NOT monotonic in "goodness" order by construction: the 148-min naive
    baseline was itself a pre-measurement guess that assumed ~100-200 tok/s prefill, which the
    Step 1 gate found was wrong by ~100x (roadmap-daily-driver.md, "Serve-gate update" section).
    So the honest ordering (best/fastest first) is reuse+kernel (90-120 min) < naive (148 min)
    < reuse-only-no-kernel (300-420 min, "the afternoon") -- the naive guess sits BETWEEN the
    other two bands, not above both. Ordering the checks by ascending minutes gets this right
    without hardcoding an assumption about which band is "worse"."""
    if minutes <= REUSE_KERNEL_PROJECTED_HIGH_MIN:
        return (f"at/below the reuse+kernel PROJECTED band "
                f"({REUSE_KERNEL_PROJECTED_LOW_MIN:.0f}-{REUSE_KERNEL_PROJECTED_HIGH_MIN:.0f} "
                "min) -- the projection holds, and beats the naive baseline too")
    if minutes <= NAIVE_BASELINE_MIN:
        return (f"above the reuse+kernel projected band but still at/below the naive "
                f"baseline ({NAIVE_BASELINE_MIN:.0f} min)")
    if minutes <= REUSE_ONLY_HIGH_MIN:
        return (f"above the naive baseline ({NAIVE_BASELINE_MIN:.0f} min) -- in or below the "
                f"reuse-only, no-Metal-prefill-kernel band "
                f"({REUSE_ONLY_LOW_MIN:.0f}-{REUSE_ONLY_HIGH_MIN:.0f} min); the kernel fix is "
                "what is supposed to close this gap")
    return (f"above even the reuse-only band ({REUSE_ONLY_HIGH_MIN:.0f} min) -- "
            "regression, investigate")


def headline_table_lines(summary):
    minutes = summary["measured_minutes"]
    lines = [
        "| reference point | minutes | source |",
        "|---|---|---|",
        f"| naive baseline (no reuse, no discipline) | {NAIVE_BASELINE_MIN:.1f} | "
        "docs/roadmap-daily-driver.md, \"Baseline today\" (measured 2026-07-14) |",
        f"| reuse-only, no Metal-prefill kernel (\"the afternoon\") | "
        f"{REUSE_ONLY_LOW_MIN:.1f}-{REUSE_ONLY_HIGH_MIN:.1f} | "
        "docs/roadmap-daily-driver.md Step 1 gate session projection (measured serve prefill "
        "~1.0-1.8 tok/s; roadmap also notes +~1h decode on top) |",
        f"| reuse + Metal-prefill kernel (PROJECTED) | "
        f"{REUSE_KERNEL_PROJECTED_LOW_MIN:.1f}-{REUSE_KERNEL_PROJECTED_HIGH_MIN:.1f} | "
        "Metal-prefill fix (commits 8d26555/cc96775/ee81bbb): measured 6-9x TTFT applied to "
        "the reuse-only band -- never measured end-to-end before this harness |",
        f"| **measured (this run)** | **{minutes:.2f}** | "
        f"session_acceptance.py, attempt_id=`{summary['attempt_id']}` |",
        "",
        f"Measured session: {classify_measured_minutes(minutes)}.",
    ]
    if summary["mock"]:
        lines += [
            "",
            "**NOTE: this run used `--mock`** (an in-process stub server, fabricated timings). "
            "These numbers exercise the harness's plumbing only and are NOT a real "
            "measurement -- see the module docstring's \"exact real-run command\".",
        ]
    return lines


# ---- output ---------------------------------------------------------------------------

CSV_COLUMNS = ["attempt_id", "turn", "kind", "context_diet", "start_mode", "mock",
              "target_new_tokens", "max_tokens", "prompt_tokens", "completion_tokens",
              "ttft_s", "decode_tok_s", "total_s", "prefill_delta_engine",
              "prefill_delta_expected", "reuse_engaged", "trim_full_chars",
              "trim_trimmed_chars", "trim_savings_pct", "finish_reason", "timestamp"]


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(turns, summary, path):
    lines = ["# session_acceptance results", ""]
    lines.append(
        f"attempt_id: `{summary['attempt_id']}` | mock: {summary['mock']} | "
        f"context_diet: {summary['context_diet']} | start_mode: {summary['start_mode']} | "
        f"turns: {summary['turns']} | generated: {summary['timestamp']}")
    lines.append("")
    lines.append(
        "Session-level acceptance measurement for the campaign's headline metric (see the "
        "module docstring for why the three reference points below were never previously "
        "measured end-to-end).")
    lines += ["", "## Headline: measured vs projected 20-turn session wall-clock", ""]
    lines += headline_table_lines(summary)

    lines += ["", "## Session totals", ""]
    lines.append(f"- session wall-clock: {summary['session_wall_s']}s "
                f"({summary['measured_minutes']} min)")
    lines.append(f"- engine time (sum of per-turn round trips): {summary['engine_time_s']}s "
                f"-- {summary['engine_time_share_pct']}% of session wall-clock")
    if summary["cold_purge_s"] is not None:
        lines.append(f"- cold-cache purge (before turn 1, not counted in session wall-clock): "
                    f"{summary['cold_purge_s']}s")
    lines.append(f"- final context depth: {summary['final_prompt_tokens']} tokens "
                f"(target ~22,150; docs/roadmap-daily-driver.md's own \"~22K ctx\" reference "
                "session)")
    lines.append(f"- sum of per-turn engine-reported prefill deltas: "
                f"{summary['sum_prefill_delta_engine']} tokens")
    lines.append(f"- mean TTFT: {summary['mean_ttft_s']}s | "
                f"mean decode: {summary['mean_decode_tok_s']} tok/s")
    if summary["context_diet"] == "on":
        lines.append(f"- context-diet final cumulative savings: "
                    f"{summary['final_trim_savings_pct']}% (measured via the real "
                    "openai_server.trim_messages()/render_chat(), not estimated; only the 6 "
                    "\"review\" turns' aged-out tool output is eligible -- see module "
                    "docstring)")
    else:
        lines.append("- context-diet: off this run (n/a) -- rerun with --context-diet on to "
                    "measure the session-level savings")
    if summary["reuse_failures"]:
        lines.append(f"- reuse-engagement failures (turns 2-20 only): {summary['reuse_failures']}")
    else:
        lines.append("- reuse-engagement failures (turns 2-20 only): none")

    lines += ["", "## Per-turn table", ""]
    header = ["turn", "kind", "target_new_tok", "max_tokens", "prompt_tokens",
              "completion_tokens", "ttft_s", "decode_tok_s", "prefill_engine",
              "prefill_expected", "reuse_engaged", "total_s"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    keys = ["turn", "kind", "target_new_tokens", "max_tokens", "prompt_tokens",
            "completion_tokens", "ttft_s", "decode_tok_s", "prefill_delta_engine",
            "prefill_delta_expected", "reuse_engaged", "total_s"]
    for row in turns:
        values = [row.get(k) for k in keys]
        lines.append("| " + " | ".join("n/a" if v is None else str(v) for v in values) + " |")

    lines += ["", "## Reuse-engagement notes", ""]
    lines.append(
        "Turn 1 always shows a full prefill (no prior slot content) -- expected, not a "
        "failure. Turns 2-20 are checked against `prefill_engine <= prefill_expected + "
        f"{SLACK_TOKENS}` (same tolerance as scripts/serve_gate.py's verdict()).")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


# ---- CLI -----------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true",
                        help="dry run against an in-process stub server; no engine, no pgrep gate")
    parser.add_argument("--mock-port", type=int, default=8898)
    parser.add_argument("--mock-model-id", default=long_ctx_profile.DEFAULT_MODEL_ID,
                        help="model id the --mock stub server advertises via GET "
                             "/v1/models (testing only)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=None,
                        help="chat-completions `model` field; default: query the "
                             "running server's GET /v1/models and use its first "
                             "advertised id, falling back to "
                             f"{long_ctx_profile.DEFAULT_MODEL_ID!r} if that query fails")
    parser.add_argument("--slot", type=int, default=0,
                        help="cache_slot for the whole session (ONE monotonic slot throughout)")
    parser.add_argument("--context-diet", choices=("on", "off"), default="off",
                        help="set trim_tool_output_tokens/trim_keep_last_turns request fields")
    parser.add_argument("--trim-tool-output-tokens", type=int, default=300,
                        help="the context-diet analysis suggested default")
    parser.add_argument("--trim-keep-last-turns", type=int, default=2,
                        help="the context-diet analysis suggested default")
    start_mode = parser.add_mutually_exclusive_group()
    start_mode.add_argument("--cold", action="store_true",
                            help="purge the OS page cache once before turn 1 (best-effort; "
                                 "see long_ctx_profile.cold_purge)")
    start_mode.add_argument("--warm", action="store_true", help="skip the purge (default)")
    parser.add_argument("--repo-root", default=str(REPO_C_DIR),
                        help="directory the corpus source files are read from (default: c/)")
    parser.add_argument("--sources", default=None,
                        help="comma-separated repo-relative source files "
                             "(default: long_ctx_profile.DEFAULT_SOURCES)")
    parser.add_argument("--attempt-id", default=None,
                        help="default: session-<timestamp>")
    parser.add_argument("--serve-log", default=None,
                        help="running ili serve's stderr log (required for a live run; "
                             "auto-set under --mock)")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.attempt_id:
        args.attempt_id = f"session-{time.strftime('%Y%m%d-%H%M%S')}"
    start_mode = "cold" if args.cold else "warm"
    sources = tuple(args.sources.split(",")) if args.sources else long_ctx_profile.DEFAULT_SOURCES

    validate_session_plan(SESSION_PLAN_SPEC)
    total_target = sum(p[2] for p in SESSION_PLAN_SPEC)
    print(f"[plan] 20 turns (10 ask/6 review/4 paste), ~{total_target} target new-content "
          f"tokens, attempt_id={args.attempt_id}, context_diet={args.context_diet}, "
          f"start_mode={start_mode}", file=sys.stderr)

    if args.mock:
        if args.serve_log:
            log_path = args.serve_log
        else:
            # A fixed shared filename here would collide across concurrent --mock
            # invocations (e.g. this script running while the test suite's own --mock
            # subprocess tests are also in flight): each run's `open(log_path, "w")`
            # truncation would race the other's in-flight appends, silently dropping
            # `[API] KV slot ...` lines the OTHER run needed. mkstemp's atomic creation
            # guarantees a unique path per run instead.
            fd, log_path = tempfile.mkstemp(prefix="session_acceptance_mock_", suffix=".log")
            os.close(fd)
        open(log_path, "w").close()
        long_ctx_profile.start_mock_server(args.host, args.mock_port, log_path,
                                           model_id=args.mock_model_id)
        args.port = args.mock_port
        args.serve_log = log_path
        print(f"[mock] stub server on http://{args.host}:{args.mock_port}, log={log_path}",
              file=sys.stderr)
    else:
        long_ctx_profile.refuse_if_bench_running()
        if not args.serve_log:
            print("error: --serve-log is required for a live run (the running `ili serve`'s "
                  "stderr log, to read [API] KV slot lines)", file=sys.stderr)
            sys.exit(2)

    if not args.model_id:
        args.model_id = long_ctx_profile.resolve_model_id(args.host, args.port)
    print(f"[model-id] using {args.model_id!r}", file=sys.stderr)

    cold_purge_s = None
    if start_mode == "cold":
        t0 = time.monotonic()
        if args.mock:
            print("[mock] --cold requested but mocked: skipping purge", file=sys.stderr)
        else:
            long_ctx_profile.cold_purge()
        cold_purge_s = time.monotonic() - t0

    summary, turns = run_session(args, sources, start_mode, cold_purge_s)
    write_csv(turns, args.out_csv)
    write_markdown(turns, summary, args.out_md)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_csv} and {args.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
