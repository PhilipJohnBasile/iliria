#!/usr/bin/env python3
"""Step 1 gate for docs/roadmap-daily-driver.md: prefix-KV reuse through `ili serve`.

Drives a scripted N-turn agent-style transcript (long code hunks, ~1-2K new prompt
tokens per turn, monotonic history) through the OpenAI-compatible API and measures,
per turn: prefill tokens (from the serve stderr log), TTFT, decode tok/s, and output
tokens. Two modes:

  reuse    monotonic history on one KV slot -> turns 2..N must prefill only the delta.
           Uses the Step 2 discipline defaults (THINK off, per-turn max_tokens,
           diffs-not-files system prompt).
  control  fresh-context baseline: a per-turn nonce at the head of the system message
           breaks the token prefix, forcing a full re-prefill at the same history
           lengths. Skipped turns use canned assistant text so context sizes match.
           No length discipline (loose max_tokens, plain system prompt).

Verdict mode compares the two result files and prints PASS/FAIL:
  * turns 2..N prefill only the delta (prefill_tokens <= new_tokens + 64);
  * reuse decode tok/s >= 90% of the control mean.

Dependency-free (stdlib only), like openai_server.py.

Typical session (from c/):
  ILI_PORT=8199 bash run-m5max-serve.sh MODEL_DIR 2> serve.log &
  python3 scripts/serve_gate.py --mode reuse   --port 8199 --serve-log serve.log --out reuse.json
  python3 scripts/serve_gate.py --mode control --port 8199 --serve-log serve.log --out control.json
  python3 scripts/serve_gate.py --verdict reuse.json control.json
"""

import argparse
import http.client
import json
import os
import re
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))

DISCIPLINED_SYSTEM = (
    "You are a senior C engineer reviewing an inference engine. "
    "Reply with unified diffs or changed hunks only, never full files. "
    "No preamble, no recap, no restating the request. Be terse."
)
PLAIN_SYSTEM = "You are a senior C engineer reviewing an inference engine."

# Deterministic filler for control turns that are not actually generated: keeps the
# rendered context length comparable with the reuse transcript (~120 words ≈ 160 tok).
CANNED_REPLY = (
    "The hunk manages the expert cache admission path. It computes the resident byte "
    "budget, subtracts the key-value pool and the working-set slabs, and derives the "
    "per-layer capacity. One issue is that the reserve arithmetic mixes double and "
    "int64 operands, which can lose precision for very large budgets. A safer version "
    "keeps the accumulation in int64 until the final division. Another detail worth "
    "checking is the fallback when MemAvailable is unreadable: the code assumes eight "
    "gigabytes, which may overshoot small machines. The truncation path also rewrites "
    "the record counter last so a crash mid-append leaves the file consistent, which "
    "is correct. Overall the logic is sound but the capacity clamp deserves a unit "
    "test around the boundary where the computed capacity drops below one." )

API_LINE = re.compile(r"\[API\] KV slot (\d+) prefix (\d+)/(\d+) token, prefill (\d+)")

# openai_server.py serve()'s own --model-id default; last-resort fallback for
# resolve_model_id() below only -- see its docstring for the 2026-07-14 incident this
# guards against (a stale/hardcoded client model-id 404ing against a renamed server).
DEFAULT_MODEL_ID = "glm-5.2-iliria"


def load_chunks(source, turns, chars_per_turn):
    """Slice ~chars_per_turn hunks out of a real source file (line-aligned)."""
    with open(source, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()
    chunks, cursor = [], 0
    for _ in range(turns):
        taken, size = [], 0
        while size < chars_per_turn:
            taken.append(lines[cursor % len(lines)])
            size += len(taken[-1]) + 1
            cursor += 1
        chunks.append("\n".join(taken))
    return chunks


def sse_request(host, port, payload, timeout):
    """POST a streaming chat completion; return per-turn timing and usage."""
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json"})
    t_sent = time.monotonic()
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:500]!r}")
    t_first, usage, finish, text = None, None, None, []
    buffer = b""
    while True:
        chunk = response.read1(65536)
        if not chunk:
            break
        buffer += chunk
        while b"\n\n" in buffer:
            event, buffer = buffer.split(b"\n\n", 1)
            for line in event.splitlines():
                if not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    continue
                parsed = json.loads(data)
                if parsed.get("usage"):
                    usage = parsed["usage"]
                for choice in parsed.get("choices", []):
                    delta = choice.get("delta") or {}
                    # keepalive pings arrive as reasoning_content: not real tokens
                    if delta.get("content"):
                        if t_first is None:
                            t_first = time.monotonic()
                        text.append(delta["content"])
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    t_done = time.monotonic()
    conn.close()
    return {"ttft_s": (t_first - t_sent) if t_first else None,
            "total_s": t_done - t_sent,
            "decode_s": (t_done - t_first) if t_first else None,
            "usage": usage, "finish_reason": finish, "text": "".join(text)}


def resolve_model_id(host, port, timeout=10.0, api_key=None, default=DEFAULT_MODEL_ID):
    """GET /v1/models from an already-running `ili serve` and return its FIRST advertised
    model id -- kept as a local copy (not imported from long_ctx_profile.py) so this script
    stays dependency-free, same convention as sse_request() below. See
    long_ctx_profile.resolve_model_id's docstring for the incident this guards against.
    Never raises -- falls back to `default` for any failure (connection refused, timeout,
    non-200, malformed body, empty list)."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/v1/models", headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        if response.status != 200:
            return default
        models = json.loads(data).get("data") or []
        return models[0]["id"] if models else default
    except Exception:
        return default


def parse_serve_log(path, offset):
    """Collect [API] prefill lines appended to the serve log after `offset`."""
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        tail = handle.read()
    return [{"slot": int(m.group(1)), "prefix_tokens": int(m.group(2)),
             "prompt_tokens": int(m.group(3)), "prefill_tokens": int(m.group(4))}
            for m in API_LINE.finditer(tail)]


def run_transcript(args):
    disciplined = args.mode == "reuse"
    system = DISCIPLINED_SYSTEM if disciplined else PLAIN_SYSTEM
    max_tokens = args.max_tokens if disciplined else args.control_max_tokens
    sample = (set(range(1, args.turns + 1)) if disciplined
              else {int(t) for t in args.sample.split(",")})
    chunks = load_chunks(args.source, args.turns, args.chars_per_turn)
    log_offset = (os.path.getsize(args.serve_log)
                  if args.serve_log and os.path.exists(args.serve_log) else 0)

    messages = [{"role": "system", "content": system}]
    turns = []
    for turn in range(1, args.turns + 1):
        task = ("Review this hunk from glm.c. Point out ONE concrete issue and show "
                "the fix." if disciplined else
                "Review this hunk from glm.c. Explain what it does and discuss any "
                "issues you find, with fixed code.")
        messages.append({"role": "user",
                         "content": f"[turn {turn}] {task}\n```c\n{chunks[turn-1]}\n```"})
        if turn not in sample:
            messages.append({"role": "assistant", "content": CANNED_REPLY})
            continue
        request_messages = messages
        if not disciplined:  # break the token prefix -> full re-prefill (control)
            nonce = uuid.uuid4().hex
            request_messages = ([{"role": "system",
                                  "content": f"session-nonce: {nonce}\n{system}"}]
                                + messages[1:])
        payload = {"model": args.model_id, "messages": request_messages,
                   "stream": True, "stream_options": {"include_usage": True},
                   "temperature": 0, "max_tokens": max_tokens,
                   "cache_slot": args.slot if disciplined else args.control_slot}
        started = time.strftime("%H:%M:%S")
        result = sse_request(args.host, args.port, payload, args.timeout)
        usage = result["usage"] or {}
        completion = usage.get("completion_tokens", 0)
        decode_tps = (completion / result["decode_s"]
                      if result["decode_s"] and completion else None)
        record = {"turn": turn, "started": started,
                  "prompt_tokens": usage.get("prompt_tokens"),
                  "completion_tokens": completion,
                  "ttft_s": round(result["ttft_s"], 2) if result["ttft_s"] else None,
                  "decode_tps": round(decode_tps, 3) if decode_tps else None,
                  "total_s": round(result["total_s"], 2),
                  "finish_reason": result["finish_reason"]}
        turns.append(record)
        print(f"[{args.mode}] turn {turn}: prompt={record['prompt_tokens']} "
              f"out={completion} ttft={record['ttft_s']}s "
              f"decode={record['decode_tps']} tok/s total={record['total_s']}s "
              f"finish={record['finish_reason']}", flush=True)
        reply = result["text"].strip() or CANNED_REPLY
        messages.append({"role": "assistant",
                         "content": reply if disciplined else CANNED_REPLY})

    prefills = parse_serve_log(args.serve_log, log_offset)
    measured = [t["turn"] for t in turns]
    for record, prefill in zip(turns, prefills):
        record.update(prefill)
    if len(prefills) != len(turns):
        print(f"warning: {len(prefills)} [API] log lines for {len(turns)} measured "
              "turns; alignment may be off", file=sys.stderr)
    out = {"mode": args.mode, "model_id": args.model_id, "turns": turns,
           "max_tokens": max_tokens, "sampled_turns": sorted(measured),
           "chars_per_turn": args.chars_per_turn,
           "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"wrote {args.out}")


def verdict(reuse_path, control_path, slack_tokens=64, tps_ratio=0.90):
    with open(reuse_path, encoding="utf-8") as handle:
        reuse = json.load(handle)
    with open(control_path, encoding="utf-8") as handle:
        control = json.load(handle)
    failures = []
    reuse_turns = reuse["turns"]
    for prev, cur in zip(reuse_turns, reuse_turns[1:]):
        if cur.get("prefill_tokens") is None or cur.get("prompt_tokens") is None:
            failures.append(f"turn {cur['turn']}: missing prefill/usage data")
            continue
        delta = cur["prompt_tokens"] - prev["prompt_tokens"]
        if cur["prefill_tokens"] > delta + slack_tokens:
            failures.append(f"turn {cur['turn']}: prefilled {cur['prefill_tokens']} "
                            f"tokens but the new-content delta is only {delta}")
    reuse_tps = [t["decode_tps"] for t in reuse_turns[1:] if t.get("decode_tps")]
    control_tps = [t["decode_tps"] for t in control["turns"] if t.get("decode_tps")]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    if not reuse_tps or not control_tps:
        failures.append("missing decode tok/s samples")
    elif mean(reuse_tps) < tps_ratio * mean(control_tps):
        failures.append(f"decode {mean(reuse_tps):.2f} tok/s (reuse) < "
                        f"{tps_ratio:.0%} of control {mean(control_tps):.2f} tok/s")
    summary = {
        "reuse_prefill_per_turn_2plus": [t.get("prefill_tokens") for t in reuse_turns[1:]],
        "reuse_ttft_s_turn_2plus": [t.get("ttft_s") for t in reuse_turns[1:]],
        "control_ttft_s": [t.get("ttft_s") for t in control["turns"]],
        "reuse_decode_tps_mean": round(mean(reuse_tps), 3),
        "control_decode_tps_mean": round(mean(control_tps), 3),
        "reuse_avg_output_tokens": round(mean(
            [t["completion_tokens"] for t in reuse_turns]), 1),
        "control_avg_output_tokens": round(mean(
            [t["completion_tokens"] for t in control["turns"]]), 1),
    }
    print(json.dumps(summary, indent=2))
    if failures:
        print("GATE: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("GATE: PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reuse", "control"))
    parser.add_argument("--verdict", nargs=2, metavar=("REUSE_JSON", "CONTROL_JSON"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=None,
                        help="chat-completions `model` field; default: query the "
                             "running server's GET /v1/models and use its first "
                             f"advertised id, falling back to {DEFAULT_MODEL_ID!r} if "
                             "that query fails")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--sample", default="2,6,10",
                        help="control mode: which turns actually hit the engine")
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="reuse mode per-turn cap (Step 2 discipline)")
    parser.add_argument("--control-max-tokens", type=int, default=512)
    parser.add_argument("--chars-per-turn", type=int, default=4500,
                        help="~1-1.5K tokens of new code context per turn")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--control-slot", type=int, default=1)
    parser.add_argument("--source", default=os.path.join(HERE, os.pardir, "glm.c"))
    parser.add_argument("--serve-log", default=None,
                        help="serve stderr log (for [API] prefill lines)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.verdict:
        sys.exit(verdict(*args.verdict))
    if not args.mode:
        parser.error("--mode or --verdict is required")
    if not args.out:
        parser.error("--out is required with --mode")
    if not args.model_id:
        args.model_id = resolve_model_id(args.host, args.port)
    print(f"[model-id] using {args.model_id!r}", file=sys.stderr)
    run_transcript(args)


if __name__ == "__main__":
    main()
