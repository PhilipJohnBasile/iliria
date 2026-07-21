#!/usr/bin/env python3
"""32K/64K/128K decode re-profile harness — the campaign's deciding measurement.

THE QUESTION THIS ANSWERS: at realistic long-agent-session context depths, how much of
the DECODE critical path is expert page-in from disk? If expert page-in is <~1% of decode
wall time at real contexts, the entire expert-prefetch category (K6/PILOT and everything
like it) is dead on arrival regardless of prediction quality — there is nothing left to
hide behind prefetch. `docs/roadmap-daily-driver.md` and the memory record already killed
K6/PILOT and eviction-veto at ~1.5-5K context; this harness exists to check whether that
verdict still holds an order of magnitude deeper, where the KV pool and expert working set
both look very different.

INSTRUMENTATION HONESTY NOTE: `c/glm.c` currently profiles PREFILL only — the per-turn
`[API] PROFILE prefill ...` line (commit 8d26555) is printed once, right after the
prefill step() call and BEFORE spec_decode() runs, so it never covers decode. There is no
engine-side decode-phase expert-disk/attention/matmul breakdown to read, and adding one
would mean editing glm.c, which is out of scope here (the quality bench owns the engine).
So this harness reports two honest, directly-observable things per context depth instead
of inventing a decode profile:
  1. measured DECODE throughput (tok/s) over ~200 tokens, via clean SSE client-side
     timing (time-to-first-token vs. time-after-first-token, same split serve_gate.py
     uses) — this is real, not estimated.
  2. the PROFILE line for the tiny delta-prefill that immediately precedes that decode
     run, reported as `expert_disk_share_of_delta_prefill_pct` — a PROXY for the
     disk-vs-compute split per exposed token at that context depth (same MoE routing,
     same LRU/pin cache state, same 75 layers x top-8 experts, whether the token being
     processed is a prefill token or a decode token), NOT a ground-truth decode number.
Treat column (2) as directional evidence for the go/no-go call, and column (1) as the
number that actually matters for wall-clock.

CTX / RAM CEILING: `c/glm.c` imposes no hard context ceiling (`int maxctx=getenv("CTX")
?...:4096`, no upper clamp; GLM-5.2's `max_position_embeddings` is 1,048,576 — far past
anything tested here). The practical cap is RAM: the KV pool costs
`(n_layers+1) * ctx * (kv_lora_rank + qk_rope_head_dim) * 4` bytes per slot
(`kv_pool_bytes()` in glm.c) = 79 * 576 * 4 = 182,016 bytes/token/slot for GLM-5.2's
dims (78 layers, kv_lora_rank=512, qk_rope_head_dim=64) — matching the "~182 KB/token"
note in `run-m5max-serve.sh`. At the default `ILI_RAM_GB=114`, `ILI_KV_SLOTS=1`:

    32K tok: KV pool  5.82 GB -> ~98.3 GB left for dense model (~9.9 GB) + expert cache
    64K tok: KV pool 11.65 GB -> ~92.5 GB left
   128K tok: KV pool 23.30 GB -> ~80.8 GB left

All three targets fit comfortably (see `check_ctx_budget()` below, which reproduces this
arithmetic and refuses to run a depth that would leave less than `MIN_EXPERT_HEADROOM_GB`
for the rest of the model). What does NOT fit is the *launcher default*:
`run-m5max-serve.sh`'s `ILI_CTX=40960` (40K) is a hard per-slot allocation
(`kv_alloc(m, maxctx)` sizes fixed arrays at server startup) — it must be raised before
starting `ili serve` or the 64K/128K targets will silently hit the context-reset path
mid-load instead of measuring the depth you asked for. See "tomorrow" below.

OPERATIONAL SEQUENCING FOR TOMORROW (engine-gated; this script never spawns `glm` itself,
it only talks HTTP to an already-running `ili serve`, exactly like `scripts/serve_gate.py`):

  1. Confirm nothing else is using the engine:
       pgrep -x glm                     # must print nothing
     This script's own preflight (`refuse_if_bench_running`) re-checks this once at
     startup and refuses to proceed if it finds a `glm` process — the quality bench and
     the container-plan agent's tools own the engine and SSD; this study must never race
     them for the disk.

  2. Start a DEDICATED serve process with a raised context ceiling (in another terminal,
     from `c/`):
       ILI_CTX=140000 ILI_KV_SLOTS=1 ILI_METAL_PREFILL=1 \\
         bash run-m5max-serve.sh /path/to/GLM-5.2-int4-with-int8-mtp \\
         2> serve.log &
     (140000 rather than 131072 to leave slack for the trailing "measurement" question
     plus ~200 generated tokens without hitting the reset-on-overflow path.)

  3. Dry-run the harness against the real port with --mock OFF but small targets first if
     paranoid, or go straight to the full matrix:
       python3 scripts/long_ctx_profile.py --serve-log serve.log \\
         --targets 32000,64000,128000 --decode-tokens 200 --cold \\
         --out-csv bench-m5max/long-ctx-profile-$(date +%Y%m%d).csv \\
         --out-md  bench-m5max/long-ctx-profile-$(date +%Y%m%d).md

  4. Read the markdown table's `decode_tok_s` and `expert_disk_share_of_delta_prefill_pct`
     columns against the <~1% threshold in the TL;DR above.

TONIGHT (no engine, per the campaign's hard constraint): everything below `--mock` is
exercised end-to-end against an in-process stub HTTP server that speaks just enough of the
`/v1/chat/completions` protocol (including SSE + the `[API] KV slot`/`[API] PROFILE`
stderr lines) to drive corpus generation, arg parsing, the CTX-budget preflight, the log
parsers, and CSV/markdown output — with fabricated, clearly-non-authoritative timings:

    python3 scripts/long_ctx_profile.py --mock --targets 4000,8000 --decode-tokens 20 \\
      --out-csv /tmp/mock.csv --out-md /tmp/mock.md
"""

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_C_DIR = HERE.parent   # c/

# ---- corpus generator: deterministic, from real repo files, never random tokens ----------

CHARS_PER_TOKEN = 4   # same rough heuristic as c/openai_server.py's context-diet feature
DEFAULT_SOURCES = ("glm.c", "openai_server.py", "backend_metal.mm", "resource_plan.py",
                   "scripts/serve_gate.py", "tok.h", "st.h", "json.h", "doctor.py")
DEFAULT_SYSTEM_PROMPT = ("You are a senior systems engineer investigating a local "
                         "inference engine. Use the provided tool output to answer "
                         "precisely and concisely.")

# openai_server.py serve()'s own --model-id default. Kept here only as the LAST-RESORT
# fallback for resolve_model_id() below -- see the 2026-07-14 incident where every
# client's hardcoded/defaulted `glm-5.2` 404'd the moment a naming scrub made the server
# start advertising `glm-5.2-iliria` instead (openai_server.py's check_model() rejects
# any `model` field that does not match the running server's model_id, HTTP 404). Never
# hardcode this value into a request payload directly -- call resolve_model_id() instead,
# so a future rename is a non-event for every script that talks HTTP.
DEFAULT_MODEL_ID = "glm-5.2-iliria"


class _LineCycler:
    """Deterministically cycles forever through the lines of `sources` under `repo_root`.
    Chunks never span two files (a chunk stops early rather than crossing a file boundary),
    which keeps the synthetic tool output readable. Same inputs -> same sequence, always:
    calling build_transcript() twice with identical arguments produces byte-identical
    transcripts, and a larger target_tokens is a strict extension of a smaller one's
    message list (both start reading the same source files in the same order from line 1)."""

    def __init__(self, repo_root, sources):
        self.repo_root = repo_root
        self.paths = [p for p in (os.path.join(repo_root, s) for s in sources) if os.path.isfile(p)]
        if not self.paths:
            raise FileNotFoundError(f"none of the corpus source files exist under {repo_root}: {list(sources)}")
        self._gen = self._lines_forever()
        self._pending = next(self._gen)

    def _lines_forever(self):
        while True:
            for path in self.paths:
                rel = os.path.relpath(path, self.repo_root)
                with open(path, encoding="utf-8", errors="replace") as handle:
                    for lineno, raw in enumerate(handle, 1):
                        yield rel, lineno, raw.rstrip("\n")

    def next_chunk(self, chars_budget):
        """Return (rel_path, start_line, end_line, chunk_text) of ~chars_budget chars."""
        rel, start_line, _ = self._pending
        lines, chars, end_line = [], 0, start_line
        while chars < chars_budget:
            cur_rel, lineno, text = self._pending
            if cur_rel != rel:
                break
            lines.append(text)
            chars += len(text) + 1
            end_line = lineno
            self._pending = next(self._gen)
        return rel, start_line, end_line, "\n".join(lines)


def build_transcript(target_tokens, repo_root, sources=DEFAULT_SOURCES, chars_per_turn=3200,
                     system_prompt=DEFAULT_SYSTEM_PROMPT):
    """A deterministic synthetic agent-session transcript: each turn is a user ask, an
    assistant tool-call, a tool response sliced from a real repo source file (code, never
    random tokens), and a short assistant reply — appended until the estimated rendered
    size (~4 chars/token) reaches target_tokens. See _LineCycler for the determinism and
    prefix-nesting guarantee across different target_tokens values."""
    cycler = _LineCycler(repo_root, sources)
    messages = [{"role": "system", "content": system_prompt}]
    target_chars = target_tokens * CHARS_PER_TOKEN
    total_chars = len(system_prompt)
    turn = 0
    while total_chars < target_chars:
        turn += 1
        rel, start_line, end_line, chunk = cycler.next_chunk(chars_per_turn)
        call_id = f"call_{turn:05d}"
        user = {"role": "user",
                "content": f"[turn {turn}] Look at {rel} around lines {start_line}-{end_line}. "
                           "Anything concerning in there?"}
        assistant_call = {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "read_file",
                         "arguments": json.dumps({"path": rel, "start": start_line, "end": end_line})}}]}
        tool_response = {"role": "tool", "tool_call_id": call_id,
                         "content": f"$ sed -n '{start_line},{end_line}p' {rel}\n{chunk}"}
        assistant_reply = {"role": "assistant",
                           "content": f"turn {turn}: {rel} lines {start_line}-{end_line} reviewed, "
                                      "nothing blocking found."}
        for message in (user, assistant_call, tool_response, assistant_reply):
            messages.append(message)
            total_chars += len(json.dumps(message))
    return messages


def estimate_prompt_tokens(messages):
    """~4 chars/token heuristic over message content + tool-call payloads. The real number
    comes from the engine's reported `usage.prompt_tokens`; this is a pre-flight estimate."""
    total_chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        for call in (message.get("tool_calls") or []):
            total_chars += len(json.dumps(call))
    return total_chars // CHARS_PER_TOKEN


# ---- CTX / RAM budget preflight (no hard engine ceiling -- RAM is the real cap) -----------

KV_BYTES_PER_TOKEN_PER_SLOT = 182_016   # GLM-5.2: (78+1)*(512+64)*4, see module docstring
DENSE_MODEL_GB = 9.9                    # always-resident dense weights (project memory record)
MIN_EXPERT_HEADROOM_GB = 20.0           # below this a run stops being daily-driver-representative


def check_ctx_budget(target_tokens, ram_gb, kv_slots):
    kv_gb = KV_BYTES_PER_TOKEN_PER_SLOT * target_tokens * kv_slots / 1e9
    headroom_gb = ram_gb - kv_gb - DENSE_MODEL_GB
    return kv_gb, headroom_gb, headroom_gb >= MIN_EXPERT_HEADROOM_GB


# ---- engine ownership gate: this script must never race the quality bench ----------------

def engine_process_count():
    """Number of running `glm` processes, or None if `pgrep` itself could not be run
    (caller must then decide). The documented workflow starts a DEDICATED `ili serve`
    (one `glm` process) before running this script, so "zero glm processes" is not the
    live-mode invariant -- "exactly the one dedicated serve, nothing else" is."""
    try:
        result = subprocess.run(["pgrep", "-x", "glm"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return 0
    return len(result.stdout.split())


def refuse_if_bench_running():
    count = engine_process_count()
    if count is None:
        print("warning: could not run `pgrep -x glm` to check for a bench run; proceeding "
              "WITHOUT the engine-ownership gate. Verify by hand that nothing else owns the "
              "engine/SSD before trusting these numbers.", file=sys.stderr)
        return
    if count == 0:
        print("error: no `glm` process found. Start the DEDICATED `ili serve` for this "
              "study first (see the module docstring step 2), then invoke this script "
              "against its --serve-log.", file=sys.stderr)
        sys.exit(2)
    if count > 1:
        print(f"refusing to start: `pgrep -x glm` found {count} running engine processes -- "
              "the quality bench (or another campaign item) is racing the dedicated serve "
              "this study is supposed to be the only thing using. Stop the other one first.",
              file=sys.stderr)
        sys.exit(2)


def cold_purge():
    """Best-effort OS page-cache purge via passwordless sudo (same convention the rest of
    the campaign uses); a no-op with a warning when that is not configured."""
    probe = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    if probe.returncode != 0:
        print("warning: passwordless `sudo purge` not available; skipping the cold-cache "
              "purge -- this depth's \"cold\" row will actually be warm.", file=sys.stderr)
        return False
    subprocess.run(["sudo", "-n", "purge"], check=False)
    return True


# ---- HTTP / SSE client (dependency-free, mirrors scripts/serve_gate.py's approach) --------

def _post_json(host, port, path, payload, timeout, api_key=None):
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", path, body, headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status} from {path}: {data[:500]!r}")
    return json.loads(data)


def resolve_model_id(host, port, timeout=10.0, api_key=None, default=DEFAULT_MODEL_ID):
    """GET /v1/models from an already-running `ili serve` and return its FIRST advertised
    model id -- the robust way to stay correct across server-side model-id renames (see the
    2026-07-14 incident: every client hardcoded/defaulted `glm-5.2`, the server started
    advertising `glm-5.2-iliria`, and openai_server.py's check_model() 404'd every one of
    them: "The model `glm-5.2` does not exist."). Never raises -- falls back to `default`
    for ANY failure (connection refused, timeout, non-200, malformed body, empty list), so
    a server that is not up yet never blocks a caller that is about to fail loudly on its
    own first real request regardless."""
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


def _sse_measure(host, port, payload, timeout, api_key=None):
    """POST a streaming chat completion; return ttft/decode timing split + usage, exactly
    like serve_gate.sse_request() (kept local here so this script has no cross-script
    import dependency)."""
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", body, headers)
    t_sent = time.monotonic()
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:500]!r}")
    t_first, usage, finish = None, None, None
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
                    if delta.get("content"):
                        if t_first is None:
                            t_first = time.monotonic()
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    t_done = time.monotonic()
    conn.close()
    return {"ttft_s": (t_first - t_sent) if t_first else None,
            "decode_s": (t_done - t_first) if t_first else None,
            "usage": usage, "finish_reason": finish}


# ---- serve-log parsing: [API] KV slot / [API] PROFILE lines ------------------------------

PROFILE_RE = re.compile(
    r"\[API\] PROFILE prefill (\d+) tok in ([\d.]+)s: expert-disk ([\d.]+)s \| "
    r"expert-matmul ([\d.]+)s \| attention ([\d.]+)s \(kvb ([\d.]+)s\) \| "
    r"lm_head ([\d.]+)s \| other ([\d.-]+)s")
KV_LINE_RE = re.compile(r"\[API\] KV slot (\d+) prefix (\d+)/(\d+) token, prefill (\d+)")


def tail_log(path, offset):
    if not path or not os.path.exists(path):
        return "", offset
    with open(path, encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        text = handle.read()
        new_offset = handle.tell()
    return text, new_offset


def parse_last_profile(text):
    matches = list(PROFILE_RE.finditer(text))
    if not matches:
        return None
    tok, wall, disk, matmul, attn, kvb, head, other = matches[-1].groups()
    return {"prefill_tokens": int(tok), "prefill_wall_s": float(wall), "expert_disk_s": float(disk),
            "expert_matmul_s": float(matmul), "attention_s": float(attn), "kvb_s": float(kvb),
            "lm_head_s": float(head), "other_s": float(other)}


def parse_last_kv_line(text):
    matches = list(KV_LINE_RE.finditer(text))
    if not matches:
        return None
    slot, prefix, prompt, prefill = matches[-1].groups()
    return {"slot": int(slot), "prefix_tokens": int(prefix), "prompt_tokens": int(prompt),
            "prefill_delta_tokens": int(prefill)}


# ---- per-depth measurement ------------------------------------------------------------

FOLLOWUP_QUESTION = {"role": "user",
                     "content": "In one sentence, summarize the last file you looked at."}


def run_one(args, transcript, target_tokens, mode, log_offset):
    """Load `transcript` (a full history up to `target_tokens`) into `args.slot`, then
    measure clean decode throughput over `args.decode_tokens` tokens on a short follow-up
    question appended to it. Returns (row_dict, new_log_offset)."""
    load_payload = {"model": args.model_id, "messages": transcript, "max_tokens": 1,
                    "temperature": 0, "cache_slot": args.slot, "stream": False}
    _post_json(args.host, args.port, "/v1/chat/completions", load_payload, args.timeout, args.api_key)
    log_text, log_offset = tail_log(args.serve_log, log_offset)
    load_profile = parse_last_profile(log_text)
    load_kv = parse_last_kv_line(log_text)

    measure_payload = {"model": args.model_id, "messages": transcript + [FOLLOWUP_QUESTION],
                       "max_tokens": args.decode_tokens, "temperature": 0,
                       "cache_slot": args.slot, "stream": True,
                       "stream_options": {"include_usage": True}}
    result = _sse_measure(args.host, args.port, measure_payload, args.timeout, args.api_key)
    log_text, log_offset = tail_log(args.serve_log, log_offset)
    measure_profile = parse_last_profile(log_text)
    measure_kv = parse_last_kv_line(log_text)

    usage = result["usage"] or {}
    completion = usage.get("completion_tokens", 0)
    decode_s = result["decode_s"]
    decode_tok_s = (completion / decode_s) if decode_s and completion else None
    profile = measure_profile or load_profile   # prefer the depth-adjacent delta-prefill sample
    disk_share_pct = (round(100.0 * profile["expert_disk_s"] / profile["prefill_wall_s"], 2)
                      if profile and profile["prefill_wall_s"] > 0 else None)

    return {
        "target_tokens": target_tokens,
        "mode": mode,
        "slot": args.slot,
        "prompt_tokens_at_load": (load_kv or {}).get("prompt_tokens"),
        "prompt_tokens_at_measure": usage.get("prompt_tokens"),
        "measure_prefill_delta_tokens": (measure_kv or {}).get("prefill_delta_tokens"),
        "ttft_s": round(result["ttft_s"], 3) if result["ttft_s"] else None,
        "decode_tokens": completion,
        "decode_s": round(decode_s, 3) if decode_s else None,
        "decode_tok_s": round(decode_tok_s, 3) if decode_tok_s else None,
        "load_expert_disk_s": (load_profile or {}).get("expert_disk_s"),
        "load_expert_matmul_s": (load_profile or {}).get("expert_matmul_s"),
        "load_attention_s": (load_profile or {}).get("attention_s"),
        "load_prefill_wall_s": (load_profile or {}).get("prefill_wall_s"),
        "measure_expert_disk_s": (measure_profile or {}).get("expert_disk_s"),
        "measure_expert_matmul_s": (measure_profile or {}).get("expert_matmul_s"),
        "measure_attention_s": (measure_profile or {}).get("attention_s"),
        "measure_prefill_wall_s": (measure_profile or {}).get("prefill_wall_s"),
        "expert_disk_share_of_delta_prefill_pct": disk_share_pct,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, log_offset


# ---- --mock: in-process stub server for tonight's dry run --------------------------------

class MockEngine:
    """Fabricated, clearly-non-authoritative timings: fast, deterministic, no disk, no GPU.
    Exists only to exercise the HTTP/SSE plumbing, the corpus generator, and the log
    parsers end to end before the real engine is available."""

    def __init__(self, log_path, decode_tps=500.0, model_id=DEFAULT_MODEL_ID):
        self.log_path = log_path
        self.decode_tps = decode_tps
        self.model_id = model_id
        self.slot_len = {}
        self.lock = threading.Lock()


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass   # keep dry-run stderr quiet

    def do_GET(self):
        if self.path == "/v1/models":
            engine = self.server.mock_engine
            body = json.dumps({"object": "list", "data": [
                {"id": engine.model_id, "object": "model", "created": 0}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        engine = self.server.mock_engine
        # Mirror openai_server.py's check_model(): reject any non-matching `model` field
        # with the same HTTP 404 the real server gives, so a stale/hardcoded client
        # default is caught right here in the mock instead of only against a real serve.
        requested_model = body.get("model")
        if requested_model != engine.model_id:
            error = json.dumps({"error": {
                "message": f"The model `{requested_model}` does not exist.",
                "type": "model_not_found", "param": "model", "code": None}}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error)))
            self.end_headers()
            self.wfile.write(error)
            return
        messages = body.get("messages") or []
        prompt_proxy = json.dumps(messages)
        cache_slot = body.get("cache_slot", 0)
        max_tokens = body.get("max_tokens", 16)
        stream = bool(body.get("stream"))
        with engine.lock:
            prev = engine.slot_len.get(cache_slot, 0)
            approx_tokens = len(prompt_proxy) // CHARS_PER_TOKEN
            prefill = max(0, approx_tokens - prev)
            engine.slot_len[cache_slot] = approx_tokens
            with open(engine.log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[API] KV slot {cache_slot} prefix {prev}/{approx_tokens} token, "
                             f"prefill {prefill}\n")
                if prefill > 0:
                    disk, matmul, attn = prefill * 0.0004, prefill * 0.006, prefill * 0.018
                    kvb, head, other = attn * 0.15, 0.010, 0.050
                    wall = disk + matmul + attn + head + other
                    handle.write(f"[API] PROFILE prefill {prefill} tok in {wall:.1f}s: "
                                 f"expert-disk {disk:.3f}s | expert-matmul {matmul:.3f}s | "
                                 f"attention {attn:.3f}s (kvb {kvb:.3f}s) | lm_head {head:.3f}s "
                                 f"| other {other:.3f}s\n")
        completion = max_tokens
        decode_s = completion / engine.decode_tps

        if not stream:
            payload = {"id": "mockcmpl", "object": "chat.completion", "created": int(time.time()),
                      "model": body.get("model", "mock"),
                      "choices": [{"index": 0, "message": {"role": "assistant", "content": "x" * completion},
                                   "finish_reason": "length"}],
                      "usage": {"prompt_tokens": approx_tokens, "completion_tokens": completion,
                               "total_tokens": approx_tokens + completion}}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(min(prefill * 0.0005, 0.05))   # separate ttft from decode_s in the mock
        step = decode_s / max(completion, 1)
        for _ in range(completion):
            time.sleep(step)
            chunk = {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        usage_chunk = {"choices": [], "usage": {"prompt_tokens": approx_tokens,
                       "completion_tokens": completion, "total_tokens": approx_tokens + completion}}
        self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def start_mock_server(host, port, log_path, model_id=DEFAULT_MODEL_ID):
    server = ThreadingHTTPServer((host, port), _MockHandler)
    server.mock_engine = MockEngine(log_path, model_id=model_id)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


# ---- output ------------------------------------------------------------------------------

CSV_COLUMNS = ["target_tokens", "mode", "slot", "prompt_tokens_at_load", "prompt_tokens_at_measure",
              "measure_prefill_delta_tokens", "ttft_s", "decode_tokens", "decode_s", "decode_tok_s",
              "load_expert_disk_s", "load_expert_matmul_s", "load_attention_s", "load_prefill_wall_s",
              "measure_expert_disk_s", "measure_expert_matmul_s", "measure_attention_s",
              "measure_prefill_wall_s", "expert_disk_share_of_delta_prefill_pct", "timestamp"]


def write_csv(rows, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows, path, meta):
    lines = ["# long_ctx_profile results", ""]
    lines.append(f"mock={meta['mock']} · ram_gb={meta['ram_gb']} · kv_slots={meta['kv_slots']} · "
                "decode_tokens={}".format(meta["decode_tokens"]))
    lines.append("")
    lines.append("Decision: if `expert_disk_share_of_delta_prefill_pct` is <~1% at real "
                "contexts AND `decode_tok_s` matches the non-disk decode floor "
                "(~2.8 tok/s, see docs/roadmap-daily-driver.md), expert-prefetch work is "
                "dead at these depths -- see the module docstring for the honesty caveat "
                "on this column (it is a delta-prefill proxy, not a direct decode profile).")
    lines.append("")
    header = ["target_tokens", "mode", "prompt_tokens", "ttft_s", "decode_tok_s",
              "expert_disk_share_pct", "load_expert_disk_s", "load_attention_s"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        values = [row.get("target_tokens"), row.get("mode"), row.get("prompt_tokens_at_measure"),
                  row.get("ttft_s"), row.get("decode_tok_s"),
                  row.get("expert_disk_share_of_delta_prefill_pct"),
                  row.get("load_expert_disk_s"), row.get("load_attention_s")]
        lines.append("| " + " | ".join("n/a" if v is None else str(v) for v in values) + " |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


# ---- CLI -----------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true",
                        help="dry run against an in-process stub server; no engine, no pgrep gate")
    parser.add_argument("--mock-port", type=int, default=8877)
    parser.add_argument("--mock-model-id", default=DEFAULT_MODEL_ID,
                        help="model id the --mock stub server advertises via GET "
                             "/v1/models (testing only)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=None,
                        help="chat-completions `model` field; default: query the "
                             "running server's GET /v1/models and use its first "
                             f"advertised id, falling back to {DEFAULT_MODEL_ID!r} if "
                             "that query fails")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--targets", default="32000,64000,128000",
                        help="comma-separated target context depths in ~tokens")
    parser.add_argument("--decode-tokens", type=int, default=200)
    parser.add_argument("--repo-root", default=str(REPO_C_DIR),
                        help="directory the corpus source files are read from (default: c/)")
    parser.add_argument("--sources", default=None,
                        help="comma-separated repo-relative source files (default: a fixed list)")
    parser.add_argument("--chars-per-turn", type=int, default=3200)
    parser.add_argument("--cold", action="store_true",
                        help="also take a cold-cache row per depth (sudo -n purge) before the warm row")
    parser.add_argument("--ram-gb", type=float, default=114.0,
                        help="ILI_RAM_GB assumption for the CTX-budget preflight (matches run-m5max-serve.sh)")
    parser.add_argument("--kv-slots", type=int, default=1)
    parser.add_argument("--force", action="store_true",
                        help="run a depth even if the CTX-budget preflight flags it as too tight")
    parser.add_argument("--serve-log", default=None,
                        help="running ili serve's stderr log (required for a live run; auto-set under --mock)")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--out-csv", default=None, required=True)
    parser.add_argument("--out-md", default=None, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    targets = sorted({int(x) for x in args.targets.split(",")})
    sources = tuple(args.sources.split(",")) if args.sources else DEFAULT_SOURCES

    if args.mock:
        log_path = args.serve_log or os.path.join(tempfile.gettempdir(), "long_ctx_profile_mock.log")
        open(log_path, "w").close()
        start_mock_server(args.host, args.mock_port, log_path, model_id=args.mock_model_id)
        args.port = args.mock_port
        args.serve_log = log_path
        print(f"[mock] stub server on http://{args.host}:{args.mock_port}, log={log_path}", file=sys.stderr)
    else:
        refuse_if_bench_running()
        if not args.serve_log:
            print("error: --serve-log is required for a live run (the running `ili serve`'s "
                  "stderr log, to read [API] PROFILE/KV lines)", file=sys.stderr)
            sys.exit(2)

    if not args.model_id:
        args.model_id = resolve_model_id(args.host, args.port, api_key=args.api_key)
    print(f"[model-id] using {args.model_id!r}", file=sys.stderr)

    print(f"[ctx-budget] ram_gb={args.ram_gb} kv_slots={args.kv_slots} "
          f"dense_model_gb={DENSE_MODEL_GB}", file=sys.stderr)
    for target in targets:
        kv_gb, headroom_gb, ok = check_ctx_budget(target, args.ram_gb, args.kv_slots)
        print(f"  {target:>7} tok: KV pool {kv_gb:6.2f} GB, expert-cache headroom "
              f"{headroom_gb:6.2f} GB [{'OK' if ok else 'TIGHT'}]", file=sys.stderr)
        if not ok and not args.force:
            print(f"error: {target} tokens leaves only {headroom_gb:.1f} GB for the dense "
                  f"model + expert cache (< {MIN_EXPERT_HEADROOM_GB} GB floor) -- not "
                  "representative of the daily-driver profile. Raise --ram-gb, lower "
                  "--kv-slots, or pass --force to override.", file=sys.stderr)
            sys.exit(2)

    log_offset = os.path.getsize(args.serve_log) if os.path.exists(args.serve_log) else 0
    rows = []
    for target in targets:
        transcript = build_transcript(target, args.repo_root, sources, args.chars_per_turn)
        print(f"[corpus] target={target} approx_tokens={estimate_prompt_tokens(transcript)} "
              f"messages={len(transcript)}", file=sys.stderr)
        modes = ["cold", "warm"] if args.cold else ["warm"]
        for mode in modes:
            if mode == "cold":
                if args.mock:
                    print("[mock] --cold requested but mocked: skipping purge", file=sys.stderr)
                else:
                    cold_purge()
            row, log_offset = run_one(args, transcript, target, mode, log_offset)
            rows.append(row)
            print(f"[{mode}] target={target}: prompt~{row['prompt_tokens_at_measure']} "
                  f"decode={row['decode_tok_s']} tok/s ttft={row['ttft_s']}s "
                  f"expert_disk_share~{row['expert_disk_share_of_delta_prefill_pct']}%",
                  file=sys.stderr)

    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md, {"mock": args.mock, "ram_gb": args.ram_gb,
                                       "kv_slots": args.kv_slots, "decode_tokens": args.decode_tokens})
    print(f"wrote {args.out_csv} and {args.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
