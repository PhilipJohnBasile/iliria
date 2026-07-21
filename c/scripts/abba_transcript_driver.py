#!/usr/bin/env python3
"""HTTP driver + mock engine for `run_abba_matrix.sh` (the same-commit serve-mode
MP=0/1 ABBA gate). Two subcommands:

  drive        Load a fixed, scripted 3-turn monotonic-history transcript into
               `--slot` of an already-running `ili serve`, recording per turn:
               TTFT, prompt/completion tokens, decode tok/s, prefill tok/s (from
               the serve log's `[API] PROFILE prefill ...` line, same regex
               family as scripts/long_ctx_profile.py), and a sha256 of the
               generated text. Writes one JSON result file per arm.

               HTTP/SSE driving reuses scripts/serve_gate.py's `sse_request` and
               `load_chunks` machinery directly (imported, not copied) per the
               same dependency-free approach documented there; this script adds
               only the prefill-wall-time parsing serve_gate.py does not do.

  mock-serve   An in-process stub HTTP server standing in for `ili serve` for
               `run_abba_matrix.sh --dry-run`: no engine, no glm. Speaks just
               enough of the `/v1/chat/completions` + `/health` protocol to
               exercise `drive` end to end, and appends synthetic
               `[API] KV slot ...` / `[API] PROFILE ...` lines to a log file in
               the exact format the real engine emits (glm.c `layer_forward`
               callers), so the SAME log-parsing code in `drive` runs unchanged
               against real or mock output. It also forks a decoy child process
               literally named `glm` (a renamed copy of /bin/sleep, so
               `pgrep -x glm` finds it exactly like the real engine) -- this
               lets a dry run genuinely exercise the "kill the wrapper AND the
               glm child" teardown path in run_abba_matrix.sh instead of just
               assuming it. `--orphan-on-term` (fault injection, dry-run only)
               makes the mock NOT reap that child on SIGTERM, the way a real
               `ili serve` wrapped in `caffeinate` can leave `glm` orphaned if
               caffeinate itself is killed without forwarding the signal --
               empirically confirmed on this machine on 2026-07-15 (killing
               `caffeinate -dimsu sleep N` left `sleep` running, reparented to
               launchd) -- proving why run_abba_matrix.sh's teardown must
               independently pgrep+kill `glm` rather than trust that signaling
               the top-level process is enough.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_gate  # noqa: E402  (load_chunks, sse_request -- see module docstring)

# ---- shared with scripts/serve_gate.py's transcript style ---------------------------------

DISCIPLINED_SYSTEM = serve_gate.DISCIPLINED_SYSTEM
DEFAULT_SOURCE = str(HERE.parent / "glm.c")

# Same PROFILE line glm.c's layer_forward callers emit (commit 8d26555, see
# scripts/long_ctx_profile.py's PROFILE_RE): tokens + wall time for the prefill
# that immediately preceded this turn's measurement. serve_gate.py's own
# API_LINE regex (KV slot/prefix/prefill-token-count) does not carry wall time,
# so this driver needs its own PROFILE_RE for the prefill-tok/s figure the ABBA
# gate records per arm.
PROFILE_RE = re.compile(
    r"\[API\] PROFILE prefill (\d+) tok in ([\d.]+)s:")


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
    tok, wall = matches[-1].groups()
    tok, wall = int(tok), float(wall)
    return {"prefill_tokens": tok, "prefill_wall_s": wall,
            "prefill_tok_s": round(tok / wall, 3) if wall > 0 else None}


def parse_last_kv_line(text):
    matches = list(serve_gate.API_LINE.finditer(text))
    if not matches:
        return None
    slot, prefix, prompt, prefill = matches[-1].groups()
    return {"slot": int(slot), "prefix_tokens": int(prefix),
            "prompt_tokens": int(prompt), "prefill_delta_tokens": int(prefill)}


# ---- drive: the scripted 3-turn transcript --------------------------------------------

def build_turns(source, turns, chars_per_turn):
    """3 monotonic-history turns sliced from a real source file (never random
    tokens) -- same shape as serve_gate.py's 'reuse' mode, simplified: ABBA's
    independent variable is ILI_METAL_PREFILL, not reuse-vs-control, so there
    is only one transcript, run identically in every arm."""
    return serve_gate.load_chunks(source, turns, chars_per_turn)


def run_drive(args):
    if not args.model_id:
        # Query the already-running ili serve (real, or this module's own mock-serve
        # subcommand under run_abba_matrix.sh --dry-run) for its advertised model id --
        # see serve_gate.resolve_model_id's docstring for the 2026-07-14 incident this
        # guards against (a stale/hardcoded client model-id 404ing against a renamed
        # server).
        args.model_id = serve_gate.resolve_model_id(args.host, args.port)
    print(f"[model-id] using {args.model_id!r}", file=sys.stderr)
    chunks = build_turns(args.source, args.turns, args.chars_per_turn)
    log_offset = (os.path.getsize(args.serve_log)
                  if args.serve_log and os.path.exists(args.serve_log) else 0)
    messages = [{"role": "system", "content": DISCIPLINED_SYSTEM}]
    turns = []
    for turn in range(1, args.turns + 1):
        task = ("Review this hunk from glm.c. Point out ONE concrete issue and show "
                "the fix.")
        messages.append({"role": "user",
                         "content": f"[turn {turn}] {task}\n```c\n{chunks[turn - 1]}\n```"})
        payload = {"model": args.model_id, "messages": messages, "stream": True,
                   "stream_options": {"include_usage": True}, "temperature": 0,
                   "max_tokens": args.max_tokens, "cache_slot": args.slot}
        result = serve_gate.sse_request(args.host, args.port, payload, args.timeout)
        usage = result["usage"] or {}
        completion = usage.get("completion_tokens", 0)
        decode_s = result["decode_s"]
        decode_tps = (completion / decode_s) if decode_s and completion else None
        text = result["text"] or ""

        log_text, log_offset = tail_log(args.serve_log, log_offset)
        profile = parse_last_profile(log_text)
        kv = parse_last_kv_line(log_text)

        record = {
            "turn": turn,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion,
            "ttft_s": round(result["ttft_s"], 3) if result["ttft_s"] else None,
            "decode_tok_s": round(decode_tps, 3) if decode_tps else None,
            "total_s": round(result["total_s"], 3),
            "finish_reason": result["finish_reason"],
            "prefill_delta_tokens": (kv or {}).get("prefill_delta_tokens"),
            "prefill_tok_s": (profile or {}).get("prefill_tok_s"),
            "prefill_wall_s": (profile or {}).get("prefill_wall_s"),
            "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        turns.append(record)
        print(f"[drive] turn {turn}: prompt={record['prompt_tokens']} "
              f"ttft={record['ttft_s']}s decode={record['decode_tok_s']} tok/s "
              f"prefill={record['prefill_tok_s']} tok/s "
              f"hash={record['output_sha256'][:12]}", flush=True)
        messages.append({"role": "assistant", "content": text.strip() or "(empty)"})

    combined = hashlib.sha256(
        "\n".join(m["content"] for m in messages if m["role"] == "assistant").encode("utf-8")
    ).hexdigest()
    out = {"turns": turns, "combined_output_sha256": combined,
           "max_tokens": args.max_tokens, "slot": args.slot,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"wrote {args.out}")


# ---- mock-serve: in-process stub engine for --dry-run -------------------------------------

class _MockState:
    def __init__(self, log_path, decode_tps, model_id=serve_gate.DEFAULT_MODEL_ID):
        self.log_path = log_path
        self.decode_tps = decode_tps
        self.model_id = model_id
        self.slot_len = {}
        self.lock = threading.Lock()


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # keep dry-run stderr quiet, like long_ctx_profile.py's mock

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/v1/models":
            state = self.server.mock_state
            body = json.dumps({"object": "list", "data": [
                {"id": state.model_id, "object": "model", "created": 0}]}).encode()
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
        state = self.server.mock_state
        # Mirror openai_server.py's check_model(): reject any non-matching `model` field
        # with the same HTTP 404 the real server gives (see run_drive()'s resolve_model_id
        # call above -- this is what would catch a regression of the 2026-07-14 incident
        # even under --dry-run, before it ever reaches a real serve).
        requested_model = body.get("model")
        if requested_model != state.model_id:
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
        with state.lock:
            prev = state.slot_len.get(cache_slot, 0)
            approx_tokens = len(prompt_proxy) // 4
            prefill = max(0, approx_tokens - prev)
            state.slot_len[cache_slot] = approx_tokens
            with open(state.log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[API] KV slot {cache_slot} prefix {prev}/{approx_tokens} "
                             f"token, prefill {prefill}\n")
                if prefill > 0:
                    # Fabricated but plausible split; only exercises the parser.
                    wall = prefill / 9000.0
                    handle.write(
                        f"[API] PROFILE prefill {prefill} tok in {wall:.3f}s: "
                        f"expert-disk {wall*0.02:.3f}s | expert-matmul {wall*0.55:.3f}s | "
                        f"attention {wall*0.35:.3f}s (kvb {wall*0.05:.3f}s) | "
                        f"lm_head {wall*0.02:.3f}s | other {wall*0.06:.3f}s\n")

        completion = max_tokens
        decode_s = completion / state.decode_tps
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(min(prefill * 0.0003, 0.05))
        step = decode_s / max(completion, 1)
        # Deterministic filler that varies by mode AND by turn (approx_tokens),
        # so the same arm's 3 turns hash differently from each other, MP=0 vs
        # MP=1 provably differ, and re-running the identical arm reproduces the
        # identical hashes -- exercising the hash column honestly rather than
        # returning a constant.
        filler = f"m{self.server.mock_mode}-{approx_tokens}-" * (completion // 2 + 1)
        for ch in filler[:completion]:
            time.sleep(step)
            chunk = {"choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        usage_chunk = {"choices": [], "usage": {"prompt_tokens": approx_tokens,
                       "completion_tokens": completion,
                       "total_tokens": approx_tokens + completion}}
        self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


# Self-limiting on purpose: this decoy deliberately collides with the real engine's
# process name so `pgrep -x glm` (used both by run_abba_matrix.sh's teardown and by
# scripts/quiesce_check.sh's "no competing engine" check) finds it exactly like the
# real thing -- that collision is the point, it is what proves the dual-kill teardown
# path for real. But that same collision makes an ESCAPED decoy (killed by nothing,
# e.g. an ad hoc `mock-serve` invocation outside run_abba_matrix.sh that nobody tore
# down) a false positive against every OTHER caller's "is an engine running?" check on
# the machine -- observed for real on 2026-07-15 (an orphaned decoy from manual
# debugging tripped a concurrent agent's two-engines check ~24 minutes later). A
# bounded lifetime, not just SIGTERM handling, keeps a missed teardown cheap: the
# decoy dies on its own well before it can plausibly outlive anyone's test session.
_DECOY_LIFETIME_S_DEFAULT = 900   # 15 min
# Plain concatenation, not str.format(): the C source's own braces (function body)
# would otherwise be parsed as format fields.
_DECOY_SOURCE = (
    "#include <unistd.h>\n#include <stdlib.h>\n"
    "int main(int argc, char **argv){\n"
    "    unsigned s = argc > 1 ? (unsigned)atoi(argv[1]) : " + str(_DECOY_LIFETIME_S_DEFAULT) + ";\n"
    "    sleep(s); return 0;\n"
    "}\n"
)


def _spawn_decoy_glm(work_dir, lifetime_s=_DECOY_LIFETIME_S_DEFAULT):
    """Compile a trivial local binary literally named `glm` and exec it, so
    `pgrep -x glm` finds a real process -- see module docstring. Self-terminates
    after `lifetime_s` regardless of whether anything ever signals it (see the
    module-level comment above `_DECOY_SOURCE`).

    NOTE: copying an existing signed system binary (e.g. /bin/sleep) to a new
    path and renaming it does NOT work here -- macOS's code-signing
    enforcement SIGKILLs a relocated platform binary on exec (verified
    empirically: exit code 137, `codesign -dv` shows the copy still carries
    the original com.apple.sleep platform identity). A freshly compiled,
    unsigned local binary is not subject to that check and runs normally.
    """
    decoy_c = Path(work_dir) / "glm_decoy.c"
    decoy = Path(work_dir) / "glm"
    decoy_c.write_text(_DECOY_SOURCE, encoding="utf-8")
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if not cc:
        print("[mock-serve] warning: no C compiler found; skipping decoy glm child "
              "(the dual-process teardown path will not be exercised)", file=sys.stderr)
        return None
    result = subprocess.run([cc, "-O0", "-o", str(decoy), str(decoy_c)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[mock-serve] warning: decoy glm compile failed, skipping: {result.stderr}",
              file=sys.stderr)
        return None
    return subprocess.Popen([str(decoy), str(int(lifetime_s))])


def _reap_decoy(decoy):
    if decoy is None or decoy.poll() is not None:
        return
    decoy.terminate()
    try:
        decoy.wait(timeout=5)
    except subprocess.TimeoutExpired:
        decoy.kill()


def run_mock_serve(args):
    open(args.serve_log, "a", encoding="utf-8").close()
    server = ThreadingHTTPServer((args.host, args.port), _MockHandler)
    server.mock_state = _MockState(args.serve_log, args.decode_tps, model_id=args.model_id)
    server.mock_mode = args.metal_prefill
    decoy = (_spawn_decoy_glm(args.work_dir, args.decoy_lifetime_s)
             if args.spawn_decoy_glm else None)
    if decoy:
        print(f"[mock-serve] decoy glm child pid={decoy.pid} "
              f"self-destructs in {args.decoy_lifetime_s}s regardless", file=sys.stderr)
        if not args.orphan_on_term:
            # Defense in depth beneath the SIGTERM path below: atexit still runs on an
            # uncaught exception or a plain sys.exit(), which do not go through
            # handle_term. It will not run on SIGKILL -- nothing can -- which is exactly
            # why the decoy also self-destructs on its own timer regardless of this.
            # Conditional on orphan_on_term like the SIGTERM path: an unconditional
            # atexit reap would silently defeat --orphan-on-term's entire purpose
            # (proving run_abba_matrix.sh's independent glm-kill fallback is actually
            # exercised, not just assumed) by cleaning up on ordinary interpreter exit
            # regardless of the flag.
            import atexit
            atexit.register(_reap_decoy, decoy)

    stop = threading.Event()

    def handle_term(signum, frame):
        # Signal handlers must be reentrant-safe: do nothing but set a flag. A real
        # teardown can (and, from the bash side, sometimes does) deliver SIGTERM more
        # than once in quick succession -- e.g. once as the port's lsof-found listener
        # and once as the tracked wrapper PID, which are frequently the SAME process.
        # Doing I/O (print) here is NOT safe: a second signal arriving while the first
        # call is mid-`print` re-enters the stdio buffered writer and Python raises
        # `RuntimeError: reentrant call inside <_io.BufferedWriter>`, crashing this
        # process with the decoy child never reaped -- i.e. it manufactures exactly
        # the orphan-wrapper failure mode this fixture exists to test, by accident,
        # on the "well-behaved" arms. All the real cleanup happens below, once, after
        # the main loop notices the flag.
        stop.set()

    import signal
    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[mock-serve] listening on http://{args.host}:{args.port} "
          f"metal_prefill={args.metal_prefill} log={args.serve_log}", file=sys.stderr)
    while not stop.is_set():
        time.sleep(0.1)
    print(f"[mock-serve] stopping, orphan_on_term={args.orphan_on_term}", file=sys.stderr)
    if not args.orphan_on_term:
        _reap_decoy(decoy)
    server.shutdown()
    server.server_close()
    print("[mock-serve] shut down", file=sys.stderr)


# ---- summarize: paired ABBA table + medians (design item d) ---------------------------

def _quiesce_passed(path):
    if not path or not os.path.exists(path):
        return None
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if "ALL CONDITIONS PASS" in text:
        return True
    if "NOT QUIESCED" in text:
        return False
    return None


def run_summarize(args):
    import statistics

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result_dir = Path(args.result_dir)
    arms = sorted(manifest["arms"], key=lambda a: a["arm"])
    for arm in arms:
        arm["data"] = json.loads((result_dir / arm["json"]).read_text(encoding="utf-8"))

    lines = [
        "# ABBA matrix: same-commit serve MP=0/1 (`ILI_METAL_PREFILL`)",
        "",
        f"attempt_id: `{manifest['attempt_id']}` | dry_run: {manifest['dry_run']} | "
        f"prefill_chunk: {manifest['prefill_chunk']} | generated: "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Arm order is ABBA (MP=0, MP=1, MP=1, MP=0) to control for monotonic drift over "
        "the run, the same convention as ab-m5max-k6-matrix.sh's off/candidate "
        "alternation.",
        "",
        "## Quiesce gate, before/after each arm",
        "",
        "| arm | mode | before | after |",
        "|---|---|---|---|",
    ]

    def fmt_pass(v):
        return "PASS" if v is True else ("NOT QUIESCED" if v is False else "n/a")

    for arm in arms:
        before = fmt_pass(_quiesce_passed(result_dir / arm["quiesce_before"]))
        after = fmt_pass(_quiesce_passed(result_dir / arm["quiesce_after"]))
        lines.append(f"| {arm['arm']} | MP={arm['metal_prefill']} | {before} | {after} |")

    lines += ["", "## Paired per-turn table", ""]
    header = ["turn", "metric"] + [f"arm{arm['arm']} (MP={arm['metal_prefill']})" for arm in arms]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    n_turns = len(arms[0]["data"]["turns"])
    metrics = [("ttft_s", "TTFT (s)"), ("prefill_tok_s", "prefill tok/s"),
               ("decode_tok_s", "decode tok/s"), ("output_sha256", "output sha256 (12c)")]
    for t in range(n_turns):
        for key, label in metrics:
            row = [str(t + 1), label]
            for arm in arms:
                v = arm["data"]["turns"][t].get(key)
                if key == "output_sha256" and v:
                    v = v[:12]
                row.append("n/a" if v is None else str(v))
            lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Per-mode medians (across the 2 same-mode arms, per turn)", ""]
    by_mode = {}
    for arm in arms:
        by_mode.setdefault(arm["metal_prefill"], []).append(arm)
    header2 = ["turn"] + [f"MP={mode} median {m[1]}" for mode in sorted(by_mode)
                          for m in (("ttft_s", "TTFT(s)"), ("prefill_tok_s", "prefill tok/s"),
                                   ("decode_tok_s", "decode tok/s"))]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("|" + "---|" * len(header2))
    for t in range(n_turns):
        row = [str(t + 1)]
        for mode in sorted(by_mode):
            for key, _ in (("ttft_s", None), ("prefill_tok_s", None), ("decode_tok_s", None)):
                vals = [a["data"]["turns"][t].get(key) for a in by_mode[mode]]
                vals = [v for v in vals if v is not None]
                row.append(f"{statistics.median(vals):.4g}" if vals else "n/a")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Determinism check (combined output sha256 per arm)", ""]
    lines.append("| arm | mode | combined_output_sha256 |")
    lines.append("|---|---|---|")
    for arm in arms:
        lines.append(f"| {arm['arm']} | MP={arm['metal_prefill']} | "
                     f"`{arm['data']['combined_output_sha256']}` |")
    for mode, group in sorted(by_mode.items()):
        hashes = {a["data"]["combined_output_sha256"] for a in group}
        arm_ids = [a["arm"] for a in group]
        verdict = "MATCH (repeatable at this mode)" if len(hashes) == 1 else "DIVERGE"
        lines.append(f"\narm {arm_ids} (MP={mode}): {verdict}")
    if len(by_mode) == 2:
        modes = sorted(by_mode)
        h0 = {a["data"]["combined_output_sha256"] for a in by_mode[modes[0]]}
        h1 = {a["data"]["combined_output_sha256"] for a in by_mode[modes[1]]}
        cross = "MATCH" if h0 == h1 else "DIVERGE"
        lines.append(
            f"\nMP={modes[0]} vs MP={modes[1]}: {cross} -- run-m5max-serve.sh documents "
            "ILI_METAL_PREFILL=1 as accepting rounding-variance greedy forks vs the "
            "byte-exact CPU default, so a DIVERGE here is the expected/accepted outcome, "
            "not a bug; this table exists to make it a measured fact instead of an "
            "assumption.")

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


# ---- CLI -----------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    drive = sub.add_parser("drive", help="drive the scripted 3-turn transcript over HTTP")
    drive.add_argument("--host", default="127.0.0.1")
    drive.add_argument("--port", type=int, required=True)
    drive.add_argument("--model-id", default=None,
                       help="chat-completions `model` field; default: query the "
                            "running server's GET /v1/models and use its first "
                            "advertised id, falling back to "
                            f"{serve_gate.DEFAULT_MODEL_ID!r} if that query fails")
    drive.add_argument("--turns", type=int, default=3)
    drive.add_argument("--max-tokens", type=int, default=300)
    drive.add_argument("--chars-per-turn", type=int, default=4500)
    drive.add_argument("--slot", type=int, default=0)
    drive.add_argument("--source", default=DEFAULT_SOURCE)
    drive.add_argument("--serve-log", default=None, help="serve stderr log, for PROFILE/KV lines")
    drive.add_argument("--out", required=True)
    drive.add_argument("--timeout", type=float, default=1800)

    mock = sub.add_parser("mock-serve", help="in-process stub engine for --dry-run")
    mock.add_argument("--host", default="127.0.0.1")
    mock.add_argument("--port", type=int, required=True)
    mock.add_argument("--serve-log", required=True)
    mock.add_argument("--model-id", default=serve_gate.DEFAULT_MODEL_ID,
                      help="model id this stub server advertises via GET /v1/models "
                           "(testing only)")
    mock.add_argument("--metal-prefill", type=int, choices=(0, 1), default=0)
    mock.add_argument("--decode-tps", type=float, default=400.0)
    mock.add_argument("--work-dir", default=None, help="where to place the decoy `glm` binary")
    mock.add_argument("--spawn-decoy-glm", action="store_true", default=True)
    mock.add_argument("--no-spawn-decoy-glm", dest="spawn_decoy_glm", action="store_false")
    mock.add_argument("--decoy-lifetime-s", type=int, default=_DECOY_LIFETIME_S_DEFAULT,
                      help="hard self-destruct bound for the decoy glm child: it exits on "
                           "its own after this many seconds even if nothing ever signals "
                           "it or this mock-serve process (a leaked decoy that collides "
                           "with `pgrep -x glm` is a false positive for every OTHER "
                           "caller's engine-running check on the machine, not just this "
                           "one's)")
    mock.add_argument("--orphan-on-term", action="store_true",
                      help="fault injection: do not reap the decoy glm child on SIGTERM "
                           "(it still self-destructs after --decoy-lifetime-s)")

    summarize = sub.add_parser("summarize", help="write results.md from a manifest.json")
    summarize.add_argument("--manifest", required=True)
    summarize.add_argument("--result-dir", required=True,
                           help="directory manifest.json's relative json/quiesce paths resolve against")
    summarize.add_argument("--out", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "mock-serve" and not args.work_dir:
        args.work_dir = str(Path(args.serve_log).resolve().parent)
    {"drive": run_drive, "mock-serve": run_mock_serve,
     "summarize": run_summarize}[args.command](args)


if __name__ == "__main__":
    main()
