#!/usr/bin/env python3
# Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE.
"""Dependency-free OpenAI-compatible HTTP gateway for the iliria engine."""

import argparse
import codecs
import collections
import contextlib
import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


def _env(name, default=None):
    """Env lookup with silent legacy fallback: ILI_* > COLI_* > FA_*."""
    for prefix in ("ILI_", "COLI_", "FA_"):
        v = os.environ.get(prefix + name)
        if v is not None:
            return v
    return default


HERE = Path(__file__).resolve().parent
END = b"\x01\x01END\x01\x01\n"
READY = b"\x01\x01READY\x01\x01\n"
PROMPT_BYTES_MAX = 16 << 20     # glm.c run_serve rejects \x02PROMPT frames beyond 16 MiB
MAX_BODY = 4 << 20
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://tauri.localhost",
    "tauri://localhost",
)


class APIError(Exception):
    def __init__(self, status, message, param=None, code=None, error_type="invalid_request_error",
                 headers=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type
        self.headers = headers or {}


class ClientCancelled(Exception):
    pass


def error_object(error):
    return {"error": {"message": error.message, "type": error.error_type,
                      "param": error.param, "code": error.code}}


class GenerationScheduler:
    """Bounded FIFO admission for the engine's single mutable KV context."""

    def __init__(self, max_queue=8, queue_timeout=300):
        if max_queue < 0:
            raise ValueError("max_queue cannot be negative")
        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.condition = threading.Condition()
        self.queue = collections.deque()
        self.active = False
        self.closed = False
        self.admitted = 0
        self.completed = 0
        self.rejected = 0
        self.timed_out = 0
        self.cancelled = 0

    @contextlib.contextmanager
    def admit(self, cancelled=None):
        ticket = object()
        queued_at = time.monotonic()
        with self.condition:
            if self.closed:
                raise APIError(503, "The inference scheduler is shutting down.", None,
                               "scheduler_closed", "server_error")
            if (self.active or self.queue) and len(self.queue) >= self.max_queue:
                self.rejected += 1
                raise APIError(429, "The inference queue is full.", None, "queue_full",
                               "rate_limit_error", {"Retry-After": "1"})
            self.queue.append(ticket)
            deadline = queued_at + self.queue_timeout
            while True:
                if self.closed:
                    self.queue.remove(ticket)
                    self.condition.notify_all()
                    raise APIError(503, "The inference scheduler is shutting down.", None,
                                   "scheduler_closed", "server_error")
                if not self.active and self.queue[0] is ticket:
                    break
                if cancelled and cancelled():
                    self.queue.remove(ticket)
                    self.cancelled += 1
                    self.condition.notify_all()
                    raise ClientCancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.queue.remove(ticket)
                    self.timed_out += 1
                    self.condition.notify_all()
                    raise APIError(429, "Timed out waiting for the inference engine.", None,
                                   "queue_timeout", "rate_limit_error", {"Retry-After": "1"})
                self.condition.wait(min(remaining, 0.25))
            self.queue.popleft()
            self.active = True
            self.admitted += 1
            wait_seconds = time.monotonic() - queued_at
        try:
            yield wait_seconds
        finally:
            with self.condition:
                self.active = False
                self.completed += 1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"active": self.active, "queued": len(self.queue),
                    "max_queue": self.max_queue, "queue_timeout_seconds": self.queue_timeout,
                    "admitted": self.admitted, "completed": self.completed,
                    "rejected": self.rejected, "timed_out": self.timed_out,
                    "cancelled": self.cancelled}

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class PrefixSlotRouter:
    """Choose a KV slot by prompt affinity without replacing engine-side token checks.

    The native engine remains authoritative: it tokenizes the prompt and verifies the
    exact common prefix before retaining any KV rows.  This router only keeps enough
    text-level history to avoid sending every stateless HTTP request to slot zero.
    """

    def __init__(self, slot_count):
        if slot_count < 1:
            raise ValueError("slot_count must be positive")
        self.prompts = [None] * slot_count
        self.last_used = [0] * slot_count
        self.clock = 0
        self.lock = threading.Lock()

    @staticmethod
    def _common_prefix_length(left, right):
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return index

    def select(self, prompt):
        with self.lock:
            # A complete prior prompt is the strongest conversation-affinity signal:
            # rendered follow-up turns append the prior assistant answer and next turn.
            full_matches = [
                slot for slot, previous in enumerate(self.prompts)
                if previous is not None and prompt.startswith(previous)
            ]
            if full_matches:
                # A longer partial prefix can preserve more KV than a shorter
                # previous prompt that happens to be fully contained.
                shared = [self._common_prefix_length(previous, prompt)
                          if previous is not None else -1 for previous in self.prompts]
                best = max(shared)
                candidates = [slot for slot, length in enumerate(shared) if length == best]
                return max(candidates, key=lambda slot: self.last_used[slot])

            # Give unrelated conversations their own slot while capacity remains.
            for slot, previous in enumerate(self.prompts):
                if previous is None:
                    return slot

            # Once full, retain the longest exact shared text prefix.  Equal-affinity
            # candidates use LRU replacement so generic chat-template prefixes do not
            # make one hot slot absorb every unrelated request.
            shared = [self._common_prefix_length(previous, prompt)
                      for previous in self.prompts]
            best = max(shared)
            candidates = [slot for slot, length in enumerate(shared) if length == best]
            return min(candidates, key=lambda slot: self.last_used[slot])

    def record(self, slot, prompt):
        with self.lock:
            self.clock += 1
            self.prompts[slot] = prompt
            self.last_used[slot] = self.clock


def content_text(content, param):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise APIError(400, "Message content must be a string or an array of text parts.", param)
    parts = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
            raise APIError(400, "Iliria currently supports text message content only.",
                           f"{param}.{index}", "unsupported_content_type")
        if not isinstance(part.get("text"), str):
            raise APIError(400, "Text content parts require a string `text` field.",
                           f"{param}.{index}.text")
        parts.append(part["text"])
    return "".join(parts)


# ---- GLM-5.2 tool calling -----------------------------------------------------------------
# The model expresses tool calls as ordinary text (from chat_template.jinja):
#   <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
# and tool results come back as <|observation|><tool_response>{content}</tool_response>.
# We render those markers into the prompt and parse them back into OpenAI `tool_calls`.
import re

BOX_START, BOX_END = "<tool_call>", "</tool_call>"
TR_OPEN,  TR_CLOSE = "<tool_response>", "</tool_response>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

_BOX_RE  = re.compile(re.escape(BOX_START) + r"(.*?)" + re.escape(BOX_END), re.DOTALL)
_ARG_RE  = re.compile(r"<arg_key>([^<]*)</arg_key><arg_value>(.*?)</arg_value>", re.DOTALL)
_NAME_RE = re.compile(r"\s*([A-Za-z0-9_.\-]+)")
_TAG_RE  = re.compile(r"</?arg_key>|</?arg_value>")

# De-mangler: opt-in recovery for heavily-quantized models that drop the
# <arg_key>K</arg_key><arg_value> structure. Default OFF (never rewrites well-formed output).
_SALVAGE = _env("TOOL_SALVAGE", "0") == "1"


def _tool_param_order(tools):
    """name -> ordered param names (required first) from the request schema, for de-mangling."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        params = ((fn.get("parameters") or {}).get("properties") or {})
        required = list((fn.get("parameters") or {}).get("required") or [])
        out[name] = required + [p for p in params if p not in required]
    return out


def parse_tool_calls(reply, tools=None):
    """Return (content, tool_calls). Strict GLM parse; optional de-mangler (ILI_TOOL_SALVAGE=1)
    rescues malformed int4 output by mapping a lone payload onto the tool's primary parameter."""
    param_order = _tool_param_order(tools)
    calls, salvaged = [], []
    for match in _BOX_RE.finditer(reply):
        inner = match.group(1)
        name_match = _NAME_RE.match(inner)
        name = name_match.group(1) if name_match else inner.strip()
        args = {}
        for arg in _ARG_RE.finditer(inner):
            key, value = arg.group(1), arg.group(2)
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
            args[key] = value
        if not args and _SALVAGE:
            rest = inner[name_match.end():] if name_match else ""
            payload = _TAG_RE.sub("", rest).strip()
            if payload.startswith("(") and payload.endswith(")"):
                payload = payload[1:-1].strip()
            if payload:
                key = (param_order.get(name) or ["input"])[0]
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                args = {key: payload}
                salvaged.append(name)
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    text = _BOX_RE.sub("", reply)
    if THINK_CLOSE in text:
        text = text.split(THINK_CLOSE, 1)[1]
    text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    if calls:
        dm = len(salvaged)
        sys.stderr.write("[api] tool-calls: %d total, %d strict, %d de-mangled [%s]%s\n"
                         % (len(calls), len(calls) - dm, dm, "CLEAN" if dm == 0 else "DE-MANGLED",
                            (" -> " + ", ".join(salvaged)) if dm else ""))
        sys.stderr.flush()
    return text.strip(), calls


# ---- context diet: opt-in server-side history trimming ------------------------------------
# Agent sessions resend the WHOLE message history every turn, and every prefill token costs
# real wall time (~0.09-0.7s measured on this engine). This trims old tool-role messages down
# to a small head+tail excerpt so growing histories cost less to re-render and re-prefill.
#
# PREFIX-STABILITY INVARIANT (delta-KV-reuse depends on this -- see the context-diet analysis):
# the engine matches the rendered prompt against the KV slot's stored text byte-for-byte and
# only prefills the delta (PrefixSlotRouter / [API] KV slot ... prefill ... above). If a
# message's rendered text ever changed between two requests, everything after it would look
# like new content and force a full re-prefill. So trimming is applied "at ingestion time":
# the FIRST time a given message's original content is trimmed, the excerpt is frozen in
# _TrimCache and every later request for that same original content gets the identical
# frozen bytes back -- never re-computed, never re-flowed, regardless of how the trim knobs
# change afterwards. A message crossing OUT of the full-fidelity window costs one extra
# re-prefill (its rendered text shrinks that turn); every request after that is stable again.
#
# Default OFF: ILI_TRIM_TOOL_OUTPUT_TOKENS unset/0 (or an explicit `trim_tool_output_tokens:
# 0` on the request) means trim_messages() is a no-op and history renders exactly as before.

_TRIM_CHARS_PER_TOKEN = 4   # rough heuristic; the server has no local tokenizer to consult
_TRIM_ELISION = "\n…[ili-trim: elided {chars} chars (~{tokens} tok) of tool output]…\n"


def _trim_excerpt(content, budget_tokens):
    """Pure, deterministic head+tail excerpt of `content` to ~budget_tokens. Same input ->
    same output, always -- callers must not vary this on any signal besides `content` and
    `budget_tokens`, or the prefix-stability invariant above breaks."""
    if budget_tokens <= 0:
        return content
    budget_chars = budget_tokens * _TRIM_CHARS_PER_TOKEN
    if len(content) <= budget_chars:
        return content   # already fits: trimming would not save anything
    head = budget_chars // 2
    tail = budget_chars - head
    elided = len(content) - head - tail
    marker = _TRIM_ELISION.format(chars=elided, tokens=elided // _TRIM_CHARS_PER_TOKEN)
    return content[:head] + marker + (content[-tail:] if tail else "")


class _TrimCache:
    """Freezes a message's trimmed form the first time it is computed. Keyed on the
    message's own original content ONLY -- never on the trim budget -- so a message already
    trimmed stays byte-identical even if ILI_TRIM_TOOL_OUTPUT_TOKENS changes mid-session.
    Bounded FIFO so a long-lived serve process cannot grow this without limit."""

    def __init__(self, max_entries=20000):
        self.max_entries = max_entries
        self._store = {}

    def get_or_compute(self, content, budget_tokens):
        cached = self._store.get(content)
        if cached is not None:
            return cached
        trimmed = _trim_excerpt(content, budget_tokens)
        if len(self._store) >= self.max_entries:
            self._store.pop(next(iter(self._store)))   # FIFO eviction
        self._store[content] = trimmed
        return trimmed


def _trim_cutoff(messages, keep_last_turns):
    """Index below which messages fall outside the full-fidelity window: everything at index
    >= the returned cutoff belongs to the last `keep_last_turns` turns (a turn starts at each
    user-role message) and is left untouched."""
    boundaries = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"]
    if keep_last_turns <= 0:
        return len(messages)             # nothing protected: everything is eligible
    if keep_last_turns >= len(boundaries):
        return 0                         # fewer turns than the window: nothing is eligible
    return boundaries[len(boundaries) - keep_last_turns]


def trim_messages(messages, keep_last_turns, tool_output_tokens, cache):
    """Return a new list with eligible tool-role messages replaced by a frozen head+tail
    excerpt (see the context-diet block comment above for the prefix-stability contract).
    No-op, returning `messages` unchanged, when tool_output_tokens <= 0 (the default-off
    state) or when there is nothing to trim."""
    if tool_output_tokens <= 0 or not messages:
        return messages
    cutoff = _trim_cutoff(messages, keep_last_turns)
    if cutoff <= 0:
        return messages
    out = list(messages)
    for i in range(cutoff):
        message = out[i]
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue   # list-content tool payloads are out of scope for this opt-in feature
        trimmed = cache.get_or_compute(content, tool_output_tokens)
        if trimmed is not content:
            message = dict(message)
            message["content"] = trimmed
            out[i] = message
    return out


def _trim_options(body):
    """Resolve the context-diet knobs: a `trim_tool_output_tokens` / `trim_keep_last_turns`
    request field overrides the matching ILI_TRIM_* env default (0 tokens => disabled)."""
    def resolve(field, env_name, default):
        value = body.get(field)
        if value is None:
            value = int(_env(env_name, str(default)))
        if isinstance(value, bool) or not isinstance(value, int):
            raise APIError(400, f"`{field}` must be an integer.", field)
        return value
    tool_output_tokens = resolve("trim_tool_output_tokens", "TRIM_TOOL_OUTPUT_TOKENS", 0)
    keep_last_turns = resolve("trim_keep_last_turns", "TRIM_KEEP_LAST_TURNS", 2)
    if tool_output_tokens < 0:
        raise APIError(400, "`trim_tool_output_tokens` cannot be negative.", "trim_tool_output_tokens")
    if keep_last_turns < 0:
        raise APIError(400, "`trim_keep_last_turns` cannot be negative.", "trim_keep_last_turns")
    return tool_output_tokens, keep_last_turns


def render_chat(messages, enable_thinking=False, reasoning_effort=None, tools=None):
    """Render the text-only subset of the official GLM-5.2 chat template."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    prompt = ["[gMASK]<sop>"]
    if enable_thinking:
        effort = "High" if reasoning_effort == "high" else "Max"
        prompt.append(f"<|system|>Reasoning Effort: {effort}")
    if tools:
        # AUTHORITATIVE GLM-5.2 tool-declaration block (byte-matches chat_template.jinja): the
        # `# Tools` + <tools></tools> XML structure is what the model was trained on. A made-up
        # preamble makes it hallucinate other frameworks' syntax (e.g. `end_action`).
        prompt.append("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the "
                      "user query.\n\nYou are provided with function signatures within <tools></tools> "
                      "XML tags:\n<tools>\n")
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            clean = {k: v for k, v in fn.items() if k not in ("defer_loading", "strict")}
            prompt.append(json.dumps(clean, ensure_ascii=False) + "\n")
        prompt.append("</tools>\n\nFor each function call, output the function name and arguments "
                      "within the following XML format:\n<tool_call>{function-name}"
                      "<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
                      "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")
    prev_tool = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role in ("system", "developer"):
            prompt.append(f"<|system|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "user":
            prompt.append(f"<|user|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "assistant":
            # content may be null when the message is purely tool_calls
            raw = message.get("content")
            text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            prompt.append(f"<|assistant|><think></think>{text.strip()}")
            for tc in (message.get("tool_calls") or []):
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                prompt.append(BOX_START + (fn.get("name") or ""))
                for key, value in (args or {}).items():
                    prompt.append(f"<arg_key>{key}</arg_key><arg_value>"
                                  + (value if isinstance(value, str)
                                     else json.dumps(value, ensure_ascii=False)) + "</arg_value>")
                prompt.append(BOX_END)
        elif role == "tool":
            if not prev_tool:                       # one <|observation|> per consecutive tool run
                prompt.append("<|observation|>")
            prompt.append(TR_OPEN + content_text(message.get("content"), f"messages.{index}.content") + TR_CLOSE)
        else:
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
        prev_tool = (role == "tool")
    prompt.append("<|assistant|><think>" if enable_thinking else
                  "<|assistant|><think></think>")
    return "".join(prompt)


def generation_options(body, limit):
    if body.get("n", 1) != 1:
        raise APIError(400, "Iliria currently supports `n=1` only.", "n", "unsupported_value")
    # `tools`/`functions` are handled by render_chat (declaration) + parse_tool_calls (output).
    if body.get("stop") is not None:
        raise APIError(400, "Custom stop sequences are not supported yet.", "stop", "unsupported_parameter")
    if body.get("logprobs"):
        raise APIError(400, "Log probabilities are not supported yet.", "logprobs", "unsupported_parameter")
    if body.get("frequency_penalty", 0) or body.get("presence_penalty", 0):
        raise APIError(400, "Token penalties are not supported yet.", None, "unsupported_parameter")
    if body.get("seed") is not None:
        raise APIError(400, "Per-request seeds are not supported yet.", "seed", "unsupported_parameter")
    response_format = body.get("response_format")
    if response_format not in (None, {"type": "text"}):
        raise APIError(400, "Only the default text response format is supported.",
                       "response_format", "unsupported_parameter")

    maximum = body.get("max_completion_tokens")
    maximum_param = "max_completion_tokens"
    if maximum is None:
        maximum = body.get("max_tokens")
        maximum_param = "max_tokens"
    if maximum is None:
        maximum = min(256, limit)
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    temperature = 0.7 if temperature is None else temperature
    top_p = 0.9 if top_p is None else top_p
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= limit:
        raise APIError(400, f"`{maximum_param}` must be an integer between 1 and {limit}.", maximum_param)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise APIError(400, "`temperature` must be between 0 and 2.", "temperature")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise APIError(400, "`top_p` must be greater than 0 and at most 1.", "top_p")
    return maximum, float(temperature), float(top_p)


def read_engine_turn(stream, sentinel, on_bytes):
    pending = b""
    while True:
        byte = stream.read(1)
        if byte == b"":
            raise RuntimeError("iliria engine exited unexpectedly")
        pending += byte
        if pending.endswith(sentinel):
            data = pending[:-len(sentinel)]
            if data:
                on_bytes(data)
            break
        if len(pending) > len(sentinel):
            on_bytes(pending[:-len(sentinel)])
            pending = pending[-len(sentinel):]

    fields = stream.readline().decode("utf-8", "replace").strip().split()
    if len(fields) < 5 or fields[0] != "STAT":
        raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
    return {
        "completion_tokens": int(fields[1]),
        "tokens_per_second": float(fields[2]),
        "cache_hit_percent": float(fields[3]),
        "rss_gb": float(fields[4]),
        "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
        "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
    }


class Engine:
    def __init__(self, executable, model, cap=8, max_tokens=1024, env=None, kv_slots=1):
        child_env = dict(env or os.environ, SNAP=str(model), SERVE="1", NGEN=str(max_tokens),
                         KV_SLOTS=str(kv_slots))
        self.process = subprocess.Popen(
            [str(executable), str(cap)], env=child_env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, bufsize=0,
        )
        self.lock = threading.Lock()
        self.kv_slots = kv_slots
        read_engine_turn(self.process.stdout, READY, lambda _: None)

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0):
        if isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or not 0 <= cache_slot < self.kv_slots:
            raise APIError(400, "Invalid cache slot.", "cache_slot")
        payload = prompt.encode("utf-8")
        if b"\0" in payload:
            raise APIError(400, "NUL bytes are not supported in prompts.", "messages")
        if len(payload) > PROMPT_BYTES_MAX:
            raise APIError(400, "Prompt exceeds the engine's 16 MiB frame limit.",
                           "messages", "context_length_exceeded")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def decode(data):
            text = decoder.decode(data)
            if text:
                on_text(text)

        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError("iliria engine is not running")
            header = (f"\x02PROMPT {len(payload)} {max_tokens} {temperature:.8g} "
                      f"{top_p:.8g} {cache_slot}\n").encode()
            # bufsize=0 makes stdin a raw FileIO: a single write() may report a short
            # count (e.g. when a signal lands while blocked on a >64 KiB payload that
            # exceeds the pipe buffer). Anything unwritten would leave the engine
            # waiting forever for the rest of the frame, so loop until it is all out.
            view = memoryview(header + payload + b"\n")
            while view:
                written = self.process.stdin.write(view)
                if written is None:     # buffered stream: everything was queued
                    break
                view = view[written:]
            self.process.stdin.flush()
            stats = read_engine_turn(self.process.stdout, END, decode)
            tail = decoder.decode(b"", final=True)
            if tail:
                on_text(tail)
            return stats

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def model_object(model_id, created):
    return {"id": model_id, "object": "model", "created": created, "owned_by": "iliria"}


class APIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine, model_id, api_key=None, max_tokens=1024,
                 cors_origins=DEFAULT_CORS_ORIGINS, max_queue=8, queue_timeout=300,
                 kv_slots=1):
        super().__init__(address, APIHandler)
        self.engine = engine
        self.model_id = model_id
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.scheduler = GenerationScheduler(max_queue, queue_timeout)
        self.kv_slots = kv_slots
        self.prefix_slots = PrefixSlotRouter(kv_slots)
        self.trim_cache = _TrimCache()
        self.cors_origins = tuple(cors_origins)
        self.created = int(time.time())


class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "iliria"
    # Slowloris guard: with no timeout, a connection that never finishes sending its
    # request line/headers/body holds a thread (and a file descriptor) open forever --
    # ThreadingHTTPServer has no cap on concurrent threads. socketserver.StreamRequestHandler
    # .setup() applies this via socket.settimeout(), and BaseHTTPRequestHandler
    # .handle_one_request() already catches the resulting TimeoutError and simply closes
    # that one connection (see the serve-security review). 60s is far above anything a
    # legitimate request needs and well above the SSE keepalive pump's 10s cadence (KA_GAP
    # below), so an in-progress generation is never affected -- only a stalled connection
    # making zero progress is dropped.
    timeout = 60

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] %s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, status, body, request_id=None, headers=None):
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if request_id:
            self.send_header("x-request-id", request_id)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def send_cors_headers(self):
        origin = self.headers.get("Origin")
        if not origin or ("*" not in self.server.cors_origins and origin not in self.server.cors_origins):
            return
        self.send_header("Access-Control-Allow-Origin", "*" if "*" in self.server.cors_origins else origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Expose-Headers",
                         "x-request-id, x-iliria-queue-wait-ms, Retry-After")
        self.send_header("Access-Control-Max-Age", "600")
        if "*" not in self.server.cors_origins:
            self.send_header("Vary", "Origin")

    def check_origin(self):
        """Reject cross-origin browser requests outside the CORS allowlist before doing
        any other work (localhost-CSRF guard, the serve-security review finding CSRF-1).
        send_cors_headers() alone only controls whether browser JS may *read* the
        response -- a "simple" cross-origin request (e.g. Content-Type: text/plain,
        which is CORS-safelisted, or a plain HTML form submission) skips the preflight
        entirely and the browser sends it regardless. Without this check, any webpage the
        user's browser has open (or a DNS-rebinding attacker) could silently drive the
        model. Non-browser clients -- curl, the OpenAI SDK, this repo's own scripts --
        never send an Origin header and are unaffected. `--cors-origin '*'` remains the
        explicit, documented opt-out for anyone who really wants to allow every origin."""
        origin = self.headers.get("Origin")
        if origin is None or "*" in self.server.cors_origins or origin in self.server.cors_origins:
            return
        raise APIError(403, "Cross-origin requests are not permitted from this origin.",
                       None, "origin_not_allowed")

    def require_auth(self):
        if self.server.api_key and self.headers.get("Authorization") != f"Bearer {self.server.api_key}":
            raise APIError(401, "Invalid or missing API key.", None, "invalid_api_key",
                           "authentication_error")

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError(400, "Invalid Content-Length header.")
        if length < 1 or length > MAX_BODY:
            raise APIError(400, f"Request body must be between 1 and {MAX_BODY} bytes.")
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise APIError(400, "Request body must be valid JSON.")
        if not isinstance(body, dict):
            raise APIError(400, "Request body must be a JSON object.")
        return body

    def check_model(self, body):
        model = body.get("model")
        if model != self.server.model_id:
            raise APIError(404, f"The model `{model}` does not exist.", "model", "model_not_found")

    def do_GET(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            self.check_origin()
            path = urlsplit(self.path).path
            if path == "/health":
                self.send_json(200, {"status": "ok", "scheduler": self.server.scheduler.snapshot(),
                                     "kv_slots": self.server.kv_slots}, request_id)
                return
            self.require_auth()
            if path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [model_object(
                    self.server.model_id, self.server.created)]}, request_id)
            elif path.startswith("/v1/models/") and unquote(path[11:]) == self.server.model_id:
                self.send_json(200, model_object(self.server.model_id, self.server.created), request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_json(error.status, error_object(error), request_id, error.headers)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            self.check_origin()
            self.require_auth()
            body = self.read_json()
            self.check_model(body)
            path = urlsplit(self.path).path
            if path == "/v1/chat/completions":
                self.chat_completion(body, request_id)
            elif path == "/v1/completions":
                self.completion(body, request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_json(error.status, error_object(error), request_id, error.headers)
        except ClientCancelled:
            pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.log_error("request failed: %s", error)
            api_error = APIError(500, "The iliria engine failed to process the request.",
                                 None, "engine_error", "server_error")
            try:
                self.send_json(500, error_object(api_error), request_id)
            except OSError:
                pass

    def generation(self, body, prompt, request_id, chat):
        maximum, temperature, top_p = generation_options(body, self.server.max_tokens)
        tools = (body.get("tools") or body.get("functions") or None) if chat else None
        if "cache_slot" in body:
            cache_slot = body["cache_slot"]
            if (isinstance(cache_slot, bool) or not isinstance(cache_slot, int)
                    or not 0 <= cache_slot < self.server.kv_slots):
                raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                               "cache_slot")
        else:
            cache_slot = None
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        stream_options = body.get("stream_options") if stream else None
        if stream and stream_options is not None and not isinstance(stream_options, dict):
            raise APIError(400, "`stream_options` must be an object.", "stream_options")
        include_usage = bool((stream_options or {}).get("include_usage"))
        object_name = "chat.completion" if chat else "text_completion"
        id_prefix = "chatcmpl-" if chat else "cmpl-"
        completion_id = id_prefix + uuid.uuid4().hex
        created = int(time.time())

        with self.server.scheduler.admit(self.client_disconnected) as queue_wait:
            queue_headers = {"x-iliria-queue-wait-ms": str(round(queue_wait * 1000))}
            if cache_slot is None:
                cache_slot = self.server.prefix_slots.select(prompt)

            def run_engine(on_text):
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, on_text, cache_slot)
                self.server.prefix_slots.record(cache_slot, prompt)
                return stats

            if not stream:
                output = []
                stats = run_engine(output.append)
                text = "".join(output)
                length_finish = "length" if stats["length_limited"] else "stop"
                if chat and tools:
                    content, calls = parse_tool_calls(text, tools)
                    message = {"role": "assistant", "content": content or None, "refusal": None}
                    if calls:
                        message["tool_calls"] = calls
                    finish = "tool_calls" if calls else length_finish
                    choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                else:
                    choice = ({"index": 0, "message": {"role": "assistant", "content": text,
                               "refusal": None}, "logprobs": None, "finish_reason": length_finish} if chat else
                              {"index": 0, "text": text, "logprobs": None, "finish_reason": length_finish})
                self.send_json(200, {"id": completion_id, "object": object_name, "created": created,
                    "model": self.server.model_id, "choices": [choice], "usage": self.usage(stats)},
                    request_id, queue_headers)
                return

            stream_object = "chat.completion.chunk" if chat else object_name
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("x-request-id", request_id)
            for name, value in queue_headers.items(): self.send_header(name, value)
            self.send_cors_headers()
            self.end_headers()
            connected = True
            # KEEPALIVE: engine.generate() blocks SILENTLY during the (minutes-long) cold
            # prefill, and the client drops the socket after its idle timeout. A background pump
            # emits a reasoning_content "." delta (the channel that reliably resets the client's
            # timer and lands in the thinking panel, so answer content stays clean) whenever no
            # event has been written for KA_GAP seconds. All wfile writes share ka_lock so the
            # pump and event() never interleave; last_write gates the pump so it stays quiet
            # while real tokens are flowing (e.g. during decode).
            ka_lock = threading.Lock()
            last_write = [time.time()]
            ka_stop = threading.Event()
            KA_GAP = 10.0
            dbg_echo = _env("DEBUG", "0") == "1"   # tee decoded tokens to stderr

            def event(choices, usage_marker=False):
                nonlocal connected
                if not connected:
                    return
                event_body = {"id": completion_id, "object": stream_object, "created": created,
                              "model": self.server.model_id, "choices": choices}
                if include_usage:
                    event_body["usage"] = None if not usage_marker else usage_marker
                data = json.dumps(event_body, ensure_ascii=False, separators=(",", ":"))
                with ka_lock:
                    try:
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected = False

            def _keepalive():
                ping = [{"index": 0, "delta": ({"reasoning_content": "."} if chat else {"content": ""}),
                         "logprobs": None, "finish_reason": None}]
                while not ka_stop.wait(1.0):
                    if not connected:
                        return
                    if time.time() - last_write[0] >= KA_GAP:
                        event(ping)

            if chat:
                event([{"index": 0, "delta": {"role": "assistant", "content": ""},
                        "logprobs": None, "finish_reason": None}])

            def emit(text):
                choice = ({"index": 0, "delta": {"content": text}, "logprobs": None,
                           "finish_reason": None} if chat else
                          {"index": 0, "text": text, "logprobs": None, "finish_reason": None})
                event([choice])

            ka_thread = threading.Thread(target=_keepalive, daemon=True)
            ka_thread.start()
            if chat and tools:
                # Suppress tool-call markers from the streamed content and parse the authoritative
                # calls from the FULL reply after generation. Hold back a marker-length tail so a
                # <tool_call> split across engine chunks is still caught.
                sp = {"buf": "", "tool": False}
                hold = len(BOX_START) - 1
                raw = []
                def emit_tools(chunk):
                    raw.append(chunk)
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    if sp["tool"]:
                        return
                    sp["buf"] += chunk
                    cut = sp["buf"].find(BOX_START)
                    if cut >= 0:
                        if cut:
                            emit(sp["buf"][:cut])
                        sp["buf"] = ""
                        sp["tool"] = True
                        return
                    flush = max(0, len(sp["buf"]) - hold)
                    if flush:
                        emit(sp["buf"][:flush])
                        sp["buf"] = sp["buf"][flush:]
                stats = run_engine(emit_tools)
                if not sp["tool"] and sp["buf"]:
                    emit(sp["buf"])                     # no tool call happened: flush held tail
                _content, calls = parse_tool_calls("".join(raw), tools)
                for i, tc in enumerate(calls):
                    event([{"index": 0, "delta": {"tool_calls": [{"index": i, "id": tc["id"],
                             "type": "function", "function": {"name": tc["function"]["name"],
                             "arguments": tc["function"]["arguments"]}}]},
                            "logprobs": None, "finish_reason": None}])
                finish = "tool_calls" if calls else ("length" if stats["length_limited"] else "stop")
            else:
                def emit_plain(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    emit(chunk)
                stats = run_engine(emit_plain)
                finish = "length" if stats["length_limited"] else "stop"
            ka_stop.set()                          # generation done: stop the keepalive pump
            ka_thread.join(timeout=2)
            final_choice = ({"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish}
                            if chat else {"index": 0, "text": "", "logprobs": None,
                                          "finish_reason": finish})
            event([final_choice])
            if include_usage:
                event([], self.usage(stats))
            if connected:
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
            self.close_connection = True

    def client_disconnected(self):
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
            return self.connection.recv(1, flags) == b""
        except (OSError, ValueError):
            return True

    @staticmethod
    def usage(stats):
        prompt = stats["prompt_tokens"]
        completion = stats["completion_tokens"]
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion}

    def chat_completion(self, body, request_id):
        reasoning_effort = body.get("reasoning_effort")
        efforts = (None, "none", "minimal", "low", "medium", "high", "xhigh")
        if reasoning_effort not in efforts:
            raise APIError(400, "`reasoning_effort` must be none, minimal, low, medium, high, or xhigh.",
                           "reasoning_effort")
        # ILI_THINK=1 makes thinking the default when the client sends NEITHER reasoning_effort
        # nor enable_thinking (a global switch, like the old server's --think). An explicit
        # client value always wins. Default off => exact OpenAI-standard behavior.
        if (reasoning_effort is None and "enable_thinking" not in body
                and _env("THINK", "0") == "1"):
            reasoning_effort = "high"
        enable_thinking = body.get("enable_thinking", reasoning_effort not in (None, "none"))
        if not isinstance(enable_thinking, bool):
            raise APIError(400, "`enable_thinking` must be a boolean.", "enable_thinking")
        tools = body.get("tools") or body.get("functions") or None
        tool_output_tokens, keep_last_turns = _trim_options(body)
        messages = body.get("messages")
        if tool_output_tokens > 0 and isinstance(messages, list):
            messages = trim_messages(messages, keep_last_turns, tool_output_tokens, self.server.trim_cache)
        prompt = render_chat(messages, enable_thinking, reasoning_effort, tools)
        self.generation(body, prompt, request_id, True)

    def completion(self, body, request_id):
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise APIError(400, "Iliria currently requires `prompt` to be a string.", "prompt")
        self.generation(body, prompt, request_id, False)


def serve(model, host="127.0.0.1", port=8000, model_id="glm-5.2-iliria", api_key=None,
          cap=8, max_tokens=1024, engine=HERE / "glm", env=None, cors_origins=None,
          max_queue=8, queue_timeout=300, kv_slots=1):
    if not 1 <= max_tokens:
        raise ValueError("max_tokens must be positive")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if max_queue < 0:
        raise ValueError("max_queue cannot be negative")
    if queue_timeout <= 0:
        raise ValueError("queue_timeout must be positive")
    if not 1 <= kv_slots <= 16:
        raise ValueError("kv_slots must be between 1 and 16")
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        print("WARNING: API is listening beyond localhost without ILI_API_KEY", file=sys.stderr)
    runtime = Engine(engine,model,cap,max_tokens,env,kv_slots)
    origins = DEFAULT_CORS_ORIGINS if cors_origins is None else tuple(cors_origins)
    server = APIServer((host, port), runtime, model_id, api_key, max_tokens, origins,
                       max_queue, queue_timeout, kv_slots)
    print(f"OpenAI-compatible API listening on http://{host}:{port}/v1", file=sys.stderr)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.scheduler.close()
        server.server_close()
        runtime.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_env("MODEL"), required=not _env("MODEL"))
    parser.add_argument("--engine", default=str(HERE / "glm"))
    # ILI_SERVE_BIND overrides the loopback-only default; an explicit --host always wins.
    # Bind beyond loopback only if you understand the exposure (the serve-security review):
    # any other process on the LAN/wifi could then reach the API, so pair it with
    # --api-key and a real firewall rule -- never leave it open on a shared network.
    parser.add_argument("--host", default=_env("SERVE_BIND", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=_env("MODEL_ID", "glm-5.2-iliria"))
    parser.add_argument("--api-key", default=_env("API_KEY"))
    parser.add_argument("--cors-origin", action="append", default=None,
                        help="allowed browser origin; repeat as needed (use '*' for any origin)")
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-queue", type=int, default=int(_env("MAX_QUEUE", "8")))
    parser.add_argument("--queue-timeout", type=float,
                        default=float(_env("QUEUE_TIMEOUT", "300")))
    parser.add_argument("--kv-slots", type=int, default=int(_env("KV_SLOTS", "1")))
    return parser


def main():
    args = build_arg_parser().parse_args()
    serve(args.model, args.host, args.port, args.model_id, args.api_key,
          args.cap,args.max_tokens,args.engine,cors_origins=args.cors_origin,
          max_queue=args.max_queue,queue_timeout=args.queue_timeout,kv_slots=args.kv_slots)


if __name__ == "__main__":
    main()
