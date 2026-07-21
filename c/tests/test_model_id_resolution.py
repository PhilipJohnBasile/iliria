"""Regression coverage for the 2026-07-14 model-id rename incident: today's naming scrub
made openai_server.py's serve() default model_id to "glm-5.2-iliria", but every HTTP
client script still hardcoded/defaulted the stale "glm-5.2" -- which check_model() in
openai_server.py 404s outright ("The model `glm-5.2` does not exist."). That is exactly
what broke the layer-capture-grid run and would have blocked the RoPE-capture, F1, ABBA,
and session runs behind it.

The fix: every affected client (scripts/long_ctx_profile.py, run_layer_capture_grid.py,
serve_gate.py, session_acceptance.py, abba_transcript_driver.py) now defaults --model-id
to None and, unless the caller explicitly overrides it, calls resolve_model_id() -- a GET
/v1/models query against the already-running server, using its FIRST advertised id -- so
a future rename is a non-event instead of a repeat blocker. Two independent
resolve_model_id() implementations exist on purpose (long_ctx_profile.py and
serve_gate.py each stay dependency-free, mirroring their existing sse_request()/
_sse_measure() convention of parallel-not-shared HTTP helpers); run_layer_capture_grid.py,
session_acceptance.py, and abba_transcript_driver.py each reuse one of the two via their
existing sibling-module imports.

This suite exercises both resolve_model_id() copies directly, then drives full CLI runs
of every affected script end to end against a purpose-built mock server that -- like the
real server's check_model() -- 404s any request whose `model` field does not match what
it advertises. That enforcement is what makes "the client used the id the server
advertised, not a hardcoded stale one" a checkable property instead of a hoped-for one: a
mock that accepted any model string could not tell a correct resolution from a lucky
coincidence, which is exactly how the original bug went unnoticed by every --mock/--dry-run
smoke test already in this suite (none of them validate the `model` field).

No real engine (`ili serve`/`glm`) is ever invoked by this suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = C_DIR / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
import long_ctx_profile   # noqa: E402  (path insert must come first)
import serve_gate         # noqa: E402

RUN_LAYER_CAPTURE_GRID = SCRIPTS_DIR / "run_layer_capture_grid.py"
SERVE_GATE = SCRIPTS_DIR / "serve_gate.py"
ABBA_DRIVER = SCRIPTS_DIR / "abba_transcript_driver.py"
LONG_CTX_PROFILE = SCRIPTS_DIR / "long_ctx_profile.py"
SESSION_ACCEPTANCE = SCRIPTS_DIR / "session_acceptance.py"

# Deliberately NOT "glm-5.2" (the old stale default) and NOT "glm-5.2-iliria" (today's
# corrected default) -- a distinctive marker so a passing test proves the client actually
# read this id from the server, rather than coincidentally matching a hardcoded fallback.
ADVERTISED_MODEL_ID = "glm-9.9-server-advertised-test-marker"


class _EnforcingServer(ThreadingHTTPServer):
    daemon_threads = True   # match openai_server.py's own APIServer convention


class _EnforcingHandler(BaseHTTPRequestHandler):
    """Minimal stand-in for openai_server.py: advertises ONE model id via GET /v1/models
    and rejects (HTTP 404, same message shape as check_model()) any /v1/chat/completions
    request whose `model` field does not match it exactly."""
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass   # keep test output quiet

    def do_GET(self):
        if self.path == "/v1/models":
            data = [] if self.server.empty_models else [
                {"id": self.server.model_id, "object": "model", "created": 0}]
            self._send_json(200, {"object": "list", "data": data})
            return
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": {"message": "Not found."}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        model = body.get("model")
        with self.server.lock:
            self.server.received_models.append(model)
        if model != self.server.model_id:
            # Same shape/status as openai_server.py's check_model().
            self._send_json(404, {"error": {
                "message": f"The model `{model}` does not exist.",
                "type": "model_not_found", "param": "model", "code": None}})
            return
        if body.get("stream"):
            self._send_sse(body)
        else:
            self._send_json(200, {
                "id": "mockcmpl", "object": "chat.completion", "created": 0, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}})

    def _send_json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, body):
        max_tokens = max(1, min(body.get("max_tokens") or 1, 3))   # keep the test fast
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for _ in range(max_tokens):
            chunk = {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        usage_chunk = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": max_tokens,
                       "total_tokens": 10 + max_tokens}}
        self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def start_enforcing_mock(model_id=ADVERTISED_MODEL_ID, empty_models=False):
    """OS-assigned free port -- this mock is only ever reached from within this test
    process or subprocesses it spawns itself, so there is no reason to risk colliding
    with another test's fixed --mock-port (several already-existing test files in this
    suite use fixed ports for their own scripts' --mock modes)."""
    server = _EnforcingServer(("127.0.0.1", 0), _EnforcingHandler)
    server.model_id = model_id
    server.empty_models = empty_models
    server.received_models = []
    server.lock = threading.Lock()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


class ResolveModelIdDirectTest(unittest.TestCase):
    """Exercises both independent resolve_model_id() copies directly (long_ctx_profile.py
    and serve_gate.py never share this code -- each stays dependency-free on purpose)."""

    def setUp(self):
        self.server, self.thread = start_enforcing_mock()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.port = self.server.server_address[1]

    def test_long_ctx_profile_resolves_the_advertised_id(self):
        self.assertEqual(long_ctx_profile.resolve_model_id("127.0.0.1", self.port),
                         ADVERTISED_MODEL_ID)

    def test_serve_gate_resolves_the_advertised_id(self):
        self.assertEqual(serve_gate.resolve_model_id("127.0.0.1", self.port),
                         ADVERTISED_MODEL_ID)

    def test_long_ctx_profile_falls_back_when_server_unreachable(self):
        # Nothing listens on port 1 (connect refused almost immediately on loopback).
        self.assertEqual(
            long_ctx_profile.resolve_model_id("127.0.0.1", 1, timeout=0.5),
            long_ctx_profile.DEFAULT_MODEL_ID)

    def test_serve_gate_falls_back_when_server_unreachable(self):
        self.assertEqual(
            serve_gate.resolve_model_id("127.0.0.1", 1, timeout=0.5),
            serve_gate.DEFAULT_MODEL_ID)

    def test_falls_back_on_empty_models_list(self):
        server, _ = start_enforcing_mock(empty_models=True)
        try:
            port = server.server_address[1]
            self.assertEqual(
                long_ctx_profile.resolve_model_id("127.0.0.1", port, default="fallback-value"),
                "fallback-value")
            self.assertEqual(
                serve_gate.resolve_model_id("127.0.0.1", port, default="fallback-value"),
                "fallback-value")
        finally:
            server.shutdown()
            server.server_close()

    def test_explicit_default_overrides_the_fallback_value(self):
        self.assertEqual(
            long_ctx_profile.resolve_model_id("127.0.0.1", 1, timeout=0.5, default="custom"),
            "custom")


class RunLayerCaptureGridEndToEndTest(unittest.TestCase):
    """scripts/run_layer_capture_grid.py -- the exact script whose stale `glm-5.2` default
    404'd against the renamed server tonight. Live ("drive") mode, never --dry-run."""

    def setUp(self):
        self.server, self.thread = start_enforcing_mock()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.port = self.server.server_address[1]

    def test_resolves_and_uses_the_servers_advertised_id_with_no_model_id_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "captures"
            result = subprocess.run(
                [sys.executable, str(RUN_LAYER_CAPTURE_GRID),
                 "--capture-dir", str(capture_dir), "--metal-prefill", "0",
                 "--host", "127.0.0.1", "--port", str(self.port),
                 "--s-values", "4", "--t-values", "128"],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((capture_dir / "grid-manifest.json").exists())
            manifest = json.loads((capture_dir / "grid-manifest.json").read_text())
            self.assertEqual(len(manifest["cells"]), 1)   # 1 T value x 1 S value

        # Both the "load" and "measure" HTTP calls used the id GET /v1/models advertised --
        # not "glm-5.2" (the old stale default) and not even "glm-5.2-iliria" (today's
        # corrected default), proving this came from the query, not a hardcoded string.
        self.assertEqual(len(self.server.received_models), 2)
        self.assertTrue(all(m == ADVERTISED_MODEL_ID for m in self.server.received_models))

    def test_explicit_model_id_flag_is_respected_verbatim_even_when_the_server_rejects_it(self):
        """An explicit --model-id must never be silently replaced by the resolved id --
        the user's override always wins, even if that means a real 404 (the same failure
        mode this whole fix exists to prevent when the id is stale)."""
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "captures"
            result = subprocess.run(
                [sys.executable, str(RUN_LAYER_CAPTURE_GRID),
                 "--capture-dir", str(capture_dir), "--metal-prefill", "0",
                 "--host", "127.0.0.1", "--port", str(self.port),
                 "--s-values", "4", "--t-values", "128",
                 "--model-id", "explicitly-wrong-id"],
                capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("404", result.stdout + result.stderr)
        self.assertIn("explicitly-wrong-id", self.server.received_models)


class ServeGateEndToEndTest(unittest.TestCase):
    """scripts/serve_gate.py -- live-only (no --mock), gets its own resolve_model_id copy."""

    def setUp(self):
        self.server, self.thread = start_enforcing_mock()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.port = self.server.server_address[1]

    def test_resolves_and_uses_the_servers_advertised_id_with_no_model_id_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reuse.json"
            result = subprocess.run(
                [sys.executable, str(SERVE_GATE), "--mode", "reuse",
                 "--host", "127.0.0.1", "--port", str(self.port),
                 "--turns", "1", "--max-tokens", "5", "--chars-per-turn", "200",
                 "--out", str(out)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(out.exists())
        self.assertEqual(self.server.received_models, [ADVERTISED_MODEL_ID])


class AbbaTranscriptDriverEndToEndTest(unittest.TestCase):
    """scripts/abba_transcript_driver.py `drive` -- reuses serve_gate.resolve_model_id via
    its existing `import serve_gate`, exercising the reuse path (not a re-copy)."""

    def setUp(self):
        self.server, self.thread = start_enforcing_mock()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.port = self.server.server_address[1]

    def test_resolves_and_uses_the_servers_advertised_id_with_no_model_id_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "arm.json"
            result = subprocess.run(
                [sys.executable, str(ABBA_DRIVER), "drive",
                 "--host", "127.0.0.1", "--port", str(self.port),
                 "--turns", "1", "--max-tokens", "5", "--chars-per-turn", "200",
                 "--out", str(out)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(out.exists())
        self.assertEqual(self.server.received_models, [ADVERTISED_MODEL_ID])


class LongCtxProfileMockWiringTest(unittest.TestCase):
    """scripts/long_ctx_profile.py has its own in-process --mock server (no external mock
    needed); --mock-model-id lets this test pin it to a distinctive id so a passing run
    proves the CLI's own resolve_model_id() wiring, not a lucky default-vs-default match."""

    def test_mock_run_succeeds_when_resolving_the_mocks_advertised_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(LONG_CTX_PROFILE), "--mock", "--mock-port", "8917",
                 "--mock-model-id", ADVERTISED_MODEL_ID,
                 "--targets", "2000", "--decode-tokens", "5",
                 "--out-csv", str(Path(tmp) / "out.csv"), "--out-md", str(Path(tmp) / "out.md")],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(repr(ADVERTISED_MODEL_ID), result.stderr)


class SessionAcceptanceMockWiringTest(unittest.TestCase):
    """scripts/session_acceptance.py: same --mock-model-id wiring, reused via its existing
    `import long_ctx_profile` (start_mock_server + resolve_model_id), not a re-copy."""

    def test_mock_run_succeeds_when_resolving_the_mocks_advertised_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(SESSION_ACCEPTANCE), "--mock", "--mock-port", "8918",
                 "--mock-model-id", ADVERTISED_MODEL_ID,
                 "--out-csv", str(Path(tmp) / "out.csv"), "--out-md", str(Path(tmp) / "out.md")],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(repr(ADVERTISED_MODEL_ID), result.stderr)


if __name__ == "__main__":
    unittest.main()
