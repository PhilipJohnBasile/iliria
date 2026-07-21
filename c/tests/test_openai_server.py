# Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE.
import io
import json
import os
import socket
import threading
import time
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai_server import (APIError, APIHandler, APIServer, ClientCancelled, END,
                           GenerationScheduler, PrefixSlotRouter, _TrimCache, _trim_cutoff,
                           _trim_excerpt, _trim_options, build_arg_parser, generation_options,
                           read_engine_turn, render_chat, serve, trim_messages)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0):
        self.calls.append((prompt, maximum, temperature, top_p, cache_slot))
        on_text("Hé")
        on_text("llo")
        return {"prompt_tokens": 7, "completion_tokens": 2, "length_limited": False}


class BlockingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0):
        self.entered.set()
        self.release.wait(2)
        return super().generate(prompt, maximum, temperature, top_p, on_text, cache_slot)


class TemplateTest(unittest.TestCase):
    def test_renders_text_subset_of_official_template(self):
        prompt = render_chat([
            {"role": "system", "content": "System"},
            {"role": "developer", "content": "Developer"},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": " Hello "},
            {"role": "user", "content": "Again"},
        ])
        self.assertEqual(
            prompt,
            "[gMASK]<sop><|system|>System<|system|>Developer<|user|>Hi"
            "<|assistant|><think></think>Hello<|user|>Again"
            "<|assistant|><think></think>",
        )

    def test_rejects_non_text_content(self):
        with self.assertRaisesRegex(APIError, "text message content only"):
            render_chat([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "x"}}
            ]}])

    def test_renders_thinking_prefix(self):
        self.assertEqual(
            render_chat([{"role": "user", "content": "Hi"}], True, "high"),
            "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>Hi<|assistant|><think>",
        )

    def test_validates_generation_limits(self):
        self.assertEqual(generation_options({"max_tokens": 4, "temperature": 0, "top_p": 1}, 8),
                         (4, 0.0, 1.0))
        with self.assertRaises(APIError):
            generation_options({"max_tokens": 9}, 8)
        self.assertEqual(generation_options({"temperature": None, "top_p": None}, 8),
                         (8, 0.7, 0.9))


class ProtocolTest(unittest.TestCase):
    def test_reads_payload_and_extended_status(self):
        stream = io.BytesIO(b"hello" + END + b"STAT 2 3.5 44 1.2 7 1\n")
        chunks = []
        stats = read_engine_turn(stream, END, chunks.append)
        self.assertEqual(b"".join(chunks), b"hello")
        self.assertEqual(stats["prompt_tokens"], 7)
        self.assertTrue(stats["length_limited"])

    def test_rejects_invalid_kv_pool_before_engine_start(self):
        with self.assertRaisesRegex(ValueError, "kv_slots"):
            serve("/missing", kv_slots=0)


class SchedulerTest(unittest.TestCase):
    def test_rejects_when_waiting_queue_is_full(self):
        scheduler = GenerationScheduler(max_queue=0, queue_timeout=1)
        with scheduler.admit():
            with self.assertRaises(APIError) as caught:
                with scheduler.admit():
                    pass
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(caught.exception.code, "queue_full")
        self.assertEqual(scheduler.snapshot()["rejected"], 1)

    def test_times_out_and_cancels_queued_requests(self):
        scheduler = GenerationScheduler(max_queue=2, queue_timeout=0.02)
        with scheduler.admit():
            with self.assertRaises(APIError) as timed_out:
                with scheduler.admit():
                    pass
            with self.assertRaises(ClientCancelled):
                with scheduler.admit(lambda: True):
                    pass
        stats = scheduler.snapshot()
        self.assertEqual(timed_out.exception.code, "queue_timeout")
        self.assertEqual(stats["timed_out"], 1)
        self.assertEqual(stats["cancelled"], 1)

    def test_admits_waiters_in_fifo_order(self):
        scheduler = GenerationScheduler(max_queue=2, queue_timeout=1)
        entered = threading.Event()
        release = threading.Event()
        order = []

        def run(name, block=False):
            with scheduler.admit():
                order.append(name)
                if block:
                    entered.set()
                    release.wait(1)

        first = threading.Thread(target=run, args=("first", True))
        second = threading.Thread(target=run, args=("second",))
        third = threading.Thread(target=run, args=("third",))
        first.start(); entered.wait(1)
        second.start()
        for _ in range(100):
            if scheduler.snapshot()["queued"] == 1: break
            threading.Event().wait(0.005)
        third.start()
        for _ in range(100):
            if scheduler.snapshot()["queued"] == 2: break
            threading.Event().wait(0.005)
        release.set()
        first.join(1); second.join(1); third.join(1)
        self.assertEqual(order, ["first", "second", "third"])
        self.assertEqual(scheduler.snapshot()["completed"], 3)

    def test_close_rejects_waiters(self):
        scheduler = GenerationScheduler(max_queue=1, queue_timeout=1)
        entered = threading.Event()
        release = threading.Event()
        errors = []

        def active():
            with scheduler.admit():
                entered.set(); release.wait(1)

        def waiting():
            try:
                with scheduler.admit(): pass
            except APIError as error:
                errors.append(error.code)

        first = threading.Thread(target=active); first.start(); entered.wait(1)
        second = threading.Thread(target=waiting); second.start()
        scheduler.close(); release.set(); first.join(1); second.join(1)
        self.assertEqual(errors, ["scheduler_closed"])


class ContextDietTest(unittest.TestCase):
    """Item A (context-diet): opt-in server-side trimming of old tool-role messages.

    Covers the three properties the feature must hold: (1) it saves real prompt bytes when
    enabled, (2) once a message is trimmed it stays byte-identical forever after -- the
    property delta-KV-reuse depends on -- and (3) it is a strict no-op when disabled."""

    @staticmethod
    def transcript(turns, tool_payload):
        """A synthetic monotonic agent history: turn N appends user/tool/assistant."""
        messages = [{"role": "system", "content": "You are a careful reviewer."}]
        for turn in range(1, turns + 1):
            messages.append({"role": "user", "content": f"[turn {turn}] please check this output"})
            messages.append({"role": "tool", "content": tool_payload(turn)})
            messages.append({"role": "assistant", "content": f"Looked at turn {turn}, looks fine."})
        return messages

    def test_disabled_by_default_is_an_exact_no_op(self):
        cache = _TrimCache()
        messages = self.transcript(5, lambda t: ("large tool output line %d\n" % t) * 500)
        out = trim_messages(messages, keep_last_turns=1, tool_output_tokens=0, cache=cache)
        self.assertIs(out, messages)   # not even copied -- true no-op
        self.assertEqual(render_chat(out), render_chat(messages))

    def test_trims_old_tool_output_and_saves_prompt_bytes(self):
        cache = _TrimCache()
        messages = self.transcript(6, lambda t: ("large tool output line %d\n" % t) * 500)
        baseline = render_chat(messages)
        trimmed = trim_messages(messages, keep_last_turns=1, tool_output_tokens=100, cache=cache)
        diet = render_chat(trimmed)
        self.assertLess(len(diet), len(baseline) * 0.5)   # big, measurable saving
        # every tool message outside the 1-turn window was actually shortened...
        for message in trimmed[:-3]:                        # last turn = last 3 messages (u/t/a)
            if message.get("role") == "tool":
                self.assertIn("ili-trim", message["content"])
        # ...but the most recent turn's tool output is untouched (full-fidelity window).
        self.assertNotIn("ili-trim", trimmed[-2]["content"])
        self.assertEqual(trimmed[-2]["content"], messages[-2]["content"])

    def test_short_tool_output_is_left_untouched_even_if_eligible(self):
        cache = _TrimCache()
        messages = self.transcript(3, lambda t: "short output")
        trimmed = trim_messages(messages, keep_last_turns=1, tool_output_tokens=500, cache=cache)
        self.assertEqual(trimmed, messages)   # nothing exceeded the budget -> nothing changes

    def test_non_tool_messages_are_never_trimmed(self):
        cache = _TrimCache()
        big_user_text = "x" * 5000
        messages = [{"role": "system", "content": "s"},
                    {"role": "user", "content": big_user_text},
                    {"role": "assistant", "content": "y" * 5000},
                    {"role": "user", "content": "next"}]
        trimmed = trim_messages(messages, keep_last_turns=0, tool_output_tokens=10, cache=cache)
        self.assertEqual(trimmed[1]["content"], big_user_text)
        self.assertEqual(trimmed[2]["content"], "y" * 5000)

    def test_trimmed_message_is_frozen_and_stable_as_history_grows(self):
        """The prefix-stability invariant: once turn 1's tool message crosses out of the
        full-fidelity window and gets trimmed, its content -- and the rendered prefix up
        through it -- must never change again, no matter how many more turns are appended."""
        cache = _TrimCache()
        keep_last_turns, budget = 1, 20
        frozen_snapshots = []
        prefix_renders = []
        for turns in range(1, 6):
            messages = self.transcript(turns, lambda t: ("payload for turn %d " % t) * 300)
            trimmed = trim_messages(messages, keep_last_turns, budget, cache)
            frozen_snapshots.append(trimmed[2]["content"])          # turn 1's tool message
            prefix_renders.append(render_chat(trimmed[:3]))         # system+user1+tool1 only
        # Turn 1 is still inside the full-fidelity window when there is only 1 turn total.
        self.assertNotIn("ili-trim", frozen_snapshots[0])
        # From turn 2 onward it has crossed out of the window and must be frozen: identical
        # bytes and an identical rendered prefix on every subsequent (larger) history.
        self.assertIn("ili-trim", frozen_snapshots[1])
        self.assertEqual(len(set(frozen_snapshots[1:])), 1,
                          "trimmed content for turn 1's tool message changed across snapshots")
        self.assertEqual(len(set(prefix_renders[1:])), 1,
                          "rendered prefix through the frozen message changed across snapshots")

    def test_frozen_form_survives_a_mid_session_budget_change(self):
        """Changing ILI_TRIM_TOOL_OUTPUT_TOKENS mid-session must not re-flow a message that
        was already trimmed under the old budget -- that would break prefix stability."""
        cache = _TrimCache()
        messages = self.transcript(3, lambda t: ("payload %d " % t) * 300)
        first = trim_messages(messages, keep_last_turns=1, tool_output_tokens=20, cache=cache)
        again = trim_messages(messages, keep_last_turns=1, tool_output_tokens=80, cache=cache)
        self.assertEqual(first[2]["content"], again[2]["content"])

    def test_excerpt_is_pure_and_deterministic(self):
        content = "abc" * 2000
        self.assertEqual(_trim_excerpt(content, 50), _trim_excerpt(content, 50))
        self.assertEqual(_trim_excerpt(content, 0), content)   # 0 budget => no-op

    def test_cutoff_protects_at_least_keep_last_turns(self):
        messages = self.transcript(4, lambda t: "x")
        # 4 turns total; asking to keep more turns than exist protects everything.
        self.assertEqual(_trim_cutoff(messages, keep_last_turns=10), 0)
        # Keeping 0 turns means nothing is protected.
        self.assertEqual(_trim_cutoff(messages, keep_last_turns=0), len(messages))

    def test_options_default_off_and_request_field_overrides_env(self):
        self.assertEqual(_trim_options({}), (0, 2))
        self.assertEqual(_trim_options({"trim_tool_output_tokens": 64}), (64, 2))
        self.assertEqual(_trim_options({"trim_tool_output_tokens": 64, "trim_keep_last_turns": 5}),
                         (64, 5))
        with self.assertRaises(APIError):
            _trim_options({"trim_tool_output_tokens": -1})
        with self.assertRaises(APIError):
            _trim_options({"trim_keep_last_turns": -1})
        with self.assertRaises(APIError):
            _trim_options({"trim_tool_output_tokens": "many"})


class PrefixSlotRouterTest(unittest.TestCase):
    def test_prefers_complete_prior_prompt(self):
        router = PrefixSlotRouter(2)
        router.record(0, "conversation A: ")
        router.record(1, "conversation B: ")
        self.assertEqual(router.select("conversation A: next turn"), 0)

    def test_longer_partial_prefix_beats_short_complete_prompt(self):
        router = PrefixSlotRouter(2)
        router.record(0, "a")
        router.record(1, "abcdef")
        self.assertEqual(router.select("abcX"), 1)

    def test_fills_empty_slots_before_replacing_unrelated_prompt(self):
        router = PrefixSlotRouter(2)
        self.assertEqual(router.select("alpha"), 0)
        router.record(0, "alpha")
        self.assertEqual(router.select("beta"), 1)

    def test_uses_longest_shared_prefix_then_lru(self):
        router = PrefixSlotRouter(3)
        router.record(0, "shared/one")
        router.record(1, "shared/two")
        router.record(2, "different")
        self.assertEqual(router.select("shared/x"), 0)
        router.record(0, "newer/common-a")
        router.record(1, "newer/common-b")
        router.record(2, "other")
        self.assertEqual(router.select("newer/common-c"), 0)


class HTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FakeEngine()
        cls.server = APIServer(("127.0.0.1", 0),cls.engine,"test-model","secret",16,kv_slots=2)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.scheduler.close()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path, body=None, key="secret"):
        headers = {"Authorization": f"Bearer {key}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        return urlopen(Request(self.base + path, data=data, headers=headers), timeout=2)

    def test_lists_models_and_checks_auth(self):
        with self.request("/v1/models") as response:
            self.assertEqual(json.load(response)["data"][0]["id"], "test-model")
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/models", key="wrong")
        self.assertEqual(caught.exception.code, 401)

    def test_health_reports_scheduler_and_kv_slots(self):
        with self.request("/health") as response:
            health = json.load(response)
            scheduler = health["scheduler"]
        self.assertEqual(scheduler["max_queue"], 8)
        self.assertIn("queued", scheduler)
        self.assertEqual(health["kv_slots"], 2)

    def test_browser_preflight(self):
        request = Request(self.base + "/v1/chat/completions", method="OPTIONS", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        })
        with urlopen(request, timeout=2) as response:
            self.assertEqual(response.status, 204)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
            self.assertIn("Authorization", response.headers["Access-Control-Allow-Headers"])

    def test_chat_completion(self):
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4, "cache_slot": 1,
        }) as response:
            body = json.load(response)
            queue_wait = response.headers.get("x-iliria-queue-wait-ms")
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "Héllo")
        self.assertEqual(body["usage"], {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9})
        self.assertIsNotNone(queue_wait)
        self.assertIn("<|user|>Hi<|assistant|><think></think>", self.engine.calls[-1][0])
        self.assertEqual(self.engine.calls[-1][4], 1)

    def test_context_diet_is_off_by_default_over_http(self):
        big_tool_output = ("stack trace line\n" * 2000)
        with self.request("/v1/chat/completions", {
            "model": "test-model", "cache_slot": 1, "max_tokens": 4,
            "messages": [{"role": "user", "content": "[turn 1] run the tests"},
                         {"role": "tool", "content": big_tool_output},
                         {"role": "assistant", "content": "ok"},
                         {"role": "user", "content": "[turn 2] run again"}],
        }) as response:
            response.read()
        self.assertIn(big_tool_output, self.engine.calls[-1][0])
        self.assertNotIn("ili-trim", self.engine.calls[-1][0])

    def test_context_diet_trims_when_requested_over_http(self):
        big_tool_output = ("stack trace line\n" * 2000)
        with self.request("/v1/chat/completions", {
            "model": "test-model", "cache_slot": 1, "max_tokens": 4,
            "trim_tool_output_tokens": 50, "trim_keep_last_turns": 0,
            "messages": [{"role": "user", "content": "[turn 1] run the tests"},
                         {"role": "tool", "content": big_tool_output},
                         {"role": "assistant", "content": "ok"},
                         {"role": "user", "content": "[turn 2] run again"}],
        }) as response:
            response.read()
        sent_prompt = self.engine.calls[-1][0]
        self.assertNotIn(big_tool_output, sent_prompt)
        self.assertIn("ili-trim", sent_prompt)

    def test_rejects_invalid_cache_slot(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "cache_slot": 2,
            })
        self.assertEqual(caught.exception.code, 400)

    def test_rejects_null_cache_slot(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "cache_slot": None,
            })
        self.assertEqual(caught.exception.code, 400)

    def test_streaming_chat_completion(self):
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "stream": True, "stream_options": {"include_usage": True},
        }) as response:
            stream = response.read().decode()
        self.assertIn('\"delta\":{\"role\":\"assistant\",\"content\":\"\"}', stream)
        self.assertIn('\"object\":\"chat.completion.chunk\"', stream)
        self.assertIn('\"content\":\"Hé\"', stream)
        self.assertIn('\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}', stream)
        self.assertTrue(stream.endswith("data: [DONE]\n\n"))

    def test_legacy_completion(self):
        with self.request("/v1/completions", {
            "model": "test-model", "prompt": "Complete me", "temperature": 0,
        }) as response:
            body = json.load(response)
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["choices"][0]["text"], "Héllo")
        self.assertEqual(self.engine.calls[-1][0], "Complete me")

    def test_rejects_invalid_stream_options(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "stream": True, "stream_options": "usage",
            })
        self.assertEqual(caught.exception.code, 400)


class SchedulerHTTPTest(unittest.TestCase):
    def setUp(self):
        self.engine = BlockingEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model",
                                max_tokens=16, max_queue=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"

    def tearDown(self):
        self.engine.release.set()
        self.server.scheduler.close()
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def request(self):
        body = json.dumps({"model": "test-model", "messages": [
            {"role": "user", "content": "Hi"}]}).encode()
        return urlopen(Request(self.url, data=body, headers={"Content-Type": "application/json"}), timeout=2)

    def test_queue_full_returns_429_before_generation(self):
        first_errors = []

        def first_request():
            try:
                with self.request() as response: response.read()
            except Exception as error:
                first_errors.append(error)

        first = threading.Thread(target=first_request); first.start()
        self.assertTrue(self.engine.entered.wait(1))
        with self.assertRaises(HTTPError) as caught:
            self.request()
        error = json.loads(caught.exception.read())["error"]
        self.assertEqual(caught.exception.code, 429)
        self.assertEqual(caught.exception.headers["Retry-After"], "1")
        self.assertEqual(error["code"], "queue_full")
        self.engine.release.set(); first.join(2)
        self.assertEqual(first_errors, [])


class PrefixSlotHTTPTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model",
                                max_tokens=16, kv_slots=2)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/completions"

    def tearDown(self):
        self.server.scheduler.close()
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def request(self, prompt, cache_slot="missing"):
        body = {"model": "test-model", "prompt": prompt, "max_tokens": 1}
        if cache_slot != "missing":
            body["cache_slot"] = cache_slot
        data = json.dumps(body).encode()
        with urlopen(Request(self.url, data=data, headers={"Content-Type": "application/json"}),
                     timeout=2) as response:
            response.read()

    def test_automatic_assignment_reuses_conversation_slot(self):
        self.request("session-a:")
        self.request("session-b:")
        self.request("session-a: continued")
        self.assertEqual([call[4] for call in self.engine.calls], [0, 1, 0])

    def test_explicit_slot_overrides_and_seeds_affinity(self):
        self.request("manual:", 1)
        self.request("manual: continued")
        self.assertEqual([call[4] for call in self.engine.calls], [1, 1])


class OriginGuardTest(unittest.TestCase):
    """Localhost-CSRF guard (the serve-security review, finding CSRF-1): a malicious webpage
    open in the user's browser reaches this loopback-bound server exactly like the desktop
    app does -- the OS makes no distinction between them. CORS response headers alone do
    not stop this: a "simple" cross-origin request (Content-Type: text/plain, which is
    CORS-safelisted, or a bare HTML form submission) skips the preflight entirely, so
    without an explicit Origin check the server would execute it blind."""

    @classmethod
    def setUpClass(cls):
        cls.engine = FakeEngine()
        cls.server = APIServer(("127.0.0.1", 0), cls.engine, "test-model", kv_slots=1)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.scheduler.close()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def post(self, origin, content_type="application/json"):
        headers = {"Content-Type": content_type}
        if origin is not None:
            headers["Origin"] = origin
        body = json.dumps({"model": "test-model",
                           "messages": [{"role": "user", "content": "Hi"}]}).encode()
        return urlopen(Request(self.base + "/v1/chat/completions", data=body, headers=headers),
                       timeout=2)

    def test_rejects_disallowed_origin_even_as_a_simple_request(self):
        # text/plain is CORS-safelisted: a real browser sends this exact request with NO
        # preflight. The fix must still reject it server-side, before the engine runs.
        before = len(self.engine.calls)
        with self.assertRaises(HTTPError) as caught:
            self.post(origin="http://evil.example", content_type="text/plain")
        self.assertEqual(caught.exception.code, 403)
        self.assertEqual(json.loads(caught.exception.read())["error"]["code"], "origin_not_allowed")
        self.assertEqual(len(self.engine.calls), before)   # rejected before reaching the engine

    def test_rejects_disallowed_origin_on_get_too(self):
        request = Request(self.base + "/health", headers={"Origin": "http://evil.example"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=2)
        self.assertEqual(caught.exception.code, 403)

    def test_allows_the_desktop_apps_own_origin(self):
        with self.post(origin="tauri://localhost") as response:
            self.assertEqual(response.status, 200)

    def test_allows_the_dev_server_origin(self):
        with self.post(origin="http://localhost:5173") as response:
            self.assertEqual(response.status, 200)

    def test_allows_requests_with_no_origin_header(self):
        # curl / the OpenAI SDK / this repo's own scripts never send Origin.
        with self.post(origin=None) as response:
            self.assertEqual(response.status, 200)

    def test_wildcard_opt_out_allows_any_origin(self):
        engine = FakeEngine()
        server = APIServer(("127.0.0.1", 0), engine, "test-model", kv_slots=1,
                           cors_origins=("*",))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = json.dumps({"model": "test-model",
                               "messages": [{"role": "user", "content": "Hi"}]}).encode()
            request = Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                              data=body, headers={"Content-Type": "application/json",
                                                  "Origin": "http://anything.example"})
            with urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 200)
        finally:
            server.scheduler.close(); server.shutdown(); server.server_close(); thread.join(timeout=2)


class SlowClientTest(unittest.TestCase):
    """Regression test for the slowloris guard (the serve-security review, finding DOS-1): a
    connection that never finishes sending its request must be dropped after
    APIHandler.timeout seconds, not held open forever."""

    def test_incomplete_request_is_dropped_after_timeout(self):
        class FastTimeoutHandler(APIHandler):
            timeout = 0.3   # production uses 60s (APIHandler.timeout); shortened for the test

        engine = FakeEngine()
        server = APIServer(("127.0.0.1", 0), engine, "test-model", kv_slots=1)
        server.RequestHandlerClass = FastTimeoutHandler
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            sock = socket.create_connection(("127.0.0.1", server.server_port), timeout=5)
            try:
                # A complete request line but no header-terminating blank line: the server
                # blocks reading headers forever without the timeout fix.
                sock.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n")
                started = time.monotonic()
                data = sock.recv(4096)
                elapsed = time.monotonic() - started
            finally:
                sock.close()
        finally:
            server.scheduler.close(); server.shutdown(); server.server_close(); thread.join(timeout=2)
        self.assertEqual(data, b"")      # server closed the connection; no response was sent
        self.assertLess(elapsed, 2.0)    # bounded by the handler's 0.3s timeout, not hung forever


class BindAddressTest(unittest.TestCase):
    """the serve-security review finding BIND-1: the CLI default must stay loopback-only, and
    ILI_SERVE_BIND must be able to override it without a code change or a hand-added
    --host flag every time -- while an explicit --host still wins over the env var."""

    @staticmethod
    def _clean_env():
        env = dict(os.environ)
        for prefix in ("ILI_", "COLI_", "FA_"):
            env.pop(prefix + "SERVE_BIND", None)
        return env

    def test_defaults_to_loopback_without_env_override(self):
        with mock.patch.dict(os.environ, self._clean_env(), clear=True):
            args = build_arg_parser().parse_args(["--model", "/tmp/does-not-matter"])
        self.assertEqual(args.host, "127.0.0.1")

    def test_env_override_changes_the_default(self):
        env = self._clean_env(); env["ILI_SERVE_BIND"] = "0.0.0.0"
        with mock.patch.dict(os.environ, env, clear=True):
            args = build_arg_parser().parse_args(["--model", "/tmp/does-not-matter"])
        self.assertEqual(args.host, "0.0.0.0")

    def test_explicit_flag_beats_the_env_override(self):
        env = self._clean_env(); env["ILI_SERVE_BIND"] = "0.0.0.0"
        with mock.patch.dict(os.environ, env, clear=True):
            args = build_arg_parser().parse_args(
                ["--model", "/tmp/does-not-matter", "--host", "127.0.0.1"])
        self.assertEqual(args.host, "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
