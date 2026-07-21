"""Serve-mode large-prompt regression tests (2026-07-14 39,749-token freeze).

Three layers of coverage:

  * EngineWritePathTest — pure protocol, no model: the real openai_server.Engine
    against a scripted child that speaks the \x02PROMPT protocol but drains its
    stdin slowly. Proves the parent's write strategy survives payloads well past
    the 64 KiB pipe buffer and that the 16 MiB frame cap rejects before writing.
  * LargePromptServeTest — the REAL `glm` serve loop on the tiny numpy fixture
    (tiny_serve_fixture.py): >128 KiB prompts round-trip, chunked prefill is
    byte-identical to the monolithic path at temperature 0, and consecutive
    large prompts stay framed.
  * RejectedFrameTest — a rejected \x02PROMPT header (bad kv slot) must DRAIN its
    payload; the next valid frame still gets a correct, in-sync reply. Before
    the drain fix the payload was replayed as interactive chat lines and the
    server read someone else's response forever.

Skips cleanly when the glm binary or numpy is unavailable (CI without a
toolchain), mirroring the fixture-dependent tests elsewhere in this directory.
"""

import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
GLM = HERE / "glm"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from openai_server import Engine, APIError, PROMPT_BYTES_MAX  # noqa: E402

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

END = b"\x01\x01END\x01\x01\n"
READY = b"\x01\x01READY\x01\x01\n"
TURN_TIMEOUT = 90       # generous: a hang, not slowness, is the failure mode


def run_with_timeout(fn, timeout=TURN_TIMEOUT):
    """Run fn in a worker; return (finished, result_or_exception)."""
    box = {}

    def work():
        try:
            box["result"] = fn()
        except Exception as error:            # surfaced by the caller
            box["error"] = error

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return False, None
    if "error" in box:
        raise box["error"]
    return True, box.get("result")


# A stand-in engine child: same wire protocol as glm's run_serve, but it sleeps
# before draining each payload (the window where glm does usage_save and
# kv_disk_append while the parent already writes the next frame) and echoes a
# short reply. Slow draining forces the parent's write() to block on payloads
# beyond the pipe buffer — the exact write path under test.
FAKE_CHILD = r"""
import os, sys, time
out = sys.stdout.buffer
out.write(b"\x01\x01READY\x01\x01\n"); out.write(b"STAT 0 0.00 0.0 0.01\n"); out.flush()
stdin = sys.stdin.buffer
while True:
    line = stdin.readline()
    if not line:
        break
    if not line.startswith(b"\x02PROMPT "):
        out.write(b"\x01\x01END\x01\x01\n"); out.write(b"STAT 0 0.00 0.0 0.01 0 0\n"); out.flush()
        continue
    nb = int(line.split()[1])
    time.sleep(0.5)                       # parent must survive a full pipe meanwhile
    got = 0
    while got < nb:
        chunk = stdin.read(min(65536, nb - got))
        if not chunk:
            sys.exit(1)
        got += len(chunk)
    stdin.read(1)                         # trailing delimiter
    out.write(b"ok:%d" % got)
    out.write(b"\x01\x01END\x01\x01\n")
    out.write(b"STAT 1 1.00 0.0 0.01 %d 0\n" % (nb // 10))
    out.flush()
"""


class EngineWritePathTest(unittest.TestCase):
    """openai_server.Engine write strategy over a real pipe, no model needed."""

    def setUp(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(FAKE_CHILD)
            self.child_script = f.name
        self.engine = Engine.__new__(Engine)
        self.engine.process = subprocess.Popen(
            [sys.executable, self.child_script], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, bufsize=0,
        )
        self.engine.lock = threading.Lock()
        self.engine.kv_slots = 1
        from openai_server import read_engine_turn
        read_engine_turn(self.engine.process.stdout, READY, lambda _: None)

    def tearDown(self):
        self.engine.close()
        os.unlink(self.child_script)

    def roundtrip(self, size):
        prompt = "x" * size
        got = []
        finished, stats = run_with_timeout(
            lambda: self.engine.generate(prompt, 8, 0.7, 0.9, got.append))
        self.assertTrue(finished, f"engine write path hung on a {size}-byte payload")
        self.assertEqual("".join(got), f"ok:{size}")
        self.assertEqual(stats["prompt_tokens"], size // 10)

    def test_payload_beyond_pipe_buffer(self):
        self.roundtrip(65 * 1024)         # just past the 64 KiB pipe buffer

    def test_payload_far_beyond_pipe_buffer(self):
        self.roundtrip(200 * 1024)        # tonight's failing order of magnitude

    def test_oversized_prompt_is_rejected_before_writing(self):
        with self.assertRaises(APIError) as ctx:
            self.engine.generate("y" * (PROMPT_BYTES_MAX + 1), 8, 0.7, 0.9,
                                 lambda _: None)
        self.assertEqual(ctx.exception.status, 400)
        # the engine child never saw a frame: a small follow-up still works
        self.roundtrip(1000)


@unittest.skipUnless(GLM.exists() and os.access(GLM, os.X_OK), "glm binary not built")
@unittest.skipUnless(HAVE_NUMPY, "numpy unavailable: cannot build the tiny fixture")
class FixtureBackedTest(unittest.TestCase):
    """Base: build the tiny model once, then drive the real serve loop."""

    @classmethod
    def setUpClass(cls):
        import tiny_serve_fixture
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = tiny_serve_fixture.build(Path(cls._tmp.name) / "glm_tiny")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def engine_env(self, **extra):
        env = dict(os.environ, CTX="1024", KVSAVE="0", MTP="0",
                   OMP_NUM_THREADS="2")
        env.update({k: str(v) for k, v in extra.items()})
        return env


class LargePromptServeTest(FixtureBackedTest):
    def generate(self, engine, prompt, max_tokens=8, temperature=0.7):
        got = []
        finished, stats = run_with_timeout(
            lambda: engine.generate(prompt, max_tokens, temperature, 0.9,
                                    got.append, cache_slot=0))
        self.assertTrue(finished, "real glm serve loop hung")
        return "".join(got), stats

    def test_prompts_past_128k_round_trip(self):
        engine = Engine(GLM, self.model, cap=8, max_tokens=32,
                        env=self.engine_env())
        try:
            for size in (1024, 65 * 1024, 140 * 1024, 256 * 1024):
                text = ("scroll %d " % size) * (size // 12 + 1)
                reply, stats = self.generate(engine, text[:size])
                # the assertions that matter: the turn completed (no hang) and
                # the stream stayed framed enough to parse a plausible STAT
                self.assertGreater(stats["prompt_tokens"], 0, f"size {size}")
                self.assertGreaterEqual(stats["completion_tokens"], 0, f"size {size}")
        finally:
            engine.close()

    def test_chunked_prefill_matches_monolithic_greedy(self):
        prompt = ("The tiny model reads a very long scroll. " * 4000)[:120000]
        replies = {}
        for label, chunk in (("mono", 0), ("chunked", 256)):
            engine = Engine(GLM, self.model, cap=8, max_tokens=64,
                            env=self.engine_env(ILI_PREFILL_CHUNK=chunk))
            try:
                reply, stats = self.generate(engine, prompt, max_tokens=24,
                                             temperature=0.0)
                replies[label] = (reply, stats["prompt_tokens"])
            finally:
                engine.close()
        self.assertEqual(replies["mono"], replies["chunked"])


class RejectedFrameTest(FixtureBackedTest):
    def test_rejected_header_drains_payload_and_stays_framed(self):
        process = subprocess.Popen(
            [str(GLM), "8"],
            env=self.engine_env(SNAP=str(self.model), SERVE="1", NGEN="32",
                                KV_SLOTS="1"),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )
        try:
            def read_turn():
                buf = b""
                while not buf.endswith(END):
                    byte = process.stdout.read(1)
                    self.assertNotEqual(byte, b"", "engine exited mid-turn")
                    buf += byte
                return buf[:-len(END)], process.stdout.readline()

            buf = b""
            while not buf.endswith(READY):
                buf += process.stdout.read(1)
            process.stdout.readline()                     # startup STAT

            # invalid slot (9 >= KV_SLOTS) with a payload past the pipe buffer
            payload = b"z" * 100_000
            frame = (b"\x02PROMPT %d 8 0.7 0.9 9\n" % len(payload)) + payload + b"\n"
            writer = threading.Thread(
                target=lambda: (process.stdin.write(frame), process.stdin.flush()),
                daemon=True)
            writer.start()
            finished, turn = run_with_timeout(read_turn)
            self.assertTrue(finished, "reject turn hung: payload was not drained")
            body, stat = turn
            self.assertEqual(body, b"", "rejected frame must produce no text")
            writer.join(TURN_TIMEOUT)
            self.assertFalse(writer.is_alive(), "writer stuck: payload not consumed")

            # the very next valid frame must be answered in sync
            prompt = b"a small valid prompt after the rejected one"
            process.stdin.write(
                (b"\x02PROMPT %d 4 0 0.9 0\n" % len(prompt)) + prompt + b"\n")
            process.stdin.flush()
            finished, turn = run_with_timeout(read_turn)
            self.assertTrue(finished, "valid frame after reject hung: stream desynced")
            _, stat = turn
            fields = stat.split()
            self.assertEqual(fields[0], b"STAT")
            # prompt_tokens (field 5) must be THIS prompt's count — one byte-level
            # token per byte — not the replayed z-payload's
            self.assertEqual(int(fields[5]), len(prompt))
        finally:
            process.stdin.close()
            process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
