"""Item B (32K/64K/128K decode re-profile harness): tests for the parts that don't need
the real engine -- corpus determinism, the CTX/RAM budget preflight, serve-log parsing, and
a full --mock dry run of the CLI. The engine-ownership gate (`refuse_if_bench_running`,
`pgrep -x glm`) is deliberately NOT exercised here: it depends on live system process state
(this machine had a real bench `glm` process running while this harness was built, and the
gate was verified against it by hand -- see the session report) and is not something a
hermetic unit test should fake by spawning a process literally named `glm`.
"""

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = C_DIR / "scripts"
SCRIPT = SCRIPTS_DIR / "long_ctx_profile.py"

sys.path.insert(0, str(SCRIPTS_DIR))
from long_ctx_profile import (          # noqa: E402  (path insert must come first)
    DENSE_MODEL_GB, KV_BYTES_PER_TOKEN_PER_SLOT, MIN_EXPERT_HEADROOM_GB, build_transcript,
    check_ctx_budget, estimate_prompt_tokens, parse_last_kv_line, parse_last_profile)


class CorpusGeneratorTest(unittest.TestCase):
    def test_deterministic_across_calls(self):
        first = build_transcript(6000, str(C_DIR))
        second = build_transcript(6000, str(C_DIR))
        self.assertEqual(first, second)

    def test_larger_target_extends_smaller_ones_message_list(self):
        small = build_transcript(4000, str(C_DIR))
        big = build_transcript(9000, str(C_DIR))
        self.assertGreater(len(big), len(small))
        self.assertEqual(small, big[:len(small)])

    def test_reaches_roughly_the_requested_depth(self):
        messages = build_transcript(5000, str(C_DIR))
        estimate = estimate_prompt_tokens(messages)
        self.assertGreaterEqual(estimate, 5000)
        self.assertLess(estimate, 5000 * 1.5)   # generous slack: chunks stop early at file boundaries

    def test_transcript_content_is_never_random_tokens(self):
        """Tool responses must be real slices of the repo's own source files."""
        messages = build_transcript(3000, str(C_DIR))
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        self.assertTrue(tool_messages)
        for message in tool_messages:
            self.assertIn("sed -n", message["content"])   # deterministic slice marker
        # at least one tool response should contain a real, recognizable line from glm.c
        combined = "\n".join(m["content"] for m in tool_messages)
        self.assertIn("#include", combined)

    def test_raises_on_no_valid_source_files(self):
        with self.assertRaises(FileNotFoundError):
            build_transcript(1000, str(C_DIR), sources=("does/not/exist.c",))


class CtxBudgetTest(unittest.TestCase):
    def test_matches_the_documented_182kb_per_token_formula(self):
        # 79 * (512 + 64) * 4 = 182,016 bytes/token/slot for GLM-5.2's dims -- see the
        # module docstring and run-m5max-serve.sh's "~182 KB/token" comment.
        self.assertEqual(KV_BYTES_PER_TOKEN_PER_SLOT, 182_016)

    def test_default_targets_all_fit_the_default_ram_budget(self):
        for target in (32_000, 64_000, 128_000):
            kv_gb, headroom_gb, ok = check_ctx_budget(target, ram_gb=114.0, kv_slots=1)
            self.assertTrue(ok, f"{target} tokens unexpectedly tight at the documented default")
            self.assertGreater(headroom_gb, MIN_EXPERT_HEADROOM_GB)
        # and the specific numbers cited in the module docstring
        kv_gb, headroom_gb, _ = check_ctx_budget(128_000, ram_gb=114.0, kv_slots=1)
        self.assertAlmostEqual(kv_gb, 23.30, places=1)
        self.assertAlmostEqual(headroom_gb, 80.8, places=1)

    def test_flags_a_too_tight_budget(self):
        _, headroom_gb, ok = check_ctx_budget(128_000, ram_gb=20.0, kv_slots=1)
        self.assertFalse(ok)
        self.assertLess(headroom_gb, MIN_EXPERT_HEADROOM_GB)

    def test_more_kv_slots_costs_proportionally_more(self):
        kv_gb_1, _, _ = check_ctx_budget(32_000, ram_gb=114.0, kv_slots=1)
        kv_gb_2, _, _ = check_ctx_budget(32_000, ram_gb=114.0, kv_slots=2)
        self.assertAlmostEqual(kv_gb_2, kv_gb_1 * 2, places=6)


class LogParsingTest(unittest.TestCase):
    def test_parses_the_real_profile_line_format(self):
        # exact format from c/glm.c (commit 8d26555)
        line = ("[API] PROFILE prefill 1361 tok in 452.3s: expert-disk 3.0s | "
               "expert-matmul 27.4s | attention 383.7s (kvb 12.1s) | lm_head 0.5s | other 38.2s\n")
        parsed = parse_last_profile(line)
        self.assertEqual(parsed["prefill_tokens"], 1361)
        self.assertEqual(parsed["expert_disk_s"], 3.0)
        self.assertEqual(parsed["attention_s"], 383.7)

    def test_parses_the_real_kv_slot_line_format(self):
        line = "[API] KV slot 0 prefix 1639/3096 token, prefill 1457\n"
        parsed = parse_last_kv_line(line)
        self.assertEqual(parsed, {"slot": 0, "prefix_tokens": 1639, "prompt_tokens": 3096,
                                  "prefill_delta_tokens": 1457})

    def test_returns_none_when_absent(self):
        self.assertIsNone(parse_last_profile("no profile lines here\n"))
        self.assertIsNone(parse_last_kv_line("no kv lines here\n"))

    def test_uses_the_last_match_when_several_are_present(self):
        text = ("[API] KV slot 0 prefix 0/100 token, prefill 100\n"
               "[API] KV slot 0 prefix 100/150 token, prefill 50\n")
        self.assertEqual(parse_last_kv_line(text)["prefill_delta_tokens"], 50)


class MockDryRunTest(unittest.TestCase):
    """End-to-end CLI smoke test against the in-process --mock server: no engine, no disk
    I/O near the model, matching the campaign's hard constraint on this study."""

    def test_mock_run_writes_a_well_formed_csv_and_markdown(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            md_path = Path(tmp) / "out.md"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--mock", "--targets", "3000,6000",
                 "--decode-tokens", "10", "--mock-port", "8879",
                 "--out-csv", str(csv_path), "--out-md", str(md_path)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())

            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)   # one row per target, warm only (no --cold)
            self.assertEqual([r["target_tokens"] for r in rows], ["3000", "6000"])
            for row in rows:
                self.assertEqual(row["mode"], "warm")
                self.assertTrue(row["decode_tok_s"])
                self.assertTrue(row["expert_disk_share_of_delta_prefill_pct"])

            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("expert_disk_share_of_delta_prefill_pct", markdown)
            self.assertIn("3000", markdown)

    def test_mock_run_respects_cold_flag_and_doubles_rows(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            md_path = Path(tmp) / "out.md"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--mock", "--targets", "3000", "--cold",
                 "--decode-tokens", "10", "--mock-port", "8880",
                 "--out-csv", str(csv_path), "--out-md", str(md_path)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["mode"] for r in rows], ["cold", "warm"])

    def test_engine_gate_is_skipped_under_mock(self):
        """--mock must never invoke the pgrep engine-ownership gate."""
        with __import__("tempfile").TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            md_path = Path(tmp) / "out.md"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--mock", "--targets", "2000",
                 "--decode-tokens", "5", "--mock-port", "8881",
                 "--out-csv", str(csv_path), "--out-md", str(md_path)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("refusing to start", result.stderr)

    def test_live_mode_without_serve_log_fails_fast(self):
        # Deterministic prerequisite contract (2026-07-16 suite classification, rows 17-18):
        # this test previously depended SILENTLY on a live `ili serve` (the script's pgrep
        # gate runs BEFORE the serve-log check, so with no glm process it failed on the wrong
        # error). A PATH shim satisfies the process gate deterministically so the serve-log
        # fast-fail behavior under test is exercised as a unit, on any machine, no live engine.
        import tempfile, os, stat
        shim_dir = tempfile.mkdtemp(prefix="pgrep-shim-")
        shim_path = os.path.join(shim_dir, "pgrep")
        with open(shim_path, "w") as fh:
            fh.write("#!/bin/sh\necho 99999\nexit 0\n")
        os.chmod(shim_path, os.stat(shim_path).st_mode | stat.S_IEXEC)
        shim_env = dict(os.environ, PATH=shim_dir + os.pathsep + os.environ.get("PATH", ""))
        """Without --mock, --serve-log is mandatory (unless the engine-ownership gate
        already refused first, which is equally acceptable -- both are safe failures that
        never attempt an HTTP call against a nonexistent/uncontrolled server)."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--targets", "2000",
             "--out-csv", "/tmp/should-not-be-created.csv", "--out-md", "/tmp/should-not-be-created.md"],
            env=shim_env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(
            "serve-log is required" in result.stderr or "refusing to start" in result.stderr,
            result.stderr)


if __name__ == "__main__":
    unittest.main()
