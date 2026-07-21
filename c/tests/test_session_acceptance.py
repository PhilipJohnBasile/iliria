"""Session-level acceptance harness (scripts/session_acceptance.py): tests for the parts that
don't need a real engine -- the 20-turn plan's invariants, the deterministic corpus-to-messages
construction (reusing long_ctx_profile's `_LineCycler`), the context-diet savings diagnostic
(reusing the REAL openai_server.trim_messages()/render_chat(), not a mock estimate), the
headline-comparison table, and a full --mock dry run of the CLI. Like test_long_ctx_profile.py,
the engine-ownership gate (`refuse_if_bench_running`, `pgrep -x glm`) is not exercised here --
it depends on live system process state and is not something a hermetic unit test should fake.

No real engine (`ili serve`/`glm`) is ever invoked by this suite; every subprocess test below
passes `--mock`.
"""

import csv
import subprocess
import sys
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = C_DIR / "scripts"
SCRIPT = SCRIPTS_DIR / "session_acceptance.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import session_acceptance as sa   # noqa: E402  (path insert must come first)


def _assemble_full_history(plan):
    """A full monotonic message history for `plan` (session_acceptance.build_session_turns'
    output) with a canned assistant reply per turn -- enough to exercise render_chat()/
    trim_messages() without needing a live server's actual reply text."""
    messages = [{"role": "system", "content": sa.serve_gate.DISCIPLINED_SYSTEM}]
    for step in plan:
        messages.extend(step["new_messages"])
        messages.append({"role": "assistant", "content": f"turn {step['turn']}: looks fine."})
    return messages


class SessionPlanTest(unittest.TestCase):
    def test_plan_has_20_turns_with_documented_kind_split(self):
        self.assertTrue(sa.validate_session_plan(sa.SESSION_PLAN_SPEC))
        kinds = [p[1] for p in sa.SESSION_PLAN_SPEC]
        self.assertEqual(len(sa.SESSION_PLAN_SPEC), 20)
        self.assertEqual(kinds.count("ask"), 10)
        self.assertEqual(kinds.count("review"), 6)
        self.assertEqual(kinds.count("paste"), 4)

    def test_turn_numbers_are_sequential(self):
        turns = [p[0] for p in sa.SESSION_PLAN_SPEC]
        self.assertEqual(turns, list(range(1, 21)))

    def test_targets_roughly_the_roadmaps_22k_ctx_session(self):
        total = sum(p[2] for p in sa.SESSION_PLAN_SPEC)
        self.assertGreaterEqual(total, 20_000)
        self.assertLessEqual(total, 24_000)

    def test_max_tokens_follows_the_200_400_700_discipline_mix(self):
        by_kind = {p[1]: p[3] for p in sa.SESSION_PLAN_SPEC}
        self.assertEqual(by_kind["ask"], 200)
        self.assertEqual(by_kind["review"], 400)
        self.assertEqual(by_kind["paste"], 700)

    def test_rejects_wrong_turn_count(self):
        with self.assertRaises(ValueError):
            sa.validate_session_plan(sa.SESSION_PLAN_SPEC[:19])

    def test_rejects_wrong_kind_split(self):
        bad = [(i + 1, "ask", 300, 200) for i in range(20)]
        with self.assertRaises(ValueError):
            sa.validate_session_plan(bad)

    def test_rejects_non_sequential_turn_numbers(self):
        bad = [(p[0] + 1, p[1], p[2], p[3]) for p in sa.SESSION_PLAN_SPEC]
        with self.assertRaises(ValueError):
            sa.validate_session_plan(bad)


class BuildSessionTurnsTest(unittest.TestCase):
    def test_deterministic_across_calls(self):
        first = sa.build_session_turns(str(C_DIR))
        second = sa.build_session_turns(str(C_DIR))
        self.assertEqual(first, second)

    def test_kind_and_size_order_matches_the_plan(self):
        plan = sa.build_session_turns(str(C_DIR))
        self.assertEqual(len(plan), 20)
        for step, spec in zip(plan, sa.SESSION_PLAN_SPEC):
            self.assertEqual(step["turn"], spec[0])
            self.assertEqual(step["kind"], spec[1])
            self.assertEqual(step["target_tokens"], spec[2])
            self.assertEqual(step["max_tokens"], spec[3])

    def test_review_turns_have_the_tool_call_shape(self):
        plan = sa.build_session_turns(str(C_DIR))
        review_steps = [s for s in plan if s["kind"] == "review"]
        self.assertEqual(len(review_steps), 6)
        for step in review_steps:
            roles = [m["role"] for m in step["new_messages"]]
            self.assertEqual(roles, ["user", "assistant", "tool"])
            self.assertIsNone(step["new_messages"][1]["content"])
            self.assertTrue(step["new_messages"][1]["tool_calls"])
            self.assertIn("sed -n", step["new_messages"][2]["content"])

    def test_ask_and_paste_turns_are_a_single_user_message(self):
        plan = sa.build_session_turns(str(C_DIR))
        for step in plan:
            if step["kind"] in ("ask", "paste"):
                self.assertEqual(len(step["new_messages"]), 1)
                self.assertEqual(step["new_messages"][0]["role"], "user")

    def test_content_is_never_random_tokens(self):
        """Every turn's attached content must be a real slice of the repo's own source."""
        plan = sa.build_session_turns(str(C_DIR))
        combined = "\n".join(
            m.get("content") or ""
            for step in plan for m in step["new_messages"]
            if isinstance(m.get("content"), str))
        self.assertIn("#include", combined)

    def test_never_repeats_content_across_turns(self):
        """The shared _LineCycler advances monotonically turn to turn -- no two turns should
        pull an identical chunk (each next_chunk call continues from where the last left off)."""
        plan = sa.build_session_turns(str(C_DIR))
        chunks = []
        for step in plan:
            for m in step["new_messages"]:
                content = m.get("content")
                if isinstance(content, str) and "```" in content:
                    chunks.append(content.split("```")[1])
        self.assertEqual(len(chunks), len(set(chunks)))

    def test_raises_on_no_valid_source_files(self):
        with self.assertRaises(FileNotFoundError):
            sa.build_session_turns(str(C_DIR), sources=("does/not/exist.c",))


class ContextDietDiagnosticTest(unittest.TestCase):
    """Exercises _context_diet_savings against the REAL openai_server.trim_messages()/
    render_chat() -- the same production code path the context-diet analysis's own reproduction
    recipe uses -- so these numbers are measured, not mocked."""

    def test_savings_are_positive_and_bounded_when_enabled(self):
        plan = sa.build_session_turns(str(C_DIR))
        messages = _assemble_full_history(plan)
        cache = sa.openai_server._TrimCache()
        full_chars, trimmed_chars, pct = sa._context_diet_savings(
            messages, keep_last_turns=2, tool_output_tokens=300, cache=cache)
        self.assertIsNotNone(full_chars)
        self.assertLess(trimmed_chars, full_chars)
        self.assertGreater(pct, 0)
        self.assertLess(pct, 100)

    def test_zero_budget_means_zero_savings(self):
        plan = sa.build_session_turns(str(C_DIR))
        messages = _assemble_full_history(plan)
        cache = sa.openai_server._TrimCache()
        full_chars, trimmed_chars, pct = sa._context_diet_savings(
            messages, keep_last_turns=2, tool_output_tokens=0, cache=cache)
        self.assertEqual(full_chars, trimmed_chars)
        self.assertEqual(pct, 0.0)

    def test_render_failure_is_handled_gracefully(self):
        """A diagnostic-only computation must never raise -- it degrades to (None, None, None)."""
        cache = sa.openai_server._TrimCache()
        bad_messages = [{"role": "not-a-real-role", "content": "x"}]
        full_chars, trimmed_chars, pct = sa._context_diet_savings(
            bad_messages, keep_last_turns=2, tool_output_tokens=300, cache=cache)
        self.assertIsNone(full_chars)
        self.assertIsNone(trimmed_chars)
        self.assertIsNone(pct)

    def test_only_review_turns_tool_output_is_eligible(self):
        """Paste turns' large content lives in the user message, never role:tool, so it must
        never shrink -- the documented honesty note in the module docstring."""
        plan = sa.build_session_turns(str(C_DIR))
        messages = _assemble_full_history(plan)
        cache = sa.openai_server._TrimCache()
        trimmed = sa.openai_server.trim_messages(
            messages, keep_last_turns=0, tool_output_tokens=50, cache=cache)
        paste_user_contents = [m["content"] for m in messages if m.get("role") == "user"
                               and m["content"].count("```") and "sed -n" not in m["content"]]
        trimmed_user_contents = [m["content"] for m in trimmed if m.get("role") == "user"
                                 and m["content"].count("```") and "sed -n" not in m["content"]]
        self.assertEqual(paste_user_contents, trimmed_user_contents)


class HeadlineTableTest(unittest.TestCase):
    def test_classify_measured_minutes_bands(self):
        # Bands are NOT monotonic: reuse+kernel (90-120) < naive (148) < reuse-only (300-420).
        self.assertIn("PROJECTED band", sa.classify_measured_minutes(90.0))
        self.assertIn("PROJECTED band", sa.classify_measured_minutes(120.0))
        self.assertIn("naive baseline", sa.classify_measured_minutes(148.0))
        self.assertIn("reuse-only", sa.classify_measured_minutes(300.0))
        self.assertIn("reuse-only", sa.classify_measured_minutes(420.0))
        self.assertIn("regression", sa.classify_measured_minutes(500.0))

    def test_headline_table_lines_cite_all_three_reference_points(self):
        summary = {"measured_minutes": 100.0, "mock": False, "attempt_id": "session-test"}
        lines = "\n".join(sa.headline_table_lines(summary))
        self.assertIn("148.0", lines)                    # naive baseline
        self.assertIn("300.0-420.0", lines)               # reuse-only band (5-7h in minutes)
        self.assertIn("90.0-120.0", lines)                # reuse+kernel projected band
        self.assertIn("session-test", lines)

    def test_mock_run_is_flagged_as_non_authoritative(self):
        summary = {"measured_minutes": 5.0, "mock": True, "attempt_id": "session-test"}
        lines = "\n".join(sa.headline_table_lines(summary))
        self.assertIn("NOTE: this run used", lines)


class MockDryRunTest(unittest.TestCase):
    """End-to-end CLI smoke tests against the in-process --mock server (reused directly from
    long_ctx_profile.start_mock_server): no engine, no disk I/O near the model."""

    def test_mock_run_writes_a_well_formed_csv_and_markdown(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            md_path = Path(tmp) / "out.md"
            # Explicit, per-test --serve-log: avoids relying solely on the script's own
            # mkstemp-based uniqueness for a shared default path, and keeps this test
            # hermetic even if another --mock run is in flight concurrently (e.g. the
            # test suite's own other subprocess tests, or a manual run alongside it).
            log_path = Path(tmp) / "serve.log"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--mock", "--mock-port", "8900",
                 "--serve-log", str(log_path), "--attempt-id", "session-test-off-warm",
                 "--out-csv", str(csv_path), "--out-md", str(md_path)],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("refusing to start", result.stderr)   # engine gate must be skipped
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())

            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            self.assertEqual([r["turn"] for r in rows], [str(i) for i in range(1, 21)])
            self.assertEqual([r["kind"] for r in rows],
                             [spec[1] for spec in sa.SESSION_PLAN_SPEC])
            self.assertTrue(all(r["context_diet"] == "off" for r in rows))
            self.assertTrue(all(r["start_mode"] == "warm" for r in rows))
            self.assertEqual(rows[0]["reuse_engaged"], "")   # turn 1: n/a, not a failure
            for r in rows[1:]:
                self.assertIn(r["reuse_engaged"], ("True", "False"))
            self.assertTrue(all(r["ttft_s"] for r in rows))
            self.assertTrue(all(r["decode_tok_s"] for r in rows))

            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("session_acceptance results", markdown)
            self.assertIn("Headline", markdown)
            self.assertIn("148.0", markdown)
            self.assertIn("session-test-off-warm", markdown)
            self.assertIn("this run used `--mock`", markdown)

    def test_context_diet_on_reports_trim_savings(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            md_path = Path(tmp) / "out.md"
            log_path = Path(tmp) / "serve.log"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--mock", "--mock-port", "8901",
                 "--serve-log", str(log_path),
                 "--attempt-id", "session-test-on-warm", "--context-diet", "on",
                 "--out-csv", str(csv_path), "--out-md", str(md_path)],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)

            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(r["context_diet"] == "on" for r in rows))
            # by the last turn, at least one "review" turn's tool output should be eligible
            # for trimming and the savings diagnostic should have produced a real number
            last_pct = float(rows[-1]["trim_savings_pct"])
            self.assertGreaterEqual(last_pct, 0.0)
            self.assertLess(last_pct, 100.0)

            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("context-diet final cumulative savings", markdown)

    def test_cold_flag_warns_and_skips_purge_under_mock(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            md_path = Path(tmp) / "out.md"
            log_path = Path(tmp) / "serve.log"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--mock", "--mock-port", "8902", "--cold",
                 "--serve-log", str(log_path), "--attempt-id", "session-test-cold",
                 "--out-csv", str(csv_path), "--out-md", str(md_path)],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("skipping purge", result.stderr)
            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(r["start_mode"] == "cold" for r in rows))

    def test_concurrent_mock_runs_do_not_share_a_default_log_path(self):
        """Regression test for a real bug found while validating this harness: without
        --serve-log, two concurrent --mock runs used to share ONE fixed tempfile path, so
        each run's startup truncation (`open(log_path, "w")`) could wipe out the other's
        in-flight [API] KV slot lines, silently blanking prefill_delta_engine/reuse_engaged
        for whichever turns lost the race. Runs two full 20-turn sessions at once, neither
        passing --serve-log, and requires BOTH to come back with every turn 2-20 populated."""
        with __import__("tempfile").TemporaryDirectory() as tmp:
            def launch(port, attempt_id):
                csv_path = Path(tmp) / f"{attempt_id}.csv"
                md_path = Path(tmp) / f"{attempt_id}.md"
                proc = subprocess.Popen(
                    [sys.executable, str(SCRIPT), "--mock", "--mock-port", str(port),
                     "--attempt-id", attempt_id,
                     "--out-csv", str(csv_path), "--out-md", str(md_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                return proc, csv_path

            proc_a, csv_a = launch(8910, "session-concurrent-a")
            proc_b, csv_b = launch(8911, "session-concurrent-b")
            _, err_a = proc_a.communicate(timeout=120)
            _, err_b = proc_b.communicate(timeout=120)
            self.assertEqual(proc_a.returncode, 0, err_a)
            self.assertEqual(proc_b.returncode, 0, err_b)

            for csv_path in (csv_a, csv_b):
                with open(csv_path, newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 20)
                for row in rows[1:]:
                    self.assertNotEqual(row["prefill_delta_engine"], "",
                                        f"{csv_path}: {row}")
                    self.assertIn(row["reuse_engaged"], ("True", "False"),
                                 f"{csv_path}: {row}")

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
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--port", "8903",
             "--out-csv", "/tmp/should-not-be-created.csv",
             "--out-md", "/tmp/should-not-be-created.md"],
            env=shim_env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(
            "serve-log is required" in result.stderr or "refusing to start" in result.stderr,
            result.stderr)

    def test_cold_and_warm_are_mutually_exclusive(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--mock", "--cold", "--warm",
             "--out-csv", "/tmp/should-not-be-created2.csv",
             "--out-md", "/tmp/should-not-be-created2.md"],
            capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not allowed with argument", result.stderr)


if __name__ == "__main__":
    unittest.main()
