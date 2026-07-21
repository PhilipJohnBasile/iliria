"""End-to-end --dry-run tests for scripts/fight_card.sh: runs the REAL shell harness (real
card resolution via tools/fight_plan.py, real ABBA-generalized row sequencing, real .fa_usage
snapshot/restore, real hash extraction, real tools/fight_report.py) against
scripts/fight_mock_engine.sh (no glm, no real engine) so the whole pipeline is proven without
a model or hardware. quiesce_check.sh itself is replaced by a trivial fixture (same
convention as c/tests/test_evening_orchestrator.py's ILI_EVENING_QUIESCE_BIN override): the
real quiesce_check.sh reads live system telemetry and takes tens of real seconds per call,
which would make these tests slow and flaky (pass/fail would depend on this machine's actual
state at test-run time, not the scenario under test).

No real engine command (`make`, `build-m5max-lab.sh`, `run-m5max-fast.sh`, `glm`) is ever
invoked by this suite -- every test passes --dry-run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fight_card.sh"
DEFAULT_CARD = ROOT / "scripts" / "fight_card.default.json"

FAST_QUIESCE = """#!/usr/bin/env bash
echo "[quiesce-fixture] instant pass (test fixture, not real telemetry)"
echo "[quiesce] ALL CONDITIONS PASS -- timing gate may begin"
exit 0
"""


def write_fixture(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


class FightCardEndToEndTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.quiesce_bin = write_fixture(self.tmp / "fast_quiesce.sh", FAST_QUIESCE)
        self.result_dir = self.tmp / "result"
        self.campaign_log = self.tmp / "campaign-log.md"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_fight_card(self, extra_args=None, extra_env=None, passes="1", cache_states="warm"):
        env = dict(os.environ)
        env.update({
            "ILI_FIGHT_QUIESCE_BIN": str(self.quiesce_bin),
            "ILI_FIGHT_RESULT_DIR": str(self.result_dir),
            "ILI_FIGHT_ATTEMPT_ID": "test-attempt-1",
            "ILI_FIGHT_CAMPAIGN_LOG": str(self.campaign_log),
            "ILI_FIGHT_PASSES": passes,
            "ILI_FIGHT_CACHE_STATES": cache_states,
        })
        if extra_env:
            env.update(extra_env)
        args = ["--dry-run"] + (extra_args or [])
        return subprocess.run(
            ["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=120,
        )


@unittest.skipUnless(sys.platform == "darwin", "fight_card.sh targets macOS")
class DefaultCardDryRunTest(unittest.TestCase):
    """One shared subprocess run (setUpClass, not per-test setUp) covering the shipped
    default card end to end; each test method below asserts on a different facet of that
    SAME run's output, instead of paying the ~15s subprocess cost three times over."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.quiesce_bin = write_fixture(cls.tmp / "fast_quiesce.sh", FAST_QUIESCE)
        cls.result_dir = cls.tmp / "result"
        cls.campaign_log = cls.tmp / "campaign-log.md"
        env = dict(os.environ)
        env.update({
            "ILI_FIGHT_QUIESCE_BIN": str(cls.quiesce_bin),
            "ILI_FIGHT_RESULT_DIR": str(cls.result_dir),
            "ILI_FIGHT_ATTEMPT_ID": "test-attempt-1",
            "ILI_FIGHT_CAMPAIGN_LOG": str(cls.campaign_log),
            "ILI_FIGHT_PASSES": "1",
            "ILI_FIGHT_CACHE_STATES": "warm",
        })
        cls.proc = subprocess.run(
            ["bash", str(SCRIPT), "--dry-run"], env=env, capture_output=True, text=True, timeout=120,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_exits_clean_and_produces_every_expected_artifact(self) -> None:
        self.assertEqual(self.proc.returncode, 0, msg=f"stdout:\n{self.proc.stdout}\nstderr:\n{self.proc.stderr}")
        for name in ("plan.json", "manifest.json", "FIGHT_CARD.md", "report-console.txt"):
            self.assertTrue((self.result_dir / name).exists(), f"missing {name}")

    def test_resolved_plan_has_baseline_stack_and_one_ablation_per_lever(self) -> None:
        plan = json.loads((self.result_dir / "plan.json").read_text())
        stack_levers = plan["stack_levers"]
        self.assertEqual(plan["arm_order"][0], "baseline")
        self.assertEqual(plan["arm_order"][1], "stack")
        self.assertEqual(
            {a for a in plan["arm_order"] if a.startswith("ablate_")},
            {f"ablate_{lv}" for lv in stack_levers},
        )

    def test_both_graveyard_revival_rows_are_skipped_with_a_reason_today(self) -> None:
        # Both triggers are unmet today (real docs/performance-theory.json, no
        # --set-measurement / --force-revival given) -- must be SKIPPED, not silently
        # dropped, and each must carry a legible reason.
        plan = json.loads((self.result_dir / "plan.json").read_text())
        skipped_names = {s["name"] for s in plan["skipped"]}
        self.assertEqual(skipped_names, {"revival_mtp_draft1", "revival_pilot_k6"})
        for s in plan["skipped"]:
            self.assertTrue(s["reason"])

    def test_manifest_runs_embed_attempt_id_in_every_log_path(self) -> None:
        manifest = json.loads((self.result_dir / "manifest.json").read_text())
        self.assertEqual(manifest["attempt_id"], "test-attempt-1")
        self.assertGreater(len(manifest["runs"]), 0)
        for r in manifest["runs"]:
            self.assertTrue((self.result_dir / r["log"]).exists())
            # attempt_id must be embedded in every artifact filename (campaign-state.json
            # convention, per roofline_run.sh / run_abba_matrix.sh's own comments).
            self.assertIn(manifest["attempt_id"], r["log"])

    def test_report_contains_hash_consistency_verdicts_and_interaction_table(self) -> None:
        fight_card_md = (self.result_dir / "FIGHT_CARD.md").read_text()
        self.assertIn("PASS -- every arm hash-matched itself", fight_card_md)
        self.assertIn("## Promotion verdicts", fight_card_md)
        self.assertIn("## Interaction: stack vs sum-of-parts", fight_card_md)
        self.assertIn("SKIPPED", fight_card_md)

    def test_campaign_log_gets_a_timestamped_attempt_stamped_line_per_step(self) -> None:
        # campaign-state.json convention -- see evening_orchestrator.sh's own campaign_log().
        log_text = self.campaign_log.read_text()
        self.assertIn("test-attempt-1", log_text)
        self.assertIn("fight_card starting", log_text)
        self.assertIn("fight_card complete", log_text)

    def test_quiesce_fixture_was_actually_invoked_not_bypassed(self) -> None:
        system_dir = self.result_dir / "system"
        quiesce_files = list(system_dir.glob("*quiesce*"))
        self.assertTrue(quiesce_files, "expected at least one quiesce artifact under system/")
        preflight = system_dir / "test-attempt-1-quiesce-preflight.txt"
        self.assertTrue(preflight.exists())
        self.assertIn("quiesce-fixture", preflight.read_text())

    def test_lab_build_decision_is_logged_but_never_actually_invoked_under_dry_run(self) -> None:
        # needs_lab_build is True for the shipped default (METAL4_MOE/PERSISTENT_STATE are
        # stack members) -- --dry-run must still LOG that decision without ever running make
        # or build-m5max-lab.sh.
        self.assertIn("would run 'ILI_M5_LAB_METAL4=1", self.proc.stdout)
        self.assertFalse((self.result_dir / "lab-build.log").exists())


@unittest.skipUnless(sys.platform == "darwin", "fight_card.sh targets macOS")
class RevivalTriggerOverrideTest(FightCardEndToEndTestBase):
    def test_force_revival_and_set_measurement_make_both_graveyard_rows_run(self) -> None:
        proc = self.run_fight_card(extra_args=[
            "--set-measurement", "miss_bytes_per_token_gb=1.2",
            "--force-revival", "revival_pilot_k6",
        ])
        self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        plan = json.loads((self.result_dir / "plan.json").read_text())
        self.assertEqual(plan["skipped"], [])
        self.assertIn("revival_mtp_draft1", plan["arms"])
        self.assertIn("revival_pilot_k6", plan["arms"])
        self.assertEqual(plan["arms"]["revival_mtp_draft1"]["env"]["ILI_DRAFT"], "1")
        self.assertEqual(plan["arms"]["revival_pilot_k6"]["env"]["ILI_PILOT"], "1")

        fight_card_md = (self.result_dir / "FIGHT_CARD.md").read_text()
        self.assertIn("MTP DRAFT=1 (graveyard revival)", fight_card_md)
        self.assertIn("PILOT K6 (graveyard revival)", fight_card_md)
        self.assertNotIn("SKIPPED", fight_card_md)  # both ran; nothing left to skip


@unittest.skipUnless(sys.platform == "darwin", "fight_card.sh targets macOS")
class CustomCardTest(FightCardEndToEndTestBase):
    """Proves --card works against a hand-rolled, minimal card, not just the shipped
    default -- i.e. the harness is genuinely config-driven, not hardwired to one file."""

    def _write_tiny_card(self) -> Path:
        card = {
            "schema_version": 1,
            "description": "test fixture",
            "containers": {"only": {"path": str(self.tmp / "fake-model")}},
            "default_container": "only",
            "matrix": {"passes": 1, "ngen": 8, "cache_states": ["warm"], "hotset_profile": "coding"},
            "quiesce_granularity": "per_cell",
            "fixed_env": {"ILI_FIXED_ONE": "yes"},
            "levers": {
                "ILI_ONLY_LEVER": {
                    "shipped_default_value": "0", "off_value": "0", "candidate_value": "1",
                    "requires_lab_build": False, "status_ref": "test", "note": "",
                },
            },
            "stack": {"levers": ["ILI_ONLY_LEVER"], "container": "only"},
            "revival": [],
        }
        path = self.tmp / "tiny_card.json"
        path.write_text(json.dumps(card))
        return path

    def test_single_lever_card_produces_exactly_baseline_stack_and_one_ablation(self) -> None:
        card = self._write_tiny_card()
        proc = self.run_fight_card(extra_args=["--card", str(card)])
        self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        plan = json.loads((self.result_dir / "plan.json").read_text())
        self.assertEqual(set(plan["arm_order"]), {"baseline", "stack", "ablate_ILI_ONLY_LEVER"})
        self.assertFalse(plan["needs_lab_build"])


@unittest.skipUnless(sys.platform == "darwin", "fight_card.sh targets macOS")
class ArgParsingTest(FightCardEndToEndTestBase):
    def test_missing_card_path_fails_fast(self) -> None:
        proc = self.run_fight_card(extra_args=["--card", str(self.tmp / "does-not-exist.json")])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("fight card not found", proc.stderr)
        self.assertFalse(self.result_dir.exists())

    def test_help_flag_prints_usage_and_exits_zero(self) -> None:
        proc = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Usage:", proc.stdout)

    def test_unknown_flag_is_rejected(self) -> None:
        proc = subprocess.run(["bash", str(SCRIPT), "--bogus-flag"], capture_output=True, text=True, timeout=20)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("unknown argument", proc.stderr)

    def test_malformed_set_measurement_fails_fast(self) -> None:
        proc = self.run_fight_card(extra_args=["--set-measurement", "no-equals-sign"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse((self.result_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
