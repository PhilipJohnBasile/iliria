"""Direct unit tests for tools/fight_report.py's aggregation and verdict logic: hash
self-consistency gating, the promote/no-harm verdict bars, and the stack-vs-sum-of-parts
interaction table. Builds manifest.json/plan.json/log fixtures directly (no fight_card.sh
subprocess, no engine) so this stays fast and isolates the aggregation math from the shell
harness; c/tests/test_fight_card.py covers the same report generation end to end."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fight_report  # noqa: E402


def log_text(tok_s: float, output_hash: str, hit_pct: float = 70.0) -> str:
    return (
        f"112 tokens in 70.00s ({tok_s:.2f} tok/s) | expert hit rate {hit_pct:.1f}% | RSS 1.0 GB\n"
        f"output token hash: {output_hash}\n"
        f"PROFILE: expert-disk 30.000s | expert-matmul 12.000s | attention 15.000s "
        f"(including kvb 0.0s) | lm_head 1.0s | other 1.0s\n"
    )


class FightReportFixture:
    """Builds a minimal, self-consistent result dir: 2 arms (baseline, stack) x 1 cache x
    1 prompt x N trials, with hashes and tok/s the caller controls per arm."""

    def __init__(self, tmp: Path):
        self.root = tmp
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        self.runs: list[dict] = []
        self.arms: dict[str, dict] = {}
        self.arm_order: list[str] = []
        self.skipped: list[dict] = []
        self.stack_levers = ["ILI_A"]

    def add_arm(self, name: str, kind: str, kernel_family: dict, omits_lever: str | None = None) -> None:
        self.arms[name] = {
            "kind": kind, "container": "c1", "container_path": "/models/c1",
            "env": {}, "kernel_family": kernel_family, "requires_lab_build": False,
        }
        if omits_lever:
            self.arms[name]["omits_lever"] = omits_lever
        if name not in self.arm_order:
            self.arm_order.append(name)

    def add_run(self, arm: str, cache: str, prompt: str, trial: int, tok_s: float, output_hash: str) -> None:
        name = f"{cache}-{prompt}-t{trial}-{arm}.log"
        (self.root / "logs" / name).write_text(log_text(tok_s, output_hash))
        self.runs.append({"arm": arm, "cache_state": cache, "prompt": prompt, "trial": trial, "log": f"logs/{name}"})

    def write(self, passes: int = 1) -> Path:
        manifest = {
            "attempt_id": "test-attempt", "passes": passes, "ngen": 112, "dry_run": True,
            "quiesce_granularity": "per_cell", "runs": self.runs,
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        plan = {
            "arms": self.arms, "arm_order": self.arm_order, "stack_levers": self.stack_levers,
            "skipped": self.skipped,
        }
        (self.root / "plan.json").write_text(json.dumps(plan))
        return self.root


class HashSelfConsistencyTest(unittest.TestCase):
    def test_pass_when_every_trial_of_one_arm_hash_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx = FightReportFixture(Path(td))
            fx.add_arm("baseline", "baseline", {"ILI_A": "0"})
            fx.add_arm("stack", "stack", {"ILI_A": "1"})
            for trial in (1, 2, 3):
                fx.add_run("baseline", "warm", "p", trial, 1.0, "aaaa")
                fx.add_run("stack", "warm", "p", trial, 1.1, "bbbb")
            root = fx.write()
            report = fight_report.build_report(root)
        self.assertIn("PASS -- every arm hash-matched itself", report)

    def test_fail_when_two_trials_of_the_same_arm_disagree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx = FightReportFixture(Path(td))
            fx.add_arm("baseline", "baseline", {"ILI_A": "0"})
            fx.add_arm("stack", "stack", {"ILI_A": "1"})
            fx.add_run("baseline", "warm", "p", 1, 1.0, "aaaa")
            fx.add_run("stack", "warm", "p", 1, 1.1, "bbbb")
            fx.add_run("stack", "warm", "p", 2, 1.1, "cccc")  # same arm/cache/prompt, different hash
            root = fx.write()
            report = fight_report.build_report(root)
        self.assertIn("**FAIL:**", report)
        self.assertIn("stack / warm / p", report)


class PromotionVerdictTest(unittest.TestCase):
    def _two_arm_report(self, baseline_rate: float, stack_rate: float, prompts=("p1", "p2", "p3")) -> str:
        with tempfile.TemporaryDirectory() as td:
            fx = FightReportFixture(Path(td))
            fx.add_arm("baseline", "baseline", {"ILI_A": "0"})
            fx.add_arm("stack", "stack", {"ILI_A": "1"})
            for p in prompts:
                fx.add_run("baseline", "warm", p, 1, baseline_rate, "aaaa")
                fx.add_run("stack", "warm", p, 1, stack_rate, "bbbb")
            root = fx.write()
            return fight_report.build_report(root)

    def test_promote_when_stack_clears_5_percent_on_every_cell(self) -> None:
        report = self._two_arm_report(1.00, 1.10)  # +10% every cell
        self.assertIn("PROMOTE", report)

    def test_keep_provisional_when_below_5_percent(self) -> None:
        report = self._two_arm_report(1.00, 1.02)  # +2%
        self.assertIn("KEEP PROVISIONAL", report)
        self.assertNotIn("PROMOTE (", report)

    def test_revert_when_stack_loses_5_percent_or_more(self) -> None:
        report = self._two_arm_report(1.00, 0.90)  # -10%
        self.assertIn("REVERT/REJECT", report)


class NoHarmVerdictAndInteractionTableTest(unittest.TestCase):
    def _three_arm_fixture(self, stack_rate: float, ablate_rate: float) -> Path:
        td = tempfile.mkdtemp()
        fx = FightReportFixture(Path(td))
        fx.stack_levers = ["ILI_A"]
        fx.add_arm("baseline", "baseline", {"ILI_A": "0"})
        fx.add_arm("stack", "stack", {"ILI_A": "1"})
        fx.add_arm("ablate_ILI_A", "ablation", {"ILI_A": "0"}, omits_lever="ILI_A")
        for p in ("p1", "p2", "p3"):
            fx.add_run("baseline", "warm", p, 1, 1.00, "aaaa")
            fx.add_run("stack", "warm", p, 1, stack_rate, "bbbb")
            fx.add_run("ablate_ILI_A", "warm", p, 1, ablate_rate, "cccc")
        return fx.write()

    def test_no_harm_when_ablation_does_not_beat_stack(self) -> None:
        root = self._three_arm_fixture(stack_rate=1.10, ablate_rate=1.00)  # ablation -9.1% vs stack
        report = fight_report.build_report(root)
        self.assertIn("NO-HARM", report)
        self.assertNotIn("LEVER APPEARS HARMFUL", report)

    def test_flags_harmful_when_ablation_clearly_beats_stack(self) -> None:
        # Removing the lever makes things MUCH better -- it's actively hurting the stack.
        root = self._three_arm_fixture(stack_rate=1.00, ablate_rate=1.20)  # ablation +20% vs stack
        report = fight_report.build_report(root)
        self.assertIn("LEVER APPEARS HARMFUL WITHIN STACK", report)

    def test_interaction_table_reports_additive_when_marginal_equals_stack_delta(self) -> None:
        # baseline=1.00, stack=1.10 (+10% vs baseline); ablation=1.00 (-9.09% vs stack, so
        # the lever's marginal contribution is +9.09%, close enough to the stack's own +10%
        # that the residual sits inside the additive/independent noise band).
        root = self._three_arm_fixture(stack_rate=1.10, ablate_rate=1.00)
        report = fight_report.build_report(root)
        self.assertIn("Sum of marginal contributions:", report)
        self.assertIn("Actual STACK delta vs baseline:", report)
        self.assertIn("additive / independent", report)


class CrossArmHashForkTableTest(unittest.TestCase):
    def test_expected_fork_when_kernel_family_differs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx = FightReportFixture(Path(td))
            fx.add_arm("baseline", "baseline", {"ILI_A": "0"})
            fx.add_arm("stack", "stack", {"ILI_A": "1"})
            fx.add_run("baseline", "warm", "p", 1, 1.0, "aaaa")
            fx.add_run("stack", "warm", "p", 1, 1.1, "bbbb")  # different hash, different kernel family
            root = fx.write()
            report = fight_report.build_report(root)
        self.assertIn("expected fork (kernel-family policy)", report)
        self.assertNotIn("UNEXPECTED FORK", report)

    def test_unexpected_fork_flagged_when_kernel_family_is_identical(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx = FightReportFixture(Path(td))
            fx.add_arm("baseline", "baseline", {"ILI_A": "0"})
            # Same kernel_family fingerprint as baseline, but a divergent hash -- this
            # should never happen for a byte-identical mechanism and must be flagged, not
            # quietly recorded as an "expected" fork.
            fx.add_arm("stack", "stack", {"ILI_A": "0"})
            fx.add_run("baseline", "warm", "p", 1, 1.0, "aaaa")
            fx.add_run("stack", "warm", "p", 1, 1.1, "bbbb")
            root = fx.write()
            report = fight_report.build_report(root)
        self.assertIn("UNEXPECTED FORK", report)

    def test_no_flag_when_hashes_actually_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx = FightReportFixture(Path(td))
            fx.add_arm("baseline", "baseline", {"ILI_A": "0"})
            fx.add_arm("stack", "stack", {"ILI_A": "1"})
            fx.add_run("baseline", "warm", "p", 1, 1.0, "aaaa")
            fx.add_run("stack", "warm", "p", 1, 1.1, "aaaa")  # identical hash despite differing lever
            root = fx.write()
            report = fight_report.build_report(root)
        # Same-hash rows print "-" in the flag column for that pairing.
        self.assertIn("| warm | p | baseline | stack | True | True | - |", report)


class RevivalRowReportingTest(unittest.TestCase):
    def test_ran_revival_row_gets_a_promote_style_verdict_vs_stack(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx = FightReportFixture(Path(td))
            fx.add_arm("baseline", "baseline", {"ILI_A": "0"})
            fx.add_arm("stack", "stack", {"ILI_A": "1"})
            fx.add_arm("revival_x", "revival", {"ILI_A": "1"})
            fx.arms["revival_x"]["display_name"] = "Revival X"
            for p in ("p1", "p2", "p3"):
                fx.add_run("baseline", "warm", p, 1, 1.00, "aaaa")
                fx.add_run("stack", "warm", p, 1, 1.05, "bbbb")
                fx.add_run("revival_x", "warm", p, 1, 1.20, "cccc")  # clear win vs stack
            root = fx.write()
            report = fight_report.build_report(root)
        self.assertIn("Revival X", report)
        self.assertIn("RAN", report)
        self.assertIn("PROMOTE", report)

    def test_skipped_revival_row_reports_its_reason(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx = FightReportFixture(Path(td))
            fx.add_arm("baseline", "baseline", {"ILI_A": "0"})
            fx.add_arm("stack", "stack", {"ILI_A": "1"})
            fx.add_run("baseline", "warm", "p", 1, 1.0, "aaaa")
            fx.add_run("stack", "warm", "p", 1, 1.05, "bbbb")
            fx.skipped.append({"name": "revival_y", "display_name": "Revival Y", "status_ref": "y",
                                "reason": "trigger not met: measured value too high"})
            root = fx.write()
            report = fight_report.build_report(root)
        self.assertIn("Revival Y", report)
        self.assertIn("SKIPPED", report)
        self.assertIn("trigger not met: measured value too high", report)


class ReportWritesFightCardFileTest(unittest.TestCase):
    def test_fight_card_md_is_written_to_the_result_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx = FightReportFixture(Path(td))
            fx.add_arm("baseline", "baseline", {"ILI_A": "0"})
            fx.add_arm("stack", "stack", {"ILI_A": "1"})
            fx.add_run("baseline", "warm", "p", 1, 1.0, "aaaa")
            fx.add_run("stack", "warm", "p", 1, 1.05, "bbbb")
            root = fx.write()
            fight_report.build_report(root)
            self.assertTrue((root / "FIGHT_CARD.md").exists())
            self.assertIn("# FIGHT Card Report", (root / "FIGHT_CARD.md").read_text())


if __name__ == "__main__":
    unittest.main()
