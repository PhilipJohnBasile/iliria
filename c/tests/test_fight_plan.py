"""Direct unit tests for tools/fight_plan.py: card resolution (baseline / stack /
mechanically-derived ablations) and graveyard-revival trigger evaluation. These import the
module directly (no subprocess) so the whole suite stays fast; c/tests/test_fight_card.py
covers the same logic end to end through the real fight_card.sh subprocess."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fight_plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARD = ROOT / "scripts" / "fight_card.default.json"
REAL_PERF_THEORY = ROOT.parent / "docs" / "performance-theory.json"


def tiny_card(**overrides) -> dict:
    card = {
        "containers": {"c1": {"path": "/models/c1"}},
        "default_container": "c1",
        "matrix": {"passes": 3, "ngen": 112, "cache_states": ["warm", "cold"], "hotset_profile": "coding"},
        "quiesce_granularity": "per_cell",
        "fixed_env": {"ILI_FIXED": "x"},
        "levers": {
            "ILI_A": {"shipped_default_value": "1", "off_value": "0", "candidate_value": "1",
                       "requires_lab_build": False, "status_ref": "a", "note": ""},
            "ILI_B": {"shipped_default_value": "0", "off_value": "0", "candidate_value": "1",
                       "requires_lab_build": True, "status_ref": "b", "note": ""},
        },
        "stack": {"levers": ["ILI_A", "ILI_B"], "container": "c1"},
        "revival": [],
    }
    card.update(overrides)
    return card


class ResolveEnvTest(unittest.TestCase):
    def test_add_env_overrides_lever_env_overrides_fixed_env(self) -> None:
        env = fight_plan.resolve_env({"K": "fixed"}, {"K": "lever"})
        self.assertEqual(env["K"], "lever")
        env = fight_plan.resolve_env({"K": "fixed"}, {"K": "lever"}, {"K": "added"})
        self.assertEqual(env["K"], "added")

    def test_all_values_stringified(self) -> None:
        env = fight_plan.resolve_env({}, {"N": 1})
        self.assertEqual(env["N"], "1")
        self.assertIsInstance(env["N"], str)


class KernelFamilyFingerprintTest(unittest.TestCase):
    def test_only_the_four_kernel_family_keys_are_included(self) -> None:
        env = {
            "ILI_METAL_PREFILL": "1", "ILI_METAL4_MOE": "0", "ILI_METAL_PERSISTENT_STATE": "1",
            "ILI_DSA": "0", "ILI_PILOT": "1", "ILI_DRAFT": "1", "ILI_CPU_GROUPED_MOE": "1",
        }
        fp = fight_plan.kernel_family_fingerprint(env)
        self.assertEqual(
            fp, {"ILI_METAL_PREFILL": "1", "ILI_METAL4_MOE": "0",
                 "ILI_METAL_PERSISTENT_STATE": "1", "ILI_DSA": "0"},
        )

    def test_missing_keys_default_to_off(self) -> None:
        self.assertEqual(fight_plan.kernel_family_fingerprint({})["ILI_DSA"], "0")


class EvaluateTriggerTest(unittest.TestCase):
    def test_forced_override_short_circuits_everything(self) -> None:
        met, reason = fight_plan.evaluate_trigger(
            {"kind": "measured_metric", "metric": "x", "comparator": "lt", "threshold": 1.0},
            "row1", measured={}, forced={"row1"}, perf_theory_path=Path("/nonexistent"),
        )
        self.assertTrue(met)
        self.assertIn("forced", reason)

    def test_measured_metric_unmet_without_a_supplied_value_fail_closed(self) -> None:
        met, reason = fight_plan.evaluate_trigger(
            {"kind": "measured_metric", "metric": "miss_bytes_per_token_gb", "comparator": "lt", "threshold": 1.5},
            "row1", measured={}, forced=set(), perf_theory_path=Path("/nonexistent"),
        )
        self.assertFalse(met)
        self.assertIn("no measured value on record", reason)

    def test_measured_metric_met_when_supplied_value_clears_threshold(self) -> None:
        met, _ = fight_plan.evaluate_trigger(
            {"kind": "measured_metric", "metric": "m", "comparator": "lt", "threshold": 1.5},
            "row1", measured={"m": 1.2}, forced=set(), perf_theory_path=Path("/nonexistent"),
        )
        self.assertTrue(met)

    def test_measured_metric_not_met_when_supplied_value_misses_threshold(self) -> None:
        met, reason = fight_plan.evaluate_trigger(
            {"kind": "measured_metric", "metric": "m", "comparator": "lt", "threshold": 1.5},
            "row1", measured={"m": 4.2}, forced=set(), perf_theory_path=Path("/nonexistent"),
        )
        self.assertFalse(met)
        self.assertIn("does not satisfy", reason)

    def test_performance_theory_status_any_of_two_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "pt.json"
            p.write_text(json.dumps({"entries": [
                {"id": "x", "status": {"shipping": {"code": "not_built"}}},
                {"id": "y", "status": {"shipping": {"code": "enabled"}}},
            ]}))
            met, reason = fight_plan.evaluate_trigger(
                {"kind": "performance_theory_status", "watch_ids": ["x", "y"],
                 "requires_shipping_in": ["enabled", "default_on"], "any_or_all": "any"},
                "row1", measured={}, forced=set(), perf_theory_path=p,
            )
        self.assertTrue(met)
        self.assertIn("y=enabled", reason)

    def test_performance_theory_status_not_met_when_all_ids_are_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "pt.json"
            p.write_text(json.dumps({"entries": [
                {"id": "x", "status": {"shipping": {"code": "not_built"}}},
            ]}))
            met, _ = fight_plan.evaluate_trigger(
                {"kind": "performance_theory_status", "watch_ids": ["x"],
                 "requires_shipping_in": ["enabled"], "any_or_all": "any"},
                "row1", measured={}, forced=set(), perf_theory_path=p,
            )
        self.assertFalse(met)

    def test_missing_perf_theory_file_fails_closed(self) -> None:
        met, reason = fight_plan.evaluate_trigger(
            {"kind": "performance_theory_status", "watch_ids": ["x"], "requires_shipping_in": ["enabled"]},
            "row1", measured={}, forced=set(), perf_theory_path=Path("/definitely/not/here.json"),
        )
        self.assertFalse(met)
        self.assertIn("not found", reason)


class BuildPlanTest(unittest.TestCase):
    def test_baseline_uses_shipped_default_stack_uses_candidate(self) -> None:
        plan = fight_plan.build_plan(tiny_card(), {}, set(), Path("/nonexistent"))
        self.assertEqual(plan["arms"]["baseline"]["env"]["ILI_A"], "1")
        self.assertEqual(plan["arms"]["baseline"]["env"]["ILI_B"], "0")
        self.assertEqual(plan["arms"]["stack"]["env"]["ILI_A"], "1")
        self.assertEqual(plan["arms"]["stack"]["env"]["ILI_B"], "1")

    def test_ablation_forces_off_value_not_shipped_default(self) -> None:
        # Regression guard for the exact bug the --dry-run proof run caught: ILI_A's
        # shipped_default_value (1) equals its candidate_value (1), so an ablation that
        # (incorrectly) reset to shipped_default_value would be indistinguishable from the
        # stack row. It must instead reset to off_value (0).
        plan = fight_plan.build_plan(tiny_card(), {}, set(), Path("/nonexistent"))
        ablate_a = plan["arms"]["ablate_ILI_A"]
        self.assertEqual(ablate_a["env"]["ILI_A"], "0")
        self.assertEqual(ablate_a["env"]["ILI_B"], "1")  # untouched
        self.assertNotEqual(ablate_a["env"], plan["arms"]["stack"]["env"])

    def test_ablations_are_mechanically_derived_one_per_stack_lever(self) -> None:
        plan = fight_plan.build_plan(tiny_card(), {}, set(), Path("/nonexistent"))
        ablation_names = {n for n, a in plan["arms"].items() if a["kind"] == "ablation"}
        self.assertEqual(ablation_names, {"ablate_ILI_A", "ablate_ILI_B"})

    def test_needs_lab_build_true_iff_some_running_arm_uses_a_lab_lever_at_candidate(self) -> None:
        plan = fight_plan.build_plan(tiny_card(), {}, set(), Path("/nonexistent"))
        # ILI_B requires_lab_build=True and is candidate(1) in stack and in ablate_ILI_A.
        self.assertTrue(plan["arms"]["stack"]["requires_lab_build"])
        self.assertTrue(plan["arms"]["ablate_ILI_A"]["requires_lab_build"])
        # ablate_ILI_B forces B to off(0) -- no lab-build lever active in that arm.
        self.assertFalse(plan["arms"]["ablate_ILI_B"]["requires_lab_build"])
        self.assertFalse(plan["arms"]["baseline"]["requires_lab_build"])
        self.assertTrue(plan["needs_lab_build"])  # union across all arms

    def test_revival_row_added_only_when_triggered(self) -> None:
        card = tiny_card(revival=[
            {"name": "rev1", "display_name": "Rev One", "status_ref": "s",
             "add_env": {"ILI_C": "1"},
             "trigger": {"kind": "measured_metric", "metric": "m", "comparator": "lt", "threshold": 1.0}},
        ])
        plan_unmet = fight_plan.build_plan(card, {}, set(), Path("/nonexistent"))
        self.assertNotIn("rev1", plan_unmet["arms"])
        self.assertEqual(len(plan_unmet["skipped"]), 1)
        self.assertEqual(plan_unmet["skipped"][0]["name"], "rev1")

        plan_met = fight_plan.build_plan(card, {"m": 0.5}, set(), Path("/nonexistent"))
        self.assertIn("rev1", plan_met["arms"])
        self.assertEqual(plan_met["arms"]["rev1"]["env"]["ILI_C"], "1")
        self.assertEqual(plan_met["arms"]["rev1"]["kind"], "revival")
        self.assertEqual(plan_met["skipped"], [])

    def test_arm_order_lists_baseline_and_stack_first(self) -> None:
        plan = fight_plan.build_plan(tiny_card(), {}, set(), Path("/nonexistent"))
        self.assertEqual(plan["arm_order"][0], "baseline")
        self.assertEqual(plan["arm_order"][1], "stack")


@unittest.skipUnless(DEFAULT_CARD.exists(), "default fight card not found")
class DefaultCardStructureTest(unittest.TestCase):
    """Locks in the shipped default card's own structural invariants (not its specific
    numeric content, which is expected to evolve as levers are promoted/killed)."""

    def test_default_card_is_valid_json_with_required_top_level_keys(self) -> None:
        card = json.loads(DEFAULT_CARD.read_text())
        for key in ("containers", "default_container", "matrix", "fixed_env", "levers", "stack", "revival"):
            self.assertIn(key, card)

    def test_default_card_resolves_against_the_real_performance_theory_json(self) -> None:
        card = json.loads(DEFAULT_CARD.read_text())
        perf_theory = REAL_PERF_THEORY if REAL_PERF_THEORY.exists() else Path("/nonexistent")
        plan = fight_plan.build_plan(card, {}, set(), perf_theory)
        stack_levers = card["stack"]["levers"]
        self.assertEqual(plan["arm_order"][:2], ["baseline", "stack"])
        ablation_names = {n for n, a in plan["arms"].items() if a["kind"] == "ablation"}
        self.assertEqual(ablation_names, {f"ablate_{lv}" for lv in stack_levers})
        # Every declared revival row is accounted for: either it ran (in arms) or it was
        # explicitly skipped with a reason -- never silently dropped.
        revival_declared = {r["name"] for r in card["revival"]}
        revival_ran = {n for n, a in plan["arms"].items() if a["kind"] == "revival"}
        revival_skipped = {s["name"] for s in plan["skipped"]}
        self.assertEqual(revival_declared, revival_ran | revival_skipped)

    def test_ablate_metal_prefill_actually_differs_from_stack(self) -> None:
        # The concrete regression case: ILI_METAL_PREFILL is already shipped default_on, so
        # its shipped_default_value equals its candidate_value -- only off_value correctly
        # removes it from the stack.
        card = json.loads(DEFAULT_CARD.read_text())
        plan = fight_plan.build_plan(card, {}, set(), Path("/nonexistent"))
        stack_env = plan["arms"]["stack"]["env"]
        ablated_env = plan["arms"]["ablate_ILI_METAL_PREFILL"]["env"]
        self.assertNotEqual(stack_env, ablated_env)
        self.assertEqual(ablated_env["ILI_METAL_PREFILL"], "0")
        self.assertEqual(stack_env["ILI_METAL_PREFILL"], "1")


if __name__ == "__main__":
    unittest.main()
