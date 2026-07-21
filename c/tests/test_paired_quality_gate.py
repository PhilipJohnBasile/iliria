"""tools/paired_quality_gate.py: unit tests.

No engine, no network, no /path/to/models reads. Covers:
  - exact McNemar p-values against hand-computed cases (math.comb by hand;
    the denominators are powers of two, so the expected floats are EXACT --
    no tolerance needed -- both standalone and through the full
    metric_block()/compute_group() pipeline)
  - bootstrap determinism under a fixed seed
  - qid-mismatch / content-mismatch hard errors (align, load_per_item)
  - the sample-size formula against independently-computed textbook
    z-table constants (not against this tool's own norm_ppf)
  - end-to-end CLI (subprocess): --selftest, a real identical-arms run
    built from eval_glm.py's own --dump-per-item, and the hard-error paths
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "paired_quality_gate.py"
EVAL_GLM = ROOT / "tools" / "eval_glm.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pqg = load_module("paired_quality_gate", SCRIPT)


def make_rec(task, qid, gold, lp, lens):
    chosen = max(range(len(lp)), key=lambda i: lp[i])
    return {"task": task, "qid": qid, "gold": gold, "chosen_acc": chosen,
            "chosen_accnorm": chosen, "correct_acc": chosen == gold,
            "correct_accnorm": chosen == gold, "lp_per_option": lp, "option_lengths": lens}


# ---------------------------------------------------------------------------
# exact McNemar: hand-computed cases
# ---------------------------------------------------------------------------

class McNemarExactHandComputedTests(unittest.TestCase):
    def test_hand_computed_cases(self):
        # All denominators below are powers of two (2**n discordant pairs),
        # so binary floating point represents them exactly -- assertEqual,
        # not assertAlmostEqual, is the correct check.
        #   n=10, k=1: p = 2*(C(10,0)+C(10,1))/2**10 = 2*11/1024 = 22/1024
        #   n=10, k=2: p = 2*(C(10,0)+C(10,1)+C(10,2))/2**10 = 2*56/1024 = 112/1024
        #   n=10, k=5: p = 2*638/1024 = 1276/1024 = 1.246... -> capped at 1.0
        #   n=0:       no discordant pairs at all -> defined as p=1.0
        #   n=3, k=0:  p = 2*C(3,0)/2**3 = 2/8 = 0.25
        cases = [
            ((1, 9), 22 / 1024),
            ((9, 1), 22 / 1024),      # symmetric in its two arguments
            ((2, 8), 112 / 1024),
            ((8, 2), 112 / 1024),
            ((5, 5), 1.0),           # sum exceeds 1 before capping
            ((0, 0), 1.0),
            ((0, 3), 0.25),
            ((3, 0), 0.25),
        ]
        for (n1, n2), expected in cases:
            with self.subTest(n1=n1, n2=n2):
                self.assertEqual(pqg.mcnemar_exact_p(n1, n2), expected)

    def test_matches_manual_binomial_sum_for_a_larger_case(self):
        # n=20, k=6: independently re-derived here (not copy-pasted from the
        # implementation) via math.comb, then cross-checked.
        n1, n2 = 6, 14
        n = n1 + n2
        k = min(n1, n2)
        tail = sum(math.comb(n, i) for i in range(k + 1))
        expected = min(1.0, 2.0 * tail / (2 ** n))
        self.assertEqual(pqg.mcnemar_exact_p(n1, n2), expected)


class MetricBlockMcNemarIntegrationTests(unittest.TestCase):
    """The hand-computed McNemar cases above, but exercised through the
    actual per-item pipeline (metric_block), not just the bare function."""

    def test_metric_block_reproduces_hand_computed_p_value(self):
        # 10 items, all discordant: 1 baseline-only-correct + 9 candidate-only-correct.
        a_vals = [1.0] + [0.0] * 9
        b_vals = [0.0] + [1.0] * 9
        block = pqg.metric_block(a_vals, b_vals, resamples=50, seed=1, alpha=0.05, margin=0.9)
        self.assertEqual(block["discordant_baseline_only"], 1)
        self.assertEqual(block["discordant_candidate_only"], 9)
        self.assertEqual(block["mcnemar_p"], 22 / 1024)
        self.assertAlmostEqual(block["delta"], (9 - 1) / 10)

    def test_compute_group_reproduces_hand_computed_p_value(self):
        a_items, b_items = {}, {}
        for i in range(10):
            baseline_correct = (i == 0)          # only item 0: baseline right, candidate wrong
            candidate_correct = (i != 0)          # items 1..9: candidate right, baseline wrong
            a_items[("t", i)] = make_rec("t", i, 0, [-1.0, -2.0] if baseline_correct else [-2.0, -1.0], [1, 1])
            b_items[("t", i)] = make_rec("t", i, 0, [-1.0, -2.0] if candidate_correct else [-2.0, -1.0], [1, 1])
        keys = sorted(a_items)
        group = pqg.compute_group(keys, a_items, b_items, resamples=50, seed=1, alpha=0.05, margin=0.9)
        self.assertEqual(group["acc"]["mcnemar_p"], 22 / 1024)
        self.assertEqual(group["acc_norm"]["mcnemar_p"], 22 / 1024)


# ---------------------------------------------------------------------------
# bootstrap determinism
# ---------------------------------------------------------------------------

class BootstrapDeterminismTests(unittest.TestCase):
    VALUES = [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.05, -0.05]

    def test_same_seed_reproduces_identical_distribution(self):
        d1 = pqg.bootstrap_mean(self.VALUES, n_resamples=500, seed=42)
        d2 = pqg.bootstrap_mean(self.VALUES, n_resamples=500, seed=42)
        self.assertEqual(d1, d2)

    def test_same_seed_reproduces_identical_percentiles(self):
        d1 = pqg.bootstrap_mean(self.VALUES, n_resamples=500, seed=42)
        d2 = pqg.bootstrap_mean(self.VALUES, n_resamples=500, seed=42)
        self.assertEqual(pqg.percentile(d1, 2.5), pqg.percentile(d2, 2.5))
        self.assertEqual(pqg.percentile(d1, 97.5), pqg.percentile(d2, 97.5))

    def test_different_seeds_need_not_match(self):
        d1 = pqg.bootstrap_mean(self.VALUES, n_resamples=500, seed=1)
        d2 = pqg.bootstrap_mean(self.VALUES, n_resamples=500, seed=2)
        self.assertNotEqual(d1, d2)

    def test_constant_values_collapse_bootstrap_to_that_constant(self):
        # every resample -- whatever indices it draws -- averages the same
        # constant, so the ENTIRE distribution must equal it exactly.
        dist = pqg.bootstrap_mean([0.375] * 20, n_resamples=100, seed=7)
        self.assertTrue(all(v == 0.375 for v in dist))

    def test_empty_values_returns_empty_distribution(self):
        self.assertEqual(pqg.bootstrap_mean([], 100, 1), [])


class PercentileTests(unittest.TestCase):
    def test_hand_computed_linear_interpolation(self):
        # k = (4-1)*0.5 = 1.5 -> interpolate between index 1 (2) and 2 (3)
        self.assertEqual(pqg.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(pqg.percentile([10, 20, 30], 0), 10)
        self.assertEqual(pqg.percentile([10, 20, 30], 100), 30)
        self.assertEqual(pqg.percentile([10, 20, 30], 50), 20)

    def test_single_value(self):
        self.assertEqual(pqg.percentile([5.0], 37), 5.0)

    def test_empty_is_nan(self):
        self.assertTrue(math.isnan(pqg.percentile([], 50)))


# ---------------------------------------------------------------------------
# norm_ppf / sample-size formula
# ---------------------------------------------------------------------------

class NormPpfTests(unittest.TestCase):
    def test_matches_textbook_quantiles(self):
        known = {0.5: 0.0, 0.8: 0.8416212335729143, 0.9: 1.2815515655446004,
                0.95: 1.6448536269514722, 0.975: 1.9599639845400545,
                0.99: 2.3263478740408408}
        for p, expected in known.items():
            with self.subTest(p=p):
                self.assertAlmostEqual(pqg.norm_ppf(p), expected, places=6)

    def test_symmetric_about_zero(self):
        self.assertAlmostEqual(pqg.norm_ppf(0.975), -pqg.norm_ppf(0.025), places=9)

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            pqg.norm_ppf(0.0)
        with self.assertRaises(ValueError):
            pqg.norm_ppf(1.0)


class RequiredPairsTests(unittest.TestCase):
    def test_matches_independent_textbook_z_computation(self):
        # z-values read off a standard normal table -- computed here
        # independently of this tool's own norm_ppf -- then plugged into
        # Connor (1987)'s formula by hand.
        z_alpha_05 = 1.6448536269514722    # one-sided alpha=0.05
        z_beta_20 = 0.8416212335729143     # power=80% (beta=0.20)
        z_beta_10 = 1.2815515655446004     # power=90% (beta=0.10)
        psi, margin = 0.1, 0.05
        inner = psi - margin ** 2
        expected_80 = math.ceil((z_alpha_05 * math.sqrt(psi) + z_beta_20 * math.sqrt(inner)) ** 2 / margin ** 2)
        expected_90 = math.ceil((z_alpha_05 * math.sqrt(psi) + z_beta_10 * math.sqrt(inner)) ** 2 / margin ** 2)
        self.assertEqual(expected_80, 246)   # pinned: catches accidental formula drift
        self.assertEqual(expected_90, 339)
        self.assertEqual(pqg.required_pairs(psi, margin, alpha=0.05, beta=0.20), expected_80)
        self.assertEqual(pqg.required_pairs(psi, margin, alpha=0.05, beta=0.10), expected_90)

    def test_zero_discordance_is_none(self):
        self.assertIsNone(pqg.required_pairs(0.0, 0.05, 0.05, 0.20))

    def test_larger_margin_needs_fewer_pairs(self):
        n_tight = pqg.required_pairs(0.1, 0.02, 0.05, 0.20)
        n_loose = pqg.required_pairs(0.1, 0.10, 0.05, 0.20)
        self.assertGreater(n_tight, n_loose)

    def test_higher_power_needs_more_pairs(self):
        n80 = pqg.required_pairs(0.1, 0.05, 0.05, 0.20)
        n90 = pqg.required_pairs(0.1, 0.05, 0.05, 0.10)
        self.assertGreater(n90, n80)

    def test_rejects_nonpositive_margin(self):
        with self.assertRaises(ValueError):
            pqg.required_pairs(0.1, 0.0, 0.05, 0.20)


# ---------------------------------------------------------------------------
# qid-alignment hard errors
# ---------------------------------------------------------------------------

class AlignmentHardErrorTests(unittest.TestCase):
    def test_disjoint_qid_sets_hard_errors(self):
        a = {("t", 0): make_rec("t", 0, 0, [-1, -2], [1, 1])}
        b = {("t", 1): make_rec("t", 1, 0, [-1, -2], [1, 1])}
        with self.assertRaises(SystemExit):
            pqg.align(a, b, "A.jsonl", "B.jsonl")

    def test_extra_item_on_one_side_hard_errors(self):
        a = {("t", 0): make_rec("t", 0, 0, [-1, -2], [1, 1]),
             ("t", 1): make_rec("t", 1, 0, [-1, -2], [1, 1])}
        b = {("t", 0): make_rec("t", 0, 0, [-1, -2], [1, 1])}
        with self.assertRaises(SystemExit):
            pqg.align(a, b, "A.jsonl", "B.jsonl")

    def test_disagreeing_gold_hard_errors(self):
        a = {("t", 0): make_rec("t", 0, 0, [-1, -2], [1, 1])}
        b = {("t", 0): make_rec("t", 0, 1, [-1, -2], [1, 1])}
        with self.assertRaises(SystemExit):
            pqg.align(a, b, "A.jsonl", "B.jsonl")

    def test_disagreeing_option_count_hard_errors(self):
        a = {("t", 0): make_rec("t", 0, 0, [-1, -2], [1, 1])}
        b = {("t", 0): make_rec("t", 0, 0, [-1, -2, -3], [1, 1, 1])}
        with self.assertRaises(SystemExit):
            pqg.align(a, b, "A.jsonl", "B.jsonl")

    def test_identical_question_set_passes_and_is_sorted(self):
        a = {("t", 1): make_rec("t", 1, 1, [-2, -1], [1, 1]),
             ("t", 0): make_rec("t", 0, 0, [-1, -2], [1, 1])}
        b = {("t", 1): make_rec("t", 1, 1, [-2, -1], [1, 1]),
             ("t", 0): make_rec("t", 0, 0, [-1, -2], [1, 1])}
        self.assertEqual(pqg.align(a, b, "A.jsonl", "B.jsonl"), [("t", 0), ("t", 1)])


class LoadPerItemHardErrorTests(unittest.TestCase):
    def test_missing_field_hard_errors(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.jsonl"
            p.write_text(json.dumps({"task": "t", "qid": 0, "gold": 0}) + "\n")
            with self.assertRaises(SystemExit):
                pqg.load_per_item(str(p))

    def test_duplicate_key_hard_errors(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dup.jsonl"
            rec = make_rec("t", 0, 0, [-1, -2], [1, 1])
            p.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
            with self.assertRaises(SystemExit):
                pqg.load_per_item(str(p))

    def test_invalid_json_hard_errors(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "invalid.jsonl"
            p.write_text("{not json\n")
            with self.assertRaises(SystemExit):
                pqg.load_per_item(str(p))

    def test_empty_file_hard_errors(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.jsonl"
            p.write_text("")
            with self.assertRaises(SystemExit):
                pqg.load_per_item(str(p))

    def test_valid_file_loads(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ok.jsonl"
            p.write_text(json.dumps(make_rec("t", 0, 0, [-1, -2], [1, 1])) + "\n")
            items = pqg.load_per_item(str(p))
            self.assertEqual(list(items.keys()), [("t", 0)])


# ---------------------------------------------------------------------------
# CLI end-to-end (subprocess)
# ---------------------------------------------------------------------------

class CLIEndToEndTests(unittest.TestCase):
    def test_selftest_cli(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--selftest"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("selftest OK", result.stdout)
        self.assertIn("VERDICT", result.stdout)

    def test_identical_arms_pass_trivially_and_are_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            dump = Path(td) / "dump.jsonl"
            r = subprocess.run([sys.executable, str(EVAL_GLM), "--snap", "/nonexistent",
                                "--selftest", "--dump-per-item", str(dump)],
                               capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)

            out_json_1 = Path(td) / "gate1.json"
            out_json_2 = Path(td) / "gate2.json"
            cmd = [sys.executable, str(SCRIPT), str(dump), str(dump),
                  "--noninferiority-margin", "0.01", "--quiet"]
            res1 = subprocess.run(cmd + ["--out-json", str(out_json_1)],
                                  capture_output=True, text=True, timeout=30)
            res2 = subprocess.run(cmd + ["--out-json", str(out_json_2)],
                                  capture_output=True, text=True, timeout=30)
            self.assertEqual(res1.returncode, 0, res1.stderr)
            self.assertEqual(res2.returncode, 0, res2.stderr)
            report1 = json.loads(out_json_1.read_text())
            report2 = json.loads(out_json_2.read_text())

        self.assertEqual(report1, report2)          # determinism across processes
        self.assertEqual(report1["verdict"], "PASS")
        self.assertEqual(report1["pooled"]["acc"]["delta"], 0.0)
        self.assertEqual(report1["pooled"]["acc"]["mcnemar_p"], 1.0)
        self.assertEqual(report1["pooled"]["acc"]["discordant_baseline_only"], 0)
        self.assertEqual(report1["pooled"]["acc"]["discordant_candidate_only"], 0)

    def test_qid_mismatch_hard_errors_at_cli(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.jsonl"
            b = Path(td) / "b.jsonl"
            a.write_text(json.dumps(make_rec("t", 0, 0, [-1, -2], [1, 1])) + "\n")
            b.write_text(json.dumps(make_rec("t", 1, 0, [-1, -2], [1, 1])) + "\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(a), str(b), "--noninferiority-margin", "0.05"],
                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("qid mismatch", result.stderr)

    def test_margin_is_required_with_no_default(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.jsonl"
            a.write_text(json.dumps(make_rec("t", 0, 0, [-1, -2], [1, 1])) + "\n")
            result = subprocess.run([sys.executable, str(SCRIPT), str(a), str(a)],
                                    capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--noninferiority-margin", result.stderr)
        self.assertIn("required", result.stderr)

    def test_out_md_file_is_written(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.jsonl"
            a.write_text(json.dumps(make_rec("t", 0, 0, [-1, -2], [1, 1])) + "\n")
            out_md = Path(td) / "gate.md"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(a), str(a),
                 "--noninferiority-margin", "0.05", "--quiet", "--out-md", str(out_md)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")   # --quiet suppresses the stdout echo
            md = out_md.read_text()
        self.assertIn("VERDICT", md)
        self.assertIn("Pooled", md)
        self.assertIn("Task: t", md)

    def test_exit_code_reflects_fail_verdict(self):
        # 40 items, margin so tight relative to sample size that even a
        # slightly-worse candidate cannot be shown noninferior.
        with tempfile.TemporaryDirectory() as td:
            a_items, b_items = [], []
            for i in range(40):
                b_wrong = i < 6   # candidate wrong on 6/40 items baseline gets right
                a_items.append(make_rec("t", i, 0, [-1.0, -2.0], [1, 1]))
                b_items.append(make_rec("t", i, 0, [-2.0, -1.0] if b_wrong else [-1.0, -2.0], [1, 1]))
            a_path, b_path = Path(td) / "a.jsonl", Path(td) / "b.jsonl"
            a_path.write_text("\n".join(json.dumps(r) for r in a_items) + "\n")
            b_path.write_text("\n".join(json.dumps(r) for r in b_items) + "\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(a_path), str(b_path),
                 "--noninferiority-margin", "0.01", "--quiet"],
                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
