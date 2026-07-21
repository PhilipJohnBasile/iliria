"""Activation-level quantizer verdict tool: synthetic-fixture validation.

No engine access, no network, no real captured activations (those are
"produced by tomorrow's engine runs" per tools/eval_activation_error.py's
module docstring -- this test builds its own tiny .npy fixture matching that
exact contract, so the tool is validated end to end without the capture
hook existing yet).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qc = load_module("quant_container", ROOT / "tools" / "quant_container.py")
ev = load_module("eval_activation_error", ROOT / "tools" / "eval_activation_error.py")

# tiny geometry: one routed layer (3), two experts, wide enough rows that the
# fitted quantizer's grid search reliably beats the defective one (mirrors
# tests/test_quant_noise_floor.py's use of a wide I to keep the comparison
# from being a coin flip on a tiny sample).
CFG = {"hidden_size": 16, "moe_intermediate_size": 32, "num_hidden_layers": 4,
       "first_k_dense_replace": 3, "n_routed_experts": 2}


def make_container(root: Path, rng) -> dict:
    """One-shard synthetic int4 container: layer 3, experts 0 and 1."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(CFG))
    weights = {}
    tensors = {}
    for expert in range(2):
        w = {}
        for proj in qc.PROJS:
            O, I = qc.expert_shape(CFG_ADAPTER(), proj)
            w[proj] = (rng.standard_normal((O, I)) * 0.3).astype(np.float32)
        weights[expert] = w
        for proj in qc.PROJS:
            name = f"model.layers.3.mlp.experts.{expert}.{proj}.weight"
            packed, scales = qc.quantize(w[proj], 4)
            tensors[name] = packed
            tensors[name + ".qs"] = scales
    qc.st_save(root / "out-00000.safetensors", tensors)
    return weights


def CFG_ADAPTER():
    """qc.expert_shape wants the internal {"hidden","moe_inter"} key names."""
    return {"hidden": CFG["hidden_size"], "moe_inter": CFG["moe_intermediate_size"]}


class EvalActivationErrorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.container = self.root / "container"
        self.activations = self.root / "activations"
        self.activations.mkdir()
        self.rng = np.random.default_rng(0)
        self.weights = make_container(self.container, self.rng)

    def write_activation(self, layer, expert, n, hidden=None, dtype=np.float32):
        X = self.rng.standard_normal((n, hidden if hidden is not None else CFG["hidden_size"])).astype(dtype)
        np.save(self.activations / f"L{layer}_E{expert}.npy", X)
        return X

    def run_eval(self, out="err.csv", **extra):
        argv = ["--container", str(self.container), "--activations", str(self.activations),
                "--out", str(self.root / out), "--allow-busy"]
        for k, v in extra.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        rc = ev.main(argv)
        self.assertEqual(rc, 0, f"eval_activation_error exited {rc}")
        return self.root / out

    def test_discovers_files_matching_the_npy_contract_only(self):
        self.write_activation(3, 0, 5)
        (self.activations / "not_a_match.npy").write_bytes(b"")
        (self.activations / "L3_E0.txt").write_bytes(b"")
        found = ev.discover_activation_files(str(self.activations))
        self.assertEqual(set(found), {(3, 0)})

    def test_end_to_end_csv_has_expected_columns_and_finite_errors(self):
        self.write_activation(3, 0, 8)
        self.write_activation(3, 1, 1)  # N=1 edge case must also work
        out = self.run_eval()
        lines = out.read_text().strip().splitlines()
        header = lines[0].split(",")
        self.assertEqual(header, [
            "layer", "expert", "n_samples", "hidden", "moe_inter",
            "err_gate_defective", "err_up_defective", "err_down_defective", "err_layer_defective",
            "err_gate_fitted", "err_up_fitted", "err_down_fitted", "err_layer_fitted",
        ])
        rows = {int(r.split(",")[1]): r.split(",") for r in lines[1:]}
        self.assertEqual(set(rows), {0, 1})
        row0 = rows[0]
        self.assertEqual(row0[0], "3")
        self.assertEqual(row0[2], "8")   # n_samples
        self.assertEqual(row0[3], "16")  # hidden
        self.assertEqual(row0[4], "32")  # moe_inter
        errs = [float(x) for x in row0[5:]]
        self.assertTrue(all(np.isfinite(e) and e >= 0 for e in errs), errs)
        row1 = rows[1]
        self.assertEqual(row1[2], "1")   # N=1 edge case did not crash

    def test_fitted_quantizer_beats_defective_on_layer_output_error(self):
        self.write_activation(3, 0, 16)
        out = self.run_eval()
        row = out.read_text().strip().splitlines()[1].split(",")
        header = out.read_text().strip().splitlines()[0].split(",")
        idx = {name: i for i, name in enumerate(header)}
        err_layer_defective = float(row[idx["err_layer_defective"]])
        err_layer_fitted = float(row[idx["err_layer_fitted"]])
        self.assertLess(
            err_layer_fitted, err_layer_defective,
            "the repaired (fitted-scale) int2 quantizer should produce a materially "
            "smaller layer-output error than the defective one on real activations, "
            "matching the weight-Frobenius result in tests/test_quant_noise_floor.py")

    def test_variants_flag_restricts_to_one_quantizer(self):
        self.write_activation(3, 0, 4)
        out = self.run_eval(variants="defective")
        header = out.read_text().splitlines()[0].split(",")
        self.assertNotIn("err_layer_fitted", header)
        self.assertIn("err_layer_defective", header)

    def test_resumable_skips_already_scored_experts(self):
        self.write_activation(3, 0, 4)
        out = self.run_eval(out="resume.csv")
        first = out.read_text()
        self.write_activation(3, 1, 4)
        ev.main(["--container", str(self.container), "--activations", str(self.activations),
                "--out", str(out), "--allow-busy"])
        second = out.read_text()
        self.assertTrue(second.startswith(first.splitlines()[0] + "\n" + first.splitlines()[1]))
        self.assertEqual(len(second.strip().splitlines()), 3)  # header + 2 experts

    def test_wrong_hidden_dimension_is_a_hard_error_not_a_silent_skip(self):
        self.write_activation(3, 0, 4, hidden=CFG["hidden_size"] + 1)
        with self.assertRaises(SystemExit):
            self.run_eval()

    def test_manifest_mismatch_is_a_hard_error(self):
        self.write_activation(3, 0, 4)
        (self.activations / "manifest.json").write_text(json.dumps({"hidden": 999}))
        with self.assertRaises(SystemExit):
            self.run_eval()

    def test_manifest_matching_hidden_is_accepted(self):
        self.write_activation(3, 0, 4)
        (self.activations / "manifest.json").write_text(
            json.dumps({"hidden": CFG["hidden_size"], "note": "smoke"}))
        out = self.run_eval()  # must not raise
        self.assertTrue(out.exists())

    def test_missing_expert_in_container_is_a_hard_error(self):
        self.write_activation(3, 5, 4)  # expert 5 does not exist in the tiny container
        with self.assertRaises(SystemExit):
            self.run_eval()

    def test_empty_activations_dir_is_a_clean_noop(self):
        rc = ev.main(["--container", str(self.container), "--activations", str(self.activations),
                     "--out", str(self.root / "empty.csv"), "--allow-busy"])
        self.assertEqual(rc, 0)
        self.assertFalse((self.root / "empty.csv").exists())


if __name__ == "__main__":
    unittest.main()
