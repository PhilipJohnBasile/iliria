"""Mixed-precision container pipeline: measure -> allocate -> build roundtrip
on tiny synthetic containers (no model access, no network, no engine)."""

from __future__ import annotations

import importlib.util
import json
import os
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
measure = load_module("measure_expert_quant_error", ROOT / "tools" / "measure_expert_quant_error.py")
alloc_mod = load_module("allocate_bit_budget", ROOT / "tools" / "allocate_bit_budget.py")
build_mod = load_module("build_mixed_container", ROOT / "tools" / "build_mixed_container.py")

# tiny geometry: routed layers 3..6, 4 experts each = 16 routed experts
CFG = {"hidden_size": 8, "moe_intermediate_size": 4, "num_hidden_layers": 7,
       "first_k_dense_replace": 3, "n_routed_experts": 4}


def expert_weights(layer, expert, rng):
    """f32 weights for one expert; (4,0) lies exactly on the int2 grid."""
    out = {}
    for proj in qc.PROJS:
        O, I = (8, 4) if proj == "down_proj" else (4, 8)
        if (layer, expert) == (4, 0):
            q = rng.integers(-1, 2, size=(O, I)).astype(np.float32)
            q[:, 0] = 1.0  # every row reaches amax = qmax*s -> exact int2
            out[proj] = q * 0.5
        else:
            out[proj] = (rng.standard_normal((O, I)) * 0.3).astype(np.float32)
    return out


def make_container(root: Path, rng) -> dict:
    """Two-shard synthetic int4 container + config + usage histograms."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(CFG))
    weights = {}
    shards = {"out-00000.safetensors": [3, 4], "out-00001.safetensors": [5, 6]}
    for fn, layers in shards.items():
        tensors = {}
        for layer in layers:
            for expert in range(4):
                w = expert_weights(layer, expert, rng)
                weights[(layer, expert)] = w
                for proj in qc.PROJS:
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    packed, scales = qc.quantize(w[proj], 4)
                    tensors[name] = packed
                    tensors[name + ".qs"] = scales
        qc.st_save(root / fn, tensors)
    # non-expert passthrough tensors in a third shard
    dense = (rng.standard_normal((4, 8)) * 0.2).astype(np.float32)
    packed, scales = qc.quantize(dense, 8)
    qc.st_save(root / "out-00002.safetensors", {
        "model.norm.weight": np.ones(8, np.float32),
        "model.layers.0.mlp.gate_proj.weight": packed,
        "model.layers.0.mlp.gate_proj.weight.qs": scales,
    })
    usage = ["# fa_hotset_v1"]
    for layer in range(3, 7):
        for expert in range(4):
            usage.append(f"{layer} {expert} {10 * layer + expert + 1}")
    (root / ".fa_usage").write_text("\n".join(usage) + "\n")
    (root / ".fa_usage.coding").write_text("\n".join(usage) + "\n")
    return weights


class QuantDequantTest(unittest.TestCase):
    def test_dequant_matches_known_nibbles(self):
        w = np.array([[1.0, -2.0, 3.0, -4.0]], np.float32)
        packed, scales = qc.quantize(w, 4)
        s = np.float32(4.0) / np.float32(7.0)
        self.assertAlmostEqual(float(scales[0]), float(s), places=6)
        # rint(w/s) in float32 = [2, -3, 5, -7] (-2/s = -3.4999998, not -3.5,
        # exactly like C lrintf); nibbles stored v+8, low nibble = even index
        self.assertEqual([int(b) for b in packed],
                         [(2 + 8) | ((-3 + 8) << 4), (5 + 8) | ((-7 + 8) << 4)])
        back = qc.dequant(packed, scales, 1, 4, 4)
        np.testing.assert_allclose(back, np.array([[2, -3, 5, -7]]) * s, rtol=1e-6)

    def test_int2_packing_order(self):
        w = np.array([[1.0, -1.0, 0.0, 1.0, -1.0]], np.float32)
        packed, scales = qc.quantize(w, 2)
        self.assertEqual(float(scales[0]), 1.0)
        # values+2 packed 4/byte, LSB-first: [3,1,2,3] -> 0b11100111; [1] -> 0b01
        self.assertEqual(list(packed), [0b11100111, 0b00000001])
        back = qc.dequant(packed, scales, 1, 5, 2)
        np.testing.assert_allclose(back, w)

    def test_requantization_is_idempotent(self):
        rng = np.random.default_rng(7)
        w = (rng.standard_normal((6, 16)) * 0.4).astype(np.float32)
        for bits in (2, 3, 4, 8):
            p1, s1 = qc.quantize(w, bits)
            back = qc.dequant(p1, s1, 6, 16, bits)
            p2, s2 = qc.quantize(back, bits)
            np.testing.assert_array_equal(p1, p2)
            np.testing.assert_allclose(s1, s2, rtol=1e-6)

    def test_infer_bits_and_sizes(self):
        self.assertEqual(qc.infer_bits(4 * 8, 4, 8), 8)
        self.assertEqual(qc.infer_bits(4 * 4, 4, 8), 4)
        self.assertEqual(qc.infer_bits(4 * 2, 4, 8), 2)
        with self.assertRaises(ValueError):
            qc.infer_bits(7, 4, 8)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / "tiny_i4"
        self.weights = make_container(self.model, np.random.default_rng(42))
        self.cfg = qc.load_config(self.model)
        self.b4 = qc.expert_total_bytes(self.cfg, 4)
        self.b2 = qc.expert_total_bytes(self.cfg, 2)

    def tearDown(self):
        self.tmp.cleanup()

    def run_measure(self, out, extra=()):
        rc = measure.main(["--model", str(self.model), "--out", str(out),
                           "--max-mb-s", "0", "--allow-busy", *extra])
        self.assertEqual(rc, 0)

    def read_csv(self, path):
        lines = Path(path).read_text().strip().split("\n")
        header = lines[0].split(",")
        rows = {}
        for line in lines[1:]:
            parts = line.split(",")
            rows[(int(parts[0]), int(parts[1]))] = dict(zip(header, parts))
        return header, rows

    def test_measure_end_to_end(self):
        csv = self.root / "err.csv"
        self.run_measure(csv)
        header, rows = self.read_csv(csv)
        self.assertEqual(len(rows), 16)
        self.assertIn("err_int2", header)
        for key, row in rows.items():
            self.assertEqual(row["src_bits"], "4")
            self.assertEqual(int(row["bytes_src"]), self.b4)
            self.assertEqual(int(row["bytes_int2"]), self.b2)
            self.assertGreaterEqual(float(row["err_int2"]), 0.0)
        # the on-grid expert reconstructs exactly; a random one does not
        self.assertAlmostEqual(float(rows[(4, 0)]["err_int2"]), 0.0, places=6)
        self.assertGreater(float(rows[(3, 1)]["err_int2"]), 0.01)
        # usage columns recorded, never folded into the error
        self.assertEqual(int(rows[(3, 1)]["usage_general"]), 32)
        self.assertAlmostEqual(float(rows[(3, 1)]["share_general"]),
                               32 / (31 + 32 + 33 + 34), places=5)

    def test_measure_resume_and_limit(self):
        csv = self.root / "err.csv"
        self.run_measure(csv, ["--limit", "5"])
        _, rows = self.read_csv(csv)
        self.assertEqual(len(rows), 5)
        self.run_measure(csv)  # resume completes without duplicating
        text = Path(csv).read_text().strip().split("\n")
        self.assertEqual(len(text), 1 + 16)
        _, rows = self.read_csv(csv)
        self.assertEqual(len(rows), 16)

    def test_measure_layer_filter(self):
        csv = self.root / "err.csv"
        self.run_measure(csv, ["--layers", "3"])
        _, rows = self.read_csv(csv)
        self.assertEqual(sorted({k[0] for k in rows}), [3])
        self.assertEqual(len(rows), 4)

    # ------------------------------------------------------------ allocator
    def synth_errors(self):
        """error grows with (layer, expert) index -> deterministic ranking"""
        return {(l, e): 0.01 * (4 * (l - 3) + e + 1)
                for l in range(3, 7) for e in range(4)}

    def test_allocator_budget_and_pinning(self):
        errors = self.synth_errors()
        forced = {3}
        all_int4 = 16 * self.b4  # nonexpert = 0
        target = all_int4 - 5 * (self.b4 - self.b2)  # forces 5 demotions
        alloc = alloc_mod.allocate(self.cfg, errors, forced, target, 0.0)
        self.assertTrue(alloc["feasible"])
        self.assertLessEqual(alloc["size_bytes"], target)
        self.assertEqual(alloc["n_int2"], 5)
        for l, e in alloc["int2"]:
            self.assertNotIn(l, forced)          # forced layers stay int4
        # greedy demotes the LOWEST-error demotable experts (layer 4 first)
        self.assertEqual(alloc["int2"], [[4, 0], [4, 1], [4, 2], [4, 3], [5, 0]]
                         if isinstance(alloc["int2"][0], list)
                         else [(4, 0), (4, 1), (4, 2), (4, 3), (5, 0)])
        self.assertLessEqual(alloc["max_demoted_err"], alloc["min_kept_err"])

    def test_allocator_monotone_in_target(self):
        errors = self.synth_errors()
        small = alloc_mod.allocate(self.cfg, errors, {3},
                                   16 * self.b4 - 9 * (self.b4 - self.b2), 0.0)
        large = alloc_mod.allocate(self.cfg, errors, {3},
                                   16 * self.b4 - 2 * (self.b4 - self.b2), 0.0)
        self.assertLess(large["n_int2"], small["n_int2"])
        self.assertTrue(set(map(tuple, large["int2"]))
                        <= set(map(tuple, small["int2"])))

    def test_allocator_infeasible_and_missing(self):
        errors = self.synth_errors()
        alloc = alloc_mod.allocate(self.cfg, errors, {3}, 1.0, 0.0)
        self.assertFalse(alloc["feasible"])      # 1 byte can never fit
        with self.assertRaises(SystemExit):
            alloc_mod.allocate(self.cfg, {}, {3}, 16 * self.b4, 0.0)
        partial = alloc_mod.allocate(self.cfg, {}, {3}, 16 * self.b4, 0.0,
                                     allow_partial=True)
        self.assertEqual(partial["n_missing"], 12)

    def test_simulate_bytes_lru(self):
        sizes = {(0, e): 10 for e in range(4)}
        requests = {0: [1, 2, 1, 3, 1, 2]}
        # budget 20 holds two experts; expert 0 pinned never requested
        n, hits, misses, mb = alloc_mod.simulate_bytes(requests, sizes, set(), 20)
        self.assertEqual((n, hits, misses, mb), (6, 2, 4, 40))
        # pinning expert 3 removes its miss
        n, hits, misses, mb = alloc_mod.simulate_bytes(requests, sizes, {(0, 3)}, 20)
        self.assertEqual((n, hits, misses, mb), (6, 4, 2, 20))

    # ------------------------------------------------------- build roundtrip
    def test_measure_allocate_build_roundtrip(self):
        csv = self.root / "err.csv"
        self.run_measure(csv)
        manifest = self.root / "manifest.json"
        all_int4 = 16 * self.b4
        target_gb = (all_int4 - 6 * (self.b4 - self.b2)) / 1e9
        rc = alloc_mod.main([
            "--csv", str(csv), "--model", str(self.model),
            "--nonexpert-gb", "0", "--force-int4-layers", "3",
            "--targets", f"{target_gb:.12g}",
            "--emit", str(manifest), "--emit-gb", f"{target_gb:.12g}"])
        self.assertEqual(rc, 0)
        m = json.loads(manifest.read_text())
        self.assertEqual(m["default_bits"], 4)
        self.assertEqual(len(m["int2"]), 6)
        int2 = {tuple(k) for k in m["int2"]}
        self.assertIn((4, 0), int2)              # the exact-grid expert is cheapest

        outdir = self.root / "tiny_mixed"
        rc = build_mod.main(["--indir", str(self.model), "--outdir", str(outdir),
                             "--manifest", str(manifest), "--min-free-gb", "0",
                             "--allow-busy"])
        self.assertEqual(rc, 0)

        src_idx = qc.st_scan(str(self.model))
        out_idx = qc.st_scan(str(outdir))
        self.assertEqual(set(src_idx), set(out_idx))
        for (layer, expert) in [(l, e) for l in range(3, 7) for e in range(4)]:
            for proj in qc.PROJS:
                name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                O, I = qc.expert_shape(self.cfg, proj)
                bits = 2 if (layer, expert) in int2 else 4
                self.assertEqual(out_idx[name]["nbytes"], qc.packed_nbytes(O, I, bits),
                                 name)
                self.assertEqual(out_idx[name + ".qs"]["nbytes"], O * 4)
                if bits == 2:  # requantized bytes match the reference math
                    w4 = qc.dequant(qc.st_read_tensor(src_idx[name]),
                                    qc.st_read_tensor(src_idx[name + ".qs"]), O, I, 4)
                    ref_p, ref_s = qc.quantize(w4, 2)
                    np.testing.assert_array_equal(qc.st_read_tensor(out_idx[name]), ref_p)
                    np.testing.assert_allclose(qc.st_read_tensor(out_idx[name + ".qs"]),
                                               ref_s, rtol=1e-6)
                else:          # kept experts byte-identical
                    np.testing.assert_array_equal(qc.st_read_tensor(out_idx[name]),
                                                  qc.st_read_tensor(src_idx[name]))
        # non-expert tensors byte-identical, metadata copied
        np.testing.assert_array_equal(
            qc.st_read_tensor(out_idx["model.layers.0.mlp.gate_proj.weight"]),
            qc.st_read_tensor(src_idx["model.layers.0.mlp.gate_proj.weight"]))
        self.assertTrue((outdir / "config.json").exists())
        self.assertTrue((outdir / ".fa_usage").exists())
        self.assertTrue((outdir / "mixed-container-manifest.json").exists())

        # verify mode agrees; re-measuring the mixed container: int2 experts
        # reconstruct their own int2 exactly (idempotence), so err_int2 == 0
        rc = build_mod.main(["--indir", str(self.model), "--outdir", str(outdir),
                             "--manifest", str(manifest), "--verify-only"])
        self.assertEqual(rc, 0)
        csv2 = self.root / "err2.csv"
        rc = measure.main(["--model", str(outdir), "--out", str(csv2),
                           "--max-mb-s", "0", "--allow-busy"])
        self.assertEqual(rc, 0)
        _, rows = self.read_csv(csv2)
        for key in int2:
            self.assertEqual(rows[key]["src_bits"], "2")
            self.assertAlmostEqual(float(rows[key]["err_int2"]), 0.0, places=6)

        # resume: a second build run rewrites nothing and still verifies
        rc = build_mod.main(["--indir", str(self.model), "--outdir", str(outdir),
                             "--manifest", str(manifest), "--min-free-gb", "0",
                             "--allow-busy"])
        self.assertEqual(rc, 0)

    def test_build_refuses_same_dir(self):
        manifest = self.root / "m.json"
        manifest.write_text(json.dumps({"version": 1, "default_bits": 4, "int2": []}))
        with self.assertRaises(SystemExit):
            build_mod.main(["--indir", str(self.model), "--outdir", str(self.model),
                            "--manifest", str(manifest), "--allow-busy"])


if __name__ == "__main__":
    unittest.main()
