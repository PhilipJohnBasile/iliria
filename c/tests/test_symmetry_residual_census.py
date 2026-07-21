"""n3 falsifier synthetic-recovery proof for
c/tools/symmetry_residual_census.py (SwiGLU intermediate-neuron
permutation-symmetry residual coding, docs/performance-theory.json n3 entry).

No model access, no network, no engine: every test here either operates on
constructed arrays directly or on tiny synthetic int4 containers built the
same way tests/test_container_pipeline.py builds its fixtures. The core
claim under test: given a 'trained' expert that is a permuted-and-noised
copy of a shared per-layer prototype (n3's exact hypothesis about how real
experts might relate to one another), does the signature-based bipartite
aligner RECOVER the planted permutation and does the aligned residual's
entropy drop accordingly -- and, as a falsification-oriented negative
control, does aligning two genuinely UNRELATED experts fail to manufacture
any such improvement?
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

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
ent = load_module("measure_expert_entropy", ROOT / "tools" / "measure_expert_entropy.py")
sym = load_module("symmetry_residual_census", ROOT / "tools" / "symmetry_residual_census.py")


# --------------------------------------------------------------- fixtures --
def make_expert_arrays(hidden, moe_inter, rng, scale=0.3):
    gate = (rng.standard_normal((moe_inter, hidden)) * scale).astype(np.float32)
    up = (rng.standard_normal((moe_inter, hidden)) * scale).astype(np.float32)
    down = (rng.standard_normal((hidden, moe_inter)) * scale).astype(np.float32)
    return gate, up, down


def quantize_expert(gate_f, up_f, down_f):
    gate_p, gate_s = qc.quantize(gate_f, 4)
    up_p, up_s = qc.quantize(up_f, 4)
    down_p, down_s = qc.quantize(down_f, 4)
    Ogu, Igu = gate_f.shape
    Od, Id = down_f.shape
    return {
        "gate_proj": {"nibbles": sym.unpack_matrix(gate_p, Ogu, Igu), "scale": gate_s},
        "up_proj": {"nibbles": sym.unpack_matrix(up_p, Ogu, Igu), "scale": up_s},
        "down_proj": {"nibbles": sym.unpack_matrix(down_p, Od, Id), "scale": down_s},
    }


def planted_pair(hidden, moe_inter, noise_std, perm_seed, val_seed, scale=0.3):
    """Prototype expert + a permuted(+noised) 'trained' expert derived from
    it, plus the ground-truth permutation -- n3's exact hypothesis about how
    a real expert might relate to a common per-layer prototype.

    expert_neuron[k] = prototype_neuron[perm[k]] + noise, so candidate-row k
    stands in for prototype-row perm[k]. The alignment pi (pi[i] = j means
    aligned_row[i] := candidate_row[j]) that recovers this is the INVERSE of
    perm, i.e. np.argsort(perm) -- see align_bipartite's pi[i]=j convention
    and BipartiteMatchingTests.test_recovers_planted_permutation_on_clean_signatures.
    """
    rng_val = np.random.default_rng(val_seed)
    rng_perm = np.random.default_rng(perm_seed)
    gate_f, up_f, down_f = make_expert_arrays(hidden, moe_inter, rng_val, scale)
    perm = rng_perm.permutation(moe_inter)

    def noise(shape):
        return (rng_val.standard_normal(shape) * noise_std).astype(np.float32)

    gate_e = gate_f[perm, :] + noise(gate_f.shape)
    up_e = up_f[perm, :] + noise(up_f.shape)
    down_e = down_f[:, perm] + noise(down_f.shape)
    proto = quantize_expert(gate_f, up_f, down_f)
    expert = quantize_expert(gate_e, up_e, down_e)
    return proto, expert, perm


# ------------------------------------------------------------- gate tests --
class GateTests(unittest.TestCase):
    def test_marker_present(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "marker"
            self.assertFalse(sym.marker_present(str(p)))
            p.write_text("x")
            self.assertTrue(sym.marker_present(str(p)))

    def test_glm_running_false_for_unknown_process(self):
        self.assertFalse(sym.glm_running("definitely-not-a-real-process-xyz123"))

    def test_glm_running_true_when_pgrep_finds_it(self):
        with mock.patch.object(sym.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0)
            self.assertTrue(sym.glm_running("glm"))
            run.assert_called_once_with(
                ["pgrep", "-x", "glm"], stdout=mock.ANY, stderr=mock.ANY)

    def test_shard_reads_allowed_true_if_marker_present_even_if_busy(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "marker"
            p.write_text("x")
            with mock.patch.object(sym.subprocess, "run") as run:
                run.return_value = mock.Mock(returncode=0)  # glm "running"
                self.assertTrue(sym.shard_reads_allowed(str(p), "glm"))

    def test_shard_reads_allowed_false_if_no_marker_and_busy(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "marker"  # never created
            with mock.patch.object(sym.subprocess, "run") as run:
                run.return_value = mock.Mock(returncode=0)
                self.assertFalse(sym.shard_reads_allowed(str(p), "glm"))

    def test_shard_reads_allowed_true_if_not_busy(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "marker"
            self.assertTrue(sym.shard_reads_allowed(
                str(p), "definitely-not-a-real-process-xyz123"))

    def test_require_shard_reads_allowed_exits_when_closed(self):
        with mock.patch.object(sym, "shard_reads_allowed", return_value=False):
            with self.assertRaises(SystemExit) as cm:
                sym.require_shard_reads_allowed(allow_busy=False, wait=False, poll_s=1)
            self.assertEqual(cm.exception.code, 2)

    def test_require_shard_reads_allowed_bypassed_by_allow_busy(self):
        with mock.patch.object(sym, "shard_reads_allowed", return_value=False):
            sym.require_shard_reads_allowed(allow_busy=True, wait=False, poll_s=1)  # no raise

    def test_wait_for_gate_polls_until_open(self):
        calls = {"n": 0}

        def fake_allowed():
            calls["n"] += 1
            return calls["n"] >= 3

        with mock.patch.object(sym, "shard_reads_allowed", side_effect=fake_allowed), \
             mock.patch.object(sym.time, "sleep") as sleep:
            sym.wait_for_gate(poll_s=1)
        self.assertEqual(calls["n"], 3)
        self.assertEqual(sleep.call_count, 2)


# --------------------------------------------------------- nibble utilities
class NibbleUtilTests(unittest.TestCase):
    def test_unpack_matrix_matches_ent_unpack_nibbles(self):
        rng = np.random.default_rng(1)
        w = (rng.standard_normal((6, 16)) * 0.4).astype(np.float32)
        packed, _scale = qc.quantize(w, 4)
        mat = sym.unpack_matrix(packed, 6, 16)
        flat_expected = ent.unpack_nibbles(packed)
        np.testing.assert_array_equal(mat.reshape(-1), flat_expected)
        self.assertEqual(mat.shape, (6, 16))

    def test_pack_nibbles_inverts_unpack(self):
        rng = np.random.default_rng(2)
        nibbles = rng.integers(0, 16, 4096).astype(np.uint8)
        packed = sym.pack_nibbles(nibbles)
        back = ent.unpack_nibbles(packed)
        np.testing.assert_array_equal(back, nibbles)

    def test_nibble_residual_is_invertible(self):
        rng = np.random.default_rng(3)
        proto = rng.integers(0, 16, 2000).astype(np.uint8)
        aligned = rng.integers(0, 16, 2000).astype(np.uint8)
        residual = sym.nibble_residual(proto, aligned)
        self.assertTrue(np.all((residual >= 0) & (residual <= 15)))
        recovered = ((residual.astype(np.int16) + proto.astype(np.int16)) % 16).astype(np.uint8)
        np.testing.assert_array_equal(recovered, aligned)

    def test_nibble_residual_zero_when_identical(self):
        rng = np.random.default_rng(4)
        nib = rng.integers(0, 16, 500).astype(np.uint8)
        residual = sym.nibble_residual(nib, nib)
        self.assertTrue(np.all(residual == 0))


class ScaleDeltaTests(unittest.TestCase):
    def test_zero_delta_is_all_center_code_and_zero_entropy(self):
        scale = np.full(200, 1.5, dtype=np.float32)
        codes = sym.scale_delta_codes(scale, scale)
        self.assertTrue(np.all(codes == 128))
        stats = sym.scale_delta_stats(scale, scale)
        self.assertEqual(stats["h_bits_per_code"], 0.0)

    def test_spread_delta_has_positive_entropy(self):
        rng = np.random.default_rng(5)
        proto_scale = np.full(500, 1.0, dtype=np.float32)
        aligned_scale = (1.0 * np.exp(rng.standard_normal(500) * 0.3)).astype(np.float32)
        stats = sym.scale_delta_stats(proto_scale, aligned_scale)
        self.assertGreater(stats["h_bits_per_code"], 0.5)


# ------------------------------------------------------- signature tests --
class SignatureTests(unittest.TestCase):
    def test_build_projection_shapes(self):
        rng = np.random.default_rng(6)
        proj = sym.build_projection(hidden=40, n_scale_feats=2, proj_dim=64, rng=rng)
        self.assertEqual(proj["gate"].shape, (40, 64))
        self.assertEqual(proj["up"].shape, (40, 64))
        self.assertEqual(proj["down"].shape, (40, 64))
        self.assertEqual(proj["scale"].shape, (2, 64))

    def test_signature_is_exactly_permutation_equivariant(self):
        """Load-bearing correctness property: permuting the neuron axis of
        every input by pi must produce exactly neuron_signatures(orig)[pi]."""
        hidden, moe_inter = 32, 20
        rng = np.random.default_rng(7)
        gate = rng.integers(-8, 8, (moe_inter, hidden)).astype(np.float32)
        up = rng.integers(-8, 8, (moe_inter, hidden)).astype(np.float32)
        down = rng.integers(-8, 8, (hidden, moe_inter)).astype(np.float32)
        gs = rng.uniform(0.1, 2.0, moe_inter).astype(np.float32)
        us = rng.uniform(0.1, 2.0, moe_inter).astype(np.float32)
        proj = sym.build_projection(hidden, 2, 64, rng)

        perm = rng.permutation(moe_inter)
        sig_orig = sym.neuron_signatures(gate, up, down, gs, us, proj)
        sig_perm = sym.neuron_signatures(gate[perm, :], up[perm, :], down[:, perm],
                                         gs[perm], us[perm], proj)
        # atol=1e-6/rtol=0 was tight enough to false-fail on CI's numpy/BLAS build (observed
        # max abs diff ~3.8e-6, max rel diff ~1.5e-5 -- float32 reassociation noise from a
        # different matmul reduction order, not a logic bug: a real non-equivariant signature
        # would miss by orders of magnitude, not parts-per-million).
        np.testing.assert_allclose(sig_perm, sig_orig[perm], rtol=1e-5, atol=1e-5)


class MedoidTests(unittest.TestCase):
    def test_picks_a_central_expert_not_the_outlier(self):
        base = np.zeros((10, 4), dtype=np.float32)
        near = base + 0.01
        outlier = base + 50.0
        sigs = {"a": base, "b": near, "c": outlier}
        self.assertIn(sym.pick_medoid(sigs), ("a", "b"))


# ------------------------------------------------------ bipartite matching
class BipartiteMatchingTests(unittest.TestCase):
    def test_recovers_planted_permutation_on_clean_signatures(self):
        rng = np.random.default_rng(8)
        n, d = 100, 16
        proto_sig = rng.standard_normal((n, d)).astype(np.float32) * 5.0
        perm = rng.permutation(n)
        # cand_sig[k] = proto_sig[perm[k]] + noise, so candidate-row k stands
        # in for prototype-row perm[k]; the alignment pi[i] = j (aligned_row[i]
        # := candidate_row[j]) that recovers this is pi = inverse(perm), i.e.
        # np.argsort(perm) (perm[argsort(perm)[i]] == i by construction).
        cand_sig = proto_sig[perm] + rng.standard_normal((n, d)).astype(np.float32) * 0.05
        cost = sym.pairwise_sqdist(proto_sig, cand_sig)
        align = sym.align_bipartite(cost, refine_rounds=60, rng=rng)
        self.assertTrue(np.array_equal(align["perm"], np.argsort(perm)))
        self.assertLess(align["optimality_gap"], 1e-6)

    def test_refine_never_worsens_greedy(self):
        rng = np.random.default_rng(9)
        n, d = 80, 8
        A = rng.standard_normal((n, d)).astype(np.float32)
        B = rng.standard_normal((n, d)).astype(np.float32)
        cost = sym.pairwise_sqdist(A, B)
        align = sym.align_bipartite(cost, refine_rounds=30, rng=rng)
        self.assertLessEqual(align["cost"], align["greedy_cost"] + 1e-9)

    def test_greedy_assignment_is_a_valid_bijection(self):
        rng = np.random.default_rng(10)
        cost = rng.uniform(0, 1, (50, 50))
        pi = sym.greedy_assignment(cost)
        self.assertEqual(sorted(pi.tolist()), list(range(50)))

    def test_greedy_assignment_completes_quickly_at_real_scale(self):
        """Performance regression guard: the real run uses moe_inter=2048,
        proj_dim=64. The greedy scan over the sorted cost list breaks out
        once all n rows are matched, but a slow/broken implementation that
        walked the full O(n^2) list would make the real 3-layer x 32-64
        expert run impractical within the throttled read window."""
        rng = np.random.default_rng(11)
        n, d = 2048, 64
        A = rng.standard_normal((n, d)).astype(np.float32)
        B = A[rng.permutation(n)] + rng.standard_normal((n, d)).astype(np.float32) * 0.1
        cost = sym.pairwise_sqdist(A, B)
        t0 = time.monotonic()
        pi = sym.greedy_assignment(cost)
        dt = time.monotonic() - t0
        self.assertEqual(sorted(pi.tolist()), list(range(n)))
        self.assertLess(dt, 20.0, f"greedy_assignment took {dt:.1f}s at n={n}")


# -------------------------------------------------------------- verdict --
class VerdictTests(unittest.TestCase):
    @staticmethod
    def make_rows(n, aligned_ratio, indep_ratio, density_all, density_gu, gap=0.01):
        return [{"aligned_stored_ratio": aligned_ratio,
                 "independent_stored_ratio": indep_ratio,
                 "residual_density_all": density_all,
                 "residual_density_gate_up": density_gu,
                 "optimality_gap": gap} for _ in range(n)]

    def test_kill_condition_triggers(self):
        # rel improve = 1-0.70/0.7379 = 5.14% (<10%); density 85% (>70%)
        rows = self.make_rows(20, aligned_ratio=0.70, indep_ratio=0.7379,
                              density_all=0.85, density_gu=0.85)
        v = sym.compute_verdict(rows)
        self.assertTrue(v["kill_condition"])
        self.assertFalse(v["proceed_condition"])
        self.assertIn("KILL", v["kill_line"])

    def test_proceed_via_storage(self):
        rows = self.make_rows(20, aligned_ratio=0.60, indep_ratio=0.7379,
                              density_all=0.5, density_gu=0.5)
        v = sym.compute_verdict(rows)
        self.assertTrue(v["proceed_condition_storage"])
        self.assertTrue(v["proceed_condition"])
        self.assertFalse(v["kill_condition"])  # rel improve 18.7% >= 10%

    def test_proceed_via_compute_saving_in_isolation(self):
        # aligned (0.75) NOT <0.70 and NOT worse than a (deliberately worse)
        # 0.90 independent comparator by less than 10%, so kill's first
        # clause is false regardless of density; gate/up density 0.50 drives
        # a real compute-saving proceed signal in isolation.
        rows = self.make_rows(20, aligned_ratio=0.75, indep_ratio=0.90,
                              density_all=0.75, density_gu=0.50)
        v = sym.compute_verdict(rows)
        self.assertAlmostEqual(v["modeled_work_fraction"], 0.625, places=6)
        self.assertAlmostEqual(v["modeled_compute_saving"], 0.375, places=6)
        self.assertTrue(v["proceed_condition_compute"])
        self.assertTrue(v["proceed_condition"])
        self.assertFalse(v["proceed_condition_storage"])
        self.assertFalse(v["kill_condition"])

    def test_neither_kill_nor_proceed(self):
        rows = self.make_rows(20, aligned_ratio=0.71, indep_ratio=0.7379,
                              density_all=0.60, density_gu=0.80)
        v = sym.compute_verdict(rows)
        self.assertFalse(v["kill_condition"])       # density 60% <= 70%
        self.assertFalse(v["proceed_condition"])    # 0.71 >= 0.70; saving 7.5% <= 15%
        self.assertIn("NOT-KILLED", v["kill_line"])
        self.assertIn("NOT-PROCEED", v["proceed_line"])


# ------------------------------------------ THE synthetic-recovery proof --
class SyntheticRecoveryProofTests(unittest.TestCase):
    """The falsifier's core claim, tested directly against constructed
    arrays (no disk I/O): does aligning a 'trained' expert (a permuted+noised
    copy of a shared prototype) to that prototype recover the planted
    permutation and measurably drop the residual's entropy -- and does
    aligning two UNRELATED experts fail to manufacture any such win?"""

    def test_low_noise_recovers_permutation_and_drops_entropy(self):
        hidden, moe_inter = 64, 128
        rng = np.random.default_rng(100)
        proto, expert, perm = planted_pair(hidden, moe_inter, noise_std=0.01,
                                           perm_seed=1, val_seed=2)
        proj = sym.build_projection(hidden, 2, 64, rng)
        proto_sig = sym.expert_signature(proto, proj)
        exp_sig = sym.expert_signature(expert, proj)
        cost = sym.pairwise_sqdist(proto_sig, exp_sig)
        align = sym.align_bipartite(cost, refine_rounds=60, rng=rng)
        recovered = align["perm"]

        agreement = float((recovered == np.argsort(perm)).mean())
        self.assertGreater(agreement, 0.95, f"only {agreement:.2%} permutation agreement")
        self.assertLess(align["optimality_gap"], 0.05)

        aligned_stats = sym.compute_expert_residual_stats(proto, expert, recovered)
        identity = np.arange(moe_inter)
        unaligned_stats = sym.compute_expert_residual_stats(proto, expert, identity)

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            key = f"{proj_name}_h_residual_bits"
            self.assertLess(aligned_stats[key], unaligned_stats[key],
                            f"{proj_name}: aligned residual entropy did not drop "
                            f"({aligned_stats[key]:.3f} vs unaligned {unaligned_stats[key]:.3f})")
            self.assertLess(aligned_stats[key], sym.N1_MEAN_BITS_PER_WEIGHT * 0.5,
                            f"{proj_name}: aligned entropy not far below the raw "
                            "int4 census baseline")

        self.assertGreater(aligned_stats["exact_zero_rate_all"], 0.3)
        self.assertLess(aligned_stats["residual_density_all"],
                        unaligned_stats["residual_density_all"])

        # end-to-end stored-ratio: aligned beats fresh independent coding
        cfg = {"hidden": hidden, "moe_inter": moe_inter}
        proto_stats = sym.independent_coding_stats(proto)
        indep_stats = sym.independent_coding_stats(expert)
        raw_bytes = qc.expert_total_bytes(
            {"hidden": hidden, "moe_inter": moe_inter}, 4)
        aligned_bytes = sym.expert_aligned_stored_bytes(aligned_stats, proto_stats, 2, cfg)
        self.assertLess(aligned_bytes / raw_bytes,
                        indep_stats["independent_bytes"] / raw_bytes,
                        "aligned stored ratio did not beat independent coding "
                        "on a clean planted-permutation synthetic pair")

    def test_moderate_noise_still_recovers_better_than_chance(self):
        hidden, moe_inter = 64, 128
        rng = np.random.default_rng(101)
        proto, expert, perm = planted_pair(hidden, moe_inter, noise_std=0.15,
                                           perm_seed=3, val_seed=4)
        proj = sym.build_projection(hidden, 2, 64, rng)
        cost = sym.pairwise_sqdist(sym.expert_signature(proto, proj),
                                   sym.expert_signature(expert, proj))
        align = sym.align_bipartite(cost, refine_rounds=60, rng=rng)
        agreement = float((align["perm"] == np.argsort(perm)).mean())
        self.assertGreater(agreement, 0.5,
                           f"only {agreement:.2%} agreement (chance ~= {1 / moe_inter:.4f})")

        aligned_stats = sym.compute_expert_residual_stats(proto, expert, align["perm"])
        identity = np.arange(moe_inter)
        unaligned_stats = sym.compute_expert_residual_stats(proto, expert, identity)
        self.assertLess(aligned_stats["gate_proj_h_residual_bits"],
                        unaligned_stats["gate_proj_h_residual_bits"])

    def test_unrelated_experts_show_no_spurious_improvement(self):
        """Negative control: two INDEPENDENT random experts share no
        structure. 'Aligning' them must not manufacture an entropy win --
        the residual entropy should land close to the mod-16 difference of
        two independent nibble streams, nowhere near the low-noise case's
        near-zero result (agreement/entropy-drop/near-total near-zero rate
        all measured well above 0.9 in
        test_low_noise_recovers_permutation_and_drops_entropy). The
        mod-16 difference of two independent, per-row-abs-max-quantized
        Gaussian nibble streams is NOT perfectly uniform (concentrated
        inputs give a somewhat concentrated difference too), so the bounds
        below are calibrated to this construction's measured baseline
        (empirically ~3.7 bits / ~68% density for this hidden/moe_inter/
        scale combination) rather than the naive uniform-alphabet estimate
        (4 bits / ~81%), with margin on the side that would falsely suggest
        a real win."""
        hidden, moe_inter = 64, 128
        rng = np.random.default_rng(102)
        gate_a, up_a, down_a = make_expert_arrays(hidden, moe_inter, np.random.default_rng(10))
        gate_b, up_b, down_b = make_expert_arrays(hidden, moe_inter, np.random.default_rng(20))
        proto = quantize_expert(gate_a, up_a, down_a)
        expert = quantize_expert(gate_b, up_b, down_b)

        proj = sym.build_projection(hidden, 2, 64, rng)
        cost = sym.pairwise_sqdist(sym.expert_signature(proto, proj),
                                   sym.expert_signature(expert, proj))
        align = sym.align_bipartite(cost, refine_rounds=60, rng=rng)
        aligned_stats = sym.compute_expert_residual_stats(proto, expert, align["perm"])
        identity_stats = sym.compute_expert_residual_stats(
            proto, expert, np.arange(moe_inter))
        self.assertGreater(aligned_stats["gate_proj_h_residual_bits"], 3.5)
        self.assertGreater(aligned_stats["residual_density_all"], 0.5)
        # the real falsification check: matching two unrelated experts must
        # not do meaningfully better than not aligning them at all
        self.assertLess(
            identity_stats["residual_density_all"] - aligned_stats["residual_density_all"],
            0.05,
            "alignment manufactured a density improvement out of unrelated experts")


# ------------------------------------------------------- verdict pure math
class PermutationMetadataTests(unittest.TestCase):
    def test_matches_ceil_log2_times_n(self):
        self.assertEqual(sym.permutation_metadata_bits(2048), 11 * 2048)
        self.assertEqual(sym.permutation_metadata_bits(256), 8 * 256)
        self.assertEqual(sym.permutation_metadata_bits(257), 9 * 257)


# ---------------------------------------------------- disk end-to-end test
# routed layers 3..27 == exactly BANDS' "early" range (3, 27), so
# --limit-layers 1 always lands on an EXISTING layer regardless of the
# seeded rng draw inside main().
TINY_CFG = {"hidden_size": 24, "moe_intermediate_size": 16, "num_hidden_layers": 28,
           "first_k_dense_replace": 3, "n_routed_experts": 6}


def make_tiny_container(root: Path, rng: np.random.Generator) -> None:
    """A per-layer prototype + permuted/noised 'trained' experts at every
    candidate 'early'-band layer (3..27), so the test doesn't need to
    predict which specific layer main()'s seeded rng will draw."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(TINY_CFG))
    hidden, moe_inter = TINY_CFG["hidden_size"], TINY_CFG["moe_intermediate_size"]
    n_experts = TINY_CFG["n_routed_experts"]
    tensors = {}
    for layer in range(TINY_CFG["first_k_dense_replace"], TINY_CFG["num_hidden_layers"]):
        gate_f, up_f, down_f = make_expert_arrays(hidden, moe_inter, rng)
        for e in range(n_experts):
            perm = rng.permutation(moe_inter)
            noise_std = 0.02
            gate_e = gate_f[perm, :] + (rng.standard_normal(gate_f.shape) * noise_std).astype(np.float32)
            up_e = up_f[perm, :] + (rng.standard_normal(up_f.shape) * noise_std).astype(np.float32)
            down_e = down_f[:, perm] + (rng.standard_normal(down_f.shape) * noise_std).astype(np.float32)
            for proj_name, w in (("gate_proj", gate_e), ("up_proj", up_e), ("down_proj", down_e)):
                name = f"model.layers.{layer}.mlp.experts.{e}.{proj_name}.weight"
                packed, scale = qc.quantize(w, 4)
                tensors[name] = packed
                tensors[name + ".qs"] = scale
    qc.st_save(root / "out-00000.safetensors", tensors)


class CliIntegrationTest(unittest.TestCase):
    def test_main_end_to_end_on_synthetic_container(self):
        rng = np.random.default_rng(999)
        with tempfile.TemporaryDirectory() as d:
            model_dir = Path(d) / "tiny_model"
            make_tiny_container(model_dir, rng)
            outdir = Path(d) / "out"
            rc = sym.main([
                "--model", str(model_dir), "--outdir", str(outdir),
                "--n-experts-per-layer", "6", "--proj-dim", "32",
                "--refine-rounds", "20", "--seed", "42",
                "--limit-layers", "1", "--allow-busy",
            ])
            self.assertEqual(rc, 0)
            summary_path = outdir / "n3-residual-summary.json"
            csv_path = outdir / "n3-residual-experts.csv"
            self.assertTrue(summary_path.exists())
            self.assertTrue(csv_path.exists())
            summary = json.loads(summary_path.read_text())
            self.assertEqual(len(summary["bands"]), 1)
            self.assertEqual(summary["bands"][0]["band"], "early")
            self.assertIn("kill_line", summary["headline"])
            self.assertIn("proceed_line", summary["headline"])
            self.assertEqual(summary["headline"]["n_experts_total"], 5)  # 6 experts - 1 prototype
            csv_lines = csv_path.read_text().strip().split("\n")
            self.assertEqual(len(csv_lines), 1 + 5)  # header + 5 follower rows

            # report-only mode must reproduce the same verdict, zero shard reads
            rc2 = sym.main(["--model", str(model_dir), "--outdir", str(outdir),
                           "--report-only"])
            self.assertEqual(rc2, 0)
            summary2 = json.loads(summary_path.read_text())
            self.assertAlmostEqual(
                summary2["headline"]["median_aligned_stored_ratio"],
                summary["headline"]["median_aligned_stored_ratio"], places=9)

    def test_main_refuses_when_gate_closed(self):
        with tempfile.TemporaryDirectory() as d:
            model_dir = Path(d) / "tiny_model"
            make_tiny_container(model_dir, np.random.default_rng(1000))
            outdir = Path(d) / "out"
            with mock.patch.object(sym, "shard_reads_allowed", return_value=False):
                with self.assertRaises(SystemExit) as cm:
                    sym.main(["--model", str(model_dir), "--outdir", str(outdir)])
                self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
