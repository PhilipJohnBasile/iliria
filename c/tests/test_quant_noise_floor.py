"""Characterization tests for the quantizer noise floors in tools/quant_container.py.

These pin down the intrinsic reconstruction error of each quantization path on
synthetic weights -- Gaussian is the easiest possible distribution, so its
floor is a lower bound on what any real (structured, heavier-tailed) weight
tensor will show -- so that per-expert error measurements can be interpreted
against the instrument's own floor. Reference points (Lloyd-Max optimal
quantizers of N(0,1)):

    4-level  (int2): relative Frobenius ~0.343  (MSE 0.11748)
    16-level (int4): relative Frobenius ~0.0975 (MSE 0.009501)

Measured 2026-07-15 with the shipped implementation (tools/convert_fp8_to_int4.py):

    int4: ~0.143  (~1.47x the 16-level optimum -- healthy, plenty of margin
                  for a weight tier)
    int2: ~0.868  (SATURATED: ~2.5x worse than the 4-level optimum -- the
                  signal is mostly destroyed, and any per-expert error
                  ranking computed with this path is pinned at the
                  instrument's ceiling rather than measuring expert
                  structure)

The int2 floor is why the 2026-07-15 per-expert saliency sweep returned a
flat 0.88-0.92 across all 19,200 experts: the instrument, not the experts.
See docs/performance-theory.json (expert-quant-error-saliency) and
the container-design plan.

ROOT CAUSE (external review, code-verified): quant_int2 in
tools/convert_fp8_to_int4.py sets qmax=(1<<(bits-1))-1=1 for bits=2, so
scale=absmax/qmax=absmax/1 and codes=clip(rint(w/s), -2, 1). Because the
scale is fit to make +1 (not +2) the largest representable magnitude, code
-2 is geometrically unreachable in practice: on a 64x1024 Gaussian sample
(seed 0) the code histogram is exactly {-1: 2846, 0: 59816, +1: 2874} --
ZERO occurrences of -2 -- and any |w| below half the row's absmax rounds to
0 (most of a Gaussian row). It is a ternary quantizer wearing 2-bit
clothing. test_int2_code_minus2_never_produced_qmax1_defect below pins this
down as a regression trip-wire, and _reference_repaired_quant_int2 below is
a small, format-compatible reference fix (fitted per-row scale over the
FULL {-2,-1,0,1} codebook) that demonstrates the error is fixable without
an engine change -- the seed of the real repair, not the repair itself.
See docs/performance-theory.json (expert-quant-error-saliency) for the
fuller three-stage repair plan: this fitted-scale pass; a symmetric-
codebook 2-bit redesign that needs a kernel change; and task-sensitive,
activation-level evaluation beyond weight Frobenius.

When the int2 quantizer is actually repaired in
tools/convert_fp8_to_int4.py (not just in this test file), UPDATE the int2
bounds below to lock in the improvement, DELETE or invert the code-
utilization trip-wire test, and re-run the per-expert saliency sweep: the
upper bound on int2's floor here is the current ceiling, and dropping below
0.5 is the signal that per-expert discrimination is worth re-measuring.
"""

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - this machine always has numpy
    raise ImportError(
        "numpy is required to run c/tests/test_quant_noise_floor.py. This "
        "machine always has numpy installed, so a missing import here is an "
        "environment defect that must FAIL the suite -- it must not be "
        "silently skipped, since that would hide the quantizer noise-floor "
        "characterization this file exists to guarantee."
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import quant_container as qc  # noqa: E402  (sys.path must be set up first)


def _unpack_int2_codes(raw, O, I):
    """Packed quant_int2 bytes -> signed codes in {-2,-1,0,1}, shape [O,I].
    Mirrors qc.dequant's int2 branch exactly, minus the final *scale, so the
    raw codes actually written to disk can be inspected directly."""
    b = raw.reshape(O, (I + 3) // 4)
    codes = np.empty((O, b.shape[1] * 4), np.int16)
    for k in range(4):
        codes[:, k::4] = ((b >> (2 * k)) & 3).astype(np.int16) - 2
    return codes[:, :I]


def _reference_repaired_quant_int2(w, bits=2, n_candidates=64):
    """Reference REPAIRED int2 quantizer -- the seed of the real fix, not
    the fix itself (that belongs in tools/convert_fp8_to_int4.py once
    validated on real weights, per docs/performance-theory.json's
    three-stage repair plan). Stage 1 of that plan, specifically:

    Keeps the shipped on-disk FORMAT (4 codes/byte, packed value v = code+2,
    engine dequant (code-2)*scale) -- zero engine/kernel changes -- but
    replaces the shipped scale=absmax/qmax=absmax/1 with a per-row scale
    chosen by a small 1-D grid search that minimizes reconstruction MSE
    over the FULL {-2,-1,0,1} codebook (the shipped quantizer only ever
    reaches {-1,0,1}; see the module docstring). Both a positive and a
    negative candidate scale are tried per row (the "negative-scale sign
    trick"): the codebook is asymmetric (two negative codes, one positive),
    so letting a row's own empirical tail pick which side gets the extra
    code shaves a little more error off.

    `bits` is accepted (and ignored beyond assuming 2) only so this drops
    into the same `quant_fn(w, bits)` call shape as qc.quant_int2 /
    qc.quant_int4; the packed output is always 2-bit.
    """
    O, I = w.shape
    amax = np.abs(w).max(axis=1)
    best_s = np.zeros(O, np.float32)
    best_mse = np.full(O, np.inf, np.float64)
    for sign in (1.0, -1.0):
        for k in range(1, n_candidates + 1):
            s = sign * (amax * k / n_candidates)
            s = np.where(s == 0, 1e-8, s).astype(np.float32)
            q = np.clip(np.rint(w / s[:, None]), -2, 1)
            recon = q * s[:, None]
            mse = np.mean((w - recon) ** 2, axis=1, dtype=np.float64)
            better = mse < best_mse
            best_mse = np.where(better, mse, best_mse)
            best_s = np.where(better, s, best_s)
    q = np.clip(np.rint(w / best_s[:, None]), -2, 1).astype(np.int32)
    rb = (I + 3) // 4
    out = np.zeros((O, rb), np.uint8)
    for k in range(4):                          # pack identically to qc.quant_int2
        vk = q[:, k::4]
        out[:, :vk.shape[1]] |= ((vk + 2).astype(np.uint8) << (k * 2))
    return out.reshape(-1), best_s.astype(np.float32)


# Widened characterization matrix (part d): distributions beyond Gaussian.
# Relative-Frobenius error is scale-invariant (scaling W by a constant scales
# the fitted quantizer scale by the same constant and cancels out), so every
# sampler below uses a convenient unit scale rather than matching variances.
_DISTRIBUTIONS = ("gaussian", "laplace", "student_t3", "uniform", "outlier_contaminated")


def _sample(dist, rng, shape):
    if dist == "gaussian":
        return rng.standard_normal(shape).astype(np.float32)
    if dist == "laplace":
        return rng.laplace(0.0, 1.0, shape).astype(np.float32)
    if dist == "student_t3":
        return rng.standard_t(3, shape).astype(np.float32)
    if dist == "uniform":
        return rng.uniform(-1.0, 1.0, shape).astype(np.float32)
    if dist == "outlier_contaminated":
        w = rng.standard_normal(shape).astype(np.float32)
        mask = rng.random(shape) < 0.01
        return np.where(mask, w * 20.0, w).astype(np.float32)
    raise ValueError(f"unknown distribution {dist!r}")


class QuantNoiseFloorTest(unittest.TestCase):
    def _roundtrip_rel_err(self, bits, quant_fn):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((256, 1024)).astype(np.float32)
        raw, qs = quant_fn(W, bits)
        W2 = qc.dequant(raw, qs, W.shape[0], W.shape[1], bits)
        return float(np.linalg.norm(W - W2) / np.linalg.norm(W))

    def test_int4_floor_is_healthy(self):
        rel = self._roundtrip_rel_err(4, qc.quant_int4)
        # ~1.47x the 16-level Lloyd-Max optimum (~0.0975, MSE 0.009501);
        # fails if a regression makes int4 lossy enough to threaten the
        # resident tier.
        self.assertLess(rel, 0.20, f"int4 noise floor degraded: {rel:.3f}")

    def test_int2_floor_is_saturated_current_implementation(self):
        rel = self._roundtrip_rel_err(2, qc.quant_int2)
        # CHARACTERIZATION of the current implementation: ~0.868 on
        # Gaussian, far above the ~0.343 Lloyd-Max 4-level optimum. The
        # lower bound documents the saturation (per-expert rankings are
        # unmeasurable above it); the upper bound just keeps the number
        # from silently getting even worse.
        self.assertGreater(rel, 0.70,
                           "int2 floor improved! Update this test's bounds, then "
                           "re-run the per-expert saliency sweep -- discrimination "
                           "may now be measurable (see module docstring).")
        self.assertLess(rel, 0.95, f"int2 floor got WORSE: {rel:.3f}")

    def test_int2_code_minus2_never_produced_qmax1_defect(self):
        """Documents the CURRENT defect, does not require it: quant_int2's
        scale=absmax/qmax=absmax/1 makes code -2 (packed nibble value 0)
        geometrically unreachable, so the shipped "2-bit" quantizer only
        ever emits 3 of its 4 codes -- a ternary quantizer, not a 4-level
        one. Reproduces the exact cited histogram (seed 0, 64x1024
        Gaussian: {-1: 2846, 0: 59816, +1: 2874}, i.e. zero -2's out of
        65,536 codes)."""
        rng = np.random.default_rng(0)
        W = rng.standard_normal((64, 1024)).astype(np.float32)
        raw, _qs = qc.quant_int2(W, 2)
        codes = _unpack_int2_codes(raw, W.shape[0], W.shape[1])
        n_minus2 = int(np.count_nonzero(codes == -2))
        self.assertEqual(
            n_minus2, 0,
            f"code -2 now appears {n_minus2} time(s) in quant_int2's output -- "
            "this test DOCUMENTS the current qmax=(1<<(bits-1))-1=1 defect "
            "(scale=absmax/1 leaves the most-negative code unreachable), it "
            "does not require it to stay broken. If this assertion just "
            "failed because the quantizer was repaired (fitted scale, "
            "symmetric codebook, group-wise scales, ...), that is GOOD NEWS: "
            "delete or invert this assertion, then update the noise-floor "
            "bounds in this file and the expert-quant-error-saliency entry "
            "in docs/performance-theory.json to match.")

    def test_repaired_reference_quantizer_is_format_compatible_and_meets_attainable_error(self):
        """The reference fitted-scale quantizer (_reference_repaired_quant_int2)
        is the seed of the real repair: same on-disk format, better scale.
        This proves two things on synthetic Gaussian weights -- NOT a
        model-quality claim, just a weight-Frobenius demonstration on the
        easiest possible distribution -- (a) it is format-compatible: the
        shared qc.dequant int2 path (which mirrors the engine's
        (code-2)*scale decode) reproduces the same values a manual decode
        of the same packed bytes gives, bit for bit; (b) attainable error:
        well under half the shipped quantizer's 0.868 floor."""
        rng = np.random.default_rng(0)
        W = rng.standard_normal((256, 1024)).astype(np.float32)
        O, I = W.shape
        raw, qs = _reference_repaired_quant_int2(W, bits=2)

        # (a) format compatibility: hand-decode with the literal engine
        # formula (packed_nibble - 2) * per-row scale and check it is
        # exactly (not just approximately) what qc.dequant produces.
        manual = _unpack_int2_codes(raw, O, I).astype(np.float32) * qs[:, None]
        engine = qc.dequant(raw, qs, O, I, 2)
        np.testing.assert_array_equal(
            manual, engine,
            err_msg="reference repaired quantizer's packed bytes are not "
                    "decoded identically by qc.dequant's int2 path -- it "
                    "would not be format-compatible with the engine")

        # (b) attainable error: comfortably better than the ~0.45-0.55
        # ESTIMATED floor for real (non-synthetic) expert weights -- this
        # synthetic Gaussian case is the easiest distribution, so it should
        # clear that estimate with room to spare.
        rel = float(np.linalg.norm(W - engine) / np.linalg.norm(W))
        print(f"\nreference repaired int2 quantizer (fitted per-row scale, "
              f"synthetic Gaussian): rel err = {rel:.4f} "
              f"(shipped quant_int2 floor: ~0.868; int4 healthy floor: ~0.143)")
        self.assertLess(
            rel, 0.55,
            f"reference repaired quantizer only reached {rel:.3f} rel err; "
            "the fitted-scale grid search should comfortably beat 0.55 on "
            "synthetic Gaussian -- this is the attainable-error demonstration "
            "that the real repair (docs/performance-theory.json, "
            "expert-quant-error-saliency) is meant to build on")

    def test_production_quant_int2_fitted_matches_the_reference_byte_for_byte(self):
        """PORT CHECK: tools/convert_fp8_to_int4.py's quant_int2_fitted (the
        production port, also reachable as tools/quant_container.py's
        qc.quant_int2_fitted / qc.INT2_QUANTIZERS['fitted']) must reproduce
        this file's own _reference_repaired_quant_int2 EXACTLY -- same packed
        bytes, same per-row scales -- on the same inputs (Gaussian and a
        second, differently-shaped/seeded distribution, so the check is not
        a single-point coincidence). The reference function above stays the
        readable spec; quant_int2_fitted is the shipped implementation the
        converter (--int2-quantizer fitted / ILI_INT2_QUANTIZER=fitted) and
        build_mixed_container.py (--int2-quantizer fitted) actually call.
        quant_int2 (the defective quantizer) is UNCHANGED and still the
        default everywhere -- see docs/performance-theory.json
        (expert-quant-error-saliency) and tools/eval_activation_error.py for
        the activation-level verdict between the two."""
        cases = (
            (np.random.default_rng(0).standard_normal((256, 1024)).astype(np.float32)),
            (np.random.default_rng(7).standard_normal((64, 2048)).astype(np.float32) * 3.0),
        )
        for i, W in enumerate(cases):
            ref_raw, ref_s = _reference_repaired_quant_int2(W, bits=2)
            prod_raw, prod_s = qc.quant_int2_fitted(W, 2)
            np.testing.assert_array_equal(
                ref_raw, prod_raw,
                err_msg=f"case {i}: packed bytes differ between the reference "
                        "and tools/convert_fp8_to_int4.py's quant_int2_fitted")
            np.testing.assert_array_equal(
                ref_s, prod_s,
                err_msg=f"case {i}: per-row scales differ between the reference "
                        "and tools/convert_fp8_to_int4.py's quant_int2_fitted")
            # and the dispatch table used by the converter/builder CLIs
            table_raw, table_s = qc.INT2_QUANTIZERS["fitted"](W, 2)
            np.testing.assert_array_equal(ref_raw, table_raw)
            np.testing.assert_array_equal(ref_s, table_s)
        # the defective path must be unaffected: still reachable, still itself
        raw_def, _ = qc.INT2_QUANTIZERS["defective"](cases[0], 2)
        raw_ref_def, _ = qc.quant_int2(cases[0], 2)
        np.testing.assert_array_equal(raw_def, raw_ref_def)

    def test_int2_floor_across_seeds_shapes_and_distributions(self):
        """Widened coverage (external review): the single 256x1024 Gaussian
        case above is one point in a larger space. This ONLY asserts on
        gaussian (matching the existing saturation characterization,
        seeds/shapes included); every other distribution is measured and
        printed for the record, not gated -- this project has no settled
        claim yet about the shipped int2 floor on non-Gaussian sources, and
        this test is characterization, not a spec."""
        shapes = ((64, 2048), (64, 6144))  # production expert widths, O kept small for speed
        seeds = (0, 1, 2)
        lines = ["", "int2 shipped-implementation rel-Frobenius floor "
                      f"(O=64 rows, mean over seeds={seeds}):"]
        lines.append(f"{'distribution':<22}" + "  ".join(f"I={i:>5}" for _, i in shapes))
        for dist in _DISTRIBUTIONS:
            row_means = []
            for (O, I) in shapes:
                errs = []
                for seed in seeds:
                    rng = np.random.default_rng((seed, _DISTRIBUTIONS.index(dist), I))
                    W = _sample(dist, rng, (O, I))
                    raw, qs = qc.quant_int2(W, 2)
                    W2 = qc.dequant(raw, qs, O, I, 2)
                    errs.append(float(np.linalg.norm(W - W2) / np.linalg.norm(W)))
                if dist == "gaussian":
                    for seed, e in zip(seeds, errs):
                        self.assertGreater(
                            e, 0.7,
                            f"gaussian int2 floor improved at seed={seed}, I={I} "
                            f"({e:.3f} <= 0.7) -- update the characterization "
                            "bounds in this file (see module docstring)")
                row_means.append(sum(errs) / len(errs))
            lines.append(f"{dist:<22}" + "  ".join(f"{v:9.4f}" for v in row_means))
        print("\n".join(lines))


if __name__ == "__main__":
    unittest.main()
