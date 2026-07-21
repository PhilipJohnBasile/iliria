"""Mixed int4/int2 routed-expert Metal MoE regression test.

Builds the tiny mixed-format fixture (tiny_mixed_moe_fixture.py: a GLM-5.2-shaped
model whose router bias GUARANTEES every decode step routes to a mix of int4 and
int2 experts, plus an on-the-fly-quantized int8 shared expert) and runs the REAL,
unmodified `glm` binary's SCORE mode CPU-only vs Metal-enabled, asserting the two
agree within a tolerance appropriate for float reassociation differences between a
serial CPU accumulation and a parallel GPU reduction.

This is the end-to-end check for the mixed-format block-builder guard in glm.c's
moe(): before that guard existed, the block builder took its Metal dispatch format
from the FIRST expert in a 64-expert union only, so a block containing (say) two
int4 experts and two int2 experts would submit the WHOLE block as one scalar fmt --
silently decoding int2 bytes as int4 nibbles (or vice versa). That failure mode
produces wildly wrong values (a different bit-packing, not a rounding difference),
so it would blow this test's tolerance by orders of magnitude, not a few percent --
this test does not need to be more precise than that to be a meaningful guard.

Skips cleanly when the glm binary, Metal, or numpy is unavailable.
"""

import importlib.util
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
GLM = HERE / "glm"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

if HAVE_NUMPY:
    fixture = _load_module(
        "tiny_mixed_moe_fixture", str(Path(__file__).resolve().parent / "tiny_mixed_moe_fixture.py"))


def _build_requests(n=8, seed=7):
    """n small byte-token (ctx, cont) requests: SCORE mode's request-file format."""
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        ctx = [rng.randint(0, 255) for _ in range(6)]
        cont = [rng.randint(0, 255) for _ in range(4)]
        full = ctx + cont
        lines.append(f"{len(ctx)} {len(cont)} " + " ".join(map(str, full)))
    return lines


def _run_score(model_dir, req_path, metal):
    """Invoke the real engine's SCORE mode; returns (per-request logprobs, stderr)."""
    env = dict(os.environ, SNAP=str(model_dir), SCORE=str(req_path))
    if metal:
        env["ILI_METAL"] = "1"
    else:
        env.pop("ILI_METAL", None)
    proc = subprocess.run([str(GLM), "32", "8", "8"], env=env,
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"engine failed (metal={metal}), rc={proc.returncode}:\n{proc.stderr[-4000:]}")
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln and ln[0] in "-0123456789"]
    return [float(ln.split()[0]) for ln in lines], proc.stderr


@unittest.skipUnless(GLM.exists(), "glm binary not built")
@unittest.skipUnless(HAVE_NUMPY, "numpy not available")
class MixedFormatMoeTest(unittest.TestCase):
    N_REQUESTS = 8
    REL_TOL = 0.05   # generous vs. the ~0.1-0.2% actually observed; a format-mismatch
                     # corruption (wrong bit-packing) would miss by orders of magnitude

    def test_cpu_and_metal_agree_on_mixed_int4_int2_container(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = fixture.build(Path(td) / "glm_tiny_mixed")
            req_path = Path(td) / "requests.txt"
            req_path.write_text("\n".join(_build_requests(self.N_REQUESTS)) + "\n")

            lp_cpu, cpu_stderr = _run_score(model_dir, req_path, metal=False)
            try:
                lp_metal, metal_stderr = _run_score(model_dir, req_path, metal=True)
            except RuntimeError as e:
                if "no Metal backend" in str(e):
                    # Portable (non-Metal) build, per this file's own documented promise
                    # ("Skips cleanly when the glm binary, Metal, or numpy is unavailable").
                    # Using the engine's own definitive signal instead of a platform guess
                    # correctly skips both on non-macOS CI and on a real Mac built via `make
                    # portable` (no Metal either).
                    self.skipTest("glm binary has no Metal backend (portable build); "
                                  "rebuild with make METAL=1 to exercise this comparison")
                raise

            self.assertNotIn("[METAL] mode:", cpu_stderr,
                             "CPU-only run unexpectedly reports Metal active")
            self.assertIn("[METAL] mode:", metal_stderr,
                         "Metal run did not report itself active -- comparison would be "
                         "meaningless (silently running CPU-only on both sides)")
            self.assertEqual(len(lp_cpu), self.N_REQUESTS)
            self.assertEqual(len(lp_metal), self.N_REQUESTS)

            worst_rel, worst_i = 0.0, -1
            for i, (a, b) in enumerate(zip(lp_cpu, lp_metal)):
                rel = abs(a - b) / max(1.0, abs(a))
                if rel > worst_rel:
                    worst_rel, worst_i = rel, i
            self.assertLess(
                worst_rel, self.REL_TOL,
                f"request {worst_i}: CPU lp={lp_cpu[worst_i]:.6f} vs Metal "
                f"lp={lp_metal[worst_i]:.6f} (rel {worst_rel:.4f}) -- mixed int4/int2 "
                "routed-expert block diverged beyond floating-point-reassociation tolerance")


if __name__ == "__main__":
    unittest.main()
