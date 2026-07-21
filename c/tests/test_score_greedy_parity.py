"""Greedy-parity regression (deliverable 4b, c/bench-m5max/factorial-streaming-causality-
the format spec): a SCORE request whose continuation IS the greedy decode path, run on BOTH the CPU
attention path and the repaired S>4 Metal-prefill-attention path, asserting they select the
SAME continuation (both "greedy"==1) with logprobs agreeing within a stated tolerance -- and,
critically, that the Metal leg actually REACHES the repaired gate (METAL-ATTN engagement
count > 0), not merely runs without crashing on a path that silently fell back to CPU.

Uses tests/glm_score_gate_fixture.py -- see that module's docstring for why a fixture at
GLM-5.2's real attention dims (not a "tiny" 128-hidden one) is unavoidable: the gate under
test hardcodes those dims as part of its own precondition, so no smaller fixture can ever
reach it.

Method (avoids any dependency on the `tokenizers` Python package or the C engine's text
tokenizer/detokenizer -- everything here is raw integer token ids, SCORE mode's own native
format, and this fixture's vocab is byte-level with ignore_merges=True so token id N is
simply the raw byte N with no encoding ambiguity to get wrong):
  1. Fix a short context (bytes of an ASCII string).
  2. Discover a K-token GREEDY continuation by, at each step, SCORE-ing all 256 possible
     next bytes as length-1 continuations of (context + tokens-found-so-far) on the CPU-only
     leg, and taking the one candidate SCORE reports greedy=1 for (the argmax) -- this is
     exactly the autoregressive greedy recurrence, just phrased as repeated one-token SCOREs
     instead of a decode loop, and confirms uniqueness (exactly one winner) along the way.
  3. SCORE the resulting (context, K-token continuation) pair as ONE request -- S=T well
     above 4 -- on the CPU leg (ILI_METAL unset) and the Metal leg (ILI_METAL=1
     ILI_METAL_PREFILL=1), and compare.

Skips cleanly when the glm binary, Metal, or numpy is unavailable (matches
test_mixed_format_moe.py's convention).
"""

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
GLM = HERE / "glm"
FIXTURE_CACHE = Path(tempfile.gettempdir()) / "ili_glm_score_gate_fixture_cache"


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
        "glm_score_gate_fixture", str(Path(__file__).resolve().parent / "glm_score_gate_fixture.py"))


def _engine_supports_metal():
    """True iff this ./glm binary was compiled with -DILI_METAL (a CPU-only build can never
    take the Metal leg, so the test must skip rather than silently pass on a no-op).
    main()'s SNAP check runs BEFORE the METAL-availability check, so a bogus (nonexistent)
    SNAP is required just to get past it -- the METAL check itself still runs first, before
    any attempt to actually open the (nonexistent) model directory."""
    if not GLM.exists():
        return False
    try:
        proc = subprocess.run([str(GLM)], env=dict(os.environ, ILI_METAL="1", SNAP="/nonexistent-glm-score-gate-fixture-probe"),
                              capture_output=True, text=True, timeout=10)
    except Exception:
        return False
    return "no Metal backend" not in (proc.stdout + proc.stderr)


def _run_score(model_dir, req_lines, metal):
    req_path = tempfile.mktemp(suffix=".txt")
    Path(req_path).write_text("\n".join(req_lines) + "\n")
    env = dict(os.environ, SNAP=str(model_dir), SCORE=req_path)
    if metal:
        env["ILI_METAL"] = "1"
        env["ILI_METAL_PREFILL"] = "1"
    else:
        env.pop("ILI_METAL", None)
        env.pop("ILI_METAL_PREFILL", None)
    try:
        # dbits=4 (3rd CLI arg) is REQUIRED: the gate's own precondition checks
        # l->kv_b.fmt==2, i.e. kv_b must be quantized to int4 on load.
        proc = subprocess.run([str(GLM), "4", "8", "4"], env=env,
                              capture_output=True, text=True, timeout=180)
    finally:
        os.remove(req_path)
    if proc.returncode != 0:
        raise RuntimeError(f"engine failed (metal={metal}), rc={proc.returncode}:\n{proc.stderr[-4000:]}\n---stdout tail---\n{proc.stdout[-2000:]}")
    out_lines = [ln for ln in proc.stdout.strip().splitlines() if ln and ln[0] in "-0123456789"]
    parsed = [(float(a), int(b), int(c)) for a, b, c in (ln.split() for ln in out_lines)]
    return parsed, proc.stdout, proc.stderr


def _discover_greedy_continuation(model_dir, ctx_ids, k):
    """Returns a list of k token ids: the autoregressive greedy continuation of ctx_ids,
    discovered via repeated CPU-leg SCORE calls (see module docstring, step 2)."""
    so_far = []
    for _step in range(k):
        cur_ctx = ctx_ids + so_far
        lines = [f"{len(cur_ctx)} 1 " + " ".join(map(str, cur_ctx + [b])) for b in range(256)]
        results, _out, _err = _run_score(model_dir, lines, metal=False)
        winners = [b for b, (_lp, _cl, greedy) in enumerate(results) if greedy == 1]
        assert len(winners) == 1, (
            f"discovery step {_step}: expected exactly one argmax winner among 256 "
            f"candidates, got {winners} -- either a tie (statistically implausible with "
            f"continuous random weights) or a SCORE-mode bug"
        )
        so_far.append(winners[0])
    return so_far


def _glm_is_stale() -> bool:
    """True when ./glm predates its own sources -- the exact condition that made this test
    fail in-suite on 2026-07-16 (row 20): a pre-dsa-gate-fix binary can never engage the
    repaired Metal path, so parity 'fails' against a binary the current source did not build.
    A stale binary must be a precise skip (with remedy), never a silent wrong-binary run."""
    try:
        bin_m = GLM.stat().st_mtime
        return any((HERE / f).stat().st_mtime > bin_m
                   for f in ("glm.c", "backend_metal.mm") if (HERE / f).exists())
    except OSError:
        return True


@unittest.skipUnless(GLM.exists(), "glm binary not built")
@unittest.skipIf(_glm_is_stale(), "stale ./glm (older than glm.c/backend_metal.mm) -- run: make glm METAL=1")
@unittest.skipUnless(HAVE_NUMPY, "numpy not available")
@unittest.skipUnless(_engine_supports_metal(), "glm binary has no Metal backend (build with METAL=1)")
class ScoreGreedyParityTest(unittest.TestCase):
    K_CONTINUATION = 3
    # Pre-registered tolerance: this is a 1-layer, int4-dense, RANDOM-weight fixture (no
    # accumulation across 78 real layers), so CPU-vs-Metal float-reassociation drift should
    # be small; e567f80's own real-model measurement (78 layers, hellaswag) was 0.0299 nats
    # max over 16 requests. 0.25 nats is a generous margin above that for a single request
    # on this fixture while still being far tighter than would tolerate an actual wrong-path
    # bug (a gate that silently fell back to CPU produces EXACT agreement, 0.0 delta, which
    # this bound obviously also accepts -- the METAL-ATTN engagement-count assertion below is
    # what specifically catches that failure mode; the logprob tolerance catches numerical
    # divergence, a different and complementary failure mode).
    LOGPROB_TOLERANCE_NATS = 0.25

    @classmethod
    def setUpClass(cls):
        cls.model_dir = fixture.build(FIXTURE_CACHE)

    def test_cpu_and_metal_paths_agree_on_the_genuinely_greedy_continuation(self):
        ctx_ids = [ord(c) for c in "hello world, this is a test context"]
        cont_ids = _discover_greedy_continuation(self.model_dir, ctx_ids, self.K_CONTINUATION)
        full = ctx_ids + cont_ids
        req_line = f"{len(ctx_ids)} {len(cont_ids)} " + " ".join(map(str, full))
        self.assertGreater(len(full), 4, "S must be >4 to even be eligible for the gate under test")

        cpu_results, cpu_out, _cpu_err = _run_score(self.model_dir, [req_line], metal=False)
        metal_results, metal_out, _metal_err = _run_score(self.model_dir, [req_line], metal=True)

        self.assertEqual(len(cpu_results), 1)
        self.assertEqual(len(metal_results), 1)
        lp_cpu, cl_cpu, greedy_cpu = cpu_results[0]
        lp_metal, cl_metal, greedy_metal = metal_results[0]

        self.assertEqual(cl_cpu, self.K_CONTINUATION)
        self.assertEqual(cl_metal, self.K_CONTINUATION)

        # The genuinely-greedy request must register as greedy on BOTH legs: this IS the
        # "same selected token" assertion (greedy==1 means every continuation position
        # matched that position's argmax, i.e. the SAME token was selected at every step).
        self.assertEqual(greedy_cpu, 1, f"CPU leg: discovered continuation did not score as greedy (lp={lp_cpu})")
        self.assertEqual(greedy_metal, 1, f"Metal leg: discovered continuation did not score as greedy (lp={lp_metal}) -- "
                                          f"the repaired gate selected a DIFFERENT token than the CPU path")

        # The Metal leg must actually have REACHED the repaired gate -- a test that merely
        # runs without crashing on a silently-CPU-fallback path is vacuous.
        m = re.search(r"METAL-ATTN: layer GPU (\d+)", metal_out)
        self.assertIsNotNone(m, f"Metal leg produced no METAL-ATTN line at all (gate never engaged):\n{metal_out[-2000:]}")
        metal_attn_count = int(m.group(1))
        self.assertGreater(metal_attn_count, 0,
            "Metal leg's METAL-ATTN engagement count is 0 -- the repaired gate did not "
            "actually route this request's S>4 attention through the Metal kernel")

        delta = abs(lp_cpu - lp_metal)
        self.assertLess(delta, self.LOGPROB_TOLERANCE_NATS,
            f"CPU vs Metal logprob delta {delta:.4f} nats exceeds the pre-registered "
            f"{self.LOGPROB_TOLERANCE_NATS} nat tolerance (lp_cpu={lp_cpu:.4f} lp_metal={lp_metal:.4f})")


if __name__ == "__main__":
    unittest.main()
