"""Stage-2 v3 boundary regressions (the stage-2 v3 registration, engine commit f622b99).

Proves on the tiny mixed-MoE fixture: fail-closed trace shape; PIPE rejected in warmup mode;
prefill/warmup telemetry cannot leak into the measured window (IOLAT ring == measured reads only,
strictly fewer than a no-boundary control); the decode-only lever does not delay prefill (wall
ordering vs legacy lever which delays ALL phases); dual EFFECTIVE-FLAGS; legacy-only unchanged.
"""
from __future__ import annotations
import importlib.util, json, os, re, subprocess, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLM = HERE.parent / "glm"
spec = importlib.util.spec_from_file_location("tiny_mixed_moe_fixture", HERE / "tiny_mixed_moe_fixture.py")
fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)

FIX = Path(tempfile.gettempdir()) / "ili_v3_boundary_fixture"
NP, WARM, MEAS = 8, 4, 8

def ref(n_decode):
    ids = list(range(2, 2 + NP + n_decode))
    p = tempfile.mktemp(suffix=".json")
    json.dump({"prompt_ids": ids[:NP], "full_ids": ids}, open(p, "w"))
    return p

def run(env_extra, n_decode=WARM+MEAS, timeout=120):
    env = dict(os.environ, SNAP=str(FIX), REPLAY="1", REF=ref(n_decode),
               AUTOPIN="0", RAM_GB="0.01")
    env.pop("PIPE", None); env.pop("ILI_PIPE", None)
    env.update(env_extra)
    return subprocess.run([str(GLM), "2", "8", "8"], env=env, capture_output=True, text=True, timeout=timeout)

@unittest.skipUnless(GLM.exists(), "glm binary not built")
class V3Boundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FIX / "config.json").exists():
            fx.build(FIX)

    def test_shape_mismatch_fails_closed(self):
        r = run({"ILI_REPLAY_WARMUP": str(WARM), "ILI_REPLAY_MEASURE": str(MEAS)}, n_decode=WARM+MEAS+1)
        self.assertEqual(r.returncode, 2, r.stderr[-400:])
        self.assertIn("trace shape mismatch", r.stderr)

    def test_pipe_rejected_in_warmup_mode(self):
        r = run({"ILI_REPLAY_WARMUP": str(WARM), "ILI_REPLAY_MEASURE": str(MEAS), "PIPE": "1"})
        self.assertEqual(r.returncode, 2, r.stderr[-400:])
        self.assertIn("PIPE must be OFF", r.stderr)

    def test_measured_window_isolated_and_flags(self):
        ctrl = run({})                                    # no boundary: whole trace counted
        v3 = run({"ILI_REPLAY_WARMUP": str(WARM), "ILI_REPLAY_MEASURE": str(MEAS),
                  "ILI_IO_DELAY_DECODE_US": "0"})
        self.assertEqual(v3.returncode, 0, v3.stderr[-400:])
        self.assertIn("warmup decode tokens under treatment", v3.stderr)
        m = re.search(r"REPLAY decode: (\d+) tokens", v3.stdout)
        self.assertEqual(int(m.group(1)), MEAS, "measured denominator must be exactly MEAS")
        def reads(out):
            g = re.search(r"reads attempted (\d+) completed (\d+)", out)
            return int(g.group(2))
        def iolat_n(out):
            g = re.search(r"IO-LATENCY:.*\(n=(\d+) samples\)", out)
            return int(g.group(1))
        self.assertLess(reads(v3.stdout), reads(ctrl.stdout),
                        "measured-window reads must exclude prefill+warmup")
        self.assertEqual(iolat_n(v3.stdout), reads(v3.stdout),
                         "IOLAT ring must contain measured-window samples only (no leak)")
        self.assertIn("EFFECTIVE-FLAGS: IO_DELAY_DECODE_US requested=0 effective=0", v3.stdout)
        self.assertIn("EFFECTIVE-FLAGS: PREFILL_DELAY_US effective=0", v3.stdout)
        self.assertIn("EFFECTIVE-FLAGS: DECODE_DELAY_US effective=0", v3.stdout)

    def test_decode_lever_spares_prefill_legacy_covers_all(self):
        D = "3000"
        base = run({"ILI_REPLAY_WARMUP": str(WARM), "ILI_REPLAY_MEASURE": str(MEAS)})
        dec  = run({"ILI_REPLAY_WARMUP": str(WARM), "ILI_REPLAY_MEASURE": str(MEAS),
                    "ILI_IO_DELAY_DECODE_US": D})
        leg  = run({"ILI_REPLAY_WARMUP": str(WARM), "ILI_REPLAY_MEASURE": str(MEAS),
                    "ILI_IO_DELAY_US": D})
        for r in (base, dec, leg): self.assertEqual(r.returncode, 0, r.stderr[-300:])
        def wall(r):
            return float(re.search(r"REPLAY decode: \d+ tokens in ([\d.]+)s", r.stdout).group(1))
        self.assertGreater(wall(dec), wall(base) * 1.5, "decode lever must slow the measured window")
        self.assertIn("DECODE_DELAY_US effective=3000", dec.stdout)
        self.assertIn("PREFILL_DELAY_US effective=0", dec.stdout)
        self.assertIn("PREFILL_DELAY_US effective=3000", leg.stdout)

    def test_legacy_only_unchanged(self):
        r = run({"ILI_IO_DELAY_US": "1000"})
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        self.assertIn("EFFECTIVE-FLAGS: IO_DELAY_US requested=1000 effective=1000", r.stdout)
        self.assertNotIn("warmup decode tokens", r.stderr)
