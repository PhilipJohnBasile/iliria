"""Stage-2 v4 snapshot regressions (the stage-2 v4 registration, engine 29a96d5).

Proves: SAVE writes snapshot+prov and exits WITHOUT decode/telemetry; LOAD restores KV and yields
BYTE-IDENTICAL greedy decisions vs live prefill (argmax-hash); provenance mismatch fails closed;
fresh expert cache (identical hit rate); legacy replay (no snapshot env) unchanged.
"""
from __future__ import annotations
import importlib.util, json, os, re, subprocess, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLM = HERE.parent / "glm"
spec = importlib.util.spec_from_file_location("tiny_mixed_moe_fixture", HERE / "tiny_mixed_moe_fixture.py")
fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)
FIX = Path(tempfile.gettempdir()) / "ili_v4t_fixture"
NP, WARM, MEAS = 8, 4, 8

def ref():
    ids = list(range(2, 2 + NP + WARM + MEAS))
    p = tempfile.mktemp(suffix=".json")
    json.dump({"prompt_ids": ids[:NP], "full_ids": ids}, open(p, "w"))
    return p

def run(extra, r=None, timeout=120):
    env = dict(os.environ, SNAP=str(FIX), REPLAY="1", REF=r or ref(), AUTOPIN="0", RAM_GB="0.01",
               ILI_REPLAY_WARMUP=str(WARM), ILI_REPLAY_MEASURE=str(MEAS))
    env.pop("PIPE", None)
    env.update(extra)
    return subprocess.run([str(GLM), "2", "8", "8"], env=env, capture_output=True, text=True, timeout=timeout)

@unittest.skipUnless(GLM.exists(), "glm binary not built")
class V4Snapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FIX / "config.json").exists(): fx.build(FIX)
        cls.snap = tempfile.mktemp(suffix=".v4snap")
        cls.r = ref()
        s = run({"ILI_KV_SNAPSHOT": cls.snap, "ILI_REPLAY_SAVE": "1"}, r=cls.r)
        assert s.returncode == 0, s.stderr[-400:]
        cls.save_stderr, cls.save_stdout = s.stderr, s.stdout

    def test_save_exits_without_decode_or_telemetry(self):
        self.assertIn("V4 SAVE", self.save_stderr)
        self.assertTrue(Path(self.snap).exists() and Path(self.snap + ".prov").exists())
        self.assertNotIn("REPLAY decode:", self.save_stdout)      # no measured window
        self.assertNotIn("warmup decode tokens", self.save_stderr)  # no treatment applied

    def test_load_parity_argmax_hash_identical(self):
        live = run({"ILI_REPLAY_ARGMAX_HASH": "1"}, r=self.r)
        snap = run({"ILI_KV_SNAPSHOT": self.snap, "ILI_REPLAY_ARGMAX_HASH": "1"}, r=self.r)
        for x in (live, snap): self.assertEqual(x.returncode, 0, x.stderr[-300:])
        h = lambda o: re.search(r"argmax-hash: ([0-9a-f]+) \(n=(\d+)\)", o).groups()
        self.assertEqual(h(live.stdout), h(snap.stdout), "snapshot restore must match live decisions")
        self.assertIn("no live prefill", snap.stderr)
        self.assertIn("expert cache FRESH", snap.stderr)

    def test_fresh_cache_identical_hitrate(self):
        live = run({}, r=self.r); snap = run({"ILI_KV_SNAPSHOT": self.snap}, r=self.r)
        hr = lambda o: re.search(r"expert hit ([\d.]+)%", o).group(1)
        self.assertEqual(hr(live.stdout), hr(snap.stdout))

    def test_prov_prompt_mismatch_fails_closed(self):
        bad = {"prompt_ids": list(range(99, 99 + NP)), "full_ids": list(range(99, 99 + NP + WARM + MEAS))}
        rp = tempfile.mktemp(suffix=".json")
        with open(rp, "w") as _f: json.dump(bad, _f)
        r = run({"ILI_KV_SNAPSHOT": self.snap}, r=rp)
        self.assertEqual(r.returncode, 2, r.stderr[-300:])
        self.assertIn("prompt_hash mismatch", r.stderr)

    def test_missing_prov_fails_closed(self):
        r = run({"ILI_KV_SNAPSHOT": tempfile.mktemp(suffix=".none")}, r=self.r)
        self.assertEqual(r.returncode, 2, r.stderr[-300:])
        self.assertIn("missing provenance", r.stderr)

    def test_legacy_replay_unchanged(self):
        r = run({})   # no snapshot env: pure legacy v3 replay path
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        self.assertIn("REPLAY decode:", r.stdout)
        self.assertNotIn("V4", r.stderr)
