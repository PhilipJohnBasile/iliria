"""End-to-end tests for tools/encode_mode15_container.py on a tiny
synthetic int4 container (no /path/to/models access, no engine, no
network) -- same tiny-fixture convention as test_container_pipeline.py's
make_container(): routed layers 3..6, 4 experts/layer, one shard with no
routed experts at all (verbatim-only shard), all built via
quant_container.st_save/quantize so the fixture is byte-for-byte what the
real int4 container's own tooling would produce.
"""
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
m15 = load_module("mode15_container", ROOT / "tools" / "mode15_container.py")
enc = load_module("encode_mode15_container", ROOT / "tools" / "encode_mode15_container.py")

CFG = {"hidden_size": 8, "moe_intermediate_size": 4, "num_hidden_layers": 7,
       "first_k_dense_replace": 3, "n_routed_experts": 4}


def make_int4_container(root: Path, rng) -> dict:
    """Tiny int4-only container: 2 shards with routed experts (layers 3-6,
    4 experts each) + 1 shard with ONLY non-expert tensors (exercises the
    "verbatim-only shard" path)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(CFG))
    weights = {}
    shards = {"out-00000.safetensors": [3, 4], "out-00001.safetensors": [5, 6]}
    for fn, layers in shards.items():
        tensors = {}
        for layer in layers:
            for expert in range(4):
                w = {}
                for proj in qc.PROJS:
                    O, I = (8, 4) if proj == "down_proj" else (4, 8)
                    w[proj] = (rng.standard_normal((O, I)) * 0.3).astype(np.float32)
                weights[(layer, expert)] = w
                for proj in qc.PROJS:
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    packed, scales = qc.quantize(w[proj], 4)
                    tensors[name] = packed
                    tensors[name + ".qs"] = scales
        qc.st_save(root / fn, tensors)
    # verbatim-only shard: no routed experts at all
    dense = (rng.standard_normal((4, 8)) * 0.2).astype(np.float32)
    packed, scales = qc.quantize(dense, 8)
    qc.st_save(root / "out-00002.safetensors", {
        "model.norm.weight": np.ones(8, np.float32),
        "model.layers.0.mlp.gate_proj.weight": packed,
        "model.layers.0.mlp.gate_proj.weight.qs": scales,
    })
    return weights


class EncodeModeE2ETests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / "tiny_i4"
        self.weights = make_int4_container(self.model, np.random.default_rng(2026))
        self.cfg = qc.load_config(self.model)

    def tearDown(self):
        self.tmp.cleanup()

    def run_encode(self, outdir, extra=()):
        return enc.main(["--indir", str(self.model), "--outdir", str(outdir),
                         "--min-free-gb", "0", "--allow-busy", *extra])

    # ---------------------------------------------------------------- core
    def test_full_encode_byte_exact_and_verbatim_passthrough(self):
        outdir = self.root / "mode15"
        rc = self.run_encode(outdir)
        self.assertEqual(rc, 0)

        src_idx = qc.st_scan(str(self.model))
        out_idx = qc.st_scan(str(outdir))
        self.assertEqual(set(src_idx), set(out_idx))

        for (layer, expert) in [(l, e) for l in range(3, 7) for e in range(4)]:
            for proj in qc.PROJS:
                name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                O, I = qc.expert_shape(self.cfg, proj)
                blob = qc.st_read_tensor(out_idx[name]).tobytes()
                got_packed = m15.decode_blob_to_packed_bytes(blob, expect_O=O, expect_I=I)
                src_packed = qc.st_read_tensor(src_idx[name]).tobytes()
                self.assertEqual(got_packed, src_packed, name)
                # .qs scales byte-identical (never touched)
                np.testing.assert_array_equal(
                    qc.st_read_tensor(out_idx[name + ".qs"]),
                    qc.st_read_tensor(src_idx[name + ".qs"]))

        # non-expert tensors (incl. the verbatim-only shard) byte-identical
        for name in ("model.norm.weight", "model.layers.0.mlp.gate_proj.weight",
                     "model.layers.0.mlp.gate_proj.weight.qs"):
            np.testing.assert_array_equal(qc.st_read_tensor(out_idx[name]),
                                          qc.st_read_tensor(src_idx[name]))

        self.assertTrue((outdir / "config.json").exists())
        manifest = json.loads((outdir / enc.MANIFEST_NAME).read_text())
        self.assertEqual(manifest["format"], m15.CONTAINER_FORMAT)
        self.assertEqual(manifest["schema_version"], m15.CONTAINER_SCHEMA_VERSION)
        self.assertEqual(manifest["model_config"], self.cfg)
        self.assertIn("tool_commit", manifest)
        self.assertEqual(len(manifest["shards"]), 3)
        for fn, entry in manifest["shards"].items():
            self.assertEqual(entry["status"], "done")
            self.assertEqual(len(entry["source_sha256"]), 64)
        # the verbatim-only shard recorded zero encoded tensors; shard 0
        # holds layers 3+4, 4 experts each, 3 projections = 24 tensors
        self.assertEqual(manifest["shards"]["out-00002.safetensors"]["n_encoded_tensors"], 0)
        self.assertEqual(manifest["shards"]["out-00000.safetensors"]["n_encoded_tensors"], 24)

    def test_verify_only_passes_on_complete_container(self):
        outdir = self.root / "mode15"
        self.assertEqual(self.run_encode(outdir), 0)
        rc = enc.main(["--indir", str(self.model), "--outdir", str(outdir), "--verify-only"])
        self.assertEqual(rc, 0)

    def test_verify_only_fails_closed_on_corruption(self):
        outdir = self.root / "mode15"
        self.assertEqual(self.run_encode(outdir), 0)
        # corrupt one byte inside an encoded expert tensor's payload region
        path = outdir / "out-00000.safetensors"
        header, data_start = qc.st_read_header(str(path))
        name = "model.layers.3.mlp.experts.0.gate_proj.weight"
        o0, o1 = header[name]["data_offsets"]
        with open(path, "r+b") as f:
            f.seek(data_start + o1 - 1)
            b = f.read(1)
            f.seek(data_start + o1 - 1)
            f.write(bytes([b[0] ^ 0xFF]))
        rc = enc.main(["--indir", str(self.model), "--outdir", str(outdir), "--verify-only"])
        self.assertEqual(rc, 1)

    # --------------------------------------------------------- resumability
    def test_limit_then_resume_completes_without_reencoding_done_shards(self):
        outdir = self.root / "mode15"
        rc = self.run_encode(outdir, ["--limit", "1"])
        self.assertEqual(rc, 0)
        manifest_after_1 = json.loads((outdir / enc.MANIFEST_NAME).read_text())
        self.assertEqual(len(manifest_after_1["shards"]), 1)

        # record built_at + mtime for the shard already done
        done_fn = next(iter(manifest_after_1["shards"]))
        built_at_1 = manifest_after_1["shards"][done_fn]["built_at"]
        mtime_1 = os.path.getmtime(outdir / done_fn)

        rc = self.run_encode(outdir)  # resume, no --limit
        self.assertEqual(rc, 0)
        manifest_final = json.loads((outdir / enc.MANIFEST_NAME).read_text())
        self.assertEqual(len(manifest_final["shards"]), 3)
        # the already-done shard was NOT rebuilt (timestamp/mtime unchanged)
        self.assertEqual(manifest_final["shards"][done_fn]["built_at"], built_at_1)
        self.assertEqual(os.path.getmtime(outdir / done_fn), mtime_1)

        src_idx = qc.st_scan(str(self.model))
        out_idx = qc.st_scan(str(outdir))
        self.assertEqual(set(src_idx), set(out_idx))

    def test_rebuilds_a_shard_whose_output_was_deleted(self):
        outdir = self.root / "mode15"
        self.assertEqual(self.run_encode(outdir), 0)
        target = outdir / "out-00001.safetensors"
        os.remove(target)
        rc = self.run_encode(outdir)
        self.assertEqual(rc, 0)
        self.assertTrue(target.exists())
        rc = enc.main(["--indir", str(self.model), "--outdir", str(outdir), "--verify-only"])
        self.assertEqual(rc, 0)

    def test_refuses_mismatched_rows_per_block_on_resume(self):
        outdir = self.root / "mode15"
        self.assertEqual(self.run_encode(outdir, ["--rows-per-block", "2"]), 0)
        with self.assertRaises(SystemExit):
            self.run_encode(outdir, ["--rows-per-block", "4"])

    def test_refuses_same_indir_outdir(self):
        with self.assertRaises(SystemExit):
            self.run_encode(self.model)

    # ------------------------------------------------------------- ratio
    def test_aggregate_ratio_is_reported_and_positive(self):
        # NOTE: this fixture's expert tensors are tiny (O=4 or O=8 rows) --
        # far too small for the ratio to be meaningful: the blob's FIXED
        # per-tensor overhead (32B header + 8B length table + (O+1)*4B
        # row_offsets + block CRC32s) dwarfs a 16-byte raw payload, so the
        # ratio here is >1 (expansion) EXPECTED, not a bug -- real expert
        # tensors (O=2048/6144 rows) show genuine ~0.74 compression, see
        # tests/test_mode15_container.py's real-row-shape tests and this
        # this test's own smoke-test-on-real-shards numbers. This test only
        # checks the aggregate accounting itself is wired up correctly.
        outdir = self.root / "mode15"
        self.assertEqual(self.run_encode(outdir), 0)
        manifest = json.loads((outdir / enc.MANIFEST_NAME).read_text())
        total_raw = sum(s["raw_expert_bytes"] for s in manifest["shards"].values())
        total_enc = sum(s["encoded_expert_bytes"] for s in manifest["shards"].values())
        self.assertGreater(total_raw, 0)
        self.assertGreater(total_enc, 0)
        # sanity bound: even at this pathological tiny size, overhead is
        # bounded and known (per-tensor blob <= 32+8+(O+1)*4+n_blocks*4+O*I/2
        # bytes) -- so the ratio can't be UNBOUNDED, just larger than 1.
        self.assertLess(total_enc / total_raw, 10.0)


if __name__ == "__main__":
    unittest.main()
