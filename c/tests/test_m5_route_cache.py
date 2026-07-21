from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sim = load_module("simulate_m5max_cache", ROOT / "tools" / "simulate_m5max_cache.py")
patcher = load_module("patch_m5max_route_trace", ROOT / "tools" / "patch_m5max_route_trace.py")


class RouteTraceTests(unittest.TestCase):
    def write_trace(self, path: Path, version: int = 2) -> None:
        records = [
            # call 0: expert 1 is repeated across rows and must be one engine lookup.
            (0, 0, 3, 0, 0, 1),
            (1, 0, 3, 0, 1, 2),
            (2, 0, 3, 1, 0, 1),
            # call 1
            (3, 1, 3, 0, 0, 2),
            (4, 1, 3, 0, 1, 3),
        ]
        with path.open("wb") as f:
            f.write(sim.HEADER_BASE.pack(sim.MAGIC, version, sim.RECORD.size))
            if version == 2:
                f.write(sim.HEADER_V2_META.pack(18_916_000, 75, 52, 1240, 0))
            for record in records:
                f.write(sim.RECORD.pack(*record))

    def test_batch_union_lru_and_v2_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "routes.bin"
            self.write_trace(path)
            requests, selections, calls, metadata = sim.load_engine_requests(path)
        self.assertEqual(selections, 5)
        self.assertEqual(calls, 2)
        self.assertEqual(requests, {3: [1, 2, 2, 3]})
        self.assertEqual(metadata.expert_bytes, 18_916_000)
        self.assertEqual(metadata.total_expert_slots, 1240 + 75 * 52)
        self.assertFalse(metadata.mtp_present)

        pin_hits, lru_hits, misses = sim.simulate_lru(requests, set(), 1)
        self.assertEqual((pin_hits, lru_hits, misses), (0, 1, 3))

        pin_hits, lru_hits, misses = sim.simulate_lru(requests, {(3, 1)}, 1)
        self.assertEqual((pin_hits, lru_hits, misses), (1, 1, 2))
        self.assertEqual(sim.simulate_optimal(requests, {(3, 1)}, 1), 2)

    def test_v1_backward_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "routes-v1.bin"
            self.write_trace(path, version=1)
            requests, selections, calls, metadata = sim.load_engine_requests(path)
        self.assertEqual(requests, {3: [1, 2, 2, 3]})
        self.assertEqual((selections, calls, metadata.version), (5, 2, 1))
        self.assertEqual(metadata.total_expert_slots, 0)

    def test_global_pin_ranking_keeps_layer_dimension(self) -> None:
        usage = {(3, 1): 10, (4, 1): 9, (3, 2): 8}
        pins = sim.select_pins(usage, 2, "global", [3, 4])
        self.assertEqual(pins, {(3, 1), (4, 1)})

    def test_rejects_truncated_trace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.bin"
            path.write_bytes(sim.HEADER_BASE.pack(sim.MAGIC, 2, sim.RECORD.size) + b"x")
            with self.assertRaises(ValueError):
                list(sim.iter_trace(path))

    def test_generated_source_patch(self) -> None:
        source = """prefix
/* MoE GLM su x[S,hidden] -> out (router sigmoid/noaux_tc, n_group=1, + shared expert).
body
    m->enr[layer]=keff[S-1]; for(int kk=0;kk<keff[S-1];kk++) m->eroute[layer][kk]=idxs[(int64_t)(S-1)*K+kk];
suffix
"""
        patched = patcher.patch_text(source)
        self.assertIn("FAROUTE1", patched)
        self.assertIn("m5_route_put32(h+8,2)", patched)
        self.assertIn("m5_route_trace_emit(m,layer,S,K,idxs,keff);", patched)
        self.assertEqual(patched.count("m5_route_trace_emit(m,layer,S,K,idxs,keff);"), 1)


if __name__ == "__main__":
    unittest.main()
