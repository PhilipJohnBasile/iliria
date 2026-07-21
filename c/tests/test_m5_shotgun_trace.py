from __future__ import annotations

import importlib.util
import struct
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


shotgun = load_module("simulate_shotgun", ROOT / "tools" / "simulate_shotgun.py")


def write_trace(path: Path, version: int, records: list[tuple]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<8sII", b"FAROUTE1", version, 24))
        if version >= 2:
            # engine header: expert_bytes u64 + cache_units/lru/pinned/flags u32 = 24 bytes
            f.write(struct.pack("<QIIII", 18915328, 75, 34, 2584, 0))
        for rec in records:
            f.write(struct.pack("<QQHHHH", *rec))


class ShotgunTraceHeader(unittest.TestCase):
    """Regression: the overnight3 run crashed on the 24-byte v2 metadata block
    because a stale copy read only 20 bytes. The v2 header is 16 base + 24 meta."""

    def test_reads_v2_header_and_records(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "routes.bin"
            recs = [(i, i // 8, 3, 0, i % 8, (i * 37) % 256) for i in range(16)]
            write_trace(p, 2, recs)
            meta, records = shotgun.read_trace(p)
            self.assertEqual(meta["expert_bytes"], 18915328)
            self.assertEqual(meta["cache_units"], 75)
            self.assertEqual(meta["lru_per_layer"], 34)
            self.assertEqual(meta["pinned_units"], 2584)
            self.assertEqual(len(records), 16)
            self.assertEqual(records[0].expert, 0)
            self.assertEqual(records[1].expert, 37)

    def test_reads_v1_header(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "routes.bin"
            write_trace(p, 1, [(0, 0, 0, 0, 0, 5)])
            meta, records = shotgun.read_trace(p)
            self.assertEqual(meta, {})
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].expert, 5)

    def test_matches_cache_sim_header_layout(self):
        cache_sim = load_module(
            "simulate_m5max_cache", ROOT / "tools" / "simulate_m5max_cache.py"
        )
        self.assertEqual(shotgun.HEADER_BASE.size, cache_sim.HEADER_BASE.size)
        self.assertEqual(shotgun.HEADER_V2_META.size, cache_sim.HEADER_V2_META.size)
        self.assertEqual(shotgun.RECORD.size, cache_sim.RECORD.size)
        self.assertEqual(shotgun.HEADER_V2_META.size, 24)


if __name__ == "__main__":
    unittest.main()
