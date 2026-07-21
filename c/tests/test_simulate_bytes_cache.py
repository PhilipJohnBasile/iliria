from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


slotsim = load_module("simulate_m5max_cache", ROOT / "tools" / "simulate_m5max_cache.py")
bsim = load_module("simulate_bytes_cache", ROOT / "tools" / "simulate_bytes_cache.py")


def make_calls(spec):
    """spec: [(layer, [experts], n_rows)] -> [Call] with sequential ids."""
    return [bsim.Call(i, layer, list(experts), n_rows, len(experts))
            for i, (layer, experts, n_rows) in enumerate(spec)]


def token_map(calls):
    return bsim.segment_tokens(calls)


class LayerLRUByteTests(unittest.TestCase):
    def test_byte_exact_eviction_order(self):
        lru = bsim.LayerLRU("bytes", 100)
        lru.insert("a", 40)
        lru.insert("b", 40)
        evicted = []
        lru.insert("c", 30, evicted)  # 40+40+30 = 110 > 100 -> evict a only
        self.assertEqual(evicted, [("a", 40)])
        self.assertEqual(set(lru.od), {"b", "c"})
        self.assertEqual(lru.used, 70)

    def test_large_item_evicts_until_it_fits(self):
        lru = bsim.LayerLRU("bytes", 100)
        lru.insert("b", 40)
        lru.insert("c", 30)
        evicted = []
        lru.insert("d", 100, evicted)  # needs the whole budget
        self.assertEqual(evicted, [("b", 40), ("c", 30)])
        self.assertEqual(set(lru.od), {"d"})
        self.assertEqual(lru.used, 100)

    def test_oversized_item_never_flushes_the_tier(self):
        lru = bsim.LayerLRU("bytes", 100)
        lru.insert("d", 100)
        evicted = []
        lru.insert("e", 101, evicted)  # cannot ever fit
        self.assertEqual(evicted, [])
        self.assertEqual(set(lru.od), {"d"})
        self.assertEqual(lru.used, 100)

    def test_touch_changes_the_victim(self):
        lru = bsim.LayerLRU("bytes", 100)
        lru.insert("a", 50)
        lru.insert("b", 50)
        lru.touch("a")  # b is now the LRU end
        evicted = []
        lru.insert("c", 50, evicted)
        self.assertEqual(evicted, [("b", 50)])
        self.assertEqual(set(lru.od), {"a", "c"})

    def test_slot_mode_ignores_size(self):
        lru = bsim.LayerLRU("slots", 2)
        lru.insert("a", 1)
        lru.insert("b", 1000)
        evicted = []
        lru.insert("c", 5, evicted)
        self.assertEqual(evicted, [("a", 1)])
        self.assertEqual(set(lru.od), {"b", "c"})


class PinFillTests(unittest.TestCase):
    USAGE = {(3, 1): 100, (3, 2): 90, (4, 1): 80}

    def test_exact_fill(self):
        sizes = {(3, 1): 40, (3, 2): 40, (4, 1): 10}
        pinned, used = bsim.fill_pins(self.USAGE, sizes, 90)
        self.assertEqual(set(pinned), {(3, 1), (3, 2), (4, 1)})
        self.assertEqual(used, 90)

    def test_truncates_at_first_overflow_like_pin_load(self):
        # (4,1) at 5 B would fit after the 45 B pair overflows, but pin_load
        # truncates the ranking -- it does not skip and continue.
        sizes = {(3, 1): 40, (3, 2): 45, (4, 1): 5}
        pinned, used = bsim.fill_pins(self.USAGE, sizes, 84)
        self.assertEqual(set(pinned), {(3, 1)})
        self.assertEqual(used, 40)

    def test_pairs_missing_from_size_table_are_ignored(self):
        usage = dict(self.USAGE)
        usage[(78, 0)] = 10_000  # e.g. an MTP-layer row
        sizes = {(3, 1): 40, (3, 2): 40, (4, 1): 10}
        pinned, used = bsim.fill_pins(usage, sizes, 90)
        self.assertEqual(set(pinned), {(3, 1), (3, 2), (4, 1)})
        self.assertEqual(used, 90)


class TwoPhaseReplayTests(unittest.TestCase):
    def test_pre_insertion_classification(self):
        # slots=2. Call 0 loads {1,2}. Call 1 requests [3, 1]: sequential
        # insertion of 3 would evict 1 before its lookup; the engine (and the
        # two-phase replay) classifies BOTH against call-entry state -> 1 hits.
        calls = make_calls([(3, [1, 2], 1), (3, [3, 1], 1)])
        token_of, n = token_map(calls)
        sizes = {(3, e): 10 for e in range(6)}
        res = bsim.replay(calls, token_of, sizes, {}, "slots", 2,
                          collect_m=True)
        self.assertEqual(res.lru_hits, 1)
        self.assertEqual(res.disk_misses, 3)
        # call 1: M=1 (expert 3), and after apply the cache holds {1, 3}
        self.assertEqual(res.call_m[1][2], 1)

    def test_apply_phase_updates_recency_and_evicts_byte_exact(self):
        # bytes budget 100; sizes: e1=60, e2=50, e3=40.
        # call0 [1]: miss (60). call1 [2]: miss -> 60+50 > 100 evicts e1 (50).
        # call2 [3]: miss -> 50+40 <= 100 fits (90). call3 [2]: hit, touch.
        # call4 [1]: miss -> 60 needs eviction of e3 then e2? LRU end is e3
        # (order after touch: e3, e2) -> evict e3 (50 used), 50+60>100 evict e2
        # -> insert e1.
        sizes = {(3, 1): 60, (3, 2): 50, (3, 3): 40}
        calls = make_calls([(3, [1], 1), (3, [2], 1), (3, [3], 1),
                            (3, [2], 1), (3, [1], 1)])
        token_of, _ = token_map(calls)
        res = bsim.replay(calls, token_of, sizes, {}, "bytes", 100)
        self.assertEqual(res.lru_hits, 1)
        self.assertEqual(res.disk_misses, 4)
        self.assertEqual(res.disk_bytes, 60 + 50 + 40 + 60)

    def test_uniform_sizes_match_slot_simulator(self):
        rng = random.Random(7)
        layers = [3, 4, 5]
        requests = {l: [rng.randrange(12) for _ in range(400)] for l in layers}
        usage = slotsim.usage_from_requests(requests)
        pinned_set = slotsim.select_pins(usage, 6, "global", layers)
        capacity = 4
        want = slotsim.simulate_lru(requests, pinned_set, capacity)

        size = 1000
        sizes = {(l, e): size for l in layers for e in range(12)}
        pinned = {p: size for p in pinned_set}
        spec = [(l, [e], 1) for l in layers for e in requests[l]]
        calls = make_calls(spec)
        token_of, _ = token_map(calls)
        for mode, budget in (("slots", capacity), ("bytes", capacity * size)):
            res = bsim.replay(calls, token_of, sizes, pinned, mode, budget)
            self.assertEqual(
                (res.pin_hits, res.lru_hits, res.disk_misses), want,
                f"mode={mode}")

    def test_byte_mode_capacity_gain_from_smaller_experts(self):
        # cyclic [1,2,1,2...] with budget 100: 60 B experts -> one resident,
        # every access misses; 50 B experts -> both resident after warmup.
        calls = make_calls([(3, [1 + (i % 2)], 1) for i in range(20)])
        token_of, _ = token_map(calls)
        big = {(3, 1): 60, (3, 2): 60}
        small = {(3, 1): 50, (3, 2): 50}
        res_big = bsim.replay(calls, token_of, big, {}, "bytes", 100)
        res_small = bsim.replay(calls, token_of, small, {}, "bytes", 100)
        self.assertEqual(res_big.lru_hits, 0)
        self.assertEqual(res_small.lru_hits, 18)
        # slot mode with 1 slot cannot benefit from the smaller size
        res_slot = bsim.replay(calls, token_of, small, {}, "slots", 1)
        self.assertEqual(res_slot.lru_hits, 0)

    def test_page_cache_second_tier(self):
        size = 10
        sizes = {(3, 1): size, (3, 2): size}
        calls = make_calls([(3, [1], 1), (3, [2], 1), (3, [1], 1)])
        token_of, _ = token_map(calls)
        # without a page cache: the third call re-reads expert 1 from disk
        res = bsim.replay(calls, token_of, sizes, {}, "slots", 1)
        self.assertEqual((res.page_hits, res.disk_bytes), (0, 3 * size))
        # with a page cache: expert 1's pages survive its RAM eviction
        res = bsim.replay(calls, token_of, sizes, {}, "slots", 1,
                          page_bytes=1000)
        self.assertEqual(res.page_hits, 1)
        self.assertEqual(res.disk_bytes, 2 * size)
        self.assertEqual(res.disk_misses, 2)
        # a page hit is still a RAM miss
        self.assertEqual(res.lru_hits, 0)


class TokenSegmentationTests(unittest.TestCase):
    def test_prefill_and_decode_split(self):
        calls = make_calls([
            (3, [1, 2], 2), (4, [1], 2),          # prefill (multi-row)
            (3, [1], 1), (4, [2], 1),             # token 0
            (3, [2], 1), (4, [1], 1),             # token 1
        ])
        token_of, n_tokens = token_map(calls)
        self.assertEqual(n_tokens, 2)
        self.assertEqual([token_of[i] for i in range(6)],
                         [None, None, 0, 0, 1, 1])

    def test_per_token_bytes(self):
        sizes = {(3, 1): 7, (3, 2): 11}
        calls = make_calls([(3, [1], 1), (3, [2], 1), (3, [1], 1)])
        token_of, _ = token_map(calls)
        res = bsim.replay(calls, token_of, sizes, {}, "slots", 2)
        self.assertEqual(res.per_token_bytes, [7, 11, 0])
        self.assertEqual(res.per_token_misses, [1, 1, 0])
        self.assertEqual(res.steady(1), ((11 + 0) / 2, 0.5, 2))
        self.assertEqual(res.warm(1), (7.0, 1.0, 1))


class MissStructureTests(unittest.TestCase):
    def build(self):
        # two layers, two tokens; slots=1 per layer, no pins.
        # L3 stream: [1],[1],[2]; L4 stream: [5],[5],[5]
        calls = make_calls([
            (3, [1], 1), (4, [5], 1),   # token 0: L3 miss, L4 miss
            (3, [1], 1), (4, [5], 1),   # token 1: both hit
            (3, [2], 1), (4, [5], 1),   # token 2: L3 miss, L4 hit
        ])
        token_of, _ = token_map(calls)
        sizes = {(3, 1): 10, (3, 2): 20, (4, 5): 30}
        return bsim.replay(calls, token_of, sizes, {}, "slots", 1,
                           collect_m=True)

    def test_pm_and_runs(self):
        res = self.build()
        s = bsim.miss_structure(res, warmup=0)
        self.assertEqual(s["n_decode_calls"], 6)
        self.assertAlmostEqual(s["pm_all"][0], 3 / 6)
        self.assertAlmostEqual(s["pm_all"][1], 3 / 6)
        self.assertAlmostEqual(s["p_le1"], 1.0)
        # all-hit layer-calls: 3 of 6; runs: token1 has a run of 2 (both
        # layers all-hit), token2 a run of 1 (L4 only)
        self.assertEqual(s["run_hist"], {1: 1, 2: 1})
        self.assertAlmostEqual(s["frac_layers_all_hit"], 3 / 6)
        self.assertAlmostEqual(s["frac_layers_in_runs_ge2"], 2 / 6)
        self.assertAlmostEqual(s["frac_layers_in_runs_ge3"], 0.0)
        # h = 3/6; per-layer h: L3 1/3, L4 2/3 (M over 1 expert -> h^8 analog
        # uses the 8-expert formula only on real traces; here just sanity)
        self.assertAlmostEqual(s["h"], 0.5)
        # bytes by M at steady=all: M=0 calls read 0 bytes; M=1 mean of 10,30,20
        self.assertEqual(s["bytes_by_m"][0]["mean"], 0.0)
        self.assertAlmostEqual(s["bytes_by_m"][1]["mean"], (10 + 30 + 20) / 3)


class TraceFileRoundTripTests(unittest.TestCase):
    def write_trace(self, path: Path):
        # v2 header: expert_bytes=100, cache_units=2, lru=1, pinned=0, flags=0
        records = [
            # prefill call 0 (layer 3, rows 0/1, expert 1 repeated across rows)
            (0, 0, 3, 0, 0, 1), (1, 0, 3, 0, 1, 2), (2, 0, 3, 1, 0, 1),
            # prefill call 1 (layer 4)
            (3, 1, 4, 0, 0, 7), (4, 1, 4, 1, 0, 7),
            # token 0
            (5, 2, 3, 0, 0, 1), (6, 2, 3, 0, 1, 3),
            (7, 3, 4, 0, 0, 7),
            # token 1
            (8, 4, 3, 0, 0, 3), (9, 4, 3, 0, 1, 2),
            (10, 5, 4, 0, 0, 8),
        ]
        with path.open("wb") as f:
            f.write(slotsim.HEADER_BASE.pack(slotsim.MAGIC, 2, slotsim.RECORD.size))
            f.write(slotsim.HEADER_V2_META.pack(100, 2, 1, 0, 0))
            for rec in records:
                f.write(slotsim.RECORD.pack(*rec))

    def test_load_calls_and_replay(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trace.bin"
            self.write_trace(path)
            meta, calls = bsim.load_calls(path)
            self.assertEqual(meta.expert_bytes, 100)
            self.assertEqual(len(calls), 6)
            self.assertEqual(calls[0].experts, [1, 2])  # batch union dedup
            self.assertEqual(calls[0].n_rows, 2)
            self.assertEqual(calls[1].experts, [7])
            token_of, n_tokens = bsim.segment_tokens(calls)
            self.assertEqual(n_tokens, 2)

            sizes = {(l, e): 100 for l in (3, 4) for e in range(10)}
            # LRU=1 slot: L3 prefill loads 1 then 2 -> cache {2}; token0
            # requests [1,3]: both miss (pre-insertion state {2}); token1
            # requests [3,2]: 3 hits (loaded by token0's apply), 2 misses.
            res = bsim.replay(calls, token_of, sizes, {}, "slots", 1,
                              collect_m=True)
            self.assertEqual(res.requests, 2 + 1 + 2 + 1 + 2 + 1)
            l3 = [c for c in res.call_m if c[1] == 3]
            self.assertEqual([c[2] for c in l3], [2, 1])

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            trace = td / "trace.bin"
            self.write_trace(trace)
            cfg = {"hidden": 8, "moe_inter": 4, "n_layers": 5,
                   "first_dense": 3, "n_experts": 10}
            manifest = {"model_config": cfg, "force_int4_layers": [3],
                        "int2": [[4, 7]]}
            mpath = td / "manifest.json"
            mpath.write_text(json.dumps(manifest))
            upath = td / "usage.txt"
            upath.write_text("3 1 50\n4 7 40\n")
            out = td / "out"
            rc = bsim.main([
                str(trace), "--usage", str(upath), "--manifest", str(mpath),
                "--layouts", "A,B,C,D", "--int4-bytes", "100",
                "--int2-bytes", "50", "--pin-bytes", "100",
                "--lru-slots", "1", "--warmup-tokens", "1",
                "--miss-structure", "--outdir", str(out),
            ])
            self.assertEqual(rc, 0)
            for name in ("layout-grid.csv", "tok-s-grid.csv", "config.json",
                         "missstruct-pm.csv", "missstruct-summary.csv",
                         "pinset-A.csv", "per-token-A-trace.csv"):
                self.assertTrue((out / name).exists(), name)
            grid = (out / "layout-grid.csv").read_text().splitlines()
            self.assertEqual(len(grid), 1 + 4)  # header + 4 layouts x 1 trace
            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["int4_bytes"], 100)
            # pin fill: budget 100 B -> under A only (3,1) fits (100 B);
            # under C, (3,1) stays int4 (forced layer) -> still 1 pin; the
            # pinset CSVs must record the layout-specific fill.
            pins_a = (out / "pinset-A.csv").read_text().splitlines()
            self.assertEqual(len(pins_a), 2)


class TokPerSecondMathTests(unittest.TestCase):
    def test_formula(self):
        # t = C + B/BW: 3.6e9 B/tok at 10 GB/s and C=0.36 -> 0.72 s -> 1.389
        c, b, bw = 0.36, 3.6e9, 10.0
        t = c + b / (bw * 1e9)
        self.assertAlmostEqual(t, 0.72)
        self.assertAlmostEqual(1 / t, 1.3889, places=4)


if __name__ == "__main__":
    unittest.main()
