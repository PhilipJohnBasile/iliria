from __future__ import annotations

import csv
import importlib.util
import json
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


bsim = load_module("simulate_bytes_cache", ROOT / "tools" / "simulate_bytes_cache.py")
ear = load_module("entropy_aware_replay", ROOT / "tools" / "entropy_aware_replay.py")
slotsim = ear.bsim.slotsim


def make_calls(spec):
    """spec: [(layer, [experts], n_rows)] -> [Call] with sequential ids."""
    return [bsim.Call(i, layer, list(experts), n_rows, len(experts))
            for i, (layer, experts, n_rows) in enumerate(spec)]


def token_map(calls):
    return bsim.segment_tokens(calls)


def make_ratio_table(early=0.70, mid=0.72, late=0.74, coder="rans", block=65536, n=85):
    """A tiny, complete synthetic census_summary dict: distinct ratios per
    band so band-conditioning is actually exercised, same ratio across the
    3 projections within a band (kept simple; per-projection variation is
    covered by test_predicted_band_bytes_matches_hand_calc instead)."""
    band_ratio = {"early": early, "mid": mid, "late": late}
    percentiles = {}
    for band, ratio in band_ratio.items():
        for proj in ("gate", "up", "down"):
            percentiles[f"{band}/{proj}/{coder}/{block}"] = {
                "p50": ratio, "p90": ratio + 0.05, "p99": ratio + 0.06, "n": n,
            }
    return {
        "n_experts": 256,
        "headline": {"best_coder": coder, "best_block_size": block},
        "percentiles": percentiles,
    }


TINY_CFG = {"hidden": 8, "moe_inter": 4, "n_layers": 6, "first_dense": 3, "n_experts": 2}


class LoadRatioTableTests(unittest.TestCase):
    def test_defaults_to_headline_coder_and_block(self):
        summary = make_ratio_table()
        table, coder, block = ear.load_ratio_table(summary)
        self.assertEqual((coder, block), ("rans", 65536))
        self.assertEqual(table[("early", "gate")]["p50"], 0.70)
        self.assertEqual(table[("mid", "down")]["p50"], 0.72)
        self.assertEqual(table[("late", "up")]["p90"], 0.79)
        # all 9 (band, proj) cells present
        self.assertEqual(len(table), 9)

    def test_override_coder_and_block(self):
        summary = make_ratio_table(coder="huff", block=4096)
        summary["percentiles"]["early/gate/rans/65536"] = {"p50": 0.99, "p90": 0.99, "n": 1}
        table, coder, block = ear.load_ratio_table(summary, coder="huff", block=4096)
        self.assertEqual((coder, block), ("huff", 4096))
        self.assertEqual(table[("early", "gate")]["p50"], 0.70)

    def test_missing_cell_raises(self):
        summary = make_ratio_table()
        del summary["percentiles"]["late/down/rans/65536"]
        with self.assertRaises(KeyError):
            ear.load_ratio_table(summary)


class PredictedBandBytesTests(unittest.TestCase):
    def test_matches_hand_calc_per_projection(self):
        cfg = {"hidden": 8, "moe_inter": 4}
        # gate/up: O,I = moe_inter,hidden = 4,8 -> packed_nbytes(4,8,4) = 4*4=16
        # down:    O,I = hidden,moe_inter = 8,4 -> packed_nbytes(8,4,4) = 8*2=16
        ratio_table = {
            ("early", "gate"): {"p50": 0.5, "p90": 0.9},
            ("early", "up"): {"p50": 0.6, "p90": 0.9},
            ("early", "down"): {"p50": 0.7, "p90": 0.9},
            ("mid", "gate"): {"p50": 0.5, "p90": 0.5},
            ("mid", "up"): {"p50": 0.5, "p90": 0.5},
            ("mid", "down"): {"p50": 0.5, "p90": 0.5},
            ("late", "gate"): {"p50": 0.5, "p90": 0.5},
            ("late", "up"): {"p50": 0.5, "p90": 0.5},
            ("late", "down"): {"p50": 0.5, "p90": 0.5},
        }
        out = ear.predicted_band_bytes(cfg, ratio_table, "p50")
        # gate: 16*0.5 + 4*4 (O=4 scale rows) = 8 + 16 = 24
        # up:   16*0.6 + 16 = 9.6 + 16 = 25.6
        # down: 16*0.7 + 8*4 (O=8 scale rows) = 11.2 + 32 = 43.2
        self.assertAlmostEqual(out["early"], 24 + 25.6 + 43.2)

    def test_p90_stat_uses_p90_column(self):
        cfg = {"hidden": 8, "moe_inter": 4}
        ratio_table = {
            (b, p): {"p50": 0.5, "p90": 0.9}
            for b in ("early", "mid", "late") for p in ("gate", "up", "down")
        }
        median = ear.predicted_band_bytes(cfg, ratio_table, "p50")
        pess = ear.predicted_band_bytes(cfg, ratio_table, "p90")
        for band in ("early", "mid", "late"):
            self.assertGreater(pess[band], median[band])

    def test_all_three_bands_present(self):
        cfg = {"hidden": 8, "moe_inter": 4}
        ratio_table = {
            (b, p): {"p50": 0.7, "p90": 0.8}
            for b in ("early", "mid", "late") for p in ("gate", "up", "down")
        }
        out = ear.predicted_band_bytes(cfg, ratio_table, "p50")
        self.assertEqual(set(out), {"early", "mid", "late"})


class BandOfBoundaryTests(unittest.TestCase):
    def test_real_band_boundaries(self):
        self.assertEqual(ear.band_of(3), "early")
        self.assertEqual(ear.band_of(27), "early")
        self.assertEqual(ear.band_of(28), "mid")
        self.assertEqual(ear.band_of(52), "mid")
        self.assertEqual(ear.band_of(53), "late")
        self.assertEqual(ear.band_of(77), "late")

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            ear.band_of(2)
        with self.assertRaises(ValueError):
            ear.band_of(78)


class RoutedPairsAndAssignmentTests(unittest.TestCase):
    def test_routed_pairs_count_and_shape(self):
        pairs = ear.routed_pairs(TINY_CFG)
        # layers 3,4,5 (all "early"); 2 experts each
        self.assertEqual(len(pairs), 3 * 2)
        self.assertEqual(set(l for l, _ in pairs), {3, 4, 5})
        self.assertEqual(set(e for _, e in pairs), {0, 1})

    def test_real_cfg_shape_gives_19200(self):
        cfg = {"hidden": 6144, "moe_inter": 2048, "n_layers": 78,
               "first_dense": 3, "n_experts": 256}
        self.assertEqual(len(ear.routed_pairs(cfg)), 19200)

    def test_assign_predicted_sizes_every_expert_gets_its_band(self):
        band_bytes = {"early": 111.4, "mid": 222.6, "late": 333.0}
        sizes = ear.assign_predicted_sizes(TINY_CFG, band_bytes)
        self.assertEqual(len(sizes), 6)
        for (l, e), b in sizes.items():
            self.assertEqual(b, round(band_bytes["early"]))  # layers 3-5 all early

    def test_assign_predicted_sizes_spans_bands(self):
        cfg = {"hidden": 8, "moe_inter": 4, "n_layers": 30,
               "first_dense": 3, "n_experts": 1}
        band_bytes = {"early": 100.0, "mid": 200.0, "late": 300.0}
        sizes = ear.assign_predicted_sizes(cfg, band_bytes)
        self.assertEqual(sizes[(3, 0)], 100)
        self.assertEqual(sizes[(27, 0)], 100)
        self.assertEqual(sizes[(28, 0)], 200)
        self.assertEqual(sizes[(29, 0)], 200)


class CaseSizeBuilderTests(unittest.TestCase):
    def test_uniform_sizes(self):
        sizes = ear.uniform_sizes(TINY_CFG, 42)
        self.assertEqual(set(sizes.values()), {42})
        self.assertEqual(set(sizes), set(ear.routed_pairs(TINY_CFG)))

    def test_hybrid_sizes_scalar_partition(self):
        forced = {3}
        sizes = ear.hybrid_sizes(TINY_CFG, forced, cold_size=5, protected_sizes=9)
        for (l, e), b in sizes.items():
            self.assertEqual(b, 9 if l == 3 else 5)
        self.assertEqual(set(sizes), set(ear.routed_pairs(TINY_CFG)))

    def test_hybrid_sizes_dict_partition(self):
        forced = {3, 4}
        protected = {(3, 0): 11, (3, 1): 12, (4, 0): 13, (4, 1): 14}
        sizes = ear.hybrid_sizes(TINY_CFG, forced, cold_size=5,
                                  protected_sizes=protected)
        self.assertEqual(sizes[(3, 0)], 11)
        self.assertEqual(sizes[(3, 1)], 12)
        self.assertEqual(sizes[(4, 0)], 13)
        self.assertEqual(sizes[(5, 0)], 5)  # cold
        self.assertEqual(sizes[(5, 1)], 5)


class BuildCasesTests(unittest.TestCase):
    def setUp(self):
        self.forced = {3}
        self.entropy_median = {"early": 9.0, "mid": 10.0, "late": 11.0}
        self.entropy_p90 = {"early": 9.5, "mid": 10.5, "late": 11.5}
        self.cases = ear.build_cases(
            TINY_CFG, self.forced, int4_b=20, int2_b=10,
            entropy_median=self.entropy_median, entropy_p90=self.entropy_p90,
            int2_cons_ratio=0.95, int2_opt_ratio=0.85)
        self.by_name = {c.name: c for c in self.cases}

    def test_expected_case_names_present(self):
        expected = {"1", "2", "2-p90", "2-mode1.5", "3", "3-p90", "4", "5",
                    "5-p90", "5-mode1.5", "6a-cons", "6b-opt"}
        self.assertEqual(set(self.by_name), expected)

    def test_lru_modes(self):
        self.assertEqual(self.by_name["1"].lru_mode, "slots")
        self.assertEqual(self.by_name["2"].lru_mode, "slots")
        self.assertEqual(self.by_name["2-mode1.5"].lru_mode, "slots")
        self.assertEqual(self.by_name["3"].lru_mode, "bytes")
        self.assertEqual(self.by_name["4"].lru_mode, "bytes")
        self.assertEqual(self.by_name["5"].lru_mode, "bytes")
        self.assertEqual(self.by_name["5-mode1.5"].lru_mode, "bytes")
        self.assertEqual(self.by_name["6a-cons"].lru_mode, "bytes")

    def test_mode2_3_cases_have_equal_resident_and_transfer(self):
        for name in ("1", "2", "3", "4", "5", "6a-cons", "6b-opt"):
            case = self.by_name[name]
            self.assertEqual(case.resident_sizes, case.transfer_sizes, name)

    def test_mode1_5_cases_diverge_only_on_protected_tier(self):
        m15_2 = self.by_name["2-mode1.5"]
        case1 = self.by_name["1"]
        case2 = self.by_name["2"]
        self.assertEqual(m15_2.resident_sizes, case1.resident_sizes)
        self.assertEqual(m15_2.transfer_sizes, case2.resident_sizes)
        self.assertNotEqual(m15_2.resident_sizes, m15_2.transfer_sizes)

        m15_5 = self.by_name["5-mode1.5"]
        case4 = self.by_name["4"]
        case5 = self.by_name["5"]
        self.assertEqual(m15_5.resident_sizes, case4.resident_sizes)
        self.assertEqual(m15_5.transfer_sizes, case5.resident_sizes)

    def test_case1_is_uniform_int4(self):
        self.assertEqual(set(self.by_name["1"].resident_sizes.values()), {20})

    def test_case4_partitions_int2_int4(self):
        sizes = self.by_name["4"].resident_sizes
        for (l, e), b in sizes.items():
            self.assertEqual(b, 20 if l in self.forced else 10)

    def test_case5_protected_tier_is_entropy_coded_not_raw(self):
        sizes = self.by_name["5"].resident_sizes
        for (l, e), b in sizes.items():
            if l in self.forced:
                self.assertEqual(b, round(self.entropy_median["early"]))
                self.assertNotEqual(b, 20)  # not raw int4
            else:
                self.assertEqual(b, 10)  # still repaired int2

    def test_case6_int2_ratios_applied_to_cold_tier_only(self):
        cons = self.by_name["6a-cons"].resident_sizes
        opt = self.by_name["6b-opt"].resident_sizes
        for (l, e) in ear.routed_pairs(TINY_CFG):
            if l in self.forced:
                # protected tier: same entropy-coded size as case 5, in both
                self.assertEqual(cons[(l, e)], round(self.entropy_median["early"]))
                self.assertEqual(opt[(l, e)], round(self.entropy_median["early"]))
            else:
                self.assertEqual(cons[(l, e)], round(10 * 0.95))
                self.assertEqual(opt[(l, e)], round(10 * 0.85))
        self.assertLess(round(10 * 0.85), round(10 * 0.95))  # optimistic = smaller


class ReplayDualConsistencyTests(unittest.TestCase):
    def test_matches_bsim_replay_when_dicts_equal(self):
        calls = make_calls([
            (3, [1, 2], 1), (4, [7], 1),
            (3, [1, 3], 1), (4, [7], 1),
            (3, [2], 1), (4, [8], 1),
        ])
        token_of, _ = token_map(calls)
        sizes = {(3, e): 100 + e for e in range(5)}
        sizes.update({(4, e): 200 + e for e in range(9)})
        pinned = {(3, 1): sizes[(3, 1)]}

        want = bsim.replay(calls, token_of, sizes, pinned, "bytes", 250,
                            layout="X", trace="Y", collect_m=True)
        got = ear.replay_dual(calls, token_of, sizes, sizes, pinned, "bytes",
                               250, layout="X", trace="Y", collect_m=True)

        self.assertEqual(got.requests, want.requests)
        self.assertEqual(got.pin_hits, want.pin_hits)
        self.assertEqual(got.lru_hits, want.lru_hits)
        self.assertEqual(got.disk_misses, want.disk_misses)
        self.assertEqual(got.disk_bytes, want.disk_bytes)
        self.assertEqual(got.per_token_bytes, want.per_token_bytes)
        self.assertEqual(got.per_token_misses, want.per_token_misses)
        self.assertEqual(got.call_m, want.call_m)

    def test_slot_mode_also_matches(self):
        calls = make_calls([(3, [1], 1), (3, [2], 1), (3, [1], 1), (3, [3], 1)])
        token_of, _ = token_map(calls)
        sizes = {(3, e): 10 * e for e in range(5)}
        want = bsim.replay(calls, token_of, sizes, {}, "slots", 1)
        got = ear.replay_dual(calls, token_of, sizes, sizes, {}, "slots", 1)
        self.assertEqual((got.lru_hits, got.disk_misses, got.disk_bytes),
                          (want.lru_hits, want.disk_misses, want.disk_bytes))

    def test_mode1_5_classification_follows_resident_bytes_charged_follows_transfer(self):
        # Budget 100 B. Raw (resident) size 60 B per expert -> only one
        # resident at a time, cyclic access always misses (matches
        # test_simulate_bytes_cache's byte-mode capacity-gain fixture).
        # Compressed (transfer) size 24 B: same miss PATTERN as the raw
        # replay (residency governed by resident=60B), but each of those
        # misses now only charges 24 B instead of 60 B.
        calls = make_calls([(3, [1 + (i % 2)], 1) for i in range(20)])
        token_of, _ = token_map(calls)
        resident = {(3, 1): 60, (3, 2): 60}
        transfer = {(3, 1): 24, (3, 2): 24}

        raw_baseline = bsim.replay(calls, token_of, resident, {}, "bytes", 100)
        got = ear.replay_dual(calls, token_of, resident, transfer, {}, "bytes", 100)

        self.assertEqual(got.lru_hits, raw_baseline.lru_hits)
        self.assertEqual(got.disk_misses, raw_baseline.disk_misses)
        self.assertEqual(raw_baseline.lru_hits, 0)  # sanity: fixture is all-miss
        self.assertEqual(got.disk_misses, 20)
        self.assertEqual(got.disk_bytes, 20 * 24)
        self.assertEqual(raw_baseline.disk_bytes, 20 * 60)

    def test_mode1_5_pins_use_resident_semantics(self):
        # A pair pinned under resident sizing must never cost transfer
        # bytes -- it is never read from disk at all.
        calls = make_calls([(3, [1], 1), (3, [1], 1)])
        token_of, _ = token_map(calls)
        resident = {(3, 1): 60}
        transfer = {(3, 1): 5}
        pinned = {(3, 1): 60}
        res = ear.replay_dual(calls, token_of, resident, transfer, pinned,
                               "bytes", 100)
        self.assertEqual(res.pin_hits, 2)
        self.assertEqual(res.disk_bytes, 0)


class RealCensusSummaryShapeTests(unittest.TestCase):
    """Light integration check against the committed (tracked-in-git) census
    output -- guards the parser against real-world schema drift without
    depending on any of the untracked, local-only bench trace/usage/manifest
    files the full replay needs."""

    CENSUS_PATH = ROOT / "bench-m5max" / "new-math-20260715" / "census-summary.json"

    @unittest.skipUnless(CENSUS_PATH.exists(), "committed census summary not present")
    def test_real_file_parses_and_yields_sane_predicted_sizes(self):
        summary = json.loads(self.CENSUS_PATH.read_text())
        table, coder, block = ear.load_ratio_table(summary)
        self.assertEqual(len(table), 9)
        for (band, proj), stats in table.items():
            self.assertGreater(stats["p50"], 0.0)
            self.assertLess(stats["p50"], 1.0)
            self.assertLessEqual(stats["p50"], stats["p90"])

        cfg = {"hidden": 6144, "moe_inter": 2048, "n_layers": 78,
               "first_dense": 3, "n_experts": 256}
        int4_b = bsim.expert_bytes_per_row_scales(cfg, 4)
        median = ear.predicted_band_bytes(cfg, table, "p50")
        p90 = ear.predicted_band_bytes(cfg, table, "p90")
        int2_b = bsim.expert_bytes_per_row_scales(cfg, 2)
        for band in ("early", "mid", "late"):
            # compressed predictions sit strictly between int2 and int4 raw
            self.assertLess(median[band], int4_b)
            self.assertGreater(median[band], int2_b)
            self.assertGreaterEqual(p90[band], median[band])


class CliEndToEndTests(unittest.TestCase):
    def write_trace(self, path: Path):
        # v2 header: expert_bytes=0 (unchecked), cache_units=2, lru=1,
        # pinned=1, flags=0. Expert ids stay within {0, 1} to match the
        # tiny n_experts=2 config below. Shape mirrors
        # test_simulate_bytes_cache.py's write_trace fixture: 2 multi-row
        # prefill calls then 2 decode tokens (layer 3 then layer 4 each).
        records = [
            # prefill call 0 (layer 3, rows 0/1)
            (0, 0, 3, 0, 0, 0), (1, 0, 3, 0, 1, 1), (2, 0, 3, 1, 0, 0),
            # prefill call 1 (layer 4, rows 0/1)
            (3, 1, 4, 0, 0, 1), (4, 1, 4, 1, 0, 1),
            # token 0
            (5, 2, 3, 0, 0, 0), (6, 2, 3, 0, 1, 1),
            (7, 3, 4, 0, 0, 1),
            # token 1
            (8, 4, 3, 0, 0, 1), (9, 4, 3, 0, 1, 0),
            (10, 5, 4, 0, 0, 0),
        ]
        with path.open("wb") as f:
            f.write(slotsim.HEADER_BASE.pack(slotsim.MAGIC, 2, slotsim.RECORD.size))
            f.write(slotsim.HEADER_V2_META.pack(0, 2, 1, 1, 0))
            for rec in records:
                f.write(slotsim.RECORD.pack(*rec))

    def test_main_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            trace = td / "trace.bin"
            self.write_trace(trace)

            cfg = {"hidden": 8, "moe_inter": 4, "n_layers": 6,
                   "first_dense": 3, "n_experts": 2}
            manifest = {"model_config": cfg, "force_int4_layers": [3], "int2": []}
            (td / "manifest.json").write_text(json.dumps(manifest))
            (td / "usage.txt").write_text("3 0 50\n3 1 40\n4 0 30\n")
            (td / "census.json").write_text(json.dumps(make_ratio_table()))

            outdir = td / "out"
            rc = ear.main([
                "--traces", str(trace),
                "--usage", str(td / "usage.txt"),
                "--manifest", str(td / "manifest.json"),
                "--census-summary", str(td / "census.json"),
                "--warmup-tokens", "0",
                "--outdir", str(outdir),
            ])

            self.assertEqual(rc, 0)
            for name in ("case-grid.csv", "tok-s-grid.csv", "config.json",
                         "predicted-sizes-by-band.csv", "missstruct-pm.csv",
                         "missstruct-summary.csv", "pinset-1.csv", "pinset-5.csv"):
                self.assertTrue((outdir / name).exists(), name)

            with (outdir / "case-grid.csv").open() as f:
                grid = list(csv.DictReader(f))
            names = {row["case"] for row in grid}
            self.assertEqual(len(grid), 12)  # 12 cases x 1 trace
            self.assertIn("5", names)
            self.assertIn("6a-cons", names)

            config = json.loads((outdir / "config.json").read_text())
            self.assertEqual(config["int4_bytes"],
                              bsim.expert_bytes_per_row_scales(cfg, 4))
            self.assertIn("PREDICTED FROM SAMPLE", config["sizing_method"])


if __name__ == "__main__":
    unittest.main()
