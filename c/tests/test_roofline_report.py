"""Unit tests for tools/roofline_report.py on synthetic PROFILE/telemetry fixtures (no
engine, no real GPU/model needed -- the module's own regexes are exercised against
hand-written text matching glm.c/backend_metal.mm's exact printf formats, and against its
own mock-log generator for a round trip)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "roofline_report.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rr = load_module("roofline_report", TOOL)


def make_log_text(
    tokens=128, wall_s=90.0, hit_pct=68.0, experts_per_token=8.2,
    t_edisk=38.0, t_ematmul=18.0, t_attn=17.0, t_kvb=1.7, t_head=3.0,
    metal=True, include_gemm=True,
) -> str:
    tok_s = tokens / wall_s
    other = max(0.0, wall_s - (t_edisk + t_ematmul + t_attn + t_head))
    lines = [
        "[m5max] RAM=114GB DRAFT=0 PIPE=1/8",
        f"\n---\n{tokens} tokens in {wall_s:.2f}s ({tok_s:.2f} tok/s) | expert hit rate {hit_pct:.1f}% | RSS 109.51 GB",
        f"experts loaded/token: {experts_per_token:.1f} (per-layer 1.07 across 75; baseline topk=8) | TOPK=8 TOPP=0.00",
        f"PROFILE: expert-disk {t_edisk:.3f}s | expert-matmul {t_ematmul:.3f}s | attention {t_attn:.3f}s "
        f"(including kvb {t_kvb:.3f}s) | lm_head {t_head:.3f}s | other {other:.3f}s",
    ]
    if metal:
        lines.append("METAL-ATTN: layer GPU 128 | gpu-wall 17.50s (kernel 11.80s | cpu-sched 2.30s gpu-sched 3.40s)")
        lines.append("METAL: blocchi GPU 128 | fallback CPU 0 | expert su GPU 719 | setup 2.68s gpu-wall 13.40s (kernel 10.22s) scatter 1.79s")
        if include_gemm:
            lines.append("METAL-GEMM: calls 4 | gpu-wall 6.11s (kernel 3.64s)")
    return "\n".join(lines) + "\n"


class ParseEngineLogTests(unittest.TestCase):
    def test_parses_all_fields_from_a_full_metal_log(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text(make_log_text())
            stats = rr.parse_engine_log(path)
        self.assertEqual(stats.tokens, 128)
        self.assertAlmostEqual(stats.wall_s, 90.0, places=2)
        self.assertAlmostEqual(stats.hit_pct, 68.0, places=2)
        self.assertAlmostEqual(stats.experts_per_token, 8.2, places=2)
        self.assertAlmostEqual(stats.t_edisk, 38.0, places=2)
        self.assertAlmostEqual(stats.t_ematmul, 18.0, places=2)
        self.assertAlmostEqual(stats.t_attn, 17.0, places=2)
        self.assertIsNotNone(stats.metal_attn)
        self.assertEqual(stats.metal_attn["ok"], 128)
        self.assertAlmostEqual(stats.metal_attn["kernel"], 11.80, places=2)
        self.assertIsNotNone(stats.metal_moe)
        self.assertEqual(stats.metal_moe["experts"], 719)
        self.assertIsNotNone(stats.metal_gemm)
        self.assertEqual(stats.metal_gemm["ok"], 4)

    def test_cpu_only_log_has_no_metal_fields_but_still_parses_cpu_profile(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text(make_log_text(metal=False))
            stats = rr.parse_engine_log(path)
        self.assertEqual(stats.tokens, 128)
        self.assertIsNone(stats.metal_attn)
        self.assertIsNone(stats.metal_moe)
        self.assertIsNone(stats.metal_gemm)

    def test_missing_gemm_line_leaves_metal_gemm_none_without_crashing(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text(make_log_text(metal=True, include_gemm=False))
            stats = rr.parse_engine_log(path)
        self.assertIsNotNone(stats.metal_attn)
        self.assertIsNotNone(stats.metal_moe)
        self.assertIsNone(stats.metal_gemm)

    def test_takes_last_occurrence_when_a_log_has_two_summaries(self):
        text = make_log_text(tokens=64, hit_pct=50.0) + "\n" + make_log_text(tokens=128, hit_pct=70.0)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text(text)
            stats = rr.parse_engine_log(path)
        self.assertEqual(stats.tokens, 128)
        self.assertAlmostEqual(stats.hit_pct, 70.0, places=2)

    def test_miss_and_hit_expert_derivation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text(make_log_text(tokens=100, hit_pct=50.0, experts_per_token=8.0))
            stats = rr.parse_engine_log(path)
        self.assertAlmostEqual(stats.miss_experts, 100 * 8.0 * 0.5, places=6)
        self.assertAlmostEqual(stats.hit_experts, 100 * 8.0 * 0.5, places=6)


class BandwidthMathTests(unittest.TestCase):
    def test_known_inputs_give_exact_read_path_bandwidth(self):
        # 100 tokens * 8 experts/token * 50% miss = 400 missed loads; 10 MB/expert =>
        # 4e9 bytes; 4.0s of expert-disk time => exactly 1.0 GB/s read-path bandwidth.
        stats = rr.EngineRunStats(log_path="x", tokens=100, experts_per_token=8.0, hit_pct=50.0, t_edisk=4.0)
        bw = rr.estimate_bandwidths([stats], bytes_per_expert_mb=10.0, device_bw_gbs=13.3, overlap_fraction=0.0)
        self.assertAlmostEqual(bw.compulsory_miss_bytes, 4.0e9, delta=1.0)
        self.assertAlmostEqual(bw.read_path_gbs, 1.0, places=6)
        self.assertAlmostEqual(bw.exposed_gbs, 1.0, places=6)
        self.assertEqual(bw.device_throughput_gbs, 13.3)

    def test_overlap_fraction_raises_exposed_above_read_path(self):
        stats = rr.EngineRunStats(log_path="x", tokens=100, experts_per_token=8.0, hit_pct=50.0, t_edisk=4.0)
        bw = rr.estimate_bandwidths([stats], bytes_per_expert_mb=10.0, overlap_fraction=0.5)
        # Same bytes, half the "exposed" stall time -> exposed bandwidth doubles vs read-path.
        self.assertAlmostEqual(bw.exposed_gbs, bw.read_path_gbs * 2.0, places=6)

    def test_missing_profile_data_reports_notes_instead_of_crashing(self):
        stats = rr.EngineRunStats(log_path="x")
        bw = rr.estimate_bandwidths([stats])
        self.assertIsNone(bw.compulsory_miss_bytes)
        self.assertIsNone(bw.read_path_gbs)
        self.assertIsNone(bw.exposed_gbs)
        self.assertTrue(any("unavailable" in n for n in bw.notes))


class ClassificationTests(unittest.TestCase):
    def test_low_occupancy_is_sync_ish_regardless_of_ai(self):
        cls, note = rr.classify(occupancy=0.2, ai=100.0)
        self.assertEqual(cls, "sync-ish")

    def test_high_ai_is_compute_ish(self):
        cls, note = rr.classify(occupancy=0.9, ai=50.0, ai_threshold=8.0)
        self.assertEqual(cls, "compute-ish")

    def test_low_ai_is_bandwidth_ish(self):
        cls, note = rr.classify(occupancy=0.9, ai=1.0, ai_threshold=8.0)
        self.assertEqual(cls, "bandwidth-ish")

    def test_no_occupancy_is_unknown_not_fabricated(self):
        cls, note = rr.classify(occupancy=None, ai=None)
        self.assertEqual(cls, "unknown")


class KernelClassTableTests(unittest.TestCase):
    def test_router_row_is_always_unmeasured(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text(make_log_text())
            stats = rr.parse_engine_log(path)
        rows = rr.build_kernel_class_table([stats])
        router = next(r for r in rows if r.name == "router")
        self.assertIn("not independently observable", router.classification)
        self.assertIsNone(router.calls)

    def test_four_required_kernel_classes_present(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text(make_log_text())
            stats = rr.parse_engine_log(path)
        rows = rr.build_kernel_class_table([stats])
        names = {r.name for r in rows}
        self.assertEqual(names, {"attention score/latent", "MoE GEMV", "router", "projections"})

    def test_projections_row_absent_calls_when_no_gemm_line(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text(make_log_text(include_gemm=False))
            stats = rr.parse_engine_log(path)
        rows = rr.build_kernel_class_table([stats])
        proj = next(r for r in rows if r.name == "projections")
        self.assertIsNone(proj.calls)
        self.assertEqual(proj.classification, "unknown")


class MockLogRoundTripTests(unittest.TestCase):
    def test_mock_log_parses_back_with_expected_shape(self):
        text = rr.gen_mock_log(phase="cold", seed=1, tokens=128)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mock.log"
            path.write_text(text)
            stats = rr.parse_engine_log(path)
        self.assertEqual(stats.tokens, 128)
        self.assertIsNotNone(stats.tok_s)
        self.assertIsNotNone(stats.metal_attn)
        self.assertIsNotNone(stats.metal_moe)
        self.assertIsNotNone(stats.metal_gemm)

    def test_steady_phase_is_seeded_slower_than_cold_on_average(self):
        # Not a single-draw guarantee (jitter is randomized), but the derate factor (0.90)
        # dominates the +-3% jitter, so the median over several seeds must show a real gap.
        import statistics

        def tok_s(phase: str, seed: int) -> float:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "mock.log"
                path.write_text(rr.gen_mock_log(phase, seed=seed))
                return rr.parse_engine_log(path).tok_s

        cold = [tok_s("cold", s) for s in range(10)]
        steady = [tok_s("steady", s) for s in range(10)]
        self.assertLess(statistics.median(steady), statistics.median(cold))


class TelemetryCsvTests(unittest.TestCase):
    def test_phase_summary_computes_median_and_ignores_blank_fields(self):
        rows = [
            {"phase": "cold", "thermal_speed_limit_pct": "", "cpu_idle_pct": "20", "disk_mb_s": "100.0"},
            {"phase": "cold", "thermal_speed_limit_pct": "", "cpu_idle_pct": "40", "disk_mb_s": "200.0"},
            {"phase": "steady", "thermal_speed_limit_pct": "80", "cpu_idle_pct": "10", "disk_mb_s": "50.0"},
        ]
        summ = rr.telemetry_phase_summary(rows, "cold")
        self.assertEqual(summ["samples"], 2)
        self.assertIsNone(summ["thermal_speed_limit_pct"]["median"])
        self.assertAlmostEqual(summ["cpu_idle_pct"]["median"], 30.0, places=6)
        self.assertAlmostEqual(summ["disk_mb_s"]["median"], 150.0, places=6)


class ReportCliTests(unittest.TestCase):
    def _build_result_dir(self, root: Path):
        (root / "logs").mkdir()
        (root / "telemetry").mkdir()
        runs = []
        for phase, hit, wall in (("cold", 70.0, 85.0), ("cold", 71.0, 86.0), ("steady", 65.0, 95.0), ("steady", 64.0, 96.0)):
            name = f"{phase}-{len(runs)}.log"
            (root / "logs" / name).write_text(make_log_text(hit_pct=hit, wall_s=wall))
            runs.append({"phase": phase, "label": "rust_queue", "trial": len(runs) + 1, "log": f"logs/{name}"})
        (root / "manifest.json").write_text(json.dumps({
            "attempt_id": "test-attempt", "mock": True, "runs": runs,
            "telemetry": {"cold": "telemetry/cold.csv", "steady": "telemetry/steady.csv"},
        }))
        for phase in ("cold", "steady"):
            (root / "telemetry" / f"{phase}.csv").write_text(
                "timestamp,phase,elapsed_s,thermal_speed_limit_pct,cpu_idle_pct,disk_mb_s,pageouts_cum\n"
                f"2026-01-01T00:00:00Z,{phase},0,,50,100.0,0\n"
            )

    def test_report_subcommand_produces_expected_sections(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._build_result_dir(root)
            out = root / "report.md"
            subprocess.run(
                [sys.executable, str(TOOL), "report", "--result-dir", str(root), "--out", str(out)],
                check=True, capture_output=True, text=True,
            )
            text = out.read_text()
        self.assertIn("# Dual roofline report", text)
        self.assertIn("## Thermal derate estimate", text)
        self.assertIn("## Bandwidth (three definitions)", text)
        self.assertIn("## Per-kernel-class roofline table", text)
        self.assertIn("attention score/latent", text)
        self.assertIn("MoE GEMV", text)
        self.assertIn("router", text)
        self.assertIn("projections", text)
        self.assertIn("## Method & limits", text)
        self.assertIn("No chip-peak GFLOP/s or GB/s figure is used", text)
        self.assertIn("steady median tok/s / cold median tok/s", text)

    def test_mock_log_cli_writes_a_parseable_file(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "mock.log"
            subprocess.run(
                [sys.executable, str(TOOL), "mock-log", "--out", str(out), "--phase", "steady", "--seed", "3"],
                check=True, capture_output=True, text=True,
            )
            stats = rr.parse_engine_log(out)
        self.assertIsNotNone(stats.tokens)
        self.assertIsNotNone(stats.metal_attn)


if __name__ == "__main__":
    unittest.main()
