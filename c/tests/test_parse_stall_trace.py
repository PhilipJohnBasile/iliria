"""tools/parse_stall_trace.py: synthetic STALL_TRACE line parsing and
per-layer aggregation (mean, percentiles, CVaR, hideable/exposed derivation).
No engine, no compilation -- pure text fixtures matching the exact line
format tools/patch_m5max_stall_trace.py emits."""

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


pst = load_module("parse_stall_trace", ROOT / "tools" / "parse_stall_trace.py")


def line(layer, nhit, nmiss, resident_start, resident_finish, miss_issue,
         miss_complete_max, reduction_start, exposed, fwd=0, nu=None):
    nu = nu if nu is not None else nhit + nmiss
    return (
        f"STALL_TRACE fwd={fwd} layer={layer} nu={nu} nhit={nhit} nmiss={nmiss} "
        f"resident_start_ms={resident_start:.4f} resident_finish_ms={resident_finish:.4f} "
        f"miss_issue_ms={miss_issue:.4f} miss_complete_max_ms={miss_complete_max:.4f} "
        f"reduction_start_ms={reduction_start:.4f} exposed_stall_ms={exposed:.4f}\n"
    )


class ParseLineTest(unittest.TestCase):
    def test_parses_all_fields_with_correct_types(self):
        record = pst.parse_line(line(7, 4, 2, 0.012, 0.450, 0.015, 0.980, 0.400, 0.530, fwd=142))
        self.assertEqual(record["fwd"], 142)
        self.assertEqual(record["layer"], 7)
        self.assertEqual(record["nhit"], 4)
        self.assertEqual(record["nmiss"], 2)
        self.assertIsInstance(record["nhit"], int)
        self.assertAlmostEqual(record["resident_start_ms"], 0.012)
        self.assertAlmostEqual(record["exposed_stall_ms"], 0.530)

    def test_ignores_non_matching_lines(self):
        self.assertIsNone(pst.parse_line("[stall-trace] enabled: ...\n"))
        self.assertIsNone(pst.parse_line("PROFILE: expert-disk 0.001s\n"))
        self.assertIsNone(pst.parse_line("\n"))

    def test_sentinel_negative_one_parses_as_a_plain_float(self):
        record = pst.parse_line(line(3, 0, 2, -1.0, -1.0, 0.0, 0.038, -1.0, 0.038))
        self.assertEqual(record["resident_start_ms"], -1.0)
        self.assertEqual(record["reduction_start_ms"], -1.0)

    def test_missing_field_raises_value_error(self):
        with self.assertRaises(ValueError):
            pst.parse_line("STALL_TRACE fwd=0 layer=3 nu=2 nhit=0 nmiss=2\n")


class CvarTest(unittest.TestCase):
    def test_cvar95_is_mean_of_worst_5_percent(self):
        values = list(range(1, 101))  # 1..100
        # worst 5% of 100 samples = top 5 values = 96..100, mean 98
        self.assertAlmostEqual(pst.cvar([float(v) for v in values], 0.95), 98.0)

    def test_cvar_with_few_samples_is_the_single_worst_value(self):
        self.assertAlmostEqual(pst.cvar([1.0, 5.0, 2.0], 0.95), 5.0)

    def test_cvar_empty_is_zero(self):
        self.assertEqual(pst.cvar([], 0.95), 0.0)


class AggregateLayerTest(unittest.TestCase):
    def test_hideable_and_exposed_when_resident_covers_the_miss(self):
        # resident window [0, 10] fully covers a miss completing at 6ms -> exposed 0
        records = [pst.parse_line(line(3, 1, 1, 0.0, 10.0, 0.0, 6.0, 1.0, 0.0))]
        agg = pst.aggregate_layer(records)
        self.assertAlmostEqual(agg["resident_compute_duration_ms"], 10.0)
        self.assertAlmostEqual(agg["final_read_completion_ms"], 6.0)
        self.assertAlmostEqual(agg["hideable_ms"], 6.0)  # min(10, 6)
        self.assertAlmostEqual(agg["exposed_stall_ms"], 0.0)
        self.assertEqual(agg["fast_path_eligible"], "true")

    def test_hideable_and_exposed_when_miss_outlasts_resident(self):
        # resident window [0, 2] finishes long before the miss lands at 20ms -> exposed 18ms
        records = [pst.parse_line(line(3, 1, 1, 0.0, 2.0, 0.0, 20.0, 1.0, 18.0))]
        agg = pst.aggregate_layer(records)
        self.assertAlmostEqual(agg["hideable_ms"], 2.0)  # min(2, 20)
        self.assertAlmostEqual(agg["exposed_stall_ms"], 18.0)
        self.assertEqual(agg["fast_path_eligible"], "false")

    def test_no_resident_no_miss_samples_report_the_na_sentinel(self):
        records = [pst.parse_line(line(3, 2, 0, 0.01, 0.02, -1.0, -1.0, 0.005, 0.0))]
        agg = pst.aggregate_layer(records)
        self.assertEqual(agg["final_read_completion_ms"], pst.PENDING_NA)
        self.assertEqual(agg["fast_path_eligible"], "true")  # zero misses -> trivially fast-path

    def test_all_miss_block_uses_route_ready_fallback_and_resident_na(self):
        # engine reports resident_start/finish as -1 (no hits this block)
        records = [pst.parse_line(line(3, 0, 2, -1.0, -1.0, 0.0, 0.038, -1.0, 0.038))]
        agg = pst.aggregate_layer(records)
        self.assertEqual(agg["resident_compute_duration_ms"], pst.PENDING_NA)
        self.assertAlmostEqual(agg["final_read_completion_ms"], 0.038)
        self.assertAlmostEqual(agg["hideable_ms"], 0.0)  # no resident duration to hide behind
        self.assertAlmostEqual(agg["exposed_stall_ms"], 0.038)

    def test_distribution_columns_over_multiple_samples(self):
        exposed_values = [0.0, 1.0, 2.0, 3.0, 100.0]
        records = [
            pst.parse_line(line(5, 1, 1, 0.0, 1.0, 0.0, 1.0 + v, 0.5, v))
            for v in exposed_values
        ]
        agg = pst.aggregate_layer(records)
        self.assertEqual(agg["n_samples"], 5)
        self.assertAlmostEqual(agg["exposed_stall_mean_ms"], sum(exposed_values) / 5)
        self.assertAlmostEqual(agg["exposed_stall_median_ms"], 2.0)
        self.assertAlmostEqual(agg["exposed_stall_max_ms"], 100.0)
        self.assertGreaterEqual(agg["exposed_stall_cvar95_ms"], agg["exposed_stall_p95_ms"])


class BuildRowsAndCsvTest(unittest.TestCase):
    def test_rows_grouped_by_layer_and_sorted(self):
        records = [
            pst.parse_line(line(7, 1, 0, 0.0, 0.01, -1.0, -1.0, 0.005, 0.0)),
            pst.parse_line(line(3, 0, 1, -1.0, -1.0, 0.0, 0.03, -1.0, 0.03)),
            pst.parse_line(line(3, 1, 0, 0.0, 0.01, -1.0, -1.0, 0.005, 0.0)),
        ]
        rows = pst.build_rows(records)
        self.assertEqual([r["layer"] for r in rows], [3, 7])
        self.assertEqual(rows[0]["n_samples"], 2)
        self.assertEqual(rows[1]["n_samples"], 1)

    def test_csv_columns_match_v1_schema_plus_extras(self):
        self.assertEqual(pst.CSV_COLUMNS[:len(pst.LOM.CSV_COLUMNS)], pst.LOM.CSV_COLUMNS)
        for extra in ("n_samples", "exposed_stall_mean_ms", "exposed_stall_cvar95_ms"):
            self.assertIn(extra, pst.CSV_COLUMNS)

    def test_write_csv_round_trips(self):
        records = [pst.parse_line(line(3, 1, 1, 0.0, 1.0, 0.0, 0.5, 0.2, 0.0))]
        rows = pst.build_rows(records)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "layer-opportunity-matrix-stall.csv"
            pst.write_csv(rows, out)
            text = out.read_text()
        header = text.splitlines()[0].split(",")
        self.assertEqual(header, pst.CSV_COLUMNS)
        self.assertIn("3,", text)


class MainCliTest(unittest.TestCase):
    def test_refuses_to_write_the_v1_filename(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "trace.log"
            log.write_text(line(3, 1, 1, 0.0, 1.0, 0.0, 0.5, 0.2, 0.0))
            out = Path(td) / "layer-opportunity-matrix.csv"
            old_argv = sys.argv
            sys.argv = ["parse_stall_trace.py", "--log", str(log), "--out", str(out)]
            try:
                with self.assertRaises(SystemExit):
                    pst.main()
            finally:
                sys.argv = old_argv
            self.assertFalse(out.exists())

    def test_end_to_end_multi_log_multi_layer(self):
        with tempfile.TemporaryDirectory() as td:
            log1 = Path(td) / "a.log"
            log2 = Path(td) / "b.log"
            log1.write_text(
                "[stall-trace] enabled: one STALL_TRACE line per decode layer per token to stderr\n"
                + line(3, 0, 2, -1.0, -1.0, 0.0, 12.07, -1.0, 12.07)
                + line(4, 1, 1, 0.02, 0.03, 0.0, 0.019, 0.025, 0.0)
            )
            log2.write_text(line(3, 1, 1, 0.03, 0.06, 0.0, 0.10, 0.05, 0.04))
            out = Path(td) / "layer-opportunity-matrix-stall.csv"
            old_argv = sys.argv
            sys.argv = ["parse_stall_trace.py", "--log", str(log1), "--log", str(log2),
                       "--out", str(out)]
            try:
                pst.main()
            finally:
                sys.argv = old_argv
            rows = out.read_text().strip().splitlines()
            self.assertEqual(len(rows), 3)  # header + layer 3 + layer 4
            layer3 = [r for r in rows if r.startswith("3,")][0].split(",")
            header = rows[0].split(",")
            n_samples = layer3[header.index("n_samples")]
            self.assertEqual(n_samples, "2")  # one from each log


if __name__ == "__main__":
    unittest.main()
