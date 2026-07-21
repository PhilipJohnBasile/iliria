"""Tests for tools/provenance_compare.py -- the gate that asserts two scripts/provenance.sh
manifests agree on everything except a declared --vary set, and hard-fails (with a diff, and
a loud callout) on a binary_sha256 / generated_source_sha256 mismatch specifically, since that
is exactly what makes an A/B comparison meaningless (see that module's own docstring).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = C_DIR / "tools" / "provenance_compare.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pc = load_module("provenance_compare", MODULE_PATH)


def base_manifest(**overrides) -> dict:
    manifest = {
        "schema_version": 1,
        "attempt_id": "arm1",
        "generated_at": "2026-07-15T00:00:00Z",
        "start_ts": "2026-07-15T00:00:00Z",
        "end_ts": "2026-07-15T00:05:00Z",
        "git_commit": "abc123",
        "git_dirty": False,
        "binary_path": "/c/glm",
        "binary_sha256": "same-hash",
        "generated_source_sha256": {"glm_m5max.c": "src-hash-1", "backend_metal_m5max.mm": "src-hash-2"},
        "model_dir": "/models/int4",
        "container_manifest_hash": "container-hash-1",
        "pin_profile_hash": "pin-hash-1",
        "launcher_env": {"ILI_METAL_PREFILL": "0", "ILI_DIRECT": "1"},
        "launcher_env_digest": "env-digest-1",
        "prompt_or_dataset_hash": "prompt-hash-1",
        "prompt_or_dataset_source": "inline",
        "quiesce": {"bin": "/c/scripts/quiesce_check.sh", "skipped": False, "pass": True,
                    "exit_code": 0, "output": ["[quiesce] sample-that-always-differs"]},
        "manifest_path": "/out/provenance-arm1.json",
    }
    manifest.update(overrides)
    return manifest


class FlattenTests(unittest.TestCase):
    def test_flatten_uses_slash_and_does_not_confuse_a_dot_in_a_filename_key(self):
        obj = {"generated_source_sha256": {"glm_m5max.c": "h1"}, "model_dir": "/x"}
        flat = pc.flatten(obj)
        self.assertEqual(flat, {"generated_source_sha256/glm_m5max.c": "h1", "model_dir": "/x"})

    def test_flatten_keeps_lists_atomic(self):
        obj = {"quiesce": {"output": ["a", "b"]}}
        flat = pc.flatten(obj)
        self.assertEqual(flat, {"quiesce/output": ["a", "b"]})


class CompareManifestsTests(unittest.TestCase):
    def test_identical_manifests_match(self):
        a = base_manifest()
        b = base_manifest()
        result = pc.compare_manifests(a, b, vary=[])
        self.assertEqual(result["mismatches"], [])

    def test_only_always_ignored_fields_differ_still_matches(self):
        a = base_manifest()
        b = base_manifest(attempt_id="arm2", generated_at="2026-07-15T09:00:00Z",
                          start_ts="...", end_ts="...", manifest_path="/out/provenance-arm2.json",
                          quiesce={**a["quiesce"], "output": ["totally different live sample"]})
        result = pc.compare_manifests(a, b, vary=[])
        self.assertEqual(result["mismatches"], [])

    def test_planted_binary_sha_mismatch_is_caught_and_flagged_critical(self):
        a = base_manifest()
        b = base_manifest(binary_sha256="DIFFERENT-hash", binary_path="/c/glm")
        result = pc.compare_manifests(a, b, vary=[])
        paths = {m["path"] for m in result["mismatches"]}
        self.assertIn("binary_sha256", paths)
        critical = [m for m in result["mismatches"] if m["path"] == "binary_sha256"]
        self.assertEqual(len(critical), 1)
        self.assertTrue(critical[0]["critical"])

    def test_planted_generated_source_mismatch_is_caught_and_flagged_critical(self):
        a = base_manifest()
        b = base_manifest(generated_source_sha256={"glm_m5max.c": "DIFFERENT",
                                                    "backend_metal_m5max.mm": "src-hash-2"})
        result = pc.compare_manifests(a, b, vary=[])
        critical_paths = {m["path"] for m in result["mismatches"] if m["critical"]}
        self.assertIn("generated_source_sha256/glm_m5max.c", critical_paths)

    def test_generated_source_key_present_only_on_one_side_is_a_mismatch(self):
        """One arm built with mac-fast (glm_m5max.c present), the other never built (absent)
        -- exactly the kind of drift this tool must not silently ignore."""
        a = base_manifest()
        b = base_manifest(generated_source_sha256={"backend_metal_m5max.mm": "src-hash-2"})
        result = pc.compare_manifests(a, b, vary=[])
        mismatched_paths = {m["path"]: m for m in result["mismatches"]}
        self.assertIn("generated_source_sha256/glm_m5max.c", mismatched_paths)
        self.assertEqual(mismatched_paths["generated_source_sha256/glm_m5max.c"]["value_b"],
                         "<absent>")

    def test_vary_ili_metal_prefill_exempts_only_that_env_var_abba_case(self):
        a = base_manifest(launcher_env={"ILI_METAL_PREFILL": "0", "ILI_DIRECT": "1"})
        b = base_manifest(launcher_env={"ILI_METAL_PREFILL": "1", "ILI_DIRECT": "1"})
        without_vary = pc.compare_manifests(a, b, vary=[])
        self.assertTrue(any(m["path"] == "launcher_env/ILI_METAL_PREFILL"
                            for m in without_vary["mismatches"]))

        with_vary = pc.compare_manifests(a, b, vary=["ILI_METAL_PREFILL"])
        self.assertEqual(with_vary["mismatches"], [])
        self.assertTrue(any(e["path"] == "launcher_env/ILI_METAL_PREFILL"
                            for e in with_vary["exempted"]))

    def test_vary_ili_metal_prefill_still_catches_an_unrelated_binary_mismatch(self):
        """--vary must be narrow: declaring ILI_METAL_PREFILL as intentionally varying must
        NOT accidentally wave through an unrelated (and invalidating) binary difference."""
        a = base_manifest(launcher_env={"ILI_METAL_PREFILL": "0"})
        b = base_manifest(launcher_env={"ILI_METAL_PREFILL": "1"}, binary_sha256="DIFFERENT")
        result = pc.compare_manifests(a, b, vary=["ILI_METAL_PREFILL"])
        self.assertTrue(any(m["path"] == "binary_sha256" and m["critical"]
                            for m in result["mismatches"]))

    def test_vary_model_dir_also_implies_container_and_pin_hash_b0_vs_b1_case(self):
        a = base_manifest(model_dir="/models/int4", container_manifest_hash="c1", pin_profile_hash="p1")
        b = base_manifest(model_dir="/models/mixed", container_manifest_hash="c2", pin_profile_hash="p2")
        without_vary = pc.compare_manifests(a, b, vary=[])
        self.assertEqual({m["path"] for m in without_vary["mismatches"]},
                         {"model_dir", "container_manifest_hash", "pin_profile_hash"})

        with_vary = pc.compare_manifests(a, b, vary=["model_dir"])
        self.assertEqual(with_vary["mismatches"], [])
        exempted_paths = {e["path"] for e in with_vary["exempted"]}
        self.assertEqual(exempted_paths, {"model_dir", "container_manifest_hash", "pin_profile_hash"})

    def test_vary_by_full_path_also_works_not_only_final_segment(self):
        a = base_manifest(launcher_env={"ILI_METAL_PREFILL": "0"})
        b = base_manifest(launcher_env={"ILI_METAL_PREFILL": "1"})
        result = pc.compare_manifests(a, b, vary=["launcher_env/ILI_METAL_PREFILL"])
        self.assertEqual(result["mismatches"], [])

    def test_launcher_env_digest_itself_is_always_ignored_not_double_counted(self):
        """launcher_env_digest is derived from launcher_env and would differ whenever ANY env
        var differs, including a declared --vary'd one -- it must never independently cause a
        mismatch (the granular launcher_env/* comparison is the real, vary-aware mechanism)."""
        a = base_manifest(launcher_env={"ILI_METAL_PREFILL": "0"}, launcher_env_digest="d1")
        b = base_manifest(launcher_env={"ILI_METAL_PREFILL": "1"}, launcher_env_digest="d2")
        result = pc.compare_manifests(a, b, vary=["ILI_METAL_PREFILL"])
        self.assertEqual(result["mismatches"], [])
        self.assertFalse(any(e["path"] == "launcher_env_digest" for e in result["exempted"]))


class CriticalDetectionTests(unittest.TestCase):
    def test_is_critical_top_level_binary_sha256(self):
        self.assertTrue(pc.is_critical("binary_sha256"))

    def test_is_critical_nested_generated_source(self):
        self.assertTrue(pc.is_critical("generated_source_sha256/glm_m5max.c"))

    def test_is_critical_false_for_unrelated_field(self):
        self.assertFalse(pc.is_critical("model_dir"))
        self.assertFalse(pc.is_critical("launcher_env/ILI_METAL_PREFILL"))


class CliEndToEndTests(unittest.TestCase):
    def write(self, name, manifest) -> str:
        p = Path(self.td.name) / name
        p.write_text(json.dumps(manifest))
        return str(p)

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.td.cleanup()

    def run_cli(self, args):
        return subprocess.run([sys.executable, str(MODULE_PATH), *args],
                              capture_output=True, text=True, timeout=30)

    def test_matching_manifests_exit_zero(self):
        a = self.write("a.json", base_manifest())
        b = self.write("b.json", base_manifest(attempt_id="arm2"))
        proc = self.run_cli(["--a", a, "--b", b])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("MATCH", proc.stdout)

    def test_binary_mismatch_exits_one_with_critical_callout(self):
        a = self.write("a.json", base_manifest())
        b = self.write("b.json", base_manifest(binary_sha256="DIFFERENT"))
        proc = self.run_cli(["--a", a, "--b", b])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("MISMATCH", proc.stdout)
        self.assertIn("CRITICAL", proc.stdout)
        self.assertIn("binary_sha256", proc.stdout)

    def test_vary_flag_on_cli_passes_the_abba_case(self):
        a = self.write("a.json", base_manifest(launcher_env={"ILI_METAL_PREFILL": "0"}))
        b = self.write("b.json", base_manifest(launcher_env={"ILI_METAL_PREFILL": "1"}))
        without = self.run_cli(["--a", a, "--b", b])
        self.assertEqual(without.returncode, 1)
        with_vary = self.run_cli(["--a", a, "--b", b, "--vary", "ILI_METAL_PREFILL"])
        self.assertEqual(with_vary.returncode, 0, with_vary.stdout + with_vary.stderr)

    def test_missing_manifest_file_exits_two(self):
        b = self.write("b.json", base_manifest())
        proc = self.run_cli(["--a", "/no/such/manifest.json", "--b", b])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("error", proc.stderr.lower())

    def test_malformed_json_exits_two(self):
        bad = Path(self.td.name) / "bad.json"
        bad.write_text("{not valid json")
        b = self.write("b.json", base_manifest())
        proc = self.run_cli(["--a", str(bad), "--b", b])
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
