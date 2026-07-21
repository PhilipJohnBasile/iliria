"""Tests for scripts/provenance.sh + tools/provenance_manifest.py: the executable provenance
manifest system (c/scripts/provenance.sh's own header comment has the full motivation and
field list). Two layers, matching the module split:

  * ProvenanceManifestUnitTests -- tools/provenance_manifest.py's functions imported directly
    (importlib, matching this test suite's dominant convention -- see e.g.
    test_compare_layer_captures.py), so each hashing/detection rule is checked in isolation.
  * ProvenanceScriptEndToEndTests -- the real scripts/provenance.sh, as a subprocess, proving
    the documented CLI contract: required-arg enforcement, --pid resolution against a REAL
    running process (the "CRITICAL" requirement in that script's header -- macOS has no
    /proc/<pid>/exe, so this is the concrete proof the chosen lsof/ps approach actually
    resolves to the right on-disk file), quiesce-bin override, --skip-quiesce, and that
    sourcing the script never executes anything or mutates the sourcing shell's options.

No real engine, real model directory, or real quiesce telemetry sampling is required for any
test here (--skip-quiesce / a fixture quiesce script stands in, exactly like
test_evening_orchestrator.py's own quiesce_pass.sh/quiesce_fail.sh fixtures).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
SCRIPT = C_DIR / "scripts" / "provenance.sh"
MODULE_PATH = C_DIR / "tools" / "provenance_manifest.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pm = load_module("provenance_manifest", MODULE_PATH)

EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version", "attempt_id", "generated_at", "start_ts", "end_ts", "git_commit",
    "git_dirty", "binary_path", "binary_sha256", "generated_source_sha256", "model_dir",
    "container_manifest_hash", "pin_profile_hash", "launcher_env", "launcher_env_digest",
    "effective_flags_predicted", "prompt_or_dataset_hash", "prompt_or_dataset_source",
    "quiesce", "manifest_path",
}
EXPECTED_QUIESCE_KEYS = {"bin", "skipped", "pass", "exit_code", "output"}


def write_script(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


class ProvenanceManifestUnitTests(unittest.TestCase):
    def test_sha256_file_matches_known_vector(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "f.txt"
            p.write_text("hello world")
            self.assertEqual(
                pm.sha256_file(p),
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9")

    def test_sha256_bytes_matches_known_vector_for_empty_string(self):
        self.assertEqual(
            pm.sha256_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    def test_generated_source_sha256_only_includes_files_that_exist(self):
        with tempfile.TemporaryDirectory() as td:
            c_dir = Path(td)
            self.assertEqual(pm.compute_generated_source_sha256(c_dir), {})

            (c_dir / "glm_m5max.c").write_text("/* generated */")
            result = pm.compute_generated_source_sha256(c_dir)
            self.assertEqual(set(result), {"glm_m5max.c"})
            self.assertEqual(result["glm_m5max.c"], pm.sha256_file(c_dir / "glm_m5max.c"))

            (c_dir / "backend_metal_m5max.mm").write_text("// generated")
            result2 = pm.compute_generated_source_sha256(c_dir)
            self.assertEqual(set(result2), {"glm_m5max.c", "backend_metal_m5max.mm"})

    def test_container_manifest_hash_none_when_model_dir_missing_or_absent(self):
        self.assertIsNone(pm.compute_container_manifest_hash(""))
        self.assertIsNone(pm.compute_container_manifest_hash(None))
        self.assertIsNone(pm.compute_container_manifest_hash("/no/such/directory/anywhere"))

    def test_container_manifest_hash_stable_across_identical_state(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text('{"a": 1}')
            (model_dir / "shard-0001.bin").write_bytes(b"x" * 100)
            h1 = pm.compute_container_manifest_hash(model_dir)
            h2 = pm.compute_container_manifest_hash(model_dir)
            self.assertEqual(h1, h2)
            self.assertIsNotNone(h1)

    def test_container_manifest_hash_changes_when_a_shard_size_changes(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text('{"a": 1}')
            (model_dir / "shard-0001.bin").write_bytes(b"x" * 100)
            before = pm.compute_container_manifest_hash(model_dir)
            (model_dir / "shard-0001.bin").write_bytes(b"x" * 101)  # size differs, bytes never read
            after = pm.compute_container_manifest_hash(model_dir)
            self.assertNotEqual(before, after)

    def test_container_manifest_hash_never_reads_shard_bytes(self):
        """A 'shard' whose content is unreadable garbage / not even valid to open as claimed
        must not matter: only its listed name/size/mtime are consulted. Simulated here by a
        shard file with no read permission -- if the implementation ever tried to read shard
        bytes this would raise PermissionError."""
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text('{"a": 1}')
            shard = model_dir / "shard-0001.bin"
            shard.write_bytes(b"x" * 100)
            os.chmod(shard, 0o000)
            try:
                h = pm.compute_container_manifest_hash(model_dir)
                self.assertIsNotNone(h)
            finally:
                os.chmod(shard, 0o644)

    def test_container_manifest_hash_distinguishes_config_absent_vs_present(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "shard-0001.bin").write_bytes(b"x" * 10)
            no_config = pm.compute_container_manifest_hash(model_dir)
            (model_dir / "config.json").write_text("{}")
            with_config = pm.compute_container_manifest_hash(model_dir)
            self.assertNotEqual(no_config, with_config)

    def test_pin_profile_hash_present_vs_absent(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            self.assertIsNone(pm.compute_pin_profile_hash(model_dir))
            self.assertIsNone(pm.compute_pin_profile_hash(""))
            (model_dir / ".fa_usage").write_bytes(b"expert-counters")
            h = pm.compute_pin_profile_hash(model_dir)
            self.assertEqual(h, hashlib.sha256(b"expert-counters").hexdigest())

    def test_launcher_env_filters_to_ili_coli_fa_prefixes_and_sorts(self):
        env_backup = dict(os.environ)
        try:
            for k in list(os.environ):
                if k.startswith(("ILI_", "COLI_", "FA_")):
                    del os.environ[k]
            os.environ["ILI_METAL_PREFILL"] = "1"
            os.environ["COLI_MODEL"] = "/legacy/path"
            os.environ["FA_DIRECT"] = "1"
            os.environ["UNRELATED_VAR"] = "should-not-appear"
            env, digest = pm.compute_launcher_env()
            self.assertEqual(env, {
                "COLI_MODEL": "/legacy/path",
                "FA_DIRECT": "1",
                "ILI_METAL_PREFILL": "1",
            })
            self.assertNotIn("UNRELATED_VAR", env)
            canonical = "\n".join(f"{k}={v}" for k, v in sorted(env.items())).encode()
            self.assertEqual(digest, hashlib.sha256(canonical).hexdigest())
        finally:
            for k in list(os.environ):
                if k.startswith(("ILI_", "COLI_", "FA_", "UNRELATED_VAR")):
                    del os.environ[k]
            os.environ.update(env_backup)

    def test_prompt_hash_inline_vs_file_vs_none(self):
        h, source = pm.compute_prompt_hash("inline", "hello world")
        self.assertEqual(h, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9")
        self.assertEqual(source, "inline")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "prompt.txt"
            p.write_text("hello world")
            h2, source2 = pm.compute_prompt_hash("file", str(p))
            self.assertEqual(h2, h)
            self.assertEqual(source2, "file")

        h3, source3 = pm.compute_prompt_hash(None, None)
        self.assertIsNone(h3)
        self.assertIsNone(source3)

    def test_git_info_none_outside_a_repository(self):
        with tempfile.TemporaryDirectory() as td:
            commit, dirty = pm.git_info(td)
            self.assertIsNone(commit)
            self.assertIsNone(dirty)

    def test_git_info_inside_this_repo_matches_rev_parse(self):
        commit, dirty = pm.git_info(str(C_DIR))
        expected = subprocess.run(["git", "-C", str(C_DIR), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(commit, expected)
        self.assertIn(dirty, (True, False))

    def test_read_quiesce_skipped(self):
        result = pm.read_quiesce(None, "", "", True)
        self.assertEqual(result, {"bin": None, "skipped": True, "pass": None,
                                  "exit_code": None, "output": []})

    def test_read_quiesce_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "q.txt"
            out.write_text("[quiesce] ok\n[quiesce] ALL CONDITIONS PASS\n")
            passed = pm.read_quiesce("/path/to/quiesce_check.sh", "0", str(out), False)
            self.assertTrue(passed["pass"])
            self.assertEqual(passed["exit_code"], 0)
            self.assertEqual(passed["output"], ["[quiesce] ok", "[quiesce] ALL CONDITIONS PASS"])

            failed = pm.read_quiesce("/path/to/quiesce_check.sh", "1", str(out), False)
            self.assertFalse(failed["pass"])
            self.assertEqual(failed["exit_code"], 1)

    def test_build_manifest_schema_completeness(self):
        """The literal deliverable: assert the manifest's key set is exactly what
        scripts/provenance.sh's header comment documents -- no silently dropped or
        accidentally added field."""
        with tempfile.TemporaryDirectory() as td:
            binary = Path(td) / "bin"
            binary.write_bytes(b"#!/bin/sh\necho hi\n")
            manifest = pm.build_manifest(
                1, "attempt-x", str(binary), None, "inline", "prompt text",
                "2026-07-15T00:00:00Z", "2026-07-15T01:00:00Z", str(C_DIR), str(C_DIR),
                None, "", "", True)
            self.assertEqual(manifest["binary_sha256"], pm.sha256_file(binary))
        self.assertEqual(set(manifest), EXPECTED_TOP_LEVEL_KEYS - {"manifest_path"})
        self.assertEqual(set(manifest["quiesce"]), EXPECTED_QUIESCE_KEYS)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["attempt_id"], "attempt-x")
        self.assertIsNone(manifest["model_dir"])
        self.assertIsNone(manifest["container_manifest_hash"])
        self.assertIsNone(manifest["pin_profile_hash"])
        self.assertEqual(manifest["prompt_or_dataset_source"], "inline")

    def test_write_atomic_adds_manifest_path_and_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = {"schema_version": 1, "attempt_id": "x"}
            path = pm.write_atomic(manifest, td, "x")
            self.assertTrue(Path(path).is_file())
            self.assertEqual(Path(path).name, "provenance-x.json")
            on_disk = json.loads(Path(path).read_text())
            self.assertEqual(on_disk["manifest_path"], path)
            # No stray temp file left behind.
            leftovers = [p for p in Path(td).iterdir() if p.name.startswith(".provenance.")]
            self.assertEqual(leftovers, [])


def run_cli(args, env=None, timeout=30) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                          timeout=timeout, env=full_env)


class ProvenanceScriptEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.artifact_dir = self.tmp_path / "artifacts"
        self.binary = self.tmp_path / "fake-glm"
        self.binary.write_text("#!/bin/sh\necho fake-engine\n")
        self.binary.chmod(0o755)
        self.quiesce_pass = write_script(self.tmp_path / "quiesce_pass.sh", "#!/bin/bash\necho ok\nexit 0\n")
        self.quiesce_fail = write_script(self.tmp_path / "quiesce_fail.sh", "#!/bin/bash\necho bad\nexit 1\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_help_prints_usage_and_exits_zero(self):
        proc = run_cli(["--help"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Executable provenance manifest", proc.stdout)

    def test_missing_attempt_id_is_rejected(self):
        proc = run_cli(["--binary", str(self.binary), "--skip-quiesce"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--attempt-id is required", proc.stderr)

    def test_missing_binary_and_pid_is_rejected(self):
        proc = run_cli(["--attempt-id", "x", "--skip-quiesce"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("required", proc.stderr)

    def test_flag_missing_its_value_as_the_last_argument_fails_fast_not_hangs(self):
        """Regression test: an earlier draft used a bare `shift 2` per flag, which -- when the
        flag is the LAST argument with no value -- fails silently without moving the
        positional parameters at all, spinning the arg-parsing loop on the same unconsumed
        flag forever (confirmed empirically during development; this is NOT bash raising an
        error, it is a true hang). A tight timeout here means this test fails loudly rather
        than blocking the whole suite if it ever regresses."""
        proc = run_cli(["--attempt-id"], timeout=5)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--attempt-id is required", proc.stderr)

    def test_binary_must_actually_exist(self):
        proc = run_cli(["--attempt-id", "x", "--binary", "/no/such/file", "--skip-quiesce"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("does not exist", proc.stderr)

    def test_binary_and_pid_are_mutually_exclusive(self):
        proc = run_cli(["--attempt-id", "x", "--binary", str(self.binary), "--pid", "1",
                        "--skip-quiesce"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("mutually exclusive", proc.stderr)

    def test_prompt_and_prompt_file_are_mutually_exclusive(self):
        proc = run_cli(["--attempt-id", "x", "--binary", str(self.binary), "--skip-quiesce",
                        "--prompt", "a", "--prompt-file", "/tmp/b"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("mutually exclusive", proc.stderr)

    def test_end_to_end_manifest_is_written_and_schema_complete(self):
        proc = run_cli(["--attempt-id", "e2e-1", "--binary", str(self.binary),
                        "--artifact-dir", str(self.artifact_dir), "--skip-quiesce",
                        "--prompt", "the prompt text"])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        manifest_path = proc.stdout.strip()
        self.assertEqual(manifest_path, str(self.artifact_dir / "provenance-e2e-1.json"))
        manifest = json.loads(Path(manifest_path).read_text())
        self.assertEqual(set(manifest), EXPECTED_TOP_LEVEL_KEYS)
        self.assertEqual(manifest["attempt_id"], "e2e-1")
        self.assertEqual(manifest["binary_sha256"], pm.sha256_file(self.binary))
        self.assertEqual(manifest["quiesce"], {"bin": None, "skipped": True, "pass": None,
                                               "exit_code": None, "output": []})
        self.assertEqual(manifest["prompt_or_dataset_source"], "inline")

    def test_quiesce_bin_override_avoids_the_real_telemetry_sampler(self):
        proc = run_cli(["--attempt-id", "qp", "--binary", str(self.binary),
                        "--artifact-dir", str(self.artifact_dir),
                        "--quiesce-bin", str(self.quiesce_pass)])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        manifest = json.loads(Path(proc.stdout.strip()).read_text())
        self.assertEqual(manifest["quiesce"]["pass"], True)
        self.assertEqual(manifest["quiesce"]["exit_code"], 0)
        self.assertFalse(manifest["quiesce"]["skipped"])
        self.assertEqual(manifest["quiesce"]["bin"], str(self.quiesce_pass))
        self.assertIn("ok", manifest["quiesce"]["output"])

        proc2 = run_cli(["--attempt-id", "qf", "--binary", str(self.binary),
                         "--artifact-dir", str(self.artifact_dir),
                         "--quiesce-bin", str(self.quiesce_fail)])
        self.assertEqual(proc2.returncode, 0, proc2.stdout + proc2.stderr)
        manifest2 = json.loads(Path(proc2.stdout.strip()).read_text())
        # A FAILED quiesce is recorded data, not a provenance.sh failure: rc must stay 0.
        self.assertEqual(manifest2["quiesce"]["pass"], False)
        self.assertEqual(manifest2["quiesce"]["exit_code"], 1)

    def test_missing_quiesce_bin_is_rejected_unless_skipped(self):
        proc = run_cli(["--attempt-id", "x", "--binary", str(self.binary),
                        "--artifact-dir", str(self.artifact_dir),
                        "--quiesce-bin", "/no/such/quiesce.sh"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("not found", proc.stderr)

    def test_pid_resolution_hash_matches_directly_hashing_the_same_executable(self):
        """The CRITICAL requirement from this script's header: for an ALREADY-RUNNING
        process, binary_sha256 must be the hash of the on-disk file actually backing that
        PID. Spawns a real, long-lived `sleep` subprocess and asserts --pid resolves to
        /bin/sleep with the identical hash a direct `--binary /bin/sleep` run produces."""
        sleep_bin = "/bin/sleep"
        if not os.path.isfile(sleep_bin):
            self.skipTest("/bin/sleep not present on this platform")
        proc_sleep = subprocess.Popen([sleep_bin, "30"])
        try:
            proc = run_cli(["--attempt-id", "pid-test", "--pid", str(proc_sleep.pid),
                            "--artifact-dir", str(self.artifact_dir), "--skip-quiesce"])
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            manifest = json.loads(Path(proc.stdout.strip()).read_text())
            # realpath both sides: on usr-merged Linux distros /bin/sleep IS /usr/bin/sleep
            # (a symlink hop), so lsof/ps can correctly resolve to the canonical path while an
            # exact string compare against the unresolved input would false-fail -- that hop is
            # irrelevant to the property under test (the hash of the actual on-disk file).
            self.assertEqual(os.path.realpath(manifest["binary_path"]), os.path.realpath(sleep_bin))
            self.assertEqual(manifest["binary_sha256"], pm.sha256_file(sleep_bin))
        finally:
            proc_sleep.terminate()
            proc_sleep.wait(timeout=5)

    def test_pid_resolution_of_a_dead_pid_fails_closed(self):
        proc_sleep = subprocess.Popen(["/bin/sleep", "0.1"])
        proc_sleep.wait(timeout=5)
        dead_pid = proc_sleep.pid
        proc = run_cli(["--attempt-id", "x", "--pid", str(dead_pid),
                        "--artifact-dir", str(self.artifact_dir), "--skip-quiesce"])
        self.assertNotEqual(proc.returncode, 0)

    def test_sourcing_never_executes_or_changes_shell_options(self):
        """`source scripts/provenance.sh` in a fresh bash must (a) not error, (b) not print a
        manifest path (nothing ran), (c) leave shell options exactly as they were before the
        source -- proving the standalone-execution guard actually guards."""
        script = (
            f'before="$-"; '
            f'source "{SCRIPT}"; '
            f'after="$-"; '
            f'if [[ "$before" != "$after" ]]; then echo "OPTIONS CHANGED: $before -> $after"; exit 1; fi; '
            f'echo "sourced-ok"; '
            f'type provenance_main >/dev/null 2>&1 && echo "function-defined"'
        )
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("sourced-ok", proc.stdout)
        self.assertIn("function-defined", proc.stdout)
        self.assertNotIn("OPTIONS CHANGED", proc.stdout)

    def test_sourced_function_can_be_called_directly(self):
        script = (
            f'source "{SCRIPT}"; '
            f'provenance_main --attempt-id sourced-1 --binary "{self.binary}" '
            f'--artifact-dir "{self.artifact_dir}" --skip-quiesce'
        )
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        manifest_path = proc.stdout.strip()
        self.assertTrue(Path(manifest_path).is_file())


if __name__ == "__main__":
    unittest.main()
