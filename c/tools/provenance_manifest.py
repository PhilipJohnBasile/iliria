#!/usr/bin/env python3
"""Provenance manifest assembly -- the computation backend for scripts/provenance.sh.

scripts/provenance.sh is the sourceable/standalone entry point (arg parsing, invoking
quiesce_check.sh, existence validation); this module does the actual hashing/git/JSON work
and is deliberately kept independently importable so c/tests/test_provenance.py can assert
each piece (container-manifest hashing, launcher-env digesting, generated-source detection,
...) in isolation, not only through a subprocess round-trip of the shell wrapper.

Manifest schema (see build_manifest() and c/tests/test_provenance.py's completeness test):
    schema_version, attempt_id, generated_at, start_ts, end_ts, git_commit, git_dirty,
    binary_path, binary_sha256, generated_source_sha256 {filename: sha256, only if present on
    disk}, model_dir, container_manifest_hash, pin_profile_hash, launcher_env {name: value},
    launcher_env_digest, effective_flags_predicted (tools/effective_flags.py's env-only,
    pre-launch prediction of METAL/METAL_PREFILL's resolved values -- see that module's
    predicted_effective_flags() docstring for why this is a prediction, not measurement: the
    engine's own ground-truth "EFFECTIVE-FLAGS:" line does not exist yet at manifest time),
    prompt_or_dataset_hash, prompt_or_dataset_source, quiesce
    {bin, skipped, pass, exit_code, output[]}, manifest_path.

container_manifest_hash: sha256 over "config.json=<sha256 or 'absent'>" plus the sorted
top-level (name, size, content-digest) listing of every regular file directly inside
model_dir. The content-digest is a REAL content check, not (name, size, mtime): small files
(config.json, .fa_usage[.profile], tokenizer files) are hashed in full; anything larger
(weight shards, which can be multi-GB) is hashed over its first+last 4 MiB only -- bounded
I/O regardless of file size, so this stays cheap even for a several-hundred-GB container
directory (reading every shard's FULL bytes, every time, really would cost minutes, not
milliseconds -- see _file_content_digest()'s own docstring for the exact bound and its one
known blind spot: a change confined entirely to the untouched middle of a file larger than
8 MiB is not caught). This is NOT a cryptographic guarantee over arbitrary tampering the way
whole-file hashing would be, but it does catch the failure mode a pure size+mtime proxy
cannot: a same-size, different-bytes rewrite (e.g. swapping --int2-quantizer variants, see
build_mixed_container.py / convert_fp8_to_int4.py) landing within the same wall-clock second,
which mtime-to-the-second alone is blind to. This deliberately includes every top-level file,
not only weight-shard-looking names, so a changed config.json, .fa_usage, or any other
container-directory file is caught without guessing at a shard-naming convention this module
has no way to verify independently. A file this process cannot read (permission-denied,
mid-rotation, ...) falls back to a distinct "unreadable" marker rather than raising, so it
cannot crash provenance recording.

launcher_env scope: every ILI_*/COLI_*/FA_* environment variable, sorted by name -- mirrors
c/ili's own documented env convention exactly (`_env()`: "silent legacy fallback: ILI_* >
COLI_* > FA_*"), so a launcher cannot vary behavior through a legacy-prefixed alias without
the digest (and the individual launcher_env entries provenance_compare.py's --vary can target)
noticing.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import effective_flags  # noqa: E402  (centralized ILI_/COLI_/FA_ resolution, see its docstring)

GENERATED_SOURCE_CANDIDATES = ("glm_m5max.c", "backend_metal_m5max.mm")
ENV_PREFIXES = ("ILI_", "COLI_", "FA_")
_HASH_CHUNK = 1 << 20
_SAMPLE_BYTES = 4 << 20  # 4 MiB: see _file_content_digest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_content_digest(path, size) -> str:
    """Content-aware digest for compute_container_manifest_hash(): sha256 of the WHOLE file
    when it's small enough that doing so is cheap (config.json, .fa_usage[.profile],
    tokenizer files -- all KB-scale), else sha256 of just its first+last _SAMPLE_BYTES --
    bounded I/O regardless of file size, so this stays a milliseconds-not-minutes operation
    even for a several-hundred-GB container directory (see the module docstring's own
    "reading every shard's FULL bytes...would cost minutes" rationale).

    This is a REAL content check, not (name, size, mtime): a same-size rewrite (e.g. swapping
    --int2-quantizer variants in build_mixed_container.py / convert_fp8_to_int4.py, which
    changes only the per-row F32 `.qs` scale bytes) changes this digest, unlike the old
    size+mtime-to-the-second proxy, which could not tell the two apart at all if the rewrite
    happened within the same wall-clock second. quant_container.st_save()'s own convention
    (every F32 tensor, including every `.qs`, serialized FIRST) puts exactly the bytes a
    routine requantization changes at the very front of any shard build_mixed_container.py /
    encode_mode15_container.py rewrite -- well inside the leading _SAMPLE_BYTES sample.

    NOT a full guarantee over arbitrary tampering: a change confined entirely to the
    untouched MIDDLE of a file larger than 2*_SAMPLE_BYTES would not be caught; hash the
    whole file instead if that guarantee is ever actually required here.

    A file this process cannot read (permission-denied, mid-rotation, ...) must not crash
    provenance recording -- falls back to a distinct "unreadable" marker rather than raising."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            if size <= 2 * _SAMPLE_BYTES:
                for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
                    h.update(chunk)
            else:
                h.update(fh.read(_SAMPLE_BYTES))
                fh.seek(-_SAMPLE_BYTES, os.SEEK_END)
                h.update(fh.read(_SAMPLE_BYTES))
        return h.hexdigest()
    except OSError as exc:
        return f"unreadable:{type(exc).__name__}"


def compute_generated_source_sha256(c_dir) -> dict:
    """glm_m5max.c / backend_metal_m5max.mm are Makefile-generated (gen_m5max_engine.py /
    gen_m5max_backend.py, from glm.c / backend_metal.mm) and `make clean`-removed -- present
    only after a mac-fast-style build, absent on a fresh checkout. Only keys for files that
    actually exist right now are included, so their absence is a missing key, not a null."""
    result = {}
    for fname in GENERATED_SOURCE_CANDIDATES:
        p = os.path.join(c_dir, fname)
        if os.path.isfile(p):
            result[fname] = sha256_file(p)
    return result


def compute_container_manifest_hash(model_dir):
    if not model_dir or not os.path.isdir(model_dir):
        return None
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "rb") as fh:
            config_sha = sha256_bytes(fh.read())
    else:
        config_sha = "absent"
    lines = [f"config.json={config_sha}"]
    for name in sorted(os.listdir(model_dir)):
        full = os.path.join(model_dir, name)
        if os.path.isfile(full):
            size = os.path.getsize(full)
            lines.append(f"{name}\t{size}\t{_file_content_digest(full, size)}")
    canonical = "\n".join(lines).encode("utf-8")
    return sha256_bytes(canonical)


def compute_pin_profile_hash(model_dir):
    if not model_dir:
        return None
    p = os.path.join(model_dir, ".fa_usage")
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as fh:
        return sha256_bytes(fh.read())


def compute_launcher_env():
    env = {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIXES)}
    ordered = dict(sorted(env.items()))
    canonical = "\n".join(f"{k}={v}" for k, v in ordered.items()).encode("utf-8")
    return ordered, sha256_bytes(canonical)


def compute_prompt_hash(mode, value):
    if mode == "inline":
        return sha256_bytes((value or "").encode("utf-8")), "inline"
    if mode == "file":
        with open(value, "rb") as fh:
            return sha256_bytes(fh.read()), "file"
    return None, None


def git_info(repo_root):
    """None (not False/'') for either field when git itself could not answer -- e.g.
    repo_root is not inside a git checkout at all -- rather than asserting a guessed state."""
    commit = None
    dirty = None
    try:
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            commit = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        out = subprocess.run(["git", "-C", repo_root, "status", "--porcelain"],
                              capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            dirty = bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return commit, dirty


def read_quiesce(quiesce_bin, rc_str, out_file, skipped: bool) -> dict:
    if skipped:
        return {"bin": None, "skipped": True, "pass": None, "exit_code": None, "output": []}
    lines = []
    if out_file and os.path.isfile(out_file):
        lines = Path(out_file).read_text(errors="replace").splitlines()
    rc = int(rc_str) if rc_str not in (None, "") else None
    return {
        "bin": quiesce_bin or None,
        "skipped": False,
        "pass": (rc == 0) if rc is not None else None,
        "exit_code": rc,
        "output": lines,
    }


def build_manifest(schema_version, attempt_id, binary_path, model_dir, prompt_mode,
                    prompt_value, start_ts, end_ts, repo_root, c_dir, quiesce_bin, quiesce_rc,
                    quiesce_out_file, skip_quiesce: bool) -> dict:
    commit, dirty = git_info(repo_root)
    prompt_hash, prompt_source = compute_prompt_hash(prompt_mode, prompt_value)
    env, env_digest = compute_launcher_env()
    return {
        "schema_version": schema_version,
        "attempt_id": attempt_id,
        "generated_at": _now_iso(),
        "start_ts": start_ts or None,
        "end_ts": end_ts or None,
        "git_commit": commit,
        "git_dirty": dirty,
        "binary_path": os.path.abspath(binary_path),
        "binary_sha256": sha256_file(binary_path),
        "generated_source_sha256": compute_generated_source_sha256(c_dir),
        "model_dir": os.path.abspath(model_dir) if model_dir else None,
        "container_manifest_hash": compute_container_manifest_hash(model_dir),
        "pin_profile_hash": compute_pin_profile_hash(model_dir),
        "launcher_env": env,
        "launcher_env_digest": env_digest,
        "effective_flags_predicted": effective_flags.predicted_effective_flags(),
        "prompt_or_dataset_hash": prompt_hash,
        "prompt_or_dataset_source": prompt_source,
        "quiesce": read_quiesce(quiesce_bin, quiesce_rc, quiesce_out_file, skip_quiesce),
    }


def write_atomic(manifest: dict, artifact_dir, attempt_id) -> str:
    """Temp file + os.replace (same filesystem): a reader never observes a torn/partial
    manifest, matching scripts/evening_orchestrator.sh's write_status() convention."""
    os.makedirs(artifact_dir, exist_ok=True)
    final_path = os.path.abspath(os.path.join(artifact_dir, f"provenance-{attempt_id}.json"))
    manifest["manifest_path"] = final_path
    fd, tmp = tempfile.mkstemp(dir=artifact_dir, prefix=".provenance.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, final_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return final_path


def main() -> None:
    # Positional contract with scripts/provenance.sh (see that file's provenance_main()):
    # schema_version attempt_id binary_path artifact_dir model_dir prompt_mode prompt_value
    # start_ts end_ts repo_root c_dir quiesce_bin quiesce_rc quiesce_out_file skip_quiesce
    (schema_version, attempt_id, binary_path, artifact_dir, model_dir, prompt_mode,
     prompt_value, start_ts, end_ts, repo_root, c_dir, quiesce_bin, quiesce_rc,
     quiesce_out_file, skip_quiesce) = sys.argv[1:16]
    try:
        manifest = build_manifest(
            int(schema_version), attempt_id, binary_path, model_dir or None,
            prompt_mode or None, prompt_value, start_ts, end_ts, repo_root, c_dir,
            quiesce_bin or None, quiesce_rc, quiesce_out_file, skip_quiesce == "1")
        path = write_atomic(manifest, artifact_dir, attempt_id)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the caller's stderr, then exit 1
        print(f"provenance_manifest.py: {exc}", file=sys.stderr)
        sys.exit(1)
    print(path)


if __name__ == "__main__":
    main()
