#!/usr/bin/env bash
# Executable provenance manifest -- answers "what actually ran" for one bench/gate attempt,
# independent of what git HEAD claims. Motivating incident: an n=100 quality bench can run
# for 8+ hours against a `./glm` binary that was compiled before the last N merges to glm.c /
# backend_metal.mm; the bench's own log may report today's git commit while the process
# actually executing tokens is hours stale. A result is not comparable to anything else --
# and an A/B pair is not valid at all -- without knowing exactly which binary, which
# generated source, which model container, and which launcher environment produced it.
#
# Sourceable + standalone:
#   source scripts/provenance.sh; provenance_main --attempt-id ID --binary PATH ...
#   bash   scripts/provenance.sh        --attempt-id ID --binary PATH ...
# Sourcing this file only defines functions (provenance_main and helpers) -- it never sets
# shell options or runs anything at source time, so it is safe to `source` from a caller that
# has its own `set -uo pipefail` (or any other options) already in effect. Shell options are
# only applied inside the standalone-execution guard at the bottom of this file.
#
# Writes <artifact-dir>/provenance-<attempt_id>.json (atomic: temp file + rename) and prints
# its absolute path on stdout (the only thing this script prints to stdout on success --
# everything else goes to stderr -- so callers can do MANIFEST=$(bash scripts/provenance.sh ...)).
#
# Required:
#   --attempt-id ID     attempt identifier; embedded in the manifest and the output filename
#   --binary PATH       the ACTUAL running engine executable for this attempt (./glm or
#                        ./ili) -- the real file, never an assumed default path. Must exist.
#                        Mutually exclusive with --pid.
#   --pid PID            resolve --binary FROM an already-running process instead of being
#                        told the path directly (the "attach provenance to an 8h-old bench
#                        you did not launch" case). macOS approach (no /proc/<pid>/exe here):
#                        `lsof -p PID`, first FD of type "txt" (the process's own program-text
#                        mapping) -- empirically the process's own main executable is listed
#                        before shared mappings every process has (dyld, locale files); see
#                        resolve_binary_from_pid() below. Falls back to `ps -o comm=` ONLY
#                        when it yields an absolute, existing path (comm is not reliably a
#                        full path for every process -- observed empirically: a backgrounded
#                        `sleep` reports bare "sleep", not "/bin/sleep"). Mutually exclusive
#                        with --binary.
#
# Optional:
#   --artifact-dir DIR   directory to write provenance-<attempt_id>.json into (default: .)
#   --model-dir DIR      model/container directory: config.json + sorted shard listing
#                        (name/size/mtime, never shard bytes) -> container_manifest_hash;
#                        DIR/.fa_usage -> pin_profile_hash. Either/both omitted if DIR absent.
#   --prompt STRING      literal prompt/dataset text -> prompt_or_dataset_hash (sha256 of the
#                        UTF-8 bytes). Mutually exclusive with --prompt-file.
#   --prompt-file PATH   file whose bytes are hashed instead of a literal string.
#   --start-ts TS        run start timestamp the CALLER tracked (bracketing convention: the
#                        caller records this before the run begins and passes it here, at
#                        whatever point it calls this script). Default: now.
#   --end-ts TS          run end timestamp. Default: now. Passing the SAME manifest call both
#                        --start-ts (captured earlier) and letting --end-ts default lets one
#                        invocation at the end of a run report both brackets; calling this
#                        script again at completion (same attempt-id/artifact-dir) refreshes
#                        end_ts and re-verifies the binary/source hashes did not change
#                        mid-run -- the file is overwritten, not appended.
#   --quiesce-bin PATH   quiesce_check.sh to invoke for the quiesce snapshot (default: the
#                        quiesce_check.sh next to this script). Overridable so tests / callers
#                        that already run under a fixture quiesce script reuse it here instead
#                        of paying a second real (tens-of-seconds) telemetry sample.
#   --skip-quiesce       do not invoke a quiesce script at all; quiesce.skipped=true, no other
#                        quiesce.* fields populated.
#   --repo-root DIR      repo root for git_commit/git_dirty (default: `git rev-parse
#                        --show-toplevel` from this script's own location)
#   --c-dir DIR          c/ directory for generated_source_sha256 lookups (default: parent of
#                        this script's own directory)
#   -h, --help
#
# Manifest schema (see c/tests/test_provenance.py for the enforced completeness contract):
#   schema_version, attempt_id, generated_at, start_ts, end_ts, git_commit, git_dirty,
#   binary_path, binary_sha256, generated_source_sha256 {filename: sha256, only if present},
#   model_dir, container_manifest_hash, pin_profile_hash, launcher_env {NAME: value},
#   launcher_env_digest, prompt_or_dataset_hash, prompt_or_dataset_source, quiesce
#   {bin, skipped, pass, exit_code, output[]}, manifest_path.
#
# launcher_env scope: every ILI_*/COLI_*/FA_* environment variable, sorted by name. This
# mirrors c/ili's own documented env convention exactly (_env(): "silent legacy fallback:
# ILI_* > COLI_* > FA_*") -- the three prefixes are first-class equivalent spellings of the
# same knobs there, so all three must be in scope or a launcher could vary behavior through a
# legacy-prefixed var without the digest ever noticing.
# No top-level `set` statement here on purpose: sourcing a file runs its top-level statements
# immediately in the CALLING shell (unlike calling a function), so a `set -uo pipefail` at file
# scope would leak into whatever sourced this script the instant it was sourced -- proven by
# this file's own test_sourcing_never_executes_or_changes_shell_options. `set -uo pipefail` is
# therefore applied ONLY inside the standalone-execution guard at the bottom of this file, and
# every function above validates its own inputs explicitly rather than depending on `set -u` to
# catch a caller's mistake.

PROVENANCE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROVENANCE_SCHEMA_VERSION=1

_prov_log()  { echo "[provenance] $*" >&2; }
_prov_die()  { echo "[provenance] FATAL: $*" >&2; return 1; }

# ---- macOS: resolve a running PID's on-disk executable path (no /proc/<pid>/exe here) -----
resolve_binary_from_pid() {
  local pid="$1" path="" comm
  path="$(lsof -p "$pid" 2>/dev/null | awk '$4=="txt"{print $NF; exit}')"
  if [[ -z "$path" || ! -f "$path" ]]; then
    comm="$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [[ "$comm" == /* && -f "$comm" ]]; then path="$comm"; fi
  fi
  printf '%s' "$path"
}

provenance_usage() {
  sed -n '2,79p' "${BASH_SOURCE[0]}"
}

provenance_main() {
  local attempt_id="" binary="" pid="" artifact_dir="." model_dir=""
  local prompt_mode="" prompt_value=""
  local start_ts="" end_ts=""
  local quiesce_bin="$PROVENANCE_SCRIPT_DIR/quiesce_check.sh" skip_quiesce=0
  local repo_root="" c_dir="" opt

  while [[ $# -gt 0 ]]; do
    # Shift the flag itself FIRST, then consume its value only if one is actually still
    # present ("${1:-}" + a conditional shift) -- never a bare `shift 2`. A flag given as the
    # very last argument with no value must degrade to an empty string, caught by this
    # function's own validation below, not (a) crash on bash's raw "$2: unbound variable"
    # under the standalone guard's `set -u`, or (b), worse, silently HANG: `shift 2` when only
    # one positional argument remains fails without moving $1 at all (confirmed empirically --
    # see this file's test suite), which would spin the case/while parser on the same
    # unconsumed flag forever.
    opt="$1"; shift
    case "$opt" in
      --attempt-id) attempt_id="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      --binary) binary="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      --pid) pid="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      --artifact-dir) artifact_dir="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      --model-dir) model_dir="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      --prompt)
        [[ -n "$prompt_mode" ]] && { _prov_die "--prompt and --prompt-file are mutually exclusive"; return 2; }
        prompt_mode="inline"; prompt_value="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      --prompt-file)
        [[ -n "$prompt_mode" ]] && { _prov_die "--prompt and --prompt-file are mutually exclusive"; return 2; }
        prompt_mode="file"; prompt_value="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      --start-ts) start_ts="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      --end-ts) end_ts="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      --quiesce-bin) quiesce_bin="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      --skip-quiesce) skip_quiesce=1 ;;
      --repo-root) repo_root="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      --c-dir) c_dir="${1:-}"; [[ $# -gt 0 ]] && shift ;;
      -h|--help) provenance_usage; return 0 ;;
      *) _prov_die "unknown argument: $opt"; return 2 ;;
    esac
  done

  [[ -n "$attempt_id" ]] || { _prov_die "--attempt-id is required"; return 2; }
  if [[ -n "$binary" && -n "$pid" ]]; then
    _prov_die "--binary and --pid are mutually exclusive"; return 2
  fi
  if [[ -z "$binary" && -z "$pid" ]]; then
    _prov_die "--binary PATH or --pid PID is required (the real running executable, never assumed)"; return 2
  fi
  if [[ -n "$pid" ]]; then
    binary="$(resolve_binary_from_pid "$pid")"
    if [[ -z "$binary" ]]; then
      _prov_die "could not resolve an on-disk executable for pid $pid via lsof/ps -- pass --binary explicitly instead"
      return 1
    fi
    _prov_log "resolved --pid $pid -> $binary"
  fi
  if [[ ! -f "$binary" ]]; then
    _prov_die "--binary $binary does not exist or is not a regular file"; return 1
  fi

  if [[ -z "$repo_root" ]]; then
    repo_root="$(cd -- "$PROVENANCE_SCRIPT_DIR/../.." && pwd)"
  fi
  if [[ -z "$c_dir" ]]; then
    c_dir="$(cd -- "$PROVENANCE_SCRIPT_DIR/.." && pwd)"
  fi

  mkdir -p "$artifact_dir" || { _prov_die "could not create --artifact-dir $artifact_dir"; return 1; }

  [[ -n "$start_ts" ]] || start_ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  [[ -n "$end_ts" ]] || end_ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  local quiesce_out quiesce_rc="" quiesce_bin_used=""
  quiesce_out="$(mktemp "${TMPDIR:-/tmp}/provenance-quiesce.XXXXXX")"
  if [[ "$skip_quiesce" == 1 ]]; then
    : > "$quiesce_out"
  else
    if [[ ! -f "$quiesce_bin" ]]; then
      rm -f "$quiesce_out"
      _prov_die "--quiesce-bin $quiesce_bin not found (pass --skip-quiesce to omit the quiesce snapshot entirely)"
      return 1
    fi
    quiesce_bin_used="$quiesce_bin"
    bash "$quiesce_bin" > "$quiesce_out" 2>&1
    quiesce_rc=$?
    _prov_log "quiesce snapshot ($quiesce_bin): $([[ "$quiesce_rc" == 0 ]] && echo PASS || echo FAIL) (rc=$quiesce_rc)"
  fi

  local manifest_path
  manifest_path="$(python3 "$PROVENANCE_SCRIPT_DIR/../tools/provenance_manifest.py" \
      "$PROVENANCE_SCHEMA_VERSION" "$attempt_id" "$binary" "$artifact_dir" "${model_dir:-}" \
      "${prompt_mode:-}" "${prompt_value:-}" "$start_ts" "$end_ts" "$repo_root" "$c_dir" \
      "${quiesce_bin_used:-}" "${quiesce_rc:-}" "$quiesce_out" "$skip_quiesce")"
  local py_rc=$?
  rm -f "$quiesce_out"
  if [[ "$py_rc" != 0 || -z "$manifest_path" ]]; then
    _prov_die "manifest assembly failed (see above)"
    return 1
  fi

  echo "$manifest_path"
  return 0
}

# ---- standalone execution guard: sourcing this file never runs main or sets shell options --
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -uo pipefail
  provenance_main "$@"
  exit $?
fi
