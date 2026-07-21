#!/usr/bin/env bash
# Timing-run exclusivity lock -- MECHANICAL, not procedural. Motivating incident: a timing
# roofline was contaminated by concurrent agent activity even though the calling agents had
# been told "wait for other agents to finish" -- a procedural convention with no enforcement.
# This file makes exclusivity a hard, testable precondition: a timing driver that cannot
# acquire this lock refuses to start, full stop, rather than trusting anyone to have checked.
#
# Sourceable + standalone, same convention as scripts/provenance.sh:
#   source scripts/timing_lock.sh; timing_lock_acquire "$ATTEMPT_ID" || die "..."
#   bash   scripts/timing_lock.sh acquire --attempt-id ID [--pid PID] [--pgid PGID]
# Sourcing this file only defines functions -- it never sets shell options or runs anything
# at source time (identical rationale to provenance.sh: a `set -uo pipefail` at file scope
# would leak into whatever sourced this the instant it was sourced). Shell options are only
# applied inside the standalone-execution guard at the bottom.
#
# Lock primitive: `mkdir` at $ILI_TIMING_LOCK_DIR (default /tmp/iliria-timing.lock) --
# atomic per POSIX (a single syscall; EEXIST if already present), no flock(1) dependency
# (macOS ships no standalone `flock` command; util-linux only). Metadata lives INSIDE the
# directory (meta.json), written via temp-file-then-rename ONLY after mkdir has already
# proven exclusive ownership, so the metadata write itself needs no separate locking.
#
# Metadata schema (meta.json inside the lock dir):
#   attempt_id        caller-supplied identifier, embedded in every artifact filename
#                     (this repo's own campaign-state.json convention)
#   driver_pid        pid of the process the lock is FOR (the timing driver itself, e.g.
#                     roofline_run.sh's own $$ -- not a launched experiment child)
#   pgid              driver_pid's process group at acquire time
#   started_at        UTC ISO8601 acquire timestamp
#   host              `hostname`
#   driver_start_time `ps -o lstart= -p driver_pid`, trimmed -- see "stale-lock reaper" below
#
# Stale-lock reaper: a lock is reclaimable when its recorded driver_pid is provably NOT the
# same process that acquired it. PID LIVENESS ALONE IS NOT ENOUGH: PIDs are recycled by the
# OS, so a dead holder's pid can, in principle, be reassigned to an unrelated later process
# before anyone reclaims the lock -- a naive `kill -0 $pid` would then report "alive" for a
# process that is not the original holder at all. This file additionally compares
# `ps -o lstart=` (the CURRENT process at that pid's start time, second-resolution) against
# the value recorded at acquire time; a mismatch (or a dead pid outright) means reclaimable.
# A lock dir that exists but has no meta.json yet (the sub-millisecond window between mkdir
# succeeding and the metadata rename landing) is treated as "busy" while younger than
# ILI_TIMING_LOCK_GRACE_S (default 5s) and "crashed mid-acquire, reclaimable" once older --
# nothing else could be writing into a directory it does not own, so an empty dir past the
# grace period can only mean the acquirer died in that exact window.
#
# timing_lock_check() is READ-ONLY on purpose: it reports whether a lock is held/stale but
# never reclaims a stale one itself -- reclaiming is an effect of a genuine acquire attempt
# only, never a side effect of merely asking. This is the documented convention non-timing
# work (builds/tests/analysis) is asked to honor via c/scripts/pre_push_guard.sh: check, and
# back off if busy. The KEY guarantee runs the other direction and is not optional: a timing
# driver (timing_lock_acquire) hard-refuses to start if a competing lock is genuinely held.
#
# timing_lock_release_attempt() exists for a cross-process release: c/tools/timing_watchdog.py
# runs as a SEPARATE process from the driver it watches, and on a confirmed contamination
# event it kills the experiment's process group and then releases the driver's lease itself
# (see that file's module docstring) rather than trusting the driver to still be able to run
# its own exit trap. It is authorized to do this because it is told the exact attempt_id it
# is watching, and this function only ever releases a lock whose recorded attempt_id matches
# the one given -- never merely because a caller wants "the current lock" gone.

TIMING_LOCK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

_timing_lock_log() { echo "[timing-lock] $*" >&2; }

timing_lock_path() {
  printf '%s' "${ILI_TIMING_LOCK_DIR:-/tmp/iliria-timing.lock}"
}

# ---- small process-introspection helpers (macOS ps; no setsid/flock dependency) -----------
_timing_lock_pgid_of() {
  ps -o pgid= -p "$1" 2>/dev/null | tr -d '[:space:]'
}

_timing_lock_start_time_of() {
  ps -o lstart= -p "$1" 2>/dev/null | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

_timing_lock_dir_age_s() {
  local dir="$1" mtime now
  mtime="$(stat -f %m "$dir" 2>/dev/null || stat -c %Y "$dir" 2>/dev/null)"
  [[ -z "$mtime" ]] && return 1
  now="$(date +%s)"
  printf '%s' "$(( now - mtime ))"
}

# ---- meta.json read/write (python3: same delegation convention as provenance.sh) -----------
_timing_lock_write_meta() {
  local lockdir="$1" attempt_id="$2" pid="$3" pgid="$4" host="$5" started_at="$6"
  local driver_start_time
  driver_start_time="$(_timing_lock_start_time_of "$pid")"
  python3 - "$lockdir" "$attempt_id" "$pid" "$pgid" "$host" "$started_at" "$driver_start_time" <<'PY'
import json, os, sys
lockdir, attempt_id, pid, pgid, host, started_at, driver_start_time = sys.argv[1:8]
meta = {
    "attempt_id": attempt_id,
    "driver_pid": int(pid),
    "pgid": int(pgid) if pgid else int(pid),
    "started_at": started_at,
    "host": host,
    "driver_start_time": driver_start_time,
}
tmp = os.path.join(lockdir, ".meta.json.tmp")
with open(tmp, "w") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
os.replace(tmp, os.path.join(lockdir, "meta.json"))
PY
}

_timing_lock_read_field() {
  python3 -c '
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    v = data.get(sys.argv[2], "")
    print(v if v is not None else "")
except Exception:
    print("")
' "$1" "$2" 2>/dev/null
}

# Returns 0 if the metadata's recorded driver_pid is PROVABLY still the same live process
# (pid alive AND start-time matches what was recorded at acquire time); 1 otherwise (dead,
# recycled, or unreadable). Shared by both the stale-reaper and timing_lock_check() so the
# two never silently disagree about what counts as "genuinely still held".
_timing_lock_holder_is_alive() {
  local meta="$1" pid recorded_start current_start
  [[ -f "$meta" ]] || return 1
  pid="$(_timing_lock_read_field "$meta" driver_pid)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  recorded_start="$(_timing_lock_read_field "$meta" driver_start_time)"
  if [[ -z "$recorded_start" ]]; then
    _timing_lock_log "warning: no driver_start_time recorded for pid=$pid -- falling back to" \
                      "pid-liveness-only (weaker: cannot rule out pid recycling)"
    return 0
  fi
  current_start="$(_timing_lock_start_time_of "$pid")"
  [[ -n "$current_start" && "$current_start" == "$recorded_start" ]]
}

# Reclaims (rm -rf) a stale lock dir and returns 0; returns 1 if it is not stale (still
# genuinely held, or nothing to reclaim). Never called except as part of a real acquire
# attempt -- see module header.
_timing_lock_reap_if_stale() {
  local lockdir="$1" meta="$lockdir/meta.json" grace_s="${ILI_TIMING_LOCK_GRACE_S:-5}" age

  [[ -d "$lockdir" ]] || return 1

  if [[ ! -f "$meta" ]]; then
    age="$(_timing_lock_dir_age_s "$lockdir")"
    if [[ -n "$age" ]] && (( age >= grace_s )); then
      _timing_lock_log "reaping lock dir with no meta.json, age=${age}s >= grace=${grace_s}s" \
                        "(acquirer crashed between mkdir and metadata write)"
      rm -rf "$lockdir"
      return 0
    fi
    return 1   # too young to call stale -- likely a concurrent acquire mid-flight
  fi

  if _timing_lock_holder_is_alive "$meta"; then
    return 1
  fi

  _timing_lock_log "reaping stale lock: recorded holder is dead or pid was recycled" \
                    "(driver_pid=$(_timing_lock_read_field "$meta" driver_pid))"
  rm -rf "$lockdir"
  return 0
}

# ---- public API -----------------------------------------------------------------------
_TIMING_LOCK_HELD=0
_TIMING_LOCK_OWN_PID=""

# Shared success tail for both acquire paths: write metadata, and if that write fails, tear
# the just-made lockdir back down rather than leaving a meta-less directory that the empty-dir
# reaper (_timing_lock_reap_if_stale) would reclaim out from under a live measurement once the
# grace period elapses. Sets the in-shell ownership state only on a fully-successful acquire.
_timing_lock_finalize_acquire() {
  local lockdir="$1" attempt_id="$2" pid="$3" pgid="$4" host="$5" started_at="$6"
  if ! _timing_lock_write_meta "$lockdir" "$attempt_id" "$pid" "$pgid" "$host" "$started_at"; then
    _timing_lock_log "ERROR: metadata write failed after mkdir -- removing $lockdir to avoid a" \
                      "meta-less lock that would be reaped mid-measurement"
    rm -rf "$lockdir"
    return 1
  fi
  _TIMING_LOCK_HELD=1
  _TIMING_LOCK_OWN_PID="$pid"
  return 0
}

# timing_lock_acquire ATTEMPT_ID [PID] [PGID]
# PID/PGID default to the CALLING shell's own $$/pgid -- correct for the primary,
# sourced-function usage (a long-lived timing driver sources this file and calls this in
# its own process). Returns 0 on success, 1 if genuinely busy, 2 on a hard error (e.g. the
# lock directory's parent is not writable) distinct from "busy" so callers can react
# differently.
timing_lock_acquire() {
  local attempt_id="${1:?timing_lock_acquire: attempt_id required}"
  local pid="${2:-$$}"
  local pgid="${3:-}"
  [[ -n "$pgid" ]] || pgid="$(_timing_lock_pgid_of "$pid")"
  [[ -n "$pgid" ]] || pgid="$pid"
  local lockdir host started_at
  lockdir="$(timing_lock_path)"
  host="$(hostname 2>/dev/null || echo unknown-host)"
  started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  if mkdir "$lockdir" 2>/dev/null; then
    if _timing_lock_finalize_acquire "$lockdir" "$attempt_id" "$pid" "$pgid" "$host" "$started_at"; then
      _timing_lock_log "acquired: attempt_id=$attempt_id driver_pid=$pid pgid=$pgid at $lockdir"
      return 0
    fi
    return 2
  fi

  if [[ ! -d "$lockdir" ]]; then
    _timing_lock_log "ERROR: mkdir $lockdir failed and it does not exist -- check permissions/parent dir"
    return 2
  fi

  # Stale-lock reclaim, SERIALIZED by a reap-mutex ($lockdir.reap). Without it, two acquirers
  # can each independently judge the SAME lock stale, and the second's `rm -rf` lands AFTER the
  # first has already reaped, re-created, and taken the lock -- silently deleting a live lock and
  # granting it twice (the exact two-concurrent-timing-drivers incident this file exists to
  # prevent; reproduced 29/30 in review). Only the reap-mutex holder may reap, and it RE-verifies
  # staleness under the mutex, so once the first reclaimer installs fresh meta the second sees a
  # live holder and backs off to BUSY. The atomic `mkdir "$lockdir"` stays the final arbiter, so
  # a fast-path acquirer racing in on the freed slot still yields exactly one winner.
  local reapdir="${lockdir}.reap"
  local reap_grace="${ILI_TIMING_REAP_GRACE_S:-30}" reap_age
  reap_age="$(_timing_lock_dir_age_s "$reapdir" 2>/dev/null)"
  if [[ -d "$reapdir" && -n "$reap_age" ]] && (( reap_age >= reap_grace )); then
    _timing_lock_log "clearing abandoned reap-mutex (age=${reap_age}s >= ${reap_grace}s): an" \
                      "acquirer died mid-reclaim (reaping is sub-second, so this is safe)"
    rm -rf "$reapdir"
  fi
  if mkdir "$reapdir" 2>/dev/null; then
    if _timing_lock_reap_if_stale "$lockdir" && mkdir "$lockdir" 2>/dev/null; then
      if _timing_lock_finalize_acquire "$lockdir" "$attempt_id" "$pid" "$pgid" "$host" "$started_at"; then
        rmdir "$reapdir" 2>/dev/null || true
        _timing_lock_log "acquired after reclaiming a stale lock: attempt_id=$attempt_id driver_pid=$pid pgid=$pgid"
        return 0
      fi
      rmdir "$reapdir" 2>/dev/null || true
      return 2
    fi
    rmdir "$reapdir" 2>/dev/null || true
  fi

  _timing_lock_log "BUSY: $(timing_lock_check 2>&1 | tail -1)"
  return 1
}

# timing_lock_release: releases the lock THIS SHELL acquired (tracked via
# _TIMING_LOCK_HELD/_TIMING_LOCK_OWN_PID). Idempotent no-op if this shell never held it
# (safe to call unconditionally from an EXIT trap regardless of how far a script got).
# Verifies the lock is still ours before removing it -- if it was already reclaimed as
# stale and re-acquired by someone else, this refuses to delete THEIR lock.
timing_lock_release() {
  if [[ "$_TIMING_LOCK_HELD" != 1 ]]; then
    return 0
  fi
  local lockdir meta recorded_pid
  lockdir="$(timing_lock_path)"
  meta="$lockdir/meta.json"
  if [[ -f "$meta" ]]; then
    recorded_pid="$(_timing_lock_read_field "$meta" driver_pid)"
    if [[ -n "$recorded_pid" && "$recorded_pid" != "$_TIMING_LOCK_OWN_PID" ]]; then
      _timing_lock_log "refusing to release: lock is now held by pid=$recorded_pid, not ours" \
                        "($_TIMING_LOCK_OWN_PID) -- it must have been reclaimed as stale while we" \
                        "still thought we held it"
      _TIMING_LOCK_HELD=0
      return 1
    fi
  fi
  rm -rf "$lockdir"
  _timing_lock_log "released: driver_pid=$_TIMING_LOCK_OWN_PID"
  _TIMING_LOCK_HELD=0
  return 0
}

# timing_lock_release_attempt ATTEMPT_ID: cross-process release authorized by attempt_id
# match (see module header -- c/tools/timing_watchdog.py's use case). Returns 0 if released
# or already absent; 1 if the current lock belongs to a DIFFERENT attempt_id (refused).
timing_lock_release_attempt() {
  local attempt_id="${1:?timing_lock_release_attempt: attempt_id required}"
  local lockdir meta recorded
  lockdir="$(timing_lock_path)"
  meta="$lockdir/meta.json"
  if [[ ! -d "$lockdir" ]]; then
    return 0
  fi
  if [[ -f "$meta" ]]; then
    recorded="$(_timing_lock_read_field "$meta" attempt_id)"
    if [[ -n "$recorded" && "$recorded" != "$attempt_id" ]]; then
      _timing_lock_log "refusing release-attempt: lock is held by attempt_id=$recorded, not $attempt_id"
      return 1
    fi
  fi
  rm -rf "$lockdir"
  _timing_lock_log "released (by attempt_id match): attempt_id=$attempt_id"
  return 0
}

# timing_lock_check: read-only status. Prints a one-line summary and returns 0 (free, or
# stale-but-not-reclaimed -- informational only) / 1 (held by a verified-alive holder).
timing_lock_check() {
  local lockdir meta
  lockdir="$(timing_lock_path)"
  if [[ ! -d "$lockdir" ]]; then
    echo "free (no lock at $lockdir)"
    return 0
  fi
  meta="$lockdir/meta.json"
  if [[ ! -f "$meta" ]]; then
    echo "present but no metadata yet at $lockdir (likely mid-acquire) -- treating as busy"
    return 1
  fi
  local pid attempt_id started_at host
  pid="$(_timing_lock_read_field "$meta" driver_pid)"
  attempt_id="$(_timing_lock_read_field "$meta" attempt_id)"
  started_at="$(_timing_lock_read_field "$meta" started_at)"
  host="$(_timing_lock_read_field "$meta" host)"
  if _timing_lock_holder_is_alive "$meta"; then
    echo "HELD: attempt_id=$attempt_id driver_pid=$pid started_at=$started_at host=$host"
    return 1
  fi
  echo "stale (recorded driver_pid=$pid is dead or recycled; attempt_id=$attempt_id) --" \
       "reclaimable, but check() never reclaims it itself"
  return 0
}

# ---- standalone execution guard: sourcing this file never runs anything or sets options --
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -uo pipefail

  _timing_lock_usage() {
    sed -n '2,55p' "${BASH_SOURCE[0]}"
  }

  cmd="${1:-}"; shift || true
  case "$cmd" in
    acquire)
      attempt_id=""; pid=""; pgid=""
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --attempt-id) attempt_id="${2:-}"; shift 2 ;;
          --pid) pid="${2:-}"; shift 2 ;;
          --pgid) pgid="${2:-}"; shift 2 ;;
          *) echo "[timing-lock] unknown argument: $1" >&2; exit 2 ;;
        esac
      done
      [[ -n "$attempt_id" ]] || { echo "[timing-lock] --attempt-id is required" >&2; exit 2; }
      timing_lock_acquire "$attempt_id" "${pid:-$$}" "$pgid"
      exit $?
      ;;
    release)
      # Standalone `release` (no ownership tracked in THIS fresh process) only makes sense
      # via attempt-id authorization -- use release-attempt instead. Kept as a clear error
      # rather than a silent no-op so a caller does not mistake this for having worked.
      echo "[timing-lock] standalone 'release' has no in-process ownership to release;" >&2
      echo "  use 'release-attempt --attempt-id ID', or source this file and call" >&2
      echo "  timing_lock_release from the SAME shell that acquired the lock." >&2
      exit 2
      ;;
    release-attempt)
      attempt_id=""
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --attempt-id) attempt_id="${2:-}"; shift 2 ;;
          *) echo "[timing-lock] unknown argument: $1" >&2; exit 2 ;;
        esac
      done
      [[ -n "$attempt_id" ]] || { echo "[timing-lock] --attempt-id is required" >&2; exit 2; }
      timing_lock_release_attempt "$attempt_id"
      exit $?
      ;;
    check)
      timing_lock_check
      exit $?
      ;;
    -h|--help|"")
      _timing_lock_usage
      exit 0
      ;;
    *)
      echo "[timing-lock] unknown subcommand: $cmd (expected acquire|release-attempt|check)" >&2
      exit 2
      ;;
  esac
fi
