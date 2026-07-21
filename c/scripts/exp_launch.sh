#!/usr/bin/env bash
# Process-group ownership for experiment children -- launch in a NEW process group/session
# and a clean, verified cancel. Motivating incident: a driver's plain `kill -9 <child_pid>`
# reliably left a respawn-loop descendant running, because killing a single LEAF pid does
# not touch a sibling/child that process had already forked (or forks a moment later, from
# a parent that was never itself signaled). Killing the whole process GROUP reaches every
# process that ever joined it -- including one forked after the signal was already sent --
# as long as it inherited (and never changed) the group id, which every ordinary fork does.
#
# Sourceable + standalone, same convention as scripts/provenance.sh and scripts/timing_lock.sh:
#   source scripts/exp_launch.sh
#   bash   scripts/exp_launch.sh start LOG_PATH CMD [ARGS...]
#   bash   scripts/exp_launch.sh cancel PGID [STATE_FILE] [TIMEOUT_S]
# Sourcing this file only defines functions -- see timing_lock.sh's header for why (a
# `set -uo pipefail` at file scope would leak into the calling shell the instant it is
# sourced). Shell options are only applied inside the standalone-execution guard below.
#
# NEW PROCESS GROUP, portably: macOS ships no `setsid(1)` (that is Linux/util-linux only),
# and there is no BSD equivalent binary either. The portable primitive here is bash's own
# job control: with monitor mode on (`set -m`), every pipeline started with `&` becomes the
# leader of its OWN new process group (pgid == its own pid), distinct from the calling
# shell's group -- a documented bash behavior, not a setsid workaround, and it works
# identically whether `set -m` is toggled at top level or inside a function (verified
# empirically during development; see c/tests/test_exp_launch.py). Monitor mode is restored
# to its prior state immediately after backgrounding so it never leaks into the rest of the
# calling script (job-control "Done"/"Terminated" chatter, SIGCHLD semantics). This is the
# same "setsid / bash -m / start_new_session=True" pattern already used, in its Python form,
# by c/tests/test_evening_orchestrator.py's run_orchestrator() (start_new_session=True +
# os.killpg on timeout) -- this file is the bash-side, production equivalent of that same
# idea, generalized into a reusable helper instead of a test-only convenience.
#
# Clean cancel protocol (exp_launch_cancel): mark STATE_FILE "cancelling" (atomic) -> SIGTERM
# the whole group (`kill -TERM -PGID`) -> wait up to TIMEOUT_S -> SIGKILL the whole group if
# anything survived -> verify no descendants remain (`pgrep -g PGID`) -> mark STATE_FILE
# "cancelled" (atomic), and ONLY after that verification, so a caller reading "cancelled"
# from the state file is a genuine guarantee, not an intention. c/tools/timing_watchdog.py
# implements this SAME protocol natively in Python (os.killpg) for its own contamination
# response rather than shelling out to this file -- a deliberate choice, not drift: the two
# are small, independent implementations of one simple, stable, well-specified protocol, each
# idiomatic to its own language, whereas the LOCK (timing_lock.sh) has real persistent shared
# state (the lock directory's metadata schema) that must stay in sync and is therefore always
# invoked through the one canonical script instead.

EXP_LAUNCH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

_exp_launch_log() { echo "[exp-launch] $*" >&2; }

_exp_launch_pgid_of() {
  ps -o pgid= -p "$1" 2>/dev/null | tr -d '[:space:]'
}

_exp_launch_group_alive() {
  pgrep -g "$1" >/dev/null 2>&1
}

_exp_launch_mark() {
  local state_file="$1" state="$2" tmp
  [[ -z "$state_file" ]] && return 0
  tmp="$(mktemp "${state_file}.XXXXXX" 2>/dev/null)" || return 0
  printf '%s\n' "$state" > "$tmp"
  mv -f "$tmp" "$state_file"
}

# ---- launch: begin/end primitives (flexible -- works around ANY existing backgrounding
#      pattern, including a compound subshell with its own redirects) + a start()
#      convenience wrapper for the common single-command case. ---------------------------
_EXP_LAUNCH_PRIOR_M=0

exp_launch_begin() {
  case "$-" in
    *m*) _EXP_LAUNCH_PRIOR_M=1 ;;
    *)   _EXP_LAUNCH_PRIOR_M=0 ;;
  esac
  set -m
}

# exp_launch_restore_m: restore the shell's prior monitor-mode setting, IN THIS SHELL. Call it
# DIRECTLY (never via $(...) or <(...)) -- a subshell's `set +m` does not propagate back, so the
# caller would leak monitor mode (job-control "Done"/"Terminated" chatter, altered SIGCHLD). Use
# this when you capture the leader pid inline (pgid==pid under `set -m`, so you don't need
# exp_launch_end's ps round-trip and its process-substitution subshell).
exp_launch_restore_m() {
  (( _EXP_LAUNCH_PRIOR_M )) || set +m
}

# exp_launch_end PID: call immediately after backgrounding (with the resulting $!). Restores
# the shell's prior monitor-mode setting and prints "PID PGID" on stdout.
exp_launch_end() {
  local pid="$1" pgid
  (( _EXP_LAUNCH_PRIOR_M )) || set +m
  pgid="$(_exp_launch_pgid_of "$pid")"
  printf '%s %s\n' "$pid" "${pgid:-$pid}"
}

# exp_launch_start LOG_PATH CMD [ARGS...]
# Convenience wrapper for the common case: CMD's stdout+stderr combined into LOG_PATH (this
# repo's own convention -- every driver script here already does "> log 2>&1"). Prints
# "PID PGID" on stdout. For a compound command (a subshell with its own internal structure,
# e.g. roofline_run.sh's `( set +e; ...; echo EXIT ) > log 2>&1 &`), use exp_launch_begin /
# exp_launch_end directly around the existing background launch instead of routing it
# through this function -- see roofline_run.sh's own run_one() for that pattern.
exp_launch_start() {
  local log_path="$1"; shift
  exp_launch_begin
  "$@" < /dev/null > "$log_path" 2>&1 &
  local leader_pid=$!
  exp_launch_end "$leader_pid"
}

# exp_launch_cancel PGID [STATE_FILE] [TIMEOUT_S]: see module header for the protocol.
# Returns 0 once verified clean, 1 if something survived even SIGKILL (should not happen
# short of an uninterruptible kernel wait -- logged loudly if it does).
exp_launch_cancel() {
  local pgid="${1:?exp_launch_cancel: pgid required}"
  local state_file="${2:-}"
  local timeout_s="${3:-10}"

  _exp_launch_mark "$state_file" "cancelling"

  if ! _exp_launch_group_alive "$pgid"; then
    _exp_launch_log "cancel($pgid): already gone, nothing to signal"
    _exp_launch_mark "$state_file" "cancelled"
    return 0
  fi

  _exp_launch_log "cancel($pgid): SIGTERM to process group"
  kill -TERM "-$pgid" 2>/dev/null || true

  local waited=0
  while (( waited < timeout_s )); do
    _exp_launch_group_alive "$pgid" || break
    sleep 1
    waited=$((waited + 1))
  done

  if _exp_launch_group_alive "$pgid"; then
    _exp_launch_log "cancel($pgid): survived ${timeout_s}s of SIGTERM, sending SIGKILL to process group"
    kill -KILL "-$pgid" 2>/dev/null || true
    sleep 1
  fi

  if _exp_launch_group_alive "$pgid"; then
    ps -o pid,ppid,pgid,command -g "$pgid" 2>/dev/null >&2 || true
    _exp_launch_mark "$state_file" "cancel-failed"
    _exp_launch_log "cancel($pgid): FAILED -- descendants survived SIGKILL (see process list above)"
    return 1
  fi

  _exp_launch_mark "$state_file" "cancelled"
  _exp_launch_log "cancel($pgid): verified clean, no descendants remain"
  return 0
}

# exp_launch_group_pids PGID: lists current member pids (one per line), empty if none.
# Standalone helper for callers/tests that want to observe group membership directly
# instead of only a boolean via _exp_launch_group_alive.
exp_launch_group_pids() {
  pgrep -g "$1" 2>/dev/null
}

# ---- standalone execution guard: sourcing this file never runs anything or sets options --
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -uo pipefail

  _exp_launch_usage() {
    sed -n '2,45p' "${BASH_SOURCE[0]}"
  }

  cmd="${1:-}"; shift || true
  case "$cmd" in
    start)
      log_path="${1:-}"; shift || true
      [[ -n "$log_path" && $# -gt 0 ]] || { echo "[exp-launch] usage: start LOG_PATH CMD [ARGS...]" >&2; exit 2; }
      exp_launch_start "$log_path" "$@"
      ;;
    cancel)
      pgid="${1:-}"; state_file="${2:-}"; timeout_s="${3:-10}"
      [[ -n "$pgid" ]] || { echo "[exp-launch] usage: cancel PGID [STATE_FILE] [TIMEOUT_S]" >&2; exit 2; }
      exp_launch_cancel "$pgid" "$state_file" "$timeout_s"
      exit $?
      ;;
    -h|--help|"")
      _exp_launch_usage
      exit 0
      ;;
    *)
      echo "[exp-launch] unknown subcommand: $cmd (expected start|cancel)" >&2
      exit 2
      ;;
  esac
fi
