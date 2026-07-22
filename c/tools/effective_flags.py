"""Centralized ILI_/COLI_/FA_ env-alias resolution + pre-launch safety guards.

Single source of truth shared by c/tools/eval_glm.py (direct import) and the shell timing
drivers c/scripts/roofline_run.sh / c/scripts/run_abba_matrix.sh (subprocess CLI call) --
see each caller for exactly how. The point of routing everyone through ONE module is that
the alias-resolution logic (ILI_<name> > COLI_<name> > FA_<name>, mirroring glm.c's own
ili_env()) and the truthiness rule (atoi()-style: non-numeric strings parse as 0/false)
must never drift between three different languages (C engine, Python, bash) -- a guard
reimplemented three times is a guard that silently diverges the next time only one copy
gets updated.

This module intentionally does NOT try to reproduce the engine's full effective-flags
resolution (e.g. whether ILI_METAL actually initialized a working GPU, or whether a
model's has_dsa ends up true) -- that ground truth only exists once the engine has
actually started and is printed on its own "EFFECTIVE-FLAGS:" stderr line (glm.c,
print_effective_flags()), which c/scripts/provenance.sh captures into the run manifest.
This module's job is narrower and cheaper: catch known footguns from the REQUESTED
environment alone, BEFORE paying for a (possibly hours-long) engine launch.

CLI:
  python3 effective_flags.py check-metal-prefill-guard
      exit 0, no output, if safe to launch.
      exit 2, one-line message on stderr, if ILI_METAL_PREFILL is truthy while
      ILI_METAL is not =1 (the footgun this guard exists for: METAL_PREFILL only ever
      takes effect inside glm.c's `if(ili_env("METAL") && atoi(ili_env("METAL")))`
      block, so setting it without METAL=1 does not error -- it silently falls back to
      an all-CPU run, which is how a prior run cost 3h14m instead of the intended
      GPU-accelerated time).
"""
import os
import re
import sys

_PREFIXES = ("ILI_", "COLI_", "FA_")

# atoi()'s real C prefix-parse: optional leading whitespace, an optional sign,
# then the LEADING run of decimal digits (stopping at the first non-digit,
# trailing garbage ignored). No match at all == atoi's "0 if no digits found".
_ATOI_RE = re.compile(r"^[ \t\n\v\f\r]*([+-]?\d+)")


def ili_env(name, env=None):
    """ILI_<name> > COLI_<name> > FA_<name>, mirroring glm.c's ili_env() exactly
    (first prefix that is SET wins, even if its value is the empty string)."""
    env = env if env is not None else os.environ
    for prefix in _PREFIXES:
        v = env.get(prefix + name)
        if v is not None:
            return v
    return None


def which_alias(name, env=None):
    """Name of the env var that actually supplied `name`'s value (for error messages),
    or the canonical ILI_ spelling if none is set."""
    env = env if env is not None else os.environ
    for prefix in _PREFIXES:
        if env.get(prefix + name) is not None:
            return prefix + name
    return "ILI_" + name


def truthy(v):
    """Mirrors glm.c's `atoi(ili_env(...))` truthiness check EXACTLY: unset/None is
    false; otherwise this replicates real C atoi() semantics via _ATOI_RE -- skip
    leading whitespace, an optional sign, then consume the LEADING run of decimal
    digits and stop at the first non-digit (ignoring any trailing garbage), 0 if no
    digits were found at all.

    This is deliberately NOT `int(v.strip()) != 0` guarded by a ValueError catch:
    that requires the WHOLE (stripped) string to be a clean Python int literal,
    which is a stricter, DIFFERENT rule than atoi()'s prefix-parse-and-stop. They
    diverge on any value with a numeric prefix followed by non-digit trailing
    content -- e.g. atoi("1x") == 1 and atoi("1.5") == 1 in C (both truthy), but
    `int("1x")`/`int("1.5")` both raise ValueError, so the old implementation
    returned False for both. That silently defeated check_metal_prefill_guard()
    below for exactly the values it exists to catch: ILI_METAL_PREFILL="1x" (or
    "1.5", etc.) with ILI_METAL unset would atoi()-truthily enable prefill-without-
    metal in the real engine, while the old truthy() reported it as falsy and let
    the guard wave the launch through."""
    if v is None:
        return False
    m = _ATOI_RE.match(v)
    if not m:
        return False
    return int(m.group(1)) != 0


def check_metal_prefill_guard(env=None):
    """Returns None if safe to launch, else a human-readable error string.

    The footgun: ILI_METAL_PREFILL set truthy while ILI_METAL is not enabled. glm.c
    only ever reads METAL_PREFILL inside the `if(ili_env("METAL") && atoi(...))` guard
    (see g_metal_prefill assignment) -- so requesting METAL_PREFILL without METAL is not
    an error at all, it is silently ignored, and the engine runs entirely on CPU. On a
    744B-class model that difference is hours, not seconds.
    """
    env = env if env is not None else os.environ
    metal = ili_env("METAL", env)
    prefill = ili_env("METAL_PREFILL", env)
    if truthy(prefill) and not truthy(metal):
        prefill_var = which_alias("METAL_PREFILL", env)
        metal_var = which_alias("METAL", env)
        return (
            "refusing to launch: {pv}={pval!r} is truthy but {mv}={mval!r} is not =1. "
            "METAL_PREFILL only takes effect when METAL is enabled (glm.c gates it inside "
            "the METAL init block) -- without METAL=1 this silently falls back to an "
            "all-CPU run instead of erroring (the footgun that once cost a 3h14m all-CPU "
            "eval). Fix: set {mv}=1, or unset {pv}."
        ).format(pv=prefill_var, pval=prefill, mv=metal_var, mval=metal)
    return None


def predicted_effective_flags(env=None):
    """Best-effort, ENGINE-FREE prediction of METAL/METAL_PREFILL's effective values from
    the launcher environment alone -- for scripts/provenance.sh's manifest (recorded
    BEFORE the engine runs, so the engine's own ground-truth "EFFECTIVE-FLAGS:" stdout
    line, glm.c's print_effective_flags(), does not exist yet to capture). "predicted"
    because it can only apply the same master-switch gating rule the guard uses; it
    cannot know whether ILI_METAL=1 would actually succeed at Metal device init (that
    is real engine ground truth, not derivable from env). METAL_PREFILL's prediction IS
    exact, though, since glm.c's own gating is exactly this same env-level rule (see
    check_metal_prefill_guard's docstring) with no runtime component.
    """
    env = env if env is not None else os.environ
    metal_req = ili_env("METAL", env)
    prefill_req = ili_env("METAL_PREFILL", env)
    metal_requested_on = truthy(metal_req)
    prefill_eff = metal_requested_on and truthy(prefill_req)
    return {
        "metal": {"requested": metal_req, "effective_if_init_succeeds": metal_requested_on},
        "metal_prefill": {
            "requested": prefill_req,
            "effective": prefill_eff,
            "reason": "" if (not truthy(prefill_req) or prefill_eff) else "master_disabled",
        },
        "note": "predicted from launcher_env only, pre-launch -- not measured from a "
                "running engine; see glm.c's own EFFECTIVE-FLAGS stdout line for ground truth",
    }


def _main(argv):
    if len(argv) != 2 or argv[1] != "check-metal-prefill-guard":
        print(__doc__, file=sys.stderr)
        return 2
    msg = check_metal_prefill_guard()
    if msg is not None:
        print(msg, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
