#!/usr/bin/env bash
# Fast pre-push gate: performance-theory drift check + non-Metal Python
# unittests + a repo-wide guard against unresolved merge-conflict markers.
#
# NOT auto-installed by this repo. To wire it up as your local pre-push
# hook, symlink or copy this script to .git/hooks/pre-push (optional).
#
# On purpose, this does NOT redo the portable engine (re)build or the C
# unit tests that `make -C c check` / .github/workflows/ci.yml also run --
# that build is the slow part (~2 minutes) and is already covered by CI
# before merge; this script exists to be fast enough to run on every
# `git push` without friction. Metal-only coverage (c/tests/*.mm, built and
# run only via `make metal-test*`) is out of scope here for the same
# reason CI skips it: it needs Xcode + a Metal-capable GPU, which a
# pre-push hook cannot assume any more than a Linux CI runner can.
#
# Git invokes a pre-push hook with two positional args (remote name,
# remote URL) and pushed ref-update lines on stdin; neither is used below
# -- every check here gates on working-tree content, not on the specific
# commit range being pushed -- so both are accepted and ignored on purpose
# (leaving stdin undrained is harmless: git does not require it read).

set -uo pipefail   # NOT -e: see scripts/run_abba_matrix.sh's own comment on
                    # why this codebase avoids it in multi-step drivers --
                    # every step below is checked explicitly instead.

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
status=0

echo "[pre-push] performance-theory drift check..."
if "$PYTHON" docs/gen_performance_theory.py --check; then
    echo "[pre-push] OK: docs/PERFORMANCE_THEORY.md matches docs/performance-theory.json"
else
    echo "[pre-push] FAILED: docs/PERFORMANCE_THEORY.md is stale. Run:" >&2
    echo "    python3 docs/gen_performance_theory.py" >&2
    echo "  and commit the regenerated file." >&2
    status=1
fi

echo "[pre-push] conflict-marker guard (repo-wide)..."
if "$PYTHON" c/tools/check_no_conflict_markers.py; then
    echo "[pre-push] OK: no unresolved conflict markers found"
else
    echo "[pre-push] FAILED: unresolved merge-conflict markers found (listed above)." >&2
    status=1
fi

echo "[pre-push] Python unittest suite (c/tests, excluding Metal-only)..."
if ( cd c && "$PYTHON" -m unittest discover -s tests ); then
    echo "[pre-push] OK: c/tests unittest suite passed"
else
    echo "[pre-push] FAILED: c/tests unittest suite did not pass." >&2
    status=1
fi

if [[ "$status" -ne 0 ]]; then
    echo "[pre-push] one or more fast checks failed above; push aborted." >&2
    echo "[pre-push] to push anyway (not recommended): git push --no-verify" >&2
fi

exit "$status"
