#!/usr/bin/env bash
# mode15_multiturn_soak.sh -- pre-merge verification-gate soak test for the
# expert_bytes_probe()/cap_for_ram() cap-sizing fix (commit d1aaa7e, "mode15:
# fix cap-sizing OOM root cause + fail-loud hardening for the multi-turn
# silent-death bug"). Drives MANY turns of ONE persistent `ili chat` session
# against the real Mode-1.5 (MH01) container, deliberately varying prompt
# TOPIC every turn (to spread MoE routing across as many distinct experts as
# possible) and sending `:reset` between every turn -- the EXACT bug
# condition: :reset clears KV/conversation state but deliberately does NOT
# clear the per-layer LRU expert cache (see expert_bytes_probe()'s and
# cap_for_ram()'s own comments in glm.c), so the cache fills gradually across
# turns and only a real multi-turn session can ever exercise the accumulation
# this bug lived in. A single `ili run`, or a chat session ended after one
# or two turns (the prior repro's limitation -- RSS "held ~50GB" and never
# reproduced the overshoot), cannot.
#
# SCOPE NOTE (scaled soak): run at a LOWERED --ram budget, not the ~114 GB
# full/production budget, specifically so this can run safely on a machine
# that may also be holding a foreign process resident (see this repo's own
# verification-gate instructions -- never approach the physical RAM ceiling).
# cap_for_ram()'s own sizing math targets projected peak RSS ~= the requested
# --ram budget REGARDLESS of its absolute value (confirmed empirically: a
# calibration run against this exact container at --ram 40 printed "cap
# raised 8->16 ... projected peak 39.2 GB"), so a clean plateau at a smaller
# budget is strong evidence the same arithmetic holds at full scale -- the
# fix's core claim (expert_bytes_probe returning the DECODED, not the
# on-disk COMPRESSED, byte count for an MH01 tensor) does not depend on the
# budget's absolute size.
#
# Usage:
#   mode15_multiturn_soak.sh MODEL_DIR [RAM_GB] [NGEN] [OUTDIR]
#
# Requires: ./glm built with MODE15=1 (see Makefile).
#
# NOTE for agent/automation callers backgrounding this script (same lesson
# mode15_e2e_certify.sh already records): launch with `nohup ... & disown`,
# not just `& disown` -- a plain backgrounded child on this machine has been
# observed to die silently with no OOM/crash trace when its launcher's own
# tracking went away.
set -uo pipefail

CDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CDIR"

MODEL_DIR="${1:?usage: mode15_multiturn_soak.sh MODEL_DIR [RAM_GB] [NGEN] [OUTDIR]}"
RAM_GB="${2:-40}"
NGEN="${3:-4}"
OUTDIR="${4:-$CDIR/bench-m5max/mode15-multiturn-soak-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/soak.log"

log() { echo "[mode15-soak $(date '+%H:%M:%S')] $*" >&2; }

[[ -x ./glm ]] || { echo "error: ./glm not built. Run: make MODE15=1 glm" >&2; exit 1; }
if ! strings ./glm 2>/dev/null | grep -q "mode-1.5 Huffman-compressed"; then
  log "warning: ./glm does not appear to reference mode-1.5 strings -- was it built with MODE15=1?"
fi
if pgrep -f "$CDIR/glm" >/dev/null 2>&1; then
  echo "error: an iliria engine process is already running under this worktree -- refusing to start a second heavy run" >&2
  pgrep -fl "$CDIR/glm" >&2
  exit 1
fi

# Deliberately wide-domain short-answer prompt set (numbers/math, geography,
# chemistry, history, biology, physics, literature, astronomy, computing,
# economics...) -- topical variety is the only lever available to spread
# MoE routing across many DISTINCT experts/layers without controlling
# routing directly. All short-answer (bounded decode length, keeps wall-clock
# cost per turn bounded even at NGEN default) -- same rationale as
# mode15_e2e_certify.sh's own prompt set.
PROMPTS=(
  "What is 2+2? Answer with just the number."
  "What is the capital of France? Answer with just the city name."
  "What is the chemical symbol for gold? Answer with just the symbol."
  "What year did World War II end? Answer with just the year."
  "What is the capital of Japan? Answer with just the city name."
  "What is the chemical symbol for iron? Answer with just the symbol."
  "What is 9 times 7? Answer with just the number."
  "Who wrote Romeo and Juliet? Answer with just the name."
  "What is the largest planet in our solar system? Answer with just the name."
  "What is the boiling point of water in Celsius? Answer with just the number."
  "What is the capital of Italy? Answer with just the city name."
  "What is the square root of 64? Answer with just the number."
  "What is the chemical symbol for sodium? Answer with just the symbol."
  "Who painted the Mona Lisa? Answer with just the name."
  "What is the tallest mountain on Earth? Answer with just the name."
  "What is the capital of Germany? Answer with just the city name."
  "How many continents are there on Earth? Answer with just the number."
  "What is the chemical symbol for oxygen? Answer with just the symbol."
  "What is the freezing point of water in Fahrenheit? Answer with just the number."
  "Who is credited with inventing the telephone? Answer with just the name."
  "What is the capital of Spain? Answer with just the city name."
  "What is 12 divided by 4? Answer with just the number."
  "What is the longest river in the world? Answer with just the name."
  "What is the chemical symbol for silver? Answer with just the symbol."
  "What planet is known as the Red Planet? Answer with just the name."
  "What is the capital of Russia? Answer with just the city name."
  "How many legs does a spider have? Answer with just the number."
  "What is the currency of Japan? Answer with just the name."
  "Who developed the theory of relativity? Answer with just the name."
  "What is the smallest prime number? Answer with just the number."
  "What is the capital of Canada? Answer with just the city name."
  "What gas do plants absorb from the atmosphere for photosynthesis? Answer with just the name."
  "What is the capital of Egypt? Answer with just the city name."
  "How many bones are in the adult human body? Answer with just the number."
  "What is the chemical symbol for carbon? Answer with just the symbol."
  "Who composed the Ninth Symphony? Answer with just the name."
  "What is the capital of Brazil? Answer with just the city name."
  "What is 15 minus 8? Answer with just the number."
  "What is the largest ocean on Earth? Answer with just the name."
  "What is the capital of India? Answer with just the city name."
)

PROMPT_FILE="$OUTDIR/prompts.txt"
: >"$PROMPT_FILE"
for p in "${PROMPTS[@]}"; do printf '%s\n:reset\n' "$p" >>"$PROMPT_FILE"; done
NPROMPTS=${#PROMPTS[@]}

log "starting multiturn soak: model=$MODEL_DIR ram=$RAM_GB ngen=$NGEN turns=$NPROMPTS"
log "log -> $LOG"
t0=$(date +%s)
SEED=1 AUTOPIN=0 REPIN=0 DRAFT=0 MTP=0 DIRECT=1 \
  ./ili chat --model "$MODEL_DIR" --ram "$RAM_GB" --temp 0 --ngen "$NGEN" \
  <"$PROMPT_FILE" >"$LOG" 2>&1
t1=$(date +%s)
log "session finished in $((t1 - t0))s"

# ---- parse + report -------------------------------------------------------
python3 - "$LOG" "$NPROMPTS" "$RAM_GB" <<'PY'
import re, sys

log_path, nprompts, ram_gb = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
text = open(log_path, errors="replace").read()

# ili's cmd_chat re-wraps long [RAM_GB=...]/[PIN]/[MTP]/[USAGE]/[DSA]/[KV]
# diagnostic lines from the engine's stderr via textwrap.wrap() to fit the
# terminal width, WITHOUT repeating the prefix on continuation lines (see
# ili's own cmd_chat, the "for l in elog.splitlines(): if l.startswith((...))"
# block) -- so the cap line can legitimately be split across 2+ physical
# lines in this log. Reconstruct logical lines before searching.
PREFIXES = ("[RAM_GB", "[PIN]", "[MTP]", "[USAGE]", "[DSA]", "[KV]")
logical_lines, cur = [], None
for raw_ln in text.splitlines():
    ln = raw_ln.strip()
    if ln.startswith(PREFIXES):
        if cur is not None: logical_lines.append(cur)
        cur = ln
    elif cur is not None and ln:
        cur += " " + ln
    else:
        if cur is not None: logical_lines.append(cur)
        cur = None
if cur is not None: logical_lines.append(cur)

# any of the three cap_for_ram() branches (glm.c):
#   "cap raised %d->%d: budget allows it (projected peak %.1f GB..."
#   "cap lowered %d->%d (projected peak %.1f GB)"
#   "cap=%d ok (projected peak %.1f GB)"
cap_m = None
for ll in logical_lines:
    m = re.search(r"\[RAM_GB=[^\]]*\]\s*(cap (?:raised|lowered) \d+->\d+.*\(.*?GB.*?\)|cap=\d+ ok.*\(.*?GB\))", ll)
    if m: cap_m = m; break
print("="*78)
print("MODE-1.5 MULTI-TURN SOAK REPORT")
print("="*78)
print(f"requested RAM_GB budget: {ram_gb}")
if cap_m:
    print(f"startup cap line: {cap_m.group(1).strip()}")
else:
    print("startup cap line: NOT FOUND (check raw log)")

# per-turn STAT footer, exactly as ili's cmd_chat prints it:
#   "  └─ {tok} tok · {tps:.2f} tok/s · hit {hit:.0f}% · RSS {rss:.1f} GB · {el:.0f}s"
turn_re = re.compile(r"└─\s*(\d+)\s*tok\s*·\s*([\d.]+)\s*tok/s\s*·\s*hit\s*(\d+)%\s*·\s*RSS\s*([\d.]+)\s*GB\s*·\s*(\d+)s")
turns = [{"tok": int(m.group(1)), "tps": float(m.group(2)), "hit": int(m.group(3)),
          "rss": float(m.group(4)), "wall_s": int(m.group(5))} for m in turn_re.finditer(text)]

deaths = [m.group(0) for m in re.finditer(r"\[engine terminated\].*", text)]

print(f"\nturns completed: {len(turns)} / {nprompts} prompts sent")
print(f"engine deaths detected ([engine terminated]): {len(deaths)}")
for d in deaths:
    print(f"  {d}")

if turns:
    print(f"\n{'turn':>4}  {'tok':>4}  {'tok/s':>7}  {'hit%':>4}  {'RSS GB':>7}  {'wall s':>6}")
    for i, t in enumerate(turns, 1):
        print(f"{i:>4}  {t['tok']:>4}  {t['tps']:>7.2f}  {t['hit']:>4}  {t['rss']:>7.1f}  {t['wall_s']:>6}")

    rss_vals = [t["rss"] for t in turns]
    max_rss = max(rss_vals)
    max_turn = rss_vals.index(max_rss) + 1
    n = len(rss_vals)
    half = max(1, n // 2)
    first_half_max = max(rss_vals[:half])
    second_half_max = max(rss_vals[half:]) if n > half else rss_vals[-1]
    # plateau heuristic: does RSS growth stop (second half doesn't meaningfully
    # exceed first half), and does the observed max stay at/under the
    # startup cap line's own projected-peak estimate (with slack for
    # measurement noise / transient activations)?
    growth_second_half = second_half_max - first_half_max
    print(f"\nmax RSS observed: {max_rss:.1f} GB (turn {max_turn})")
    print(f"first-half max RSS: {first_half_max:.1f} GB | second-half max RSS: {second_half_max:.1f} GB "
          f"| second-half growth over first-half: {growth_second_half:+.1f} GB")
    if cap_m:
        peak_m = re.search(r"projected peak ([\d.]+) GB", cap_m.group(1))
        if peak_m:
            projected = float(peak_m.group(1))
            print(f"startup projected peak: {projected:.1f} GB | max observed vs projected: "
                  f"{max_rss - projected:+.1f} GB")
    print(f"\nPLATEAU CHECK: {'PASS -- RSS did not grow materially in the second half of the run' if growth_second_half <= 2.0 else 'FAIL -- RSS kept climbing in the second half'}")
else:
    print("\nNO TURNS PARSED -- check the raw log for FATAL/error output or an early death.")

print(f"\nraw log: {log_path}")
PY
log "done."
