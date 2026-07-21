#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 ]] || { echo "Usage: ./benchmark-m5max.sh MODEL_DIR [PROMPT]" >&2; exit 2; }
MODEL_DIR="${1%/}"
PROMPT="${2:-Explain how a compiler optimizer transforms an SSA graph.}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p bench-m5max
NGEN="${ILI_BENCH_NGEN:-112}"

for ram in 110 114 118; do
  for draft in 0 2 4 6; do
    out="bench-m5max/ram-${ram}-draft-${draft}.log"
    echo "=== RAM=$ram DRAFT=$draft ===" | tee "$out"
    TEMP=0 SEED=1 ILI_IGNORE_PROFILE=1 ILI_AUTOPIN=0 ILI_REPIN=0 \
    ILI_RAM_GB="$ram" ILI_DRAFT="$draft" ILI_NGEN="$NGEN" \
      ./run-m5max-fast.sh "$MODEL_DIR" run "$PROMPT" 2>&1 | tee -a "$out"
  done
done

python3 - "$PWD/bench-m5max" <<'PY'
from pathlib import Path
import re, sys

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.log")):
    text = path.read_text(errors="replace")
    rates = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?) tok/s", text)]
    hits = [int(x) for x in re.findall(r"hit ([0-9]+)%", text)]
    fw = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?) tok/fw", text)]
    rows.append((path.name, rates[-1] if rates else 0.0, hits[-1] if hits else -1, fw[-1] if fw else 0.0))

print("\nfile\tfinal_tok_s\tlast_hit_pct\tlast_tok_fw")
for row in sorted(rows, key=lambda r: r[1], reverse=True):
    print(f"{row[0]}\t{row[1]:.3f}\t{row[2]}\t{row[3]:.2f}")
PY
