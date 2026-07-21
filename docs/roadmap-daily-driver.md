# iliria → agent daily driver: unified roadmap (2026-07-14)

Derived from a multi-pass analysis over the measured benchmark record. Unit that matters:
**wall-clock minutes per 20-turn agent session** (~22K ctx, ~500 out tok/turn).
Baseline today ≈ 148 min/session, TTFT 100–200 s/turn. Target ≤60 min, TTFT ≤15 s.
Non-disk floor ≈ 0.36–0.38 s/token ⇒ decode caps ~2.8 tok/s even at 100% hit.

## Steps (each gated before the next)
1. **Wire harness to `ili serve` + prefix KV reuse** (1–2 days, config only): long-lived serve,
   monotonic history, `.ili_kv`, `--kv-slots 1–2` (4 slots would steal ~29 GB from expert cache).
   → ~114 min, TTFT 5–15 s. Gate: turns 2–10 prefill only the delta; decode unchanged.
2. **Response-length discipline** (hours): THINK off, per-turn max_tokens, diffs-not-files.
   → ~69 min if 500→300 tok/turn holds. Gate: measured avg on 3 real sessions.
3. **Offline replay sim for tiered container** (½ day, CPU-only, after live matrix finishes):
   per-expert sizes (int4 salient / int2 bulk) in simulate_m5max_cache.py; refit disk model to BOTH
   measured points (K6 shows sublinear). Gate: ≥ +15% predicted, else stop.
4. **Mixed-precision container** (~5–6 days): saliency = routing mass + int2 reconstruction MSE;
   early layers forced int4 (uniform int2 collapsed 38/46→6/46); ~223 GB+ fits disk. Only lever
   consistent with settled facts: fewer bytes + more capacity. → ~1.65–1.8 tok/s, ~60 min.
   Gates: int4 `ili bench` baseline first; MC parity; long-generation collapse gate; frozen-state
   paired 3-prompt A/B at +5%. Contingency: int3 mid-tier.
5. **Arrival-order MoE overlap** (1–2 weeks): async miss reads at router time, cached GEMV while
   reads fly. Same bytes, exact routing (not prefetch). Evidence of headroom: blocked-pipe ≈ disk
   timer today. Ceiling ~+13–16%. → ~1.9–2.0 tok/s. Gate: blocked-pipe must drop strictly below
   expert-disk, plus paired +5%.
6. **`--topp 0.7` adaptive experts A/B** (hours): 8→~5.6 experts, never measured here; full quality
   gate required. → ~2.0–2.2 tok/s if it lands.

Stretch (cheap-probe first): prefill tool output on arrival; grammar tool-call drafts (fix XML
syntax + process-global GRAMMAR trap first); continuous batching only after measuring cross-stream
expert-union from existing traces.

## Step 1 gate — measured 2026-07-14 (M5 Max, `run-m5max-serve.sh`, 2×40960-token KV slots)

**Verdict: prefix-KV reuse PASS · delta-only prefill PASS · decode unchanged PASS ·
TTFT ≤15 s target FAIL for normal-size turn deltas — prefill throughput is the new wall.**

Harness: `c/scripts/serve_gate.py` (agent-style transcript, ~1.5K new code-context
tokens/turn, monotonic history, THINK off, max_tokens 300, greedy). Raw data in
`c/bench-m5max/serve-gate-20260714/` (local, gitignored).

- **Delta-only prefill works exactly.** Engine `[API]` log per turn: turn 2 prefix
  1639/3096 → prefill 1457; turn 3 prefix 3395/4932 → prefill 1537. Never a full
  re-prefill. `.ili_kv` persistence verified across a server restart: slot 0 resumed
  3249 tokens in **0.1 s**, and the next request prefilled **0** tokens (TTFT 3.07 s
  vs 436.65 s for the same prompt cold — **142×**, N=1). This 142× figure was later
  retired as uncitable (the cold and resume observations used different prompts, and
  neither stated whether TTFT included process/model load); the certified replacement
  is **6.8× [5.0×, 8.0×]**, a same-prompt ABBA result — see
  `c/bench-m5max/ili-kv-resume-abba/REPORT.md`.
- **Decode is unchanged with KV reuse:** 1.56–1.66 tok/s across reuse turns vs
  1.45 tok/s on the fresh-context control turn (settled warm band 1.42–1.58).
  Control (nonce-broken prefix, same history length): full prefill of 2955 tokens,
  TTFT **1884 s**; the reuse turn at the same point prefilled 1457 and hit TTFT
  **821 s** — TTFT is exactly proportional to prefilled tokens (~1.6–1.8 tok/s
  prefill both ways).
- **TTFT scales with delta size, not history size** — that is the win — but measured
  serve **prefill throughput is only ~1.0–1.8 tok/s** (turn 1 cold 3.1 tok/s at 1.4K
  ctx, degrading with context). Batch-union prefill *is* engaged (layer-by-layer over
  the whole batch), but each MoE layer streams essentially all 256 experts (~4.9 GB)
  from disk for a >1K-token batch: prefill is disk-bound at roughly decode speed.
  Measured TTFT: 3 s (delta 0), 821 s (delta 1457), 1523 s (delta 1537 at 4.9K ctx).
- **The roadmap's "TTFT 100–200 s → 5–15 s" assumed ~100–200 tok/s prefill; that was
  wrong by ~100×.** Session projection at measured rates (20 turns, ~1.5K new
  tok/turn, 300 out/turn): without reuse ≈ 380K prefill tokens ≈ 35–60 h; with reuse
  ≈ 30K delta tokens ≈ 5–7 h + ~1 h decode. Prefix reuse delivers its ~8–12×
  prefill reduction as promised, but ≤60 min sessions additionally require raising
  prefill throughput (fewer bytes/expert — Step 4 — helps prefill proportionally;
  I/O overlap — Step 5 — likewise). TTFT ≤15 s holds only for deltas ≤ ~25 tokens
  (e.g. short user turns without new tool output).

### Serve-gate update — prefill fix landed (2026-07-14, commits 8d26555/cc96775/ee81bbb)

The "prefill is disk-bound" clause above was falsified by the prefill-I/O study:
in-engine PROFILE showed a 1361-token prefill = 84.8% CPU attention, 0.7% disk.
Fix shipped: S-row projection GEMMs + Metal S>4 prefill attention behind
`ILI_METAL_PREFILL`. User ruling 2026-07-14: kernel-family rounding variance
counts as a faithful forward, so the greedy prefill-continuation forks vs the CPU
path are acceptable; the ruling tied default-ON promotion to turn-2 TTFT <=120 s,
which measured 130.4 s (9% miss) — launcher default stays 0, export
`ILI_METAL_PREFILL=1` (or flip the one-line default) for the measured 6-9x TTFT.
Serve-gate rerun (3 reuse turns + control, same recipe, ILI_METAL_PREFILL=1):

- TTFT: turn 1 40.7 s (390-tok delta), turn 2 **130.4 s** (1432-tok delta at 3.1K
  ctx, was 821 s → 6.3x), turn 3 **168.3 s** (1492-tok delta at 4.9K ctx, was
  1523 s → 9.1x). Prefill throughput ~1.6 → **~9-11 tok/s**.
- Decode unchanged with the feature on: 1.62-1.64 tok/s on clean turns (warm band
  1.42-1.58; turn-3's 1.43 decode and the first control run were contaminated by a
  stray mid-run `sudo purge`; the clean control rerun: TTFT 430.1 s for a 4567-token
  full re-prefill, decode 1.43 tok/s). Formal `serve_gate.py --verdict`: **PASS**
  (delta-exact prefill; reuse decode mean 1.527 >= 90% of control 1.427).
- Delta-only prefill still engages exactly (prefix 948/1338, 1637/3069, 3368/4860).
- **Turn-2 TTFT <=120 s target: MISSED by ~9%** (130.4 s). Post-fix profile of that
  turn: attention 57.9 s (GPU), expert-MoE-matmul 32.2 s, other (router/norms/
  scatter) 37.6 s, disk 1.8 s — the next TTFT lever is MoE GEMM + glue, not I/O.
  TTFT <=15 s remains delta<=25-token territory; a 1.5K-token delta now costs ~2 min
  instead of ~14-25 min.

## Step 2 — status (2026-07-14)

- Per-request `max_tokens`/`max_completion_tokens` and thinking passthrough
  (`enable_thinking`, `reasoning_effort`, global `ILI_THINK` default **off**) were
  already wired in `openai_server.py` — verified in code and exercised by the gate;
  no server changes needed. `run-m5max-serve.sh` documents the defaults
  (THINK off; ~200 tokens for tool-call turns, ~400 for code edits, ~700 for
  from-scratch generation; diffs-not-files system-prompt snippet).
- Measured on the gate transcript: with the discipline defaults every turn capped at
  300 output tokens (`finish_reason: length` — the cap binds on code-review turns).
  At the measured ~1.5 tok/s decode (warm band 1.42–1.58 tok/s), each 100 output tokens ≈ 63 s, so 500→300
  tok/turn saves ~2 min/turn ≈ 40 min per 20-turn session — the Step 2 arithmetic
  holds; the cap (plus THINK off) is the binding lever, the prompt snippet is belt
  and braces.

## Campaign item 1 — DSA lightning-indexer extraction + A/B (2026-07-14)

**Verdict: weights extracted and installed; DSA stays OPT-IN (`ILI_DSA=0` launcher default).**
With `ILI_METAL_PREFILL=1` now the default, DSA is a net loss at daily-driver contexts:
its selection path is CPU-only and disqualifies the Metal prefill kernel (`!dsel` precondition
in `attention()`), so enabling it trades a 6-9x GPU prefill win for a linear-but-CPU sparse path.

**Extraction.** `convert_fp8_to_int4.py --indexer` from `zai-org/GLM-5.2-FP8`: 20 of 141 shards
touched (~108 GB traffic, not the ~756 GB the converter's warning suggests — the indexer lives in
20 shards), 21 'full' indexer layers x 8 tensors = 168 tensors, **188 MB** int8 (+f32 norms),
`out-idx-00000..00019.safetensors` installed in the model dir. Engine autodetect confirmed:
`[DSA] indexer active: top-2048 sparse attention beyond 2048 context tokens`.

**A/B, frozen `.fa_usage`, greedy, M5 Max** (short = rust_queue 28-token prompt, 112 gen;
long = 3.8K-token code-review prompt, 32 gen):

| cell | wall | attention timer | tok/s | hash |
|---|---|---|---|---|
| short, DSA=0 (a) | 76.6s | 17.6s | 1.46 | `135a40223a294e1f` |
| short, DSA=0 (b) | 75.4s | 17.2s | 1.49 | `5ebb95b9a8ea17d1` |
| short, DSA=1 | 83.9s | 23.1s | 1.33 | `79efb41af085c0f9` |
| long, MP=0, DSA=0 | 2998s | 2781.9s | — | `3581e1f7deb1f3e9` |
| long, MP=0, DSA=1 | 1750s | 1557.1s | — | `6a5e3cf0b6ffc6e6` |
| long, MP=1, DSA=0 | **362s** | **161.7s** | — | `aa42b37d987a87c4` |
| long, MP=1, DSA=1 | 1780s | 1587.3s | — | `89d457115f0d7898` |

(MP = `ILI_METAL_PREFILL`. Long TTFT ~= wall minus ~25s decode+load. The MP=0 long pair ran
before the usage-freeze protocol; hit rates matched 41.0/41.3% so the timing comparison holds.)

- **Short context (<2048): DSA is pure overhead** — ~-10% decode, +5.5s attention per 112
  tokens, from per-token indexer-key upkeep on the 21 full layers (selection never engages).
- **CPU prefill (MP=0): DSA wins big** — -42% wall / -44% attention at 3.8K ctx, growing with T.
- **GPU prefill (MP=1, the new default): DSA loses 4.9x** at 3.8K ctx. MP=1+DSA=1 ==
  MP=0+DSA=1 (1780 vs 1750s): the GPU kernel never engages once DSA allocates selection.
  Est. crossover where CPU-sparse beats GPU-dense: ~36K ctx (161.7s x (T/3.8K)^2 vs 1557s x T/3.8K).
- **Hashes / exactness.** Single-thread CPU + frozen `.fa_usage` is bit-deterministic (det0a ==
  det0b, `45328897213df08b`). Forced select-all (`DSA_FORCE=1`) under those same deterministic
  conditions still forks (`25a1adcc19c5cf31`): the top-k pass permutes key order (threshold pass,
  then ties), changing FP summation order — bitwise equality vs dense is unattainable by
  construction; output stays coherent (argmax-tie class, accepted per user ruling 2026-07-14).
  Note: the standard multi-threaded Metal config self-forks run-to-run even with frozen usage
  (short DSA=0 a vs b above), so hash comparison is meaningless there.
- **Follow-up that would change the verdict:** a DSA-aware Metal prefill kernel (score+select on
  GPU, or CPU select + GPU sparse attention). Until then DSA only helps CPU-bound prefill or
  >~36K contexts, which current KV slots (40K cap) barely reach.

## Will NOT do (kill numbers)
More prefetch (K6 −6.96%); cache-policy work (veto −26%, zero hit change — capacity is the wall);
Metal 4 as perf claim (+2.05%, infra only); static pinning/hotset (flat/diffuse); MTP (3× slower);
hetero CPU MoE (loss); uniform int2 / full residency (quality collapse / needs ~1 bit).

## Bottom line
Decode tops out ~2.0–2.2 tok/s single-stream on this hardware; 10–15 tok/s interactive is
unreachable. The daily-driver bar doesn't need it: Steps 1–2 (two days, zero engine risk) deliver
~69 min sessions with 5–15 s TTFT — 80% of the win. Verified floor 2.1× effective; plausible
endgame ~3×. Ship Steps 1–2 the day the matrix finishes.
