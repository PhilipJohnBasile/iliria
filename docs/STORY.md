# iliria: running a 744B model off a laptop's SSD — an honest account

**How to read the confidence tags in this document** (the discipline this
project applies to its own internal theory database, carried into this
outward-facing account):

- **[MATRIX-MEASURED]** — passed the house confirmation matrix: 3 held-out
  prompts, warm/cold, 3 interleaved ABBA trials, frozen `.fa_usage`,
  hash-gated on the exact output-token stream, +5% median-paired promotion
  bar (`c/ab-m5max-k6-matrix.sh`).
- **[MEASURED, weaker class]** — a real before/after measurement that has
  *not* cleared the house matrix above (`measured/cross-session` in this
  project's own evidence taxonomy) — real, but a softer claim.
- **[PROJECTED]** — arithmetic derived from measured inputs, not itself a
  direct measurement.
- **[PENDING]** — pre-registered, gated, not yet run. Treated as a loud
  placeholder, not a rounding-up to "basically done."

---

## 1. TL;DR

iliria is a ~4,000-line, dependency-free C engine that runs **GLM-5.2, a
744-billion-parameter Mixture-of-Experts model, entirely locally on a single
M5 Max MacBook Pro with 128 GB of unified memory** — by keeping the ~17B-parameter
dense core resident in RAM and streaming the ~19,200 per-token-routed expert
tensors live from internal NVMe on every forward pass
(`README.md`). The model itself is never pruned,
distilled, fine-tuned, or otherwise degraded from its shipped int4
quantization — what runs is the full 744B model, not a smaller stand-in for
it.

Five results carry this account, each with a citation trail an owner or future
collaborator can re-walk without re-asking the author:

1. **744B, unmodified, local.** ~370 GB on disk, 9.9 GB resident, 1.42–1.58
   tok/s warm decode **[MATRIX-MEASURED]** (§3).
2. **A logical-bandwidth normalization against a much smaller, non-MLX
   competitor — a diagnostic, not a performance comparison.** Normalized
   for gross active-expert bytes/token, iliria and `anemll/flash-moe` both
   land around **~17 logical routed-expert GB/s** at their respective
   chosen configs — stated in the same breath as the raw, un-normalized
   gap this normalization is explaining: **1.42–1.58 tok/s vs. flash-moe's
   12.9 tok/s**, an ~8–9x difference that is real and uncorrected. The
   normalization shows *some* of that gap traces to per-token payload
   size; it is not a claim that the gap is "entirely" model size —
   flash-moe's 12.9 tok/s config runs its **top-4** experts (vs. the
   upstream Qwen3.5-397B reference config's **top-10**) at **Q3**
   quantization (vs. iliria's int4), so routing width and quantization
   format differ too, and neither system's output quality at its
   benchmarked config has been cross-measured here (§4).
3. **A causal, not merely correlational, proof of the I/O-bound design.**
   A pre-registered, randomized delay-injection experiment (a 16 ms
   synthetic delay injected per expert read) shows that adding synthetic
   latency to the expert-read path causes a matching increase in decode
   latency: mean +365.2s (95% CI [359, 369]s — the CI is on the mean, not
   the median) across **5 paired blocks**, median +368s, 0 failures (§5).
4. **A quality gate that killed a real bad result before it shipped**
   (compression Gate 3, case D — a single int2-tier design killed on 5
   layers / 1 capture, an activation-error blowup an external reviewer predicted the
   mechanism for in advance; this closes one design, not int2 quantization
   generally) (§8).
5. **Honest negative results, kept rather than buried**: Apple's own SpecMD
   "Least-Stale" eviction policy is bit-for-bit identical to iliria's
   plain per-layer LRU **in a demand-only replay** — this does not test
   the async, out-of-order prefetch regime or the shared-cache condition
   Least-Stale was actually built for, so "identical here" is not "proven
   equivalent" (§6a); iliria's own router-lookahead prefetch (PILOT K6)
   *wins* on hit rate and *loses* on wall-clock, and was reverted (§6b).

**Two placeholders from the prior day's revision have since moved — neither to
a ship.** Mode-1.5's predicted decode-speed gain is no longer unmeasured: it
has been measured, end-to-end, on the real container, and the measurement is
a slowdown, not the predicted gain (§9a). A sibling project's (the sibling engine's, not
iliria's) layer-pruning safety work remains governance-blocked and outside
this release (§9b). Section 12 states plainly what this document is *not*
claiming.

---

## 2. What iliria is, in one paragraph

iliria (Apache-2.0, a fork of `JustVugg/colibri`) runs GLM-5.2
(744B-parameter MoE, top-8-of-256 experts/layer across 75 MoE layers, MLA
attention, DeepSeek-V3-style sigmoid router) at int4 precision, quantized
from the MIT-licensed `zai-org/GLM-5.2-FP8` base model
(`docs/DEFENSIBLE-ASSETS.md` §e). A 744B MoE model
activates only ~40B parameters per token, and only the ~11 GB belonging to
that token's routed experts actually changes token-to-token
(`README.md` §"The idea"). iliria's design follows directly from that fact:
the dense core (~17B params, ~9.9 GB at int4) stays resident; the routed
experts (~370 GB at int4) live on disk and are streamed on demand through a
per-layer LRU cache, a learned pinned hot-store, and the OS page cache as a
free second tier. No BLAS, no Python, no GPU requirement at runtime — Metal
acceleration is opt-in and additive.

---

## 3. The headline number: 744B, unmodified, local

**Honest numbers** (M5 Max, 128 GB unified memory; `README.md`
§"Honest numbers", `DEFENSIBLE-ASSETS.md` §d):

| metric | value | confidence |
|---|---|---|
| Model on disk (int4 container) | ~370 GB (README's headline figure). A byte-math cross-check in §4 uses a figure reported as "359 GB" for the same container — **this is very likely a GB-vs-GiB unit mismatch, not 3% rounding**: 359.35 GiB (binary) converts to 385.85 GB (decimal), which is *larger* than the ~370 GB headline, not ~3% smaller. The two citations have not been reconciled to a single unit-consistent number as of this account — treat "~370 GB" as the citable headline figure and the "359" figure in §4 as unit-ambiguous pending reconciliation, not as independent confirmation of it | measured, pending unit reconciliation |
| Resident RAM (dense core, int4) | 9.9 GB | measured |
| Warm decode | **1.42–1.58 tok/s** | **[MATRIX-MEASURED]**, `c/bench-m5max/k6-matrix-20260714-090059/` (evidence dir not included in this repository's staged subset — see §13) |
| Prefill TTFT, pre- → post-kernel | 821–1,523s → 130.4–168.3s (**6.3–9.1×**) | **[MEASURED, weaker class]** — shipped default, but this specific number has **never been run through iliria's own house confirmation matrix**; the project's own theory database calls its status "Serve-ABBA-pending," not confirmed, despite already being default-on (`DEFENSIBLE-ASSETS.md` gap 1) |
| Quality baseline (int4 vs published FP) | 62.5% mean acc_norm (HellaSwag/ARC/MMLU) | n=40/task — flagged in-repo as **too small** to distinguish a real effect from noise beyond the roughest sanity check; the planned n=100 rerun has not produced a committed result (`DEFENSIBLE-ASSETS.md` gap 3) |

"Unmodified" here means what it says and nothing more: past its shipped int4
quantization, the weights are never pruned, merged, fine-tuned, or
router-recalibrated (verified by absence — there is no training loop
anywhere in this codebase, `DEFENSIBLE-ASSETS.md` §e). It does not mean the
int4 conversion step itself is lossless (quantization is a lossy step by
construction — see "Quality baseline" in the table above for what it
actually costs), and it does not mean "fast." At 1.5 tok/s, this is a
744B-parameter model answering correctly on a laptop, not a production
serving target — the honest framing this whole account tries to hold to.

---

## 4. A logical-bandwidth normalization against flash-moe — a diagnostic, not a parity claim

The closest publicly-known large-MoE disk-streaming competitor is
`Anemll/flash-moe` (C+Metal, **not** MLX — a fork of `danveloper/flash-moe`),
which runs Qwen3.5-397B (Q3 quantization, **top-4** experts at its
benchmarked config) at 12.9 tok/s
(an internal cross-engine competitive-survey pass, 2026-07-19; not part of
this release — see §13). Read naively, 12.9 tok/s vs. iliria's 1.42–1.58
tok/s looks like an
order-of-magnitude engine-quality gap — **and the raw gap is real; nothing
below erases it.** Normalizing for bytes moved per token adds a narrower,
diagnostic point on top of that gap, not a rebuttal of it — the arithmetic
is simple enough to show in full rather than assert:

```
gross_active_expert_bytes_per_second = tok/s × GB/token
```

| system | model | tok/s | GB/token (routed-expert bytes/token) | derived GB/s |
|---|---|---:|---:|---:|
| iliria | GLM-5.2, 744B int4, top-8-of-256 | 1.42–1.58 | 11.3 | **~16.0–17.8** |
| flash-moe | Qwen3.5-397B, Q3, top-4 (vs. the upstream reference config's top-10) | 12.9 | 1.31 | **~16.9** |

**Read this table narrowly.** Both systems land at roughly the same **~17
logical routed-expert GB/s** at their respective chosen configs — that is a
diagnostic normalization, **not** a performance or quality comparison, and
it does not mean the two systems are equivalent. Stated in the same
breath, because it must be: the raw, un-normalized gap is **1.5 vs. 12.9
tok/s**, roughly 8–9× and entirely real. What the normalization shows is
that *part* of that gap traces to iliria moving an ~8.6× larger per-token
payload (11.3 vs. 1.31 GB/token). It does **not** show the gap is
"entirely" model size: flash-moe's benchmarked config runs its **top-4**
experts, versus the **top-10** experts of the upstream Qwen3.5-397B
reference configuration, and at **Q3** quantization versus iliria's
**int4** — so routing width and quantization format also differ between
the two measurements, not just model size, and neither system's output
quality at its benchmarked config has been cross-measured here. "Gross
active-expert GB/s" is a logical routed-volume figure, not a
physical-bandwidth measurement, and should not be read as one.

The GB/token input on iliria's side is independently cross-validated, not
just asserted: total routed-expert bytes derived from architecture (experts
× per-expert size × layers) predicts a 362.3 GB container against an
actual on-disk measurement reported as "359 GB" — but see §3's footnote on
that figure's likely GB-vs-GiB unit mismatch, which this cross-check
inherits and has not independently resolved; treat the "~1% match" this
would otherwise imply as **provisional pending unit reconciliation**, not
confirmed. This account has **not** independently re-derived flash-moe's 1.31
GB/token figure beyond the source cited above; it is reported, not
re-verified from flash-moe's own repository.

One correction worth stating plainly, because an earlier internal read of
the data got it wrong and the record should say so: a "12.9 tok/s MLX
slot-bank" figure that circulated internally conflated two different
repositories — `anemll-flash-mlx` (an MLX-based slot-bank implementation,
35B-only) and `flash-moe` (the C/Metal implementation actually benchmarked
at 397B, the source of the real 12.9 tok/s number). The number above is from
the correct repository.

**What iliria leads on, per the same competitive survey — stated as
bounded claims, not a scoreboard:** scale (744B is the largest of the
compared systems, ~1.9× the next-largest), a cross-restart KV-resume
mechanism now backed by iliria's own certified 6.8× same-prompt ABBA
result (§7) — a harder, cross-restart claim than a separately-measured,
same-process-reuse figure on a sibling engine, and not directly comparable
in magnitude to it — a causal delay-injection proof of I/O-boundedness
(§5), and a quality gate that killed a real bad result (§8). No competitor
surveyed (`flash-moe`,
`anemll-flash-mlx`, `ssd-moe`/`deepseek-v4-flash-mlx`, `SharpAI/SwiftLM`)
has published any of these four; the two most-starred of that set ship
with no license file at all (same source, §2c).

---

## 5. A causal proof, stated the way the data actually supports it

It is easy to *assert* a disk-streaming engine is I/O-bound. iliria's team
instead ran a pre-registered, randomized experiment designed to make that
claim falsifiable — and it survived.

**Design** (registered *before* the measurement infrastructure existed:
the stage-2 v4 registration — not included in this
repository's staged subset, see §13; built via commits `e876276`
→ `30a1cc8` → `087214a` → `bee19b4` → `5a70329`; run against the frozen,
hash-verified binary `validation/epoch-20260717-05/`, `glm` sha256
`29a96d5e...`): a byte-identical restored KV state (via snapshot, so every
treatment arm starts from *exactly* the same point — this is what makes the
comparison paired rather than noisy) plus a synthetic completion-delay
injected into the expert-read path at a fixed 8K-context/warm-cache/capacity
condition. A manipulation check confirmed the injected delay actually
landed and scaled monotonically with dose (p50 read-completion time rose
2.23ms → 7.24ms → 14.73ms → 22.21ms across four dose levels — exactly the
shape a real, working manipulation should produce). The headline comparison
is the paired high-dose-vs-zero-dose (labeled "16000-vs-0" in the source —
**16000 here denotes 16 ms of injected delay per expert read**, not the
literal unit the label suggests) decode-latency difference across **5
paired blocks**.

**Result:** all 5 paired blocks completed, 0 failures, argmax output
identical to the zero-delay control in every case (the delay changes
*timing*, never *correctness* — a parity check, not an assumption). Paired
decode-latency differences: [354, 369, 370, 368, 365]s — **mean +365.2s
(95% CI [359, 369]s), median +368s**. The CI above is a bootstrap interval
on the **mean**; it is attached to the mean here specifically because an
earlier internal framing paired it with the median instead, which is not
the statistic the CI was computed for. Both the mean and the median sit
well above zero on only 5 paired blocks — a real, directionally clear
result, but a small-n one, not yet a large-sample confirmatory claim.

**The licensed conclusion, quoted verbatim because the precision matters
more than a punchier paraphrase would:**

> "Added expert-read completion delay causally increases decode token
> latency under the registered 8K/warm/32+256 treatment. NOT bandwidth, NOT
> compute-floor intercept, NOT compression payoff (proves slowing reads
> slows decode; faster→faster remains extrapolation)."

That last clause is the discipline worth keeping in any retelling: this
experiment proves that **adding** read latency **causes** slower decode. It
does **not**, by itself, prove that **removing** read latency would produce
a symmetric speedup — that direction is a reasonable extrapolation, not
something this experiment measured. Any future claim of the form "cutting
read latency by X will buy Y tok/s" should be treated as a prediction to
test, not a corollary of this result.

---

## 6. Cache policy, precisely: three numbers that are not the same number

This is the section most worth getting exactly right, because an earlier
internal framing blurred it and the correction is itself informative.

At iliria's real production capacity — **2,584 pinned experts + 34 LRU
slots/layer**, roughly a 97 GB / 5,134-slot budget
(`docs/frontier/iliria-expert-pruning.md` §2.2; consistent with the
capacity used in the SpecMD A/B below) — three genuinely different
quantities exist, measured across **3 separate GLM-5.2 route-trace corpora
(baseline/coding/shotgun), ~251.6K requests combined** — not a single
~83.8K-request corpus. An earlier internal framing described this as one
~83,850-request corpus (cited as 83,848 requests in the SpecMD A/B and
83,853 in an earlier survey pass); that framing treated what is actually
one trace's approximate size as if it were the combined total across all
three, understating the true combined corpus size roughly threefold:

| tier | what it is | hit rate | shipped? |
|---|---|---:|---|
| **Plain per-layer LRU** | today's actual default | **69.5–70.4%** (70.40 / 70.04 / 69.47% on baseline/coding/shotgun traces) | **Yes — default** |
| **PILOT K6** | opt-in router-lookahead prefetch/admission policy | **77.8–79.5%** | **No — reverted** (see below) |
| **Belady oracle** | offline-optimal ceiling, same capacity, same traces | **79.4–80.2%** (80.18 / 79.99 / 79.42%) | n/a — theoretical ceiling only |

**The correction, stated plainly:** an earlier internal framing described
"77.8–79.5% at the Belady ceiling" as if that were the *shipped* hit rate.
It was not — that figure belongs to PILOT K6, an opt-in feature that is
explicitly **not** the default. Plain LRU, what's actually running today,
sits roughly 9–10 points below the Belady ceiling, not on it
(an internal cross-engine competitive-survey pass, 2026-07-19, correcting
the prior framing explicitly — see §13). This matters because it
changes what "the cache policy frontier is closed" actually means: it does
not mean today's shipped hit rate is unimprovable — PILOT K6 shows a real
+13-hit-rate-point lever exists — it means **no policy tried so far converts
that hit-rate lever into a wall-clock win**, which is a narrower and more
defensible claim.

### 6a. Honest negative #1, narrowed: identical to LRU only in a demand-only replay

Apple's SpecMD (arXiv:2602.03921, ICML 2026, Hoang/Jaiswal/Samragh/Cho)
proposes a "Least-Stale" eviction policy, reporting >88% hit rate / 34.7%
TTFT improvement on OLMoE with a 0.6 GB cache. iliria's team A/B'd
Least-Stale directly against plain per-layer LRU on iliria's own real
traces at real production capacity: **Least-Stale is bit-for-bit identical
to LRU** — 70.40 / 70.04 / 69.47%, to the integer, on all three traces
(same source, §2a). The mechanism reason is architectural, not a measurement
artifact: iliria's `ecache` is already partitioned per layer
(`c/glm.c:162`), so the shared-pool "collision miss" that Least-Stale is
designed to fix cannot occur in this system.

**The scope this result actually covers, stated precisely because an
earlier internal pass overstated it:** the A/B above — including a
shared-pool stress test built to probe the collision-miss condition — is
**demand-only, in-order, and reactive**. It does not exercise the **async,
out-of-order prefetch** regime Least-Stale was actually designed for, nor a
genuinely shared cache under that regime. Under a deeper-lookahead
prefetch condition, Least-Stale has been shown to **win by more than 2
percentage points** — the opposite direction of the demand-only result
above. The honest claim is therefore **"identical to LRU in a demand-only
replay; this does not test async prefetch or a shared cache"** — not
"proven equivalent," and not a general verdict on Least-Stale's mechanism.

This is not a refutation of SpecMD's own claims — SpecMD's regime (1–5%
cache fraction on a much smaller model, OLMoE) is genuinely different from
iliria's ~27% resident coverage on a 744B model, and the demand-only null
result says exactly that: **a regime mismatch under the tested condition,
not a quality gap — and not a closed question under the untested async
condition.** It is the fourth independent eviction-policy null iliria has
found on this system under demand-only replay (alongside PILOT-veto, SNRD,
and GDS-style approaches) — evidence the cache-policy frontier is closed
*for this specific replay regime*, not necessarily for async prefetch,
which the deeper-lookahead result above says remains genuinely open.

### 6b. Honest negative #2: PILOT K6 wins hit rate, loses wall-clock

PILOT K6's own confirmation-matrix result, already the house standard for
"don't trust a single-prompt A/B" (`README.md` § "Settled verdicts"):
**−6.96% median paired throughput**, despite **+13 hit-rate points** over
plain LRU. Attention contention (+4.8–6.3s) and a layer-ordering barrier
outweigh the disk-time saved. Decision: **reverted to opt-in**, not shipped
as default. The hit-rate prediction itself was correct and landed at the
Belady ceiling — hit rate was never the problem; wall-clock time was the
actual objective, and this is the result of measuring the right thing
instead of stopping at the metric that looked good.

---

## 7. Cross-restart memory: iliria's own certified 6.8× result, and why it isn't the same claim as a sibling engine's larger one

iliria persists KV cache to disk (`.ili_kv`) and reloads it across process
restarts (`kv_disk_save`/`kv_disk_load` in `c/glm.c`, file-based, per-slot,
version-checked). This section previously cited **142× TTFT speedup**,
3.07s vs. 436.65s cold, as an **N=1** observation with two known flaws: the
cold and resume arms used different prompts, and both sides' TTFT excluded
process/model/KV-load time. That specific number was retired as uncitable,
not upgraded.

**iliria now has its own certified replacement, built and run directly
against its own `.ili_kv` mechanism** (`c/bench-m5max/ili-kv-resume-abba/`,
full run 2026-07-20): a same-prompt, randomized ABBA/BAAB cold-vs-resume
harness that fixes all three flaws that retired the old number — one fixed,
sha256-pinned prompt submitted byte-identical to both arms every trial; a
resume-hit proved from the engine's own `[API]`/`[KV]` telemetry each
trial, never assumed from the schedule; and one explicit `TTFT_DEFINITION`,
carried verbatim into the JSON output, identical for both arms.

| | median TTFT | 95% CI |
|---|---:|---:|
| cold | 153.92s | [145.99, 163.49] |
| resume | 22.56s | [19.18, 30.97] |
| **speedup (cold/resume)** | **6.8×** | **[5.0×, 8.0×]** (percentile bootstrap) |

n=6 trials/arm (3 randomized ABBA/BAAB blocks, order CRRCRCCRRCCR), all
12/12 trials showed a resume-hit proof matching the arm's expectation (cold:
`prefix=0, prefill=414`; resume: `prefix=414, prefill=0`, read directly off
the engine's own counters, never assumed), and cold/resume completions were
byte-identical (greedy decode, temperature 0) across every trial. Every
trial ran contended (an unrelated pair of servers plus a router were live on
the same GPU throughout, by design — real-workload conditions, not a
clean-machine benchmark) — the cold/resume **ratio** (both arms interleaved
by ABBA through the same contention regime) is the headline number for
exactly that reason.

**Read this number precisely, because a superficially similar, much larger
multiplier exists elsewhere and the two should not be conflated.** A
separate, sibling project's engine (same machine, different codebase,
different mechanism) has its own certified **147.8× [137.1, 157.8]**
same-prompt ABBA result for its own KV-cache mechanism. That is a
**different, easier claim on a different engine**: one server process held
constant across every trial (no restart), with process spawn and one-time
model load explicitly excluded from the timed window. iliria's own claim
is specifically **cross-restart, disk-persisted**: every trial in both arms
restarts the iliria engine process from scratch, and process spawn plus
the one-time 744B model-weight load are deliberately left *inside*
iliria's timed window — symmetric costs paid in both arms, because
excluding them would silently reintroduce the exact TTFT-accounting
ambiguity that retired the old 142× number. Model load is a substantial
fixed cost paid by both of iliria's arms, so a smaller ratio here is the
expected, correctly-documented result of measuring the harder, more
complete restart claim iliria actually makes — not a weaker result, and
not the same mechanism measured twice. What resume actually eliminates in
iliria's own number is the ~131s of contended prefill (414 prompt tokens
down to 0 re-prefilled); the remaining ~22.6s is dominated by process spawn
and model load, which no KV cache can remove. Citing the sibling project's
147.8× for iliria's own cross-restart claim (or vice versa) would
misattribute one engine's result to the other — the two numbers are
individually honest and are not substitutes for each other. (Full
methodology, the resume-hit proof design, and a side-by-side comparison
table: `c/bench-m5max/ili-kv-resume-abba/REPORT.md` §5.)

Context for the general "KV-cache resume buys a large TTFT win" claim, with
the denominators kept separate rather than merged into one comparison: an
independent adversarial literature-review pass (2026-07-14, extensive parallel analysis,
22 sources, 25 claims adversarially verified) found exactly
one surviving verified session-level agentic-serving number in the public
literature: Ollama's ~23% latency reduction / ~80%+ hit rate from
same-process prefix reuse on a fully-resident dense model (Qwen-2.5-Coder,
M2 Ultra, no restart, no disk streaming at all) — the same easier,
same-process-reuse regime the 147.8× figure above occupies, not iliria's
own harder claim. In iliria's actual regime (cross-restart, disk-persisted
KV), the surviving literature had, as of that pass, zero verified claims —
the closest candidate, a "136×" figure from arXiv:2603.04428, **failed
adversarial verification** (1–2 vote). iliria's own cross-restart
mechanism is real, shipped, and — as of this writeup — has its own certified
number for the first time: **6.8×**, not borrowed from anywhere else.

---

## 8. When the gates catch you: Gate 3

The discipline behind these numbers is only worth crediting if it has
actually caught something. It has, once, concretely: a **separate**,
lossy-compression research thread (a tiered int4/int2 mixed-precision
container — not the lossless Mode-1.5 codec discussed in §9, a different
technique) had its numeric ship thresholds fixed **before the container was
built** (the compression-gate threshold record — not included in
this repository's staged subset, see §13). Gate 3 **FAILED**
(commit `bb7f0e5`): case D showed activation error 3–6× over its registered
threshold, caused by the MLP amplifying error from a fitted int2 quantizer —
a mechanism this project's standing adversarial reviewer had predicted
in advance rather than diagnosed after the fact. The container was
**deleted**, with user authorization, rather than shipped
(`DEFENSIBLE-ASSETS.md` §b2; session log, 2026-07-18, "GATE 3 FAIL: case D
KILLED"). This external review was not a one-time check — it recurs across this
project's commit history (`e72040b`, `f370851`, `0babfca`, `7608967`) as a
standing role, not a single audit.

**Scope, stated precisely so this result isn't over-read later:** Gate 3
killed **one specific int2-tier design** — on **5 layers, 1 capture** — not
int2 quantization in general. "Int2 is proven unusable" would be an
overclaim this result does not support; the narrower, defensible claim is
that this particular fitted-int2 quantizer design failed its own
pre-registered threshold on the evidence gathered so far.

---

## 9. What is not yet true — placeholders, marked loudly on purpose

### 9a. MEASURED — Mode-1.5 is correct and lossless end-to-end, and measured slower, not faster

Mode-1.5 is a **lossless** row-Huffman compression codec for the routed-expert
container (`c/codec_row_huff.h`, `c/mode15_reader.c/.h`) — a different,
separate effort from the lossy container killed in §8. What is real and
measured today:

- **Bit-exactness**: 895 tensors / 3,057,664 rows byte-identical between the
  Metal decode kernel and an independent CPU cross-check decoder
  (`gate_m15_g1.py`, commit `345c9ef`; see
  `c/bench-m5max/compression-gates/` in this repository for the gate script
  and its full-container run results).
- **Compression ratio**: a trace-weighted encoder estimate of **0.7526**
  (pre-registered), and — more recently — a completed full 164/164-shard
  encode measuring **0.7521** (362 GB → 273 GB, physical container size;
  session log, 2026-07-19 morning; see `c/bench-m5max/mode15-full-encode.log`
  and `mode15-verify-0637.log` in this repository). These two independent
  measurements agree to within 0.05 percentage points.
- **GPU decode bandwidth**: 36.51 GB/s solo, ≥32 GB/s under contention — 2.4×
  over the 13.5 GB/s streaming bar, meaning decode itself is not the
  bottleneck **when run on the GPU path** (see below for why that
  matters).
- **On the "physical vs. logical bytes" point specifically**: this
  project's own falsifier design is explicit that the compression-ratio
  instrumentation must count *compressed (physical)* bytes actually read
  off disk, never the post-decode *expanded (logical)* size — "or F1's
  ratio is definitionally 1.0" (this project's Mode-1.5 falsifier design
  notes, not part of this repository's staged subset — see §13). This is
  exactly the discipline this claim needs and the one place a compression
  number is easiest to accidentally inflate.

**What was pending as of yesterday is measured today, and the measurement is
a regression, not the predicted gain.** As of 2026-07-19, this section
described Mode-1.5's end-to-end decode-speed gain — a predicted **1.132×
speedup**, projecting to roughly 1.81 tok/s — as a pre-registered estimate,
with the four-cell factorial {baseline, compression, prefetch, both} that
would arbitrate it not yet run. That factorial's underlying question — does
the codec's own throughput beat plain int4 end-to-end? — has since been
answered directly: Mode-1.5 was run against the **real 744B/272GB
container** and is **proven correct/lossless end-to-end** (5/5
token-identical vs. the int4 reference — a genuine milestone, the first
full-container proof this codec chain has had) but measured **~1.6–1.7×
SLOWER** than plain int4 (tested cold, `AUTOPIN=0`). The cause is
straightforward, not mysterious: the engine still runs the codec's **CPU**
Huffman-decode path (wired by commit `1ac31b9`), whose own overhead exceeds
the NVMe bytes it saves. This is not in tension with the GPU bandwidth
number above — that number is a standalone probe of the GPU decoder, which
is independently fast enough to reverse the regression, but it is **not yet
wired into the live `expert_load`/`qt_from_disk` decode path**. Wiring that
GPU path in is the real next lever (referred to in session notes as
"step-5"), not another round of the four-cell factorial.

**A second, independent blocker surfaced this same session:** a
reproducible multi-turn **silent-death** bug in `ili serve` with Mode-1.5
active. The fix is in flight, not yet committed as of this writeup. This is
distinct from the fail-loud guards against running `matmul_i2` on
Huffman-compressed bytes on a mode15-unaware build (commits `a331368`,
`977d822`, `0e9e30a` — that gap was closed before this session); the
silent-death bug is a new, separate defect found by actually serving
multi-turn traffic through the now-correctness-proven container.

**Net, stated plainly:** Mode-1.5 today is a correctness milestone, not a
shippable speed win, and it carries an open reliability bug on top of the
speed regression. Current shipping status: `not_built` — true both
yesterday and today, but for a different reason: not "unmeasured," now
"measured slower, plus an open serving bug."

### 9b. PENDING — a sibling project's layer-pruning safety work (the sibling engine, not iliria)

A separate sibling project's (the sibling engine's, not iliria's) layer-pruning speed
result has a safety-evaluation gate that was found miscalibrated and is
being rebuilt pending human ratification, so that work is governance-blocked
and outside this release, with no numbers cited here pending a sound,
ratified gate — iliria's own, separate expert-*count*-pruning research (a
different mechanism, unaffected by any of this) is covered instead in
`docs/frontier/iliria-expert-pruning.md`.

---

## 10. Standing on prior work — credit, not competition

None of this exists in a vacuum, and the honest version of this story says
so explicitly.

- **Apple's "LLM in a flash" (arXiv:2312.11514)** is the foundational
  flash-streaming paper for this entire space. Its results are real and
  validated — on dense, ReLU-sparsified models (OPT-6.7B, sparsified
  Falcon-7B/Phi-2/Persimmon-8B/Llama-2-7B). Its windowing mechanism earns its
  speedup by reusing weight rows across consecutive tokens, which requires
  cross-token activation overlap. GLM-5.2's top-8-of-256 MoE routing has
  measured consecutive-token expert Jaccard similarity of **~0.18** — the
  premise the windowing technique depends on is architecturally close to
  absent here, not because the paper is wrong, but because MoE routing is a
  different regime than the dense models it was validated on (an internal
  flash-streaming literature survey, 2026-07; iliria's own
  independent route-trace analysis reached the identical 0.18 figure by a
  completely different method — a real corroboration, not a coincidence).
- **Apple's SpecMD (arXiv:2602.03921, ICML 2026)** is the MoE-era successor
  in this lineage, and its Least-Stale eviction policy is a real, published
  contribution. §6a above is a regime-mismatch null result on iliria's much
  larger, much-higher-cache-fraction system, not a claim that SpecMD's own
  reported numbers (on OLMoE, at 1–5% cache) are wrong.
- **The community's disk-offload work for MLX** is active and ahead of any
  private effort here in terms of what's already public: `mlx-lm` **PR #1588**
  (opened 2026-07-19, opt-in disk-backed expert offload with an LRU slot
  table) and thread **#1438** (which already independently tested and killed
  both expert-pinning and cross-layer prefetch — measuring a Jaccard of
  0.017 vs. 0.016, matching iliria's own near-memoryless-routing finding
  from a different codebase entirely). Maintainer statements confirm the gap
  is real and known, not overlooked: `awni`, `mlx#615` (2024, "can't mmap
  available to GPU"), `angeloskath`, `mlx#3371` (2026-04, rejecting a naive
  streaming PR on scope grounds) (this project's own MLX gap-analysis notes
  and the competitive-survey pass cited above; not part of this release).
- **`Anemll/flash-moe`** (fork of `danveloper/flash-moe`) is the closest
  existing large-MoE disk-streaming implementation in absolute terms, and
  §4 above is offered as a fair normalized comparison against it, not a
  claim of superiority.

A bounded, three-point evidence comment sharing this project's own
Least-Stale/LRU parity result, a prefetch hit-rate-vs-wall-clock caution,
and the byte-normalized parity figure from §4 was posted to issue #1438
on 2026-07-20 — not as a pitch for anything of this project's own, only as
data that might be useful against that thread's own numbers. What that
thread did with a follow-up, in-engine measurement later that same day is
its own closing note (§14).

---

## 11. A concrete MLX path — proposed, staged, not yet posted

**Verified directly from the MLX source** (the sibling engine/quant-sdpa branch): stock
MLX cannot run a model larger than RAM today. `ParallelFileReader` is
read-based, not mmap-based (`mlx/io/load.cpp:347–363`, `mlx/io/load.h:59–69`);
`mx.load`'s `Load` primitive defers the read but does not bound residency —
a forward pass materializes every weight tensor it touches, with no
eviction (`primitives.h:1299`). No offload, streaming, or eviction machinery
exists anywhere in `mlx/io/`. A ~359–386 GB int4 MoE (see §3's GB/GiB unit
caveat on this figure — the point holds under either reading) on a 128 GB
Mac is simply unloadable in MLX as it stands today (this project's own MLX
gap-analysis notes, not part of this release — see §13).

What iliria has that could fill this gap is an architecture and an evidence
trail, not an MLX code dump — iliria is not built on MLX. The proposal
under consideration (drafted, **not posted anywhere**, pending owner
approval and an ecosystem-verification pass) is narrow and staged on
purpose:

1. A **streaming loader** registering large tensors as disk-backed
   (offset + length into an existing container file) instead of eager
   `Load`, materializing on demand into a bounded residency pool.
2. A **weight-eviction policy** — plain per-layer LRU as the proven
   baseline (§6), with an optional learned-pinning hook as a documented
   opt-in, not a default.
3. An **optional lossless entropy row-decode path**, with an honest caveat
   attached rather than omitted: Mode-1.5's real compression headroom
   depends on iliria's own *coarse* per-tensor quantization-scale format.
   MLX's native *fine-grained* per-group quantization format was
   independently measured to have far less entropy headroom for this
   purpose (~7–8%, versus iliria's ~25%) — this is **not** a drop-in port,
   and any pitch to MLX maintainers should say so up front rather than let
   them discover it.
4. **The falsifier/provenance harness itself** — pre-registered kill-shots,
   the confirmation-matrix standard, epoch-freezing, the discipline that
   caught Gate 3 (§8) — offered as the more durable part of the
   contribution, per this project's own standing second-opinion review's
   framing: the moat is the discipline plus the residency policy plus the
   measured evidence, not any single kernel.

**Staged ask, no rival PR, no code-dump:** (1) open an MLX discussion/issue
laying out the gap, the evidence, and the design; (2) gauge maintainer
signal before investing further; (3) only on a green light, a narrow
reference-extension draft PR plus the parity/falsifier harness — following
the same authorship etiquette already used for an unrelated, smaller,
already-submitted MLX contribution (PR #3026, a fused quantized-KV SDPA
kernel with a small measured payoff — help, don't overwrite).

---

## 12. What this document is not claiming

Stated plainly, because the temptation to overreach on a genuinely good
result is exactly what the discipline in this project is designed to catch:

- **This is not a "world first."** No comprehensive claim of exclusivity is
  made anywhere in this account. Section 10 names real, credited prior art;
  §4 names real, comparable competing systems. Where iliria leads (scale,
  the cross-restart KV-resume mechanism, the causal proof, Gate 3), that is
  stated as a specific, bounded claim against a specific, named comparison
  set — not a general claim that nothing else exists, and, for KV-resume
  specifically, not a claim stronger than what was actually measured:
  iliria's own certified 6.8× cross-restart, disk-persisted result (§7) —
  real and shipped, but a different, harder claim than a separately-measured
  147.8× same-process-cache result on a sibling engine, which this account
  does not borrow for iliria.
- **This is not "Apple failed" at anything.** Every Apple result referenced
  in this account (LLM-in-a-flash, SpecMD) is treated as real, peer-reviewed
  work that targets a different regime than iliria's — dense
  ReLU-sparsified models, or a much smaller MoE cache fraction — not as a
  paper that got something wrong. §6a says this explicitly for SpecMD; §10
  says it explicitly for LLM-in-a-flash.
- **This is not a production serving claim.** 1.5 tok/s is not a serving
  target. This is a systems and measurement story, not a product pitch.

---

## 13. Reproducibility and licensing posture (brief)

Engine: Apache-2.0, fork of `JustVugg/colibri`. Weights: MIT
(`GLM-5.2-FP8` derivative, `zai-org`), no fine-tuning or training loop
anywhere in this codebase — verified by absence, not merely asserted
(`DEFENSIBLE-ASSETS.md` §e). Raw run data mostly lives in a local,
gitignored directory (`c/bench-m5max/`); as of 2026-07-18 the text tier
(logs, gate JSON, campaign log, registration docs, determinism snapshots —
1,454 files / 25.42 MB) is committed in this project's full internal
history, but the binary tier (six KV-snapshot files underlying the causal
proof in §5, ~8.7 GB) remains machine-local, regenerable in roughly 1 hour
per snapshot but not itself committed (`DEFENSIBLE-ASSETS.md` §d, gap 2).

**What's actually in *this* repository, specifically:** this is a public
release, not a mirror of the full internal research history. `c/bench-m5max/`
here ships only three curated evidence categories — the KV-resume ABBA
harness and results (§7), the step0-iokind diagnostic, and the Mode-1.5
certification evidence (§9a: `mode15-*.log`, `compression-gates/`). Several
`c/bench-m5max/*` and internal-findings-log paths cited elsewhere in this
document (registration docs, earlier survey passes, session logs) are part
of the broader internal research history and are **not** included in this
repository — they're cited for provenance and precision, not as an implicit
claim that a reader can open them here. A reader who wants to check a number
backed by an included path should expect to **re-run** the cited command;
for a number backed by a path not included here, treat the citation as
provenance, not a reproducibility guarantee this repository itself makes.

---

## 14. External validation, unsolicited: mlx-lm issue #1438

Section 10 above describes a three-point evidence comment posted to
`ml-explore/mlx-lm` issue #1438 on 2026-07-20. What happened next on that
same thread is worth recording as its own closing note, because it is
independent, adversarial, cross-engine validation of one specific, narrow
claim — not a general endorsement solicited or written by this project.

Over roughly a week, external contributors `mabaeyens`, `doramirdor`, and
`lBroth` had converged on a shared cost model for MoE expert-streaming I/O —
a "coalescing ladder" running from scattered per-tensor reads, through
whole-expert reads, to a fully coalesced per-expert container read, with a
measured residual gain, `g`, at each rung. Closing the loop on the
GLM-class case the thread had been extrapolating toward, this project's own
account (posted under this project's GitHub handle, `PhilipJohnBasile`)
supplied the missing in-engine measurement: iliria's container format
already stores each expert's three projections contiguously, so the fetch
path has always been **one ~18.9 MB pread per expert-load** — a real
744B-model, real-NVMe measurement (44,316 weight preads, 829.8 GB, 6.84
ms/op, 13.4–13.6 GB/s; 131,892 scale preads, 122 µs/op) placing iliria's
residual coalescing opportunity at `g = 1.053`, an implied **~1.3%
wall-clock gain** — below this project's own 5% promotion gate, and so,
correctly, not worth taking.

`doramirdor`'s reply named what that number meant for the ladder itself:

> "One ~18.9 MB pread per expert-load is the endpoint of the coalescing
> ladder, shipped as a format decision — which is the strongest endorsement
> of the lever this thread could ask for."
>
> — [`doramirdor`, `ml-explore/mlx-lm` issue #1438](https://github.com/ml-explore/mlx-lm/issues/1438#issuecomment-5028289242), 2026-07-20

Read narrowly, because that is the honest scope of what several independent
people on several different engines actually measured: this is external
confirmation of **one specific, low-level design choice** — iliria's
expert container already reads at the coalesced end of the I/O-granularity
spectrum the thread spent a week characterizing — not a general benchmark of
iliria against anything else in this account, and not a claim any of the
thread's other participants would necessarily extend further. It is,
however, a genuinely unsolicited convergence: nobody on that thread was
pitching iliria, the comparison arose because the thread's own cost model
needed a fourth data point, and the model and the measurement agreed.

---

## Appendix: source map

Primary sources for this account (read in full):

- `docs/DEFENSIBLE-ASSETS.md`
- Two internal cross-project findings logs (2026-07-19, 2026-07-20) — not
  part of this release — supplied the competitive-survey and layer-skip
  session data cited above; they are provenance citations, not files this
  repository ships (see §13).

Supporting sources read directly to source specific numbers used above:

- `README.md` (headline architecture/performance figures)
- `docs/frontier/iliria-expert-pruning.md` (capacity/hit-rate denominators,
  iliria's own separate expert-pruning track)
- `docs/PERFORMANCE_THEORY.md` (cross-source conflict discipline,
  evidence-class taxonomy)
- `c/bench-m5max/ili-kv-resume-abba/REPORT.md` (§7's certified 6.8×
  iliria-side cross-restart KV-resume ABBA)
- `c/bench-m5max/step0-iokind-diag/RESULTS.md` (§ referenced I/O-kind
  diagnostic)
- `c/bench-m5max/compression-gates/` and `c/bench-m5max/mode15-*.log`
  (§9a's Mode-1.5 bit-exactness and full-container certification evidence)
- GitHub, `ml-explore/mlx-lm` issue #1438 (§14's external-validation closing
  note; comments by `doramirdor` and `PhilipJohnBasile`, 2026-07-20)

Some sources cited in an earlier internal revision of this document (an MLX
gap-analysis writeup, additional registration/design docs under
`c/bench-m5max/`, and further internal findings logs) are themselves not
part of this release; §8, §9a, §10, §11, and §13 above are written so a
reader without access to them can still follow the argument, with the
provenance stated rather than hidden.
