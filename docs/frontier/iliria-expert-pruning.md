# Iliria expert pruning — utilization verdict, prize estimate, prune+heal plan

**Scope of this pass: banked-data analysis + research + web only.** No model load, no GPU,
no engine run — the sibling engine layer-pruning sweep has the GPU. Every number below is either (a)
recomputed from `.fa_usage` histograms and `FAROUTE1` route traces already sitting in
`c/bench-m5max/`, (b) lifted from this project's own existing verdicts
(the offline math-verdict record, `docs/PERFORMANCE_THEORY.md`,
the model-surgery frontier notes §2.3), or (c) fresh web/HF
recon on the community GLM-5.2 REAP checkpoints. Where a number required an actual pruned
checkpoint or new engine instrumentation, that is stated explicitly and deferred, not guessed.

**Context for this document's existence:** iliria's charter — "the model itself is never
modified... no fine-tuning, expert merging, or router recalibration" (`README.md`) — remains the
standing guarantee for everything iliria ships. This document is purely a *research*
exploration of whether structured expert pruning could ever relax it; it changes nothing shipped.

---

## TL;DR

1. **Utilization is SKEWED, not uniform — but it is a moderate, mid-band skew, not a
   long-tail of dead weight.** Gini ≈ 0.50 pooled across all 19,200 (layer, expert) slots;
   literal never-routed experts are ≤0.3% of slots even at 12–15M accumulated selections.
   This independently reproduces — and, on a 5–180x larger sample, *sharpens downward* — this
   project's own prior verdict (the offline math-verdict record Study 2: "~2/256 dead
   experts... diffuse, not Pareto-heavy"), including the same standout layer (L19, highest
   concentration) and the same early-layer diffuseness (L3–10).
2. **The "free" prunable fraction is small (~1–10% depending on threshold); the
   community-validated aggressive fraction is 31–37.5%; this project's own prior precedent
   on a different GLM-5.2 build reached 77% with domain-targeted healing** (confounded causal
   story, not clean proof).
3. **The intuitive framing of the prize — "fewer expert bytes streamed per token" — is
   mechanistically wrong, and this project's own survey already corrected it**
   (the model-surgery frontier notes §2.3(c), citing NAEE arXiv:2402.14800): every token still
   reads exactly k=8 experts/layer regardless of pool size. Pruning is a **capacity /
   cache-hit-rate lever, not a direct per-token-bytes lever**, for a cold miss.
4. **New measurement (this pass): out-of-sample route-trace replay through iliria's own
   calibrated cache simulator** shows a real but modest indirect effect: **+3.6 to +8.2
   percentage points of cache hit rate** at fixed ~97 GB RAM budget, for 25–37.5% prune
   ratios. First-principles disk-time arithmetic anchored to this project's own twice-measured
   13.3–13.5 GB/s raw SSD bandwidth translates this to roughly **+3.6% to +12.1% decode tok/s**
   (≈1.5 → 1.55–1.79 tok/s) — real, but nowhere near a "prune X% → X% faster" free lunch.
5. **GO, conditionally** — on the cheapest possible next step: acquire an *already-existing*
   community GLM-5.2 REAP checkpoint and measure real hit-rate/tok/s/quality on iliria's own
   engine, before building any in-house prune pipeline. This is real work (a model load), so
   it is not done in this pass; it is designed and registered below.

---

## 0. The make-or-break question, answered directly

> "Is GLM-5.2's expert utilization SKEWED (prunable dead weight) or UNIFORM (nothing to
> prune)?"

**Neither extreme. It is a moderate, structured skew** — real enough to prune into, not
extreme enough to prune for free. This is the harder middle case anticipated here, and
it is why the rest of this document is a quantified risk/reward analysis rather than a
one-line kill or a green light.

### Method

`c/analyze-m5max-hotset.py` already exists in-repo for exactly this question (per-layer static
top-N pin coverage over `.fa_usage` cumulative histograms — it deliberately keeps the layer
dimension, since expert ID 5 in layer 3 is a different tensor than expert ID 5 in layer 40).
I ran it, plus a small ad hoc tail/Gini script, against the **largest banked snapshots**
found in `c/bench-m5max/`:

- `rope-variance-233636/fa_usage.snapshot` — **12,201,600 selections**, 75 layers × 256
  experts, general/mixed workload.
- `rope-variance-233636/fa_usage.coding.snapshot` — **15,079,800 selections**, same shape,
  coding-profile workload.

These are 5.5x and ~180x larger than the two histograms the offline math-verdict record
Study 2 used (2.2M general / 83K coding), which matters for the "how many are truly dead"
question below.

### Results — static per-layer coverage (general profile)

| N/layer (of 256) | % of experts | routing-mass coverage |
|---:|---:|---:|
| 1 | 0.4% | 3.47% |
| 8 | 3.1% | 17.36% |
| 32 | 12.5% | 41.13% |
| 64 | 25% | 59.91% |
| 128 | 50% | 82.58% |
| 168 (community 34% ratio) | 65.6% | ~91.2% (in-sample) |
| 192 (in-house 25% "knee") | 75% | 94.97% |
| 256 | 100% | 100.00% |

The coding-profile histogram is **almost identical** (top-8 = 16.97% vs 17.36%; top-128 =
82.10% vs 82.58%) — coding prompts do not meaningfully concentrate the distribution further
at this resolution, consistent with the existing `hotset_training_coverage_shift_pct` finding
(`docs/PERFORMANCE_THEORY.md`: more hotset training *broadened* coverage 69→95.8%, it did not
concentrate it — an internal synthesis note (not staged here) #47).

### Results — tail/skew quantification (new this pass)

| Metric | General | Coding |
|---|---:|---:|
| Bottom decile (25/layer, 9.8% of experts) — mass share | **1.14%** | 1.12% |
| Bottom quartile (64/layer) — mass share | 5.03% | 5.15% |
| Bottom half (128/layer) — mass share | 17.42% | 17.90% |
| Literal zero-count (layer,expert) slots | 13/19,200 (0.068%) | 61/19,200 (0.318%) |
| Gini coefficient, pooled | **0.4987** | 0.4948 |
| Per-layer Gini: min / median / max | 0.236 (L4) / 0.490 / **0.683 (L19)** | 0.239 (L7) / 0.456 / **0.802 (L19)** |

**Cross-validation, unprompted and independent:** this pass's L19-max-concentration and
L3–10-most-diffuse findings **exactly reproduce** Study 2's own independently-derived result
("L12–22 concentrates hard... Gini 0.76 at L19... earliest layers L3–10 are the MOST diffuse")
on a completely different, much larger snapshot from a different session. Two independent
measurements agreeing on which *specific layer* is the outlier is a much stronger signal than
either alone.

**The sample-size correction, stated honestly:** Study 2's "~2/256 (~0.8%) dead" was measured
on histograms 5.5–180x smaller than this pass's. A smaller sample makes a rare-but-real expert
look literally dead by chance; this pass's zero-count (0.068–0.318% of slots) is **lower**
than Study 2's implied ~0.8%, exactly the direction more data should move a finite-sample
"dead" estimate. This *sharpens* Study 2's conclusion rather than contradicting it: there is
even less literal free dead weight than previously estimated, not more.

### Verdict

- **Not uniform**: Gini ≈ 0.5 is a real, moderate concentration (for reference, a Gini of 0
  is perfectly flat; Zipf-like word-frequency distributions run 0.7–0.9). Top-8-of-256/layer
  (3% of experts) carries 5.4x their "fair share" of mass (17.4% vs 3.1% uniform expectation).
- **Not a long tail of dead weight either**: essentially no expert is *literally* never
  routed at this sample depth (≤0.3% of slots), and even a generous "bottom decile" cut
  removes only ~1.1% of routing mass while touching ~9.8% of experts — real, but small.
  **This is exactly why the existing static-pinning strategy was already killed on this
  container** (internal findings (not staged); `README.md`'s own honest-numbers table lists
  "static pinning / hotset training: flat/diffuse → dead") — a *purely frequency-based* cut
  has very little obviously-free fat.
- **This is why REAP-style approaches outperform naive frequency cuts on GLM-5.2-class
  models**: REAP's actual saliency (`gate_value × ‖expert_output‖₂`, decoupling frequency from
  *impact*) and redundancy-aware merging can find prunable capacity that a raw popularity
  histogram cannot see — a rarely-routed expert can still be low-impact-when-routed (safe to
  cut) or a narrow, high-impact specialist (unsafe to cut), and **frequency alone cannot tell
  these apart**. This project has already measured two frequency/reconstruction-error-based
  saliency proxies FLAT on this exact container (internal findings (not staged) routing frequency,
  #56 quantization reconstruction error) — a real reason for caution, not dismissal, about
  whether a third proxy (REAP's) will discriminate any better. See §4.3.

---

## 1. Community-pruned GLM-5.2 checkpoints — recon (web/HF, verified 2026-07-19)

Four real, released checkpoints exist, all using REAP saliency
(`gate_value × ‖expert_output‖₂`, verified directly from two of the four model cards below).
This is not a hypothetical technique for this model — it has already shipped multiple times.

| Checkpoint | Kept/pruned | Saliency method | Recovery | Reported quality signal |
|---|---|---|---|---|
| `0xSero/GLM-5.2-REAP-504B-GGUF` | 168/256 (34% pruned) | gate × ‖output‖ | gate-only Router-KD, 0.016% params | Loop rate 7.2% vs unpruned-teacher 3.6% (~2x more looping); card flags "not yet imatrix-calibrated" |
| `brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172` | 172/256 (33%) | **real** activation capture — ran actual `GlmMoeDsaNaiveMoe` modules over **7,368,253 active tokens**, 75 layers (not a proxy) | 12,228-sample rebalanced calibration (general/legal/code/reasoning) | Baseline REAP: wrong capital of Kentucky (said Lexington) and Texas, empty/looping on Marbury v. Madison — **fixed** by broadening calibration, not by changing the prune ratio |
| `pipenetwork/GLM-5.2-REAP37-MLX-4bit` | 160/256 (37.5%) | gate × ‖output‖ | none stated; calibration = 192 seqs × 1024 tok (≈196K tokens — far smaller than brandonmusic's 7.37M) | PPL +7.3% vs full model (1.553 vs 1.447) — card calls this "modest"; MLX 4-bit, 265 GB |
| `0xSero/GLM-5.2-REAP-NU176-526B` | 176/256 (31.25%) | gate × ‖output‖ | — | (not independently re-verified beyond existence; consistent family with the above) |

### Real-user reports (Level1Techs forum, direct quotes, fetched this pass)

- *"if all the coding related experts are still in there it would make sense that it still
  performs on coding tasks"* — calibration coverage of the target domain is what matters, not
  the raw ratio.
- **"reap lobotomizes the model for tasks other than coding"** — a direct, unprompted
  real-user statement of exactly the narrow-deployment tradeoff §5 below examines.
- *"REAP50 and the model is pretty much useless, it can't hold a chain of thought"* — 50% is
  a real cliff, consistent with EASY-EP's own measured 75%-reduction cliff on the closest
  architectural analog (DeepSeek-R1/V3).
- *"If you do 5–15% you may get away with it"* — one experienced user's conservative
  recommendation, notably **more conservative** than every released checkpoint above (31–37.5%)
  and far more conservative than this project's own 77% precedent.
- *"REAP legitimately just removes the information altogether"* (vs. quantization's fidelity
  reduction) — a sharp, correct mechanistic framing: this is deletion, not degradation, and
  the "Expert Strikes Back" paper's finding of narrow *specialist* experts (chemistry-suffix
  vocabulary, legal/patent continuations — **on an actual GLM checkpoint, GLM-4.7-Flash**)
  is the concrete mechanism behind why that distinction matters.

### The recurring, cross-source pattern

Every source in this section and in the model-surgery frontier notes §2.3(b) converges on the same
shape: **short-form/coding benchmarks hold up near-flat; world-knowledge, long
chain-of-thought, and hard multi-step reasoning are what breaks**, and **calibration-set
breadth — not the prune ratio per se — is the load-bearing variable** determining how badly.
This is the single most consistent, most-corroborated finding across independent
sources (external papers, community checkpoints, forum users, and this project's own 77%-prune
history) in the entire investigation.

---

## 2. The prize, corrected and quantified

### 2.1 The mechanism correction (do not skip this)

The framing under study here states: *"expert pruning removes
experts = fewer expert bytes streamed per token... directly faster decode."* **This is
mechanistically incorrect, and this project's own survey already found and corrected the
same error** (the model-surgery frontier notes §2.3(c), citing NAEE, arXiv:2402.14800 directly):
*"each token is still processed by k selected experts, without a reduction in runtime
FLOPs."* Extended to bytes: **per-token routed-expert disk bytes on a cold miss =
expert_size × k × num_layers — an expression with no pool-size term at all.** A genuine miss
costs exactly the same bytes whether the candidate pool is 256 or 160 experts, because the
same k=8 experts' worth of bytes get read either way. Pruning does not touch this term.

**The only real mechanism is indirect**: shrinking the pool from N to N′ raises effective
cache coverage at a *fixed RAM budget* C (C/N′ > C/N), which should raise hit rate and lower
the *blended average* bytes/token (mix of cheap hits and expensive misses) — mediated entirely
through cache economics, not a direct per-read cut. The survey's own words: **"no source found
anywhere quantifies this specific pruning-to-hit-rate bridge, for any streamed MoE system,
iliria's own included."** That gap is what this section fills.

### 2.2 New measurement: out-of-sample route-trace replay

Iliria already has a calibrated LRU+pin cache simulator (`c/tools/simulate_m5max_cache.py`,
validated within ~1 point of measured hit rate in the offline math-verdict record). It has no
built-in notion of "pool size" — its cache model just tracks whatever experts actually appear
in a route trace, against a fixed RAM-budget slot count. I reused its own tested functions
(imported, not reimplemented) to simulate a smaller candidate pool honestly:

**Method** (train/test split, out-of-sample — the same discipline a real calibration/eval
split would use):
1. **Rank** experts per layer using the large, independent `.fa_usage`/`.fa_usage.coding`
   histograms (12.2M / 15.1M selections, §0) — playing the role of a REAP calibration set
   (a **frequency proxy**, explicitly not REAP's real gate×output-norm criterion — see §4.3
   for why that criterion cannot be computed from any banked data).
2. **Replay** a *different, held-out* session's route traces (`routes-baseline.bin`,
   `routes-coding.bin`, `routes-shotgun.bin` — 96,000 selections each, `overnight-20260714`
   sessions) filtered to only the surviving top-N′ experts per layer, through the **same**
   ~97.1 GB / 5,134-slot RAM budget (read directly from the trace's own v2 metadata: 2,584
   pin + 34/layer LRU × 75 layers) and the same pin+LRU derivation `simulate_m5max_cache.py`'s
   own `main()` uses.
3. Compare hit rate on the survivor-only sub-stream vs. the unfiltered baseline, at the
   identical RAM budget.

**Reproduction check (baseline, N=256):** this run reproduced **64.4% / 65.5% / 63.9%**
hit rate on the three held-out traces — squarely inside the 64.1–66.9% simulated / 64.7–66.1%
measured range Study 1/2/5 already established. The tool and methodology check out.

**Results:**

| N kept/layer | prune ratio | coverage (held-out) | hit rate, same RAM | Δ vs. baseline |
|---:|---:|---:|---:|---:|
| 256 | 0% | 100% | 64.4–65.5% | — |
| 192 | 25% (in-house "knee") | 93.6–95.2% | 67.8–69.4% | **+3.6 to +3.9 pts** |
| 176 | 31.25% (NU176) | 90.7–92.9% | 69.7–71.1% | +5.3 to +5.8 pts |
| 172 | 32.8% (brandonmusic) | 90.0–92.1% | 70.2–71.6% | +5.8 to +6.4 pts |
| 168 | 34% (0xSero-504B) | 89.1–91.4% | 70.7–72.1% | **+6.3 to +6.9 pts** |
| 160 | 37.5% (REAP37) | 87.2–89.7% | 71.9–73.2% | +7.5 to +8.2 pts |

Consistent across all three independent held-out traces and both ranking histograms — this
is a real, reproducible effect, not noise. **Explicit limitation, stated plainly**: this
measures hit rate *conditional on a token already landing on a surviving expert*. The
7–13% of traffic that historically hit a since-pruned expert is excluded from the
denominator, not rerouted — because simulating where a retrained router would send that
traffic requires the router's real gate scores, and `FAROUTE1` traces store **rank + expert
ID only, no gate/router scores** (confirmed directly from the trace format,
`docs/m5max-route-trace-cache-sim.md`: 24-byte record = `event_id, moe_call_id, layer,
batch_row, route_rank, expert_id`). This is a real gap this simulation cannot close from
banked data; it is why the actual post-heal hit rate can only be confirmed by loading a real
pruned+healed checkpoint (§4.2).

### 2.3 Translating to tok/s (first-principles, explicitly a projection, not a measurement)

Anchoring to numbers this project has already measured twice independently — raw SSD
bandwidth **13.5 GB/s** at the engine's exact read pattern (the prefill-I/O study;
`docs/PERFORMANCE_THEORY.md` `serve_prefill_raw_ssd_bandwidth_and_ttft_gap`, `mode15`
registration) and **~11.1 GB/token** full routed-expert bytes at 0% hit
(`README.md`: 8 experts × 75 layers × ~18.9 MB) — plus today's baseline hit rate (~65%,
consistent across all three traces §2.2 reproduced: 63.9–65.5%) and warm decode speed
(~1.5–1.6 tok/s, `README.md`):

`miss_GB(hit%) = 11.1 × (1 − hit/100)`; `disk_time = miss_GB ⁄ 13.5`; `total_time = disk_time
+ compute_and_other (held fixed — pruning doesn't change per-token FLOPs, per the NAEE
mechanism)`. Calibrating `compute_and_other` once against each baseline endpoint (1.5 and 1.6
tok/s) at 65% hit, then applying each prune ratio's **simulated** hit-rate range from §2.2's
table (not a re-derived estimate) gives:

| Prune ratio | Simulated hit range | Projected tok/s | Projected gain |
|---|---|---|---|
| 25% (N=192) | 67.8–69.4% | 1.55–1.70 | **+3.6% to +6.1%** |
| 31.25% (N=176) | 69.7–71.1% | 1.59–1.74 | +6.2% to +8.7% |
| 32.8% (N=172) | 70.2–71.6% | 1.60–1.75 | +6.9% to +9.5% |
| 34% (N=168) | 70.7–72.1% | 1.61–1.77 | +7.6% to +10.3% |
| 37.5% (N=160) | 71.9–73.2% | 1.64–1.79 | **+9.3% to +12.1%** |

Today: miss bytes/token ≈ 11.1 × 0.35 ≈ **3.9 GB/token** (consistent with Study 4's
independently measured 4.2 GB/token at hit 66.3%) → disk time/token ≈ 3.9⁄13.5 ≈ **0.29 s**,
against ~0.63–0.67 s/token total — disk is roughly 43–46% of the per-token budget, the rest is
compute/dispatch/other overhead this lever does not touch.

**This is a real, positive, but modest range (+3.6% to +12.1% across the whole 25–37.5%
sweep) — not the dramatic "prune X% → X% faster" outcome the direct-bytes framing would
predict.** It is a capacity/cache-economics win, correctly sized this time. (An earlier draft
of this arithmetic mixed two different bytes-per-hit-point slopes and produced an inflated
+5–21% range; the table above is the corrected, self-consistent version, recomputed and
verified with a script rather than by hand.)

### 2.4 The reason for real caution: this project's own pin-scaling precedent

Growing effective resident coverage does not reliably buy speed on this specific engine —
already measured directly: pin scaling from 23.5 GB (1,240 experts) → 46.9 GB (2,480 experts)
raised hit rate 66.2% → 69.9% but **dropped** throughput 1.43 → 1.32 tok/s, because "larger
pins steal adaptive-LRU capacity and add Metal dispatch overhead"
(`docs/PERFORMANCE_THEORY.md`, static-pinning entry; `internal ledger (not staged)` #30). This is not the
identical mechanism (that experiment grew the pin at fixed pool size; pruning shrinks the pool
at fixed pin), but it is a same-engine, same-general-shape counter-example proving that a
measured hit-rate gain **does not automatically convert to a speed gain** here. Treat §2.3's
range as optimistic pending a real measurement, not as a floor.

---

## 3. Prune + heal plan

### 3.1 What's already banked vs. what a real prune decision needs

| Signal | Banked? | Where | Sufficient for a prune decision? |
|---|---|---|---|
| Routing frequency (`.fa_usage`) | Yes, extensively | this doc §0 | No — already measured flat as a saliency proxy (internal findings (not staged)) |
| Quantization reconstruction error | Yes | `measure_expert_quant_error.py` outputs | No — already measured flat (internal findings (not staged)), and it's a different signal than REAP's anyway |
| Route order/sequence (`FAROUTE1` traces) | Yes | `c/bench-m5max/*/routes-*.bin` | Enables the cache-hit-rate simulation (§2.2) but **not** REAP's real criterion — no gate/router scores stored |
| **REAP's actual saliency** (`gate_value × ‖expert_output‖₂`) | **No** | — | This is the one signal that matters and is not recoverable from anything currently banked; requires a new instrumented calibration pass (§3.3) |

### 3.2 Cheapest decisive first experiment: don't build anything, measure an existing checkpoint

Per the model-surgery frontier notes §2.3(f)(1)'s own recommendation, sharpened here: **acquire
one already-released community checkpoint** — `0xSero/GLM-5.2-REAP-504B-GGUF` (34% pruned,
168/256 kept, the ratio closest to this pass's simulated sweet spot) is the natural first
pick; it costs zero training, only a format conversion and a benchmark run.

**Pre-registered gates** (matching this project's own house convention —
the compression-campaign registration, the stage-2 v4 registration):

1. **Gate 1 — format/load.** Convert to iliria's int4 container (or confirm GGUF
   compatibility) and load on the M5 Max. FAIL condition: won't load / won't decode a single
   token cleanly → stop, try the next checkpoint (NU176 or REAP37-MLX).
2. **Gate 2 — real hit-rate/tok/s.** Capture a `FAROUTE1` trace via the existing
   `run-m5max-route-study.sh` harness (already used for every prior route study in this repo)
   and run it through `simulate_m5max_cache.py` unmodified. Compare against §2.2's projected
   +6.3 to +6.9 pts and §2.3's +7.6% to +10.3% tok/s (34% ratio row). This is the single
   measurement that resolves whether the pin-scaling counter-example (§2.4) applies here or
   not — decisive either way.
3. **Gate 3 — quality, extending `ili bench`.** The current gate (MMLU/HellaSwag/ARC,
   `README.md`) is exactly the short-form battery that already let a compression regression
   through undetected once on this project's own history (the tool-call regression record:
   45/46 eval score, malformed tool-call tags invisible to the suite). Any prune quality gate
   must add, at minimum: (a) the long-generation collapse probe this project's own prior
   finding already flags as necessary (`glm52-quantization-and-finetuning-findings.md` #1 —
   GLM-5.2 degrades specifically on *long* generations); (b) a small, deliberately-broad
   factual-recall spot-check (reuse brandonmusic's own documented failures — capital of
   Kentucky/Texas, Marbury v. Madison — as a free, already-diagnosed regression test); (c) a
   tool-call-format check if iliria's serving path uses native tool-call tags at all, given
   the exact same regression already happened once on a sibling model.
4. **Decision rule:** if Gates 1–3 all clear at 34%, iliria has a real, cheap, in-hand
   answer and this document's projections are confirmed or corrected with actual numbers —
   no in-house REAP computation needed yet. If quality fails narrowly, try NU176 (31.25%,
   the mildest available community ratio) before concluding the technique doesn't transfer.

### 3.3 If the checkpoint route is inconclusive: compute REAP's real saliency in-house

the model-surgery frontier notes §2.3(f)(2) already specifies this correctly: compute
`gate_value × ‖expert_output‖₂` directly on GLM-5.2's own routed experts over a **domain-matched
calibration set** — and per this project's own Study 2, that calibration set must be
**pooled/broad, not coding-only**: general-vs-coding top-25% expert sets overlap only **62%**
on iliria's own measured data, and a general-only tier would capture just 47% of coding mass
(the offline math-verdict record Study 2) — a smaller-scale echo of EASY-EP's sharper finding
on the closest architectural analog (DeepSeek-R1/V3): a math-calibrated pruning mask collapsed
LiveCodeBench 63.32→38.00, a ~40% relative loss, from a mask that looked fine on its own
domain. **This requires a calibration forward pass over the live model — a real model load —
and is explicitly deferred, not attempted in this banked-data pass.**

### 3.4 Heal: domain/soul-targeted, per this project's own load-bearing lesson

This project's own precedent (the GLM-5.2 evaluation record) is unambiguous on one point even
though its causal attribution is confounded: v1 (77% prune + generic calibration) broke; v2/v3
(same 77% prune, domain/code-soul-targeted calibration) shipped clean. Whether the *heal*
specifically (vs. the confounded bit-width change) was the load-bearing variable is **not
cleanly isolated** — the record itself says so — but **every other data point gathered in
this pass points the same direction**: brandonmusic's factual-recall fix was a calibration
rebalancing, not a ratio change; the Level1Techs "if all the coding experts are still in
there" comment; this project's own Qwen3.6-35B-A3B result where healing *hurt*
(71.7% vs. prune-only 82.6%) makes clear **heal is not automatically beneficial and must be
A/B'd against a prune-only control every time**, not assumed — a real, receipted caution
against skipping straight to "heal will fix it."

---

## 4. The narrow-deployment angle — does the community failure mode even apply to us?

**The case for "we might not care":**
- The community's own recurring complaint — *"reap lobotomizes the model for tasks other
  than coding"* — is, read differently, a description of a filter that might be **free** for
  a single-user, coding/agentic-focused deployment that was never going to ask GLM-5.2 for
  the capital of Kentucky or a Marbury v. Madison explanation in the first place.
- `ili` is already a single-user, narrow-task engine by design (`README.md`), not a
  general-knowledge assistant competing on MMLU trivia.

**The case for caution anyway, stated with equal weight:**
- **"Coding" is not as narrow a calibration target as it sounds, on this project's own
  measured data.** Study 2's 62%-overlap finding means even a pure-coding calibration set
  measurably misses non-coding-but-still-relevant expert mass — and the failure mode is not
  always "wrong trivia," it can be **format-level** (this project's own
  the tool-call regression record: a 25% expert prune on a sibling model silently broke native
  tool-call tag formatting, invisible to a 45/46 eval pass, only caught by incidental adjacent
  use). A single coding-focused user still depends on tool-calling, agentic formatting,
  occasional general reasoning mid-task, and whatever the model's own chain-of-thought
  scaffolding needs — none of which is "world trivia," all of which could plausibly route
  through an expert a coding-only calibration set ranks as safe to cut. The "Expert Strikes
  Back" finding that tail experts are frequently **narrow specialists** rather than fungible
  capacity (demonstrated on an actual GLM checkpoint, GLM-4.7-Flash) is the mechanistic reason
  a low-frequency expert cannot be assumed irrelevant just because this deployment's typical
  prompts don't obviously need it.
- **The real, actionable synthesis**: narrow deployment is a legitimate reason to *accept* a
  known, scoped capability loss (e.g., "we know this model can no longer reliably state U.S.
  state capitals, and we've verified we don't care") — it is not a reason to *skip verifying*
  what was actually lost. The gate in §3.2/Gate 3 exists precisely so "we probably don't care"
  becomes "we checked, and we don't," which is the same discipline this project already
  applies everywhere else (receipts-first, per the GLM-5.2 evaluation record #5).

---

## 5. GO / NO-GO

**GO — conditionally, on the cheapest experiment, not on building a prune pipeline.**

The utilization data rules out the two easy answers ("uniform, forget it" / "huge free dead
tail, prune anything"). What is left is a real, moderate, quantified opportunity — **+3.6 to
+8.2 hit-rate points, projecting to roughly +3.6% to +12.1% decode tok/s at
community-validated prune ratios (25–37.5%)** — gated on a real, not yet closed, uncertainty
(does the hit-rate
gain survive this engine's own LRU/dispatch overhead, per the pin-scaling precedent?) and a
real, not yet closed, quality risk (does iliria's own narrow use case actually survive the
same calibration-narrowness failure mode every other GLM-5.2 REAP attempt has hit?).

**Cheapest decisive next experiment** (requires a model load — explicitly the next
non-banked-data step, not part of this pass): pull `0xSero/GLM-5.2-REAP-504B-GGUF` (already
exists, zero training cost), run it through the three gates in §3.2. That single experiment
resolves both open uncertainties — the real tok/s number (confirms or kills §2.3's
projection) and a real quality readout on an extended `ili bench` — before any decision to
invest in an in-house REAP computation (§3.3) or a from-scratch prune+heal build.

---

## Sources

**In-repo (verified by direct read/recompute this pass):**
- the model-surgery frontier notes §2.3 (the source survey underlying this analysis)
- the offline math-verdict record Study 2 (routing-mass/saliency),
  Study 1 & 5 (hit-rate fits), Study 4 (miss-bytes/hit-rate at varying k)
- `docs/PERFORMANCE_THEORY.md` (Jaccard/memorylessness, pin-scaling,
  SSD bandwidth, hotset-training-broadened-not-concentrated entries)
- the prefill-I/O study, `docs/m5max-route-trace-cache-sim.md`
- `c/analyze-m5max-hotset.py`, `c/tools/simulate_m5max_cache.py`,
  `c/tools/measure_expert_quant_error.py` (read/run directly this pass)
- `c/bench-m5max/rope-variance-233636/fa_usage{,.coding}.snapshot`,
  `overnight-20260714-001239/routes-{baseline,coding}.bin`,
  `overnight3-20260714-003224/routes-shotgun.bin` (raw data recomputed/replayed this pass;
  note: these specific raw-data paths are not part of this repo's staged
  `c/bench-m5max/` subset)
- (an internal cross-project findings log, not part of this release, provided
  additional community-context and prior-negative-result citations for this pass)

**Web/HF (fetched and verified this pass, 2026-07-19):**
- REAP paper: ["REAP the Experts"](https://arxiv.org/abs/2510.13999), arXiv:2510.13999 (ICLR
  2026, Cerebras) — cited via the survey's own already-verified reading
- [`0xSero/GLM-5.2-REAP-504B-GGUF`](https://huggingface.co/0xSero/GLM-5.2-REAP-504B-GGUF)
- [`brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172`](https://huggingface.co/brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172)
- [`pipenetwork/GLM-5.2-REAP37-MLX-4bit`](https://huggingface.co/pipenetwork/GLM-5.2-REAP37-MLX-4bit)
- [`0xSero/GLM-5.2-REAP-NU176-526B`](https://huggingface.co/0xSero/GLM-5.2-REAP-NU176-526B)
- [Level1Techs forum: "Glm 5.2 reap"](https://forum.level1techs.com/t/glm-5-2-reap/251814)
- NAEE: [arXiv:2402.14800](https://arxiv.org/abs/2402.14800); EASY-EP:
  [arXiv:2504.06792](https://arxiv.org/abs/2504.06792); "Expert Strikes Back":
  arXiv:2604.02178 (all cited via the survey's own verified readings)
