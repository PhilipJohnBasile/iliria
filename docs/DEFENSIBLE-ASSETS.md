# Defensible assets — iliria



Written in response to an external portfolio review flag (2026-07-18): "the moat is
perishable — document reproducibility, licensing and the defensible assets:
scheduler, data layout, falsifiers and accumulated measurements." This is a
provenance and reproducibility record: what's real, what's measured, what's
cited, and where the receipts are. Terse, pointer-heavy; every claim below
carries a file path or commit hash. **NOTE: commit hashes and some cited
artifacts throughout this document refer to the project's private research
history, not this public repo** — see `STORY.md` §13 for which evidence
categories are actually shipped here.
Written at iliria HEAD `c433062` (2026-07-18). Both repos have background
jobs actively committing during normal operation (long encodes, overnight
matrices) — treat any specific HEAD hash as a snapshot, not a constant; the
per-claim commits cited in section (d) are the stable references.

A sibling fast-tier engine ([trailbrake](https://github.com/PhilipJohnBasile/trailbrake)) shares the falsifier methodology described in (b).

## (a) The measurement corpus

**Index:** `docs/PERFORMANCE_THEORY.md` — 43 entries, GENERATED (do not
hand-edit) from `docs/performance-theory.json` by `docs/gen_performance_theory.py`.
Regenerate: `python3 docs/gen_performance_theory.py`. Each entry carries:
regime (phase/context/cache/batch/container/backend), hypothesis, pre-measurement
prediction, evidence class (`measured/same-commit` > `measured/cross-session`
> `synthetic-fixture` > `offline-simulation` > `calibrated-model` >
`literature-only`), one or more cited measurements, and three independent
status axes — **mechanism** / **performance** / **shipping** — because a
mechanism can be real while the perf claim is dead, or confirmed while
shipping stays opt-in. Two cross-source numeric conflicts are flagged, not
silently resolved (`## Cross-source conflicts` section at the end of the
file). Full source list: `## Sources` section of the same file (this repo's own docs/README/commits and internal synthesis notes that are not staged here).

**Shipping-status breakdown** (mechanical, reproducible with
`grep "Status:" docs/PERFORMANCE_THEORY.md`), 43 entries total:

| shipping status | count | examples (entry id) |
|---|---:|---|
| `default_on` | 4 | `serve-prefix-kv-reuse-delta-prefill`, `metal-prefill-attention-kernel-gate-2`, `mixed-format-moe-dispatch-hazard`, `a-rope-threadgroup-race` |
| `enabled` | 2 | `s-row-projection-gemms-prefill-gate-1`, `chunked-serve-prefill` |
| `opt_in` | 5 | `pilot-k6-router-lookahead-prefetch`, `dsa-sparse-indexer`, `context-diet-trimming`, `a4-mla-native-tiled-dense-attention`, `kv-snapshot-replay-fidelity` |
| `retained` (infra/reference value only) | 4 | `metal4-moe-submission`, `persistent-metal-state`, `quality-baseline-ili-bench`, `kv-slot-allocation-economics` |
| `dead` / `shipped_anyway`-then-dead | 6 | `mtp-speculative-decoding`, `static-pinning-hotset-training`, `bomb-shotgun-bulk-prewarming-prefetch`, `pilot-salvage-prefetch-aware-eviction-veto`, `heterogeneous-cpu-gpu-moe-tail`, `tiered-int4-int2-mixed-precision-container` |
| `not_built` (proposed, modeled, or gated-out) | 21 | includes `mode15-lossless-huffman-burst-decode` (kill-shots passed, primary forward lever), `a2-route-known-arrival-order-overlap` (FALSIFIED, the a2-falsifier verdict, commit `c627421`) |
| split/other | 1 | `expert-quant-error-saliency` (ranking use falsified, absolute-magnitude use unvalidated) |

Read literally: of 43 registered ideas, **6 are dead**, **21 never got built**
(mostly because an earlier gate killed the premise or a cheaper prior lever
closed the gap first), and only **~11** are actually live in the shipped
binary today. That ratio — not the 11 — is the asset: it's the evidence the
discipline in (b) is load-bearing, not decorative.

**Raw data locations** (mostly gitignored, i.e. NOT pushed — see the
reproducibility gap list in (d)):
- `c/bench-m5max/` — every dated run directory, `campaign-log.md` (110 KB
  running log), `compression-gates/results/*.json`, `stage2-v4/` (traces +
  KV snapshots + `VERDICT.txt`), `salvaged-allocation-manifest.json`.
- `Internal cross-project session log`
  — the dated (2026-07-13 → 07-18) cross-project session log the theory DB
  cites directly; treat it as the narrative complement to the theory DB's
  structured entries.

## (b) The falsifier methodology (the actual moat)

The mechanism, not any single result, is what survives personnel or context
loss. In order:

1. **Pre-register before data exists.** Every campaign has a registration doc
   committed *before* the measurement or the build: the a2-overlap falsifier registration
   (commit `15a70329`-adjacent, measurement-only, explicit PASS/FAIL/INCONCLUSIVE
   criteria before instrumentation), the stage-2 v4 registration
   (before the KV-snapshot engine existed), the Mode-15 lossless-pipeline registration
   and the compression-campaign registration (gates registered
   before the container was built).
2. **Kill-shots with numeric thresholds signed off in advance.** E.g. Mode-1.5's
   F1–F8 integration falsifiers (the Mode-15 integration design notes,
   registered commit `18dfeaa`, "Review ruling: prevent hindsight drift — update
   THIS entry after the factorial"); compression Gates 2–3 numeric thresholds
   fixed in the compression-gate threshold record *before* the
   container existed. **The discipline caught a real bad result**: Gate 3
   FAILED (commit `bb7f0e5`, case D killed — activation error 3–6x over
   threshold, MLP amplifying a fitted-int2 quantizer error the external reviewer had predicted
   the mechanism for) and the container was deleted rather than shipped
   (the compression-campaign registration, "GATE 3 VERDICT ...
   FAIL"). This is the concrete evidence the gates aren't theater. **Scope
   note:** this killed **one specific int2-tier design** (5 layers, 1
   capture) — not int2 quantization in general; "int2 is proven unusable"
   would overstate what this single result supports.
3. **A standing external adversarial reviewer**, not a one-time check —
   commits `e72040b` ("External review adopted: math reset — 1.14x not
   2.0–2.4x"), `f370851` ("External review round-3 amendments"), `0babfca` ("Adopt
   external-review gate amendments"), `7608967` ("External review round-2 adopted").
   Raw review artifacts are in the private research history, not staged here.
4. **Epoch freezing — "engine-freeze, not repo-freeze."** `validation/epoch-YYYYMMDD-NN/`
   is an immutable validation snapshot —
   `manifest.json` recording `epoch_id`, `created_utc`, `source_commit`, suite
   status, and the engine binary's own sha256, e.g. `validation/epoch-20260718-07/manifest.json`
   → `source_commit: dac28621...`, `glm_sha256: 46b17a50...` — so day-to-day doc
   or tooling commits on `main` can't silently invalidate a result mid-epoch. (The
   compiled `glm`/`ili` binaries are recorded by hash but not redistributed in this
   release; rebuild from `source_commit` to reproduce and verify them.)
   Seven epochs exist (`01`–`07`, 2026-07-15 → 07-18). The concept is itself a
   theory-DB entry (`provenance-status-split-and-engine-freeze`) that admits
   its own gap: epoch-01 hashed only the `glm` binary, not `ili`, launcher
   scripts, or generated Metal source — flagged `EXPAND-NEEDED` in the same
   pass that created it, not swept under the rug. Generator: `c/scripts/provenance.sh`
   + `c/tools/provenance_manifest.py`; diff tool: `c/tools/provenance_compare.py`
   (hard-fails an A/B pair on any unintended binary/source drift).
5. **Measurement-validity discipline** ("quiet machine" enforcement, casually
   called "launchd doctrine" for the class of bug it catches — a torn-down
   benchmark server reparenting to `launchd` and lingering as an unkilled,
   contaminating background process): `c/tools/timing_watchdog.py` (generic
   foreign-USER-process detector, commit `0052b7b` — not just a name-checker;
   flags any same-uid process outside an allow-list that persists or spikes
   CPU), `c/scripts/quiesce_check.sh` (per-run quiet-machine sample, embedded
   in every epoch/provenance manifest), `c/scripts/run_abba_matrix.sh` (hard-aborts
   the whole matrix if server teardown can't be verified clean — see its own
   comment at line ~201), `c/scripts/timing_lock.sh` (single-writer lock so two
   timing-sensitive runs can't silently contend). The provenance-vs-timing
   validity split itself is pre-registered as a named gap, not a finished
   system — see (d) for what that means for reproducing any *timing* number
   specifically (correctness/quality numbers are less exposed to this).
6. **A single confirmation-matrix standard for any performance claim** —
   `README.md` § "Benchmarking," `c/ab-m5max-k6-matrix.sh`: 3 held-out coding
   prompts × warm/cold × 3 interleaved ABBA trials, snapshots/restores
   `.fa_usage` before every run, hash-gates on the exact output-token stream,
   +5% median-paired-throughput promotion bar. This is the bar that killed
   PILOT K6 despite it looking good on a single prompt (README § "Settled
   verdicts").

## (c) Engine-specific know-how (pointers, not re-explanation)

- **Expert streaming scheduler** — `c/glm.c` (single file, ~4,000 lines, no
  BLAS/Python at runtime): `expert_load()` (~line 1165, the hot-path pread +
  fallback), the per-layer LRU (`ESlot **ecache`, struct field ~line 162),
  learned hot-store pinning `pin_wire()`/`pin_load()` (~lines 3474/3491),
  live re-pin at turn boundaries `repin_pick()`/`repin_pass()` (~lines
  3019/3033), async readahead (`madvise`/`WILLNEED` comments ~lines
  1434, 1968). `c/tools/README.md` and `c/tools/*.py` are the offline
  companions (conversion, fixtures, cache/trace simulators) — never a runtime
  dependency.
- **KV snapshot / provenance system** — `kv_disk_*` save/load functions in
  `c/glm.c` (fp32, byte-exact; parity proof commit `087214a`, argmax-hash
  `1b4e675e`); `c/scripts/provenance.sh` + `c/tools/provenance_manifest.py`
  (executable provenance manifest, captures the engine's own
  `EFFECTIVE-FLAGS:` stderr line); `c/tools/provenance_compare.py` (A/B
  manifest diff). This is the same machinery epoch-freezing and Stage-2 v4's
  causal result both depend on.
- **Mode-1.5 lossless codec chain** (not yet shipped; kill-shots passed) —
  `c/codec_row_huff.h` (canonical row-Huffman, shared bit-for-bit between the
  Metal kernel and a CPU cross-check decoder), `c/mode15_reader.c`/`.h`,
  `c/tools/mode15_container.py` + `c/tools/encode_mode15_container.py`
  (resumable, checksum-verified encoder), gate:
  `c/bench-m5max/compression-gates/gate_m15_g1.py` (bit-exactness: 895
  tensors / 3,057,664 rows byte-identical, commit `345c9ef`). Design doc:
  the Mode-15 integration design notes; registration:
  the Mode-15 lossless-pipeline registration (F1–F8 falsifiers,
  review-signed thresholds).
- **Env-var contract** — `c/tools/effective_flags.py`: single source of
  truth for `ILI_` > `COLI_` > `FA_` prefix resolution + `atoi()`-style
  truthiness, deliberately re-derived (not copy-pasted) in three languages
  (C engine's own `ili_env()`, this Python module, shell callers) so the
  alias logic can't silently diverge; consumed by `c/tools/eval_glm.py` and
  the shell timing drivers (`c/scripts/roofline_run.sh`,
  `c/scripts/run_abba_matrix.sh`). Also encodes known footguns directly,
  e.g. `check-metal-prefill-guard` (catches `ILI_METAL_PREFILL=1` set
  without `ILI_METAL=1`, which silently no-ops to all-CPU — a mistake that
  once cost 3h14m of wall time).
- **Quiesce / watchdog doctrine** — same pointer set as (b)5.

## (d) Reproducibility — per major claim

| Claim | Number | Epoch / commit | Command | Raw data |
|---|---|---|---|---|
| Warm decode baseline | **1.42–1.58 tok/s** (matrix-confirmed range; the round "~1.6 tok/s" figure in `docs/roadmap-daily-driver.md` is a conversational reference to this range, not its own hash-gated run) | `c/bench-m5max/k6-matrix-20260714-090059/` | `cd c && bash ab-m5max-k6-matrix.sh /path/to/model` | frozen `.fa_usage`, hash-gated output |
| Metal-4 MoE submission | +2.05% median | `c/bench-m5max/m4-matrix-20260714-082044/` (+`-090059`) | `cd c && ILI_K6_AXIS=metal4 bash ab-m5max-k6-matrix.sh /path/to/model` | matrix output in that dir |
| Metal prefill kernel TTFT | 6.3–9.1x (821s→130.4s turn 2, 1523s→168.3s turn 3) | commits `ee81bbb`/`9d0390b` (serve-gate before/after); canonical ratio 9.049 per `docs/PERFORMANCE_THEORY.md`'s own cross-source-conflict resolution | `c/scripts/serve_gate.py` for the serve-path replay | **GAP — see below: this number has never been run through the house confirmation matrix** |
| Causal stall result (expert-read delay → decode latency) | mean +365.2s paired decode-latency increase (95% CI [359,369]s **on the mean**, not the median); median +368s; **5 paired blocks** (not 20); dose = **16 ms** injected delay per expert read (source label "16000" denotes 16 ms, not a raw-unit literal) | registration the stage-2 v4 registration (pre-data) → commits `e876276` (registration) → `30a1cc8` (kv_disk engine) → `087214a` (parity proof) → `bee19b4` (canary pass) → `5a70329` (CONFIRMED); frozen binary `validation/epoch-20260717-05/` (`glm` sha256 `29a96d5e...`) | replay against the frozen epoch-05 binary + the 6 KV snapshots via `c/bench-m5max/stage2-v4/`'s runner (see `validation/epoch-20260717-04/stage2_run.sh`/`canary_v3.sh` for the shape) | `c/bench-m5max/stage2-v4/VERDICT.txt`; **GAP — the 6 KV snapshot files themselves are large binary artifacts and, like the rest of `c/bench-m5max/`, are not committed** |
| Mode-1.5 compression ratio | 0.7526 (trace-weighted encoder ratio); bit-exactness 895 tensors / 3,057,664 rows | commit `efcdfd3` (smoke), commit `345c9ef` (G1 gate) | `python3 tools/encode_mode15_container.py ...` (see script `--help`); gate: `python3 bench-m5max/compression-gates/gate_m15_g1.py` | `c/bench-m5max/mode15-smoke-20260718.log`; `c/bench-m5max/compression-gates/results/gate_m15_g1-GLM-5.2-mode15-huffman-20260718T195921Z.json` |
| Mode-1.5 GPU row-decoder bandwidth | 36.51 GB/s solo; ≥32 GB/s contended (2.4x over the 13.5 GB/s bar) | mode-1.5 registration kill-shot section | `bash c/bench-m5max/contention-killshot.sh` (committed, runnable as-is) | `c/bench-m5max/contention-killshot-142659/`, `contention-killshot.log` |
| Quality baseline (int4 vs published FP) | 62.5% mean acc_norm (hellaswag/arc/mmlu, n=40/task) | `quality-baseline-20260715-take5.log` | `cd c && ./ili bench` | **GAP — see below** |

**Reproducibility gaps found (explicit findings, not just table footnotes):**

1. **The Metal prefill kernel's headline 6.3–9.1x TTFT number has never been
   run through the project's own house confirmation matrix.** The theory DB
   says this in its own words (`metal-prefill-attention-kernel-gate-2`
   entry): performance status is `pending` ("Serve-ABBA-pending"), not
   `confirmed`, "despite already being the shipped default." It's a real
   before/after comparison, not fabricated, but it's a weaker evidence class
   (`measured/cross-session`) than the matrix standard everything else in
   README's "Settled verdicts" table meets.
2. **`c/bench-m5max/` — where nearly all raw run data lives — is a local,
   gitignored directory, not committed.** Every epoch's KV snapshots, gate
   JSON results, campaign logs, and contention-killshot logs cited above and
   throughout the theory DB exist only on this machine. If this machine is
   lost, the *narrative* (theory DB, registration docs, commit messages)
   survives in git; the underlying raw measurements do not. Anyone
   reproducing a claim must re-run it, not diff against the original run.
   **Status: mitigated for the text tier 2026-07-18, in the private research
   repo** (1,454 files / 25.42 MB force-added there — logs, gate JSON, CSVs,
   campaign log, registration/design docs, `.fa_usage` determinism snapshots
   — see that repo's preservation manifest for the full
   inventory and regeneration recipes). **Neither that manifest nor the
   1,454-file text tier is part of this public repo**, which ships only the
   curated evidence subset described in `STORY.md` §13; **the binary tier (6 KV-snapshot files, ~8.7 GB in
   `stage2-v4/snapshots/`) is still machine-local only**, regenerable in
   ~1h/snapshot per the same manifest but not itself committed.
3. **The quality-baseline n=100 rerun was planned and never completed.**
   Theory DB's own notes: the rerun "was armed to auto-chain after the
   container build finished, per the latest committed record; no committed
   result from it exists yet." The only committed quality number is the
   n=40/task run, explicitly flagged as too small to distinguish a real
   effect from noise for anything beyond the roughest sanity check.
4. **UPDATED 2026-07-20 — Mode-1.5's end-to-end decode speedup is no longer
   unmeasured; it has been measured, and the result is a regression, not the
   predicted 1.132x gain.** Proven correct/lossless end-to-end on the real
   744B/272GB container (5/5 token-identical vs. the int4 reference) — the
   bit-exactness and raw decoder-bandwidth kill-shots remain real and
   reproducible (see table above) — but measured **~1.6–1.7x SLOWER** than
   plain int4 (cold, `AUTOPIN=0`): the CPU Huffman-decode path's own overhead
   exceeds the NVMe bytes it saves. The GPU row-decoder is independently
   known fast enough to reverse this (36.51 GB/s solo, ≥32 GB/s contended —
   see the row above), but is not yet wired into the live
   `expert_load`/`qt_from_disk` decode path (only the CPU path is wired,
   commit `1ac31b9`); that wiring is the actual next lever, not the
   four-cell factorial this gap previously named — the factorial's
   underlying question (does the codec's own throughput beat plain int4
   end-to-end?) has now been answered on the CPU path: no. A second,
   independent blocker was also found this session: a reproducible
   multi-turn **silent-death** bug in `ili serve` with Mode-1.5 active (fix
   in flight, not yet committed). `shipping: not_built` remains accurate,
   but for a different reason than before: not "unmeasured," now "measured
   slower, plus an open serving bug." Full citations:
   an internal findings note
   §3.
5. **Epoch-01 (`validation/epoch-20260715-01/`) is an incomplete freeze** —
   it hashes only the `glm` binary, not `ili`, launcher scripts, or build
   flags — acknowledged in-repo (see (b)4) but still worth flagging here:
   any claim whose only epoch anchor is `epoch-01` should be treated as
   weaker provenance than one anchored to `epoch-02` onward.

## (e) Licensing / provenance

- **This engine**: Apache 2.0 (`LICENSE`). Fork of
  [JustVugg/colibri](https://github.com/JustVugg/colibri) (also Apache 2.0)
  — `NOTICE` names what's original to this fork (the M5 Max specialized
  engine and Metal backend generation, iliria-specific modifications to
  PILOT router-lookahead prefetch -- PILOT itself existed upstream in
  colibri -- the route-trace/cache-simulation tooling, the
  confirmation-matrix benchmark harness, the iliria rebrand) versus
  inherited.
- **Model weights actually in use**
  (`/path/to/models/GLM-5.2-int4-with-int8-mtp/README.md`):
  front-matter declares `license: mit`, `base_model: zai-org/GLM-5.2-FP8`,
  `base_model_relation: quantized`. The "Provenance & license" section states
  the conversion is from zai-org/GLM-5.2-FP8 (MIT), "this derivative is
  likewise MIT," done with colibrì's official, unmodified converter, and that
  this specific mirror was cloned from jlnsrk/GLM-5.2-colibri-int4 with the
  int4 MTP heads replaced by int8 ones (the file-size fingerprints for
  telling them apart are in this repo's own `README.md`). No separate license
  file exists inside the model directory beyond this front-matter — that
  front-matter block is the license record.
- **No fine-tuning, expert merging, or router recalibration exists anywhere
  in this codebase** — verified by absence: `c/tools/` is conversion
  (`convert_fp8_to_int4.py`), evaluation (`eval_glm.py`), and offline analysis
  only; there is no training loop. This matches README's own claim ("the
  model itself is never modified") and matters for licensing because it means
  the weights-in-use inherit GLM-5.2-FP8's MIT terms cleanly, with no
  derivative-training question to adjudicate.
- No GPL/copyleft dependency was found in `LICENSE`/`NOTICE`; the engine
  is dependency-free C at runtime (README: "No BLAS, no Python at runtime").
