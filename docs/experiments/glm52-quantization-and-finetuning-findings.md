# GLM-5.2 quantization and fine-tuning findings

Prior work on GLM-5.2 (a separate compression/heal project — internal
findings, not staged as part of this repository) produced several findings directly relevant
to iliria's int4 streaming path. They are recorded here so the next
quantization sweep or fine-tuning attempt starts from measured evidence, not
folklore.

## 1. Low-bit collapse is length-dependent, not uniform

At 3-bit, GLM-5.2 does not degrade evenly — it degrades **specifically on long
generations**. In a per-facet scorecard the 2 elite outputs were short (~1258
chars); the 13 rejected outputs were long (5–7K chars) and collapsed into
repetition and corrupted tokens. Short output from the identical weights was
fine.

**Fix (confirmed by two independent parties):** saliency-dynamic mixed
precision — protect salient + early-layer experts at 4-bit+, leave the rest
lower. Both Zhipu's own official GLM-5.2 quantization and Unsloth's community
"Dynamic 2.0" quantization use exactly this recipe (per-layer sensitivity-based
bit-width assignment, not uniform). This is not a decoding trick; the collapse
is fixable only by mixed precision on the critical experts.

**What this means for iliria:** before spending GPU time on an int2/int3
quality sweep, expect long-generation collapse and design the sweep around
mixed-precision (saliency-tiered bit-width), not uniform requant. The engine
already supports per-tensor bit-width (`fmt` 0/1/2/3 in `QT`), so a
mixed-precision container is a converter change, not an engine change. When
running the quality benchmark (`ili bench`), include a long-generation gate
in addition to the multiple-choice scoring — the multiple-choice harness may
not exercise the collapse zone.

## 2. Requant math can cancel its own memory win

If experts are already at native 2-bit (gate/up), a uniform 3-bit requant
**grows** those tensors 50%, netting ~0.96× the original size — canceling the
prune or compression it was supposed to help.

**Rule:** before any mixed-precision experiment, compute the actual before/after
byte count, not the intuitive direction. A "smaller bit-width" lever can be
net-neutral or net-negative depending on the starting point.

## 3. Quantization-lineage mismatch is a distinct MTP failure mode

iliria documents that an int4 MTP head gives 0% draft acceptance and the head
must be int8. A separate, fifth failure mode exists: pairing a bf16/fp8
*official* MTP head with a *community requant* of the trunk also gives 0%
acceptance (measured on DeepSeek V4 Flash). The head itself isn't broken —
industry practice of pairing a bf16 MTP head with a quantized trunk is normally
fine — but for a specific community requantization + official sidecar pairing,
it collapsed.

**What this means for iliria:** the current int4-trunk + int8-head pairing
works. If anyone changes the trunk precision (e.g., a mixed-precision container
per finding #1), do not assume the existing MTP head still matches — re-verify
draft acceptance. The mismatch is a distinct mechanism from the int4-head
issue, not the same bug.

## 4. Aggressive LoRA healing degrades on a compressed base

On a base near its compression ceiling (heavily quantized GLM-5.2), an
aggressive LoRA heal (rank 16, 200 iters) **degraded** quality — the base gave
an elite, correct SQLi answer; the healed model gave a generic, worse one. A
gentle heal (rank 8, 60 iters) was neutral (no lift, no degrade), confirmed by
a controlled A/B.

**Rule:** a heal's effect is regime-dependent — re-test "does this heal help"
at each compression level. Neutral on a base with headroom can become actively
harmful on a base pushed closer to its ceiling. int4 GLM-5.2 is near that
ceiling.

## 5. Validation loss ≠ behavioral quality

A 170× larger training corpus achieved **better** validation loss (1.033 vs
1.250) but **worse** behavior: it introduced an unterminated `<reflection>`
loop the smaller corpus never exhibited. The best-behaving model was the small
corpus at iteration 200, not the big corpus at iteration 300.

| approach | val loss (best) | behavior |
|---|---|---|
| small corpus (238 examples, iter 200) | 1.250 | clean stop, correct diagnosis + fix |
| large corpus (40,473 examples, iter 300) | **1.033** | reflection loop, missed the real bug, repetition |

**Rules:**
- Watch behavioral output at each checkpoint; don't trust val loss alone.
- Stop at the measured peak — the exact iteration is data-specific, not a
  number you can borrow from a prior experiment.
- A corpus that trains "better" by the loss metric can introduce a new failure
  mode (here: an unterminated reflection habit from training-data content
  shape). The fix is cleaning the corpus to exclude the problematic style, not
  adding more data.

## 6. Patch tooling for quantized models needs whitespace-flexible matching

Quantized models mis-indent measurably more than full-precision ones (Aider
benchmark: BF16 71.4% → q4 53.4% edit success, an ~18-point drop). If
code-editing agents are built on top of iliria, design edit tools assuming
imperfect formatting is common, not an edge case. Use changed-line-anchored,
whitespace-flexible matching, not exact-string matching.

## Process lessons

- **Receipts-first.** Every claim backed by a JSON receipt, not vibes. This
  practice caught the EOS bug, a truncation bug, and verifier miscalibrations
  in the next project — none by guesses.
- **Audit before GPU.** Reading a prior project's incident log before spending
  compute saves days. This doc IS that incident log for GLM-5.2 compression
  work.
- **Recognition ≠ loading ≠ correct output ≠ improvement over baseline.** Only
  the last one, measured, counts.
- **Negative results are the deliverable.** Documenting what doesn't work (MTP
  on memory-bound MoE, 3-bit uniform requant, aggressive healing on a
  compressed base) is worth more than shipping a broken feature with a nice
  card.
