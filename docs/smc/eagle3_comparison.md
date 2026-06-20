# SMC-SD vs EAGLE3 vs vanilla — Qwen3-8B (B200)

Head-to-head of our finetuned-draft SMC-SD against EAGLE3 speculative decoding
and plain batched decoding, same target (Qwen3-8B), same prompts/scoring,
temperature 0.7, `--disable-thinking`, seed 0, single B200.

- **SMC-SD**: production draft (Qwen3-0.6B, β1.5+klmix0.5), N=4 (GSM8K) / N=8
  (HumanEval), γ=8. `mem-fraction 0.4` (two model runners).
- **EAGLE3**: `Tengyunw/qwen3_8b_eagle3` head via sglang
  (`--speculative-algorithm EAGLE3 --num-steps 6 --eagle-topk 10
  --num-draft-tokens 32`), mem 0.85, accept length ≈ 4.6.
- **vanilla**: no speculative decoding.

Metric = aggregate **output (useful) tokens/s** at a given number of concurrent
requests (batch size); accuracy in parentheses.

## GSM8K throughput scaling

| batch | vanilla | EAGLE3 | SMC (prod, N=4) |
|------:|:---:|:---:|:---:|
| 1   | 264 (91.0) | **509 (92.5)** | 460 (79.0) |
| 32  | 3537 (89.5) | **3779 (93.5)** | 2846 (79.5) |
| 64  | **6204 (93.8)** | 4278 (93.0) | 4394 (78.1) |
| 128 | **9738 (91.4)** | 4700 (91.4) | 5143 (78.5) |

## HumanEval (latency vs throughput)

| batch | vanilla | EAGLE3 | SMC (prod, N=8) |
|------:|:---:|:---:|:---:|
| 1  | 252 (84.1) | **499 (84.1)** | 437 (65.9) |
| 32 | 2724 (82.9) | **3075 (83.5)** | 1792 (65.9) |

## bs=1 (single-stream latency) — best-vs-best, GSM8K

The headline comparison should use SMC's *best-accuracy* config, not its
best-speed one. At bs=1:

| method | accuracy | tok/s |
|--------|:---:|:---:|
| vanilla | 91.0 | 264 |
| **EAGLE3** | **92.5** | **509** |
| SMC — prod draft (β1.5+klmix0.5), N=4 | 79.0 | 460 |
| SMC — best draft (round-1 χ², math), N=8 | 84.4 | 441 |

**Important: the first SMC numbers were run WITHOUT the perf flags.** Enabling
`SMC_CYCLE_GRAPH=1 SMC_ENABLE_OVERLAP=1 SMC_DEFER_BONUS=1` (full-cycle CUDA graph
+ overlapped scheduler + deferred bonus, from the `bs1-deferred-cycle-graph`
work, merged here) gives ~+11–12% and closes the speed gap:

| method | accuracy | tok/s |
|--------|:---:|:---:|
| EAGLE3 | **92.5** | 509 |
| SMC renyi2 (math χ²) N=8 + perf | 83.6 | 491 |
| SMC prod (general) N=8 + perf | 82.4 | 480 |
| SMC prod N=4 + perf | 77.3 | **514** |
| *(SMC renyi2 N=8, no flags)* | *84.4* | *441* |
| *(SMC prod N=4, no flags)* | *79.0* | *460* |

**Revised bs=1 verdict:** with the perf flags, **SMC is speed-competitive with
EAGLE3** (491–514 vs 509 tok/s — faster at N=4, ~match at N=8). The original
"slower" finding was a missing-implementation artifact. The remaining gap is
**accuracy** (~84% best-SMC vs 92.5% lossless EAGLE3). Since SMC's accuracy
ceiling at α=1/T=0.7 is the target's own ~90%, more proposal finetuning (+ an
α>1 / lower-T sharpening sweep) can narrow — not erase — this gap; EAGLE3 is
already at the ceiling losslessly. (The 86.5% GSM8K figure quoted elsewhere is
the round-1 math χ² draft at N=8 on 200 q; ≈84% on 256 q, within noise.)

## Findings (honest)

1. **EAGLE3 is near-lossless** (≈ vanilla accuracy, accept length ≈ 4.6) and
   wins the latency regime (bs ≤ 32) on *both* speed and accuracy. At bs=1 it is
   ≈2× vanilla; SMC is ≈1.7× vanilla but ~13pp less accurate.
2. **SMC overtakes EAGLE3 on throughput at high concurrency** (bs ≥ 64: 4394 vs
   4278; bs=128: **5143 vs 4700, ≈ +9%**). EAGLE3's accept-based speedup
   plateaus under batching; SMC's dense batched draft keeps scaling — the SMC
   "throughput via arithmetic intensity" thesis holds *against EAGLE3*.
3. **But vanilla batched decoding beats both at high batch** (bs=128: 9738 tok/s,
   ≈2× either spec method). Speculative decoding trades compute for latency; at
   high throughput the GPU is already compute-bound, so the extra draft/verify
   (EAGLE) or N-particle (SMC) work is pure overhead. This is a general property,
   not specific to SMC.
4. **The accuracy gap is large, not small**: SMC ~78–79% vs EAGLE3 ~91–93% on
   GSM8K; 65.9% vs 83.5% on HumanEval. Finetuning lifts SMC a lot (base 66 → 84
   at N=8) but even the best SMC trails lossless EAGLE3. Raising N closes the gap
   but costs the throughput that is SMC's only edge.

## Verdict and when SMC could still win

For the **Qwen3 0.6B/8B pair on B200, SMC-SD does not present a favorable
speed/accuracy tradeoff vs EAGLE3**: EAGLE3 dominates the latency regime on both
axes, and although SMC is ~9% faster at bs≥64, it pays ~13–18pp accuracy and
both lose to vanilla at that batch size.

SMC-SD is plausibly competitive only outside this setting — directions to test
before claiming a win:
- **Large target / no good draft head**: EAGLE3 needs a trained, target-specific
  head; for very large or proprietary targets (e.g. 70B+) where heads are
  expensive/unavailable, SMC's cheap batched AR draft + the χ² finetuning recipe
  may be the better option. (Re-run this table for 1B→70B.)
- **Sampling/uncertainty use cases**: SMC returns an importance-weighted
  *posterior* of N samples (calibrated weights), not one sequence — valuable for
  best-of-N, self-consistency, or uncertainty, where EAGLE's single lossless
  stream is not directly comparable.
- **Larger/closer draft**: a 1.7B draft narrows the accuracy gap; whether it does
  so without erasing the throughput edge is open (lever #3).

The proposal-finetuning contribution (χ² objective) stands independently — it
substantially improves SMC-SD — but this comparison shows it does not, by
itself, make SMC-SD beat EAGLE3 for this model pair.
