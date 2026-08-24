# Semantic-only test-time scaling: verifier gate

## Outcome

The Qwen3.8-27B semantic verifier has real predictive signal and that signal
generalizes to a disjoint MATH-500 slice. A new long-form OlympiadBench pilot
also creates substantial selection headroom. However, exact-token checkpoints
show that the current pointwise score predicts problem difficulty much better
than it ranks sibling trajectories from the same problem. A frozen follow-up
separated validity from progress, added Qwen3-32B as a second verifier, and
combined all four signals out of fold. That experiment also failed the
within-problem actionability and matched-cost policy gates. Large external
semantic verifiers are therefore not a compute-competitive online allocation
mechanism for the current system.

No generator-likelihood score is used. The verifier distribution is computed
from the exact next-token logprobs of 20 ordered, one-token labels (A--T), then
renormalized within that set to obtain a continuous score in [0, 1]. Prefixes
are cut using the saved Qwen3.5-9B output token IDs, not by re-tokenizing text.

## Experiments

Both studies use eight Qwen3.5-9B samples per problem at temperature 0.7,
thinking disabled, and one Qwen3.8-27B semantic verifier. Confidence intervals
are 500-sample cluster bootstraps over problems.

### GSM8K train diagnostic (200 problems, 1,600 trajectories)

| Prefix | AUROC (95% CI) | Within-problem rank accuracy (95% CI) | Top-1 eventual correctness |
| --- | ---: | ---: | ---: |
| 25% | 0.738 [0.597, 0.858] | 0.675 [0.512, 0.818] | 96.5% |
| 50% | 0.844 [0.736, 0.943] | 0.825 [0.704, 0.944] | 97.5% |
| 75% | 0.859 [0.720, 0.976] | 0.865 [0.739, 0.974] | 98.0% |
| 100% | 0.886 [0.791, 0.977] | 0.937 [0.845, 0.993] | 97.5% |

This set is saturated: 1,538/1,600 trajectories are correct, only 14 problems
contain both correct and incorrect samples, and random top-half retention is
already 99.35%. Consequently it establishes early signal but cannot establish
useful pruning.

Self-consistency is 97.0%, terminal pointwise Best-of-8 is 97.5%, and oracle
pass@8 is 98.5%. Scoring all four checkpoints consumed 3,204,992 verifier
prompt tokens (500.8 per prefix) and 211.7 seconds on one B200, with 100%
selected-token-logprob coverage.

### MATH-500 hard diagnostic (first 100 problems, 800 trajectories)

The generation cap was raised to 2,048 after a 1,024-token probe exposed a
large truncation confound. The final diagnostic contains 653 correct and 147
incorrect trajectories; 22 problems contain mixed outcomes.

| Prefix | AUROC (95% CI) | Within-problem rank accuracy (95% CI) | Top-half lift over random (95% CI) | Top-1 correctness |
| --- | ---: | ---: | ---: | ---: |
| 25% | 0.851 [0.783, 0.908] | 0.615 [0.483, 0.750] | +1.46 pp [-0.46, +4.06] | 83.0% |
| 50% | 0.772 [0.669, 0.857] | 0.638 [0.506, 0.767] | -0.76 pp [-3.65, +1.92] | 83.0% |
| 75% | 0.763 [0.634, 0.857] | 0.671 [0.538, 0.804] | +2.57 pp [+0.66, +5.23] | 84.0% |
| 100% | 0.844 [0.742, 0.937] | 0.836 [0.736, 0.919] | +1.46 pp [-0.68, +3.92] | 85.0% |

Self-consistency is 89.0%, terminal pointwise Best-of-8 is 85.0%, and oracle
pass@8 is 90.0%. The verifier therefore ranks outcomes globally but its
pointwise terminal selection is four points worse than answer aggregation.
It is also overconfident (ECE 0.104--0.126).

Scoring all checkpoints consumed 2,797,725 verifier prompt tokens (874.3 per
prefix) and 185.3 seconds on one B200. The A--T labels accounted for 69.6% of
raw next-token probability mass on average, though all requested label
logprobs were returned.

## Terminal pairwise LLM-as-a-Verifier

The same Qwen3.8-27B verifier compared all 28 pairs among the eight terminal
solutions. Each pair was evaluated in both A/B orders, and the two canonical
win probabilities were averaged before candidates were ranked by mean
pairwise win probability.

| Selector | Accuracy | Difference from pairwise (paired 95% CI) |
| --- | ---: | ---: |
| Pointwise semantic Best-of-8 | 85.0% | +5 pp [+1, +10] |
| Self-consistency@8 | 89.0% | +1 pp [0, +3] |
| Order-swapped pairwise ranker | 90.0% | -- |
| Oracle pass@8 | 90.0% | -- |

Pairwise selection recovered a correct trajectory on all 90 problems where
one existed. It beat pointwise selection on five problems with no losses and
self-consistency on one problem with no losses. On the 213 mixed
correct/incorrect candidate pairs, its discrimination accuracy was 97.9%
[94.8%, 100%], assigning the correct candidate 72.3% probability on average.

This establishes that the semantic model is capable of distinguishing
solution quality; the weak terminal pointwise result came from the scoring
interface, not an absence of verifier knowledge.

The result is not a deployable policy. It required 5,600 calls and 11,887,820
verifier prompt tokens (2,122.8 per call), taking 168.6 seconds across four
B200 replicas. It also had severe first-position bias: the hard decision was
consistent across swapped orders for only 21.8% of pairs, and the mean
absolute order effect was 0.490. Order swapping and probability
symmetrization are therefore mandatory for this baseline.

### Cost-reduced terminal rankers

Two cheaper interfaces were evaluated on the identical 100 problems and
saved candidates. The knockout uses a fixed bracket over sample IDs and two
A/B orders for each of its seven matches. Because each match is independent,
its result can be computed exactly from the corresponding saved all-pairs
calls. The listwise ranker places all eight candidates in one prompt using a
deterministic random candidate order per problem and ranks their exact A--H
label-token probabilities.

| Selector | Accuracy | Calls/problem | Total verifier prompt tokens |
| --- | ---: | ---: | ---: |
| Self-consistency@8 | 89.0% | 0 | 0 |
| Pointwise terminal Best-of-8 | 85.0% | 8 | 992,090 |
| One-call randomized listwise | 88.0% | 1 | 802,810 |
| Order-swapped knockout | 90.0% | 14 | 2,967,802 |
| Order-swapped all-pairs | 90.0% | 56 | 11,887,820 |
| Oracle pass@8 | 90.0% | -- | -- |

The knockout is the useful result: it exactly preserves the all-pairs
correct/incorrect outcome on every problem and reaches oracle accuracy with
75% fewer calls and prompt tokens. It beats
pointwise Best-of-8 on five problems with no losses and self-consistency on
one problem with no losses.

Listwise selection beats pointwise Best-of-8 by 3 points (five paired wins,
two losses), but loses 1 point to self-consistency and 2 points to both
pairwise methods. Its 95% bootstrap interval for accuracy is [81%, 94%]. The
failure is dominated by label/position bias: although candidates were
randomized and A was correct on only 83% of problems, the renormalized ranker
selected A on 73/100 problems. The A--H labels contained only 39.3% of raw
next-token probability mass on average, and the model's unconstrained output
was outside A--H on 50/100 calls. The exact requested label logprobs were still
returned on every call, so 88% is the specified ranker's result rather than a
parsing artifact.

## Offline allocation ablation

This ablation prunes eight saved MATH trajectories at an exact prefix position
and completes only the top-m. It approximates one ParticleScale decision but
does not branch, replenish particles, or claim to be the online baseline.

| Policy | Accuracy | Generator tokens/problem | Verifier prompt tokens/problem |
| --- | ---: | ---: | ---: |
| Uniform majority@4 | 87.1% | 3,910 | 0 |
| Uniform majority@5 | 87.8% | 4,887 | 0 |
| 25%-prefix, keep 3 | 88.0% | 4,147 | 4,064 |
| 25%-prefix, keep 4 | 89.0% | 4,872 | 4,064 |
| Full self-consistency@8 | 89.0% | 7,821 | 0 |
| Agreement-adaptive control | 89.0% | 4,131 | 0 |

The semantic policy saves generator tokens, but a verifier-free agreement
control reaches the same accuracy with fewer generator tokens and no 27B
prompt-processing cost. This is the main reason not to integrate the current
score into the online SMC engine yet.

## Pairwise-informed pointwise verifier

The pairwise result motivated a new pointwise rubric that explicitly audits
substantive mathematical errors and refuses to reward confidence, polish, or
length. It uses five ordered A--E labels and still scores each prefix
independently: no sibling candidate and no generator likelihood is visible.

| Score | 25% AUROC | 25% within-problem rank | 25% top-1 | 100% AUROC | Terminal BoN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recoverability v1 | 0.851 | 0.615 | 83% | 0.844 | 85% |
| Error audit v1 | **0.893** | **0.737** | **86%** | **0.917** | **89%** |
| Pairwise-preference student (OOF) | 0.858 | 0.648 | 83% | 0.915 | 89% |

The error audit is the new best pointwise interface. At the terminal checkpoint
it recovers four problems lost by recoverability v1 with no losses, tying
self-consistency at 89% and falling one point short of the knockout/oracle.
Across all checkpoints it used 3,200 verifier calls and 2,954,525 prompt tokens
(923.3 per prefix). Exact A--E logprobs were returned on every call and the
labels contained 99.97% of raw next-token probability mass on average.

The teacher-distillation ablation fits a two-feature logistic pairwise ranker
separately at each checkpoint. Its inputs are only the two pointwise semantic
scores. Training and prediction use five problem-level folds, so none of a
problem's pairwise judgments train its own score. It matches the error audit at
the terminal checkpoint but is worse at early checkpoints, and doubles
verifier cost to 6,400 calls and 5,752,250 prompt tokens. This is an exploratory
cross-validation result, not a held-out test, and it does not justify keeping
the fusion.

The improved diagnostic scores still do not pass the allocation gate:

| Offline policy | Accuracy | Generator tokens/problem | Verifier prompt tokens/problem |
| --- | ---: | ---: | ---: |
| Error audit at 25%, keep 4 | 88% | 4,809 | 4,456 |
| Distilled score at 25%, keep 4 | 88% | 4,867 | 8,520 |
| Agreement-adaptive control | 89% | 4,131 | 0 |

Thus the prompt fix improves semantic discrimination substantially, but it
still does not make online semantic particle reallocation compute-competitive.

## Frozen disjoint validation (MATH-500 problems 100--199)

All prompts, models, generation settings, and the adaptive threshold were
frozen from problems 0--99 before evaluating this slice. The 800 new
trajectories contain 581 correct and 219 incorrect solutions: 64 problems are
all-correct, 19 are mixed, and 17 are all-wrong. The 2,048-token cap was reached
by 172 trajectories. Full generation used 886,068 output tokens and 57.0
steady-state seconds on one B200.

### Pointwise generalization

| Prefix | AUROC (95% CI) | Within-problem rank accuracy (95% CI) | Top-1 correctness |
| --- | ---: | ---: | ---: |
| 25% | 0.842 [0.760, 0.909] | 0.621 [0.514, 0.739] | 74% |
| 50% | 0.861 [0.785, 0.925] | 0.683 [0.563, 0.798] | 76% |
| 75% | 0.869 [0.800, 0.937] | 0.775 [0.688, 0.865] | 78% |
| 100% | 0.897 [0.809, 0.972] | 0.881 [0.794, 0.962] | 82% |

The semantic error audit therefore generalizes as a discriminator, but
terminal Best-of-8 is 82%, versus 83% for self-consistency and an 83% oracle.
All four checkpoints required 3,200 calls and 3,271,862 verifier prompt tokens.
Exact requested label-logprob coverage was 100%, with 99.96% mean probability
mass on A--E.

### Terminal knockout

The actual stage-wise knockout generated only the seven matches reached by
each bracket, in both A/B orders. It also scored 82%, with zero wins and one
loss against self-consistency. On its 59 mixed correct/incorrect matches,
accuracy was 83.1%, but hard A/B order agreement was only 20.7% and mean
absolute order effect was 0.503. It used 1,400 calls, 3,372,466 prompt tokens,
and 41.43 seconds across four B200s. Counting all four verifier GPUs plus the
generator gives 222.7 GPU-seconds, versus 57.0 for self-consistency alone.
The development-set knockout gain did not reproduce.

### Frozen allocation policies

The semantic adaptive threshold, 0.913739, is the median two-prefix score from
development problems 0--99. It was not recalibrated on the holdout. Randomized
figures are means over 500 candidate-order trials.

| Policy | Accuracy | Generator tokens/problem | Verifier prompt tokens/problem | Estimated steady-state GPU-seconds |
| --- | ---: | ---: | ---: | ---: |
| Full self-consistency@8 | 83.00% | 8,861 | 0 | 57.0 |
| Error audit at 50%, keep 4 | 83.00% | 6,602 | 7,073 | -- |
| Frozen semantic adaptive, fixed order | 83.00% | 7,027 | 1,210 | 51.6 |
| Frozen semantic adaptive, randomized order | 82.972% | 7,043 | 1,215 | 51.8 |
| Agreement-adaptive control | 82.91% | 5,276 | 0 | 33.9 |

The semantic adaptive policy saves 20.7% of generator tokens and an estimated
9.4% of total steady-state GPU time while matching full self-consistency in the
fixed order. Its randomized-order accuracy is effectively the same. But the
agreement-only control is within 0.062 percentage points of the randomized
semantic policy, uses 25% fewer generator tokens than it, and needs no 27B
verifier. The estimate combines directly measured component inference times
and scales them by observed token counts; it excludes initialization and is not
a directly timed dynamic serving run.

## Long-form OlympiadBench pilot

The harder benchmark is the official text-only, English, open-ended
competition-mathematics subset. The pilot uses 50 problems from a frozen
seed-0 permutation and eight Qwen3.5-9B thinking-enabled samples per problem.
Trajectories initially received 8,192 tokens; every length-capped trajectory
was then continued from its exact saved token IDs for up to another 8,192
tokens. The continuation is a conditional second sampling stage with seed 1,
not a fresh one-shot 16K sample.

| Metric | 8K outcomes | Continued 16K outcomes |
| --- | ---: | ---: |
| Sample accuracy | 13.75% | 32.25% |
| Self-consistency@8 | 26.0% | 48.0% |
| Oracle pass@8 | 38.0% | 56.0% |
| Mixed-outcome problems | 38.0% | 52.0% |
| Oracle minus self-consistency | 12 pp | 8 pp |
| Trajectories still at cap | 79.75% | 48.5% |
| Median completion tokens | 8,192 | 14,969.5 |

The continued set contains 129/400 correct trajectories, 26 mixed problems,
22 all-wrong problems, and two all-correct problems. It therefore passes the
predeclared benchmark eligibility gate and exposes a genuine 8-point selector
gap between self-consistency and the oracle. Generation consumed 5,058,513
output tokens and 616.1 steady-state GPU-seconds on one B200. The remaining
194 capped trajectories are a caveat, but are much less confounded than the 8K
labels.

### Why fractional prefixes were replaced

At 8K, error-audit AUROC rose from 0.721 at 10% to 0.803 at 50% and 0.873 at
the terminal checkpoint. Those fractional cuts are not an honest online
comparison: a 25% prefix is shorter for an eventually short solution than for
an eventually long one, so it leaks final trajectory length and gives
different particles different reasoning budgets. All allocation conclusions
below therefore use the same absolute generator-token horizon for every
trajectory. The saved verifier scores were relabeled after continuation, so
the fixed prefixes are evaluated against the cleaner 16K outcomes without
being rescored.

| Fixed checkpoint | AUROC (95% CI) | Within-problem rank (95% CI) | Top-1 eventual correctness |
| --- | ---: | ---: | ---: |
| 512 tokens | 0.681 [0.575, 0.777] | 0.477 [0.375, 0.581] | 32.0% |
| 1,024 tokens | 0.683 [0.579, 0.771] | 0.467 [0.375, 0.559] | 36.0% |
| 2,048 tokens | 0.722 [0.614, 0.818] | 0.456 [0.353, 0.585] | 30.0% |

The verifier separates easy from hard trajectories globally, but all three
within-problem intervals contain chance and their point estimates are below
0.5 at 1,024 and 2,048 tokens. At 2,048 tokens, retaining the top-scored half
preserves 89.3% of problems having any correct particle versus an 88.3% random
baseline: only a 1.0-point lift.

### Agreement-conditioned hybrid

The offline hybrid generates two complete solutions and stops without a
verifier call if their answers agree. On disagreement it scores both at one
fixed checkpoint, then either stops with the higher-scored solution or expands
to eight. Checkpoint, score aggregation, and threshold are selected inside
each training fold and evaluated on its held-out problem fold. Results average
500 candidate-order trials; confidence intervals are 5,000 problem-level
paired bootstraps.

| Policy | Accuracy | Generator tokens/problem | Verifier tokens/problem | GPU-seconds/problem |
| --- | ---: | ---: | ---: | ---: |
| Agreement-adaptive control | 47.02% | 86,044 | 0 | 10.480 |
| Semantic hybrid, 90% budget | 40.94% | 76,273 | 1,414 | 9.363 |
| Semantic hybrid, matched budget | 46.18% | 84,632 | 1,884 | 10.406 |

At the 90% budget, semantics saves 1.117 GPU-seconds/problem but loses 6.084
accuracy points (95% CI: -10.896 to -2.040). At matched compute it loses 0.836
points (95% CI: -2.436 to 0.000) and its compute difference is statistically
unclear. Even an optimistically biased policy selected and evaluated on the
same 50 problems reaches 47.024%, only 0.004 points above the control. This is
strong evidence that the current pointwise score is not useful for the middle
allocation decision, despite its global AUROC.

### Equal-horizon relative-verifier screen

The next frozen screen directly compared every correct/incorrect sibling pair
on the 26 mixed problems. At each of 512, 1,024, and 2,048 generator tokens,
there are 287 canonical pairs. Each was judged in both A/B orders and the
canonical probabilities were averaged. This is a label-selected diagnostic,
not a deployable selector.

The predeclared gate required problem-balanced accuracy of at least 60% and a
problem-cluster-bootstrap lower bound above 50%.

| Checkpoint | Relative verifier (95% CI) | Pointwise error audit | A/B order agreement |
| --- | ---: | ---: | ---: |
| 512 tokens | 42.53% [31.46%, 53.69%] | 46.04% | 52.96% |
| 1,024 tokens | 39.58% [30.33%, 49.02%] | 45.89% | 65.85% |
| 2,048 tokens | 50.12% [38.32%, 62.18%] | 50.81% | 70.73% |

All three checkpoints fail the gate, and none improves on the identical-prefix
pointwise score. The relative-only versus pointwise-only correct decisive-pair
counts are 47 versus 65 at 512, 30 versus 55 at 1,024, and 34 versus 36 at
2,048 tokens. Thus there are different errors, but no evidence that this
relative interface adds net ranking information.

Position bias remains material. Candidate A is selected on 73.69%, 64.98%,
and 55.05% of raw calls as the horizon grows, so order swapping cannot be
removed. Exact A/B logprobs were returned for all 1,722 calls and contained
91.13% of raw next-token probability mass on average. The screen used
4,548,510 verifier prompt tokens and 56.02 inference seconds across four B200s,
or 224.07 allocated verifier GPU-seconds.

At 1,024 tokens the verifier is significantly anti-predictive. This is not a
label-mapping bug: the saved generated choices, exact token probabilities, and
canonical orientation mapping agree. The effect is strongest when the
incorrect trajectory later reaches the 16K cap: problem-balanced accuracy is
32.36% [18.04%, 47.89%], versus 42.92% [29.74%, 56.38%] against incorrect
trajectories that stop naturally. At 512 tokens, the naturally stopped subset
is 50.78% [36.22%, 65.34%], still only chance. The judge tends to prefer
plausible, detailed paths that make insufficient progress and later exhaust
their budget.

Inverting the 1,024-token choice after seeing the labels would yield 60.42%
[50.98%, 69.67%]. This is a post-hoc observation, not a valid policy or a
passed gate. It must not be used without freezing that hypothesis and testing
it on a disjoint set.

### Multi-verifier validity/progress screen

The final frozen screen tested the proposed remedy directly. Two independent
semantic rubrics were used:

- **validity:** probability that the visible reasoning contains no substantive
  mathematical error;
- **progress:** probability that the same generator can reach a complete boxed
  answer within the fixed remaining 16,384-token budget, explicitly independent
  of whether that answer will be correct.

Both Qwen3.8-27B and Qwen3-32B scored all 400 trajectories at 512, 1,024, and
2,048 generator tokens. No generator likelihood was exposed. Five-fold
problem-level out-of-fold pairwise logistic rankers combined scores without
training on the problem being evaluated. The primary 1,024-token gate required
the all-four ensemble to reach at least 60% problem-balanced sibling ranking,
have a problem-bootstrap lower bound above 50%, and exceed every single score.

| 1,024-token score | Problem-balanced correct/incorrect sibling rank (95% CI) |
| --- | ---: |
| Qwen3.8 validity | 45.89% [36.20%, 56.15%] |
| Qwen3.8 progress | 46.74% [35.98%, 57.80%] |
| Qwen3-32B validity | 41.72% [32.12%, 51.46%] |
| Qwen3-32B progress | 53.02% [41.35%, 63.84%] |
| Qwen3.8 validity + progress | 47.28% [36.54%, 58.38%] |
| Qwen3-32B validity + progress | 55.66% [46.22%, 65.60%] |
| Both validity scores | **56.60% [46.46%, 66.31%]** |
| Both progress scores | 43.79% [32.63%, 55.27%] |
| All four scores | 52.52% [41.91%, 63.93%] |

The all-four ensemble fails all three primary-gate conditions. At 512 tokens
it reaches 57.83% [47.15%, 68.17%]. The strongest exploratory single result is
Qwen3-32B progress at 2,048 tokens, 61.24% [49.39%, 72.85%], but its lower bound
still crosses chance and the corresponding online decision simulation is
negative.

The Qwen3-32B progress hybrid scores two disagreeing prefixes and chooses
whether to stop or expand from two to eight trajectories using five-fold
cross-validation. At a 90% compute target it scores 42.11% versus 47.02% for
the verifier-free agreement control, saving 1.097 GPU-seconds/problem but
losing 4.908 accuracy points [−9.332, −1.072]. At the matched-compute setting
it scores 44.62%, losing 2.400 points [−5.656, 0.000] while saving only 0.285
GPU-seconds/problem. Even the optimistically biased in-sample upper bound is
46.30%, below the control.

This was also not a free signal. The four features required 4,800 calls and
7,361,690 prompt tokens for the 1,200 saved prefixes. Their measured inference
cost totals 396.70 allocated B200-seconds. At the primary checkpoint, scoring
two live particles with all four features costs an estimated 0.589 allocated
GPU-seconds: 19.1% of generating two average complete trajectories, or 5.6% of
the full agreement-control query cost. The estimate linearly scales measured
inference time by prompt tokens and excludes initialization and online queueing.

### True online ParticleScale and deterministic forking

The final development experiment performs real post-checkpoint branching
rather than replaying fixed completed trajectories. Every method starts from
the same eight saved Qwen3.5-9B prefixes at exactly 2,048 generator tokens.
After each 2,048-token interval, Qwen3.8-27B assigns the frozen validity score.
Semantic SMC applies the incremental update
`12 * (S_t - S_{t-1})`, resampling systematically when ESS is below `0.75N`.
The deterministic control keeps the highest-scored half and independently
forks each survivor twice. All continuations after the common initial prefix
are newly sampled online, and no generator likelihood is used.

The terminal conditions use the same original eight complete trajectories.
Pointwise Best-of-8 uses exactly the same validity score. The 14-call knockout
uses order-swapped A/B comparisons as the cost-reduced LLM-as-a-Verifier-style
terminal ranker. Accuracy intervals and differences below are 5,000
problem-level paired bootstraps. Active component cost is the favorable sum of
measured model inference GPU-seconds; it excludes initialization.

| Method | Accuracy (95% CI) | Generator tokens/problem | Verifier prompt tokens/problem | Active GPU-s/problem |
| --- | ---: | ---: | ---: | ---: |
| Self-consistency@8 | **48% [34%, 62%]** | 101,170 | 0 | **12.32** |
| Terminal pointwise semantic Best-of-8 | 40% [26%, 54%] | 101,170 | 103,877 | 18.59 |
| Terminal order-swapped knockout | 14% [4%, 24%] | 101,170 | 360,557 | 43.72 |
| Semantic SMC | 46% [32%, 60%] | 100,393 | 533,184 | 46.77 |
| Deterministic semantic fork | 40% [26%, 54%] | 96,519 | 425,058 | 35.27 |
| Original-pool oracle pass@8 | 56% [42%, 70%] | -- | -- | -- |

Semantic SMC loses 2 points to self-consistency [−12, +8], with three paired
wins and four losses. It gains 6 points over terminal pointwise selection
[−4, +16] and 6 over deterministic forking [−8, +20], neither significant.
It is strictly active-cost dominated by self-consistency: 3.80 times the cost
for a lower point estimate. If all four resident generator/verifier GPUs are
charged even while idle in this staged harness, its static allocation rises to
108.69 GPU-seconds/problem.

SMC preserves substantially more diversity than hard forking. It resamples on
102/350 problem-checkpoint events, retains 5.36 original roots and 7.92 unique
final token sequences per problem, and obtains 58% oracle pass@8 in its newly
branched pool. But its own terminal semantic selector converts that headroom to
only 46%; majority voting the same final particles gives 48%. Hard forking
collapses to 1.22 original roots and 4.10 unique final sequences, with only 44%
oracle accuracy. Stochastic resampling therefore preserves diversity, but the
semantic score still cannot spend or select that diversity productively.

The terminal results reinforce the interface failure. Pointwise terminal
AUROC is 0.799 and sibling ranking accuracy is 66.6%, yet Best-of-8 is only
40% because 251/400 trajectories fall in the 0.9--1.0 score bin while only
49.4% of that bin is correct (ECE 0.530). The order-swapped knockout is worse:
correct/incorrect match accuracy is 26.6%, it loses 17 problems to
self-consistency with no wins, and consumes 18,027,842 verifier prompt tokens.

The predeclared online gate fails. This is direct evidence, not an offline
proxy: intermediate semantic SMC, deterministic semantic forking, terminal
pointwise Best-of-N, and terminal pairwise LLM-as-a-Verifier all lose the
quality/cost comparison to verifier-free self-consistency on this development
set.

### Optimized likelihood SMC and terminal likelihood-semantic hybrid

The follow-up ran the repository's optimized likelihood-only `SMCEngine` on
the same 50 problems with Qwen3.5-9B as target, Qwen3.5-2B as draft, `N=8`,
`gamma=8`, target and draft temperature 0.7, power `alpha=1`, ESS threshold
`0.5N`, seed 0, and a 16,384-token cap. Four SMC groups ran concurrently in a
fixed CUDA-graph shape covering 32 particles. The engine returned all final
particles and likelihood weights, so posterior sampling, majority,
likelihood-weighted answer voting, maximum weight, and pool oracle use the
same online run.

The terminal hybrid then scored those 400 returned particles once with the
frozen Qwen3.8-27B validity verifier. Before scoring, `beta` was frozen to
`{0,1,2,4,8,12,16}`. The exact terminal combined weight was
`log_w_likelihood + beta * S_terminal`; both particle argmax and combined
answer-cluster voting were evaluated. This is an exact L1+S objective with one
semantic checkpoint at termination, not periodic intermediate semantic
resampling.

| Method | Accuracy (95% CI) | Active GPU-s/problem |
| --- | ---: | ---: |
| Self-consistency@8 | **48% [34%, 62%]** | **12.32** |
| Likelihood SMC posterior sample | 24% [14%, 36%] | 13.38 |
| Likelihood SMC particle majority | 26% [14%, 38%] | 13.38 |
| Likelihood-weighted answer majority | 26% [14%, 38%] | 13.38 |
| Maximum likelihood weight | 24% [12%, 36%] | 13.38 |
| Terminal semantic argmax on likelihood pool | 26% [14%, 38%] | 18.53 |
| Combined weighted answer, every tested nonzero beta | 26% [14%, 38%] | 18.53 |
| Likelihood-pool oracle pass@8 | 26% [14%, 38%] | -- |

The best combined rule loses 22 points to self-consistency, with paired 95%
CI `[−34, −10]`: one win, twelve losses, and 37 ties. Every semantic beta
produces the same answer-level result as likelihood majority, so the verifier
adds 5.14 active GPU-seconds/problem and 5,308,913 prompt tokens without
changing one outcome. Its within-pool terminal AUROC is 0.780 and sibling
ranking accuracy is 0.714, but there is no selectable headroom: terminal
semantic argmax, majority, and the pool oracle are all 26%.

The failure is proposal/diversity collapse rather than a weak terminal rule.
Only 96/400 likelihood-SMC particles are correct (24%), versus 129/400
independent target samples (32.25%). Of 50 SMC groups, 37 contain zero correct
particles, eleven contain eight, one contains seven, and one contains one.
The pool retains only 4.56 unique final token sequences/problem. Mean final
ESS is 6.37, but that tail statistic is reset after resampling and therefore
does not recover ancestry already discarded earlier in decoding.

This development configuration is strictly dominated by self-consistency in
both accuracy and active accelerator cost. Periodic likelihood-plus-semantic
resampling with the same 27B scorer is not justified: the earlier prefix gate
is negative, the external scorer is expensive, and the likelihood population
already loses the correct roots that a terminal factor would need. First fix
draft adequacy and ancestry survival with likelihood only.

## Decision and project direction

Do not integrate the tested 27B/32B external semantic verifiers into online
SMC and do not spend a new 50-problem disjoint set on this family. Pointwise,
relative, validity/progress, multi-model fusion, and now true branched online
resampling have all failed a predeclared gate. The actual online method is
strictly dominated by self-consistency, while the earlier adaptive policy loses
to a zero-verifier agreement control. More prompt engineering on these same
labels would be development-set tuning, not new evidence.

Keep the semantic code and results as a reusable diagnostic and negative
baseline. Keep the order-swapped knockout as the paper-style terminal
LLM-as-a-Verifier baseline, but not as the serving policy. Self-consistency@8
is the best current quality/cost operating point. The first optimized
likelihood-SMC configuration is also a no-go: 26% at 13.38 active
GPU-seconds/problem versus 48% at 12.32 for self-consistency.

The next milestone is not another verifier prompt or a periodic hybrid. Run a
proposal-adequacy grid over draft size, `gamma`, and ESS threshold while
recording root ancestry at every resample. Require the likelihood pool oracle
and particle accuracy to recover the independent-target baseline before doing
another serving or composite-objective sweep. In particular, ablate the 2B
and 4B drafts, `gamma in {4,8}`, and thresholds `{0,0.25,0.5}` at `N=8` on
this development set. Promote only configurations that preserve diversity;
then freeze one configuration for disjoint testing and matched-load HTTP
serving.

For online allocation, first use signals that are already in the generation
loop: answer agreement, ESS, ancestry collapse, completion state, and cheap
deterministic progress features. In parallel, wire the measured cascade-decode
kernel into dual CUDA graphs for 8K+ contexts. Only reopen semantic checkpoints
if a scorer can be made an in-engine marginal cost—target self-scoring with KV
forking—or distilled to a small model and then passes the same frozen gate.
Training such a scorer is a deliberate scope change, not a continuation of the
current no-training experiment.

The disjoint MATH protocol is frozen in
`configs/semantic/semantic_verifier_math500_holdout_v1.json`; the long-form
pilot is frozen in
`configs/semantic/semantic_verifier_olympiadbench_pilot_v1.json`; the relative
screen is frozen in
`configs/semantic/prefix_pairwise_olympiadbench_screen_v1.json`; the
multi-verifier screen is frozen in
`configs/semantic/multiverifier_progress_olympiadbench_v1.json`; the true online
development comparison is frozen in
`configs/semantic/online_particlescale_olympiadbench_dev_v1.json`; the
likelihood and terminal-combination follow-up is frozen in
`configs/semantic/likelihood_semantic_hybrid_olympiadbench_dev_v1.json`.
Configurations remain under `configs/semantic/`; MATH development artifacts
are in `work_dirs/semantic_verifier_v1/`, MATH holdout artifacts are in
`work_dirs/semantic_verifier_holdout_v1/`, and long-form artifacts are in
`work_dirs/semantic_olympiadbench_pilot_v1/`. Generated artifacts are
intentionally git-ignored.
