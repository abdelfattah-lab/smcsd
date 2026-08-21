# TTS: Test-Time-Scaling Serving with Composite SMC Objectives

Status: project roadmap — revised 2026-08-21 (CUDA-graph Qwen3.5 agent serving, matched target AR, frozen Terminal-Bench dev matrix, and joined agent/server metrics complete; semantic scoring not started)

Branch: `tts`  
Starting point: the existing one-draft, one-target SM-CSD implementation

## 1. Project thesis

Build a fast serving system for test-time scaling that allocates generation compute online with Sequential Monte Carlo (SMC). The system should support several inference objectives without requiring model training:

1. one target model's likelihood;
2. multiple target models' likelihoods;
3. one or more semantic verifier scores;
4. likelihood and semantic scores together.

The central paper result should be:

> At matched task accuracy and on the same hardware budget, the proposed system serves more queries per second, has lower time per query, or produces more correct answers per GPU-hour than strong test-time-scaling baselines.

Raw tokens per second is a diagnostic metric, not the main result. Different methods may generate different numbers of draft, target, and verifier tokens, so useful work must be measured at the query level and all accelerator time must be counted.

This is primarily an MLSys project. The algorithmic contribution is a general composite SMC objective and a scheduling policy; the core contribution is making it efficient through KV-cache sharing, batched checkpoints, duplicate-prefix compaction, fused weight/resampling operations, fixed execution shapes, and cross-request pipelining.

> **Scope in one sentence:** online compute reallocation for test-time scaling
> via SMC, made cheap by KV-fork-based particle lineage and zero-copy semantic
> checkpoints. Multi-likelihood aggregation is an ablation unless the Phase 2
> offline evidence promotes it; cross-tokenizer aggregation is future work.

## 2. Scope and initial assumptions



### In scope

- Inference-only test-time scaling.
- One draft model `q`.
- One likelihood target model `p_1`; additional same-tokenizer targets are
  evidence-gated by the Phase 2(c) complementarity study.
- Zero or more semantic scorer models `v_j`, with two implementations:
  KV-fork self-scoring (design A) and an external scorer engine (design B) —
  see Section 5.
- An independently configurable speculative block size and semantic-checkpoint interval.
- Online particle reallocation using ESS and resampling.
- Offline and online evaluation on reasoning tasks.
- Single-node, multi-GPU serving first; distributed execution only after the single-node design works.



### Initially out of scope

- Training or fine-tuning a verifier, policy, or reward model.
- Reinforcement learning.
- Cross-tokenizer likelihood aggregation.
- Tool execution or code execution as a semantic signal.
- Fully adaptive, learned scheduling in the first implementation.
- Mixing this project with the separate megakernel research branch.

All likelihood models in the first version must use an identical token-ID mapping, vocabulary, special tokens, EOS behavior, and compatible chat template. Checking vocabulary size alone is insufficient.

## 3. What already exists

The current SMC-SD path is already the first experimental condition:

- one draft model proposes tokens;
- one target model verifies blocks;
- particle log weights receive the target-versus-draft likelihood ratio;
- ESS controls resampling;
- duplicated particles share history and KV state through the existing lineage machinery;
- the target contributes a bonus token after each draft block;
- final output can be sampled from the normalized particle weights.

Therefore, “one draft + one target + likelihood only” should be preserved as an exact compatibility mode, not rewritten from scratch.

Important integration points:

- `smcsd/engine.py`: public engine construction and request/output API.
- `smcsd/core/worker.py`: draft generation, target verification, and likelihood updates.
- `smcsd/core/scheduler.py`: worker orchestration, ESS, and resampling.
- `smcsd/core/req_state.py`: per-particle tokens, weights, histories, and ancestry.
- `smcsd/core/kernels/fused_collect.py`: normalization, ESS, compaction, and systematic resampling.
- `smcsd/core/kernels/fused_resample_kv.py`: KV and block-table updates after resampling.
- `smcsd/mem_cache/allocator.py`: `SMCRefCountedTokenAllocator` and
  `copy_block_table` — refcounted KV sharing and zero-copy block-table
  forking; the machinery design A builds on.

### Engine facts to plan around (v0.5.17 review, 2026-08-19/20)

- The KV co-budget configurator is dead code at HEAD (review finding F5): the
  target silently gets stock sizing and ignores the computed target+draft
  split. Prerequisite fix for any multi-model memory plan (Phase 0).
- Decode-dispatch observability is available through `SMC_GRAPH_STATS`; the
  Terminal-Bench launcher exposes it and a B200 tool-contract run reported
  eight `cycle_graph` hits in eight cycles with no fallback.
- Fixed-seed determinism holds in the sequential event loop (verified
  seed-for-seed on B200); the overlap loop intentionally shifts RNG
  consumption — golden tests must pin the loop mode.
- SMC runs only through `SMCEngine` (offline) and `smcsd.http_server`
  (server); `sgl.Engine(speculative_algorithm="SMC")` deliberately raises on
  the vendored tree. The Phase 8 baseline harness must respect this.
- The current branch contains the earlier hybrid-state fixes and now restores
  the real full-attention layer map for dense AR draft workers. Live B200
  Qwen3.5-2B/Qwen3.5-9B runs pass streamed tool calls and capture target,
  draft, and deferred-cycle CUDA graphs; runtime logs confirm graph dispatch.



## 4. Objective definitions

Use the following model roles consistently:

- `q`: draft/proposal model.
- `p_m`: likelihood target model `m`.
- `v_j`: semantic verifier/scorer `j`.
- `gamma`: number of draft tokens proposed per speculative cycle.
- `H`: semantic checkpoint interval measured in newly generated tokens. It is independent of `gamma`.
- `N`: number of particles.

For a complete response `y` to prompt `x`, define the combined unnormalized target:

`target(y | x) proportional to product_m p_m(y | x)^alpha_m times exp(sum_j beta_j S_j(x, y))`

For the first multi-likelihood implementation:

- require `alpha_m >= 0`;
- normalize `sum_m alpha_m = 1`;
- treat `beta_j` as calibrated semantic temperatures;
- retain each factor separately in logs for debugging and ablations.

For a proposed token block `b`, the incremental likelihood weight is:

`Delta_L = sum_m alpha_m log p_m(b | prefix) - log q(b | prefix)`

At a semantic checkpoint, use the score difference:

`Delta_S = sum_j beta_j [S_j(x, new_prefix) - S_j(x, previous_scored_prefix)]`

The total incremental log weight is `Delta_L + Delta_S`. The score difference is necessary so repeated checkpoints do not repeatedly count the same accumulated semantic evidence.

### Objective modes


| Mode                           | Distribution being approximated               | Incremental weight                                     |
| ------------------------------ | --------------------------------------------- | ------------------------------------------------------ |
| L1: one likelihood model       | `p_1(y                                        | x)`                                                    |
| LM: multiple likelihood models | `product_m p_m^alpha_m`                       | `sum_m alpha_m log p_m - log q`                        |
| S: semantic only               | `q(y                                          | x) exp(sum_j beta_j S_j)`                              |
| L1+S                           | `p_1(y                                        | x) exp(sum_j beta_j S_j)`                              |
| LM+S                           | `product_m p_m^alpha_m exp(sum_j beta_j S_j)` | multi-model likelihood ratio plus semantic differences |


The semantic-only mode deliberately uses `q` as its base distribution. Setting all likelihood coefficients to zero would otherwise imply an unintended uniform sequence prior.

### Required compatibility invariant

With one likelihood target, no semantic scorer, `alpha_1 = 1`, and `beta = 0`, the new implementation must match current SM-CSD within numerical tolerance and fixed-seed sampling tolerance.

## 5. Semantic scorer design



### First scoring criterion

The first criterion should ask the verifier for:

> The probability that this partial response can still be completed into a fully correct solution, accounting for any irreversible logical, factual, or arithmetic errors already present.

This is better than asking whether an unfinished prefix is currently “correct.” It estimates recoverability and can kill particles that have entered unrecoverable states while retaining incomplete but promising work.

The scorer prompt must clearly delimit:

- the original problem;
- the partial response;
- that the response is unfinished;
- the criterion;
- an ordered score scale;
- the required single scoring-token answer.

Keep the template order problem → partial response → criterion → scale →
answer format: the growing text then stays a cache/KV *prefix* (radix reuse
in design B, literal KV reuse in design A) and only the short fixed tail
re-prefills at each checkpoint. Do not move instructions ahead of the partial
response without re-measuring both scorer cost and score quality (Phase 2b).

Use expected score under the scoring-token probability distribution, following LLM-as-a-Verifier, rather than parsing only the argmax label. Start with one evaluation and either 5 or 20 ordered labels. Later ablate:

- score granularity;
- repeated evaluations;
- criteria decomposition;
- scorer model identity;
- semantic temperature `beta`.

Possible later criteria:

- recoverability/can still finish correctly;
- probability of a fatal error;
- correctness of the reasoning so far;
- progress toward a complete answer;
- compliance with task-specific constraints.



### Configurability

The API should allow:

- zero, one, or multiple scorer model paths;
- a weight per scorer;
- a prompt template per criterion;
- a set of ordered scoring tokens and their numeric values;
- `semantic_checkpoint_interval_tokens` independent of `gamma`;
- semantic temperature per criterion/scorer;
- calibration parameters;
- semantic batching policy;
- final answer-selection policy.

A proposed initial configuration shape:

```yaml
draft_model: ...
likelihood_targets:
  - model: ...
    alpha: 1.0
semantic_scorers:
  - model: ...
    beta: 0.5
    criterion: recoverability
    score_tokens: ["1", "2", "3", "4", "5"]
gamma: 8
semantic_checkpoint_interval_tokens: 64
num_particles: 32
ess_threshold: 0.5
final_selection: posterior_sample
```



### Semantic checkpoint behavior

Start with fixed checkpoints, for example `H in {32, 64, 128}`. Reasoning-step boundaries such as double newlines can be evaluated later, but fixed intervals are easier to batch, benchmark, and capture in CUDA graphs.

At each checkpoint:

1. finish the current likelihood verification cycle;
2. optionally perform likelihood-only ESS/resampling;
3. identify unique live prefixes after ancestry compaction;
4. score only those unique prefixes;
5. scatter scores back to their particles;
6. apply score differences to log weights;
7. recompute ESS and resample if needed;
8. preserve the latest score for every criterion through ancestry changes.

This staged update is useful because it avoids scoring doomed or duplicated particles.

### Two scorer implementations

**Design A — KV-fork self-scoring (default candidate).** The scorer is a
model that already holds the context's KV (the likelihood target, or the
draft). A checkpoint then costs: fork the particle's block table (zero-copy —
the same refcount machinery as particle cloning), append a **fixed** scoring
suffix as token IDs, run one short extend (the same operation shape as
TARGET_VERIFY, which the engine executes every cycle), and read the
score-token logits at the final position. No detokenization, no
retokenization, no second engine; a fixed-length suffix means fixed shapes,
which means the checkpoint can join the existing CUDA-graph families. Cost ≈
suffix-length prefill × unique prefixes per checkpoint. Constraints: scorer
choice is limited to in-engine models, and suffix-scoring an unfinished
assistant turn is off-distribution relative to the clean re-prompted form —
Phase 2(b) must validate A's score quality against B before A becomes the
default. If it holds, zero-copy semantic checkpoints are a headline systems
feature no external-verifier baseline can replicate cheaply.

**Design B — external scorer engine (quality reference; arbitrary models).**
A separate *stock* sglang engine with radix cache ON (the SMC engine forces
radix off, so the scorer cannot live inside `SMCScheduler`), fed detokenized
context over an explicit queue. Cost is the incremental prefill of the
growing context (radix reuse) plus the host round trip (detokenize →
template → retokenize) every checkpoint. This is the faithful reproduction of
published verifier setups and the quality yardstick for A.

### Final answer selection

Keep sampling and search semantics separate:

- `posterior_sample`: sample a final particle from normalized SMC weights;
- `map_objective`: choose the particle with the highest explicitly stored target objective;
- `best_semantic`: choose the particle with the highest terminal semantic verifier score.

Never label “largest importance weight” as the best semantic answer. Importance weights correct a proposal distribution; they are not themselves the final quality score.

## 6. Multiple likelihood targets

Each likelihood target must score the same chosen token block. Aggregate only selected-token log probabilities or their weighted sum; do not communicate full-vocabulary logits between GPUs.

The existing target bonus token creates a correctness question when there are multiple targets. Resolve it before claiming the LM or LM+S objective.

Recommended first design:

1. designate one target as the bonus-token proposal;
2. sample the bonus token from that model;
3. broadcast only the selected token ID;
4. have every other likelihood target score the selected token;
5. apply the correct proposal-ratio term for that token.

Implement a slow reference version and verify it by enumeration on a tiny vocabulary. If this path proves too complex initially, add a no-bonus/all-draft proposal mode for the first multi-target correctness prototype. Do not silently reuse a one-target correction under a multi-target proposal.

## 7. Proposed software architecture



### New modules

- `smcsd/core/composite_target.py`: factor definitions and reference log-weight equations.
- `smcsd/core/target_group.py`: one logical interface for multiple likelihood workers.
- `smcsd/core/semantic_scoring.py`: semantic requests, batching, score extraction, and calibration.
- `smcsd/core/score_templates.py`: versioned criteria and prompt templates.

Keep the first implementation in PyTorch/eager code. Optimize only after objective-level tests pass.

### Request state additions

Store, per particle:

- total log importance weight;
- likelihood energy for each target;
- semantic score and previous checkpoint score for each scorer/criterion;
- accumulated semantic energy;
- composite objective value;
- last semantic checkpoint length;
- ancestry/lineage ID;
- finished/EOS state.

Do not collapse these fields into one scalar. Separate accounting is required for debugging, calibration, and objective ablations.

### Model and KV ownership

Every model has its own KV cache and block table. A shared logical lineage identifies a particle across models, while per-model physical blocks use reference counting and copy-on-write behavior.

On resampling:

- update logical ancestry once;
- apply the same ancestry map to every active model;
- increment/decrement per-model KV references safely;
- compact duplicate prefixes before verifier work;
- ensure finished particles are not accidentally advanced or rescored.



### Scheduling

Use two graph families:

- a frequent fixed-shape likelihood cycle for draft + target work;
- a less frequent semantic-checkpoint cycle.

Across requests, pipeline likelihood generation for one microbatch with semantic scoring for another when GPU placement permits. Add:

- queue backpressure;
- maximum verifier queue age;
- fair scheduling across requests;
- a global KV-memory admission limit;
- explicit accounting for idle but reserved accelerators.

Adaptive scoring can be added after the fixed schedule works. If semantic calls are skipped based on a heuristic uncertainty rule, report it as a search/scheduling heuristic unless its effect on the target distribution is formally corrected.

### Fused fast path

After the eager implementation is correct, build or extend fused operations for:

- selected-token log-probability extraction;
- weighted aggregation across likelihood factors;
- semantic delta addition;
- log-weight normalization;
- ESS and degeneracy checks;
- systematic resampling;
- ancestry compaction;
- zeroing interval weights;
- per-model block-table/KV remapping.

Cross-GPU traffic should be token IDs, scalar factor contributions, scores, and ancestry maps—not logits or KV tensors.

## 8. Implementation phases

Ordering principle: three cheap offline studies (Phase 2) pick the build
order before any major engine work, and the first online experiment
(Phase 3) decides the paper's central claim before optimization begins.

### Phase 0 — Freeze the experiment contract + engine prerequisites

**Decisions recorded 2026-08-20:**

- **Model family: Qwen3 dense.** Draft candidates: Qwen3-0.6B / Qwen3-1.7B.
  Target ladder: Qwen3-4B → 8B → 14B → 32B, one B200 each (the 8×B200 node
  makes target-size scaling a first-class study axis alongside `N` and
  `gamma`). Scorer candidate #1 is the target itself (design A); external
  candidates from the same family (design B). Hybrid Qwen3.5 deferred until
  F4/F7 land. Qwen3 thinking/non-thinking mode must be fixed once in the
  prompt contract and held constant across all methods.
- **First paper claim (recorded):** at matched task accuracy with strong
  TTS baselines on identical hardware, the system serves faster — higher
  QPS / lower p95 latency at matched accuracy, equivalently more correct
  answers per GPU-hour — with every allocated accelerator counted. Raw TPS
  stays a diagnostic (Section 1).
- **Scaling studies:** target size × `N` × `gamma` sweeps are part of the
  main characterization (Phase 6), not an afterthought.

- [x] Choose the primary draft/target/scorer model family and GPU topology
      (Qwen3 dense, above; tokenizer identity across sizes still needs the
      programmatic check below).
- [ ] Validate exact tokenizer compatibility for likelihood models.
- [ ] Define prompt templates and answer extraction once.
- [ ] Define what accelerator time includes.
- [ ] Choose development and held-out test splits.
- [ ] Record dependency versions, model revisions, seeds, and hardware.
- [x] Add a run manifest and machine-readable result schema
      (`configs/terminal_bench/likelihood_dev_v1.json`; Harbor result plus
      `experiment.json` and Prometheus snapshots).
- [ ] Decide the first paper claim and go/no-go thresholds before optimizing.
- [ ] Fix and validate the KV co-budget configurator (review finding F5) —
      every multi-model memory plan below depends on it.
- [x] Expose and validate the `SMC_GRAPH_STATS` dispatch counters, including a
      configurable debug print interval for short contract tests.
- [ ] Pin golden fixed-seed tests to the sequential event loop.

Exit criterion: one written experiment contract applies unchanged to every
method, and the engine prerequisites are merged.

### Phase 1 — Reproduce, instrument, and test draft adequacy

- [x] Add a turn-local Pi/Terminal-Bench 2.1 harness through the OpenAI chat
      server, including automatic streamed tool-call and tool-history contract
      validation (`docs/terminal_bench.md`).
- [x] Run the first B200 `fix-git` correctness smoke at `N={1,4,8}` with
      Qwen3.5-2B/Qwen3.5-9B: rewards `{0,0,1}`. This is a harness milestone,
      not statistical evidence.
- [x] Freeze a 12-task Terminal-Bench 2.1 development set, three seeds, and the
      AR plus `N={1,4,8,16}` × `gamma={2,4,8}` likelihood-only matrix.
- [ ] Add the self-consistency control to the same manifest and runner.
- [x] Fix and validate Qwen3.5 draft CUDA-graph capture on B200, including
      target/draft/cycle capture, eight-of-eight cycle-graph dispatch, and
      streamed tool calls.
- [x] Add a true target-only stock-SGLang AR launcher with matched agent API,
      seed, metrics, and request limit; validate it end to end on B200.
- [ ] Run one draft + one target + likelihood-only on a small reasoning set.
- [ ] Confirm deterministic fixed-seed repeatability where expected
      (verified 2026-08-19 in the sequential loop).
- [ ] Record exact-match accuracy and answer-parser failures.
- [ ] Add end-to-end request timing: queue, prefill, draft, verify, resample,
      decode, postprocess.
- [ ] Record draft, target, and accepted/generated token counts.
- [ ] Record ESS, resampling count, unique ancestors, KV usage, peak memory.
- [x] Add per-job active-model QPS/time per query, mean TTFT/inter-token
      latency, agent wall time, and correct tasks per agent GPU-hour by joining
      SGLang Prometheus snapshots with Harbor/Pi results.
- [ ] Add saturated QPS, p50/p95 latency, and accelerator-seconds/query through
      controlled frozen-trace replay.
- [ ] Save a golden configuration and result for regression testing.
- [ ] **Draft-adequacy diagnostic:** ESS decay rate and unique-ancestor
      half-life on GSM8K vs a MATH500 subset, across `N` and `gamma`.

Exit criterion / gate: a reproducible quality/cost/latency baseline exists,
and particle diversity survives the hard task. SMC-SD is importance sampling
in sequence space with a small proposal; if `N_eff` collapses toward 1 on
MATH500-class problems, fix the proposal story first (larger draft, draft
temperature, smaller `gamma`) — no composite objective rescues a degenerate
particle population.

### Phase 2 — Three offline studies (the decision gate)

All three run on saved trajectories sampled from the current engine; no
engine changes. Budget: days.

**(a) Semantic predictiveness**

- [ ] Sample diverse partial trajectories at several completion fractions.
- [ ] Ask the chosen scorer for expected recoverability scores.
- [ ] Measure AUROC/AUPRC for eventual correctness.
- [ ] Measure calibration error and reliability curves.
- [ ] Measure ranking accuracy within candidates for the same prompt.
- [ ] Compare 5 versus 20 labels; sweep criterion wording and a small set of
      scorer models.
- [ ] Estimate scorer latency, tokens, and batching efficiency.
- [ ] Freeze a versioned first criterion and calibration mapping.

**(b) Design A vs design B score quality.** Score the same prefixes via the
suffix-form KV-fork prompt and the clean re-prompted form. If A tracks B
(rank correlation, AUROC delta within noise), A becomes the online default;
if not, B's host-loop and prefill costs define the checkpoint budget.

**(c) Two-target likelihood complementarity.** Score saved trajectories under
a second same-tokenizer target; measure ranking complementarity and
alpha-mixture accuracy versus `p_1` alone. Same-family models are expected to
be highly correlated — this is the cheap test of whether LM mode deserves
engine work at all.

Replay caveat (unchanged): offline predictiveness supports, but does not
prove, online reallocation value — that is Phase 3's question.

Gate: the evidence sets the build order. S-family goes first if (a) holds
(expected). Phase 5 (multi-target) is built only if (c) shows real
complementarity; otherwise LM is demoted to the `M=1` identity test plus one
ablation row.

### Phase 3 — First online experiment (specification in Section 17)

The centerpiece question, answered before any optimization: **does
intermediate semantic reallocation create value beyond spending the same
verifier budget terminally?** One draft, one target, one scorer; the six
conditions of Section 17 at matched allocated cost on GSM8K + a MATH500
subset.

Implementation needed — the minimal eager composite path only:

- [ ] Factor interfaces in `composite_target.py`; likelihood and semantic
      components tracked separately.
- [ ] Score-difference checkpoint updates; semantic-only with `q` base.
- [ ] Objective modes L1, S, L1+S; explicit final-selection policies.
- [ ] The Phase-2-winning scorer design (A or B).
- [ ] Ordinary PyTorch throughout; the Section 10 objective tests pass first.

Exit criterion / gate: a matched-cost answer either way. Positive → Phases
4–7. Negative → the pivot is explicit: "terminal reranking, served fast"
(still a systems paper), or LM-led (only if Phase 2(c) was positive).

### Phase 4 — Productionize the online scorer

- [ ] Scorer configuration and prompt templates (Section 5 configurability).
- [ ] Batch semantic checkpoints across particles and requests.
- [ ] Compact identical prefixes (ancestry-based) before scoring.
- [ ] Extract expected score from scoring-token logits.
- [ ] Cache the previous score and apply only score differences.
- [ ] Propagate score state through resampling.
- [ ] Support `H` independent of `gamma`; report effective `H` (checkpoints
      quantize to cycle boundaries).
- [ ] Terminal semantic pass for final selection and fair baselines.
- [ ] Design A: fixed-shape suffix extend on forked KV sharing the
      TARGET_VERIFY execution path. Design B: external stock-sglang scorer
      engine (radix cache on) with explicit queue, backpressure, and cost
      accounting.
- [ ] Validate semantic-only and L1+S end to end.

Exit criterion: the online scorer changes ancestry correctly, preserves or
improves development-set quality, and produces complete cost traces.

### Phase 5 — Multiple likelihood targets (conditional on Phase 2(c))

- [ ] Introduce `TargetGroup` and parallel target workers.
- [ ] Validate tokenizer identity before engine startup.
- [ ] Score selected tokens on each target; aggregate weighted log
      likelihoods.
- [ ] Implement and test the designated-target bonus proposal by enumeration;
      provide a no-bonus reference mode. Do not silently reuse a one-target
      correction under a multi-target proposal.
- [ ] Apply one shared ancestry map to every target's KV state.
- [ ] Compare serial versus parallel target execution.
- [ ] Validate LM with `M=1` against L1.

Exit criterion: LM passes exact small-model tests and reports the cost of all
likelihood models.

### Phase 6 — Combine and characterize objectives

- [ ] Run the implemented modes with the same draft and budget.
- [ ] Sweep target size (4B→32B), `N`, `gamma`, `H`, ESS threshold, `alpha`, and `beta`.
- [ ] Measure factor correlation and complementarity.
- [ ] Confirm the Phase 3 result at scale: online semantic versus terminal
      reranking; multi-model likelihood versus one target (if built).
- [ ] Select a small Pareto-optimal configuration family for optimization.

Do not optimize every configuration. The system section should focus on the
objective variants that show real quality/cost value.

Exit criterion: at least one online composite configuration beats a
terminal-only control at matched accelerator cost.

### Phase 7 — Optimize one bottleneck at a time

- [ ] Profile the eager implementation with timeline traces.
- [ ] Add unique-prefix semantic batching.
- [ ] Add per-model KV reference counting and copy-on-write.
- [ ] Fuse selected-token likelihood extraction/aggregation.
- [ ] Fuse weight normalization, ESS, and resampling.
- [ ] Add graph-friendly fixed shapes for the two cycle types.
- [ ] Overlap independent target/scorer work.
- [ ] Add cross-request pipeline scheduling.
- [ ] Add queueing, fairness, and memory admission control.
- [ ] Reprofile after every optimization.

For every optimization, preserve an off switch and record both the
performance delta and objective-equivalence check.

Exit criterion: the cumulative optimized system produces a clear
throughput/latency improvement over the eager composite implementation
without changing accuracy outside confidence intervals.

### Phase 8 — Full baseline implementation

- [x] Add and validate the target-only AR baseline on the same vendored stock
      SGLang serving stack and Pi contract.
- [ ] Implement the baseline list in Section 9 through one harness (SMC
      methods via `SMCEngine`/`smcsd.http_server`; stock methods via stock
      engines on the same vendored tree).
- [ ] Add paper-faithful settings where feasible.
- [ ] Add shared-component variants using the same models, prompts, candidate
      budgets, and verifier.
- [ ] Validate every baseline on a tiny set manually.
- [ ] Sweep each method's natural compute knob to obtain a curve, not one
      point.
- [ ] Save per-request traces so quality and systems metrics can be
      recomputed.

Exit criterion: every main baseline has a reproducible command/config and a
verified cost trace.

### Phase 9 — Main evaluation

- [ ] Freeze code and configs before running the test set.
- [ ] Run at least three seeds where stochastic variation is material.
- [ ] Bootstrap confidence intervals over prompts.
- [ ] Measure isolated latency and saturated-serving throughput.
- [ ] Run at several concurrency/SLO settings.
- [ ] Account for every assigned accelerator, including pipeline bubbles.
- [ ] Generate all figures and tables from saved result files.
- [ ] Perform error analysis by task difficulty, output length, and scorer
      failure.

Exit criterion: all main claims are supported by matched-budget curves and
uncertainty estimates.

### Phase 10 — Artifact and paper

- [ ] Provide a minimal reproducible model/config matrix.
- [ ] Add scripts for downloading data and running each baseline.
- [ ] Add a one-command small-scale correctness experiment.
- [ ] Add a one-command systems benchmark.
- [ ] Document hardware, software, prompts, and cost accounting.
- [ ] Release raw aggregate results and plotting scripts where licensing
      permits.
- [ ] Write limitations, including tokenizer compatibility and extra-model
      memory.

Exit criterion: another researcher can reproduce the small result and
understand how the full result was obtained.

## 9. Baselines



### Core generation baselines

1. **Target autoregressive decoding.** One sample from the main target; establishes ordinary serving latency and accuracy.
2. **Target self-consistency.** Generate `n` target samples and majority-vote extracted answers.
3. **Target best-of-n.** Generate `n` target samples and select with the same terminal semantic verifier used by our method.
4. **Draft best-of-n.** Generate `n` draft samples and select with the same verifier.
5. **Current SM-CSD.** One draft, one target, likelihood-only, `beta=0`, followed by the same final-selection rule where applicable. This is the closest algorithmic and system baseline.



### Required paper baselines

1. **LLM-as-a-Verifier: A General-Purpose Verification Framework.**
  - Paper-faithful: expected scoring-token probabilities, its reported granularity/repetition/criteria choices where relevant, and its cost-efficient candidate ranking procedure.
  - Shared-component: our generated candidate pool, the same scorer and criterion, then terminal LLM-as-a-Verifier ranking.
  - This isolates the value of online particle reallocation from terminal verification.
2. **Guided Speculative Inference (GSI).**
  - Paper-faithful: reasoning-step proposals from the small model, reward plus target-versus-small likelihood correction, threshold-triggered target fallback, and soft best-of-n selection.
  - For mathematical tasks, reproduce its step delimiter and PRM setup if model access permits.
  - Shared-component: use the same draft, target, semantic scorer/prompt, candidate count, and hardware as our method.
  - Report its own paper-faithful configuration separately from the controlled variant.



### Closely related algorithmic baselines

1. **Rollout Roulette / particle-filtering test-time scaling.** Use the closest public implementation or a clearly documented reproduction.
2. **Sampling for Quality.** Compare its inference-time sampling allocation under matched compute.
3. **Verifier-guided beam search or reward-guided speculative decoding.** Use as a simpler online search control.
4. **Reward-Guided Speculative Decoding (RSD).** Include if needed to connect directly to the GSI comparison set.
5. **Syntactic and Semantic Control via SMC** and **SMC-SD.** At minimum discuss; run when implementations and compatible tasks are available.

If a public implementation is unavailable or incompatible, document the gap and implement only the part needed for a fair controlled comparison. Do not call a controlled reimplementation “paper-faithful.”

### Systems baselines and ablations

- Naive eager composite SMC: serial model calls, no semantic compaction, no KV optimization.
- Existing optimized SM-CSD likelihood-only path.
- Optimized serving engine best-of-n, using vLLM or SGLang if compatible.
- Cumulative system ladder:
  1. eager;
  2. + unique-prefix compaction;
  3. + KV sharing/copy-on-write;
  4. + fused factor and resampling kernels;
  5. + fixed-shape graph capture;
  6. + inter-model overlap;
  7. + cross-request pipeline.



## 10. Correctness test plan



### Objective tests

- [ ] L1 with `M=1` and `beta=0` matches current SM-CSD.
- [ ] If `q=p` and there is no semantic score, likelihood increments are zero.
- [ ] Multi-likelihood aggregation matches a scalar CPU reference.
- [ ] Semantic deltas telescope to final score minus initial score.
- [ ] Constant semantic scores are a no-op.
- [ ] `beta=0` is a no-op.
- [ ] Duplicate scorers with split weights equal one scorer with combined weight.
- [ ] Zero/negative-infinity weights do not create NaNs.
- [ ] Normalized weights sum to one after every collection step.
- [ ] Toy finite-vocabulary enumeration matches the intended posterior by TV distance/KL.
- [ ] Designated-target bonus correction matches enumeration.



### State and scheduling tests

- [ ] Ancestry copies previous semantic scores correctly.
- [ ] Every model receives exactly the same logical ancestry map.
- [ ] KV reference counts remain balanced through repeated duplication and deletion.
- [ ] Finished/EOS particles remain finished across checkpoints.
- [ ] `H < gamma`, `H = gamma`, and `H > gamma` behave as specified.
- [ ] Checkpoints crossing request maximum length are handled once.
- [ ] Variable prompt lengths and early termination do not corrupt batches.
- [ ] Unique-prefix scoring equals uncompressed scoring.
- [ ] Eager and fused paths agree within tolerance.
- [ ] Final-selection policies choose according to their documented value, not importance weight by accident.



### Regression tests

- [ ] Golden fixed-seed output for current SM-CSD.
- [ ] Golden factor trace for a tiny L1+S request.
- [ ] Golden ancestry trace for a forced-resampling request.
- [ ] Peak-memory smoke test for all configured models.
- [ ] Multi-request fairness and cancellation tests.



## 11. Workloads



### Development

- GSM8K for fast iteration and answer-parser debugging.
- A 100–200 problem subset of MATH500 for harder reasoning.
- Synthetic tiny-vocabulary tasks for exact posterior checks.



### Main reasoning evaluation

- MATH500.
- AIME, with year/split frozen and contamination caveats documented.
- OlympiadBench.
- GPQA or GPQA Diamond for a non-math reasoning domain.



### Optional generalization

- HumanEval+ or MBPP+ only if code execution is added consistently to evaluation.
- A long-form agentic benchmark only if semantic-prefix scoring remains meaningful and serving runs are affordable.

Start with math. Adding many domains before the algorithm and accounting are stable will dilute the systems study.

## 12. Metrics and cost accounting



### Primary quality/cost metrics

- Accuracy versus allocated accelerator-seconds per query.
- QPS at matched accuracy.
- Correct answers per GPU-hour.
- p50 and p95 end-to-end latency at matched accuracy.
- Maximum throughput under a specified p95 latency SLO.

“Allocated accelerator-seconds” means wall-clock duration multiplied by every accelerator reserved for that run, including draft, likelihood targets, semantic scorers, idle pipeline time, and communication stalls. Also report active GPU-seconds as a diagnostic, but do not use it to hide poor utilization.

### Diagnostic metrics

- Raw output tokens/second.
- Draft, target, and verifier tokens per query.
- Number of semantic calls and verifier tokens avoided.
- Acceptance rate and generated tokens per cycle.
- ESS before/after each factor.
- Resampling frequency.
- Number of unique live prefixes/ancestors.
- Semantic-score predictiveness and calibration.
- GPU utilization per model.
- Peak KV-cache and total device memory.
- Host/device and inter-GPU bytes.
- CUDA graph hit rate.
- Queue delay and pipeline bubble time.
- Stage breakdown: prefill, draft, likelihood, semantic, resample, KV movement, postprocess.



### Statistical reporting

- Report prompt-level bootstrap 95% confidence intervals.
- Use paired comparisons on identical prompts.
- Use at least three seeds for stochastic methods when affordable.
- Do not claim superiority from overlapping/noisy single-point estimates.



## 13. Fair comparison protocol

For every matched comparison, hold constant:

- model checkpoints and numeric precision;
- tokenizer and chat templates;
- prompt text and stopping criteria;
- maximum output tokens;
- sampling temperature/top-p unless it is an algorithmic knob;
- dataset split and answer parser;
- hardware type and software stack;
- terminal scorer and score calibration for controlled variants.

Sweep each algorithm's natural compute knob:

- number of samples for BoN/self-consistency;
- number of particles `N`;
- `gamma`;
- semantic interval `H`;
- GSI candidate count and fallback threshold;
- verifier granularity/repetition;
- semantic temperature `beta`.

Compare Pareto frontiers rather than hand-picked points. Use both:

1. equal allocated accelerator-seconds per query;
2. equal latency/SLO at a fixed concurrency.

For methods whose papers assume different reward models or hardware mappings, report two rows:

- paper-faithful reproduction;
- shared-component controlled comparison.



## 14. Required ablations



### Objective ablations

- L1 versus LM versus S versus L1+S versus LM+S.
- One versus multiple likelihood targets.
- One versus multiple semantic scorers.
- Recoverability versus alternative criteria.
- Expected token score versus argmax score.
- Calibrated versus raw semantic score.
- Terminal-only verifier versus online semantic checkpoints.



### Scheduling ablations

- Semantic interval `H`.
- `H` coupled to `gamma` versus independent `H`.
- Fixed token intervals versus reasoning-step boundaries.
- One-stage versus staged likelihood/semantic resampling.
- Scoring all particles versus unique prefixes.
- Serial versus overlapped/pipelined model execution.



### SMC ablations

- Target model scale (4B → 32B) at fixed draft.
- Particle count.
- ESS threshold.
- Systematic resampling versus alternatives if implemented.
- Target bonus proposal versus no bonus for multi-target mode.
- Final posterior sample versus terminal best-semantic selection.



### Systems ablations

- Each cumulative optimization in Section 9.
- Per-model KV sharing.
- CUDA graph capture.
- Fused kernels.
- Cross-request batching.
- GPU placement and tensor-parallel degree.



## 15. Paper figures and tables

Minimum main-paper artifacts:

1. Accuracy versus allocated accelerator-seconds/query.
2. Accuracy versus p50/p95 latency.
3. QPS and correct answers/GPU-hour at matched accuracy.
4. Objective comparison: L1, LM, S, L1+S, LM+S.
5. Cumulative system optimization ablation.
6. Latency-stage breakdown.
7. GPU-utilization timeline showing cross-request overlap.
8. ESS, unique ancestors, and semantic checkpoints over generation time.
9. Baseline table with paper-faithful and shared-component variants.
10. Peak-memory and KV-cache table.

Every headline figure should identify particle count, model placement, concurrency, and whether cost is allocated or active accelerator time.

## 16. Go/no-go decisions

Proceed past Phase 1 only if particle diversity survives the hard task
(draft-adequacy diagnostic). If `N_eff` collapses toward 1 on MATH500-class
problems, fix the proposal (draft size/temperature, `gamma`) before building
any composite objective.

Proceed with semantic online SMC only if:

- prefix recoverability score predicts eventual correctness;
- score calibration is stable enough to choose `beta`;
- online checkpoints outperform terminal-only reranking at matched total cost.

Proceed with multiple likelihood targets only if:

- the Phase 2(c) offline complementarity study was positive;
- the models provide complementary rankings or accuracy;
- the gain survives the full multi-GPU cost calculation;
- the bonus-proposal correction passes exact tests.

Proceed with heavy kernel work only after profiling shows the relevant operation is material. If verifier compute dominates, prioritize compaction, batching, caching, and pipeline overlap before resampling micro-optimizations.

If the final system raises accuracy but cannot improve query-level serving efficiency after charging all GPUs, the thesis claim must be narrowed; raw TPS is not sufficient.

## 17. First online experiment (Phase 3 specification)

Use one draft, one target, and one semantic verifier on GSM8K plus a small MATH500 subset:

- `N in {8, 16, 32}`;
- current default `gamma` plus one smaller and one larger value;
- `H in {32, 64, 128}`;
- `beta in {0, small, medium}`;
- fixed recoverability criterion;
- terminal semantic reranking for every method.

Run:

1. target AR;
2. target BoN;
3. current SM-CSD/L1;
4. semantic-only SMC;
5. L1+S;
6. L1 with terminal-only semantic reranking.

This experiment answers the first important scientific question: does intermediate semantic reallocation create value beyond spending the same verifier budget at the end? Only after that result should multiple likelihood targets and deeper systems optimization become the main focus.

## 18. Definition of done

The project is complete when:

- all four objective families are implemented and correctness-tested;
- current SM-CSD remains an exact compatibility mode;
- semantic scorer model(s), criteria, and intervals are configurable;
- at least the required LLM-as-a-Verifier and GSI baselines run through the same harness;
- all accelerator costs are counted;
- results include quality/cost Pareto curves and serving-load experiments;
- the optimized system beats strong baselines at matched accuracy on at least two substantive workloads;
- the artifact can reproduce a small end-to-end result.

An eventual headline should have this form:

> At matched accuracy on [benchmark] and counting all allocated accelerators, TTS-SMC serves X times more queries per second / reduces p95 latency by Y% / produces Z times more correct answers per GPU-hour than [strongest baseline].

Do not choose X, Y, or Z until the frozen evaluation produces them.

## 19. Experiment matrix (v2, 2026-08-20 — supersedes v1 below)

**Family decision (recorded):** Qwen3.5-generation (shared tokenizer, vocab
248077). Default draft **Qwen3.5-2B** (won the draft-economics sweep:
17/20 @ 563 tok/s vs the 397B target; 27B draft confirmed 2.3× slower with
no accuracy gain — active-params trap). Alternate draft 4B. Target ladder
**9B → Qwen3.8-27B → Qwen3.5-397B-A17B-FP8 (TP4)** — a capability ladder,
not a FLOPs ladder (397B has 17B active). Thinking mode off everywhere.
Comm flags for TP runs: disable_custom_all_reduce +
enforce_disable_flashinfer_allreduce_fusion; CUDA_HOME=/data/home/yahya/cuda-13.0.

**Baselines (Phase 1):** target AR; self-consistency majority@n (stock
engines, n matched to N). Vanilla speculative decoding dropped — not a TTS
method; the TTS controls are majority@n and terminal reranking. Paper-faithful
set (GSI, LLM-as-a-Verifier, Rollout Roulette) unchanged in Phase 8.

**Particle scaling to N=64 is a first-class requirement:** cuda_graph_max_bs
(→ _decode) and SMC_DRAFT_GRAPH_MAX_BS must cover groups×N at every point,
verified via SMC_GRAPH_STATS showing cycle_graph serving ~100% of cycles.
Untuned graph caps silently demote large-N runs to fallback tiers and
invalidate throughput numbers.

- **E1.1 Reference grid:** {9B, 27B, 397B} × {2B, 4B} × 3 seeds ×
  {GSM8K-200, MATH500-100}, N=8, γ=8.
- **E1.2 Draft adequacy (gate):** ESS/ancestry diagnostics on {2B, 4B} ×
  {9B, 397B} × N∈{8, 32, 64} × γ∈{4, 8}, MATH500-100 — the kill-question is
  lineage collapse on hard tasks at scale.
- **E1.3 N-sweep vs TTS controls:** N∈{1,2,4,8,16,32,64} at 9B+2B and
  397B+2B on both benchmarks, γ=8, graph-tuned; against target-AR and
  majority@n at matched candidate counts. First Pareto sketch.

Exit: results in hand → decide experiment scaling, further baselines, and
algorithm changes for particle/generation scaling.

## 19b. Experiment matrix (v1, superseded)

Model roster (tokenizer identity across the ladder verified 2026-08-20):
drafts Qwen3-0.6B / 1.7B; targets Qwen3-4B / 8B / 14B / 32B (one B200
each); scorer candidates: the target itself (design A), Qwen3-8B/32B as
external judge (design B). All runs non-thinking unless a later ablation
says otherwise.

### Phase 1 (current engine + counters only)

- **E1.1 Reference grid:** {4 targets} × {2 drafts} × 3 seeds, N=8, γ=8,
  GSM8K 200q → accuracy / tok/s / cost reference and the first
  target-scaling curve.
- **E1.2 Draft adequacy (gate):** {0.6B, 1.7B} × {8B, 32B} × N∈{8,16,32}
  × γ∈{4,8} on GSM8K and MATH500-100. Readouts: per-cycle ESS decay,
  resample frequency, unique-ancestor half-life, accuracy. Key
  interaction: larger target ⇒ larger q–p gap ⇒ faster ESS collapse.
- **E1.3 N buys accuracy?** Likelihood-only N∈{1..32} at fixed γ vs
  target-AR and target-BoN-N controls through stock sglang — earliest
  signal that SMC reallocation is worth anything.
- Build: per-cycle ESS/resample/ancestor counters (SMC_GRAPH_STATS
  pattern), --dump-trajectories JSONL on the eval script, run-manifest
  JSON. No algorithm changes.

### Phase 2 (offline, zero engine code)

- **E2.a predictiveness:** prefixes at 25/50/75% of saved trajectories,
  scored by 8B-self and 32B-judge, 5 vs 20 labels → AUROC/calibration.
- **E2.b design A vs B:** same prefixes, suffix-form prompt vs clean
  re-prompt (offline, both are just prompts) → rank correlation decides
  the online scorer design.
- **E2.c complementarity:** rescore trajectories under 14B/32B; does the
  α-mixture rank better than p₁ alone? Decides whether Phase 5 is built.

### Phase 3 (first engine build, scoped by the gates)

Six §17 conditions at matched allocated cost; N∈{8,16,32} ×
H∈{32,64,128} × β∈{0, small, med}; GSM8K + MATH500-100; 3 seeds.
Build: composite_target.py, score-difference checkpoints, L1/S/L1+S,
final-selection policies, one scorer design (per E2.b).

### Phases 5–7

Phase 5 only if E2.c surprises; Phase 6 re-runs winning modes across
target scale × N × γ × H (headline scaling figures); Phase 7 optimizes
against the Phase 3 eager implementation.

## 20. References

- [LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391) ([PDF](https://arxiv.org/pdf/2607.05391))
- [Guided Speculative Inference for Efficient Test-Time Alignment of LLMs](https://arxiv.org/abs/2506.04118) ([PDF](https://arxiv.org/pdf/2506.04118))
- [Rollout Roulette](https://arxiv.org/abs/2502.01618)
- [Sampling for Quality](https://arxiv.org/abs/2604.16453)
- [Syntactic and Semantic Control of Large Language Model Generation via Sequential Monte Carlo](https://arxiv.org/abs/2504.13139)
- [SMC-SD](https://arxiv.org/abs/2604.15672)
- [Reward-Guided Speculative Decoding](https://arxiv.org/abs/2501.19324)
- [ETS](https://arxiv.org/abs/2502.13575)
