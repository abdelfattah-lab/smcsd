# TTS: Test-Time-Scaling Serving with Composite SMC Objectives

Status: project roadmap  
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

## 2. Scope and initial assumptions



### In scope

- Inference-only test-time scaling.
- One draft model `q`.
- One or more likelihood target models `p_m`.
- Zero or more selectable semantic scorer models `v_j`.
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

The current SM-CSD path is already the first experimental condition:

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
- `smcsd/core/fused_collect.py`: normalization, ESS, compaction, and systematic resampling.
- `smcsd/core/fused_resample_kv.py`: KV and block-table updates after resampling.



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

The initial scorer implementation may use a separate verifier prompt with its own prefix cache. A more aggressive suffix-form KV fork is valid only if the scorer input can literally reuse the generation prefix followed by a fixed scoring suffix.

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



### Phase 0 — Freeze the experiment contract

- [ ] Choose the primary draft/target/scorer model family and GPU topology.
- [ ] Validate exact tokenizer compatibility for likelihood models.
- [ ] Define prompt templates and answer extraction once.
- [ ] Define what accelerator time includes.
- [ ] Choose development and held-out test splits.
- [ ] Record dependency versions, model revisions, seeds, and hardware.
- [ ] Add a run manifest and machine-readable result schema.
- [ ] Decide the first paper claim and go/no-go thresholds before optimizing.

Exit criterion: one written experiment contract can be applied unchanged to every method.

### Phase 1 — Reproduce and instrument current SM-CSD

- [ ] Run one draft + one target + likelihood-only on a small reasoning set.
- [ ] Confirm deterministic fixed-seed repeatability where expected.
- [ ] Record exact-match accuracy and answer-parser failures.
- [ ] Add end-to-end request timing: queue, prefill, draft, verify, resample, decode, postprocess.
- [ ] Record draft, target, and accepted/generated token counts.
- [ ] Record ESS, resampling count, unique ancestors, KV usage, and peak memory.
- [ ] Add QPS, p50/p95 latency, accelerator-seconds/query, and correct answers/GPU-hour.
- [ ] Save a golden configuration and result for regression testing.

Exit criterion: the current system has a reproducible quality/cost/latency baseline and no semantic code is involved.

### Phase 2 — Test semantic signal offline

- [ ] Sample diverse partial trajectories at several completion fractions.
- [ ] Ask the chosen scorer for expected recoverability scores.
- [ ] Measure AUROC/AUPRC for eventual correctness.
- [ ] Measure calibration error and reliability curves.
- [ ] Measure ranking accuracy within candidates for the same prompt.
- [ ] Compare 5 versus 20 labels.
- [ ] Sweep criterion wording and a small set of scorer models.
- [ ] Estimate scorer latency, tokens, and batching efficiency.
- [ ] Freeze a versioned first criterion and calibration mapping.

Important limitation: scoring saved complete trajectories can establish that the score is predictive, but not that online reallocation improves outcomes. A faithful replay study needs a precomputed branching continuation tree, or it must regenerate new suffixes after each simulated resampling decision.

Exit criterion: the semantic score predicts eventual correctness well enough to justify an online prototype and has a stable prompt/calibration.

### Phase 3 — Build a slow, testable composite-objective reference

- [ ] Implement factor interfaces in `composite_target.py`.
- [ ] Track likelihood and semantic components separately.
- [ ] Implement score-difference checkpoint updates.
- [ ] Implement semantic-only with `q` as the base distribution.
- [ ] Implement the four objective modes L1, LM, S, and combined.
- [ ] Add explicit final-selection modes.
- [ ] Use ordinary PyTorch operations without custom-kernel optimization.

Exit criterion: exact toy tests and all degeneracy/invariance tests in Section 10 pass.

### Phase 4 — Add one online semantic scorer

- [ ] Add scorer configuration and prompt templates.
- [ ] Batch semantic checkpoints across particles and requests.
- [ ] Compact identical prefixes before scoring.
- [ ] Extract expected score from scoring-token logits.
- [ ] Cache the previous score and apply only score differences.
- [ ] Propagate score state through resampling.
- [ ] Support `H` values independent of `gamma`.
- [ ] Add a terminal semantic pass for final selection and fair baselines.
- [ ] Validate semantic-only and L1+S end to end.

Exit criterion: with one scorer, the online eager implementation changes ancestry correctly, improves or preserves quality on the development set, and produces complete cost traces.

### Phase 5 — Add multiple likelihood targets

- [ ] Introduce `TargetGroup` and parallel target workers.
- [ ] Validate tokenizer identity before engine startup.
- [ ] Score selected tokens on each target.
- [ ] Aggregate weighted log likelihoods.
- [ ] Implement and test the designated-target bonus proposal.
- [ ] Provide a no-bonus reference mode.
- [ ] Apply one shared ancestry map to every target's KV state.
- [ ] Compare serial versus parallel target execution.
- [ ] Validate LM with `M=1` against L1.

Exit criterion: LM passes exact small-model tests and reports the cost of all likelihood models.

### Phase 6 — Combine and characterize objectives

- [ ] Run L1, LM, S, L1+S, and LM+S with the same draft and budget.
- [ ] Sweep `N`, `gamma`, `H`, ESS threshold, `alpha`, and `beta`.
- [ ] Measure factor correlation and complementarity.
- [ ] Check whether multi-model likelihood adds quality beyond one target.
- [ ] Check whether semantic scoring adds quality beyond terminal reranking.
- [ ] Select a small Pareto-optimal configuration family for optimization.

Do not optimize every configuration. The system section should focus on the objective variants that show real quality/cost value.

Exit criterion: at least one online composite configuration beats a terminal-only control at matched accelerator cost.

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

For every optimization, preserve an off switch and record both the performance delta and objective-equivalence check.

Exit criterion: the cumulative optimized system produces a clear throughput/latency improvement over the eager composite implementation without changing accuracy outside confidence intervals.

### Phase 8 — Full baseline implementation

- [ ] Implement the baseline list in Section 9 through one harness.
- [ ] Add paper-faithful settings where feasible.
- [ ] Add shared-component variants using the same models, prompts, candidate budgets, and verifier.
- [ ] Validate every baseline on a tiny set manually.
- [ ] Sweep each method's natural compute knob to obtain a curve, not one point.
- [ ] Save per-request traces so quality and systems metrics can be recomputed.

Exit criterion: every main baseline has a reproducible command/config and a verified cost trace.

### Phase 9 — Main evaluation

- [ ] Freeze code and configs before running the test set.
- [ ] Run at least three seeds where stochastic variation is material.
- [ ] Bootstrap confidence intervals over prompts.
- [ ] Measure isolated latency and saturated-serving throughput.
- [ ] Run at several concurrency/SLO settings.
- [ ] Account for every assigned accelerator, including pipeline bubbles.
- [ ] Generate all figures and tables from saved result files.
- [ ] Perform error analysis by task difficulty, output length, and scorer failure.

Exit criterion: all main claims are supported by matched-budget curves and uncertainty estimates.

### Phase 10 — Artifact and paper

- [ ] Provide a minimal reproducible model/config matrix.
- [ ] Add scripts for downloading data and running each baseline.
- [ ] Add a one-command small-scale correctness experiment.
- [ ] Add a one-command systems benchmark.
- [ ] Document hardware, software, prompts, and cost accounting.
- [ ] Release raw aggregate results and plotting scripts where licensing permits.
- [ ] Write limitations, including tokenizer compatibility and extra-model memory.

Exit criterion: another researcher can reproduce the small result and understand how the full result was obtained.

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

Proceed with semantic online SMC only if:

- prefix recoverability score predicts eventual correctness;
- score calibration is stable enough to choose `beta`;
- online checkpoints outperform terminal-only reranking at matched total cost.

Proceed with multiple likelihood targets only if:

- the models provide complementary rankings or accuracy;
- the gain survives the full multi-GPU cost calculation;
- the bonus-proposal correction passes exact tests.

Proceed with heavy kernel work only after profiling shows the relevant operation is material. If verifier compute dominates, prioritize compaction, batching, caching, and pipeline overlap before resampling micro-optimizations.

If the final system raises accuracy but cannot improve query-level serving efficiency after charging all GPUs, the thesis claim must be narrowed; raw TPS is not sufficient.

## 17. Recommended first experiment

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

## 19. References

- [LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391) ([PDF](https://arxiv.org/pdf/2607.05391))
- [Guided Speculative Inference for Efficient Test-Time Alignment of LLMs](https://arxiv.org/abs/2506.04118) ([PDF](https://arxiv.org/pdf/2506.04118))
- [Rollout Roulette](https://arxiv.org/abs/2502.01618)
- [Sampling for Quality](https://arxiv.org/abs/2604.16453)
- [Syntactic and Semantic Control of Large Language Model Generation via Sequential Monte Carlo](https://arxiv.org/abs/2504.13139)
- [SMC-SD](https://arxiv.org/abs/2604.15672)
- [Reward-Guided Speculative Decoding](https://arxiv.org/abs/2501.19324)
- [ETS](https://arxiv.org/abs/2502.13575)

