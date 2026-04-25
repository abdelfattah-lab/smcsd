# EAGLE / SMC-SD Proposal Learning Plan

## Goal

Train a better proposal/draft model for SMC-SD, with the initial target being:

```bash
python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --particles 12 --gamma 8 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --num-questions 400 \
  --max-running-requests 24 \
  --cuda-graph-max-bs 24
```

Current reference point:

```text
Target: meta-llama/Llama-3.1-8B-Instruct
AR draft: meta-llama/Llama-3.2-1B-Instruct
Task: GSM8K
N particles: 12
gamma: 8
temperature: 0.7
Accuracy: ~75%
Throughput: ~310 TPS
```

We want to improve the accuracy/speed Pareto frontier by training an SMC-native proposal. The current AR Stage A/B draft distillation experiments were not sufficiently helpful, so the plan is to build correct EAGLE support and then train an EAGLE-style proposal with an SMC-native objective.

The clean implementation base should be:

```text
branch: smc-slot-refactor
```

Existing EAGLE-related branches may contain useful code, but they should be treated as prototypes until verified:

```text
eagle
eagle-train
eagle3-cg
smc-eagle-tree
test-eagle
```

If those branches are incorrect or too tangled, implement EAGLE support from scratch on top of `smc-slot-refactor`.

---

## Core hypothesis

SMC-SD does not need a draft model that is merely good under next-token CE. It needs a proposal distribution `q` that produces low-variance importance weights under the target model `p`.

For a proposed block/path:

```text
y = y_1, ..., y_K
```

SMC uses:

```text
log w(y) = sum_t log p(y_t | prefix, y_<t) - sum_t log q(y_t | proposal_state, y_<t)
```

Good SMC proposals should have:

```text
low log-weight variance
high ESS / N
low resampling frequency
high target probability on sampled paths
good downstream accuracy
high throughput
```

Plain AR distillation can improve token-level KL while still failing at path-level SMC metrics because it is off-policy and does not directly optimize the `p/q` weight distribution.

---

## Why EAGLE is promising

For targets that do not have a natural small sibling draft model, e.g. GPT-OSS-20B, a normal small AR draft may be unavailable or weak. EAGLE-style draft heads are attractive because they use target hidden states as input.

EAGLE3 proposal form:

```text
q_eagle(y_{t+1} | target_hidden(prefix), previous_draft_tokens)
```

This is different from normal AR drafts:

```text
q_ar(y_{t+1} | token_prefix)
```

EAGLE can be much smaller because the target model has already computed rich hidden-state features for the prefix.

However, for SMC-SD this only works if:

1. EAGLE is wired correctly into the SMC decode loop.
2. EAGLE produces valid proposal logprobs for sampled tokens.
3. EAGLE has full target-vocabulary support, or an explicitly corrected full-support mixture proposal.
4. EAGLE is trained with an SMC-native objective, not only the standard speculative-decoding objective.

---

## Important: full-vocabulary EAGLE for SMC

For valid importance sampling, the proposal should have support wherever the target has support:

```text
if p(token | prefix) > 0, q(token | state) should also be > 0
```

Hot-vocab EAGLE configs such as:

```json
"vocab_size": 201088,
"draft_vocab_size": 32000
```

are risky for SMC because tokens outside the draft vocabulary have:

```text
q(token) = 0
```

while the target may assign nonzero probability.

Therefore, for SMC-native EAGLE, prefer:

```json
"draft_vocab_size": "same as target vocab_size"
```

For Llama-3.1-8B:

```text
vocab_size = 128256
draft_vocab_size = 128256
```

For GPT-OSS-20B:

```text
vocab_size = 201088
draft_vocab_size = 201088
```

If reduced-vocab EAGLE is needed for speed, then implement a full-support mixture proposal:

```text
q_mix = (1 - epsilon) q_hot_vocab + epsilon q_fallback_full_vocab
```

and compute exact logprobs under `q_mix`. Do not use reduced-vocab EAGLE naively for SMC weights.

---

## Phase 0: verify current baseline and instrumentation

Before changing the draft type, lock down the baseline. Using smc_slot_refactor branch and the smc_v2_clean sglang fork branch

### Baseline command

```bash
python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --particles 12 --gamma 8 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --num-questions 400 \
  --max-running-requests 24 \
  --cuda-graph-max-bs 24
```

Expected reference:

```text
Accuracy: ~75%
Throughput: ~310 TPS
```

### Add/confirm metrics

For every SMC-SD run, log:

```text
accuracy / task score
tokens/sec
ESS / N per decode step
resampling frequency
mean log_weight
variance of log_weight
min/max normalized particle weight
number of target verify calls
number of draft steps
average emitted tokens per round
```

These metrics are more important than token CE alone. A proposal is useful if it improves the accuracy/throughput frontier and improves ESS/log-weight statistics.

---

## Phase 1: build correct EAGLE support in SMC-SD

### Implementation target

Implement or verify EAGLE support on top of:

```text
smc-slot-refactor
```

Start with the simplest correct mode:

```text
EAGLE chain mode
topk = 1
full vocabulary
gamma = 4 initially, then 8
no EAGLE tree yet
```

Do not start with tree mode. Tree mode adds branching, path selection, and KV management complexity. Chain mode is enough to verify correctness of hidden-state-conditioned proposal sampling and `p/q` weighting.

### Required EAGLE decode loop

At prefill/extend:

1. Run target prefill with hidden-state capture enabled.
2. Extract target hidden states from the required layers.
3. Feed those hidden states into the EAGLE draft.
4. Produce the first draft token and its logprob.
5. Store EAGLE hidden state in `SMCDraftInput` for subsequent decode.

At decode:

1. Start from stored EAGLE hidden state.
2. For each proposal step `t = 1..gamma`:
  - run EAGLE one step using previous hidden state and previous token embedding,
  - produce logits over full target vocab,
  - sample token from EAGLE proposal,
  - record `log q(token)` under the exact proposal distribution,
  - update EAGLE hidden state.
3. Run target verify on the proposed tokens.
4. Gather target logprobs `log p(token)`.
5. Compute:

```text
logprob_diff = sum_t log p_t - sum_t log q_t
```

1. Sample target bonus token as current SMC-SD does.
2. Return emitted tokens and `logprob_diff` to SMC coordinator.

### Correctness checks

Run tiny deterministic tests first:

```text
N = 1 or 2
gamma = 1 or 2
temperature = 0.7
max questions = 5
```

Validate:

```text
EAGLE logits are finite
sampled tokens are valid target-vocab ids
q logprobs are finite
p logprobs are finite
logprob_diff is finite
ESS is not NaN
output length advances by gamma+1 per round
no KV corruption after resampling
```

For `temperature=0`, decide whether this is a deterministic proposal. SMC importance weights with deterministic proposals are subtle because q is zero for non-argmax tokens. Use stochastic temperature for SMC proposal training/eval unless deterministic mode is explicitly handled.

### Comparison tests

Compare EAGLE chain mode against AR draft mode on a few prompts:

```text
same target
same N, gamma, temperature
same max tokens
```

Expected initially:

```text
EAGLE may not be better before SMC-native training.
But it must produce stable finite weights and reasonable outputs.
```

---

## Phase 2: train standard full-vocab EAGLE warm start

Before SMC-native proposal learning, EAGLE should be a reasonable token predictor.

Use SpecForge or an equivalent local training script.

### Hidden states in EAGLE3

EAGLE3 uses target hidden states from multiple layers:

```text
h_low, h_mid, h_high
```

Then:

```text
h_cat = concat(h_low, h_mid, h_high)
h_proj = fc(h_cat)
```

At each draft step, EAGLE consumes:

```text
projected target/draft hidden state
embedding(previous token)
```

and outputs:

```text
new draft hidden state
logits over draft vocabulary
```

For SMC-native EAGLE, `draft_vocab_size` should be the full target vocab.

### Training objective for warm start

Use standard EAGLE3 token prediction first:

```text
loss = CE / log-softmax loss against target next-token distribution
```

Use datasets appropriate to the target:

For Llama-8B/GSM8K:

```text
GSM8K train
MetaMath / math reasoning data
ShareGPT/UltraChat mixture for generality
```

For GPT-OSS/code/DS1000 later:

```text
code instruction data
StackOverflow-like Python/data-science prompts
BigCodeBench / CodeFeedback
DS1000-like synthetic prompts
```

Do not train directly on DS1000 test if clean evaluation matters.

### Full-vocab config

For Llama-3.1-8B EAGLE:

```json
{
  "architectures": ["LlamaForCausalLMEagle3"],
  "num_hidden_layers": 1,
  "hidden_size": 4096,
  "vocab_size": 128256,
  "draft_vocab_size": 128256
}
```

Initialize embeddings from target if supported. Initialize full-vocab `lm_head` from target if available and shape-compatible.

---

## Phase 3: SMC-native proposal learning objective

This is the most important phase. It is the direct bridge from `finetuning_proposal_toy` to SMC-SD.

### Objective

For each prefix state `s`, sample `R` candidate paths from the current proposal:

```text
y_i = y_{i,1:K} ~ q_old(. | s)
```

Compute:

```text
logq_old_i = sum_t log q_old(y_i,t | s, y_i,<t)
logp_i     = sum_t log p_target(y_i,t | s, y_i,<t)
logw_i     = logp_i - logq_old_i
w_i        = softmax(logw_i)
```

Train the new proposal with weighted path MLE:

```text
L_smc = - sum_i stopgrad(w_i) * sum_t log q_new(y_i,t | s, y_i,<t)
```

This differs from existing Stage B if Stage B uses only:

```text
weights_i = softmax(sum_t target_logp_i / tau)
```

SMC-native proposal training must include the proposal correction:

```text
weights_i = softmax(sum_t target_logp_i - sum_t old_draft_logq_i)
```

That is the key proposal-learning correction.

### Stabilized loss

Use:

```text
L_total = L_smc
        + lambda_anchor * KL(q_new || q_warm_start)
        + lambda_ce * CE(target top-1)
        + lambda_entropy * entropy_penalty
```

Suggested initial hyperparameters:

```text
lambda_anchor = 0.1 to 1.0
lambda_ce = 0.01 to 0.05
entropy penalty: prevent over-sharpening, do not push to uniform
learning rate = 1e-5 to 5e-5
gamma_train = 4 first, then 8
R candidates per prompt = 8 or 16
temperature = 0.7
```

### Why this should help SMC-SD

This objective directly minimizes the mismatch that causes SMC degeneracy:

```text
log p - log q
```

Expected improvements:

```text
lower log-weight variance
higher ESS/N
less resampling
better accuracy at same N/gamma
or same accuracy at larger gamma / fewer particles
```

---

## Phase 4: data collection for SMC-native EAGLE training

Create a script:

```text
scripts/draft_train/collect_eagle_smc_rollouts.py
```

For each prompt:

1. Run target prefill and capture hidden states.
2. Run current EAGLE proposal to sample `R` paths of length `gamma_train`.
3. Store:

```text
prompt_ids
prompt_len
initial target hidden state or references to hidden-state cache
candidate_tokens: [R, gamma_train]
draft_logps_old: [R, gamma_train]
target_logps: [R, gamma_train]
source / prompt_id
```

For memory reasons, consider two modes:

### Online mode

Recompute target hidden states during training.

Pros:

```text
less disk
simpler storage
```

Cons:

```text
requires target model during training
more GPU memory
```

### Offline mode

Store hidden states.

Pros:

```text
can train draft without target resident
```

Cons:

```text
huge disk usage
```

For Llama-8B experiments, online mode is probably fine. For GPT-OSS-20B, decide based on memory.

---

## Phase 5: train SMC-native EAGLE proposal

Create a script:

```text
scripts/draft_train/train_eagle_smc_proposal.py
```

Training loop:

1. Load warm-start EAGLE.
2. Load rollout batch.
3. Recompute or load initial hidden states.
4. Teacher-force candidate paths through EAGLE to compute `log q_new`.
5. Use stored `target_logps` and `draft_logps_old` to compute normalized SMC weights.
6. Apply weighted path MLE.
7. Add anchor/stability losses.
8. Log SMC metrics.

Important logged metrics:

```text
training/loss_total
training/loss_smc
training/loss_anchor
training/entropy
training/logw_mean
training/logw_var
training/ess_mean
training/path_top1_match
training/weighted_logq
training/target_path_logp
```

Validation:

```text
heldout logw variance
heldout ESS/N
heldout path top1 agreement
GSM8K SMC-SD accuracy / TPS sweep
```

---

## Phase 6: integrate and evaluate in SMC-SD

After training, run SMC-SD with EAGLE proposal.

Initial sweep:

```text
N = 4, 8, 12
gamma = 4, 8, 12
temperature = 0.7
max-running-requests = 24
cuda-graph-max-bs = 24
```

For GSM8K, compare to baseline:

```text
AR 1B -> 8B, N=12, gamma=8: ~75%, ~310 TPS
```

For each EAGLE checkpoint:

```text
EAGLE warm-start only
EAGLE + SMC proposal finetune
EAGLE + SMC proposal finetune + anchor variants
```

Report:

```text
accuracy
TPS
ESS/N
log-weight variance
resampling rate
```

Success criteria:

```text
same accuracy with higher TPS
or higher accuracy at similar TPS
or higher ESS/log-weight quality enabling larger gamma
```

---

## Phase 7: optional EAGLE tree proposal

Only after chain mode works and improves metrics.

Tree mode can propose multiple branches per step. For SMC, a tree proposal can be useful, but it complicates the proposal probability.

If using tree mode, ensure the path logprob is correct:

```text
log q(path) = sum over selected branch conditional probabilities
```

If path selection uses target scores or oracle selection, that selection distribution must either:

1. be included in `q`, or
2. be treated as an algorithmic heuristic rather than valid importance sampling.

Start with sampled tree paths where `q(path)` is exactly known.

Avoid oracle mode for final results unless clearly labeled as an upper bound.

---

## Relationship to `finetuning_proposal_toy`

`finetuning_proposal_toy` trains learned SMC proposals with weighted MLE:

```text
sample particles from q
score under target / potential
normalize importance weights
train q on high-weight particles
```

For SMC-SD, use the same principle:

```text
sample K-token paths from draft proposal q
score them under target p
compute p/q weights
train proposal with weighted path MLE
```

The key transferable idea is not the vLLM/LoRA implementation; it is the objective:

```text
L = - sum_i normalized_weight_i * log q(path_i)
```

This should be applied to EAGLE hidden-state-conditioned proposals.

---

## Potential pitfalls

### 1. Using EAGLE as a normal AR model

Invalid. EAGLE needs target hidden states.

### 2. Reduced vocabulary

Dangerous for SMC unless using a full-support mixture proposal.

### 3. Training only teacher-forced CE/KL

May improve token metrics but not SMC path metrics.

### 4. Ignoring proposal logprob in training weights

SMC-native proposal learning needs:

```text
logw = logp - logq_old
```

not just target path logprob.

### 5. Deterministic proposal at temperature 0

Importance sampling with deterministic q has support issues. Use stochastic temperature for SMC training/eval unless deterministic support is explicitly handled.

### 6. Target/draft temperature mismatch

Be explicit about whether weights use raw target logprobs or tempered target logprobs. Match the inference semantics.

### 7. Hidden-state OOD/exposure bias

Standard EAGLE hidden states are from target/gold trajectories. SMC proposal finetuning must expose EAGLE to its own sampled paths.

---

## Near-term task checklist

### A. Baseline and metrics

- Reproduce Llama 1B -> 8B GSM8K baseline: ~75%, ~310 TPS.
- Add SMC metrics logging: ESS/N, logw variance, resample frequency.
- Save metrics per decode step and aggregate per request.

### B. Correct EAGLE support

- Inspect existing `smc-eagle-tree` and `eagle3-cg` branches.
- Decide whether to port or rewrite on `smc-slot-refactor`.
- Implement EAGLE chain mode first.
- Verify full-vocab EAGLE config.
- Verify finite q logprobs and p logprobs.
- Verify KV/resampling correctness.

### C. Warm-start EAGLE

- Train/obtain full-vocab Llama-3.1-8B EAGLE warm start.
- Evaluate standard EAGLE token metrics.
- Run SMC-SD with EAGLE chain mode.

### D. SMC-native proposal learning

- Implement EAGLE SMC rollout collector.
- Implement weighted path-MLE trainer.
- Use weights `softmax(logp_target - logq_old)`.
- Add KL/CE/entropy anchors.
- Evaluate heldout ESS/logw metrics.

### E. End-to-end evaluation

- Sweep N/gamma/temperature for AR baseline, EAGLE warm start, and EAGLE SMC-trained.
- Compare accuracy/TPS frontier.
- Choose checkpoint based on frontier, not just CE.

---

## Expected first milestone

The first milestone is not beating the AR 1B baseline yet. It is:

```text
EAGLE chain mode runs in SMC-SD on Llama-3.1-8B.
q logprobs and target p logprobs are finite.
SMC metrics are logged.
No KV/resampling corruption.
GSM8K completes with stable output.
```

Once that is true, the second milestone is:

```text
SMC-native EAGLE training reduces heldout logw variance and improves ESS/N.
```

Only then should we expect accuracy/TPS improvements.