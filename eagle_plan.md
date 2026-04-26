# EAGLE / SMC-SD Proposal Learning Plan

> Status updated: 2026-04-26. This document now includes completed Phase 0 instrumentation, EAGLE chain porting, full-vocab EAGLE warm-start training, first SMC-native proposal-learning attempts, results, failures, and recommended next steps.

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

## Completed implementation and experiment log

### Branches and commits

Work has been done on:

```text
smcsd branch: phase0-smc-metrics
sglang submodule branch: phase0-smc-metrics
```

Relevant pushed PR URLs:

```text
SMCSD:  https://github.com/abdelfattah-lab/smcsd/pull/new/phase0-smc-metrics
SGLang: https://github.com/abdelfattah-lab/sglang/pull/new/phase0-smc-metrics
```

Important commits in `smcsd`:

```text
f09cbfd1e Add Phase 0 SMC diagnostics
c85d0a451 Port experimental SMC EAGLE chain support
89c11df37 Add full-vocab EAGLE expansion utility
f1a0f6fb2 Add SMC-native EAGLE proposal training scripts
```

Important commits in `3rdparty/sglang`:

```text
0ca7c9197 Add SMC metrics server arguments
5821cf5ea Add SMC EAGLE draft mode arguments
628f3d3b4 Route EAGLE3 draft configs in SMC
```

Untracked/local generated artifacts may exist and should not be committed blindly:

```text
checkpoints/
data/
todo.md
```

---

## Completed Phase 0: metrics instrumentation

Added SMC diagnostic flags to SGLang server args:

```bash
--smc-metrics
--smc-metrics-log-interval
--smc-metrics-jsonl
```

Implemented `SMCMetricLogger` in:

```text
smcsd/core/scheduler.py
```

Metrics logged per decode step:

```text
ESS
ESS / N
interval log-weight variance
cumulative log-weight variance
max normalized particle weight
resampled groups
resample jobs
```

Important implementation detail: metrics are snapshotted **before** fused resampling because the fused collect kernel mutates/zeros interval/cumulative weights for resampled rows.

Plumbed metrics through:

```text
smcsd/engine.py
scripts/accuracy_test_gsm8k.py
scripts/tps_benchmark_scripts/bench_offline_throughput.py
```

Baseline command used:

```bash
cd /home/yahya/smcsd
python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --particles 12 --gamma 8 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --num-questions 400 \
  --max-running-requests 24 \
  --cuda-graph-max-bs 24 \
  --smc-metrics \
  --smc-metrics-log-interval 25 \
  --smc-metrics-jsonl outputs/metrics_llama1b_8b_gsm8k_N12_g8_t07.jsonl
```

Baseline result:

```text
Accuracy:          299/400 (74.8%)
Invalid:           6/400 (1.5%)
Output throughput: 298.8 tok/s
Total tokens:      82898
Wall time:         277.4s
```

Baseline metrics:

```text
steps:             9964
ESS mean:          5.79 / 12
ESS/N mean:        0.482
ESS/N median:      0.464
ESS/N p10:         0.168
ESS/N p90:         0.833
logw_var mean:     13.73
logw_var median:   5.42
logw_var p90:      29.51
logw_var max:      941.27
max_weight mean:   0.345
max_weight median: 0.283
max_weight p90:    0.673
resampled steps:   54.6%
total resamples:   5441 groups
```

Interpretation:

```text
The AR 1B draft is imperfect but much better than all current EAGLE attempts.
It has moderate ESS but far lower log-weight variance than EAGLE.
```

---

## Completed Phase 1: EAGLE chain support port

Ported experimental EAGLE chain/tree code from `smc-eagle-tree` onto `phase0-smc-metrics` / `smc-slot-refactor` base. The tree modes are still experimental; use chain mode first.

EAGLE draft modes added:

```text
dense
eagle3
eagle3_chain
eagle3_tree_probe
eagle3_tree_smc
eagle3_tree_oracle
```

Main implementation files touched:

```text
smcsd/core/info.py
smcsd/core/req_state.py
smcsd/core/scheduler.py
smcsd/core/worker.py
smcsd/engine.py
smcsd/model_executor/smc_cuda_graph_runner.py
smcsd/model_executor/smc_model_runner.py
3rdparty/sglang/python/sglang/srt/server_args.py
3rdparty/sglang/python/sglang/srt/configs/model_config.py
```

Important SGLang loading fix:

```text
If a draft model config contains draft_vocab_size and architecture is LlamaForCausalLM,
route it to LlamaForCausalLMEagle3 so hot-vocab EAGLE checkpoints load correctly.
```

Without this fix, `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` fails loading with:

```text
AssertionError: self.org_vocab_size=128256 ... loaded_weight.shape[output_dim]=32000
```

because its config has:

```json
"architectures": ["LlamaForCausalLM"],
"vocab_size": 128256,
"draft_vocab_size": 32000
```

but its lm_head has only 32k rows.

---

## EAGLE correctness/evaluation experiments

### Experiment A: off-the-shelf hot-vocab EAGLE, N=2, gamma=1, q3

Command skeleton:

```bash
python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --draft-mode eagle3_chain \
  --particles 2 --gamma 1 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --num-questions 3 \
  --max-new-tokens 64 \
  --max-running-requests 4 \
  --cuda-graph-max-bs 4 \
  --smc-metrics
```

Result:

```text
Runs end-to-end, but output is nonsensical.
Accuracy: 0/3
Invalid: 1/3
```

Tiny metrics:

```text
ESS/N mean:      0.561
logw_var mean:   196.7
max_weight mean: 0.941
```

### Experiment B: off-the-shelf hot-vocab EAGLE, N=8, gamma=1, q20

Command:

```bash
python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --draft-mode eagle3_chain \
  --particles 8 --gamma 1 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --num-questions 20 \
  --max-new-tokens 128 \
  --max-running-requests 12 \
  --cuda-graph-max-bs 12 \
  --smc-metrics \
  --smc-metrics-jsonl outputs/metrics_eagle_chain_gsm8k_N8_g1_q20.jsonl
```

Result:

```text
Accuracy:          1/20 (5.0%)
Invalid:           5/20 (25.0%)
Output throughput: 138.9 tok/s
```

Metrics:

```text
ESS/N mean:        0.616
logw_var mean:     41.9
max_weight mean:   0.344
resampled steps:   36.9%
```

Conclusion:

```text
It runs, but off-the-shelf hot-vocab EAGLE is a very bad SMC proposal.
```

---

## Option A: full-vocab EAGLE support

Added utility:

```text
scripts/draft_train/expand_eagle3_to_full_vocab.py
```

Purpose:

```text
Convert a hot-vocab EAGLE3 checkpoint into a full-vocab checkpoint by creating
lm_head rows for the full target vocabulary, initializing non-hot rows from the
target lm_head, and copying trained hot rows into their d2t-mapped positions.
```

Command used:

```bash
python scripts/draft_train/expand_eagle3_to_full_vocab.py \
  --eagle yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --target meta-llama/Llama-3.1-8B-Instruct \
  --output checkpoints/eagle3_llama31_8b_full_vocab
```

Output:

```text
Wrote full-vocab EAGLE3 checkpoint to checkpoints/eagle3_llama31_8b_full_vocab
old draft vocab: 32000
target vocab: 128256
hidden: 4096
hot rows copied: 32000
non-hot init: target lm_head
```

Smoke result:

```text
Accuracy: 0/3
Invalid: 2/3
ESS/N mean: 0.545
logw_var mean: 212.5
```

Conclusion:

```text
Expansion gives full support, but not quality. Need actual full-vocab EAGLE training.
```

---

## Full-vocab EAGLE warm-start training

SpecForge was used for standard EAGLE warm-start training.

Relevant repo:

```text
/home/yahya/SpecForge
```

Relevant config:

```text
/home/yahya/SpecForge/configs/llama3-8B-eagle3-smc.json
```

Important property:

```json
"draft_vocab_size": 128256
```

SpecForge already supports:

```bash
--init-lm-head-from-target
```

which initializes full-vocab draft lm_head from target lm_head.

Prepared GSM8K data:

```bash
cd /home/yahya/SpecForge
python scripts/prepare_data.py \
  --dataset gsm8k \
  --sample-size 2000 \
  --output-path cache/dataset/gsm8k_train_2k.jsonl
```

Actual data file:

```text
/home/yahya/SpecForge/cache/dataset/gsm8k_train_2k.jsonl/gsm8k_train.jsonl
```

Smoke training succeeded and saved:

```text
/home/yahya/SpecForge/outputs/llama31-8b-eagle3-smc-gsm8k-smoke/epoch_0_step_2/
```

Full warm-start training command:

```bash
cd /home/yahya/SpecForge
CUDA_VISIBLE_DEVICES=0 \
TORCHINDUCTOR_CACHE_DIR=/home/yahya/SpecForge/cache/compiled_kernels \
PYTHONUNBUFFERED=1 \
nohup /home/yahya/miniconda3/envs/specforge/bin/torchrun \
  --standalone \
  --nproc_per_node 1 \
  scripts/train_eagle3.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-model-config configs/llama3-8B-eagle3-smc.json \
  --train-data-path cache/dataset/gsm8k_train_2k.jsonl/gsm8k_train.jsonl \
  --build-dataset-num-proc 4 \
  --dataloader-num-workers 0 \
  --output-dir outputs/llama31-8b-eagle3-smc-gsm8k-2k \
  --num-epochs 1 \
  --batch-size 1 \
  --tp-size 1 \
  --learning-rate 1e-4 \
  --max-length 1024 \
  --chat-template llama3 \
  --cache-dir cache \
  --attention-backend sdpa \
  --target-model-backend sglang \
  --log-interval 25 \
  --save-interval 500 \
  --eval-interval 999999 \
  --sglang-attention-backend fa3 \
  --sglang-mem-fraction-static 0.25 \
  --init-lm-head-from-target \
  --report-to none
```

Completed 1 epoch / 2000 examples. Final checkpoint:

```text
/home/yahya/SpecForge/outputs/llama31-8b-eagle3-smc-gsm8k-2k/epoch_0_step_2000/
```

Intermediate checkpoints:

```text
epoch_0_step_500/
epoch_0_step_1000/
epoch_0_step_1500/
epoch_0_step_2000/
```

Evaluation of warm-start checkpoint:

```bash
cd /home/yahya/smcsd
python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model /home/yahya/SpecForge/outputs/llama31-8b-eagle3-smc-gsm8k-2k/epoch_0_step_2000 \
  --draft-mode eagle3_chain \
  --particles 8 --gamma 1 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --num-questions 20 \
  --max-new-tokens 128 \
  --max-running-requests 12 \
  --cuda-graph-max-bs 12 \
  --smc-metrics \
  --smc-metrics-jsonl outputs/metrics_eagle_full_vocab_trained_gsm8k_N8_g1_q20.jsonl
```

Result:

```text
Accuracy:          3/20 (15.0%)
Invalid:           4/20 (20.0%)
Output throughput: 130.6 tok/s
```

Metrics:

```text
ESS/N mean:        0.523
logw_var mean:     64.2
max_weight mean:   0.423
resampled steps:   50.0%
```

Conclusion:

```text
Full-vocab warm-start training improves accuracy over off-the-shelf EAGLE (5% -> 15%)
but is still far from AR baseline and has worse log-weight variance.
```

---

## SMC-native proposal-learning scripts

Added:

```text
scripts/draft_train/collect_eagle_smc_rollouts.py
scripts/draft_train/train_eagle_smc_proposal.py
```

### `collect_eagle_smc_rollouts.py`

Initial version supports `gamma_train=1`.

For each prompt:

```text
1. target samples seed x0 from p(. | prompt)
2. target computes hidden state for prompt+x0
3. EAGLE proposes R candidate y1 tokens
4. store q_old(y1), p_target(y1), target hidden, seed token, candidate tokens
```

Important bug found and fixed:

```text
Using absolute target position ids in one-step EAGLE draft caused CUDA device-side asserts.
Fix: use local one-token draft position ids:
    pos = torch.zeros((1, 1), dtype=torch.long, device=device)
This matches SpecForge EAGLE training; target hidden state carries prefix information.
```

### `train_eagle_smc_proposal.py`

Initial pure SMC objective:

```text
logw_i = log p_target(y_i) - log q_old(y_i)
w_i = softmax(logw_i)
L = -sum_i stopgrad(w_i) log q_new(y_i)
```

Smoke tests passed.

---

## Pure SMC proposal finetune experiment

Rollout collection:

```bash
python scripts/draft_train/collect_eagle_smc_rollouts.py \
  --target meta-llama/Llama-3.1-8B-Instruct \
  --draft /home/yahya/SpecForge/outputs/llama31-8b-eagle3-smc-gsm8k-2k/epoch_0_step_2000 \
  --data /home/yahya/SpecForge/cache/dataset/gsm8k_train_2k.jsonl/gsm8k_train.jsonl \
  --output-dir data/eagle_smc_rollouts_gsm8k_2k_R8_g1 \
  --num-prompts 2000 \
  --num-candidates 8 \
  --temperature 0.7 \
  --max-prompt-tokens 512 \
  --shard-size 256
```

Result:

```text
Done: 2000 prompts, 8 shards
```

Training:

```bash
python scripts/draft_train/train_eagle_smc_proposal.py \
  --init /home/yahya/SpecForge/outputs/llama31-8b-eagle3-smc-gsm8k-2k/epoch_0_step_2000 \
  --data-dir data/eagle_smc_rollouts_gsm8k_2k_R8_g1 \
  --output-dir outputs/eagle_smc_proposal_gsm8k_2k_R8_g1 \
  --batch-size 8 \
  --epochs 1 \
  --lr 2e-5 \
  --warmup-steps 20 \
  --anchor-weight 0.1 \
  --log-interval 10 \
  --save-interval 200
```

Final checkpoint:

```text
outputs/eagle_smc_proposal_gsm8k_2k_R8_g1/final
```

Evaluation result:

```text
Accuracy:          0/20 (0.0%)
Invalid:           1/20 (5.0%)
Output throughput: 123.0 tok/s
```

Metrics:

```text
ESS/N mean:        0.533
logw_var mean:     89.5
max_weight mean:   0.413
resampled steps:   49.0%
```

Conclusion:

```text
Pure SMC one-step update destabilized proposal quality and made task accuracy worse.
```

---

## Hybrid target-topk + SMC + anchor experiment

Modified rollout collector to store:

```text
target_topk_ids
target_topk_logps
```

Modified trainer objective to:

```text
L = topk_weight * KL(target_topk || q_new)
  + smc_weight  * SMC_weighted_MLE
  + anchor_weight * KL(q_old || q_new on sampled candidates)
```

Smoke tests passed.

Full rollout collection:

```bash
python scripts/draft_train/collect_eagle_smc_rollouts.py \
  --target meta-llama/Llama-3.1-8B-Instruct \
  --draft /home/yahya/SpecForge/outputs/llama31-8b-eagle3-smc-gsm8k-2k/epoch_0_step_2000 \
  --data /home/yahya/SpecForge/cache/dataset/gsm8k_train_2k.jsonl/gsm8k_train.jsonl \
  --output-dir data/eagle_smc_rollouts_gsm8k_2k_R8_g1_topk64 \
  --num-prompts 2000 \
  --num-candidates 8 \
  --target-topk 64 \
  --temperature 0.7 \
  --max-prompt-tokens 512 \
  --shard-size 256
```

Training:

```bash
python scripts/draft_train/train_eagle_smc_proposal.py \
  --init /home/yahya/SpecForge/outputs/llama31-8b-eagle3-smc-gsm8k-2k/epoch_0_step_2000 \
  --data-dir data/eagle_smc_rollouts_gsm8k_2k_R8_g1_topk64 \
  --output-dir outputs/eagle_smc_hybrid_gsm8k_2k_R8_g1 \
  --batch-size 8 \
  --epochs 1 \
  --lr 5e-6 \
  --warmup-steps 20 \
  --smc-weight 0.1 \
  --topk-weight 1.0 \
  --anchor-weight 1.0 \
  --log-interval 10 \
  --save-interval 200
```

Final checkpoint:

```text
outputs/eagle_smc_hybrid_gsm8k_2k_R8_g1/final
```

Evaluation:

```bash
python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model outputs/eagle_smc_hybrid_gsm8k_2k_R8_g1/final \
  --draft-mode eagle3_chain \
  --particles 8 --gamma 1 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --num-questions 20 \
  --max-new-tokens 128 \
  --max-running-requests 12 \
  --cuda-graph-max-bs 12 \
  --smc-metrics \
  --smc-metrics-jsonl outputs/metrics_eagle_hybrid_gsm8k_N8_g1_q20.jsonl
```

Result:

```text
Accuracy:          2/20 (10.0%)
Invalid:           2/20 (10.0%)
Output throughput: 127.2 tok/s
```

Metrics:

```text
ESS/N mean:        0.527
logw_var mean:     97.4
max_weight mean:   0.414
resampled steps:   49.3%
```

Conclusion:

```text
Hybrid loss did not fix proposal quality. It reduced target top-k KL during training,
but generation quality and SMC metrics remained poor or worse.
```

---

## Summary table of results so far

| Draft / training | Eval config | Accuracy | TPS | ESS/N mean | logw_var mean | max_w mean | Resample rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| AR Llama-3.2-1B -> Llama-3.1-8B | N=12, gamma=8, q400 | 74.8% | 298.8 | 0.482 | 13.7 | 0.345 | 54.6% |
| Off-shelf hot-vocab EAGLE3 | N=8, gamma=1, q20 | 5.0% | 138.9 | 0.616 | 41.9 | 0.344 | 36.9% |
| Expanded full-vocab EAGLE3, no training | N=2, gamma=1, q3 | 0.0% | 121.4 | 0.545 | 212.5 | 0.959 | 0.0% |
| Full-vocab EAGLE warm-start on GSM8K 2k | N=8, gamma=1, q20 | 15.0% | 130.6 | 0.523 | 64.2 | 0.423 | 50.0% |
| Pure SMC one-step finetune | N=8, gamma=1, q20 | 0.0% | 123.0 | 0.533 | 89.5 | 0.413 | 49.0% |
| Hybrid topk + SMC one-step finetune | N=8, gamma=1, q20 | 10.0% | 127.2 | 0.527 | 97.4 | 0.414 | 49.3% |

---

## Current conclusion

The EAGLE integration is functional enough to run, and full-vocab EAGLE checkpoints can be trained and loaded. However:

```text
All current EAGLE proposals are far worse than the AR 1B draft for SMC-SD.
```

The one-step SMC proposal-learning setup did not work. Even the hybrid top-k + SMC objective did not improve downstream SMC-SD performance.

Main likely reason:

```text
The collector/trainer only trains isolated one-step EAGLE states.
But SMC-SD inference uses recurrent EAGLE hidden-state dynamics across accepted tokens.
```

So the training distribution is still mismatched from inference. We need recurrent path-level training.

---

## Recommended next steps for the next agent

### 1. Stop single-step `gamma_train=1` training

Do not spend more time tuning the current one-step objective unless doing a very small ablation. It is unlikely to solve the core mismatch.

### 2. Implement recurrent EAGLE rollout collection

Create collector v2:

```text
scripts/draft_train/collect_eagle_smc_rollouts_v2.py
```

For each prompt:

1. Run target prefill.
2. Sample seed token `x0` from target.
3. Capture target hidden state for `prompt + x0`.
4. Run EAGLE recurrently for `gamma_train = 2 or 4` steps:
   ```text
   h_t, logits_t = EAGLE(h_{t-1}, token_{t-1})
   y_t ~ q_old(logits_t)
   store log q_old(y_t)
   ```
5. Score the full sampled path with target in one teacher-forced pass:
   ```text
   log p_target(y_1:K | prompt, x0)
   ```
6. Store per-step:
   ```text
   candidate paths [R, gamma_train]
   draft_logps_old [R, gamma_train]
   target_logps [R, gamma_train]
   target_topk_ids/logps per step if using KL
   initial target hidden state
   seed token
   ```

### 3. Implement recurrent path-level EAGLE trainer

Create trainer v2:

```text
scripts/draft_train/train_eagle_smc_recurrent.py
```

Teacher-force full candidate paths through EAGLE recurrence, matching inference:

```text
h_0 = initial target hidden state
for t in 1..K:
    logits_t, h_t = EAGLE(h_{t-1}, y_{t-1})
    compute log q_new(y_t)
```

Loss:

```text
logw_i = sum_t log p_i,t - sum_t log q_old_i,t
w_i = softmax(logw_i)

L_smc = - sum_i stopgrad(w_i) * sum_t log q_new_i,t
```

Add stabilizers:

```text
L = alpha * recurrent_target_topk_KL
  + beta  * L_smc
  + gamma * KL(q_old || q_new on sampled path tokens)
```

Start conservative:

```text
gamma_train = 2
R = 8
lr = 1e-6 to 5e-6
alpha = 1.0
beta = 0.01 to 0.1
gamma = 1.0 to 5.0
```

### 4. Evaluate at gamma=1 first, then gamma=2

Do not jump to gamma=8.

Evaluation sequence:

```text
N=8, gamma=1, q20 GSM8K
N=8, gamma=2, q20 GSM8K
then larger q100/q400 if promising
```

Success criterion:

```text
Accuracy improves above warm-start 15% on q20
logw_var mean decreases below 64
max_weight decreases
outputs become less degenerate
```

### 5. If recurrent EAGLE still fails

Then likely issue is not just objective but proposal family or implementation. Options:

1. Train longer standard full-vocab EAGLE on much larger data before SMC finetune.
2. Implement full-support mixture proposal using hot-vocab EAGLE + AR fallback.
3. Revisit target hidden-state capture / position semantics in SMC EAGLE chain.
4. Compare EAGLE logits from SMC runtime vs SpecForge training on the same state to confirm they match.

---

## Useful paths for continuation

SMCSD repo:

```text
/home/yahya/smcsd
```

SpecForge repo:

```text
/home/yahya/SpecForge
```

Baseline metrics:

```text
/home/yahya/smcsd/outputs/metrics_llama1b_8b_gsm8k_N12_g8_t07.jsonl
```

Warm-start EAGLE checkpoint:

```text
/home/yahya/SpecForge/outputs/llama31-8b-eagle3-smc-gsm8k-2k/epoch_0_step_2000
```

Pure SMC proposal checkpoint:

```text
/home/yahya/smcsd/outputs/eagle_smc_proposal_gsm8k_2k_R8_g1/final
```

Hybrid proposal checkpoint:

```text
/home/yahya/smcsd/outputs/eagle_smc_hybrid_gsm8k_2k_R8_g1/final
```

Important scripts:

```text
scripts/accuracy_test_gsm8k.py
scripts/draft_train/expand_eagle3_to_full_vocab.py
scripts/draft_train/collect_eagle_smc_rollouts.py
scripts/draft_train/train_eagle_smc_proposal.py
```

SpecForge full-vocab config:

```text
/home/yahya/SpecForge/configs/llama3-8B-eagle3-smc.json
```

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