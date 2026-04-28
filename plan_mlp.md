# Layerwise MLP-Head KL Distillation — Plan

**Branch**: `layerwise-kl` (off `phase0-smc-metrics`)
**Created**: 2026-04-28
**Status**: planning. No layerwise code written yet.

This document is the entry point for an agent picking up this work cold.
Read [updates.md](updates.md) first for the full project context (what we
tried before this direction, why, what worked, what didn't). This file is
focused on what to do next.

---

## TL;DR

For each intermediate layer ℓ ∈ {1, …, L−1} of a frozen target LM, train a
small MLP head `g_ℓ` such that

```
  KL( softmax(target.lm_head(h_L))  ‖  softmax(g_ℓ(h_ℓ)) )
```

is minimized at every position, where `h_ℓ` is the hidden state at the  
output of layer ℓ. After convergence, use each `g_ℓ` as the proposal `q`  
in SMC-SD and report how each layer performs.

---

## Why this is promising

Compared to the EAGLE3 work documented in [updates.md](updates.md), the
early-exit MLP approach is:


|                         | EAGLE3 (prior)                                              | Early-exit MLP (this plan)                                            |
| ----------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------- |
| Proposal architecture   | Separate small recurrent draft model on aux hidden states   | Just an MLP per layer of the same target                              |
| Trajectory at inference | EAGLE recurses on its own state                             | Target's own forward, stopped at layer ℓ                              |
| Exposure bias           | Real (EAGLE drifts)                                         | None — same target, same hidden states                                |
| Training objective      | On-policy KL between target and EAGLE on EAGLE-rolled paths | Off-policy KL between final head and intermediate head, position-wise |
| Code complexity         | High                                                        | Low for training; partial-forward inference is the engineering piece  |
| Per-layer ablation      | Not naturally available                                     | Direct readout: KL vs layer index                                     |


The training objective is **the right thing to optimize directly** — we
want a proposal `q` close to target `p` in KL, and we just minimize that.
No exposure bias, no separate model, no recurrent state to manage.

This builds on the same KL-paper insight that drove our `train_eagle_kl_onpolicy.py`
trainer (which DID work — pushed γ=2 from 52% to 64% on Llama-8B GSM8K),
just applied to a different proposal family.

---

## What's already on this branch (carry-over from phase0-smc-metrics)

Everything below is reusable infrastructure. Don't re-port any of it.

### Data

- `cache/dataset/llama31_8b_smc_warmstart_200k.jsonl` — 199,973 mixed
conversations (7.5k GSM8K + 7.5k hendrycks_math + 185k
PerfectBlend-Regenerated-Llama-3.1-8B-Instruct). On `/mnt/raid0` via
symlink. Use **the first ~199,773 rows for training; the LAST 200 rows
are the heldout set used by `scripts/heldout_diag.py`**.
- `scripts/prepare_smc_warmstart_mix.py` — rebuilds the 200k file from
HF datasets if needed.

### Heldout multi-domain evaluator

- `scripts/heldout_diag.py` — runs SMC-SD on the last 200 prompts of the
200k file. Reports per-cycle chain diagnostics: top-20 overlap, target
rank, target_minus_draft_lp, ESS/N, logw_var. **Reusable as-is** once
we have an SMC-SD inference path that takes our trained MLP heads as
the proposal.
- Baseline numbers for the EAGLE 200k+curriculum 2-layer ckpt:
top-20 overlap 0.50, lp gap median −1.76, ESS/N median 0.41,
logw_var median 58.9. These are the numbers to beat.

### Verification harness

- `scripts/verify_eagle_integration.py` — Tests A/B/C as a numerical
ground truth. Reuse the v2-collector helper `_eagle_prefill` for
reference, but more importantly **port Test A**: feed text through
target, capture per-layer hidden states + final logits, verify your
MLP-head training matches plain HF. This is the gold-standard
correctness check.

### Storage layout

- `outputs/`, `data/`, `checkpoints/` are symlinks to `/mnt/raid0/yahya/…`.
Always write large artifacts under those paths; never to root.
- `~/.cache/huggingface/` is also symlinked to raid0.
- `HF_HOME=/mnt/raid0/yahya/huggingface_cache` set in `~/.bashrc`.

### Conda envs

- `/home/yahya/miniconda3/envs/smcsd/bin/python` — sglang + smcsd, runs
the SMC-SD engine and the existing trainers.
- `/home/yahya/miniconda3/envs/specforge/bin/python` — has SpecForge +
flash_attn deps. Used by `train_eagle_kl_onpolicy.py` and the
collectors. **For pure-HF training (which is what the layerwise
training is), the smcsd env is sufficient.**

### Decision log

- `updates.md` — append a new "## Layerwise MLP-Head KL Distillation"
section as you progress. Use the same format as existing entries:
dated entries, headline numbers in tables, what worked / what didn't /
why.

---

## Architecture choices (defaults — confirm with advisor if uncertain)

### Per-layer head

```
g_ℓ(h) = LayerNorm(h)
       → Linear(hidden, hidden)         # ~16M params at hidden=4096
       → SiLU
       → Linear(hidden, hidden)         # ~16M
       → LayerNorm
       → tied target.lm_head            # SHARED, frozen
```

Parameters per head ≈ 33M for Llama-3.1-8B (hidden=4096). 32 layers
(Llama-8B has 32 transformer blocks; we train heads on layers 1..L−1,
so L=32 means 31 heads). Total trainable ≈ 1B params, plus the shared
frozen target lm_head.

**Sharing the lm_head is critical**. If each head had its own
`Linear(hidden, vocab)` projection, that's ~525M extra params per layer
(vocab × hidden), and we'd run out of memory long before convergence.

If the advisor wants something simpler ("just the lm_head with a
per-layer bias/projection"), substitute that. The architecture above is
a reasonable default that's been validated as the kind of head that
EAGLE3 uses.

### Loss

```
loss_ℓ_per_position = sum_v p(v) * (log p(v) - log q_ℓ(v))      # forward KL
                      where p = softmax(target.lm_head(h_L) / T)
                            q_ℓ = softmax(g_ℓ(h_ℓ) / T)
total_loss = sum over positions, then sum over layers ℓ
```

- **Forward KL** (`p ‖ q`), mode-covering. Matches the Rao-Blackwell
estimator from Amini et al. 2025 (which is the paper Ryan co-authored,
the same one we used for `train_eagle_kl_onpolicy.py`).
- Temperature T = 0.7 (matches inference and our existing recipe; see
`accuracy_test_gsm8k.py --temperature 0.7`).
- Sum (not mean) over the L−1 layers. Equivalent gradient signal per
head as training each independently.

### Optimizer

- AdamW, lr 1e-4 to 5e-4 (reasonable for fresh MLP heads — much higher
than the 5e-6 we used for fine-tuning EAGLE). Tune by watching the
loss.
- Linear warmup (200 steps) then cosine decay over the full training.
- Gradient clipping at 1.0.
- bf16 mixed precision (target frozen in bf16; MLP heads keep fp32
master weights).

### Training data + config

- Use the same 200k mixed jsonl. **Skip the last 200 prompts** —
they're heldout for `scripts/heldout_diag.py`.
- max_length 1024 (matches our EAGLE warmstart). Truncate longer; pad
shorter to nothing (no per-batch padding needed if you batch by
similar length).
- batch_size 2 per GPU is conservative; 4 should fit on H100. Tune
upward until OOM.
- ~4-6 epochs over 200k examples is probably enough to converge. If
budget is tight, 1-2 epochs is fine for a first pass.
- Save checkpoints every 5k steps in case of failure.

---

## Stages

### Stage 0 — Setup verification (~1 hour)

**Goal**: confirm infrastructure is in place before training.

TODOs:

- `git checkout layerwise-kl` and verify branch is current.
- Verify `cache/dataset/llama31_8b_smc_warmstart_200k.jsonl` exists
and has 199,973 lines.
- Verify `~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/`
exists (already cached on raid0).
- Run `scripts/heldout_diag.py` against the existing
`outputs/eagle_kl_onpolicy_200k_2layer_curriculum/final` ckpt to
confirm the heldout pipeline works end-to-end. Expected:
γ=4 q=200 ≈ ESS/N median 0.41, logw_var median 58.9.
- Sanity-check disk: `df -h /mnt/raid0` should show >25T free.

### Stage 1 — Training script (~half day to write, ~few hours to run)

**Goal**: `scripts/draft_train/train_layerwise_kl.py`.

Implementation outline (write this from scratch — not built on top of
`train_eagle_kl_onpolicy.py`):

```python
# Pseudocode — see scripts/heldout_diag.py for tokenizer/chat-template
# patterns and scripts/draft_train/train_eagle_kl_onpolicy.py for the
# data-loader / optimizer pattern.

# 1. Load Llama-3.1-8B-Instruct with output_hidden_states capability,
#    bf16, frozen.
target = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
).eval()
for p in target.parameters():
    p.requires_grad_(False)

L = target.config.num_hidden_layers   # 32 for Llama-8B
H = target.config.hidden_size         # 4096
V = target.config.vocab_size          # 128256

# 2. Define per-layer MLP heads. Shared lm_head from target.
class LayerHead(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.norm1 = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, h)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(h, h)
        self.norm2 = nn.LayerNorm(h)
    def forward(self, x):
        return self.norm2(self.fc2(self.act(self.fc1(self.norm1(x)))))

heads = nn.ModuleList([LayerHead(H) for _ in range(L - 1)])
heads = heads.to("cuda:0", dtype=torch.bfloat16)

optim = torch.optim.AdamW(heads.parameters(), lr=2e-4, betas=(0.9, 0.95))

# 3. Training step.
def step(input_ids):
    with torch.no_grad():
        out = target(input_ids, output_hidden_states=True)
        # out.hidden_states is a tuple of (L+1) tensors:
        #   hs[0] = embedding output
        #   hs[ℓ+1] = output of transformer layer ℓ (1-indexed in HF)
        # We want layer-ℓ outputs for ℓ = 1..L-1.
        hs = out.hidden_states           # tuple of (B, T, H), len L+1
        z_L = target.lm_head(hs[L]).float()    # (B, T, V)
        # Mask out pad positions before computing KL.
        log_p = F.log_softmax(z_L / 0.7, dim=-1)
        p = log_p.exp()

    total_loss = 0.0
    for ell in range(1, L):              # train heads at layers 1..L-1
        h_ell = hs[ell]                  # (B, T, H), no grad
        # Apply per-layer MLP -> tied lm_head.
        proj = heads[ell - 1](h_ell)
        z_ell = target.lm_head(proj).float()
        log_q = F.log_softmax(z_ell / 0.7, dim=-1)
        # Per-position KL: sum_v p(v) (log p(v) - log q(v))
        kl_per_pos = (p * (log_p.detach() - log_q)).sum(dim=-1)  # (B, T)
        total_loss = total_loss + kl_per_pos.mean()

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    return total_loss.item()
```

CLI / args:

- `--target` (default Llama-3.1-8B)
- `--data` (default the 200k jsonl)
- `--max-length` (default 1024)
- `--batch-size` (default 2)
- `--lr` (default 2e-4)
- `--epochs` (default 3)
- `--save-interval` (default 5000)
- `--output-dir` (under `outputs/layerwise_kl_<target>/`)
- `--temperature` (default 0.7)
- `--hold-out-last` (default 200; skip the last N rows for heldout
consistency)

Memory considerations:

- Holding all `hidden_states` simultaneously is ~33 layers × 4 KB × 1024 × bs = 134 MB per example for Llama-8B at bs=1, max_len=1024 in bf16. Feasible.
- Backward through 31 MLP heads at once is the dominant memory cost.
With bs=2 and the architecture above, expect ~30-40 GB used on an
H100 (well under the 80 GB budget).

Multi-GPU: DDP across 4 GPUs (each replicates the target + all heads;
data-parallel over batches). Standard PyTorch DDP via `torchrun --nproc_per_node 4`. Don't use FSDP for the heads themselves — they're
small enough to fit. Do gradient sync across ranks.

TODOs:

- Write `scripts/draft_train/train_layerwise_kl.py`.
- Smoke test on 100 steps to confirm forward/backward works.
- Multi-GPU DDP run on the full 200k for ~3 epochs.
Expected wall time on 4×H100 ≈ 4-6 hours.
- Save final checkpoint to `outputs/layerwise_kl_llama31_8b/final/`.
Save state for each head separately (so Stage 3 can load just the
relevant ℓ).

### Stage 2 — Per-layer KL evaluation (~30 min)

**Goal**: produce the per-layer KL curve. This is the **first
deliverable** Ryan asked for ("how well each head works").

Reuse the heldout slice (last 200 prompts of the 200k file). For each
ℓ ∈ {1, …, L−1}, on heldout:

1. Run target with `output_hidden_states=True` over the prompt
  (and optionally a sampled or gold continuation).
2. For each position, compute `KL( softmax(target.lm_head(h_L)/T)
  || softmax(g_ℓ(h_ℓ)/T) )` using the trained head.
3. Report mean KL across positions (and, separately, over only the
  high-probability positions where SMC samples concentrate).

Output: a JSON or CSV with `(layer_idx, mean_kl, median_kl, weighted_kl)` and a printed table. Also useful: `target_minus_q_lp` at
the gold continuation tokens — directly comparable to the existing
EAGLE numbers in `outputs/heldout_baseline_200kcurr/summary.json`.

TODOs:

- Write `scripts/draft_train/eval_layerwise_kl.py`.
- Run on heldout. Save `outputs/layerwise_kl_llama31_8b/per_layer_kl.json`.
- Print + save the curve. Identify the layer ℓ* with the lowest KL
(or most favorable accuracy/cost tradeoff).
- **Append a dated entry to `updates.md` decision log**: report
per-layer KL curve, mark ℓ* as the candidate proposal layer, and
note any layer-range that's cheap-and-good (e.g. "layers 24-28
have KL < 0.5").

### Stage 3 — SMC-SD with each head as proposal (~few days of engineering)

**Goal**: actually run SMC-SD with the trained heads as the proposal
distribution and measure accuracy + ESS/N + TPS at γ-sweep. This is
the **second deliverable** Ryan asked for.

This is the heavier engineering piece. Two paths:

#### Path A — Pure-HF inference (slow, correct, easier to write)

Standalone Python script that:

1. For each prompt, runs target through layer ℓ once (capturing h_ℓ
  and the KV cache for layers 1..ℓ).
2. Applies `g_ℓ` to h_ℓ → logits → samples y_1.
3. Continues drafting: feeds y_1's embedding back into layers 1..ℓ
  using the cached KV, gets new h_ℓ, applies g_ℓ, samples y_2.
   Repeat γ times. **This requires manually running target's first ℓ
   transformer blocks.**
4. After γ draft tokens, runs the FULL target on
  `[prompt + y_1..y_γ]` (re-prefilling) to get target's true
   distribution and finalize the SMC weight.

This bypasses sglang entirely. Pro: simple, correct. Con: slow — won't
give us competitive TPS measurements.

For the **first SMC-SD measurement**, this is fine. We just want
accuracy, ESS/N, and logw_var. TPS is nice-to-have but the first
question is "does the proposal work for SMC at all".

#### Path B — sglang chain-decode integration (fast, harder)

Add a new draft mode to `smcsd/core/worker.py` (e.g.,
`early_exit_layer<ℓ>`). The mode runs target's first ℓ transformer
blocks for each draft step, applies the trained MLP head, samples,
and updates the partial KV cache. Verify uses the existing sglang
verify path (full target forward).

This requires:

- Slicing the target into "first-ℓ" and "last L−ℓ" halves.
- Maintaining a partial KV cache for layers 1..ℓ during drafting.
- Plumbing the trained MLP heads as a separate model the worker can
call.

Estimated effort: 1-2 weeks. **Don't do this unless Path A's results
look promising.**

TODOs:

- Write `scripts/eval_smc_sd_with_layerhead.py` (Path A).
- Run SMC-SD on GSM8K q100 N=12 at γ ∈ {2, 4, 8} for 3-5 chosen
layers (e.g., L/4, L/2, 3L/4, L−2). Report accuracy + ESS/N +
logw_var per layer per γ.
- Also run `heldout_diag.py`-style multi-domain heldout for the
best layer's head.
- Append the table to `updates.md`.

If Path A's results are promising:

- Path B: integrate into sglang chain decode for fast TPS
measurements.

### Stage 4 — Decision: ship, iterate, or pivot

Compare against:

- AR-1B baseline: 74% / 306 TPS at γ=8 (Pareto target for Llama-8B)
- EAGLE 2-layer best: γ=2 67% / 157 TPS, γ=8 33% / 217 TPS
- Heldout target_minus_draft_lp median: −1.76 (EAGLE), gate ≥ −1.90

If layerwise heads beat EAGLE on heldout multi-domain KL: **ship the
recipe**, write up. Pivot to GPT-OSS-20B (the target where this recipe
has unique value, since no good AR sibling).

If not: investigate. Could be (a) MLP architecture too small,
(b) need on-policy refinement, (c) certain layers are noisy. Iterate
in updates.md.

---

## Gotchas / things to know

1. **HF `output_hidden_states` is 0-indexed and includes the embedding
  output**. `hs[0]` is post-embedding, `hs[1]` through `hs[L]` are
   transformer-block outputs. Train heads on `hs[1]..hs[L-1]`; use
   `hs[L]` (last block output, fed to lm_head) as the teacher h.
2. **The dense AR path's target temperature bug was fixed on 2026-04-28**.
  See the `c818a3f26` commit. `worker.py:520` now applies
   `/ self.smc_target_temperature`. If you implement Path B in Stage 3,
   make sure the new draft mode applies the temperature correctly too.
3. **Heldout slice**: the LAST 200 rows of the 200k jsonl. Hold them
  out from training. `scripts/heldout_diag.py` uses them by default.
4. **Storage**: all artifacts must land on `/mnt/raid0` via the
  `outputs/`, `data/`, `checkpoints/` symlinks. **Never** write large
   files under `/home/yahya/` directly — root is small and gets full.
5. **gpt-oss chat template** (if you pivot to GPT-OSS-20B later): the
  `gpt-oss` template uses the openai-harmony parser which expects
   role-specific channel tags (`assistant_analysis`, `assistant_final`,
   etc.) and rejects plain `assistant`. Use `gpt-oss-naive` if your
   data has plain roles. See the smoke-test history in
   `SpecForge/logs/gptoss_smoke.log`.
6. **bf16 numerical noise in target attention**: per Test B in the
  verification suite, sglang and HF disagree by ~0.5 nat mean / 0.09
   prob-weighted on the same input. This is bf16 + kernel differences,
   not a bug. For layerwise *training* (which is pure HF), this is a
   non-issue. For Path B SMC-SD (which uses sglang), expect the same
   small drift.
7. **Gradient checkpointing**: if memory is tight at training time,
  wrap the per-layer MLP application in `torch.utils.checkpoint`. The
   target is frozen so doesn't checkpoint; the MLP heads can.

---

## Open questions for the advisor (already in flight)

These were in the message sent to Ryan on 2026-04-28. Don't act on
defaults until he replies if any of these matter to you:

- KL direction (forward vs reverse). Default: forward `KL(p || q)`.
- MLP architecture (tied lm_head + small bottleneck, or unconstrained
per-layer head). Default: tied lm_head + bottleneck.
- Joint vs per-layer training. Default: joint (single target forward,
summed loss over all heads).
- "Show with SMC-SD" — KL ablation only, or full inference. Default:
KL first (Stage 2), then inference for selected layers (Stage 3).
- Target model — Llama-3.1-8B or other. Default: Llama-3.1-8B
(infrastructure ready).

---

## Commands cheat sheet

```bash
# Verify environment
df -h /mnt/raid0                         # should show >25T free
ls cache/dataset/                        # 200k jsonl present
nvidia-smi                               # 4×H100 available

# Smoke test (50 steps, 1 GPU, batch 1)
CUDA_VISIBLE_DEVICES=0 \
/home/yahya/miniconda3/envs/smcsd/bin/python \
  scripts/draft_train/train_layerwise_kl.py \
  --target meta-llama/Llama-3.1-8B-Instruct \
  --data cache/dataset/llama31_8b_smc_warmstart_200k.jsonl \
  --output-dir outputs/layerwise_kl_smoke \
  --max-steps 50 --batch-size 1

# Full 4-GPU DDP training (3 epochs)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/yahya/miniconda3/envs/smcsd/bin/torchrun --standalone --nproc_per_node 4 \
  scripts/draft_train/train_layerwise_kl.py \
  --target meta-llama/Llama-3.1-8B-Instruct \
  --data cache/dataset/llama31_8b_smc_warmstart_200k.jsonl \
  --output-dir outputs/layerwise_kl_llama31_8b \
  --epochs 3 --batch-size 2 --lr 2e-4

# Per-layer KL eval on heldout
CUDA_VISIBLE_DEVICES=0 \
/home/yahya/miniconda3/envs/smcsd/bin/python \
  scripts/draft_train/eval_layerwise_kl.py \
  --heads outputs/layerwise_kl_llama31_8b/final \
  --output outputs/layerwise_kl_llama31_8b/per_layer_kl.json
```

---

## File map (at the end of this work)

New files:

- `scripts/draft_train/train_layerwise_kl.py` — Stage 1 trainer
- `scripts/draft_train/eval_layerwise_kl.py` — Stage 2 evaluator
- `scripts/eval_smc_sd_with_layerhead.py` — Stage 3 (Path A) inference
- `outputs/layerwise_kl_llama31_8b/` — checkpoints + heads (on raid0)
- `outputs/layerwise_kl_llama31_8b/per_layer_kl.json` — Stage 2 result
- `outputs/layerwise_smc_sd_sweep/` — Stage 3 results

Modified:

- `updates.md` — append "## Layerwise MLP-Head KL Distillation" section
with dated decision-log entries as you progress.

If Path B (sglang integration):

- `smcsd/core/worker.py` — new `early_exit_layer*` draft mode
- `smcsd/engine.py` — plumbing for the new mode
- `smcsd/core/early_exit.py` — partial-forward inference primitive

