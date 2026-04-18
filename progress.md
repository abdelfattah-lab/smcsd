# SMCSD EAGLE Draft Integration Progress

Date: 2026-04-17
Branch: `eagle`

## Goal

Add EAGLE-based drafting to SMCSD without rewriting the existing SMC
scheduler, slot-state, or resampling logic. The first milestone is to define
an SMC-compatible stochastic proposal clearly enough that we can implement and
test it in small steps.

## Current Status

- Checkpoint 1 is now defined at the design level.
- Checkpoint 2 interface plumbing is implemented:
  - `smc_draft_kind = {lm, eagle}`
  - `smc_eagle_topk`
  - clear validation for unsupported / invalid EAGLE-SMC configurations
- Checkpoint 3 draft-state plumbing is implemented:
  - `SMCEagleDraftInputV2` exists as a distinct SMC-side carrier
  - `ScheduleBatchSMC` can now construct either LM or EAGLE draft carriers
  - `SMCWorkerV2` accepts either carrier type without changing LM behavior
- Checkpoint 4 prefill-only EAGLE initialization is implemented:
  - EAGLE mode now captures target hidden states during prefill
  - the draft model runs one EAGLE-style prefill step to initialize
    `hidden_states`, `topk_p`, and `topk_index`
  - decode is still intentionally blocked in EAGLE mode until the next
    checkpoint
- Checkpoint 5 flat-chain EAGLE decode is implemented at the worker/state level:
  - SMC EAGLE decode now samples one linear chain per particle from retained
    top-k EAGLE proposals
  - `log q` is accumulated from the renormalized retained top-k proposal
  - decode now writes a real next-cycle `SMCEagleDraftInputV2` back into
    slot-state instead of dropping EAGLE state after prefill
  - focused CPU/unit tests pass
  - end-to-end cleanup was fixed after adding orphan slot cleanup in the
    scheduler / slot-state layer
- We are **not** implementing full tree-aware SMC in v1.
- We are targeting an MVP that keeps the current SMC contract:
  - draft `gamma` proposal-scored tokens,
  - target-score those same `gamma` tokens,
  - then sample one target-side bonus token,
  - carry that bonus token into the next decode cycle as `verified_id`.

## Handoff Summary

This section is meant to let another agent pick the work up without chat
history.

### What Is Working Today

- `smc_draft_kind=lm` still runs normally.
- `smc_draft_kind=eagle` now runs end to end without crashing the scheduler.
- The worker executes a flat-chain EAGLE proposal:
  - start from `verified_id`
  - sample `gamma` draft tokens from retained EAGLE top-k proposals
  - accumulate `log q`
  - verify the drafted chain under the target
  - compute `logprob_diff = sum(log p - log q)`
  - sample a target bonus token
  - build a next-cycle `SMCEagleDraftInputV2`
- The scheduler/slot-state now persists EAGLE state across cycles rather than
  dropping it after prefill.
- The one-question GPU smoke exits cleanly and prints the evaluation summary.

### What Is Still Wrong

The current problem is no longer plumbing or cleanup. The current problem is
generation quality / behavioral correctness.

Observed runtime result on the real smoke test:

- command:
  - `PYTHONPATH=/home/yahya/smcsd:/home/yahya/smcsd/3rdparty/sglang/python CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/accuracy_test_gsm8k.py --mode smc_engine --model meta-llama/Llama-3.1-8B-Instruct --draft-model meta-llama/Llama-3.2-1B-Instruct --particles 4 --gamma 4 --temperature 0.7 --attention-backend fa3 --num-questions 1 --max-running-requests 8 --cuda-graph-max-bs 8 --smc-draft-kind eagle --smc-eagle-topk 4`
- result:
  - generation completed and the process exited cleanly
  - output was only one token: `To`
  - accuracy was `0/1`
  - output marked invalid

So the project has crossed the "make it run" boundary, but it has not crossed
the "make it generate sensible continuations" boundary.

### Most Likely Remaining Problem Area

The likely remaining bug is in proposal-state semantics rather than raw
scheduler plumbing.

The most suspicious areas are:

- what exactly the carried `hidden_states` represent from one cycle to the next
- whether the next-cycle EAGLE state should be bootstrapped from target hidden
  states, draft hidden states, or a draft-extend pass over the accepted path
- whether `verified_id` and the carried hidden state are aligned to the same
  history
- whether the decode-time draft step is writing / reading KV locations with the
  correct timing relative to the sampled chain
- whether the retained `topk_p` / `topk_index` correspond to the same token
  history as the carried hidden state

This means the next work should be correctness debugging, not more scheduler
infrastructure.

## Files Changed So Far

### Top-level repo

- `progress.md`
- `scripts/accuracy_test_gsm8k.py`
- `smcsd/engine.py`
- `smcsd/v2/info.py`
- `smcsd/v2/req_state.py`
- `smcsd/v2/scheduler.py`
- `smcsd/v2/worker.py`
- `tests/test_smc_v2_scheduler.py`

### Vendored submodule

- `3rdparty/sglang/python/sglang/srt/server_args.py`

## Detailed Checkpoint Notes

### Checkpoint 2: Config Plumbing

Implemented:

- `smc_draft_kind = {lm, eagle}`
- `smc_eagle_topk`
- validation that `smc_eagle_topk > 1` for EAGLE mode
- engine/script wiring so the new args reach `ServerArgs`

Important detail:

- the original hard failure for all `smc_draft_kind=eagle` runs was later
  removed once Checkpoint 4/5 were implemented, so EAGLE mode now passes
  initialization.

### Checkpoint 3: Draft Carrier

Implemented:

- `SMCEagleDraftInputV2` in `smcsd/v2/info.py`
- `ScheduleBatchSMC.prepare_for_decode()` can now return either:
  - `SMCDraftInputV2`
  - `SMCEagleDraftInputV2`
- worker and tests accept both carrier shapes

Purpose:

- create a stable SMC-side state object before adding real EAGLE logic

### Checkpoint 4: Prefill-Only EAGLE Initialization

Implemented:

- target prefill captures hidden states
- EAGLE draft prefill runs once to initialize:
  - `verified_id`
  - `hidden_states`
  - `topk_p`
  - `topk_index`
- EAGLE decode was intentionally blocked at this stage

Important detail:

- this checkpoint was used to validate that EAGLE state could be created
  correctly at the prefill/decode boundary without touching the SMC resampler

### Checkpoint 5: Flat-Chain EAGLE Decode

Implemented:

- EAGLE decode path in `smcsd/v2/worker.py`
- per-step sampling from renormalized retained top-k probabilities
- `log q` accumulation
- target verify reuse from the existing SMC path
- next-cycle `SMCEagleDraftInputV2` construction after decode

Also implemented to support this:

- per-slot EAGLE state persistence in `smcsd/v2/req_state.py`
  - hidden states
  - retained top-k probabilities
  - retained top-k indices
  - validity flags
- prefill seeding of slot-state from `result.next_draft_input`
- decode write-back of next-cycle EAGLE state into slot-state

### Cleanup Work Done After Checkpoint 5

The first runtime version of Checkpoint 5 still crashed after generation due to
idle-time memory checks.

Observed failures that were fixed:

1. First decode cycle had no persisted EAGLE state in slot-state.
   Fix:
   - seed slot-state with prefill-produced `SMCEagleDraftInputV2`
   - persist next-cycle EAGLE state after each decode result

2. Scheduler could reach idle with orphaned slot-held tokens/reqs.
   Fix:
   - add idle-time orphan cleanup hook in `SMCSchedulerV2`
   - add lower-level `force_free_all_slots()` support in `ScheduleBatchSMC`

3. Forced orphan cleanup hit partially-freed req objects.
   Fix:
   - make slot freeing tolerate reqs whose `req_pool_idx` is already `None`

Result:

- the one-question EAGLE smoke now exits cleanly

## Current Verification State

### Unit / CPU checks

Passing:

- `python3 -m py_compile` on the edited Python files
- focused test suite:
  - `tests/test_smc_v2_scheduler.py`
  - current result: `10 tests, OK`

These tests currently cover:

- decode-carrier selection
- EAGLE prefill carrier initialization
- mocked flat-chain EAGLE decode flow
- scheduler admission
- finalize-group behavior
- slow-path resampling behavior

### Runtime checks

Confirmed working:

- LM SMC path still runs
- EAGLE SMC path now runs end to end and exits cleanly

Confirmed still bad:

- EAGLE generation quality is currently poor / degenerate on the smoke test

## Recommended Next Steps

If another agent picks this up, the best next move is to debug correctness of
the EAGLE state transition, not to add more features.

Recommended order:

1. Add temporary instrumentation around one decode cycle in `smcsd/v2/worker.py`
   to log:
   - incoming `verified_id`
   - whether `topk_p/topk_index` are present
   - sampled draft tokens
   - per-step `log q`
   - verified tokens under the target
   - outgoing `verified_id`
2. Check that the carried `hidden_states` and `verified_id` correspond to the
   same history.
3. Check whether the next-cycle state should come from:
   - the current draft-extend path over `next_token_ids`
   - or a different bootstrap source
4. Compare one-cycle behavior against upstream EAGLE worker logic for the same
   accepted path.
5. Only after generations look sane should we run larger GSM8K tests or touch
   performance.

## Commands Used For Verification

Focused unit suite:

```bash
.venv/bin/python -c "import sys, unittest; sys.path[:0]=['/home/yahya/smcsd','/home/yahya/smcsd/3rdparty/sglang/python']; suite=unittest.defaultTestLoader.discover('/home/yahya/smcsd/tests', pattern='test_smc_v2_scheduler.py'); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(0 if result.wasSuccessful() else 1)"
```

Main EAGLE runtime smoke:

```bash
PYTHONPATH=/home/yahya/smcsd:/home/yahya/smcsd/3rdparty/sglang/python \
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --particles 4 --gamma 4 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --num-questions 1 \
  --max-running-requests 8 \
  --cuda-graph-max-bs 8 \
  --smc-draft-kind eagle \
  --smc-eagle-topk 4
```

## Checkpoint 1

### Decision

The first implementation will use **stochastic EAGLE as a flat-chain proposal**
for SMC.

This means:

- We will reuse EAGLE draft machinery to get draft hidden states and candidate
  token distributions.
- We will **not** reuse EAGLE's current branch-selection / accept-reject logic
  as the SMC proposal.
- Each particle will sample exactly one next token per draft step, producing a
  single linear chain per particle.
- SMC resampling and target verification stay linear and unchanged.

### Why This Is The Right First Step

- It preserves the current SMC worker/scheduler contract.
- It gives us an explicit proposal density `q`, which SMC needs.
- It keeps debugging local to the worker instead of mixing in tree-aware
  scheduler changes.
- It avoids a harder and more error-prone "SMC over EAGLE trees" design for v1.

## Proposal Specification

### Existing SMC Contract We Must Preserve

The current decode contract in `smcsd/v2/worker.py` is:

1. Start the decode cycle from a per-particle `verified_id`.
2. Draft `gamma` proposal-scored tokens.
3. Target-score those `gamma` tokens and compute:

   `logprob_diff = sum_t [log p_t(x_t) - log q_t(x_t)]`

4. Sample one target-side bonus token.
5. Emit:
   - `next_token_ids = [x_1, ..., x_gamma, bonus]`
   - `accept_lens = gamma + 1`
   - `next_verified_id = bonus`

Important: the importance weight only uses the first `gamma` drafted tokens.
The final bonus token is sampled from the target and is **not** part of the
proposal correction term.

### New EAGLE-SMC Proposal

For each particle and each SMC decode cycle:

- Let `v0` be the carried-in `verified_id`.
- Let `s1` be the EAGLE draft state after conditioning on `v0`.
- For each draft step `t in {1, ..., gamma}`:
  - EAGLE produces a candidate set `K_t = {k_1, ..., k_m}` where `m = topk`.
  - EAGLE also produces unnormalized or partially normalized candidate masses
    associated with that candidate set.
  - We define the actual proposal distribution as a **renormalized truncated
    categorical over the retained EAGLE candidates**:

    `q_t(x | s_t) = mass_t(x) / sum_{y in K_t} mass_t(y)` for `x in K_t`

    and `q_t(x | s_t) = 0` otherwise.
  - Sample one token:

    `x_t ~ q_t(. | s_t)`

  - Accumulate:

    `log q_t(x_t | s_t)`

  - Advance the EAGLE draft state along the sampled token `x_t` to obtain
    `s_{t+1}`.

After `gamma` draft steps:

- We have a linear drafted chain `(x_1, ..., x_gamma)`.
- We score that chain under the target exactly as current SMC does.
- We compute:

  `logprob_diff = sum_{t=1..gamma} [log p_t(x_t) - log q_t(x_t)]`

- We then sample one target bonus token exactly as current SMC does.

### What "SMC Over The Full EAGLE Tree" Would Mean

This phrase means a different design from the MVP above.

In full-tree EAGLE SMC:

- A particle would not be a single linear token chain during drafting.
- Instead, a particle's proposal state would include an entire EAGLE draft
  tree, or at least a structured subset of that tree.
- Weighting would have to reason about probability mass over branches in that
  tree, not just over one sampled token per step.
- Verification would also become tree-aware: we would need to compare the
  target model against branch proposals and decide how target probability mass
  maps onto the drafted tree.
- Resampling would likely need branch-aware state movement, because what gets
  copied between particles is no longer just a flat sequence plus KV cache; it
  is a structured draft state with parent/child relationships.

Another way to say it:

- Current SMC assumes each particle proposes a flat sequence
  `(x_1, ..., x_gamma)`.
- Full-tree EAGLE SMC would treat a particle as proposing a **tree of
  possible continuations**, with the target interacting with that tree more
  directly.

That is a much bigger algorithmic change, because then we must answer all of
these questions explicitly:

- What exactly is the particle state: one chosen branch, or the whole tree?
- What is the proposal density `q`: branch probability, path probability, or
  total retained tree mass?
- How is `log p - log q` defined when multiple future continuations are kept at
  once?
- How do we resample and copy structured branch state safely?
- How do we debug correctness when errors can come from tree construction,
  branch scoring, verification, or resampling?

### How That Differs From What We Will Do Now

The MVP in this document is **not** full-tree EAGLE SMC.

Instead, we are using EAGLE only as a way to construct a stochastic local
proposal at each draft step.

Concretely:

- EAGLE gives us a retained candidate set and candidate masses.
- We turn that into a simple truncated categorical distribution.
- We sample exactly one token from that distribution for each particle and each
  step.
- After sampling, we immediately collapse back to a single linear chain.
- All later SMC logic stays sequence-based rather than tree-based.

So the difference is:

- **Full-tree EAGLE SMC:** the particle remains tree-valued during the draft /
  verify / possibly resample phases.
- **Our MVP:** the particle is always sequence-valued from SMC's point of view;
  EAGLE only helps define the next-step proposal distribution.

This is the main simplification that keeps the first implementation small:

- no tree-valued particle state in the scheduler,
- no tree-aware resampling,
- no need to redefine the current SMC decode contract,
- no need to invent a new structured importance-weight formula in v1.

## Checkpoint 4

### Scope

Checkpoint 4 makes EAGLE mode real only in the prefill path.

That means:

- target prefill now runs with hidden-state capture enabled
- EAGLE mode uses those target hidden states plus the target-sampled
  `verified_id` to run one draft-model prefill step
- the SMC-side `SMCEagleDraftInputV2` returned from prefill is now populated
  with:
  - `verified_id`
  - `hidden_states`
  - `topk_p`
  - `topk_index`
- the decode path still raises an explicit "not implemented yet" error in
  EAGLE mode so failures stay local and understandable

### What Changed Conceptually

Before Checkpoint 4, `smc_draft_kind=eagle` only changed configuration and
carrier types.

After Checkpoint 4:

1. The target prefill generates the first real token as usual.
2. That token becomes the carried `verified_id` for EAGLE state init.
3. The draft model computes the first retained EAGLE candidate distribution.
4. We store that state in `SMCEagleDraftInputV2` for the next checkpoint.

So EAGLE mode now has a real prefill boundary, but still no decode-time
sampling logic yet.

### Side-By-Side Diagram

```text
Starting point for one particle:

  verified_id = v0


Full-tree EAGLE SMC idea
------------------------

  particle state
      |
      v
      v0
      |
      +-- a1 -- a2
      |
      +-- b1 -- b2
      |
      +-- c1 -- c2

  Interpretation:
  - the particle carries a structured proposal object
  - multiple branches remain alive inside the particle
  - weighting / verification must reason over the tree
  - resampling may need to copy structured branch state


Our MVP: flat-chain EAGLE proposal
----------------------------------

  particle state
      |
      v
      v0
      |
      +-- candidate set K1 = {a1, b1, c1}
              |
              +-- sample one token, say b1
                      |
                      +-- candidate set K2 conditioned on b1 = {d2, e2, f2}
                              |
                              +-- sample one token, say e2
                                      |
                                      +-- continue until gamma drafted tokens

  Resulting particle path:

      v0 -> b1 -> e2 -> ...

  Interpretation:
  - EAGLE provides local candidate distributions
  - we immediately sample one token and collapse to one path
  - the particle stays sequence-valued from SMC's perspective
  - weighting stays path-based: sum_t [log p_t - log q_t]
```

### Numbered Example

Assume one particle, `gamma = 2`, and `topk = 3`.

#### Full-tree EAGLE SMC

1. The particle starts from `verified_id = v0`.
2. EAGLE proposes a tree of depth 2, for example:
   - step 1 children: `{a1, b1, c1}`
   - step 2 descendants under those children:
     - under `a1`: `{a2, a2'}`
     - under `b1`: `{b2, b2'}`
     - under `c1`: `{c2, c2'}`
3. The particle may keep that whole tree, not just one branch.
4. Target verification would have to decide how target probability mass
   interacts with the retained branches.
5. The importance weight would need a tree-aware definition:
   - path probability,
   - branch probability,
   - or some retained-tree mass quantity.
6. Resampling would potentially copy or discard structured branch state, not
   just a flat sequence.

#### Our MVP

1. The particle starts from `verified_id = v0`.
2. EAGLE proposes step-1 candidates:
   - `K1 = {a1, b1, c1}`
   - with retained masses, for example `{0.5, 0.3, 0.2}`
3. We renormalize over `K1` and sample one token:
   - suppose we draw `b1`
   - we add `log q_1(b1 | v0)` to the proposal logprob
4. We advance the EAGLE draft state conditioned on `b1`.
5. EAGLE proposes step-2 candidates conditioned on that sampled history:
   - `K2 = {d2, e2, f2}`
   - with retained masses, for example `{0.6, 0.25, 0.15}`
6. We sample one token again:
   - suppose we draw `e2`
   - we add `log q_2(e2 | v0, b1)`
7. The particle's drafted path is now the flat sequence `(b1, e2)`.
8. We score that flat sequence under the target:
   - compute `log p_1(b1 | v0)`
   - compute `log p_2(e2 | v0, b1)`
9. We form the usual SMC correction term:
   - `logprob_diff = [log p_1 - log q_1] + [log p_2 - log q_2]`
10. Then we sample the usual target-side bonus token and carry that as the
    next `verified_id`.

The key difference is that in the MVP, the particle never remains a tree after
each proposal step. We sample immediately and continue with one path.

### Temperature

For the MVP, EAGLE draft mode will reuse `smc_draft_temperature`.

The intended semantics are:

- Apply the draft temperature before candidate selection / sampling.
- Build the EAGLE candidate proposal from that temperature-adjusted draft
  distribution.
- Renormalize the retained top-k candidate masses to form a valid categorical.

This keeps the exploration control aligned with the existing SMC LM-draft path.

### Stochasticity Requirement

To make EAGLE useful as an SMC proposal, the proposal must allow particle
diversity.

Therefore:

- The normal EAGLE-SMC mode should require `topk > 1`.
- `topk = 1` is effectively deterministic and should be treated as a debugging
  mode at most, not the default intended path.

For the first implementation, the expected default is:

- `smc_draft_kind = "eagle"`
- `smc_eagle_topk` in the range `4` to `8`

## Explicit Non-Goals For V1

We are **not** doing these in the first implementation:

- SMC over the full EAGLE tree
- reuse of EAGLE accept/reject verification as the SMC weighting rule
- branch-level particle state in the scheduler
- tree-aware resampling
- changing SMC's current `gamma` plus bonus-token decode contract

## Design Consequences

### What We Will Reuse

- EAGLE draft-model loading and hidden-state machinery
- EAGLE draft-state structures, or a reduced version of them
- EAGLE draft-step kernels / helpers where they help advance the proposal state

### What We Will Not Reuse Directly

- EAGLE's current speculative verification path as the source of SMC weights
- EAGLE branch-selection logic that keeps top-scoring branches
- accept/reject semantics from EAGLE speculative decoding

Reason:

- Those paths are built for EAGLE verification, not for exposing a clean SMC
  proposal density `q`.

## Acceptance Criteria For Checkpoint 1

Checkpoint 1 is complete if the following statements are true:

- We can write the proposal distribution `q` precisely.
- We know exactly which tokens contribute to `logprob_diff`.
- We know exactly which token becomes `next_verified_id`.
- We have decided that the first implementation is a sampled linear chain, not
  a tree-valued particle.
- We have decided that EAGLE-SMC needs stochastic top-k sampling, not greedy
  `topk = 1` by default.

## Open Questions

These are still open and should be resolved during implementation, not in
Checkpoint 1:

- Whether to store the new EAGLE draft state inside a dedicated
  `SMCEagleDraftInputV2` type or extend `SMCDraftInputV2`.
- How much of the existing EAGLE draft helpers can be reused before the code
  becomes more confusing than a small custom EAGLE-for-SMC helper.
- Whether `smc_draft_temperature` should be applied before top-k selection,
  after top-k selection, or both. Current design intent is "before selection,
  then renormalize retained masses."
- Whether we want an explicit debug override for deterministic `topk = 1`.

## Next Checkpoints

### Checkpoint 2

Add config plumbing only:

- `smc_draft_kind = {lm, eagle}`
- `smc_eagle_topk`
- validation for stochastic mode

Status:

- implemented
- `lm` remains the default runnable path
- `eagle` currently fails fast with a clear not-implemented error until the
  later checkpoints add the actual draft-state and worker behavior

### Checkpoint 3

Add an SMC-side EAGLE draft-state carrier that can survive prefill/decode
boundaries.

Status:

- implemented
- no real EAGLE drafting yet
- LM path remains unchanged
- EAGLE mode now has a concrete internal carrier type for later checkpoints

### Checkpoint 4

Implement prefill-only EAGLE initialization for SMC draft mode.

### Checkpoint 5

Implement stochastic EAGLE flat-chain drafting in isolation, including exact
`log q` accumulation.

### Checkpoint 6

Plug that proposal into the current SMC verify / weight / resample flow.

### Checkpoint 7

Run small end-to-end GSM8K smoke tests before any performance tuning.

---

# Session 2 — 2026-04-18 — End-to-end debugging & quality push

This session picked up from the state above (Checkpoint 5 "looks implemented"
but produces only 1 token `To` on the smoke) and drove it to a pipeline that
actually generates coherent text end to end. Along the way we discovered that
the previous "Checkpoint 5 is implemented" claim was misleading — at runtime
the group was being aborted silently before decode ever ran. Below is the full
diagnostic chain and what each fix actually did.

## Starting picture

- Smoke run: 1 token output (`To`), marked invalid, accuracy 0/1.
- Earlier hypothesis in this file ("carried hidden_states semantics / KV
  timing") was only partly right — the real chain of bugs was more plumbing
  than semantics.

## Step 1 — Instrument one decode cycle

Added env-gated debug prints (`SMC_EAGLE_DEBUG=1`, with optional
`SMC_EAGLE_DEBUG_MAX_CYCLES`, default 2) to
[smcsd/v2/worker.py](smcsd/v2/worker.py) inside `_forward_decode_eagle` and
covering the dispatch (`forward_batch_generation`), extend entry/exit, and
decode entry. Also added scheduler-side prints in
[smcsd/v2/scheduler.py](smcsd/v2/scheduler.py) (`_process_prefill_result`,
`_prepare_decode_batch`) behind the same env var.

These prints were the cheapest way to find out that `_forward_decode_eagle`
was **never actually being called** — the group was being aborted at
`_materialize_group` with a tensor-size error the earlier `try/except`
swallowed as a soft abort.

## Bug cascade (in the order we uncovered it)

For each, the symptom, cause, and fix.

### 1. Slot-buffer hidden_size used target's, not draft's

- Symptom: `SMC slot allocation failed: The expanded size of the tensor
  (4096) must match the existing size (2048) at non-singleton dimension 0`.
- Cause: [smcsd/v2/req_state.py](smcsd/v2/req_state.py)'s `eagle_hidden_size`
  was read from the (target) `model_config.hidden_size`. The draft in the
  smoke command was vanilla Llama-3.2-1B (hidden=2048), so writing
  `hidden_states[0]` (2048) into the slot buffer (4096) raised and the group
  was aborted. Nothing downstream ever ran.
- Fix path taken: sidestep by using a **real EAGLE head** whose `hidden_size`
  matches the target (4096). We did not add a draft-hidden_size plumbing
  path, because later analysis showed a vanilla 1B was the wrong thing to
  plug in to EAGLE semantics anyway.

### 2. EAGLE head `max_position_embeddings=2048` < target 131072

- Cause: EAGLE head configs ship short RoPE ranges; the draft worker's
  `ModelConfig._derive_context_length` refuses to stretch the draft past its
  config without an override.
- Fix: set `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.

### 3. EAGLE head `torch_dtype=float16` ≠ target `bfloat16`

- Symptom: `AssertionError: Buffer input_embeds has different dtype than
  before.` during CUDA-graph input-buffer sharing.
- Cause: target loaded as bf16, EAGLE head config declares fp16; the shared
  CUDA-graph input buffers detect the dtype clash on capture.
- Fix: added `--dtype` flag to
  [scripts/accuracy_test_gsm8k.py](scripts/accuracy_test_gsm8k.py) that
  threads through `SMCEngine(**kwargs)` to `ServerArgs.dtype`. Use
  `--dtype bfloat16` to force both models into the same dtype.

### 4. Draft CUDA-graph capture with no `spec_info`

- Symptom: `AttributeError: 'NoneType' object has no attribute
  'hidden_states'` inside `LlamaForCausalLMEagle.forward` at
  `forward_batch.spec_info.hidden_states`.
- Cause: `TpModelWorker`'s CUDA-graph capture runs a dummy forward without
  setting spec_info. Standard `EAGLEWorker` injects a dummy
  `EagleDraftInput` at capture time. `SMCWorkerV2` does not.
- Fix: force-disable draft CUDA-graph capture in EAGLE mode (set
  `backup_disable_cuda_graph = True` before the conditional capture call in
  [smcsd/v2/worker.py](smcsd/v2/worker.py)). Acceptable — target graphs are
  also disabled in EAGLE mode (see bug 6).

### Bug Y — draft FA backend pushed into tree branch (topk > 1)

- Symptom: `RuntimeError: shape '[-1, 0]' is invalid for input of size 4` at
  [flashattention_backend.py:477](3rdparty/sglang/python/sglang/srt/layers/attention/flashattention_backend.py#L477),
  in `cache_loc.view(-1, self.speculative_num_steps)`.
- Cause: SMC sets `server_args.speculative_eagle_topk = smc_eagle_topk = 4`.
  The FA backend reads `self.topk = server_args.speculative_eagle_topk` and
  enters the EAGLE-tree decode branch, which divides `out_cache_loc` by
  `speculative_num_steps` — left at its default 0 on the draft runner's
  standalone FA backend (it's only set on the multi-step draft backend
  `SMCWorkerV2` constructs separately and never actually uses in the
  per-step decode).
- Realization: SMC's "top-k" (number of retained candidates for the
  renormalized proposal) is a different concept from the attention
  backend's "topk" (tree width). SMC draws **one** token per step — the
  correct attention path is `topk<=1`.
- Fix: override `self.draft_runner.attn_backend.topk = 1` after draft init
  for EAGLE mode (in [smcsd/v2/worker.py](smcsd/v2/worker.py)). This routes
  per-step decode into the clean single-chain branch.

### Bug Z — target verify expected a tree `custom_mask`

- Symptom: `TypeError: 'NoneType' object is not subscriptable` at
  [flashattention_backend.py:2161](3rdparty/sglang/python/sglang/srt/layers/attention/flashattention_backend.py#L2161),
  `spec_info.custom_mask[mask_extraction_indices]`, during CUDA-graph replay
  for target verify.
- Cause: same `speculative_eagle_topk > 1` pushes target into tree-verify
  mode expecting a `custom_mask` that SMC's flat-chain verify never
  constructs.
- Fix (two parts):
  1. Set `self.score_runner.attn_backend.topk = 1` so the target-verify
     eager path takes the single-chain branch that needs no custom_mask.
  2. Force `server_args.disable_cuda_graph = True` in
     [3rdparty/sglang/python/sglang/srt/server_args.py](3rdparty/sglang/python/sglang/srt/server_args.py)
     when `smc_draft_kind == "eagle"`. The target's CUDA graphs were
     captured during `super().__init__()` (before `SMCWorkerV2` can touch
     topk), so the graph-replay path is baked for tree mode; easier to skip
     capture entirely for EAGLE mode. Eager path + topk=1 works.

### Missing hidden states from target verify

- Symptom: `RuntimeError: SMC EAGLE decode requires target hidden states to
  initialize the next draft state.` inside
  `_build_eagle_next_draft_input_from_decode`.
- Cause: `SMCDecodeContext.prepare_for_verify` hardcoded
  `capture_hidden_mode=CaptureHiddenMode.NULL`. EAGLE's cycle transition
  needs the target's hidden state at the last verify position.
- Fix: added a `capture_hidden_mode` kwarg to
  [smcsd/v2/info.py](smcsd/v2/info.py) `prepare_for_verify` (default NULL
  for LM), and the EAGLE caller in worker.py passes
  `CaptureHiddenMode.FULL`.

### `prepare_for_extend_to_fill_draft_kvcache` crash

- Symptom: `TypeError: Mismatched type on argument #4 when calling
  store_cache ... Expected DLTensor* but got None` during the
  end-of-cycle draft-extend forward.
- Cause: the extend's `out_cache_loc` was inherited from the decode batch
  and didn't match the KV-pool layout the draft-extend path expects; the
  machinery assumes a standard EAGLE worker's KV-pool plumbing that
  `SMCWorkerV2` doesn't fully provide.
- Fix (pragmatic): replaced the draft-extend with the simplest thing that
  works — carry target's last-verify hidden directly, set
  `topk_p=topk_index=None`, let the next cycle's bootstrap step regenerate
  top-k from `_run_eagle_decode_step` on the bonus token. The bootstrap
  case (kv_step_offset=1) fills all gamma+1 slots, so there is **no KV gap
  across cycles** — the perceived "bonus position missing draft KV" worry
  in the original progress notes is moot under this simplification.

### Fix X — shared embed_tokens / lm_head for the draft

This was the quality-breaking bug.

- Observation from Step-1 instrumentation once the pipeline started running:
  top-k was **exactly** `[0.25, 0.25, 0.25, 0.25]` with indices `[1, 0, 2,
  3]` at every step across cycles — classic all-zero-logits signature.
- Inspected the EAGLE-v1 checkpoint (`lmsys/sglang-EAGLE-LLaMA3-Instruct-8B`)
  and found it ships 10 tensors: `embed_tokens`, `fc`, 1 transformer layer's
  weights, `post_attention_layernorm`. No `lm_head.weight`. Config says
  `tie_word_embeddings: false`. The checkpoint expects the **target's
  lm_head and embed to be wired in at runtime** — which sglang's real
  `EAGLEWorker.init_lm_head()` does via `set_embed_and_head(embed, head)`.
  `SMCWorkerV2` used plain `TpModelWorker`, so the draft's `lm_head` stayed
  at init (all-zero rows) → uniform-over-vocab logits → top-k returned the
  lowest 4 ids.
- Fix: after creating the draft runner in
  [smcsd/v2/worker.py](smcsd/v2/worker.py), for `smc_draft_kind == "eagle"`
  call `target_model.get_embed_and_head()` and `draft_model.set_embed_and_head(embed, head)`.

Post-fix, top-k is real, hidden states evolve meaningfully across steps, and
a 1-question smoke emits:

```
ToStep 1: Calculate the number of eggs Janet is the total number of each day
minus daily consumption ...
```

## Where we stood after the plumbing fixes

**Gold signal: pipeline runs end to end, producing real English.**

Single-question smoke at `gamma=4, N=4, topk=4, temperature=0.7`:
| | before Step 1 | after X fix |
|---|---|---|
| Output | `"To"` (1 token) | 128 real tokens |
| TPS | 5 | 111 |
| GSM8K @ 1 question | 0/1 (format invalid) | 0/1 (format valid, wrong answer) |

50-question GSM8K eval against the LM-draft baseline (Llama-3.2-1B):

| | **LM SMC** | **EAGLE v1** |
|---|---|---|
| Accuracy | **32/50 (64%)** | 0/50 (0%) |
| Invalid | 1/50 (2%) | 39/50 (78%) |
| Throughput | 251 tps | 185 tps |
| Wall time | 37s | 125s |

Output text was **partially coherent but garbled** — recurring repetitions
("Total daily Total daily ..."), disjointed clauses, never hitting the
`#### <answer>` pattern. Hypothesized root cause: EAGLE-v1 head was trained
on Llama-3, not Llama-3.1; combined with SMC's accept-all-drafts rule (4 of
every 5 output tokens come from the draft), small draft biases compound
into incoherent output.

## B1 — Fix the cycle-transition draft-extend properly

Attempted to replace the simplified "carry target-last hidden with
`topk_p=None`" path with a one-step draft forward at the bonus position to
fill draft KV there and carry fresh top-k.

**Result: no accuracy improvement.** 0/50 stayed 0/50; invalid rate moved
78% → 84%. Output character changed (from rambling to looping:
`"ToStep 1: First, Janet calculates\nTotal daily eggs eggs\n..." → "Total
daily\nTotal daily\nTotal daily"`), but quality was not rescued. B1
confirmed my prior analysis was wrong about where the ceiling came from:
the missing piece isn't the cycle-transition math, it's draft quality +
accept-all. The B1 code is still in
`_build_eagle_next_draft_input_from_decode` — it's a correctness
improvement in its own right, just not a quality lever.

## B2 — EAGLE3 support

The cached `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B` head is actually
trained for Llama-3.1 (vs v1's Llama-3). EAGLE3 is a different architecture:

- Reduced draft vocab (`draft_vocab_size=32000`) with a `d2t` map that
  carries per-draft-id offsets to the full 128256-token target vocab.
- `fc` takes 3× target hidden (`hidden_size * 3 = 12288`) — aux hidden
  states from three target layers, not just the last.
- Ships its own `lm_head` (over 32000 draft tokens) and needs only
  `embed_tokens` shared from target.

Changes made in [smcsd/v2/worker.py](smcsd/v2/worker.py):
- Detect EAGLE3 by presence of `draft_model.hot_token_id` (populated by
  `LlamaForCausalLMEagle3.load_weights` from `d2t`).
- For EAGLE3: call `draft_model.set_embed(embed)` (embed-only) and
  `target_model.set_eagle3_layers_to_capture()` (spec_algorithm is "SMC"
  not "EAGLE3", so target's runner does not auto-enable aux capture).
- Cache `hot_token_id` on the worker and remap `topk_index` at the end of
  `_extract_eagle_state` and `_build_eagle_prefill_next_draft_input` —
  every target-vocab-facing consumer (verify, bonus, next input, carrier)
  sees target-vocab ids.

**Result (EAGLE3 @ gamma=4, N=4):**

| | LM SMC | EAGLE v1 | **EAGLE3** |
|---|---|---|---|
| Accuracy | 64% | 0% | **0%** |
| Invalid | 2% | 78% | **58%** |
| Throughput | 251 tps | 185 | 179 |

EAGLE3 **measurably improves draft quality** (invalid rate 78% → 58%,
first sentences grammatical, real prompt words appear) but still 0%
accuracy. Same failure mode as v1 at larger scale: decent start, then
degenerate word-salad / repetition collapse.

## C3 — `gamma` ablation, and the breakthrough

The accept-all-draft hypothesis predicts that **reducing gamma** (thus the
draft:target token ratio) should monotonically lift accuracy.

EAGLE3 with N=4, topk=4, temp=0.7, 50 questions:

| gamma | Accuracy | Invalid | TPS | Wall |
|---|---|---|---|---|
| 4 (prev default) | 0/50 (0%) | 58% | 179 | 127 s |
| 2 | 0/50 (0%) | 28% | 122 | 203 s |
| **1** | **12/50 (24%)** | **0/50 (0%)** | 87 | 205 s |
| 2, N=8 | 0/50 (0%) | 38% | 118 | 197 s |

**`gamma=1` jumps from 0% → 24% accuracy.** Invalid rate drops to zero —
every output hits the GSM8K format. At gamma=1 the chain is 1 draft + 1
bonus = 50% target tokens (vs 20% at gamma=4), and the accept-all
contamination essentially disappears.

This is a decisive empirical confirmation that the architectural ceiling
seen earlier is **gamma × draft-quality**, not plumbing.

## What worked vs. what didn't

### Worked
- Env-gated instrumentation as the first diagnostic move (cheap, decisive).
- Treating the original "KV chain" hypothesis as a theory to falsify rather
  than implement. The actual plumbing bugs had nothing to do with it.
- Using sglang's own `set_embed_and_head` / `set_embed` +
  `set_eagle3_layers_to_capture` primitives rather than re-implementing.
- Running LM SMC as a live control during diagnosis (confirmed "prefill →
  N-particle decode" was working for LM, so the aborting group was
  EAGLE-specific).
- Asking the user which draft model they intended — using a vanilla 1B as
  "EAGLE draft" was the root of cascading shape errors.
- Aggressively force-disabling target/draft CUDA graphs for EAGLE mode.
  Graph capture for target is baked in before `SMCWorkerV2` can adjust
  config, so retrofits are fragile. Eager is fine for now.
- Confirming B1 didn't help before doubling down on harder versions.

### Didn't work / misleading
- Progress.md's pre-session "most likely remaining problem area" list put
  carried-hidden-states semantics and KV-timing at the top. Real blockers
  were all plumbing: slot-buffer shape, attention backend topk branch,
  custom_mask, CUDA-graph / spec_info interaction, and an uninitialized
  lm_head. Takeaway: reason from traces, not from design intuition.
- The "one-question EAGLE smoke exits cleanly" note in the original
  progress.md was **misleading**. It exited cleanly only because
  `_materialize_group` swallowed the slot-alloc error as a soft abort,
  emitting 1 token (the prefill bonus) and finalizing. No decode had ever
  run.
- Proper draft-extend at cycle transition (B1) gave zero accuracy uplift.
  At this setup the quality lever is gamma, not cycle-transition fidelity.
- Passing `--disable-cuda-graph` via the CLI did not propagate to the
  scheduler subprocess (observed `disable_cuda_graph=False` at worker
  init); forcing it inside `ServerArgs.__post_init__` for EAGLE-SMC was
  the reliable path.

## Current knob defaults recommendation

For EAGLE-SMC against Llama-3.1-8B:
- `--draft-model lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`
- `--dtype bfloat16`
- `--gamma 1 --smc-eagle-topk 4 --particles 4 --temperature 0.7`
- `--attention-backend fa3` (LM path; EAGLE force-disables CUDA graphs)
- env: `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`

Larger gamma is currently unsafe at any quality bar. `--max-running-requests
16` is a comfortable setting for the 50-question eval.

## Remaining debug instrumentation

Gated behind `SMC_EAGLE_DEBUG=1` + `SMC_EAGLE_DEBUG_MAX_CYCLES` (default 2).
Still present in [smcsd/v2/worker.py](smcsd/v2/worker.py) and
[smcsd/v2/scheduler.py](smcsd/v2/scheduler.py). Zero cost when env var is
unset. Worth leaving until the next wave of correctness work settles.

## Files touched this session

- [smcsd/v2/worker.py](smcsd/v2/worker.py) — instrumentation; force-disable
  draft CUDA graphs for EAGLE; override draft+target `attn_backend.topk=1`
  for EAGLE; share `embed_tokens` + (v1) `lm_head` or (v3) `embed` only; cache
  `hot_token_id` and remap `topk_index` through it for EAGLE3; cycle
  transition rewritten to skip the failing draft-extend and carry
  target-last hidden (then later upgraded in B1 to a one-step bonus draft
  forward that fills the bonus-position KV and carries fresh top-k).
- [smcsd/v2/info.py](smcsd/v2/info.py) — `prepare_for_verify` accepts a
  `capture_hidden_mode` kwarg (default NULL), EAGLE caller passes
  `CaptureHiddenMode.FULL`.
- [smcsd/v2/scheduler.py](smcsd/v2/scheduler.py) — scheduler-side debug
  prints around `_process_prefill_result` and `_prepare_decode_batch`.
- [3rdparty/sglang/python/sglang/srt/server_args.py](3rdparty/sglang/python/sglang/srt/server_args.py)
  — for `smc_draft_kind == "eagle"`, force `disable_cuda_graph = True` with
  an explanatory warning.
- [scripts/accuracy_test_gsm8k.py](scripts/accuracy_test_gsm8k.py) —
  `--dtype` flag threaded through to `SMCEngine` kwargs.

## Open questions / next work

- **D1**: does `gamma=1, N=8` or `N=16` push past 24%? Untested.
- **D2**: does `gamma=1` with wider `--smc-eagle-topk` (8, 16) change
  resampling behavior? Untested.
- **D3**: does `--smc-resample-threshold` below default 0.5 (more
  aggressive resampling) help at `gamma=1`? Untested.
- **D5**: apples-to-apples LM baseline at `gamma=1` — unknown whether the
  EAGLE3-vs-LM gap persists at the same cycle shape. Untested.
- Alternative: introduce a per-token rejection at the worker level (breaks
  the SMC "accept all drafts" spec but may be necessary to make EAGLE-class
  drafts usable at higher gamma).
- The B1 bonus draft-step writes into the same KV slot target used at the
  bonus position. Different KV-pool layers → no collision, but this relies
  on sglang's shared-pool conventions; worth an explicit assertion if the
  memory-pool config changes.
- The debug prints should eventually be removed or moved behind a proper
  logging level once the shape of "correct" is settled.
