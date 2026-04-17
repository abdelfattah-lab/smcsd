# SMCSD EAGLE Draft Integration Progress

Date: 2026-04-17
Branch: `new_release`

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
- We are **not** implementing full tree-aware SMC in v1.
- We are targeting an MVP that keeps the current SMC contract:
  - draft `gamma` proposal-scored tokens,
  - target-score those same `gamma` tokens,
  - then sample one target-side bonus token,
  - carry that bonus token into the next decode cycle as `verified_id`.

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
