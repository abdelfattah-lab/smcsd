"""Wire protocol for decoupled SMC speculative decoding.

Verifier (target engine) <-> drafter (draft engine) messages, exchanged over a
ZMQ PUSH/PULL pair (requests down, replies up) with ``send_pyobj``.  All
payloads are CPU data (lists / numpy arrays) — no torch tensors cross the wire.

Lifecycle per group (FIFO order on one channel = total order both sides see):

    DraftPrefillReq ──▶ DraftPrefillResp          (blocking, prompt prefill + x0)
    DraftMaterializeGroup                          (parent → N particle fan-out)
    repeat per round:
        DraftStepReq ──▶ DraftStepResp             (blocking, AR tokens+logprobs)
        DraftCommitResample                        (only when resampling fired)
    DraftCloseGroup                                (finalize/abort; idempotent)

Modeled on upstream SGLang decoupled-spec IO (DraftSync / VerifyCommit /
DraftClose in PR #22520), adapted for SMC: lockstep rounds, particle
materialization, resample commits, and per-position draft logprobs in the
step reply (needed for SMC importance weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class GroupSamplingParams:
    """Subset of SamplingParams the drafter needs to reproduce x0 sampling."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_new_tokens: int = 128


@dataclass
class DraftPrefillReq:
    """Prefill the prompt(s) with the draft model and sample x0 per group."""

    group_ids: List[str]
    input_ids: List[List[int]]
    sampling: List[GroupSamplingParams]


@dataclass
class DraftPrefillResp:
    group_ids: List[str]
    next_token_ids: List[int]  # x0 per group, drafter-sampled


@dataclass
class DraftMaterializeGroup:
    """Fan the parent's prompt KV out to N particle slots (refcounted share)."""

    group_id: str
    slots: List[int]  # slot ids assigned by the verifier's ScheduleBatchSMC
    shared_seq_len: int  # must equal the drafter's parent committed length


@dataclass
class DraftStepReq:
    """One draft round over a set of active slots.

    ``seq_lens`` is the committed prefix BEFORE this round's advance
    (== verifier ctx.orig_seq_lens); the drafter asserts it against its
    mirror to fail fast on divergence.  ``tag`` is echoed in the reply so a
    pipelined verifier with several rounds in flight can assert FIFO matching.
    """

    slots: List[int]
    verified_ids: List[int]  # x0 per slot (last round's anchor token)
    seq_lens: List[int]
    tag: int = 0
    # Train counter, echoed back unchanged by the drafter (a pure reactor that
    # cannot tell speculative from committed). The async scheduler uses it as a
    # fail-fast fence for the speculative-barrier-prefetch path; default 0 keeps
    # the lockstep / pipelined callers wire-compatible.
    epoch: int = 0


@dataclass
class DraftStepResp:
    # n_emit columns per slot: gamma (bonus mode, x1..x_gamma) or gamma+1
    # (no-bonus mode, x1..x_{gamma+1} — the last is the next-round anchor).
    tokens: np.ndarray  # (bs, n_emit) int64 — drafted tokens per slot
    logprobs: np.ndarray  # (bs, n_emit) float32 — draft logprob of each token
    tag: int = 0
    epoch: int = 0  # echoed from the matching DraftStepReq (see above)


@dataclass
class DraftCommitResample:
    """Apply the verifier's resample plan: dst slot state <- src slot state."""

    dst_slots: List[int]
    src_slots: List[int]


@dataclass
class DraftCloseGroup:
    """Free everything held for a group (pending parent or particle slots)."""

    group_id: str
    slots: List[int] = field(default_factory=list)


@dataclass
class DraftPing:
    pass


@dataclass
class DraftPong:
    info: Dict = field(default_factory=dict)


@dataclass
class DraftShutdown:
    pass
