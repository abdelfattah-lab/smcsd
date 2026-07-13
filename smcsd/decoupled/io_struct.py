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
DraftClose in PR #22520), adapted for SMC: serial rounds, particle
materialization, resample commits, and per-position draft logprobs in the step
reply (needed for SMC importance weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

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
    mirror to fail fast on divergence.  ``tag`` is echoed in the reply so
    async callers can assert FIFO matching.
    """

    slots: List[int]
    verified_ids: List[int]  # x0 per slot (last round's anchor token)
    seq_lens: List[int]
    tag: int = 0
    # Train counter, echoed back unchanged by the drafter. The async scheduler
    # uses it as a fail-fast fence; default 0 keeps the serial caller
    # wire-compatible.
    epoch: int = 0
    # Lag-1 may send per-slot rollback values when one mixed StepReq advances
    # ready rows and re-drafts stale rows in place. 0 = no-op.
    rollback: Union[int, List[int]] = 0
    # After applying rollback, also lower the drafter mirror's kv_allocated_lens
    # to the rolled-back seq_len before allocation. Lag-1 uses this for stale
    # suffix privatization so catch-up rows do not overwrite a shared run-ahead
    # window inherited through resampling.
    truncate_kv: Union[bool, List[bool]] = False
    # Width-2 anchor tree (SMCSD_LAG1_ANCHOR_WIDTH=2): per-slot alt run-ahead seed
    # (the c1 top-2 anchor candidate).  Aligned with ``slots``; -1 = no alt branch
    # for that slot (stale/cold rows, or a peaked anchor where c0==c1).  When any
    # entry is >= 0 the drafter clones each such slot's prefix into an internal alt
    # slot, seeds it with c1, runs the same AR, and returns the alt window so the
    # verifier can keep whichever branch the committed bonus b matches.  None =
    # width-1 (wire-unchanged for every other caller).
    bet_alt: Optional[List[int]] = None


@dataclass
class DraftStepResp:
    # n_emit columns per slot: gamma (plain serial mode, x1..x_gamma) or
    # gamma+1 (lag-1 mode, x1..x_{gamma+1} — the last is the drafter-known
    # anchor bet).
    tokens: np.ndarray  # (bs, n_emit) int64 — drafted tokens per slot
    logprobs: np.ndarray  # (bs, n_emit) float32 — draft logprob of each token
    tag: int = 0
    epoch: int = 0  # echoed from the matching DraftStepReq (see above)
    # Width-W anchor tree (SMCSD_LAG1_ANCHOR_WIDTH>=2): the top-W most-likely
    # anchor candidates at the bet position (column gamma), under the anchor-temp
    # distribution, plus their draft logprobs.  The verifier hedges the run-ahead
    # seed over these W candidates and keeps whichever the committed bonus b
    # matches (committed token is still the verified b -> accuracy-neutral).  None
    # when width=1 (wire-compatible with every other caller).
    bet_topk: Optional[np.ndarray] = None  # (bs, W) int64
    bet_topk_logprobs: Optional[np.ndarray] = None  # (bs, W) float32
    # Width-2 anchor tree: the c1-branch run-ahead window per slot (valid only
    # where the matching DraftStepReq.bet_alt[i] >= 0), plus the alt window's own
    # top-W next-anchor candidates (used when the alt branch is promoted to be the
    # slot's primary lineage).  None = width-1 / no alt requested.
    alt_tokens: Optional[np.ndarray] = None  # (bs, n_emit) int64
    alt_logprobs: Optional[np.ndarray] = None  # (bs, n_emit) float32
    alt_bet_topk: Optional[np.ndarray] = None  # (bs, W) int64


@dataclass
class DraftPromoteAlt:
    """Width-2 branch resolution (verifier -> drafter), sent after the verify
    commits b and the winning branch is known.  For each slot: promote=True swaps
    the slot's draft KV with its alt slot (the c1 branch won -> the alt run-ahead
    becomes the slot's lineage); promote=False frees the alt slot (primary won, or
    the slot went stale)."""

    slots: List[int]
    promote: List[bool]


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
