from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    ModelWorkerBatch,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    clamp_position,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info_v2 import (
    assign_draft_cache_locs_page_size_1,
    assign_extend_cache_locs_func,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.smc_draft_cuda_graph_runner import (
        SMCDraftCudaGraphRunner,
    )
SMC_MIN_TEMPERATURE = 1e-5


@dataclass
class SMCParticleState:
    target_req: Req
    draft_req: Req
    log_weight: float = 0.0

    @property
    def is_finished(self) -> bool:
        return self.target_req.finished()

    @property
    def output_ids(self) -> List[int]:
        return self.target_req.output_ids


@dataclass
class SMCRequestState:
    parent_req_id: str
    particles: List[SMCParticleState]
    n_particles: int
    gamma: int
    resample_threshold: float
    resample_method: str
    released: bool = False
    step_index: int = 0
    best_particle_idx: int = 0
    draft_prefix_materialized: bool = False

    def active_particles(self) -> List[SMCParticleState]:
        return [particle for particle in self.particles if not particle.is_finished]

    def is_terminal(self) -> bool:
        return all(particle.is_finished for particle in self.particles)

    def get_best_particle(self) -> SMCParticleState:
        self.best_particle_idx = max(
            range(len(self.particles)),
            key=lambda idx: (
                self.particles[idx].log_weight,
                len(self.particles[idx].output_ids),
            ),
        )
        return self.particles[self.best_particle_idx]


@dataclass
class SMCParentUpdate:
    done: bool
    best_particle_idx: int
    best_output_ids: Optional[List[int]] = None
    finish_reason: Optional[BaseFinishReason] = None
    finished_len: Optional[int] = None


def validate_smc_parent_req(req: Req) -> Optional[str]:
    if req.__dict__.get("multimodal_inputs") is not None:
        return "SMC speculative decoding does not yet support multimodal inputs."
    if req.__dict__.get("input_embeds") is not None:
        return "SMC speculative decoding does not yet support input_embeds."
    if req.grammar is not None:
        return "SMC speculative decoding does not yet support constrained decoding."
    if req.return_logprob:
        return "SMC speculative decoding does not yet support return_logprob."
    if req.return_hidden_states:
        return "SMC speculative decoding does not yet support return_hidden_states."
    if req.return_routed_experts:
        return "SMC speculative decoding does not yet support return_routed_experts."
    if req.sampling_params.stop_strs:
        return "SMC speculative decoding does not yet support stop strings."
    if req.sampling_params.stop_regex_strs:
        return "SMC speculative decoding does not yet support stop regex."
    return None


def clone_req_for_smc_particle(
    parent_req: Req,
    particle_idx: int,
    role: str,
    temperature: float,
    return_logprob: bool,
    output_ids: Optional[Sequence[int]] = None,
) -> Req:
    sampling_params = copy.copy(parent_req.sampling_params)
    sampling_params.temperature = max(temperature, SMC_MIN_TEMPERATURE)
    if isinstance(sampling_params.custom_params, dict):
        sampling_params.custom_params = dict(sampling_params.custom_params)

    particle_req = Req(
        rid=f"{parent_req.rid}_smc_p{particle_idx}_{role}",
        origin_input_text=parent_req.origin_input_text,
        origin_input_ids=list(parent_req.origin_input_ids),
        sampling_params=sampling_params,
        return_logprob=return_logprob,
        top_logprobs_num=0,
        dllm_config=None,
        token_ids_logprob=None,
        stream=False,
        origin_input_ids_unpadded=tuple(parent_req.origin_input_ids_unpadded),
        lora_id=parent_req.lora_id,
        input_embeds=parent_req.input_embeds,
        token_type_ids=parent_req.token_type_ids,
        session=None,
        custom_logit_processor=parent_req.custom_logit_processor,
        require_reasoning=parent_req.require_reasoning,
        return_hidden_states=False,
        return_routed_experts=False,
        eos_token_ids=parent_req.eos_token_ids,
        bootstrap_host=None,
        bootstrap_port=None,
        bootstrap_room=None,
        disagg_mode=None,
        routed_dp_rank=None,
        disagg_prefill_dp_rank=None,
        vocab_size=parent_req.vocab_size,
        priority=parent_req.priority,
        metrics_collector=None,
        extra_key=parent_req.extra_key,
        routing_key=parent_req.routing_key,
        dimensions=parent_req.dimensions,
        http_worker_ipc=None,
        time_stats=None,
    )
    particle_req.output_ids = list(
        parent_req.output_ids if output_ids is None else output_ids
    )
    particle_req.tokenizer = parent_req.tokenizer
    particle_req.decoded_text = parent_req.decoded_text
    particle_req.surr_offset = parent_req.surr_offset
    particle_req.read_offset = parent_req.read_offset
    particle_req.smc_parent = parent_req
    particle_req.smc_particle_idx = particle_idx
    return particle_req


def initialize_smc_request_state(
    req: Req,
    *,
    server_args,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    seed_output_ids: Optional[Sequence[int]] = None,
) -> Optional[str]:
    error = validate_smc_parent_req(req)
    if error is not None:
        return error

    base_output_ids = list(req.output_ids if seed_output_ids is None else seed_output_ids)

    particle_states = []
    target_reqs = []
    draft_reqs = []
    for i in range(server_args.smc_n_particles):
        target_reqs.append(
            clone_req_for_smc_particle(
                req,
                particle_idx=i,
                role="target",
                temperature=server_args.smc_target_temperature,
                return_logprob=True,
                output_ids=base_output_ids,
            )
        )
        draft_reqs.append(
            clone_req_for_smc_particle(
                req,
                particle_idx=i,
                role="draft",
                temperature=server_args.smc_draft_temperature,
                return_logprob=True,
                output_ids=base_output_ids,
            )
        )

    particle_reqs = target_reqs + draft_reqs
    if req_to_token_pool.alloc(particle_reqs) is None:
        return "SMC particle allocation failed because req_to_token_pool is full."

    if base_output_ids:
        # SMC advances through ordinary extend batches, so keep one generated token
        # outside the committed prefix when possible. This handles overlap paths where
        # the parent request may already have committed the latest output token.
        desired_seq_len = len(req.origin_input_ids) + len(base_output_ids) - 1
        shared_seq_len = min(req.kv_committed_len, desired_seq_len)
    else:
        shared_seq_len = req.kv_committed_len
    for particle_req in particle_reqs:
        req_to_token_pool.copy_block_table(
            req.req_pool_idx,
            particle_req.req_pool_idx,
            shared_seq_len,
            token_to_kv_pool_allocator,
        )
        particle_req.kv_committed_len = shared_seq_len
        particle_req.kv_allocated_len = shared_seq_len
        particle_req.prefix_indices = req_to_token_pool.req_to_token[
            particle_req.req_pool_idx, :shared_seq_len
        ].to(dtype=torch.int64, copy=True)
        particle_req.cache_protected_len = shared_seq_len

    for target_req, draft_req in zip(target_reqs, draft_reqs, strict=True):
        particle_states.append(
            SMCParticleState(target_req=target_req, draft_req=draft_req)
        )

    req.smc_state = SMCRequestState(
        parent_req_id=req.rid,
        particles=particle_states,
        n_particles=server_args.smc_n_particles,
        gamma=server_args.smc_gamma,
        resample_threshold=server_args.smc_resample_threshold,
        resample_method=server_args.smc_resample_method,
    )
    return None


def resolve_smc_seed_output_ids(
    req: Req,
    *,
    overlap_last_token_id: Optional[int],
    overlap_new_seq_len: Optional[int],
) -> List[int]:
    seed_output_ids = list(req.output_ids)
    if overlap_last_token_id is None or overlap_new_seq_len is None:
        return seed_output_ids

    desired_output_len = int(overlap_new_seq_len) - len(req.origin_input_ids)
    if desired_output_len < 0:
        raise ValueError(
            "SMC overlap lazy init received a sequence length shorter than the prompt."
        )

    current_output_len = len(seed_output_ids)
    if desired_output_len == current_output_len:
        return seed_output_ids
    if desired_output_len == current_output_len + 1:
        seed_output_ids.append(int(overlap_last_token_id))
        return seed_output_ids

    raise ValueError(
        "SMC overlap lazy init can only recover at most one missing output token "
        f"(current={current_output_len}, desired={desired_output_len})."
    )


def collect_smc_stop_token_ids(req: Req) -> List[int]:
    stop_token_ids = set(req.sampling_params.stop_token_ids or [])
    if req.eos_token_ids:
        stop_token_ids.update(req.eos_token_ids)
    if req.tokenizer is not None:
        eos_token_id = getattr(req.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_token_ids.add(eos_token_id)
        additional_stop_token_ids = getattr(
            req.tokenizer, "additional_stop_token_ids", None
        )
        if additional_stop_token_ids:
            stop_token_ids.update(additional_stop_token_ids)
    return sorted(int(token_id) for token_id in stop_token_ids)


def resolve_smc_proposal_length(
    req: Req,
    proposal_tokens: Sequence[int],
    *,
    current_output_len: Optional[int] = None,
) -> tuple[int, bool]:
    stop_token_ids = set(collect_smc_stop_token_ids(req))

    current_output_len = (
        len(req.output_ids) if current_output_len is None else current_output_len
    )
    proposal_len = 0
    for token_id in proposal_tokens:
        proposal_len += 1
        if current_output_len + proposal_len >= req.sampling_params.max_new_tokens:
            return proposal_len, True
        if req.vocab_size is not None and (token_id > req.vocab_size or token_id < 0):
            return proposal_len, True
        if not req.sampling_params.ignore_eos and token_id in stop_token_ids:
            return proposal_len, True

    return proposal_len, False


def resolve_smc_proposal_batch(
    reqs: Sequence[Req],
    proposal_tokens: torch.Tensor,
    proposal_logprobs: torch.Tensor,
    current_output_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if proposal_tokens.numel() == 0:
        empty_len = torch.empty((0,), dtype=torch.int32, device=proposal_tokens.device)
        empty_finished = torch.empty(
            (0,), dtype=torch.bool, device=proposal_tokens.device
        )
        empty_logprobs = torch.empty(
            (0,), dtype=torch.float32, device=proposal_tokens.device
        )
        return empty_len, empty_finished, empty_logprobs

    device = proposal_tokens.device
    bs, gamma = proposal_tokens.shape
    steps = torch.arange(1, gamma + 1, dtype=torch.int64, device=device).unsqueeze(0)
    current_output_lens = current_output_lens.to(dtype=torch.int64, device=device)
    max_new_tokens = torch.tensor(
        [
            int(req.sampling_params.max_new_tokens)
            if req.sampling_params.max_new_tokens is not None
            else torch.iinfo(torch.int64).max
            for req in reqs
        ],
        dtype=torch.int64,
        device=device,
    )
    vocab_sizes = torch.tensor(
        [
            int(req.vocab_size)
            if req.vocab_size is not None
            else torch.iinfo(torch.int64).max
            for req in reqs
        ],
        dtype=torch.int64,
        device=device,
    )

    max_token_mask = (
        current_output_lens.unsqueeze(1) + steps >= max_new_tokens.unsqueeze(1)
    )
    invalid_token_mask = (proposal_tokens < 0) | (
        proposal_tokens > vocab_sizes.unsqueeze(1)
    )

    stop_token_lists = [
        [] if req.sampling_params.ignore_eos else collect_smc_stop_token_ids(req)
        for req in reqs
    ]
    max_stop_tokens = max((len(stop_ids) for stop_ids in stop_token_lists), default=0)
    if max_stop_tokens > 0:
        stop_ids = torch.full(
            (bs, max_stop_tokens),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )
        stop_ids_mask = torch.zeros(
            (bs, max_stop_tokens), dtype=torch.bool, device=device
        )
        for row, stop_token_ids in enumerate(stop_token_lists):
            if not stop_token_ids:
                continue
            stop_token_tensor = torch.tensor(
                stop_token_ids, dtype=torch.int64, device=device
            )
            stop_ids[row, : stop_token_tensor.numel()] = stop_token_tensor
            stop_ids_mask[row, : stop_token_tensor.numel()] = True
        stop_token_mask = (
            proposal_tokens.unsqueeze(-1) == stop_ids.unsqueeze(1)
        ) & stop_ids_mask.unsqueeze(1)
        stop_token_mask = stop_token_mask.any(dim=-1)
    else:
        stop_token_mask = torch.zeros_like(proposal_tokens, dtype=torch.bool)

    finish_mask = invalid_token_mask | max_token_mask | stop_token_mask
    proposal_finished = finish_mask.any(dim=1)
    first_finish = torch.argmax(finish_mask.to(torch.int32), dim=1)
    draft_lengths = torch.where(
        proposal_finished,
        first_finish + 1,
        torch.full_like(first_finish, gamma),
    )
    active_mask = (
        torch.arange(gamma, dtype=torch.int64, device=device).unsqueeze(0)
        < draft_lengths.unsqueeze(1)
    )
    draft_logprobs = torch.sum(
        proposal_logprobs.to(torch.float32) * active_mask, dim=1
    )
    return (
        draft_lengths.to(torch.int32),
        proposal_finished,
        draft_logprobs.to(torch.float32),
    )


def _copy_generation_state(src_req: Req, dst_req: Req):
    dst_req.output_ids = list(src_req.output_ids)
    dst_req.finished_reason = copy.copy(src_req.finished_reason)
    dst_req.finished_len = src_req.finished_len
    dst_req.finished_output = src_req.finished_output
    dst_req.to_finish = copy.copy(src_req.to_finish)
    dst_req.kv_committed_len = src_req.kv_committed_len
    dst_req.kv_allocated_len = src_req.kv_allocated_len
    dst_req.decoded_text = src_req.decoded_text
    dst_req.surr_offset = src_req.surr_offset
    dst_req.read_offset = src_req.read_offset
    dst_req.cache_protected_len = src_req.cache_protected_len
    dst_req.logprob_start_len = src_req.logprob_start_len
    set_smc_reserved_kv_len(dst_req, get_smc_reserved_kv_len(src_req))


def _empty_prefix_indices() -> torch.Tensor:
    return torch.empty((0,), dtype=torch.int64)


def get_smc_reserved_kv_len(req: Req) -> int:
    reserved_len = getattr(req, "_smc_reserved_kv_len", None)
    if reserved_len is None:
        return int(req.kv_allocated_len)
    return max(int(reserved_len), int(req.kv_allocated_len))


def set_smc_reserved_kv_len(req: Req, reserved_len: int) -> None:
    req._smc_reserved_kv_len = max(int(reserved_len), 0)


def _release_internal_req(
    req: Req,
    req_to_token_pool,
    token_to_kv_pool_allocator,
):
    if req.req_pool_idx is None:
        return

    reserved_len = get_smc_reserved_kv_len(req)
    if reserved_len > 0:
        indices = req_to_token_pool.req_to_token[
            req.req_pool_idx, :reserved_len
        ].to(dtype=torch.int64, copy=True)
        token_to_kv_pool_allocator.dec_ref_and_free(indices)

    req_to_token_pool.free(req)
    req.prefix_indices = _empty_prefix_indices()
    req.kv_committed_len = 0
    req.kv_allocated_len = 0
    set_smc_reserved_kv_len(req, 0)


def _release_smc_parent_req(
    req: Req,
    tree_cache,
    req_to_token_pool,
    token_to_kv_pool_allocator,
):
    """Release an SMC parent req after its KV has been shared to particles.

    `copy_block_table()` increments slot refcounts for the shared parent prefix.
    The normal `release_kv_cache(..., is_insert=False)` path uses raw
    allocator `free(...)` for committed KV, which drops those shared slots to
    zero instead of removing only the parent's reference. Use `dec_ref` here so
    the particle-owned copies keep correct lifetime accounting.
    """
    if req.req_pool_idx is None:
        return

    kv_committed_len = req.pop_committed_kv_cache()
    if req.cache_protected_len < kv_committed_len:
        committed_indices = req_to_token_pool.req_to_token[
            req.req_pool_idx, req.cache_protected_len : kv_committed_len
        ].to(dtype=torch.int64, copy=True)
        token_to_kv_pool_allocator.dec_ref_and_free(committed_indices)

    start_p, end_p = req.pop_overallocated_kv_cache()
    page_size = get_global_server_args().page_size
    if page_size > 1:
        start_p = ceil_align(start_p, page_size)
    if start_p < end_p:
        overalloc_indices = req_to_token_pool.req_to_token[
            req.req_pool_idx, start_p:end_p
        ].to(dtype=torch.int64, copy=True)
        token_to_kv_pool_allocator.dec_ref_and_free(overalloc_indices)

    req_to_token_pool.free(req)
    if req.last_node is not None:
        tree_cache.dec_lock_ref(req.last_node)


def _alias_req_state(
    src_req: Req,
    dst_req: Req,
    req_to_token_pool,
    token_to_kv_pool_allocator,
):
    if src_req.req_pool_idx == dst_req.req_pool_idx:
        _copy_generation_state(src_req, dst_req)
        if dst_req.kv_committed_len > 0:
            dst_req.prefix_indices = req_to_token_pool.req_to_token[
                dst_req.req_pool_idx, : dst_req.kv_committed_len
            ].to(dtype=torch.int64, copy=True)
        else:
            dst_req.prefix_indices = _empty_prefix_indices()
        return

    dst_reserved_len = get_smc_reserved_kv_len(dst_req)
    if dst_req.req_pool_idx is not None and dst_reserved_len > 0:
        old_indices = req_to_token_pool.req_to_token[
            dst_req.req_pool_idx, :dst_reserved_len
        ].to(dtype=torch.int64, copy=True)
        token_to_kv_pool_allocator.dec_ref_and_free(old_indices)

    seq_len = src_req.kv_committed_len
    if seq_len > 0:
        copied = req_to_token_pool.req_to_token[
            src_req.req_pool_idx, :seq_len
        ].clone()
        token_to_kv_pool_allocator.inc_ref(copied.to(torch.int64))
        req_to_token_pool.write((dst_req.req_pool_idx, slice(0, seq_len)), copied)
        dst_req.prefix_indices = copied.to(dtype=torch.int64, copy=True)
    else:
        dst_req.prefix_indices = _empty_prefix_indices()

    _copy_generation_state(src_req, dst_req)
    dst_req.kv_allocated_len = seq_len
    set_smc_reserved_kv_len(dst_req, seq_len)


def alias_smc_req_state(
    src_req: Req,
    dst_req: Req,
    req_to_token_pool,
    token_to_kv_pool_allocator,
):
    _alias_req_state(
        src_req,
        dst_req,
        req_to_token_pool,
        token_to_kv_pool_allocator,
    )
    dst_req.draft_prefix_materialized = src_req.draft_prefix_materialized


def alias_particle_state(
    src_particle: SMCParticleState,
    dst_particle: SMCParticleState,
    req_to_token_pool,
    token_to_kv_pool_allocator,
):
    _alias_req_state(
        src_particle.target_req,
        dst_particle.target_req,
        req_to_token_pool,
        token_to_kv_pool_allocator,
    )
    _alias_req_state(
        src_particle.draft_req,
        dst_particle.draft_req,
        req_to_token_pool,
        token_to_kv_pool_allocator,
    )
    dst_particle.log_weight = src_particle.log_weight


def cleanup_smc_request_state(
    state: Optional[SMCRequestState],
    req_to_token_pool,
    token_to_kv_pool_allocator,
):
    if state is None or state.released:
        return

    for particle in state.particles:
        _release_internal_req(
            particle.target_req,
            req_to_token_pool,
            token_to_kv_pool_allocator,
        )
        _release_internal_req(
            particle.draft_req,
            req_to_token_pool,
            token_to_kv_pool_allocator,
        )

    state.released = True


def normalize_log_weights(
    log_weights: Sequence[float] | torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    weights = torch.as_tensor(log_weights, dtype=torch.float64, device=device)
    if weights.numel() == 0:
        return weights
    weights = weights - torch.logsumexp(weights, dim=0)
    return torch.exp(weights)


def effective_sample_size(
    weights: Sequence[float] | torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> float:
    weights_t = torch.as_tensor(weights, dtype=torch.float64, device=device)
    if weights_t.numel() == 0:
        return 0.0
    return float(1.0 / torch.sum(weights_t * weights_t).item())


def systematic_resample(
    weights: Sequence[float] | torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> List[int]:
    weights_t = torch.as_tensor(weights, dtype=torch.float64, device=device)
    if weights_t.numel() == 0:
        return []
    cdf = torch.cumsum(weights_t, dim=0)
    step = 1.0 / weights_t.numel()
    start = torch.rand((), dtype=torch.float64).item() * step
    positions = start + step * torch.arange(
        weights_t.numel(),
        dtype=torch.float64,
        device=weights_t.device,
    )
    return torch.searchsorted(cdf, positions, right=False).tolist()


def multinomial_resample(
    weights: Sequence[float] | torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> List[int]:
    weights_t = torch.as_tensor(weights, dtype=torch.float64, device=device)
    if weights_t.numel() == 0:
        return []
    return torch.multinomial(weights_t, num_samples=weights_t.numel(), replacement=True).tolist()


@dataclass
class SMCDraftInput(SpecInput):
    last_token_ids: torch.Tensor
    new_seq_lens: torch.Tensor
    future_indices: Optional[FutureIndices] = None
    verify_done: Optional[torch.cuda.Event] = None
    committed_seq_lens_cpu: Optional[torch.Tensor] = None

    # Transient — set by prepare_for_v2_draft(), consumed by replay(), not persisted.
    seq_lens_steps: Optional[torch.Tensor] = None  # [gamma, bs]
    seq_lens_cpu_steps: Optional[torch.Tensor] = None  # [gamma, bs], CPU
    seq_lens_sum_steps: Optional[torch.Tensor] = None  # [gamma], CPU
    out_cache_loc_steps: Optional[torch.Tensor] = None  # [gamma, bs]
    positions: Optional[torch.Tensor] = None  # [bs]

    def __post_init__(self):
        super().__init__(SpecInputType.SMC_DRAFT)

    def get_spec_adjust_token_coefficient(self):
        return 1, 1

    @classmethod
    def create_idle_input(cls, device: torch.device):
        return cls(
            last_token_ids=torch.empty((0,), device=device, dtype=torch.int32),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            committed_seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
        )

    def prepare_for_v2_draft(
        self,
        req_to_token_pool,
        batch: ModelWorkerBatch,
        cuda_graph_runner: Optional[SMCDraftCudaGraphRunner],
        draft_model_runner: ModelRunner,
        gamma: int,
        draft_sampling_info: SamplingBatchInfo,
    ):
        bs = len(batch.seq_lens)
        device = batch.seq_lens.device

        # Compute draft_committed_lens in native int64 — no .to() cast.
        # .copy_() to int32 CG buffers handles conversion implicitly.
        draft_committed_lens = batch.seq_lens[:bs] - 1

        # Per-step seq_lens: [gamma, bs]
        # `draft_committed_lens` is the KV prefix length before replaying the
        # visible anchor token. Decode positions are derived from `seq_lens - 1`,
        # so replay step 0 must use `draft_committed_lens + 1` to place that
        # anchor at the current visible position while still reusing the tail
        # slots that start at `draft_committed_lens`.
        step_offsets = torch.arange(gamma, dtype=torch.int64, device=device)
        self.seq_lens_steps = (
            draft_committed_lens.unsqueeze(0) + 1 + step_offsets.unsqueeze(1)
        )

        # CPU mirror for attention backend metadata
        seq_lens_cpu = (
            batch.seq_lens_cpu
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.cpu()
        )
        draft_committed_lens_cpu = seq_lens_cpu[:bs] - 1
        step_offsets_cpu = torch.arange(gamma, dtype=torch.int64)
        self.seq_lens_cpu_steps = (
            draft_committed_lens_cpu.unsqueeze(0) + 1 + step_offsets_cpu.unsqueeze(1)
        )
        self.seq_lens_sum_steps = self.seq_lens_cpu_steps.sum(dim=1, dtype=torch.int64)

        # Compute out_cache_loc via Triton kernel (same kernel EAGLE uses).
        # Triton JIT handles int64 draft_committed_lens — no int32 cast needed.
        flat_out = torch.empty((bs * gamma,), dtype=torch.int64, device=device)
        assign_draft_cache_locs_page_size_1[(bs,)](
            batch.req_pool_indices[:bs],
            req_to_token_pool.req_to_token,
            draft_committed_lens,
            flat_out,
            req_to_token_pool.req_to_token.shape[1],
            1,  # topk=1 for SMC
            gamma,
        )
        self.out_cache_loc_steps = flat_out.view(bs, gamma).t().contiguous()

        # Positions for step 0 (captured graph increments via working_positions.add_(1))
        self.positions = clamp_position(self.seq_lens_steps[0])

        # Build a modified batch for ForwardBatch.init_new().
        # ForwardBatch.init_new picks up spec_info.positions at forward_batch_info.py:537.
        draft_batch = dataclasses.replace(
            batch,
            forward_mode=ForwardMode.DECODE,
            input_ids=self.last_token_ids,
            seq_lens=self.seq_lens_steps[0],
            seq_lens_cpu=self.seq_lens_cpu_steps[0],
            seq_lens_sum=int(self.seq_lens_sum_steps[0].item()),
            out_cache_loc=self.out_cache_loc_steps[0],
            sampling_info=draft_sampling_info,
            spec_info=self,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_logprob=True,
        )
        forward_batch = ForwardBatch.init_new(draft_batch, draft_model_runner)

        can_cuda_graph = bool(
            cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        )
        return forward_batch, can_cuda_graph

    def prepare_for_decode(self, batch: ScheduleBatch):
        if self.committed_seq_lens_cpu is None and self.verify_done is not None:
            self.verify_done.synchronize()
        batch.maybe_evict_swa()

        if self.committed_seq_lens_cpu is not None:
            batch.seq_lens_cpu = self.committed_seq_lens_cpu
        else:
            # Mirror EAGLE v2: overlap updates seq_lens first, then refresh the CPU
            # mirror before any allocation or metadata work consumes it.
            batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())

        server_args = get_global_server_args()
        gamma = max(int(server_args.smc_gamma or 1), 1)
        score_token_num = max(
            int(server_args.speculative_num_draft_tokens or gamma),
            gamma,
        )
        visible_seq_lens_cpu = batch.seq_lens_cpu.to(dtype=torch.int32)
        draft_committed_lens_cpu = (visible_seq_lens_cpu - 1).clamp_min_(0)
        current_allocated_lens_cpu = torch.tensor(
            [get_smc_reserved_kv_len(req) for req in batch.reqs],
            dtype=torch.int32,
            device="cpu",
        )
        required_lens_cpu = torch.maximum(
            current_allocated_lens_cpu,
            draft_committed_lens_cpu + score_token_num,
        )
        missing_lens_cpu = required_lens_cpu - current_allocated_lens_cpu
        num_needed_tokens = int(missing_lens_cpu.sum().item())

        if num_needed_tokens > 0:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                current_allocated_lens_cpu.to(device=batch.device),
                required_lens_cpu.to(device=batch.device),
                out_cache_loc,
                batch.batch_size(),
            )

        for req, required_len in zip(
            batch.reqs,
            required_lens_cpu.tolist(),
            strict=True,
        ):
            req.kv_allocated_len = int(required_len)
            set_smc_reserved_kv_len(req, required_len)
            req.decode_batch_idx += 1

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[new_indices]
            if self.committed_seq_lens_cpu is not None:
                self.committed_seq_lens_cpu = self.committed_seq_lens_cpu[
                    new_indices.cpu()
                ]
            return
        self.last_token_ids = self.last_token_ids[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]
        if self.committed_seq_lens_cpu is not None:
            self.committed_seq_lens_cpu = self.committed_seq_lens_cpu[new_indices.cpu()]

    def merge_batch(self, spec_info: "SMCDraftInput"):
        if self.committed_seq_lens_cpu is None or spec_info.committed_seq_lens_cpu is None:
            future_committed_seq_lens_cpu = None
        else:
            future_committed_seq_lens_cpu = torch.cat(
                [self.committed_seq_lens_cpu, spec_info.committed_seq_lens_cpu]
            )
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices]
                )
            )
            self.committed_seq_lens_cpu = future_committed_seq_lens_cpu
            return
        if self.committed_seq_lens_cpu is None and spec_info.committed_seq_lens_cpu is None:
            committed_seq_lens_cpu = None
        else:
            committed_seq_lens_cpu = torch.cat(
                [
                    (
                        self.committed_seq_lens_cpu
                        if self.committed_seq_lens_cpu is not None
                        else self.new_seq_lens.cpu()
                    ),
                    (
                        spec_info.committed_seq_lens_cpu
                        if spec_info.committed_seq_lens_cpu is not None
                        else spec_info.new_seq_lens.cpu()
                    ),
                ]
            )
        self.last_token_ids = torch.cat([self.last_token_ids, spec_info.last_token_ids])
        self.new_seq_lens = torch.cat([self.new_seq_lens, spec_info.new_seq_lens])
        self.committed_seq_lens_cpu = committed_seq_lens_cpu


@dataclass
class SMCScoreInput(SpecInput):
    draft_token: torch.Tensor
    draft_lengths: torch.Tensor
    draft_logprobs: torch.Tensor
    positions: torch.Tensor
    custom_mask: Optional[torch.Tensor]
    draft_token_num: int
    target_temperature: float
    linear_target_verify: bool = True
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL

    def __post_init__(self):
        super().__init__(SpecInputType.SMC_SCORE)
        self.num_tokens_per_req = self.draft_token_num

    def get_spec_adjust_token_coefficient(self):
        return self.draft_token_num, self.draft_token_num

    def use_linear_target_verify(self) -> bool:
        return self.linear_target_verify

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * batch_size,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )

        if self.custom_mask is None:
            raise RuntimeError(
                "SMC custom_mask is required for attention backends that do not "
                "natively support linear TARGET_VERIFY."
            )
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * batch_size
        )
        if self.custom_mask.numel() < mask_numel:
            self.custom_mask = torch.cat(
                [
                    self.custom_mask,
                    torch.full(
                        (mask_numel - self.custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )
        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    def _populate_linear_verify_metadata(self, forward_batch: ForwardBatch) -> None:
        batch_size = len(forward_batch.req_pool_indices)
        device = forward_batch.seq_lens.device
        prefix_lens = forward_batch.seq_lens.to(dtype=torch.int32)
        extend_seq_lens = torch.full(
            (batch_size,),
            self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        forward_batch.extend_prefix_lens = prefix_lens
        forward_batch.extend_seq_lens = extend_seq_lens
        forward_batch.extend_num_tokens = batch_size * self.draft_token_num
        forward_batch.extend_start_loc = torch.arange(
            0,
            forward_batch.extend_num_tokens,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = forward_batch.seq_lens.cpu()
        forward_batch.extend_prefix_lens_cpu = seq_lens_cpu
        forward_batch.extend_seq_lens_cpu = torch.full(
            (batch_size,),
            self.draft_token_num,
            dtype=torch.int32,
        )
        forward_batch.extend_logprob_start_lens_cpu = (
            forward_batch.extend_prefix_lens_cpu
        )

    def prepare_for_v2_verify(
        self,
        req_to_token_pool,
        batch: ModelWorkerBatch,
        target_worker,
    ):
        if not batch.forward_mode.is_idle():
            bs = len(batch.req_pool_indices)
            device = self.draft_token.device
            batch.input_ids = self.draft_token
            # Compute out_cache_loc at verify time (like EAGLE v2)
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )

        batch.forward_mode = (
            ForwardMode.IDLE if batch.forward_mode.is_idle() else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = self.capture_hidden_mode
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)
        if not batch.forward_mode.is_idle() and self.use_linear_target_verify():
            self._populate_linear_verify_metadata(verify_forward_batch)

        graph_runner = target_worker.model_runner.graph_runner
        model_runner_server_args = getattr(target_worker.model_runner, "server_args", None)
        captured_target_temperature = max(
            float(
                getattr(
                    model_runner_server_args,
                    "smc_target_temperature",
                    self.target_temperature,
                )
            ),
            SMC_MIN_TEMPERATURE,
        )
        target_temperature_matches_capture = math.isclose(
            float(self.target_temperature),
            captured_target_temperature,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        can_run_cuda_graph = bool(
            graph_runner
            and target_temperature_matches_capture
            and graph_runner.can_run(verify_forward_batch)
        )
        # (ccc) Keep the verify-prep graph decision on the batch so ModelRunner
        # does not independently re-enter graph replay on this path.
        verify_forward_batch.disable_graph_runner = not can_run_cuda_graph
        if can_run_cuda_graph:
            graph_runner.replay_prepare(verify_forward_batch)
        else:
            if not batch.forward_mode.is_idle():
                target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
                )

        return verify_forward_batch, can_run_cuda_graph

    def sample(
        self,
        batch: ModelWorkerBatch,
        logits_output,
    ):
        device = self.draft_token.device
        if batch.forward_mode.is_idle():
            empty_int = torch.empty((0,), dtype=torch.int32, device=device)
            empty_float = torch.empty((0,), dtype=torch.float32, device=device)
            empty_seq = torch.empty((0,), dtype=batch.seq_lens.dtype, device=device)
            return empty_int, empty_seq, empty_int, empty_float

        logprob_diffs = self.compute_logprob_diffs(batch, logits_output)
        bs = int(self.draft_lengths.shape[0])
        score_len = self.draft_token_num
        draft_token = self.draft_token.view(bs, score_len)
        committed_seq_lens = batch.seq_lens + 1 + self.draft_lengths
        safe_last_indices = torch.clamp(
            self.draft_lengths.to(torch.int64) - 1,
            min=0,
        )
        gathered_last_tokens = draft_token[:, 1:].gather(
            1, safe_last_indices.unsqueeze(1)
        ).squeeze(1)
        next_last_token_ids = torch.where(
            self.draft_lengths > 0,
            gathered_last_tokens,
            draft_token[:, 0],
        )

        return (
            self.draft_lengths,
            committed_seq_lens,
            next_last_token_ids,
            logprob_diffs,
        )

    def compute_logprob_diffs(
        self,
        batch: ModelWorkerBatch,
        logits_output,
    ) -> torch.Tensor:
        device = self.draft_token.device
        if batch.forward_mode.is_idle():
            return torch.empty((0,), dtype=torch.float32, device=device)

        bs = int(self.draft_lengths.shape[0])
        score_len = self.draft_token_num
        draft_token = self.draft_token.view(bs, score_len)
        # logits_output.next_token_logits is already log_probs
        # (log_softmax(logits/T) applied in CG capture or non-CG fallback)
        log_probs = logits_output.next_token_logits.view(bs, score_len, -1)
        # TARGET_VERIFY rows score the next token after each input token. The
        # proposal tensor is [anchor, draft_1, ..., draft_n], so the target
        # logprob for draft_i lives on row i - 1, not row i.
        gathered = log_probs[:, :-1].gather(
            2, draft_token[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        mask = (
            torch.arange(score_len - 1, device=device).unsqueeze(0)
            < self.draft_lengths.unsqueeze(1)
        )
        target_logprobs = torch.sum(gathered * mask, dim=1)
        return target_logprobs - self.draft_logprobs


def build_smc_positions(seq_lens: torch.Tensor, score_token_num: int) -> torch.Tensor:
    offsets = torch.arange(
        score_token_num, device=seq_lens.device, dtype=seq_lens.dtype
    )
    return (seq_lens.unsqueeze(1) + offsets).reshape(-1)


def build_smc_causal_mask(seq_lens: torch.Tensor, score_token_num: int) -> torch.Tensor:
    masks = []
    tril = torch.tril(
        torch.ones(
            (score_token_num, score_token_num),
            dtype=torch.bool,
            device=seq_lens.device,
        )
    )
    for seq_len in seq_lens.tolist():
        prefix = torch.ones(
            (score_token_num, seq_len), dtype=torch.bool, device=seq_lens.device
        )
        masks.append(torch.cat([prefix, tril], dim=1).reshape(-1))
    if masks:
        return torch.cat(masks)
    return torch.empty((0,), dtype=torch.bool, device=seq_lens.device)
