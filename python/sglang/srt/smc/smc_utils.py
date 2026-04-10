"""Shared SMC utilities: particle management, KV release, resampling."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Optional, Sequence

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    pass

SMC_MIN_TEMPERATURE = 1e-5


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
    temperature: float,
    return_logprob: bool,
    output_ids: Optional[Sequence[int]] = None,
) -> Req:
    sampling_params = copy.copy(parent_req.sampling_params)
    sampling_params.temperature = max(temperature, SMC_MIN_TEMPERATURE)
    if isinstance(sampling_params.custom_params, dict):
        sampling_params.custom_params = dict(sampling_params.custom_params)

    particle_req = Req(
        rid=f"{parent_req.rid}_smc_p{particle_idx}_particle",
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
    particle_req.smc_particle_idx = particle_idx
    return particle_req


def _empty_prefix_indices() -> torch.Tensor:
    return torch.empty((0,), dtype=torch.int64)


def compute_smc_shared_prefix_len(
    req: Req,
    *,
    output_ids: Optional[Sequence[int]] = None,
) -> int:
    """Return the seq_lens for this SMC particle.

    With bonus tokens, kv_committed_len = visible_seq_len - 1 is a maintained
    invariant (the bonus token is the next anchor but has no KV yet).  So
    seq_lens = kv_committed_len directly.
    """
    return int(req.kv_committed_len)


def _release_internal_req(
    req: Req,
    req_to_token_pool,
    token_to_kv_pool_allocator,
):
    if req.req_pool_idx is None:
        return

    allocated_len = int(req.kv_allocated_len)
    if allocated_len > 0:
        indices = req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ].to(dtype=torch.int64, copy=True)
        token_to_kv_pool_allocator.dec_ref_and_free(indices)

    req_to_token_pool.free(req)
    req.prefix_indices = _empty_prefix_indices()
    req.kv_committed_len = 0
    req.kv_allocated_len = 0


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


def should_resample(
    normalized_weights: torch.Tensor,
    n_particles: int,
    threshold: float,
    device: Optional[torch.device | str] = None,
) -> bool:
    """Fused ESS threshold check — does comparison on GPU, syncs only the boolean."""
    weights_t = torch.as_tensor(normalized_weights, dtype=torch.float64, device=device)
    if weights_t.numel() == 0:
        return False
    # ESS = 1 / sum(w^2).  Resample when ESS < n * threshold.
    # Equivalent: sum(w^2) > 1 / (n * threshold)
    sum_sq = torch.sum(weights_t * weights_t)
    return bool((sum_sq > 1.0 / (n_particles * threshold)).item())


def systematic_resample(
    weights: Sequence[float] | torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Returns ancestor indices as a GPU tensor (no GPU→CPU sync)."""
    weights_t = torch.as_tensor(weights, dtype=torch.float64, device=device)
    if weights_t.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    cdf = torch.cumsum(weights_t, dim=0)
    n = weights_t.numel()
    step = 1.0 / n
    start = torch.rand((), dtype=torch.float64, device=weights_t.device) * step
    positions = start + step * torch.arange(
        n,
        dtype=torch.float64,
        device=weights_t.device,
    )
    return torch.searchsorted(cdf, positions, right=False)


def multinomial_resample(
    weights: Sequence[float] | torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Returns ancestor indices as a GPU tensor (no GPU→CPU sync)."""
    weights_t = torch.as_tensor(weights, dtype=torch.float64, device=device)
    if weights_t.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.multinomial(
        weights_t, num_samples=weights_t.numel(), replacement=True
    )
