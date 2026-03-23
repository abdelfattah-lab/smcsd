from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

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
from sglang.srt.managers.utils import get_alloc_len_per_decode
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs_func
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

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
    temperature_multiplier: float,
    return_logprob: bool,
    output_ids: Optional[Sequence[int]] = None,
) -> Req:
    sampling_params = copy.copy(parent_req.sampling_params)
    parent_temperature = sampling_params.temperature
    if parent_temperature is None or parent_temperature <= 0:
        sampling_params.temperature = SMC_MIN_TEMPERATURE
    else:
        sampling_params.temperature = max(
            parent_temperature * temperature_multiplier,
            SMC_MIN_TEMPERATURE,
        )
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
                temperature_multiplier=server_args.smc_target_temperature,
                return_logprob=True,
                output_ids=base_output_ids,
            )
        )
        draft_reqs.append(
            clone_req_for_smc_particle(
                req,
                particle_idx=i,
                role="draft",
                temperature_multiplier=server_args.smc_draft_temperature,
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


def resolve_smc_proposal_length(
    req: Req, proposal_tokens: Sequence[int]
) -> tuple[int, bool]:
    if req.finished():
        return 0, True

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

    current_output_len = len(req.output_ids)
    proposal_len = 0
    for token_id in proposal_tokens:
        proposal_len += 1
        if req.to_finish:
            return proposal_len, True
        if current_output_len + proposal_len >= req.sampling_params.max_new_tokens:
            return proposal_len, True
        if req.vocab_size is not None and (token_id > req.vocab_size or token_id < 0):
            return proposal_len, True
        if not req.sampling_params.ignore_eos and token_id in stop_token_ids:
            return proposal_len, True

    return proposal_len, False


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


def _empty_prefix_indices() -> torch.Tensor:
    return torch.empty((0,), dtype=torch.int64)


def _release_internal_req(
    req: Req,
    req_to_token_pool,
    token_to_kv_pool_allocator,
):
    if req.req_pool_idx is None:
        return

    if req.kv_allocated_len > 0:
        indices = req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv_allocated_len
        ].to(dtype=torch.int64, copy=True)
        token_to_kv_pool_allocator.dec_ref_and_free(indices)

    req_to_token_pool.free(req)
    req.prefix_indices = _empty_prefix_indices()
    req.kv_committed_len = 0
    req.kv_allocated_len = 0


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

    if dst_req.req_pool_idx is not None and dst_req.kv_allocated_len > 0:
        old_indices = req_to_token_pool.req_to_token[
            dst_req.req_pool_idx, : dst_req.kv_allocated_len
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


def normalize_log_weights(log_weights: Sequence[float]) -> torch.Tensor:
    weights = torch.tensor(log_weights, dtype=torch.float64)
    if weights.numel() == 0:
        return weights
    weights = weights - torch.logsumexp(weights, dim=0)
    return torch.exp(weights)


def effective_sample_size(weights: Sequence[float]) -> float:
    weights_t = torch.as_tensor(weights, dtype=torch.float64)
    if weights_t.numel() == 0:
        return 0.0
    return float(1.0 / torch.sum(weights_t * weights_t).item())


def systematic_resample(weights: Sequence[float]) -> List[int]:
    weights_t = torch.as_tensor(weights, dtype=torch.float64)
    if weights_t.numel() == 0:
        return []
    cdf = torch.cumsum(weights_t, dim=0)
    step = 1.0 / weights_t.numel()
    start = torch.rand((), dtype=torch.float64).item() * step
    positions = start + step * torch.arange(weights_t.numel(), dtype=torch.float64)
    return torch.searchsorted(cdf, positions, right=False).tolist()


def multinomial_resample(weights: Sequence[float]) -> List[int]:
    weights_t = torch.as_tensor(weights, dtype=torch.float64)
    if weights_t.numel() == 0:
        return []
    return torch.multinomial(weights_t, num_samples=weights_t.numel(), replacement=True).tolist()


@dataclass
class SMCDraftInput(SpecInput):
    last_token_ids: torch.Tensor
    new_seq_lens: torch.Tensor
    future_indices: Optional[FutureIndices] = None
    verify_done: Optional[torch.cuda.Event] = None

    def __post_init__(self):
        super().__init__(SpecInputType.SMC_DRAFT)

    def get_spec_adjust_token_coefficient(self):
        return 1, 1

    @classmethod
    def create_idle_input(cls, device: torch.device):
        return cls(
            last_token_ids=torch.empty((0,), device=device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
        )

    def prepare_for_decode(self, batch: ScheduleBatch):
        if self.verify_done is not None:
            self.verify_done.synchronize()
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[new_indices]
            return
        self.last_token_ids = self.last_token_ids[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]

    def merge_batch(self, spec_info: "SMCDraftInput"):
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices]
                )
            )
            return
        self.last_token_ids = torch.cat([self.last_token_ids, spec_info.last_token_ids])
        self.new_seq_lens = torch.cat([self.new_seq_lens, spec_info.new_seq_lens])


@dataclass
class SMCScoreInput(SpecInput):
    draft_token: torch.Tensor
    draft_lengths: torch.Tensor
    draft_logprobs: torch.Tensor
    draft_finished: torch.Tensor
    positions: torch.Tensor
    custom_mask: torch.Tensor
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL

    def __post_init__(self):
        super().__init__(SpecInputType.SMC_SCORE)

    def get_spec_adjust_token_coefficient(self):
        return self.draft_token_num, self.draft_token_num

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

    def prepare_for_v2_verify(
        self,
        req_to_token_pool,
        batch: ModelWorkerBatch,
        target_worker,
    ):
        if not batch.forward_mode.is_idle():
            self._ensure_allocated_score_slots(req_to_token_pool, batch, target_worker)
            batch.input_ids = self.draft_token
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=len(batch.req_pool_indices),
                draft_token_num=self.draft_token_num,
                device=batch.input_ids.device,
            )

        batch.forward_mode = (
            ForwardMode.IDLE if batch.forward_mode.is_idle() else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = self.capture_hidden_mode
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        if can_run_cuda_graph:
            target_worker.model_runner.graph_runner.replay_prepare(verify_forward_batch)
        else:
            if not batch.forward_mode.is_idle():
                target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
                )

        return verify_forward_batch, can_run_cuda_graph

    def _ensure_allocated_score_slots(
        self,
        req_to_token_pool,
        batch: ModelWorkerBatch,
        target_worker,
    ) -> None:
        allocator = target_worker.model_runner.token_to_kv_pool_allocator
        required_lens = batch.seq_lens + self.draft_token_num
        for req, required_len in zip(batch.reqs, required_lens.tolist(), strict=True):
            required_len = int(required_len)
            if req.kv_allocated_len >= required_len:
                continue
            missing = required_len - req.kv_allocated_len
            new_slots = allocator.alloc(missing)
            if new_slots is None:
                raise RuntimeError(
                    "SMC target scoring could not allocate verify slots: "
                    f"rid={req.rid}, required_len={required_len}, "
                    f"allocated_len={req.kv_allocated_len}, missing={missing}"
                )
            req_to_token_pool.write(
                (req.req_pool_idx, slice(req.kv_allocated_len, required_len)),
                new_slots.to(torch.int32),
            )
            req.kv_allocated_len = required_len

    def sample(
        self,
        batch: ModelWorkerBatch,
        logits_output,
        req_to_token_pool,
        token_to_kv_pool_allocator,
    ):
        device = self.draft_token.device
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int32, device=device)
            return empty, empty, empty, empty, empty

        bs = len(batch.reqs)
        score_len = self.draft_token_num
        draft_token = self.draft_token.view(bs, score_len)
        temperatures = torch.repeat_interleave(
            batch.sampling_info.temperatures, score_len, dim=0
        )
        log_probs = F.log_softmax(logits_output.next_token_logits / temperatures, dim=-1)
        gathered = log_probs.gather(1, self.draft_token.view(-1, 1)).view(bs, score_len)
        mask = (
            torch.arange(score_len - 1, device=device).unsqueeze(0)
            < self.draft_lengths.unsqueeze(1)
        )
        target_logprobs = torch.sum(gathered[:, 1:] * mask, dim=1)

        rewritten_tokens = draft_token[:, 1:].clone()
        rewritten_lengths = self.draft_lengths.clone()
        next_last_token_ids = torch.empty((bs,), dtype=torch.int64, device=device)
        committed_seq_lens = batch.seq_lens + 1 + rewritten_lengths
        allocated_seq_lens = batch.seq_lens + score_len

        parent_to_rows: Dict[Req, List[int]] = {}
        for row, req in enumerate(batch.reqs):
            parent_to_rows.setdefault(req.smc_parent, []).append(row)
            req.smc_parent.smc_state.particles[req.smc_particle_idx].log_weight += (
                float((target_logprobs[row] - self.draft_logprobs[row]).item())
            )

        for row, req in enumerate(batch.reqs):
            if rewritten_lengths[row] > 0:
                next_last_token_ids[row] = rewritten_tokens[
                    row, rewritten_lengths[row] - 1
                ]
            else:
                next_last_token_ids[row] = req.output_ids[-1]

        for row, req in enumerate(batch.reqs):
            req.kv_committed_len = int(committed_seq_lens[row].item())
            req.kv_allocated_len = int(allocated_seq_lens[row].item())
            req.prefix_indices = req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.kv_committed_len
            ].to(dtype=torch.int64, copy=True)

        for parent_req, rows in parent_to_rows.items():
            state: SMCRequestState = parent_req.smc_state
            active_rows = [row for row in rows if not bool(self.draft_finished[row].item())]
            if len(active_rows) <= 1:
                continue

            active_particles = [state.particles[batch.reqs[row].smc_particle_idx] for row in active_rows]
            normalized = normalize_log_weights([particle.log_weight for particle in active_particles])
            ess = effective_sample_size(normalized)
            if ess >= len(active_rows) * state.resample_threshold:
                continue

            if state.resample_method == "multinomial":
                ancestors = multinomial_resample(normalized)
            else:
                ancestors = systematic_resample(normalized)

            source_rows = [active_rows[idx] for idx in ancestors]
            token_snapshots = []
            output_snapshots = []
            weight_snapshots = []
            committed_len_snapshots = []
            for src_row in source_rows:
                src_req = batch.reqs[src_row]
                src_new_len = int(allocated_seq_lens[src_row].item())
                token_snapshots.append(
                    req_to_token_pool.req_to_token[src_req.req_pool_idx, :src_new_len].clone()
                )
                output_snapshots.append(list(src_req.output_ids))
                weight_snapshots.append(
                    state.particles[src_req.smc_particle_idx].log_weight
                )
                committed_len_snapshots.append(int(committed_seq_lens[src_row].item()))

            for (
                dst_row,
                src_row,
                snapshot,
                output_snapshot,
                weight_snapshot,
                committed_len_snapshot,
            ) in zip(
                active_rows,
                source_rows,
                token_snapshots,
                output_snapshots,
                weight_snapshots,
                committed_len_snapshots,
                strict=True,
            ):
                dst_req = batch.reqs[dst_row]
                if dst_row != src_row and dst_req.kv_allocated_len > 0:
                    old_indices = req_to_token_pool.req_to_token[
                        dst_req.req_pool_idx, : dst_req.kv_allocated_len
                    ].to(torch.int64, copy=True)
                    token_to_kv_pool_allocator.dec_ref_and_free(old_indices)
                    token_to_kv_pool_allocator.inc_ref(snapshot.to(torch.int64))
                    req_to_token_pool.write(
                        (dst_req.req_pool_idx, slice(0, snapshot.shape[0])),
                        snapshot.to(torch.int32),
                    )
                    dst_req.output_ids = output_snapshot
                    dst_req.finished_reason = None
                    dst_req.finished_len = None
                    dst_req.kv_committed_len = committed_len_snapshot
                    dst_req.kv_allocated_len = snapshot.shape[0]
                    dst_req.prefix_indices = req_to_token_pool.req_to_token[
                        dst_req.req_pool_idx, : dst_req.kv_committed_len
                    ].to(dtype=torch.int64, copy=True)
                rewritten_tokens[dst_row] = rewritten_tokens[src_row]
                rewritten_lengths[dst_row] = rewritten_lengths[src_row]
                next_last_token_ids[dst_row] = next_last_token_ids[src_row]
                committed_seq_lens[dst_row] = committed_seq_lens[src_row]
                state.particles[dst_req.smc_particle_idx].log_weight = weight_snapshot

            for row in active_rows:
                state.particles[batch.reqs[row].smc_particle_idx].log_weight = 0.0

        predict = rewritten_tokens.flatten().to(torch.int32)
        accept_index = torch.empty((0,), dtype=torch.int32, device=device)
        return (
            predict,
            rewritten_lengths.to(torch.int32),
            accept_index,
            committed_seq_lens.to(torch.int32),
            next_last_token_ids,
        )


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
