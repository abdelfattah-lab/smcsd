from __future__ import annotations
import logging
import copy
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
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


logger = logging.getLogger(__name__)

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
    output_ids = req.output_ids if output_ids is None else output_ids
    if output_ids:
        visible_seq_len = len(req.origin_input_ids) + len(output_ids)
        # Keep the latest visible token outside the committed prefix so it can
        # act as the anchor input for the next draft/verify step.
        return min(int(req.kv_committed_len), visible_seq_len - 1)
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
class SMCDraftInputV2Mixin:
    @classmethod
    def create_idle_input(cls, device: torch.device):
        return cls(
            last_token_ids=torch.empty((0,), device=device, dtype=torch.int32),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
        )

    def prepare_for_v2_draft(
        self: "SMCDraftInput",
        req_to_token_pool,
        batch: ModelWorkerBatch,
        cuda_graph_runner: Optional[SMCDraftCudaGraphRunner],
        draft_model_runner: ModelRunner,
        gamma: int,
        draft_sampling_info: SamplingBatchInfo,
    ):
        if not batch.forward_mode.is_idle():
            bs = len(batch.seq_lens)
            batch.out_cache_loc = torch.empty(
                (bs * gamma,),
                dtype=torch.int64,
                device=self.last_token_ids.device,
            )
            assign_draft_cache_locs_page_size_1[(bs,)](
                batch.req_pool_indices[:bs],
                req_to_token_pool.req_to_token,
                batch.seq_lens[:bs],
                batch.out_cache_loc,
                req_to_token_pool.req_to_token.shape[1],
                1,
                gamma,
            )

        # Mirror Eagle v2: the live batch already carries the committed prefix
        # preceding the carried anchor token (`last_token_ids`).
        self.positions = batch.seq_lens.to(torch.int64)
        draft_batch = dataclasses.replace(
            batch,
            input_ids=self.last_token_ids,
            sampling_info=draft_sampling_info,
            spec_info=self,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_logprob=True,
            top_logprobs_nums=[0] * len(batch.seq_lens),
            token_ids_logprobs=[None] * len(batch.seq_lens),
        )
        forward_batch = ForwardBatch.init_new(draft_batch, draft_model_runner)

        can_cuda_graph = bool(
            cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        )
        return forward_batch, can_cuda_graph

    def prepare_for_decode(self: "SMCDraftInput", batch: ScheduleBatch):
        batch.maybe_evict_swa()
        batch.maybe_wait_verify_done()
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())

        # Pre-allocate KV slots covering the Eagle-style topk=1 draft replay
        # plus the verify pass. The forward stream only reads from the shared
        # req_to_token mapping after this point.
        server_args = get_global_server_args()
        gamma = max(int(server_args.smc_gamma or 1), 1)
        score_token_num = max(
            int(server_args.speculative_num_draft_tokens or gamma),
            gamma,
        )
        committed_seq_lens_cpu = batch.seq_lens_cpu.to(dtype=torch.int32)
        cur_kv_lens_cpu = torch.tensor(
            [int(req.kv_allocated_len) for req in batch.reqs],
            dtype=torch.int32,
            device="cpu",
        )
        nxt_kv_lens_cpu = torch.maximum(
            cur_kv_lens_cpu,
            committed_seq_lens_cpu + score_token_num,
        )
        missing_lens_cpu = nxt_kv_lens_cpu - cur_kv_lens_cpu
        num_needed_tokens = int(missing_lens_cpu.sum().item())

        if num_needed_tokens > 0:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                cur_kv_lens_cpu.to(device=batch.device),
                nxt_kv_lens_cpu.to(device=batch.device),
                out_cache_loc,
                batch.batch_size(),
            )

        for req, next_len in zip(
            batch.reqs,
            nxt_kv_lens_cpu.tolist(),
            strict=True,
        ):
            req.kv_allocated_len = int(next_len)
            req.decode_batch_idx += 1

    def filter_batch(
        self: "SMCDraftInput", new_indices: torch.Tensor, has_been_filtered: bool = True
    ):
        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[new_indices]
            return
        self.last_token_ids = self.last_token_ids[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]

    def merge_batch(self: "SMCDraftInput", spec_info: "SMCDraftInput"):
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
class SMCDraftInput(SpecInput, SMCDraftInputV2Mixin):
    last_token_ids: torch.Tensor
    new_seq_lens: torch.Tensor
    future_indices: Optional[FutureIndices] = None
    verify_done: Optional[torch.cuda.Event] = None
    positions: Optional[torch.Tensor] = None  # [bs]

    def __post_init__(self):
        super().__init__(SpecInputType.SMC_DRAFT)

    def get_spec_adjust_token_coefficient(self):
        return 1, 1


@dataclass
class SMCScoreInput(SpecInput):
    draft_token: torch.Tensor
    draft_lengths: torch.Tensor
    draft_logprobs: torch.Tensor
    positions: torch.Tensor
    custom_mask: Optional[torch.Tensor]
    draft_token_num: int
    spec_steps: int  # number of draft tokens (gamma), mirrors EAGLE's spec_steps
    target_temperature: float
    linear_target_verify: bool = True
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL
    smc_logprob_diffs: Optional[torch.Tensor] = None  # filled by sample()

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
        verify_seq_lens = batch.seq_lens
        verify_seq_lens_cpu = (
            batch.seq_lens_cpu
            if batch.seq_lens_cpu is not None
            else verify_seq_lens.cpu()
        )
        verify_seq_lens_sum = int(verify_seq_lens_cpu.sum().item())

        if not batch.forward_mode.is_idle():
            bs = len(batch.req_pool_indices)
            device = self.draft_token.device
            batch.input_ids = self.draft_token
            # The live SMC batch already carries the committed prefix before the
            # anchor token, matching Eagle's verify start-offset contract.
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=verify_seq_lens,
                end_offset=verify_seq_lens + self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )

        batch.forward_mode = (
            ForwardMode.IDLE if batch.forward_mode.is_idle() else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = self.capture_hidden_mode
        verify_batch = copy.copy(batch)
        verify_batch.seq_lens = verify_seq_lens
        verify_batch.seq_lens_cpu = verify_seq_lens_cpu
        verify_batch.seq_lens_sum = verify_seq_lens_sum
        verify_forward_batch = ForwardBatch.init_new(
            verify_batch, target_worker.model_runner
        )
        if not batch.forward_mode.is_idle() and self.use_linear_target_verify():
            self._populate_linear_verify_metadata(verify_forward_batch)

        graph_runner = target_worker.model_runner.graph_runner
        can_run_cuda_graph = bool(
            graph_runner and graph_runner.can_run(verify_forward_batch)
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
        """Accept all draft tokens, compute bonus + logprob diffs.

        Mirrors EagleVerifyInputV2Mixin.sample() (eagle_info_v2.py:272-388).
        Returns the same 3-tuple: (predict, accept_length, accept_index).
        SMC logprob diffs are stored on self.smc_logprob_diffs.
        """
        device = self.draft_token.device
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.int32, device=device)
            accept_length = torch.empty(0, dtype=torch.int32, device=device)
            accept_index = torch.empty(0, dtype=torch.int32, device=device)
            return predict, accept_length, accept_index

        bs = len(batch.seq_lens)
        next_token_logits = logits_output.next_token_logits
        ss = self.spec_steps

        # Allocate predict and accept tensors (EAGLE naming)
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(next_token_logits.shape)[:-1]
        predict = torch.zeros(
            predict_shape, dtype=torch.int32, device=device
        ).flatten()
        accept_length = torch.full(
            (bs,), ss, dtype=torch.int32, device=device
        )
        # accept_index must use GLOBAL flat offsets into predict (not local
        # per-request [0..gamma]).  predict is flat (bs * draft_token_num,) and
        # fill_new_verified_id indexes it with accept_index to extract the
        # bonus token for each request.
        req_offsets = (
            torch.arange(bs, device=device, dtype=torch.int32).unsqueeze(1)
            * self.draft_token_num
        )
        accept_index = (
            torch.arange(ss + 1, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(bs, -1)
            + req_offsets
        ).contiguous()

        # Fill predict: [d0, ..., d_{ss-1}, bonus] per request
        predict_view = predict.reshape(bs, self.draft_token_num)
        predict_view[:, :ss] = candidates[:, 1:].to(torch.int32)

        logits = next_token_logits.view(bs, self.draft_token_num, -1)
        base_log_probs = F.log_softmax(logits, dim=-1)

        # SMC logprob diffs (importance weights)
        # Aligned with the native SMC reference: score the drafted continuation
        # with the target model's base logprobs and apply temperature as a
        # post-hoc 1 / T scaling.
        gathered = base_log_probs[:, :-1].gather(
            2, candidates[:, 1:].long().unsqueeze(-1)
        ).squeeze(-1)
        target_logprobs = gathered.sum(dim=1) / self.target_temperature
        self.smc_logprob_diffs = target_logprobs - self.draft_logprobs

        return predict, accept_length, accept_index


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
