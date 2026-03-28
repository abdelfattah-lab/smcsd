from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Union

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.smc_debug_utils import append_smc_diag_record
from sglang.srt.speculative.smc_draft_cuda_graph_runner import SMCDraftCudaGraphRunner
from sglang.srt.speculative.smc_info import (
    SMCDraftInput,
    SMC_MIN_TEMPERATURE,
    SMCScoreInput,
    build_smc_positions,
    compute_smc_shared_prefix_len,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.standalone_worker_v2 import (
    StandaloneDraftWorker,
    _get_plan_stream,
)
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)


class SMCDraftWorker(StandaloneDraftWorker):
    """StandaloneDraftWorker with SMC-specific attention backend and CUDA graphs."""

    def init_attention_backend(self):
        super().init_attention_backend()
        self.smc_draft_attn_backend = None
        if self.server_args.smc_gamma > 1:
            self.smc_draft_attn_backend = DraftBackendFactory(
                self.server_args,
                self.draft_runner,
                topk=1,
                speculative_num_steps=self.server_args.smc_gamma + 1,
            ).create_decode_backend()

    def init_cuda_graphs(self):
        self.smc_draft_cuda_graph_runner = None
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return
        if self.server_args.model_impl == "mindspore":
            return
        if self.smc_draft_attn_backend is None:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture SMC draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.smc_draft_cuda_graph_runner = SMCDraftCudaGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture SMC draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
        )

        # Capture draft-extend CUDA graph (same conditions as EagleDraftWorker).
        # This accelerates the incremental draft KV fill after verify.
        from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
            EAGLEDraftExtendCudaGraphRunner,
        )
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLAMultiStepDraftBackend,
        )
        from sglang.srt.utils import is_cuda as _is_cuda_fn
        from sglang.srt.utils import is_hip as _is_hip_fn
        from sglang.srt.utils import is_npu as _is_npu_fn

        _is_npu_val = _is_npu_fn()
        _is_hip_val = _is_hip_fn()
        _is_cuda_val = _is_cuda_fn()

        supports_hip_aiter_draft_extend_graph = False
        if _is_hip_val:
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        supports_cuda_draft_extend_graph = _is_cuda_val and (
            isinstance(self.draft_attn_backend, TritonMultiStepDraftBackend)
            or isinstance(self.draft_attn_backend, TRTLLMMLAMultiStepDraftBackend)
        )

        Device2ExtendCudaGraphRunner = {}
        if _is_npu_val:
            from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
                EAGLEDraftExtendNpuGraphRunner,
            )

            Device2ExtendCudaGraphRunner["npu"] = EAGLEDraftExtendNpuGraphRunner
        if _is_cuda_val or _is_hip_val:
            Device2ExtendCudaGraphRunner["cuda"] = EAGLEDraftExtendCudaGraphRunner

        if self.draft_extend_attn_backend and (
            _is_npu_val
            or supports_cuda_draft_extend_graph
            or supports_hip_aiter_draft_extend_graph
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture SMC draft extend cuda graph begin. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture SMC draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
                f"mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )


    def draft(self, model_worker_batch: ModelWorkerBatch) -> SMCScoreInput:
        """Run SMC draft phase: propose gamma tokens per request.

        Mirrors EagleDraftWorker.draft(model_worker_batch) interface.
        Returns SMCScoreInput ready for verify, like EAGLE returns EagleVerifyInput.
        """
        outer = self._smc_outer_worker
        reqs = list(model_worker_batch.reqs) if hasattr(model_worker_batch, 'reqs') and model_worker_batch.reqs else []
        draft_input = model_worker_batch.spec_info
        if draft_input is not None and hasattr(draft_input, 'last_token_ids') and draft_input.last_token_ids is not None:
            last_token_ids = draft_input.last_token_ids
        else:
            last_token_ids = torch.tensor(
                [
                    req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
                    for req in reqs
                ],
                dtype=torch.int32,
                device=outer.device,
            )

        draft_tokens, draft_logprobs, draft_lengths, _ = outer._run_eagle_style_draft_reqs(
            reqs=reqs,
            model_worker_batch=model_worker_batch,
            last_token_ids=last_token_ids,
            draft_sampling_info=model_worker_batch.sampling_info,
        )

        # Build score tokens: [anchor, d0, ..., d_{gamma-1}]
        smc_gamma = outer.smc_gamma
        score_token_num = outer.server_args.speculative_num_draft_tokens
        score_tokens = torch.cat(
            [last_token_ids.unsqueeze(1), draft_tokens], dim=1
        )[:, :score_token_num]
        if score_tokens.shape[1] < score_token_num:
            pad = score_tokens[:, -1:].expand(
                -1, score_token_num - score_tokens.shape[1]
            )
            score_tokens = torch.cat([score_tokens, pad], dim=1)

        # Build SMCScoreInput (= verify input for SMC)
        target_temperature = max(
            float(outer.server_args.smc_target_temperature), SMC_MIN_TEMPERATURE
        )

        score_input = SMCScoreInput(
            draft_token=score_tokens.reshape(-1).contiguous(),
            draft_lengths=draft_lengths,
            draft_logprobs=draft_logprobs,
            positions=build_smc_positions(model_worker_batch.seq_lens, score_token_num),
            custom_mask=None,
            draft_token_num=score_token_num,
            spec_steps=smc_gamma,
            target_temperature=target_temperature,
            # ServerArgs enforces a Triton-only target path for SMC.
            linear_target_verify=True,
        )
        return score_input

    def draft_forward(
        self, forward_batch: ForwardBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = self.server_args.smc_gamma
        batch_size = forward_batch.batch_size
        sampled_token_ids = torch.empty(
            (gamma, batch_size), dtype=torch.int32, device=self.device
        )
        sampled_token_logprobs = torch.empty(
            (gamma, batch_size), dtype=torch.float32, device=self.device
        )
        base_seq_lens = forward_batch.seq_lens
        base_seq_lens_cpu = (
            forward_batch.seq_lens_cpu
            if forward_batch.seq_lens_cpu is not None
            else forward_batch.seq_lens.cpu()
        )
        base_seq_lens_sum = int(base_seq_lens_cpu.sum().item())
        out_cache_loc_steps = (
            forward_batch.out_cache_loc.view(batch_size, gamma).t().contiguous()
        )
        working_input_ids = forward_batch.input_ids.clone()
        working_positions = forward_batch.positions.clone()

        for step in range(gamma):
            forward_batch.input_ids = working_input_ids
            forward_batch.seq_lens = base_seq_lens + step
            forward_batch.seq_lens_cpu = base_seq_lens_cpu + step
            forward_batch.seq_lens_sum = base_seq_lens_sum + batch_size * step
            forward_batch.out_cache_loc = out_cache_loc_steps[step]
            forward_batch.positions = working_positions

            if self.smc_draft_attn_backend is not None:
                forward_batch.attn_backend = self.smc_draft_attn_backend.attn_backends[
                    step
                ]
                logits_output = self.draft_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output
            else:
                logits_output = self.draft_runner.forward(forward_batch).logits_output

            next_token_ids = self.draft_runner.sample(logits_output, forward_batch)
            sampled_token_ids[step].copy_(next_token_ids.to(dtype=torch.int32))
            sampled_token_logprobs[step].copy_(
                logits_output.next_token_logprobs.to(dtype=torch.float32)
            )
            working_input_ids = next_token_ids.to(dtype=working_input_ids.dtype)
            working_positions.add_(1)

        return (
            sampled_token_ids.transpose(0, 1).contiguous(),
            sampled_token_logprobs.transpose(0, 1).contiguous(),
        )


class _SMCInternalTreeCache:
    """Minimal allocator wrapper for worker-local internal SMC batches."""

    def __init__(self, token_to_kv_pool_allocator):
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = token_to_kv_pool_allocator.page_size

    def is_chunk_cache(self) -> bool:
        return False

    def evict(self, *args, **kwargs) -> None:
        return None

    def pretty_print(self) -> None:
        return None

    def supports_mamba(self) -> bool:
        return False

    def supports_swa(self) -> bool:
        return False


class SMCWorkerV2(EAGLEWorkerV2):
    def __init__(
        self,
        server_args,
        gpu_id,
        tp_rank,
        dp_rank,
        moe_ep_rank,
        attn_cp_rank,
        moe_dp_rank,
        nccl_port,
        target_worker,
    ):
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = SMCDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

        self.smc_gamma = server_args.smc_gamma
        self._draft_worker._smc_outer_worker = self
        self._internal_tree_cache = _SMCInternalTreeCache(
            self.token_to_kv_pool_allocator
        )

        # Note: verify post-processing (bonus + logprob diff) is computed
        # inside SMCScoreInput.sample() — no separate CUDA graph needed.

    @property
    def model_runner(self):
        return self.target_worker.model_runner

    @property
    def model_config(self):
        return self.target_worker.model_config

    def _get_forward_model_worker_batch(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        is_overlap_batch: bool,
    ) -> ModelWorkerBatch:
        model_worker_batch = batch if is_overlap_batch else batch.get_model_worker_batch()
        if (
            not is_overlap_batch
            and model_worker_batch.sampling_info is not None
        ):
            # Mirror overlap mode: keep a forward-local sampling copy so draft/verify
            # sampling kernels cannot poison the reusable ScheduleBatch state.
            model_worker_batch.sampling_info = (
                model_worker_batch.sampling_info.copy_for_forward()
            )
        return model_worker_batch

    def forward_batch_generation(
        self, batch: Union[ScheduleBatch, ModelWorkerBatch]
    ) -> GenerationBatchResult:
        is_overlap_batch = isinstance(batch, ModelWorkerBatch)
        draft_input = (
            batch.spec_info if isinstance(batch.spec_info, SMCDraftInput) else None
        )
        
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = self._get_forward_model_worker_batch(
                batch, is_overlap_batch
            )
            result = self.target_worker.forward_batch_generation(model_worker_batch)
            if is_overlap_batch:
                result.next_draft_input = self._build_prefill_overlap_input(
                    model_worker_batch, result
                )
            return result

        if not batch.reqs:
            return self._build_empty_decode_result(is_overlap_batch)

        if not is_overlap_batch and draft_input is not None:
            draft_input.prepare_for_decode(batch)

        model_worker_batch = self._get_forward_model_worker_batch(
            batch, is_overlap_batch
        )

        # 3-step EAGLE-style flow: draft → verify → draft_extend
        with self.draft_worker.draft_tp_context(
            self.draft_worker.draft_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            verify_input = self.draft_worker.draft(model_worker_batch)

        model_worker_batch.spec_info = verify_input
        batch_output = self.verify(model_worker_batch)

        # No draft_extend needed: without a bonus token, the last draft
        # token is the anchor for the next step and its draft KV already
        # exists from draft_forward().

        return batch_output

    def verify(self, batch: ModelWorkerBatch) -> GenerationBatchResult:
        """SMC verify: accept ALL draft tokens, compute logprob diffs.

        Mirrors EAGLEWorkerV2.verify(batch) — same single-arg interface.
        Reads SMCScoreInput from batch.spec_info (set by draft()).
        """
        from sglang.srt.speculative.eagle_info_v2 import fill_new_verified_id

        # Record stream (EAGLE line 745)
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        # Parse args (EAGLE line 750-752)
        verify_input: SMCScoreInput = batch.spec_info
        verify_input.num_tokens_per_req = self.smc_gamma + 1
        bs = len(batch.seq_lens)

        # Prepare for target verify (EAGLE line 756-763)
        # No -1 adjustment: with bonus + draft_extend, all tokens have KV.
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool, batch, self.target_worker,
                )
            )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        # Run target verify (EAGLE line 792-798)
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # Sample — EAGLE-style call (EAGLE line 822-826)
        predict, accept_length, accept_index = verify_input.sample(
            batch, logits_output
        )
        new_seq_lens = batch.seq_lens + accept_length

        # Record verify_done (EAGLE line 838-839)
        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        # Extract verified_id (EAGLE line 841-851)
        if not batch.forward_mode.is_idle():
            all_verified_id = predict[accept_index]
            verified_id = torch.empty_like(accept_length, dtype=torch.int32)
            fill_new_verified_id[(bs,)](
                all_verified_id,
                accept_length,
                verified_id,
                self.speculative_num_draft_tokens,
            )
        else:
            verified_id = torch.empty((0,), device=self.device, dtype=torch.int32)

        # Construct next_draft_input (EAGLE line 857-861)
        next_draft_input = SMCDraftInput(
            last_token_ids=verified_id,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            smc_logprob_diffs=verify_input.smc_logprob_diffs,
        )

    def _build_empty_decode_result(self, is_overlap_batch: bool) -> GenerationBatchResult:
        next_draft_input = (
            SMCDraftInput.create_idle_input(self.device) if is_overlap_batch else None
        )
        return GenerationBatchResult(
            logits_output=self._empty_logits_output() if is_overlap_batch else None,
            next_token_ids=torch.empty((0,), dtype=torch.int32, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            smc_logprob_diffs=torch.empty((0,), dtype=torch.float32, device=self.device),
            can_run_cuda_graph=False,
            next_draft_input=next_draft_input,
        )

    def _empty_logits_output(self) -> LogitsProcessorOutput:
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=None)

    def _build_prefill_overlap_input(
        self, batch: ModelWorkerBatch, result: GenerationBatchResult
    ) -> SMCDraftInput:
        assert result.next_token_ids is not None
        return SMCDraftInput(
            last_token_ids=result.next_token_ids,
            new_seq_lens=batch.seq_lens,
        )

    def materialize_smc_parent_draft_prefix(self, req: Req) -> None:
        committed_seq_len = compute_smc_shared_prefix_len(req)
        if committed_seq_len <= 0:
            return

        with self.draft_worker.draft_tp_context(
            self.draft_worker.draft_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self._run_draft_prefix_fill_batch([req], [committed_seq_len])

    def _run_draft_prefix_fill_batch(
        self,
        reqs: Sequence[Req],
        committed_seq_lens: Sequence[int],
    ) -> None:
        """Fill draft-model KV for the committed parent prefix before cloning."""
        if not reqs:
            return

        worker = self.draft_worker.draft_worker
        batch = ScheduleBatch.init_new(
            reqs=list(reqs),
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self._internal_tree_cache,
            model_config=worker.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.forward_mode = ForwardMode.EXTEND
        batch.return_logprob = False

        input_ids: List[int] = []
        out_cache_loc: List[torch.Tensor] = []
        seq_lens: List[int] = []
        prefix_lens: List[int] = []
        extend_lens: List[int] = []

        for idx, (req, committed_seq_len) in enumerate(
            zip(reqs, committed_seq_lens, strict=True)
        ):
            prompt_len = len(req.origin_input_ids)
            committed_output_len = committed_seq_len - prompt_len
            if committed_output_len < 0 or committed_output_len > len(req.output_ids):
                raise AssertionError(
                    "SMC draft prefix fill received inconsistent lengths: "
                    f"rid={req.rid}, prompt_len={prompt_len}, "
                    f"committed_seq_len={committed_seq_len}, "
                    f"output_len={len(req.output_ids)}"
                )

            fill_ids = req.origin_input_ids + req.output_ids[:committed_output_len]
            input_ids.extend(fill_ids)
            out_cache_loc.append(
                self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :committed_seq_len
                ].to(dtype=torch.int64, copy=True)
            )
            seq_lens.append(committed_seq_len)
            prefix_lens.append(0)
            extend_lens.append(committed_seq_len)

        batch.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        batch.req_pool_indices = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64, device=self.device
        )
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.seq_lens_sum = sum(seq_lens)
        batch.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        batch.out_cache_loc = torch.cat(out_cache_loc)
        batch.prefix_lens = prefix_lens
        batch.extend_lens = extend_lens
        batch.extend_num_tokens = sum(extend_lens)
        batch.extend_logprob_start_lens = [0] * len(reqs)
        batch.extend_input_logprob_token_ids = None
        batch.top_logprobs_nums = None
        batch.token_ids_logprobs = None
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            worker.model_config.vocab_size,
        )

        worker.forward_batch_generation(batch.get_model_worker_batch(), is_verify=True)

    def _run_eagle_style_draft_reqs(
        self,
        reqs: Sequence[Req],
        model_worker_batch: ModelWorkerBatch,
        last_token_ids: torch.Tensor,
        draft_sampling_info: SamplingBatchInfo,
    ):
        runner = self.draft_worker.smc_draft_cuda_graph_runner
        draft_input = SMCDraftInput(
            last_token_ids=last_token_ids,
            new_seq_lens=model_worker_batch.seq_lens,
        )
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            req_to_token_pool=self.req_to_token_pool,
            batch=model_worker_batch,
            cuda_graph_runner=runner,
            draft_model_runner=self.draft_worker.draft_runner,
            gamma=self.smc_gamma,
            draft_sampling_info=draft_sampling_info,
        )
        if can_cuda_graph:
            token_matrix, logprob_matrix = runner.replay(forward_batch)
        else:
            if not forward_batch.forward_mode.is_idle() and self.smc_gamma > 1:
                self.draft_worker.smc_draft_attn_backend.init_forward_metadata(
                    forward_batch
                )
            token_matrix, logprob_matrix = self.draft_worker.draft_forward(forward_batch)

        batch_size = len(reqs)
        draft_lengths = torch.full(
            (batch_size,), self.smc_gamma, dtype=torch.int32, device=self.device
        )
        draft_logprobs = logprob_matrix.sum(dim=1)

        append_smc_diag_record(
            {
                "type": "draft_result",
                "mode": "cuda_graph" if can_cuda_graph else "draft_forward",
                "group_ids": [getattr(req, "smc_group_id", None) for req in reqs],
                "prompt_prefixes": [
                    getattr(req, "origin_input_text", None)[:80]
                    if getattr(req, "origin_input_text", None) is not None
                    else None
                    for req in reqs
                ],
                "particle_indices": [
                    getattr(req, "smc_particle_idx", None) for req in reqs
                ],
                "committed_seq_lens": model_worker_batch.seq_lens.tolist(),
                "last_token_ids": last_token_ids.tolist(),
                "draft_tokens": token_matrix.tolist(),
                "draft_lengths": draft_lengths.tolist(),
                "draft_logprobs": [float(x) for x in draft_logprobs.tolist()],
                "can_run_cuda_graph": bool(can_cuda_graph),
            }
        )

        return (
            token_matrix,
            draft_logprobs,
            draft_lengths,
            can_cuda_graph,
        )
