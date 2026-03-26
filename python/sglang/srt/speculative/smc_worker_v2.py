from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
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
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.smc_draft_cuda_graph_runner import SMCDraftCudaGraphRunner
from sglang.srt.speculative.smc_info import (
    SMCDraftInput,
    SMC_MIN_TEMPERATURE,
    SMCScoreInput,
    build_smc_positions,
    get_smc_reserved_kv_len,
    set_smc_reserved_kv_len,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.standalone_worker_v2 import (
    StandaloneDraftWorker,
    _get_plan_stream,
)
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.utils import get_available_gpu_memory


logger = logging.getLogger(__name__)

_SMC_DIAG_PATH_ENV = "SGLANG_SMC_DIAG_PATH"


def _append_smc_diag_record(record: dict) -> None:
    record_path = os.environ.get(_SMC_DIAG_PATH_ENV)
    if not record_path:
        return
    payload = dict(record)
    payload["pid"] = os.getpid()
    payload["timestamp_ns"] = time.perf_counter_ns()
    with open(record_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(payload, sort_keys=True) + "\n")


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
        visible_seq_lens = model_worker_batch.seq_lens
        visible_seq_lens_cpu = (
            model_worker_batch.seq_lens_cpu
            if model_worker_batch.seq_lens_cpu is not None
            else model_worker_batch.seq_lens.cpu()
        )
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

        # With bonus + draft_extend, all tokens (including previous bonus)
        # have KV committed. No -1 adjustment needed — same as EAGLE.
        seq_lens = visible_seq_lens
        seq_lens_cpu = visible_seq_lens_cpu

        outer._ensure_draft_prefix_filled(reqs, seq_lens_cpu.tolist())

        if outer._can_use_fused_draft_cuda_graph(
            reqs, model_worker_batch.sampling_info
        ):
            draft_tokens, draft_logprobs, draft_lengths, _ = (
                outer._run_fused_draft_reqs(
                    reqs, model_worker_batch, last_token_ids,
                    model_worker_batch.sampling_info,
                    visible_seq_lens, seq_lens,
                )
            )
        else:
            draft_tokens, draft_logprobs, draft_lengths, _ = (
                outer._run_stepwise_draft_reqs(
                    reqs, visible_seq_lens, seq_lens, last_token_ids,
                )
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
        use_linear = outer.server_args.attention_backend in {"flashinfer", "triton"}
        custom_mask = None
        if not use_linear:
            from sglang.srt.speculative.smc_info import build_smc_causal_mask
            custom_mask = build_smc_causal_mask(seq_lens, score_token_num)

        return SMCScoreInput(
            draft_token=score_tokens.reshape(-1).contiguous(),
            draft_lengths=draft_lengths,
            draft_logprobs=draft_logprobs,
            positions=build_smc_positions(seq_lens, score_token_num),
            custom_mask=custom_mask,
            draft_token_num=score_token_num,
            spec_steps=smc_gamma,
            target_temperature=target_temperature,
            linear_target_verify=use_linear,
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

        with self.draft_worker.draft_tp_context(
            self.draft_worker.draft_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.draft_worker._draft_extend_for_decode(
                model_worker_batch, batch_output
            )

        return batch_output

    def verify(self, batch: ModelWorkerBatch) -> GenerationBatchResult:
        """SMC verify: accept ALL draft tokens, compute logprob diffs, add bonus.

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
        next_seq_lens = batch.seq_lens + 1
        next_seq_lens_cpu = (
            batch.seq_lens_cpu + 1
            if batch.seq_lens_cpu is not None
            else next_seq_lens.cpu()
        )
        return SMCDraftInput(
            last_token_ids=result.next_token_ids,
            new_seq_lens=next_seq_lens,
            committed_seq_lens_cpu=next_seq_lens_cpu,
        )

    def _ensure_draft_prefix_filled(
        self,
        reqs: Sequence[Req],
        draft_committed_lens: Sequence[int],
    ) -> None:
        fill_reqs: List[Req] = []
        fill_lens: List[int] = []
        for req, committed_seq_len in zip(reqs, draft_committed_lens, strict=True):
            if req.draft_prefix_materialized or committed_seq_len <= 0:
                req.draft_prefix_materialized = True
                continue
            fill_reqs.append(req)
            fill_lens.append(int(committed_seq_len))

        if not fill_reqs:
            return

        with self.draft_worker.draft_tp_context(
            self.draft_worker.draft_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self._run_draft_prefix_fill_batch(
                fill_reqs,
                fill_lens,
                worker=self.draft_worker.draft_worker,
            )

        for req in fill_reqs:
            req.draft_prefix_materialized = True

    def _run_draft_prefix_fill_batch(
        self,
        reqs: Sequence[Req],
        committed_seq_lens: Sequence[int],
        worker,
        existing_prefix_lens: Optional[Sequence[int]] = None,
        explicit_fill_ids: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        """Fill draft model KV cache for committed positions.

        When ``existing_prefix_lens`` is provided, the draft model already has
        KV at ``[0, existing_prefix_len)`` and only the range
        ``[existing_prefix_len, committed_seq_len)`` is filled (incremental).
        Otherwise the entire ``[0, committed_seq_len)`` is filled from scratch.

        When ``explicit_fill_ids`` is provided, token IDs for the fill range
        are taken from it instead of ``req.origin_input_ids + req.output_ids``.
        This is needed when the fill runs before the scheduler commits tokens
        to ``output_ids`` (e.g. incremental draft extend after verify).
        """
        if not reqs:
            return

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
            plen = (
                existing_prefix_lens[idx]
                if existing_prefix_lens is not None
                else 0
            )
            elen = committed_seq_len - plen

            if explicit_fill_ids is not None:
                fill_ids = list(explicit_fill_ids[idx])
            else:
                prompt_len = len(req.origin_input_ids)
                committed_output_len = committed_seq_len - prompt_len
                if committed_output_len < 0 or committed_output_len > len(
                    req.output_ids
                ):
                    raise AssertionError(
                        "SMC draft prefix fill received inconsistent lengths: "
                        f"rid={req.rid}, prompt_len={prompt_len}, "
                        f"committed_seq_len={committed_seq_len}, "
                        f"output_len={len(req.output_ids)}"
                    )
                all_ids = req.origin_input_ids + req.output_ids[:committed_output_len]
                fill_ids = all_ids[plen:]

            input_ids.extend(fill_ids)
            out_cache_loc.append(
                self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, plen:committed_seq_len
                ].to(dtype=torch.int64, copy=True)
            )
            seq_lens.append(committed_seq_len)
            prefix_lens.append(plen)
            extend_lens.append(elen)

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

    def _can_use_fused_draft_cuda_graph(
        self,
        reqs: Sequence[Req],
        sampling_info: SamplingBatchInfo,
    ) -> bool:
        runner = getattr(self.draft_worker, "smc_draft_cuda_graph_runner", None)
        # Fast gate: supports_replay checks fundamental incompatibilities.
        # The actual can_run check happens inside prepare_for_v2_draft.
        return bool(runner and runner.supports_replay(reqs, sampling_info))

    def _run_fused_draft_reqs(
        self,
        reqs: Sequence[Req],
        model_worker_batch: ModelWorkerBatch,
        last_token_ids: torch.Tensor,
        draft_sampling_info: SamplingBatchInfo,
        visible_seq_lens: torch.Tensor,
        draft_committed_lens: torch.Tensor,
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
        if not can_cuda_graph:
            # Large or mixed batches can still fail the exact replay check even after
            # supports_replay() passes. Fall back to the stepwise draft path instead
            # of tearing down the server.
            return self._run_stepwise_draft_reqs(
                reqs,
                visible_seq_lens,
                draft_committed_lens,
                last_token_ids,
            )
        token_matrix, logprob_matrix = runner.replay(forward_batch)
        for req in reqs:
            # Fused replay keeps the accepted draft prefix resident in KV, so
            # the next draft iteration can continue without a prefix refill.
            req.draft_prefix_materialized = True
        # SMC accepts all draft tokens unconditionally. EOS/max_new_tokens
        # are handled by the scheduler's check_finished after commit.
        batch_size = len(reqs)
        draft_lengths = torch.full(
            (batch_size,), self.smc_gamma, dtype=torch.int32, device=self.device
        )
        draft_logprobs = logprob_matrix.sum(dim=1)
        return token_matrix, draft_logprobs, draft_lengths, True

    def _run_stepwise_draft_reqs(
        self,
        reqs: Sequence[Req],
        visible_seq_lens: torch.Tensor,
        draft_committed_lens: torch.Tensor,
        last_token_ids: torch.Tensor,
    ):
        batch_size = len(reqs)
        draft_tokens = torch.zeros(
            (batch_size, self.smc_gamma), dtype=torch.int32, device=self.device
        )
        draft_logprobs = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)
        draft_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)

        snapshots = []
        seed_token_ids = [
            int(token_id) for token_id in last_token_ids.detach().cpu().tolist()
        ]
        for req, draft_committed_len, last_token_id in zip(
            reqs,
            draft_committed_lens.tolist(),
            seed_token_ids,
            strict=True,
        ):
            snapshot_allocated_len = get_smc_reserved_kv_len(req)
            if snapshot_allocated_len > 0:
                snapshot_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :snapshot_allocated_len
                ].to(dtype=torch.int64, copy=True)
            else:
                snapshot_indices = torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                )
            snapshots.append(
                {
                    "indices": snapshot_indices,
                    "output_ids": list(req.output_ids),
                    "kv_committed_len": req.kv_committed_len,
                    "kv_allocated_len": snapshot_allocated_len,
                    "finished_reason": copy.copy(req.finished_reason),
                    "finished_len": req.finished_len,
                    "finished_output": req.finished_output,
                    "to_finish": copy.copy(req.to_finish),
                    "decode_batch_idx": req.decode_batch_idx,
                }
            )
            req.output_ids = [int(last_token_id)]
            req.kv_committed_len = int(draft_committed_len)
            req.finished_reason = None
            req.finished_len = None
            req.finished_output = None
            req.to_finish = None

        can_run_cuda_graph = False
        for step in range(self.smc_gamma):
            step_reqs = list(reqs)
            decode_result = self._run_decode_batch(
                step_reqs,
                worker=self.draft_worker.draft_worker,
            )
            step_token_ids, step_token_logprobs = self._fill_draft_step_outputs(
                decode_result
            )
            can_run_cuda_graph = can_run_cuda_graph or decode_result.can_run_cuda_graph

            for idx in range(batch_size):
                token_id = int(step_token_ids[idx])
                token_logprob = float(step_token_logprobs[idx])
                draft_tokens[idx, step] = token_id
                draft_logprobs[idx] += token_logprob
                draft_lengths[idx] += 1
                reqs[idx].output_ids.append(token_id)

        for req, snapshot in zip(reqs, snapshots, strict=True):
            snapshot_indices = snapshot["indices"]
            snapshot_allocated_len = snapshot["kv_allocated_len"]
            current_allocated_len = req.kv_allocated_len
            indices_to_free = []
            if current_allocated_len > 0:
                current_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :current_allocated_len
                ].to(dtype=torch.int64, copy=True)
            else:
                current_indices = torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                )

            if snapshot_allocated_len > 0:
                current_prefix = current_indices[:snapshot_allocated_len]
                changed_mask = current_prefix != snapshot_indices
                if bool(changed_mask.any().item()):
                    changed_indices = current_prefix[changed_mask]
                    changed_indices = changed_indices[changed_indices != 0]
                    if changed_indices.numel() > 0:
                        indices_to_free.append(changed_indices)
                    self.req_to_token_pool.write(
                        (req.req_pool_idx, slice(0, snapshot_allocated_len)),
                        snapshot_indices.to(dtype=torch.int32),
                    )

            if current_allocated_len > snapshot_allocated_len:
                tail_indices = current_indices[snapshot_allocated_len:current_allocated_len]
                tail_indices = tail_indices[tail_indices != 0]
                if tail_indices.numel() > 0:
                    indices_to_free.append(tail_indices)

            if indices_to_free:
                self.token_to_kv_pool_allocator.dec_ref_and_free(
                    torch.cat(indices_to_free)
                )
            req.output_ids = snapshot["output_ids"]
            req.kv_committed_len = snapshot["kv_committed_len"]
            req.kv_allocated_len = snapshot_allocated_len
            set_smc_reserved_kv_len(req, snapshot_allocated_len)
            req.finished_reason = snapshot["finished_reason"]
            req.finished_len = snapshot["finished_len"]
            req.finished_output = snapshot["finished_output"]
            req.to_finish = snapshot["to_finish"]
            req.decode_batch_idx = snapshot["decode_batch_idx"]
            # Stepwise draft restores the pre-proposal KV snapshot, so the
            # newly committed prefix is no longer materialized for draft decode.
            req.draft_prefix_materialized = False

        _append_smc_diag_record(
            {
                "type": "draft_result",
                "mode": "stepwise",
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
                "visible_seq_lens": visible_seq_lens.tolist(),
                "last_token_ids": seed_token_ids,
                "draft_tokens": draft_tokens.tolist(),
                "draft_lengths": draft_lengths.tolist(),
                "draft_logprobs": [float(x) for x in draft_logprobs.tolist()],
                "can_run_cuda_graph": bool(can_run_cuda_graph),
            }
        )

        return (
            draft_tokens,
            draft_logprobs,
            draft_lengths,
            can_run_cuda_graph,
        )

    def _run_decode_batch(self, reqs: List[Req], worker) -> GenerationBatchResult:
        batch = self._make_decode_batch(reqs, worker.model_config)
        return worker.forward_batch_generation(batch.get_model_worker_batch())

    def _fill_draft_step_outputs(
        self,
        result: GenerationBatchResult,
    ) -> tuple[List[int], List[float]]:
        assert result.next_token_ids is not None
        assert result.logits_output is not None
        assert result.logits_output.next_token_logprobs is not None
        return (
            [int(token_id) for token_id in result.next_token_ids.tolist()],
            [float(x) for x in result.logits_output.next_token_logprobs.tolist()],
        )

    def _make_decode_batch(self, reqs: List[Req], model_config) -> ScheduleBatch:
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self._internal_tree_cache,
            model_config=model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.req_pool_indices = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64, device=self.device
        )
        batch.seq_lens = torch.tensor(
            [req.kv_committed_len for req in reqs],
            dtype=torch.int64,
            device=self.device,
        )
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
        batch.orig_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        batch.output_ids = torch.tensor(
            [req.output_ids[-1] for req in reqs],
            dtype=torch.int64,
            device=self.device,
        )
        batch.top_logprobs_nums = [0] * len(reqs)
        batch.token_ids_logprobs = [None] * len(reqs)
        batch.return_logprob = True
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            model_config.vocab_size,
        )
        decode_locs = batch.seq_lens.clone()
        reserved_out_cache_loc = self.req_to_token_pool.req_to_token[
            batch.req_pool_indices, decode_locs
        ].to(dtype=torch.int64, copy=True)
        batch.prepare_for_decode()
        new_out_cache_loc = batch.out_cache_loc
        reuse_reserved_mask = reserved_out_cache_loc != 0
        if bool(reuse_reserved_mask.any().item()):
            reused_req_pool_indices = batch.req_pool_indices[reuse_reserved_mask]
            reused_decode_locs = decode_locs[reuse_reserved_mask]
            reused_reserved_out_cache_loc = reserved_out_cache_loc[reuse_reserved_mask]
            self.req_to_token_pool.write(
                (reused_req_pool_indices, reused_decode_locs),
                reused_reserved_out_cache_loc.to(dtype=torch.int32),
            )
            self.token_to_kv_pool_allocator.dec_ref_and_free(
                new_out_cache_loc[reuse_reserved_mask].to(dtype=torch.int64, copy=True)
            )
        batch.out_cache_loc = torch.where(
            reuse_reserved_mask,
            reserved_out_cache_loc,
            new_out_cache_loc,
        )

        for req in reqs:
            req.kv_allocated_len = max(
                get_smc_reserved_kv_len(req),
                req.kv_committed_len,
            )

        return batch
