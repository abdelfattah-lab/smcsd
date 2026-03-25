from __future__ import annotations

import copy
import dataclasses
import logging
import time
from typing import List, Sequence, Union

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
    resolve_smc_proposal_length,
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
        self._internal_tree_cache = _SMCInternalTreeCache(
            self.token_to_kv_pool_allocator
        )

    @property
    def model_runner(self):
        return self.target_worker.model_runner

    @property
    def model_config(self):
        return self.target_worker.model_config

    def forward_batch_generation(
        self, batch: Union[ScheduleBatch, ModelWorkerBatch]
    ) -> GenerationBatchResult:
        is_overlap_batch = isinstance(batch, ModelWorkerBatch)
        draft_input = (
            batch.spec_info if isinstance(batch.spec_info, SMCDraftInput) else None
        )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch if is_overlap_batch else batch.get_model_worker_batch()
            result = self.target_worker.forward_batch_generation(model_worker_batch)
            if is_overlap_batch:
                result.next_draft_input = self._build_prefill_overlap_input(
                    model_worker_batch, result
                )
            return result

        if not batch.reqs:
            return self._build_empty_decode_result(is_overlap_batch)

        reqs = list(batch.reqs)
        visible_seq_lens = batch.seq_lens
        visible_seq_lens_cpu = (
            batch.seq_lens_cpu if batch.seq_lens_cpu is not None else batch.seq_lens.cpu()
        )
        if draft_input is None:
            last_token_ids = torch.tensor(
                [
                    req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
                    for req in reqs
                ],
                dtype=torch.int32,
                device=self.device,
            )
        else:
            last_token_ids = draft_input.last_token_ids
        draft_committed_lens = visible_seq_lens - 1
        draft_committed_lens_cpu = visible_seq_lens_cpu - 1

        self._ensure_draft_prefix_filled(reqs, draft_committed_lens_cpu.tolist())

        model_worker_batch = batch if is_overlap_batch else batch.get_model_worker_batch()
        with self.draft_worker.draft_tp_context(
            self.draft_worker.draft_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            if self._can_use_fused_draft_cuda_graph(reqs, batch.sampling_info):
                (
                    draft_tokens,
                    draft_logprobs,
                    draft_lengths,
                    draft_can_run_cuda_graph,
                ) = self._run_fused_draft_reqs(
                    reqs,
                    model_worker_batch,
                    last_token_ids,
                    model_worker_batch.sampling_info,
                    visible_seq_lens,
                    draft_committed_lens,
                )
            else:
                (
                    draft_tokens,
                    draft_logprobs,
                    draft_lengths,
                    draft_can_run_cuda_graph,
                ) = self._run_stepwise_draft_reqs(
                    reqs,
                    visible_seq_lens,
                    draft_committed_lens,
                    last_token_ids,
                )

        (
            accept_lens,
            committed_seq_lens,
            next_last_token_ids,
            smc_logprob_diffs,
            score_can_run_cuda_graph,
        ) = self._run_score_batch(
            base_model_worker_batch=(
                batch if is_overlap_batch else batch.get_model_worker_batch()
            ),
            draft_committed_lens=draft_committed_lens,
            anchor_token_ids=last_token_ids,
            draft_tokens=draft_tokens,
            draft_logprobs=draft_logprobs,
            draft_lengths=draft_lengths,
        )

        stride = self.server_args.speculative_num_draft_tokens
        flat_predict = torch.zeros(
            (len(reqs), stride),
            dtype=torch.int32,
            device=self.device,
        )
        flat_predict[:, : draft_tokens.shape[1]] = draft_tokens

        verify_done = None
        if is_overlap_batch:
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()

        next_draft_input = SMCDraftInput(
            last_token_ids=next_last_token_ids,
            new_seq_lens=committed_seq_lens,
            verify_done=verify_done,
        )
        return GenerationBatchResult(
            logits_output=self._empty_logits_output() if is_overlap_batch else None,
            next_token_ids=flat_predict.reshape(-1),
            accept_lens=accept_lens,
            smc_logprob_diffs=smc_logprob_diffs,
            can_run_cuda_graph=draft_can_run_cuda_graph or score_can_run_cuda_graph,
            next_draft_input=next_draft_input,
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
            new_seq_lens=batch.seq_lens + 1,
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
    ) -> None:
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

        for req, committed_seq_len in zip(reqs, committed_seq_lens, strict=True):
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
        batch.extend_num_tokens = batch.seq_lens_sum
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
        bs = token_matrix.shape[0]
        draft_lengths = torch.full(
            (bs,), self.smc_gamma, dtype=torch.int32, device=self.device
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
        draft_finished = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        current_output_lens = [
            int(visible_seq_len) - len(req.origin_input_ids)
            for req, visible_seq_len in zip(reqs, visible_seq_lens.tolist(), strict=True)
        ]

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
            step_indices = [
                idx for idx in range(batch_size) if not bool(draft_finished[idx].item())
            ]
            if not step_indices:
                break

            step_reqs = [reqs[idx] for idx in step_indices]
            decode_result = self._run_decode_batch(
                step_reqs,
                worker=self.draft_worker.draft_worker,
            )
            step_token_ids, step_token_logprobs = self._fill_draft_step_outputs(
                decode_result
            )
            can_run_cuda_graph = can_run_cuda_graph or decode_result.can_run_cuda_graph

            for local_idx, row_idx in enumerate(step_indices):
                token_id = int(step_token_ids[local_idx])
                token_logprob = float(step_token_logprobs[local_idx])
                current_output_len = current_output_lens[row_idx]
                committed_steps, proposal_finished = resolve_smc_proposal_length(
                    reqs[row_idx],
                    [token_id],
                    current_output_len=current_output_len,
                )
                if committed_steps > 0:
                    draft_tokens[row_idx, step] = token_id
                    draft_logprobs[row_idx] += token_logprob
                    draft_lengths[row_idx] += 1
                    current_output_lens[row_idx] += 1
                    reqs[row_idx].output_ids.append(token_id)
                draft_finished[row_idx] = proposal_finished

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
                    indices_to_free.append(current_prefix[changed_mask])
                    self.req_to_token_pool.write(
                        (req.req_pool_idx, slice(0, snapshot_allocated_len)),
                        snapshot_indices.to(dtype=torch.int32),
                    )

            if current_allocated_len > snapshot_allocated_len:
                indices_to_free.append(
                    current_indices[snapshot_allocated_len:current_allocated_len]
                )

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

        return (
            draft_tokens,
            draft_logprobs,
            draft_lengths,
            can_run_cuda_graph,
        )

    def _run_score_batch(
        self,
        base_model_worker_batch: ModelWorkerBatch,
        draft_committed_lens: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logprobs: torch.Tensor,
        draft_lengths: torch.Tensor,
    ):
        model_worker_batch = self._make_score_model_worker_batch(
            base_model_worker_batch=base_model_worker_batch,
            draft_committed_lens=draft_committed_lens,
            anchor_token_ids=anchor_token_ids,
            draft_tokens=draft_tokens,
            draft_logprobs=draft_logprobs,
            draft_lengths=draft_lengths,
        )
        score_input: SMCScoreInput = model_worker_batch.spec_info
        verify_forward_batch, can_run_cuda_graph = score_input.prepare_for_v2_verify(
            self.req_to_token_pool,
            model_worker_batch,
            self.target_worker,
        )
        forward_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        assert forward_output.logits_output is not None
        # CG path bakes log_softmax(logits/T) into the graph.
        # Non-CG fallback needs it applied here so sample() gets log_probs.
        if not can_run_cuda_graph:
            logits = forward_output.logits_output.next_token_logits
            logits = logits.view(
                score_input.draft_lengths.shape[0],
                score_input.draft_token_num,
                -1,
            )
            forward_output.logits_output.next_token_logits = (
                torch.nn.functional.log_softmax(
                    logits / score_input.target_temperature, dim=-1
                )
                .view_as(forward_output.logits_output.next_token_logits)
            )
        (
            accept_lens,
            committed_seq_lens,
            next_last_token_ids,
            smc_logprob_diffs,
        ) = score_input.sample(
            model_worker_batch,
            forward_output.logits_output,
        )
        return (
            accept_lens,
            committed_seq_lens,
            next_last_token_ids,
            smc_logprob_diffs,
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

    def _make_score_model_worker_batch(
        self,
        base_model_worker_batch: ModelWorkerBatch,
        draft_committed_lens: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logprobs: torch.Tensor,
        draft_lengths: torch.Tensor,
    ) -> ModelWorkerBatch:
        score_token_num = self.server_args.speculative_num_draft_tokens
        score_tokens = torch.cat(
            [
                anchor_token_ids.unsqueeze(1),
                draft_tokens,
            ],
            dim=1,
        )
        score_tokens = score_tokens[:, :score_token_num]
        if score_tokens.shape[1] < score_token_num:
            pad = score_tokens[:, -1:].expand(-1, score_token_num - score_tokens.shape[1])
            score_tokens = torch.cat([score_tokens, pad], dim=1)

        seq_lens = draft_committed_lens
        batch_size = int(seq_lens.shape[0])
        if base_model_worker_batch.seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.cpu()
            seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            seq_lens_cpu = base_model_worker_batch.seq_lens_cpu - 1
            seq_lens_sum = base_model_worker_batch.seq_lens_sum - batch_size
        use_linear_target_verify = self.server_args.attention_backend in {
            "flashinfer",
            "triton",
        }
        custom_mask = None
        if not use_linear_target_verify:
            from sglang.srt.speculative.smc_info import build_smc_causal_mask

            custom_mask = build_smc_causal_mask(seq_lens, score_token_num)
        score_input = SMCScoreInput(
            draft_token=score_tokens.reshape(-1).contiguous(),
            draft_lengths=draft_lengths,
            draft_logprobs=draft_logprobs,
            positions=build_smc_positions(seq_lens, score_token_num),
            custom_mask=custom_mask,
            draft_token_num=score_token_num,
            # (ccc) SMC target verify intentionally uses one global target
            # temperature. Do not reintroduce per-request verify temperatures
            # without updating both the eager and CUDA-graph paths together.
            target_temperature=max(
                float(self.server_args.smc_target_temperature), SMC_MIN_TEMPERATURE
            ),
            linear_target_verify=use_linear_target_verify,
        )
        return dataclasses.replace(
            base_model_worker_batch,
            forward_mode=ForwardMode.DECODE,
            input_ids=None,
            seq_lens=seq_lens,
            out_cache_loc=None,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=seq_lens_sum,
            spec_algorithm=SpeculativeAlgorithm.SMC,
            spec_info=score_input,
            capture_hidden_mode=score_input.capture_hidden_mode,
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
        batch.orig_seq_lens = torch.tensor(
            [len(req.origin_input_ids) for req in reqs],
            dtype=torch.int32,
            device=self.device,
        )
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
        self.req_to_token_pool.write(
            (batch.req_pool_indices, decode_locs),
            reserved_out_cache_loc.to(dtype=torch.int32),
        )
        self.token_to_kv_pool_allocator.dec_ref_and_free(
            new_out_cache_loc.to(dtype=torch.int64, copy=True)
        )
        batch.out_cache_loc = reserved_out_cache_loc

        for req in reqs:
            req.kv_allocated_len = max(
                get_smc_reserved_kv_len(req),
                req.kv_committed_len,
            )

        return batch
