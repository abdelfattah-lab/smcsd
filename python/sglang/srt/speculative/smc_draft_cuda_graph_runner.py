from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.layers.sampler import TOP_K_ALL
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)


@dataclass(frozen=True)
class SMCDraftSamplingSignature:
    is_all_greedy: bool
    need_top_p_sampling: bool
    need_top_k_sampling: bool
    need_min_p_sampling: bool


@dataclass
class SMCDraftInputBuffers(ForwardInputBuffers):
    input_ids: torch.Tensor
    working_input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens_steps: torch.Tensor
    seq_lens_cpu_steps: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor
    working_positions: torch.Tensor
    mrope_positions: torch.Tensor
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor
    sampling_seed: torch.Tensor
    sampled_token_ids: torch.Tensor
    sampled_token_logprobs: torch.Tensor
    global_num_tokens_gpu: Optional[torch.Tensor]
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]


class SMCDraftCudaGraphRunner:
    def __init__(self, draft_worker):
        self.draft_worker = draft_worker
        self.model_runner = model_runner = draft_worker.draft_runner
        self.gamma = draft_worker.server_args.smc_gamma
        self.device = model_runner.device
        self.graphs = {}
        self.output_buffers = {}
        self.enable_profile_cuda_graph = False
        self.enable_pdmux = False
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = model_runner.tp_size
        self.dp_size = model_runner.dp_size
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.num_tokens_per_bs = 1
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs
        self.sampling_signature = SMCDraftSamplingSignature(
            is_all_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
        )
        self.enable_deterministic = bool(model_runner.sampler.enable_deterministic)
        self.use_log_softmax_logprob = bool(model_runner.sampler.use_log_softmax_logprob)
        self.use_ascend_backend = bool(model_runner.sampler.use_ascend_backend)
        self.seq_len_fill_value = model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        self.multi_step_attn_backend = draft_worker.smc_draft_attn_backend
        self.step_attn_backends = draft_worker.smc_draft_attn_backend.attn_backends

        draft_worker.smc_draft_attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )

        with torch.device(model_runner.device):
            input_ids = torch.zeros((self.max_bs,), dtype=torch.int64)
            working_input_ids = torch.zeros((self.max_bs,), dtype=torch.int64)
            req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            seq_lens_steps = torch.full(
                (self.gamma, self.max_bs),
                self.seq_len_fill_value,
                dtype=torch.int32,
            )
            seq_lens_cpu_steps = torch.full(
                (self.gamma, self.max_bs),
                self.seq_len_fill_value,
                dtype=torch.int32,
                device="cpu",
            )
            out_cache_loc = torch.zeros(
                (self.gamma, self.max_bs), dtype=torch.int64, device=self.device
            )
            positions = torch.zeros((self.max_bs,), dtype=torch.int64)
            working_positions = torch.zeros((self.max_bs,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)
            temperatures = torch.ones((self.max_bs, 1), dtype=torch.float32)
            top_ps = torch.ones((self.max_bs,), dtype=torch.float32)
            top_ks = torch.full((self.max_bs,), TOP_K_ALL, dtype=torch.int32)
            min_ps = torch.zeros((self.max_bs,), dtype=torch.float32)
            sampling_seed = torch.zeros((self.max_bs,), dtype=torch.int64)
            sampled_token_ids = torch.zeros(
                (self.gamma, self.max_bs), dtype=torch.int32
            )
            sampled_token_logprobs = torch.zeros(
                (self.gamma, self.max_bs), dtype=torch.float32
            )

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

        self.buffers = SMCDraftInputBuffers(
            input_ids=input_ids,
            working_input_ids=working_input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens_steps=seq_lens_steps,
            seq_lens_cpu_steps=seq_lens_cpu_steps,
            out_cache_loc=out_cache_loc,
            positions=positions,
            working_positions=working_positions,
            mrope_positions=mrope_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            sampling_seed=sampling_seed,
            sampled_token_ids=sampled_token_ids,
            sampled_token_logprobs=sampled_token_logprobs,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        )
        self.buffers.share_buffers()

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def capture(self):
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(
        self, num_seqs: int, forward: Callable, stream_idx: int = 0
    ):
        del forward, stream_idx

        buffers = self.buffers
        graph = self._create_graph()
        stream = self.stream
        num_tokens = num_seqs * self.num_tokens_per_bs

        req_pool_indices = buffers.req_pool_indices[:num_seqs]
        positions = buffers.positions[:num_seqs]
        working_positions = buffers.working_positions[:num_seqs]
        input_ids = buffers.input_ids[:num_seqs]
        working_input_ids = buffers.working_input_ids[:num_seqs]
        seq_lens_steps = buffers.seq_lens_steps[:, :num_seqs]
        seq_lens_cpu_steps = buffers.seq_lens_cpu_steps[:, :num_seqs]
        out_cache_loc = buffers.out_cache_loc[:, :num_seqs]
        sampled_token_ids = buffers.sampled_token_ids[:, :num_seqs]
        sampled_token_logprobs = buffers.sampled_token_logprobs[:, :num_seqs]

        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor([num_tokens], dtype=torch.int32, device=input_ids.device)
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor([num_tokens], dtype=torch.int32, device=input_ids.device)
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        sampling_info = SamplingBatchInfo(
            temperatures=buffers.temperatures[:num_seqs],
            top_ps=buffers.top_ps[:num_seqs],
            top_ks=buffers.top_ks[:num_seqs],
            min_ps=buffers.min_ps[:num_seqs],
            sampling_seed=(
                buffers.sampling_seed[:num_seqs] if self.enable_deterministic else None
            ),
            is_all_greedy=self.sampling_signature.is_all_greedy,
            need_top_p_sampling=self.sampling_signature.need_top_p_sampling,
            need_top_k_sampling=self.sampling_signature.need_top_k_sampling,
            need_min_p_sampling=self.sampling_signature.need_min_p_sampling,
            vocab_size=self.model_runner.model_config.vocab_size,
            penalizer_orchestrator=None,
            has_custom_logit_processor=False,
            custom_params=None,
            custom_logit_processor=None,
            device=self.device,
            logit_bias=None,
        )

        for step, attn_backend in enumerate(self.step_attn_backends):
            seq_lens_step = seq_lens_steps[step]
            attn_backend.init_forward_metadata_capture_cuda_graph(
                num_seqs,
                num_tokens,
                req_pool_indices,
                seq_lens_step,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=None,
            )

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=num_seqs,
            input_ids=working_input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens_steps[0],
            seq_lens_cpu=seq_lens_cpu_steps[0],
            out_cache_loc=out_cache_loc[0],
            seq_lens_sum=int(seq_lens_cpu_steps[0].sum().item()),
            return_logprob=True,
            top_logprobs_nums=[0] * num_seqs,
            token_ids_logprobs=[None] * num_seqs,
            positions=working_positions,
            sampling_info=sampling_info,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            attn_backend=self.step_attn_backends[0],
            mrope_positions=None,
        )

        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            working_input_ids.copy_(input_ids)
            working_positions.copy_(positions)

            for step, attn_backend in enumerate(self.step_attn_backends):
                forward_batch.input_ids = working_input_ids
                forward_batch.seq_lens = seq_lens_steps[step]
                forward_batch.seq_lens_cpu = seq_lens_cpu_steps[step]
                forward_batch.seq_lens_sum = int(seq_lens_cpu_steps[step].sum().item())
                forward_batch.out_cache_loc = out_cache_loc[step]
                forward_batch.positions = working_positions
                forward_batch.attn_backend = attn_backend

                logits_output = self.model_runner.forward_decode(
                    forward_batch, skip_attn_backend_init=True
                )
                next_token_ids = self.model_runner.sample(logits_output, forward_batch)
                sampled_token_ids[step].copy_(next_token_ids)
                sampled_token_logprobs[step].copy_(logits_output.next_token_logprobs)
                working_input_ids.copy_(next_token_ids)
                working_positions.add_(1)

            return sampled_token_ids, sampled_token_logprobs

        self.deepep_adapter.capture(is_extend_in_batch=False)
        self._capture_init(run_once)
        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )
        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def _supports_sampling_info(self, sampling_info: SamplingBatchInfo) -> bool:
        penalizer = sampling_info.penalizer_orchestrator
        return (
            sampling_info.grammars is None
            and not sampling_info.has_custom_logit_processor
            and sampling_info.logit_bias is None
            and (penalizer is None or not penalizer.is_required)
        )

    def supports_replay(
        self,
        reqs: Sequence[Req],
        sampling_info: SamplingBatchInfo,
    ) -> bool:
        if not reqs:
            return False
        if self.gamma <= 1:
            return False
        if self.model_runner.server_args.enable_lora:
            return False
        if self.model_runner.is_hybrid_swa:
            return False
        if self.model_runner.use_ngram_embedding:
            return False
        if self.model_runner.model_is_mrope:
            return False
        if self.model_runner.pp_size > 1:
            return False
        if self.model_runner.token_to_kv_pool_allocator.page_size != 1:
            return False
        return self._supports_sampling_info(sampling_info)

    def _sampling_signature(
        self, sampling_info: SamplingBatchInfo
    ) -> SMCDraftSamplingSignature:
        return SMCDraftSamplingSignature(
            is_all_greedy=sampling_info.is_all_greedy,
            need_top_p_sampling=sampling_info.need_top_p_sampling,
            need_top_k_sampling=sampling_info.need_top_k_sampling,
            need_min_p_sampling=sampling_info.need_min_p_sampling,
        )

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        raw_bs = forward_batch.batch_size
        sampling_info = forward_batch.sampling_info
        signature = self._sampling_signature(sampling_info)
        if signature != self.sampling_signature:
            self.sampling_signature = signature
            self.capture()

        if self.require_mlp_sync:
            return False

        return (
            raw_bs in self.graphs if self.disable_padding else raw_bs <= self.max_bs
        )

    def replay(self, forward_batch: ForwardBatch) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang.srt.speculative.smc_info import SMCDraftInput

        raw_bs = forward_batch.batch_size
        buffers = self.buffers
        spec_info: SMCDraftInput = forward_batch.spec_info

        # Determine padded batch size
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]

        # Pad unused slots if needed
        if bs != raw_bs:
            buffers.input_ids.zero_()
            buffers.req_pool_indices.zero_()
            buffers.positions.zero_()
            buffers.out_cache_loc.zero_()
            buffers.seq_lens_steps.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu_steps.fill_(self.seq_len_fill_value)
            buffers.temperatures.fill_(1.0)
            buffers.top_ps.fill_(1.0)
            buffers.top_ks.fill_(TOP_K_ALL)
            buffers.min_ps.zero_()
            buffers.sampling_seed.zero_()

        # Copy data to CG buffers.
        # .copy_() implicitly handles dtype conversion (e.g. int64 -> int32).
        buffers.input_ids[:raw_bs].copy_(forward_batch.input_ids[:raw_bs])
        buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices[:raw_bs])
        buffers.positions[:raw_bs].copy_(forward_batch.positions[:raw_bs])

        # Multi-step data from spec_info (populated by prepare_for_v2_draft)
        buffers.out_cache_loc[:, :raw_bs].copy_(spec_info.out_cache_loc_steps)
        buffers.seq_lens_steps[:, :raw_bs].copy_(spec_info.seq_lens_steps)
        buffers.seq_lens_cpu_steps[:, :raw_bs].copy_(spec_info.seq_lens_cpu_steps)

        # Sampling parameters
        sampling_info = forward_batch.sampling_info
        buffers.temperatures[:raw_bs].copy_(sampling_info.temperatures)
        buffers.top_ps[:raw_bs].copy_(sampling_info.top_ps)
        buffers.top_ks[:raw_bs].copy_(sampling_info.top_ks)
        buffers.min_ps[:raw_bs].copy_(sampling_info.min_ps)
        if self.enable_deterministic and sampling_info.sampling_seed is not None:
            buffers.sampling_seed[:raw_bs].copy_(sampling_info.sampling_seed)

        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(bs)
            buffers.global_num_tokens_for_logprob_gpu.fill_(bs)

        # Initialize attention metadata for multi-step replay
        self._init_replay_metadata(
            bs=bs,
            raw_bs=raw_bs,
            req_pool_indices=buffers.req_pool_indices[:bs],
            seq_lens_steps=buffers.seq_lens_steps[:, :bs],
            seq_lens_cpu_steps=buffers.seq_lens_cpu_steps[:, :bs],
            seq_lens_sum_steps=spec_info.seq_lens_sum_steps,
        )

        # Run the captured CUDA graph
        self.deepep_adapter.replay()
        self.raw_bs = raw_bs
        self.bs = bs
        self.graphs[bs].replay()
        sampled_token_ids, sampled_token_logprobs = self.output_buffers[bs]
        return (
            sampled_token_ids[:, :raw_bs].transpose(0, 1).contiguous(),
            sampled_token_logprobs[:, :raw_bs].transpose(0, 1).contiguous(),
        )

    def _init_replay_metadata(
        self,
        *,
        bs: int,
        raw_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens_steps: torch.Tensor,
        seq_lens_cpu_steps: torch.Tensor,
        seq_lens_sum_steps: torch.Tensor,
    ) -> None:
        if bs != raw_bs:
            padded_step_sums = seq_lens_sum_steps + (bs - raw_bs) * self.seq_len_fill_value
        else:
            padded_step_sums = seq_lens_sum_steps
        if (
            self.multi_step_attn_backend is not None
            and hasattr(
                self.multi_step_attn_backend,
                "init_smc_forward_metadata_replay_cuda_graph",
            )
        ):
            self.multi_step_attn_backend.init_smc_forward_metadata_replay_cuda_graph(
                bs=bs,
                raw_bs=raw_bs,
                req_pool_indices=req_pool_indices,
                seq_lens_steps=seq_lens_steps,
                seq_lens_cpu_steps=seq_lens_cpu_steps,
                seq_lens_sum_steps=padded_step_sums,
            )
            return

        for step, attn_backend in enumerate(self.step_attn_backends):
            step_seq_lens = seq_lens_steps[step, :bs]
            step_seq_lens_cpu = seq_lens_cpu_steps[step]
            attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                step_seq_lens,
                int(padded_step_sums[step].item()),
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=None,
                seq_lens_cpu=step_seq_lens_cpu,
            )
