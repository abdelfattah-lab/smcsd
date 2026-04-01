"""SMC Worker: A speculative-decoding-like algorithm with two independent models.

Draft model performs gamma+1 autoregressive decode steps.
Score model performs one extend forward pass on the drafted tokens.
Computes logprob difference between the two models per request.
No rejection — all drafted tokens are accepted.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.smc.smc_info import SMCDraftInput
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

logger = logging.getLogger(__name__)


class SMCWorker(BaseSpecWorker):
    """SMC worker composing two independent TpModelWorkers (draft + score)."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.device = server_args.device
        self._target_worker = target_worker  # score model

        self.gamma = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = self.gamma + 1
        self.smc_draft_temperature = server_args.smc_draft_temperature
        self.smc_target_temperature = max(
            float(server_args.smc_target_temperature), 1e-5
        )

        # Share req_to_token_pool, separate KV caches
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Set class-level constant for KV allocation
        SMCDraftInput.ALLOC_LEN_PER_DECODE = self.speculative_num_draft_tokens

        # Override context length of draft model to match score model
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph during TpModelWorker init —
        # we capture manually after the draft model is fully set up
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Create draft TpModelWorker — fully independent, no shared lm_head/embed
        self._draft_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            pp_rank=0,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=target_worker.model_runner.memory_pool_config,
        )

        self.draft_runner = self._draft_worker.model_runner
        self.score_runner = self._target_worker.model_runner

        # Create multi-step draft attention backend (FA3, triton, etc.)
        # This pre-computes per-step attention metadata in one call,
        # avoiding per-step replay_prepare overhead.
        from sglang.srt.speculative.draft_utils import DraftBackendFactory

        # MultiStepBackend creates (num_steps - 1) sub-backends.
        # SMC needs gamma+1 forwards, so pass gamma+2 to get gamma+1 backends.
        factory = DraftBackendFactory(
            server_args,
            self.draft_runner,
            topk=1,
            speculative_num_steps=self.gamma + 2,
        )
        self.draft_attn_backend = factory.create_decode_backend()

        # Restore cuda graph flag and capture graphs for the draft model
        server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        if not backup_disable_cuda_graph:
            self.draft_runner.init_device_graphs()

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def model_config(self):
        return self._target_worker.model_config

    @property
    def model_runner(self):
        return self._target_worker.model_runner

    def clear_cache_pool(self):
        pass

    def materialize_smc_parent_draft_prefix(self, req) -> None:
        """No-op: _forward_extend already prefills both models."""
        pass

    # ------------------------------------------------------------------ #
    #  Main entry point — called by the scheduler
    # ------------------------------------------------------------------ #

    def forward_batch_generation(self, batch):
        # Non-overlap scheduler passes ScheduleBatch; convert to ModelWorkerBatch.
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        if isinstance(batch, ScheduleBatch):
            batch = batch.get_model_worker_batch()

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_extend(batch)
        else:
            return self._forward_decode(batch)

    # ------------------------------------------------------------------ #
    #  EXTEND (prefill) — prefill both models, return score result
    # ------------------------------------------------------------------ #

    def _forward_extend(self, batch: ModelWorkerBatch):
        # 1. Score model prefill
        score_result = self._target_worker.forward_batch_generation(batch)

        # 2. Draft model prefill — samples the first token (x0)
        draft_batch = self._make_clean_batch(batch)
        draft_result = self._draft_worker.forward_batch_generation(draft_batch)

        # 3. Use draft model's sampled token as verified_id (draft drives generation)
        bs = len(batch.seq_lens)
        score_result.next_token_ids = draft_result.next_token_ids

        # x0 is sampled but its KV is NOT written during prefill.
        # Keep seq_lens at prompt_len (not +1) so the first decode cycle
        # starts at the correct position and writes x0's KV.
        # batch.seq_lens is a schedule-stream tensor — safe without verify_done.
        score_result.next_draft_input = SMCDraftInput(
            verified_id=draft_result.next_token_ids,
            new_seq_lens=batch.seq_lens,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )
        score_result.accept_lens = torch.zeros(bs, dtype=torch.int32, device=self.device)
        return score_result

    # ------------------------------------------------------------------ #
    #  DECODE — draft AR → score extend → logprob diff
    # ------------------------------------------------------------------ #

    def _forward_decode(self, batch: ModelWorkerBatch):
        if batch.forward_mode.is_idle():
            return self._forward_idle(batch)

        # record_stream on tensors created on the previous forward stream
        # that we'll read on this forward stream. seq_lens is now advanced
        # on the schedule stream (in prepare_for_decode), so it's protected
        # by forward_stream.wait_stream(schedule_stream) — no record_stream needed.
        current_stream = torch.get_device_module(self.device).current_stream()
        if batch.req_pool_indices is not None:
            batch.req_pool_indices.record_stream(current_stream)

        draft_input: SMCDraftInput = batch.spec_info
        if draft_input.verified_id is not None:
            draft_input.verified_id.record_stream(current_stream)

        bs = len(draft_input._orig_seq_lens)
        gamma = self.gamma

        # ---- 1. Prepare draft: assign cache locs, create ForwardBatch ----
        draft_fb, can_cuda_graph, cache_locs, all_positions, all_seq_lens = (
            draft_input.prepare_for_draft(
                self.req_to_token_pool,
                batch,
                self.draft_runner.graph_runner if hasattr(self.draft_runner, 'graph_runner') else None,
                self.draft_runner,
                gamma,
            )
        )

        # ---- 2. Draft AR: gamma+1 decode steps ----
        # Initialize multi-step attention metadata ONCE for all steps.
        # Each sub-backend uses its speculative_step_id to compute the
        # correct per-step cache_seqlens and page_table.
        use_multistep = (
            self.draft_attn_backend is not None and not can_cuda_graph
        )
        if use_multistep and not draft_fb.forward_mode.is_idle():
            # Set spec_info and base prefix seq_lens for multi-step metadata init.
            # Each sub-backend adds its speculative_step_id to get per-step values.
            draft_fb.spec_info = draft_input
            draft_fb.seq_lens = draft_input._orig_seq_lens
            draft_fb.seq_lens_cpu = draft_input._orig_seq_lens_cpu
            self.draft_attn_backend.init_forward_metadata(draft_fb)

        x0 = draft_input.verified_id  # (bs,)
        all_tokens = [x0]
        draft_logprobs = []
        current_ids = x0

        for step in range(gamma + 1):
            # Update only the fields that change per step — no GPU→CPU sync
            draft_fb.input_ids = current_ids
            draft_fb.positions = all_positions[:, step].contiguous()
            draft_fb.out_cache_loc = cache_locs[:, step].contiguous()

            if use_multistep:
                # Swap to this step's pre-initialized attention backend
                draft_fb.attn_backend = self.draft_attn_backend.attn_backends[step]
                draft_out = self.draft_runner.forward(
                    draft_fb, skip_attn_backend_init=True
                )
            else:
                # Fallback: per-step forward with full metadata init
                draft_fb.seq_lens = all_seq_lens[:, step].contiguous()
                draft_fb.seq_lens_sum = (
                    draft_input._orig_seq_lens_sum + bs * (step + 1)
                )
                draft_fb.seq_lens_cpu = (
                    draft_input._orig_seq_lens_cpu + (step + 1)
                )
                draft_out = self.draft_runner.forward(draft_fb)
            logits = draft_out.logits_output.next_token_logits  # (bs, vocab)

            # Sample next token
            if self.smc_draft_temperature > 0:
                probs = torch.softmax(logits / self.smc_draft_temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(logits, dim=-1)

            # Collect logprob (only first gamma steps)
            if step < gamma:
                log_probs = torch.log(probs + 1e-8)
                token_logprob = log_probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
                draft_logprobs.append(token_logprob)

            all_tokens.append(next_token)
            current_ids = next_token

        # draft_logprobs: gamma entries — logprob(x1),...,logprob(x(gamma))
        draft_logprobs_stacked = torch.stack(draft_logprobs, dim=1)  # (bs, gamma)

        # ---- 3. Score verify: [x0, ..., x(gamma)] via TARGET_VERIFY ----
        verify_forward_batch, can_run_cuda_graph = draft_input.prepare_for_verify(
            self.req_to_token_pool,
            batch,
            self._target_worker,
            all_tokens,
            cache_locs,
            gamma,
        )

        score_result = self._target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )

        # ---- 4. Extract score logprobs from logits (first gamma only) ----
        # TARGET_VERIFY returns raw logits; compute logprobs directly
        score_logits = score_result.logits_output.next_token_logits  # (bs*(gamma+1), vocab)
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]} rows, "
            f"expected {expected_rows} (bs={bs}, gamma+1={gamma+1}, "
            f"cuda_graph={can_run_cuda_graph})"
        )
        # NOTE(CCC): Check whether we need a logprob for now.
        score_log_probs = torch.log_softmax(score_logits, dim=-1)
        score_log_probs = score_log_probs.reshape(bs, gamma + 1, -1)
        # Gather logprobs for tokens [x1, ..., x_gamma] at positions [0, ..., gamma-1]
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)  # (bs, gamma)
        score_logprobs_stacked = score_log_probs[:, :gamma, :].gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)  # (bs, gamma)

        # ---- 5. Compute logprob diff ----
        logprob_diff = (score_logprobs_stacked - draft_logprobs_stacked).sum(dim=1)

        # ---- 6. Sample bonus token from tempered target logits ----
        bonus_logits = score_logits.reshape(bs, gamma + 1, -1)[:, -1, :]
        bonus_log_probs = torch.log_softmax(
            bonus_logits / self.smc_target_temperature, dim=-1
        )
        bonus = torch.multinomial(bonus_log_probs.exp(), num_samples=1).squeeze(-1)

        # ---- 7. Build output ----
        # Output: [x1, ..., x_r, bonus] — bonus replaces overdraft x_{r+1}
        output_token_ids = torch.stack(
            all_tokens[1 : gamma + 1] + [bonus], dim=1
        )  # (bs, gamma+1)
        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full((bs,), gamma + 1, dtype=torch.int32, device=self.device)
        next_verified_id = bonus

        # record_stream on output tensors: created on forward stream,
        # read by scheduler via copy_to_cpu on schedule stream.
        next_token_ids.record_stream(current_stream)
        accept_lens.record_stream(current_stream)
        next_verified_id.record_stream(current_stream)
        logprob_diff.record_stream(current_stream)

        # Use schedule-stream new_seq_lens (stashed in prepare_for_decode).
        # No forward-stream tensor crosses to the scheduler → no verify_done needed.
        next_draft_input = SMCDraftInput(
            verified_id=next_verified_id,
            new_seq_lens=draft_input._new_seq_lens,
            logprob_diff=logprob_diff,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )

        return GenerationBatchResult(
            logits_output=score_result.logits_output,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            next_draft_input=next_draft_input,
            logprob_diff=logprob_diff,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _forward_idle(self, batch: ModelWorkerBatch):
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.empty(0, dtype=torch.int64, device=self.device),
            accept_lens=torch.empty(0, dtype=torch.int32, device=self.device),
            next_draft_input=SMCDraftInput.create_idle_input(self.device),
        )

    def _make_clean_batch(self, batch: ModelWorkerBatch) -> ModelWorkerBatch:
        """Create a copy of the batch with no spec_info (for draft model)."""
        return dataclasses.replace(
            batch, spec_info=None, capture_hidden_mode=CaptureHiddenMode.NULL
        )
