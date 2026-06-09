"""SMC worker: dense-AR draft path.

Draft model performs gamma+1 autoregressive decode steps.
Score model performs one extend forward pass on the drafted tokens.
Computes logprob difference between the two models per request.
No rejection — all drafted tokens are accepted.

Supports any (target, draft) pair where the draft can be loaded as a
standalone autoregressive LM. Hybrid (Mamba+attention) targets whose draft
has a different recurrent-state shape get an isolated draft Mamba pool via
``_maybe_isolate_dense_hybrid_draft_state``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.server_args import ServerArgs
from smcsd.core.info import SMCDecodeContext, SMCDraftInput
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

logger = logging.getLogger(__name__)


class SMCDenseDraftTpModelWorker(TpModelWorker):
    """Draft worker that keeps a standalone draft model as a normal LM.

    Upstream SGLang rewrites several hybrid architectures (including Qwen3.5)
    to their MTP draft variants whenever ``is_draft_model=True``. That is
    correct for NEXTN/MTP speculative decoding, but SMC's dense mode expects a
    fully autoregressive draft model. Keep ``is_draft_worker=True`` for shared
    request/KV-pool semantics, while loading the draft config without the MTP
    architecture rewrite.
    """

    def _init_model_config(self):
        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=self.server_args.speculative_draft_model_path,
            model_revision=self.server_args.speculative_draft_model_revision,
            is_draft_model=False,
        )


class SMCWorker(BaseSpecWorker):
    """Standalone SMC worker (SMCDecodeContext + SMCDraftInput)."""

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
        self.tp_size = target_worker.tp_size
        self.device = server_args.device
        self._target_worker = target_worker  # score model

        self.gamma = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = self.gamma + 1
        self.smc_draft_temperature = server_args.smc_draft_temperature
        self.smc_target_temperature = max(
            float(server_args.smc_target_temperature), 1e-5
        )
        # Only the dense-AR draft path is supported here.
        self._dense_draft_hybrid_req_to_token_pool = None

        # Share req_to_token_pool, separate KV caches
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Set class-level constant for KV allocation
        SMCDraftInput.ALLOC_LEN_PER_DECODE = self.speculative_num_draft_tokens

        server_args.context_length = target_worker.model_runner.model_config.context_len
        self.score_runner = self._target_worker.model_runner

        # Do not capture cuda graph during TpModelWorker init —
        # we capture manually after the draft model is fully set up
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Create draft TpModelWorker — fully independent, no shared lm_head/embed
        self._draft_worker = SMCDenseDraftTpModelWorker(
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

        # ---- EDA Phase 1: Variance Reduction (Better Sobol) ----
        self.sampling_method = os.environ.get("SMCSD_SAMPLING_METHOD", "multinomial")
        if self.sampling_method == "qmc":
            logger.info("SMCWorker: Using torch.quasirandom.SobolEngine for QMC sampling")
            self.sobol_engine = torch.quasirandom.SobolEngine(
                dimension=self.gamma + 1, scramble=True
            )

        # ---- EDA Phase 2b: Divergence Pruning (Inverse Early-Halt) ----
        self.use_early_halt = os.environ.get("SMCSD_USE_EARLY_HALT", "false").lower() == "true"
        self.early_halt_threshold = float(os.environ.get("SMCSD_EARLY_HALT_THRESHOLD", "8.0"))
        if self.use_early_halt:
            logger.info(f"SMCWorker: Inverse Early-Halt enabled (threshold={self.early_halt_threshold})")

        # Hybrid Qwen3.5/3.6 drafts need an isolated MambaPool sized to the
        # draft's recurrent state shape (different from the target's).
        self._maybe_isolate_dense_hybrid_draft_state()

        # Multi-step draft attention backend.
        if (
            self.draft_runner.model_config.is_hybrid
            and self.draft_runner.model_config.attention_arch
        ):
            from smcsd.core.hybrid_multistep_backend import (
                TritonHybridMultiStepDraftBackend,
            )

            self.draft_attn_backend = TritonHybridMultiStepDraftBackend(
                self.draft_runner,
                topk=1,
                speculative_num_steps=self.gamma + 2,
            )
        else:
            from sglang.srt.speculative.draft_utils import DraftBackendFactory

            factory = DraftBackendFactory(
                server_args,
                self.draft_runner,
                topk=1,
                speculative_num_steps=self.gamma + 2,
            )
            self.draft_attn_backend = factory.create_decode_backend()

        # Restore cuda graph
        server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_runner.init_device_graphs()

    def _maybe_isolate_dense_hybrid_draft_state(self) -> None:
        """Create an isolated draft recurrent pool for hybrid drafts.

        Asymmetric hybrid pairs (e.g. 9B target + 2B draft) have different
        SSM state shapes. Upstream SGLang uses a single global MambaPool,
        so we must manually initialize a second pool for the draft model.
        """
        if (
            not self.draft_runner.model_config.is_hybrid
            or not self.draft_runner.model_config.recurrent_arch
        ):
            return

        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            HybridReqToTokenPool,
        )

        target_pool = self.req_to_token_pool
        draft_config = self.draft_runner.mambaish_config
        _smc_debug = bool(os.environ.get("SMCSD_HYBRID_DEBUG"))
        if _smc_debug:
            print(
                f"[SMC] Isolating draft recurrent state: "
                f"target_shape={self.score_runner.mambaish_config.recurrent_state_shape}, "
                f"draft_shape={draft_config.recurrent_state_shape}"
            )

        self._dense_draft_hybrid_req_to_token_pool = HybridReqToTokenPool(
            max_num_reqs=target_pool.max_num_reqs,
            device=self.device,
            recurrent_state_shape=draft_config.recurrent_state_shape,
            recurrent_state_dtype=draft_config.recurrent_state_dtype,
        )

        # Patch the draft runner to use the isolated pool
        self.draft_runner.req_to_token_pool = self._dense_draft_hybrid_req_to_token_pool

    def _commit_target_mamba_state_after_verify(
        self, verify_forward_batch: ForwardBatch, accepted_steps: torch.Tensor
    ):
        """Advance the target model's recurrent state after verification.

        SMC accepted all drafted tokens; the verification pass was an extend
        pass on the full drafted sequence. We need to scatter the recurrent
        state from the end of that sequence back into the permanent request
        slots in the target's MambaPool.
        """
        from sglang.srt.layers.attention.linear.gdn_backend import (
            commit_mamba_state_after_verify,
        )

        commit_mamba_state_after_verify(
            verify_forward_batch, accepted_steps, self.score_runner.req_to_token_pool
        )

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> TpModelWorker:
        return self._draft_worker

    @property
    def model_config(self):
        return self._target_worker.model_config

    @property
    def model_runner(self):
        return self._target_worker.model_runner

    def clear_cache_pool(self):
        self.draft_runner.req_to_token_pool.clear()

    def forward_batch_generation(
        self,
        batch: ScheduleBatch | ModelWorkerBatch,
    ) -> GenerationBatchResult:
        if isinstance(batch, ScheduleBatch):
            batch = batch.get_model_worker_batch()

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_extend(batch)
        else:
            return self._forward_decode(batch)

    # ── EXTEND (prefill) ──

    def _forward_extend(self, batch: ModelWorkerBatch):
        # Score model prefill
        score_result = self._target_worker.forward_batch_generation(batch)

        # Draft model prefill — samples the first token (x0)
        draft_batch = self._make_clean_batch(batch)
        draft_result = self._draft_worker.forward_batch_generation(draft_batch)

        # Use draft model's sampled token as verified_id
        bs = len(batch.seq_lens)
        score_result.next_token_ids = draft_result.next_token_ids

        # x0 KV is NOT written during prefill — first decode writes it.
        score_result.next_draft_input = SMCDraftInput(
            verified_id=draft_result.next_token_ids,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )
        # Pre-initialize context for the first decode round
        score_result.next_draft_input.prepare_for_decode(batch)

        score_result.accept_lens = torch.zeros(
            bs, dtype=torch.int32, device=self.device
        )
        return score_result

    # ── DECODE ──

    def _forward_decode(self, batch: ModelWorkerBatch):
        if batch.forward_mode.is_idle():
            return self._forward_idle(batch)

        current_stream = torch.get_device_module(self.device).current_stream()
        if batch.req_pool_indices is not None:
            batch.req_pool_indices.record_stream(current_stream)

        draft_input: SMCDraftInput = batch.spec_info
        if draft_input is None:
            # Fallback for unexpected null spec_info
            draft_input = SMCDraftInput(
                verified_id=batch.input_ids,
                num_tokens_per_req=self.speculative_num_draft_tokens,
            )
            batch.spec_info = draft_input

        if draft_input.decode_ctx is None:
            draft_input.prepare_for_decode(batch)

        ctx: SMCDecodeContext = draft_input.decode_ctx

        if draft_input.verified_id is not None:
            draft_input.verified_id.record_stream(current_stream)

        # ---- 1. Prepare draft ----
        draft_fb, can_cuda_graph, cache_locs, all_positions, all_seq_lens = (
            ctx.prepare_for_draft(
                draft_input.verified_id,
                self.req_to_token_pool,
                batch,
                self.draft_runner.graph_runner
                if hasattr(self.draft_runner, "graph_runner")
                else None,
                self.draft_runner,
            )
        )

        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma

        # ---- 2. Dense draft AR: gamma+1 decode steps ----
        use_multistep = (
            self.draft_attn_backend is not None
            and not can_cuda_graph
        )
        if use_multistep and not draft_fb.forward_mode.is_idle():
            draft_fb.spec_info = draft_input
            draft_fb.seq_lens = ctx.orig_seq_lens
            draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu
            self.draft_attn_backend.init_forward_metadata(draft_fb)

        x0 = draft_input.verified_id
        all_tokens = [x0]
        draft_logprobs = []
        current_ids = x0

        # ---- 2.1 Prepare sampling random variables (EDA Phase 1) ----
        u_samples_t = None
        if not draft_fb.forward_mode.is_idle():
            if self.sampling_method == "qmc":
                # Continuous Sobol sequence across the whole batch.
                # Drawing N*M points ensures each request group gets a unique 
                # low-discrepancy set without artificial correlations or identity.
                u_samples_t = self.sobol_engine.draw(bs).to(self.device)
            elif self.use_antithetic:
                u_samples = np.random.uniform(0, 1, size=(bs, gamma + 1))
                # Couple pairs: i and i+1
                for i in range(0, bs, 2):
                    if i + 1 < bs:
                        u_samples[i+1] = 1.0 - u_samples[i]
                u_samples_t = torch.tensor(u_samples, dtype=torch.float32, device=self.device)

        actual_steps = gamma + 1
        sum_neg_logprob = torch.zeros(bs, device=self.device)
        target_vocab_limit = self.score_runner.model.model.embed_tokens.weight.shape[0]

        for step in range(gamma + 1):
            draft_fb.input_ids = torch.clamp(current_ids, 0, target_vocab_limit - 1)
            draft_fb.positions = all_positions[:, step].contiguous()
            draft_fb.out_cache_loc = cache_locs[:, step].contiguous()

            if use_multistep:
                draft_fb.attn_backend = self.draft_attn_backend.attn_backends[step]
                draft_out = self.draft_runner.forward(
                    draft_fb, skip_attn_backend_init=True
                )
            else:
                draft_fb.seq_lens = all_seq_lens[:, step].contiguous()
                draft_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * (step + 1)
                draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + (step + 1)
                draft_out = self.draft_runner.forward(draft_fb)

            logits = draft_out.logits_output.next_token_logits
            scaled_logits = logits / self.smc_draft_temperature
            log_probs = torch.log_softmax(scaled_logits, dim=-1)
            
            if u_samples_t is not None:
                # Coordinated sampling (QMC or Antithetic) via Inverse Transform
                u = u_samples_t[:, step]
                cdf = torch.cumsum(log_probs.exp(), dim=-1)
                draft_idx = torch.clamp(torch.searchsorted(cdf, u.unsqueeze(1)).squeeze(-1), max=log_probs.shape[1] - 1)
            elif self.smc_draft_temperature > 0:
                draft_idx = torch.multinomial(
                    log_probs.exp(), num_samples=1
                ).squeeze(-1)
            else:
                draft_idx = torch.argmax(logits, dim=-1)

            next_token = draft_idx

            if step < gamma:
                token_logprob = log_probs.gather(
                    1, draft_idx.unsqueeze(1)
                ).squeeze(1)
                draft_logprobs.append(token_logprob)
                sum_neg_logprob -= token_logprob

            all_tokens.append(next_token)
            current_ids = next_token

            # ---- EDA Phase 2b: Divergence Pruning (Inverse Early-Halt) ----
            if self.use_early_halt and step > 0:
                avg_neg_logprob = sum_neg_logprob / (step + 1)
                # Halt if the semantic path is lost (very high negative logprob)
                if (avg_neg_logprob > self.early_halt_threshold).all():
                    actual_steps = step + 1
                    break

        if actual_steps < gamma + 1:
            # Adjust remaining steps and tokens
            all_tokens = all_tokens[:actual_steps + 1]
            gamma = actual_steps - 1
            # Truncate cache_locs and positions for verify
            cache_locs = cache_locs[:, :actual_steps]
            all_positions = all_positions[:, :actual_steps]

        draft_logprobs_stacked = torch.stack(draft_logprobs, dim=1)

        # ---- 3. Score verify ----
        verify_forward_batch, can_run_cuda_graph = ctx.prepare_for_verify(
            self.req_to_token_pool,
            batch,
            self._target_worker,
            all_tokens,
            cache_locs,
        )

        score_result = self._target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )

        if self.score_runner.hybrid_gdn_config is not None:
            accepted_steps = torch.full(
                (bs,), gamma, dtype=torch.int64, device=self.device
            )
            self._commit_target_mamba_state_after_verify(
                verify_forward_batch, accepted_steps
            )

        # ---- 4. Extract score logprobs ----
        score_logits = score_result.logits_output.next_token_logits
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]} rows, "
            f"expected {expected_rows} (bs={bs}, gamma+1={gamma + 1})"
        )
        score_log_probs = torch.log_softmax(score_logits, dim=-1)
        score_log_probs = score_log_probs.reshape(bs, gamma + 1, -1)
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)
        score_logprobs_stacked = score_log_probs[:, :gamma, :].gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)

        logprob_diff = (score_logprobs_stacked - draft_logprobs_stacked).sum(dim=1)

        # ---- 6. Bonus token ----
        bonus_logits = score_logits.reshape(bs, gamma + 1, -1)[:, -1, :]
        bonus_log_probs = torch.log_softmax(
            bonus_logits / self.smc_target_temperature, dim=-1
        )
        bonus = torch.multinomial(bonus_log_probs.exp(), num_samples=1).squeeze(-1)

        # ---- 7. Output ----
        output_token_ids = torch.stack(
            all_tokens[1 : gamma + 1] + [bonus], dim=1
        )
        next_verified_id = bonus

        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=self.device
        )

        next_token_ids.record_stream(current_stream)
        accept_lens.record_stream(current_stream)
        next_verified_id.record_stream(current_stream)
        logprob_diff.record_stream(current_stream)

        next_draft_input = SMCDraftInput(
            verified_id=next_verified_id,
            logprob_diff=logprob_diff,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )
        next_draft_input.prepare_for_decode(batch)

        return GenerationBatchResult(
            logits_output=score_result.logits_output,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            next_draft_input=next_draft_input,
            logprob_diff=logprob_diff,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _forward_idle(self, batch: ModelWorkerBatch) -> GenerationBatchResult:
        """Handle idle batch (no active requests)."""
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.empty(0, dtype=torch.int64, device=self.device),
            accept_lens=torch.empty(0, dtype=torch.int32, device=self.device),
            next_draft_input=SMCDraftInput.create_idle_input(self.device),
        )

    def _make_clean_batch(self, batch: ModelWorkerBatch) -> ModelWorkerBatch:
        """Copy batch with no spec_info (for draft model)."""
        return dataclasses.replace(
            batch, spec_info=None, capture_hidden_mode=CaptureHiddenMode.NULL
        )
