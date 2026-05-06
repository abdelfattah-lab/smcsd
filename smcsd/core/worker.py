"""SMC worker: standalone worker using the ``SMCDecodeContext`` API.

Draft model performs ``gamma + 1`` autoregressive decode steps.  Target
model performs one extend forward pass on the drafted tokens.  Logprob
difference between the two models is computed per request.  No rejection —
all drafted tokens are accepted (SMC accepts every draft and reweights).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from smcsd.core.info import SMCDecodeContext, SMCDraftInput
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

import os
import json

logger = logging.getLogger(__name__)

# Per-step debug dump for SMC pipeline. Set SMCSD_DEBUG_DUMP=/tmp/smc_debug.jsonl
# to enable; one JSON line per decode step with (step_idx, log_weights,
# logprob_diff, draft_logprobs, target_logprobs at matching positions, bonus
# tokens, accepted_tokens). Only safe to use with very small workloads
# (overhead = serialize tensors + jsonl write per step).
_SMC_DEBUG_PATH = os.environ.get("SMCSD_DEBUG_DUMP", "")
_smc_debug_step = [0]  # mutable counter wrapper


class SMCWorker(BaseSpecWorker):
    """Standalone SMC worker.  Uses ``SMCDecodeContext`` + ``SMCDraftInput``."""

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

        # Create draft TpModelWorker — fully independent, no shared lm_head/embed.
        # Use SMCDraftTpModelWorker so a hybrid draft gets its own MambaPool
        # (DraftHybridReqToTokenPool) sized to the draft's state shapes, with
        # the req_to_token block table still aliased to the target's pool.
        # On a dense draft this is a no-op pass-through.
        from smcsd.managers.smc_tp_worker import SMCDraftTpModelWorker

        self._draft_worker = SMCDraftTpModelWorker(
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

        # Draft's req_to_token_pool — when the draft is hybrid, this is a
        # DraftHybridReqToTokenPool with its own MambaPool. When the draft
        # is dense, it's the target's pool (alias). The SMC scheduler
        # reads this to alloc/copy/free draft mamba state in lockstep
        # with target mamba state during fan-out and resampling.
        self.draft_req_to_token_pool = self._draft_worker.req_to_token_pool

        # Multi-step draft attention backend.
        #
        # ``DraftBackendFactory.create_decode_backend`` returns a flat-attention
        # multi-step backend (FlashAttentionMultiStepBackend, etc.). That's
        # correct for dense Qwen / Llama drafts, but it doesn't know about
        # Mamba/SSM layers — the linear_attn forward in qwen3_next.py takes
        # the slow path (forward_batch.attn_backend.forward(layer=..., mixed_qkv=..., a=..., b=...))
        # and crashes with::
        #
        #     TypeError: AttentionBackend.forward() missing 3 required positional
        #     arguments: 'q', 'k', and 'v'
        #
        # because the multi-step backend's signature is the dense (q, k, v, ...)
        # one. For hybrid drafts (Qwen3-Next, Qwen3.5, Falcon-H1, ...), skip
        # the multi-step path and let per-step draft decode go through the
        # draft model_runner's regular ``HybridLinearAttnBackend`` (set up by
        # upstream's ``attn_backend_wrapper`` when ``mambaish_config`` is set),
        # which dispatches mamba layers through the GDN / Mamba2 backend
        # correctly.
        draft_is_hybrid = (
            getattr(self.draft_runner, "mambaish_config", None) is not None
        )
        if draft_is_hybrid:
            self.draft_attn_backend = None
        else:
            from sglang.srt.speculative.draft_utils import DraftBackendFactory

            factory = DraftBackendFactory(
                server_args,
                self.draft_runner,
                topk=1,
                speculative_num_steps=self.gamma + 2,
            )
            self.draft_attn_backend = factory.create_decode_backend()

        # Restore cuda graph and capture for draft model
        server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        if not backup_disable_cuda_graph:
            self.draft_runner.init_device_graphs()

    # ── Properties (required by BaseSpecWorker / scheduler) ──

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

    # ── Main entry point ──

    def forward_batch_generation(self, batch):
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
        ctx: SMCDecodeContext = draft_input.decode_ctx

        if draft_input.verified_id is not None:
            draft_input.verified_id.record_stream(current_stream)

        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma

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

        # ---- 2. Draft AR: gamma+1 decode steps ----
        use_multistep = self.draft_attn_backend is not None and not can_cuda_graph
        if use_multistep and not draft_fb.forward_mode.is_idle():
            draft_fb.spec_info = draft_input
            draft_fb.seq_lens = ctx.orig_seq_lens
            draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu
            self.draft_attn_backend.init_forward_metadata(draft_fb)

        x0 = draft_input.verified_id
        all_tokens = [x0]
        draft_logprobs = []
        draft_argmax_steps = []
        draft_top1_logits = []
        current_ids = x0

        for step in range(gamma + 1):
            draft_fb.input_ids = current_ids
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
            if self.smc_draft_temperature > 0:
                next_token = torch.multinomial(
                    log_probs.exp(), num_samples=1
                ).squeeze(-1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            if step < gamma:
                token_logprob = log_probs.gather(
                    1, next_token.unsqueeze(1)
                ).squeeze(1)
                draft_logprobs.append(token_logprob)
                if _SMC_DEBUG_PATH:
                    draft_argmax_steps.append(torch.argmax(logits, dim=-1))
                    draft_top1_logits.append(logits.max(dim=-1).values)

            all_tokens.append(next_token)
            current_ids = next_token

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

        # ---- 4. Extract score logprobs ----
        score_logits = score_result.logits_output.next_token_logits
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]} rows, "
            f"expected {expected_rows} (bs={bs}, gamma+1={gamma + 1}, "
            f"cuda_graph={can_run_cuda_graph})"
        )
        # IS weight = log p_target(x) - log p_draft(x), where both p's are
        # evaluated at their respective sampling temperatures. The draft
        # logprobs (above) are computed at smc_draft_temperature. The target
        # logprobs MUST be computed at smc_target_temperature to match — the
        # bonus-token sampling at the bottom of this method already does
        # that; this verify step previously did not, which made IS weights
        # systematically non-zero even when target = draft (same model). On
        # an SMC self-pair with temp=0.7 that bug pushed accuracy from ~96%
        # to ~20% via spurious resampling.
        score_log_probs = torch.log_softmax(
            score_logits / self.smc_target_temperature, dim=-1
        )
        score_log_probs = score_log_probs.reshape(bs, gamma + 1, -1)
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)
        score_logprobs_stacked = score_log_probs[:, :gamma, :].gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)

        # ---- 5. Logprob diff (IS weight in log-space) ----
        logprob_diff = (score_logprobs_stacked - draft_logprobs_stacked).sum(dim=1)

        if _SMC_DEBUG_PATH:
            try:
                step_idx = _smc_debug_step[0]
                _smc_debug_step[0] += 1
                target_logits_3d = score_logits.reshape(bs, gamma + 1, -1)[:, :gamma, :]
                target_argmax = target_logits_3d.argmax(dim=-1)
                target_top1_val = target_logits_3d.max(dim=-1).values
                target_logits_at_drafted = target_logits_3d.gather(
                    2, target_tokens.unsqueeze(2)
                ).squeeze(2)
                draft_argmax = torch.stack(draft_argmax_steps, dim=1) if draft_argmax_steps else None
                draft_top1 = torch.stack(draft_top1_logits, dim=1) if draft_top1_logits else None
                rec = {
                    "step": step_idx,
                    "bs": int(bs),
                    "gamma": int(gamma),
                    "draft_temp": float(self.smc_draft_temperature),
                    "target_temp": float(self.smc_target_temperature),
                    "draft_logprobs": draft_logprobs_stacked.detach().to(torch.float32).cpu().tolist(),
                    "target_logprobs": score_logprobs_stacked.detach().to(torch.float32).cpu().tolist(),
                    "logprob_diff": logprob_diff.detach().to(torch.float32).cpu().tolist(),
                    "drafted_tokens": [t.detach().cpu().tolist() for t in all_tokens[1:gamma+1]],
                    "target_argmax": target_argmax.detach().cpu().tolist(),
                    "draft_argmax": draft_argmax.detach().cpu().tolist() if draft_argmax is not None else None,
                    "target_top1_logit": target_top1_val.detach().to(torch.float32).cpu().tolist(),
                    "draft_top1_logit": draft_top1.detach().to(torch.float32).cpu().tolist() if draft_top1 is not None else None,
                    "target_logit_at_drafted": target_logits_at_drafted.detach().to(torch.float32).cpu().tolist(),
                }
                with open(_SMC_DEBUG_PATH, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                logger.warning(f"SMC debug dump failed: {e}")

        # ---- 5b. Commit target's intermediate mamba state to ssm_state.
        #
        # Critical: the spec verify path uses
        # ``selective_state_update(disable_state_update=True, ...)`` and
        # ``causal_conv1d_update_triton(intermediate_conv_window=...)`` which
        # write per-step intermediate states to ``intermediate_ssm`` /
        # ``intermediate_conv_window`` but DO NOT advance the request's
        # actual ``ssm_state`` / ``conv_state``. The next cycle's verify
        # would then read STALE state (frozen at prefill across all cycles)
        # producing logits conditioned only on the prompt + this cycle's
        # γ+1 tokens, missing all prior cycles' history. EAGLE / dflash /
        # MTP all call ``update_mamba_state_after_mtp_verify`` after verify
        # to scatter intermediate_ssm[step] -> ssm_state. SMC didn't, which
        # caused same-model 9B+9B at temp=0.1 to drop to 42% (the model
        # forgets all output past the prompt).
        attn_backend = getattr(self.score_runner, "attn_backend", None)
        if attn_backend is not None and hasattr(
            attn_backend, "update_mamba_state_after_mtp_verify"
        ):
            # SMC accepts all γ drafted tokens. The chunk has γ+1 positions
            # (verified_id at 0, x_1..x_γ at 1..γ); intermediate_ssm[γ] is
            # the state after consuming x_γ — the conditioning needed by the
            # next cycle's draft AR step 0 (which feeds bonus_k as input).
            accepted_steps = torch.full(
                (bs,), gamma, dtype=torch.int32, device=self.device
            )
            attn_backend.update_mamba_state_after_mtp_verify(
                accepted_steps=accepted_steps,
                mamba_track_indices=None,
                mamba_steps_to_track=None,
                model=self.score_runner.model,
            )

        # ---- 5c. Sync draft mamba state from target's now-committed state.
        # On hybrid+hybrid the draft maintains a separate MambaPool. Copy
        # target's just-committed extend-kernel state into the draft pool so
        # the next cycle's draft AR step 0 reads the same conditioning the
        # target verify position 0 will read. No-op on dense drafts (the
        # draft pool aliases the target's). Without this, the draft's
        # decode-kernel ssm_state and the target's extend-kernel ssm_state
        # diverge by 2-13 logit units within a single cycle and cause
        # spurious IS-weight inflation that drives ESS-collapse resampling.
        from smcsd.mem_cache.allocator import sync_draft_from_target_mamba
        sync_draft_from_target_mamba(
            self.req_to_token_pool,
            self.draft_req_to_token_pool,
            batch.req_pool_indices,
        )

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
        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=self.device
        )
        next_verified_id = bonus

        next_token_ids.record_stream(current_stream)
        accept_lens.record_stream(current_stream)
        next_verified_id.record_stream(current_stream)
        logprob_diff.record_stream(current_stream)

        next_draft_input = SMCDraftInput(
            verified_id=next_verified_id,
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
        """Copy batch with no spec_info (for draft model)."""
        return dataclasses.replace(
            batch, spec_info=None, capture_hidden_mode=CaptureHiddenMode.NULL
        )
