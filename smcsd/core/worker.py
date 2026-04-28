"""SMC worker: standalone worker using the ``SMCDecodeContext`` API.

Draft model performs ``gamma + 1`` autoregressive decode steps.  Target
model performs one extend forward pass on the drafted tokens.  Logprob
difference between the two models is computed per request.  No rejection —
all drafted tokens are accepted (SMC accepts every draft and reweights).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Optional

import math

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from smcsd.core.info import SMCDecodeContext, SMCDraftInput
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

logger = logging.getLogger(__name__)


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
        self.smc_draft_mode = getattr(server_args, "smc_draft_mode", "dense")
        if self.smc_draft_mode == "eagle3":
            self.smc_draft_mode = "eagle3_chain"
        self.is_eagle3 = str(self.smc_draft_mode).startswith("eagle3")
        self.is_eagle3_chain = self.smc_draft_mode == "eagle3_chain"
        self.is_eagle3_tree = self.smc_draft_mode in (
            "eagle3_tree_probe",
            "eagle3_tree_smc",
            "eagle3_tree_oracle",
        )
        self.is_eagle3_full_tree = self.smc_draft_mode in (
            "eagle3_tree_smc",
            "eagle3_tree_oracle",
        )
        self.eagle_topk = int(
            getattr(
                server_args,
                "smc_eagle_topk",
                getattr(server_args, "speculative_eagle_topk", 1),
            )
            or 1
        )
        self.eagle_num_draft_tokens = int(
            getattr(
                server_args,
                "smc_eagle_num_draft_tokens",
                getattr(server_args, "speculative_num_draft_tokens", None),
            )
            or (self.gamma + 1)
        )
        self.eagle_num_draft_tokens = max(self.eagle_num_draft_tokens, self.gamma + 1)
        self.eagle_diag_path = getattr(server_args, "smc_eagle3_collect_path", None)
        self._eagle_diag_file = None
        self._eagle_diag_counter = 0
        self.eagle_eps_uniform = float(
            getattr(server_args, "smc_eagle_eps_uniform", 0.0) or 0.0
        )
        if self.eagle_eps_uniform < 0.0 or self.eagle_eps_uniform >= 1.0:
            raise ValueError(
                f"smc_eagle_eps_uniform must be in [0, 1), got {self.eagle_eps_uniform}"
            )
        self._eagle_eps_log1me = (
            math.log1p(-self.eagle_eps_uniform) if self.eagle_eps_uniform > 0 else 0.0
        )
        self._eagle_eps_log_eps = (
            math.log(self.eagle_eps_uniform) if self.eagle_eps_uniform > 0 else 0.0
        )
        if self.is_eagle3_tree:
            logger.warning(
                "SMC EAGLE tree mode is experimental. eagle3_tree_probe keeps "
                "the original shallow first-branch behavior; eagle3_tree_smc "
                "and eagle3_tree_oracle build and score a native full-depth "
                "EAGLE tree before committing a path."
            )

        # Share req_to_token_pool, separate KV caches
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Set class-level constant for KV allocation
        SMCDraftInput.ALLOC_LEN_PER_DECODE = self.speculative_num_draft_tokens

        # Override context length of draft model to match score model
        server_args.context_length = target_worker.model_runner.model_config.context_len
        prev_allow_overwrite = os.environ.get(
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"
        )
        prev_dtype = server_args.dtype
        if self.is_eagle3:
            os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
            target_dtype = target_worker.model_runner.model_config.dtype
            dtype_to_str = {
                torch.bfloat16: "bfloat16",
                torch.float16: "float16",
                torch.float32: "float32",
            }
            server_args.dtype = dtype_to_str.get(target_dtype, "bfloat16")
            if hasattr(target_worker.model_runner.model, "set_eagle3_layers_to_capture"):
                target_worker.model_runner.model.set_eagle3_layers_to_capture()

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
        if self.is_eagle3:
            if prev_allow_overwrite is None:
                os.environ.pop("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", None)
            else:
                os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = (
                    prev_allow_overwrite
                )
            server_args.dtype = prev_dtype

        self.draft_runner = self._draft_worker.model_runner
        self.score_runner = self._target_worker.model_runner
        self.eagle_use_aux = False
        self._eagle3_hidden_dtype: Optional[torch.dtype] = None
        self.hot_token_id: Optional[torch.Tensor] = None

        if self.is_eagle3:
            eagle_cfg = getattr(
                self.draft_runner.model_config.hf_config, "eagle_config", {}
            ) or {}
            self.eagle_use_aux = eagle_cfg.get("use_aux_hidden_state", True)
            embed, head = self._target_worker.model_runner.model.get_embed_and_head()
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_runner.model.set_embed(embed)

            fc_layer = getattr(self.draft_runner.model.model, "fc", None)
            if fc_layer is not None and hasattr(fc_layer, "weight"):
                self._eagle3_hidden_dtype = fc_layer.weight.dtype
            else:
                self._eagle3_hidden_dtype = next(
                    p for p in self.draft_runner.model.parameters() if p is not embed
                ).dtype
            hot = getattr(self.draft_runner.model, "hot_token_id", None)
            if hot is not None:
                self.hot_token_id = hot.to(embed.device)
            backup_disable_cuda_graph = True

        # Multi-step draft attention backend
        from sglang.srt.speculative.draft_utils import DraftBackendFactory

        backend_topk = self.eagle_topk if self.is_eagle3_tree else 1
        factory = DraftBackendFactory(
            server_args,
            self.draft_runner,
            topk=backend_topk,
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

    def _eagle_sample_with_eps_mix(self, log_probs: torch.Tensor):
        """Sample from q_mix = (1-eps)*q_eagle + eps*Uniform(V) and return
        (idx, log_q_mix(idx)).

        log_probs has shape (bs, V) and is the log of q_eagle. When
        eagle_eps_uniform is 0, this is identical to the legacy multinomial
        sampling + gather. With eps>0:
          - With probability eps a row is sampled uniformly at random.
          - log_q_mix(t) = logsumexp(log(1-eps) + log_q_eagle(t), log(eps/V))
        bounds the worst-case log q from below by log(eps/V), which caps the
        worst-case logp - logq weight. This is the universal SMC-SD safety
        belt; it does not require any sibling model.
        """
        eps = self.eagle_eps_uniform
        if eps <= 0.0:
            if self.smc_draft_temperature > 0:
                idx = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
            else:
                idx = torch.argmax(log_probs, dim=-1)
            log_q = log_probs.gather(1, idx.unsqueeze(1)).squeeze(1)
            return idx, log_q

        bs, vocab = log_probs.shape
        # branch: with prob eps sample uniform, else sample q_eagle
        coin = torch.rand(bs, device=log_probs.device) < eps
        eagle_idx = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
        unif_idx = torch.randint(0, vocab, (bs,), device=log_probs.device)
        idx = torch.where(coin, unif_idx, eagle_idx)
        # exact log q_mix(idx) = logaddexp(log(1-eps) + log_q_eagle(idx),
        #                                  log(eps) - log(V))
        log_q_eagle = log_probs.gather(1, idx.unsqueeze(1)).squeeze(1)
        log_eps_v = self._eagle_eps_log_eps - math.log(vocab)
        log_q_mix = torch.logaddexp(
            log_q_eagle + self._eagle_eps_log1me,
            torch.full_like(log_q_eagle, log_eps_v),
        )
        return idx, log_q_mix

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
        if self.is_eagle3:
            return self._forward_extend_eagle3(batch)

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

    def _forward_extend_eagle3(self, batch: ModelWorkerBatch):
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        bs = len(batch.seq_lens)
        chm = CaptureHiddenMode.FULL if self.eagle_use_aux else CaptureHiddenMode.LAST
        batch.capture_hidden_mode = chm
        score_result = self._target_worker.forward_batch_generation(batch)
        target_h = score_result.logits_output.hidden_states.to(self._eagle3_hidden_dtype)

        draft_batch = self._make_clean_batch(batch)
        shifted_ids = batch.input_ids.clone()
        pt = 0
        for i, extend_len in enumerate(batch.extend_seq_lens):
            extend_len = int(extend_len)
            if extend_len > 0:
                shifted_ids[pt : pt + extend_len - 1] = batch.input_ids[
                    pt + 1 : pt + extend_len
                ]
                shifted_ids[pt + extend_len - 1] = score_result.next_token_ids[i]
            pt += extend_len

        draft_batch.input_ids = shifted_ids
        draft_batch.spec_info = EagleDraftInput(
            hidden_states=target_h,
            verified_id=score_result.next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        draft_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        draft_result = self._draft_worker.forward_batch_generation(draft_batch)

        h0 = draft_result.logits_output.hidden_states.contiguous().to(
            self._eagle3_hidden_dtype
        )
        logits = draft_result.logits_output.next_token_logits
        scaled = logits / self.smc_draft_temperature
        probs = torch.softmax(scaled, dim=-1)
        topk_p, topk_index = torch.topk(probs, self.eagle_topk, dim=-1)

        log_probs = torch.log_softmax(scaled, dim=-1)
        first_idx, first_lp = self._eagle_sample_with_eps_mix(log_probs)
        first_target_id = (
            self.hot_token_id[first_idx] if self.hot_token_id is not None else first_idx
        )

        score_result.next_draft_input = SMCDraftInput(
            verified_id=score_result.next_token_ids,
            num_tokens_per_req=self.speculative_num_draft_tokens,
            hidden_state=h0,
            first_draft_token_id=first_target_id,
            first_draft_logprob=first_lp,
            eagle_topk_p=topk_p,
            eagle_topk_index=topk_index,
        )
        score_result.accept_lens = torch.zeros(bs, dtype=torch.int32, device=self.device)
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

        if self.is_eagle3_chain:
            return self._forward_decode_eagle3_chain(
                batch,
                draft_input,
                ctx,
                draft_fb,
                cache_locs,
                all_positions,
                all_seq_lens,
                current_stream,
            )
        if self.is_eagle3_tree:
            return self._forward_decode_eagle3_tree(
                batch,
                draft_input,
                ctx,
                draft_fb,
                cache_locs,
                all_positions,
                all_seq_lens,
                current_stream,
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
        draft_topk_ids_steps: list[torch.Tensor] = []
        draft_topk_logp_steps: list[torch.Tensor] = []
        diag_topk = max(int(getattr(self, "eagle_topk", 1)), 1)
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

                if self.eagle_diag_path is not None:
                    step_topk_logp, step_topk_ids = torch.topk(
                        log_probs, diag_topk, dim=-1
                    )
                    draft_topk_ids_steps.append(step_topk_ids)
                    draft_topk_logp_steps.append(step_topk_logp)

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
        # NOTE: must apply smc_target_temperature so target log-p matches the
        # temperature regime the draft sampled at. Dropping this divisor was
        # a bug caught by scripts/verify_eagle_integration.py Test C
        # (pass-through with draft=target gave median |logprob_diff|=0.056
        # instead of 0). The EAGLE chain path at line 1462 applies it; the
        # dense path now matches.
        score_log_probs = torch.log_softmax(
            score_logits / self.smc_target_temperature, dim=-1
        )
        score_log_probs = score_log_probs.reshape(bs, gamma + 1, -1)
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)
        score_logprobs_stacked = score_log_probs[:, :gamma, :].gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)

        if self.eagle_diag_path is not None and len(draft_topk_ids_steps) == gamma:
            self._write_eagle_chain_diagnostics(
                target_log_probs=score_log_probs,
                target_tokens=target_tokens,
                target_logprobs=score_logprobs_stacked,
                draft_logprobs=draft_logprobs_stacked,
                draft_topk_ids_steps=draft_topk_ids_steps,
                draft_topk_logp_steps=draft_topk_logp_steps,
            )

        # ---- 5. Logprob diff ----
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

    # ─────────────────────────────────────────────────────────
    #  EAGLE3 DECODE MODES
    # ─────────────────────────────────────────────────────────

    def _target_score_x0_for_eagle_tree_probe(
        self,
        batch: ModelWorkerBatch,
        draft_input: SMCDraftInput,
        ctx: SMCDecodeContext,
        cache_locs: torch.Tensor,
    ) -> torch.Tensor:
        """Score x1 candidates from the target at the x0 position.

        This is the first spike of tree-SMC: use the EAGLE top-k branch set
        for the first edge, then continue with the known-correct EAGLE chain
        machinery.  It gives us immediate signal on whether target-corrected
        branching helps without taking on full tree KV compaction yet.
        """
        bs = len(ctx.orig_seq_lens)
        # Reuse the normal SMC verify preparation path for backend metadata
        # compatibility. We only need logits after x0; trailing dummy tokens are
        # overwritten later by the real verification pass.
        all_tokens = [draft_input.verified_id] + [
            draft_input.verified_id for _ in range(ctx.gamma)
        ]
        target_runner = self._target_worker.model_runner
        graph_runner = getattr(target_runner, "graph_runner", None)
        target_runner.graph_runner = None
        try:
            verify_forward_batch, _ = ctx.prepare_for_verify(
                self.req_to_token_pool,
                batch,
                self._target_worker,
                all_tokens,
                cache_locs,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            result = self._target_worker.forward_batch_generation(
                model_worker_batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
                skip_attn_backend_init=True,
            )
        finally:
            target_runner.graph_runner = graph_runner
        logits = result.logits_output.next_token_logits
        assert logits.shape[0] == bs * (ctx.gamma + 1)
        logits = logits.reshape(bs, ctx.gamma + 1, -1)[:, 0, :]
        return torch.log_softmax(logits / self.smc_target_temperature, dim=-1)

    def _select_eagle_tree_first_token(
        self,
        batch: ModelWorkerBatch,
        draft_input: SMCDraftInput,
        ctx: SMCDecodeContext,
        cache_locs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if draft_input.eagle_topk_p is None or draft_input.eagle_topk_index is None:
            raise RuntimeError("EAGLE tree mode requires top-k state from prefill/rewrite.")

        topk_p = torch.clamp(draft_input.eagle_topk_p.to(torch.float32), min=1e-20)
        topk_index = draft_input.eagle_topk_index
        topk_target_id = (
            self.hot_token_id[topk_index] if self.hot_token_id is not None else topk_index
        )

        if self.smc_draft_mode == "eagle3_tree_probe":
            target_log_probs = self._target_score_x0_for_eagle_tree_probe(
                batch, draft_input, ctx, cache_locs
            )
            branch_target_lp = target_log_probs.gather(1, topk_target_id)
            chosen = torch.argmax(branch_target_lp, dim=1)
        else:
            probs = topk_p / topk_p.sum(dim=1, keepdim=True)
            chosen = torch.multinomial(probs, num_samples=1).squeeze(1)

        row = torch.arange(topk_index.shape[0], device=topk_index.device)
        chosen_target_id = topk_target_id[row, chosen]
        chosen_lp = torch.log(topk_p[row, chosen])

        return chosen_target_id, chosen_lp

    def _forward_decode_eagle3_tree(
        self,
        batch: ModelWorkerBatch,
        draft_input: SMCDraftInput,
        ctx: SMCDecodeContext,
        draft_fb,
        cache_locs: torch.Tensor,
        all_positions: torch.Tensor,
        all_seq_lens: torch.Tensor,
        current_stream,
    ):
        if self.is_eagle3_full_tree:
            tree_choice = self._select_eagle3_full_tree_path(
                batch, draft_input, ctx
            )
            if tree_choice is not None:
                path_tokens, path_logq, diagnostics = tree_choice
                self._write_eagle_tree_diagnostics(diagnostics)
                return self._forward_decode_eagle3_path(
                    batch,
                    draft_input,
                    ctx,
                    draft_fb,
                    cache_locs,
                    all_positions,
                    all_seq_lens,
                    current_stream,
                    path_tokens,
                    path_logq,
                )

        x1, x1_lp = self._select_eagle_tree_first_token(
            batch, draft_input, ctx, cache_locs
        )
        tree_seed = dataclasses.replace(
            draft_input,
            first_draft_token_id=x1,
            first_draft_logprob=x1_lp,
        )
        return self._forward_decode_eagle3_chain(
            batch,
            tree_seed,
            ctx,
            draft_fb,
            cache_locs,
            all_positions,
            all_seq_lens,
            current_stream,
        )

    def _ensure_eagle_diag_file(self):
        if self.eagle_diag_path is None:
            return None
        if self._eagle_diag_file is not None:
            return self._eagle_diag_file

        path = self.eagle_diag_path
        if path.endswith(".jsonl"):
            file_path = path
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(
                path, f"eagle_tree_rank{self.tp_rank}_pid{os.getpid()}.jsonl"
            )
        self._eagle_diag_file = open(file_path, "a", buffering=1)
        return self._eagle_diag_file

    def _write_eagle_diagnostics(self, diagnostics: dict) -> None:
        self._eagle_diag_counter += 1
        diagnostics["cycle"] = self._eagle_diag_counter
        diagnostics["time"] = time.time()
        fh = self._ensure_eagle_diag_file()
        if fh is not None:
            fh.write(json.dumps(diagnostics, sort_keys=True) + "\n")

    def _write_eagle_tree_diagnostics(self, diagnostics: dict) -> None:
        self._write_eagle_diagnostics(diagnostics)

    def _write_eagle_chain_diagnostics(
        self,
        *,
        target_log_probs: torch.Tensor,
        target_tokens: torch.Tensor,
        target_logprobs: torch.Tensor,
        draft_logprobs: torch.Tensor,
        draft_topk_ids_steps: list[torch.Tensor],
        draft_topk_logp_steps: list[torch.Tensor],
    ) -> None:
        if self.eagle_diag_path is None:
            return

        gamma = target_tokens.shape[1]
        topk = min(len(draft_topk_ids_steps[0][0]), target_log_probs.shape[-1])
        target_topk_logp, target_topk_ids = torch.topk(
            target_log_probs[:, :gamma, :], topk, dim=-1
        )
        draft_topk_ids = torch.stack(draft_topk_ids_steps[:gamma], dim=1)
        draft_topk_logp = torch.stack(draft_topk_logp_steps[:gamma], dim=1)
        overlap = (
            draft_topk_ids[:, :, :, None] == target_topk_ids[:, :, None, :]
        ).any(dim=-1).float()

        token_target_rank = (
            (target_log_probs[:, :gamma, :] > target_logprobs[:, :, None])
            .sum(dim=-1)
            .to(torch.float32)
            + 1.0
        )
        token_in_draft_topk = (
            draft_topk_ids == target_tokens[:, :, None]
        ).any(dim=-1).float()
        token_in_target_topk = (
            target_topk_ids == target_tokens[:, :, None]
        ).any(dim=-1).float()

        rec = {
            "mode": self.smc_draft_mode,
            "bs": int(target_tokens.shape[0]),
            "gamma": int(gamma),
            "topk": int(topk),
            "draft_target_topk_overlap_mean": float(overlap.mean().item()),
            "sample_in_draft_topk_mean": float(token_in_draft_topk.mean().item()),
            "sample_in_target_topk_mean": float(token_in_target_topk.mean().item()),
            "sample_target_rank_mean": float(token_target_rank.mean().item()),
            "sample_target_rank_median": float(
                token_target_rank.flatten().median().item()
            ),
            "target_lp_mean": float(target_logprobs.mean().item()),
            "draft_lp_mean": float(draft_logprobs.mean().item()),
            "target_minus_draft_lp_mean": float(
                (target_logprobs - draft_logprobs).mean().item()
            ),
            "draft_topk_lp_mean": float(draft_topk_logp.mean().item()),
            "target_topk_lp_mean": float(target_topk_logp.mean().item()),
        }
        self._write_eagle_diagnostics(rec)
        # Side-channel: when SMC_EAGLE_PER_TOKEN_TRACE is set, expose the raw
        # per-batch per-position tensors to the chain-decode caller so it
        # can write a separate trace file with x0 and bonus included.
        # See scripts/verify_eagle_integration.py Test B.
        self._last_chain_per_token = (
            target_tokens.to(torch.int32).cpu(),
            target_logprobs.float().cpu(),
            draft_logprobs.float().cpu(),
        )

    def _build_eagle3_tree(self, batch: ModelWorkerBatch, draft_input: SMCDraftInput):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.speculative.eagle_info import EagleDraftInput
        from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
        from sglang.srt.speculative.spec_utils import (
            assign_draft_cache_locs,
            select_top_k_tokens,
        )
        from sglang.srt.utils import next_power_of_2

        if draft_input.eagle_topk_p is None or draft_input.eagle_topk_index is None:
            raise RuntimeError("Full EAGLE tree requires top-k state.")
        if draft_input.hidden_state is None:
            raise RuntimeError("Full EAGLE tree requires hidden seed state.")

        bs = len(batch.seq_lens)
        topk = self.eagle_topk
        spec_steps = self.gamma
        num_verify_tokens = self.eagle_num_draft_tokens
        if topk <= 1:
            return None

        alloc_state = self.token_to_kv_pool_allocator.backup_state()
        out_cache_loc = self.token_to_kv_pool_allocator.alloc(bs * topk * spec_steps)
        if out_cache_loc is None:
            raise RuntimeError("EAGLE tree draft temporary KV allocation failed.")
        max_draft_end = int((batch.seq_lens_cpu + topk * spec_steps).max().item())
        rows = batch.req_pool_indices
        req_to_token_backup = self.req_to_token_pool.req_to_token[
            rows, :max_draft_end
        ].clone()
        extend_lens = torch.full(
            (bs,), spec_steps, dtype=torch.int32, device=self.device
        )
        num_new_pages_per_topk = torch.ones(
            (bs,), dtype=torch.int32, device=self.device
        )
        assign_draft_cache_locs[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            None,
            None,
            None,
            0,
            self.req_to_token_pool.req_to_token.shape[1],
            topk,
            spec_steps,
            1,
            next_power_of_2(bs),
            next_power_of_2(spec_steps + 1),
        )

        draft_batch = self._make_clean_batch(batch)
        topk_p = torch.clamp(draft_input.eagle_topk_p.to(torch.float32), min=1e-20)
        topk_index = draft_input.eagle_topk_index
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        spec_info = EagleDraftInput(
            hidden_states=draft_input.hidden_state.to(self._eagle3_hidden_dtype),
            verified_id=draft_input.verified_id,
            topk_p=topk_p,
            topk_index=topk_index,
            num_tokens_per_req=topk,
            num_tokens_for_logprob_per_req=topk,
        )
        spec_info.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
        draft_batch.spec_info = spec_info
        draft_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        draft_batch.out_cache_loc = out_cache_loc
        draft_batch.seq_lens = batch.seq_lens
        draft_batch.seq_lens_cpu = batch.seq_lens_cpu
        draft_batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())

        forward_batch = ForwardBatch.init_new(draft_batch, self.draft_runner)
        forward_batch.can_run_dp_cuda_graph = False
        if spec_steps > 1:
            self.draft_attn_backend.init_forward_metadata(forward_batch)

        out_cache_by_step = out_cache_loc.reshape(bs, topk, spec_steps)
        out_cache_by_step = out_cache_by_step.permute(2, 0, 1).reshape(
            spec_steps, -1
        )

        score_list = []
        token_list = []
        parents_list = []
        scores = None
        hidden_states = spec_info.hidden_states

        try:
            for i in range(spec_steps):
                input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                    i, topk_p, topk_index, hidden_states, scores, topk
                )
                score_list.append(tree_info[0])
                token_list.append(tree_info[1])
                parents_list.append(tree_info[2])

                if i == spec_steps - 1:
                    break

                forward_batch.input_ids = input_ids
                forward_batch.out_cache_loc = out_cache_by_step[i]
                forward_batch.positions.add_(1)
                forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
                spec_info.hidden_states = hidden_states

                logits_output = self.draft_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output
                scaled = logits_output.next_token_logits / self.smc_draft_temperature
                probs = torch.softmax(scaled, dim=-1)
                topk_p, topk_index = torch.topk(probs, topk, dim=-1)
                if self.hot_token_id is not None:
                    topk_index = self.hot_token_id[topk_index]
                hidden_states = logits_output.hidden_states
        finally:
            self.req_to_token_pool.req_to_token[rows, :max_draft_end] = (
                req_to_token_backup
            )
            self.token_to_kv_pool_allocator.restore_state(alloc_state)

        score_flat = torch.cat(score_list, dim=1).flatten(1)
        token_flat = torch.cat(token_list, dim=1)
        top_scores = torch.topk(score_flat, num_verify_tokens - 1, dim=-1)
        top_scores_index = torch.sort(top_scores.indices).values
        draft_tokens = torch.gather(token_flat, index=top_scores_index, dim=1)
        node_scores = torch.cat(
            [
                torch.ones((bs, 1), dtype=torch.float32, device=self.device),
                torch.gather(score_flat, 1, top_scores_index).to(torch.float32),
            ],
            dim=1,
        )

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            parent_list = torch.empty(bs, 0, dtype=torch.long, device=self.device)

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            int(batch.seq_lens_cpu.sum().item()),
            topk,
            spec_steps,
            num_verify_tokens,
        )

        return {
            "draft_tokens": draft_tokens,
            "node_scores": node_scores,
            "tree_mask": tree_mask,
            "positions": position,
            "retrive_index": retrive_index,
            "retrive_next_token": retrive_next_token,
            "retrive_next_sibling": retrive_next_sibling,
            "top_scores_index": top_scores_index,
            "parent_list": parent_list,
        }

    def _target_verify_eagle3_tree(
        self,
        batch: ModelWorkerBatch,
        ctx: SMCDecodeContext,
        tree: dict,
    ):
        import copy

        from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        num_verify_tokens = self.eagle_num_draft_tokens
        bs = len(ctx.orig_seq_lens)
        alloc_state = self.token_to_kv_pool_allocator.backup_state()
        out_cache_loc = self.token_to_kv_pool_allocator.alloc(bs * num_verify_tokens)
        if out_cache_loc is None:
            raise RuntimeError("EAGLE tree target temporary KV allocation failed.")

        max_end = int((ctx.orig_seq_lens_cpu + num_verify_tokens).max().item())
        rows = batch.req_pool_indices
        req_to_token_backup = self.req_to_token_pool.req_to_token[
            rows, :max_end
        ].clone()

        assign_req_to_token_pool_func(
            rows,
            self.req_to_token_pool.req_to_token,
            ctx.orig_seq_lens.to(torch.int32),
            (ctx.orig_seq_lens + num_verify_tokens).to(torch.int32),
            out_cache_loc,
            bs,
        )

        spec_info = EagleVerifyInput(
            draft_token=tree["draft_tokens"],
            custom_mask=tree["tree_mask"],
            positions=tree["positions"],
            retrive_index=tree["retrive_index"],
            retrive_next_token=tree["retrive_next_token"],
            retrive_next_sibling=tree["retrive_next_sibling"],
            retrive_cum_len=None,
            spec_steps=self.gamma,
            topk=self.eagle_topk,
            draft_token_num=num_verify_tokens,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=ctx.orig_seq_lens_sum,
            seq_lens_cpu=ctx.orig_seq_lens_cpu,
            num_tokens_per_req=num_verify_tokens,
        )

        verify_batch = copy.copy(batch)
        verify_batch.input_ids = tree["draft_tokens"]
        verify_batch.out_cache_loc = out_cache_loc
        verify_batch.seq_lens = ctx.orig_seq_lens
        verify_batch.seq_lens_cpu = ctx.orig_seq_lens_cpu
        verify_batch.seq_lens_sum = ctx.orig_seq_lens_sum
        verify_batch.spec_info = spec_info
        verify_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        verify_batch.forward_mode = ForwardMode.TARGET_VERIFY

        fb = ForwardBatch.init_new(verify_batch, self._target_worker.model_runner)
        graph_runner = getattr(self._target_worker.model_runner, "graph_runner", None)
        self._target_worker.model_runner.graph_runner = None
        try:
            self._target_worker.model_runner.attn_backend.init_forward_metadata(fb)
            result = self._target_worker.forward_batch_generation(
                model_worker_batch=None,
                forward_batch=fb,
                is_verify=True,
                skip_attn_backend_init=True,
            )
        finally:
            self._target_worker.model_runner.graph_runner = graph_runner
            self.req_to_token_pool.req_to_token[rows, :max_end] = req_to_token_backup
            self.token_to_kv_pool_allocator.restore_state(alloc_state)

        return result.logits_output.next_token_logits

    @staticmethod
    def _enumerate_eagle_paths_cpu(next_child, next_sibling, gamma: int):
        paths = []

        def rec(node: int, path: list[int]):
            child = int(next_child[node])
            if child < 0 or len(path) - 1 >= gamma:
                paths.append(path[:])
                return
            while child >= 0:
                rec(child, path + [child])
                child = int(next_sibling[child])

        rec(0, [0])
        return paths

    def _select_eagle3_full_tree_path(
        self,
        batch: ModelWorkerBatch,
        draft_input: SMCDraftInput,
        ctx: SMCDecodeContext,
    ):
        tree = self._build_eagle3_tree(batch, draft_input)
        if tree is None:
            return None

        target_logits = self._target_verify_eagle3_tree(batch, ctx, tree)
        bs = len(ctx.orig_seq_lens)
        num_verify_tokens = self.eagle_num_draft_tokens
        target_log_probs = torch.log_softmax(
            target_logits / self.smc_target_temperature, dim=-1
        ).reshape(bs, num_verify_tokens, -1)
        candidates = tree["draft_tokens"].reshape(bs, num_verify_tokens)
        node_scores = torch.clamp(tree["node_scores"], min=1e-30)
        node_logq = torch.log(node_scores)

        selected_tokens = []
        selected_logq = []
        path_counts = []
        exact_counts = []
        oracle_target_scores = []
        draft_target_scores = []
        oracle_logq_scores = []
        draft_logq_scores = []
        top1_edge_hits = []

        for b in range(bs):
            paths = self._enumerate_eagle_paths_cpu(
                tree["retrive_next_token"][b].detach().cpu().tolist(),
                tree["retrive_next_sibling"][b].detach().cpu().tolist(),
                self.gamma,
            )
            exact_paths = [p for p in paths if len(p) - 1 >= self.gamma]
            usable_paths = exact_paths or paths
            if not usable_paths:
                return None

            best_target_score = None
            best_target_path = None
            best_draft_target_score = None
            best_draft_logq = None
            best_draft_path = None
            path_logqs = []
            path_target_scores = []

            for path in usable_paths:
                path = path[: self.gamma + 1]
                target_score = torch.zeros((), dtype=torch.float32, device=self.device)
                for parent, child in zip(path[:-1], path[1:], strict=True):
                    token = candidates[b, child]
                    target_score = target_score + target_log_probs[b, parent, token]
                logq = node_logq[b, path[-1]]
                path_logqs.append(logq)
                path_target_scores.append(target_score)
                if best_target_score is None or target_score > best_target_score:
                    best_target_score = target_score
                    best_target_path = path
                if best_draft_logq is None or logq > best_draft_logq:
                    best_draft_logq = logq
                    best_draft_target_score = target_score
                    best_draft_path = path

            if self.smc_draft_mode == "eagle3_tree_oracle":
                chosen_path = best_target_path
            else:
                logqs = torch.stack(path_logqs)
                probs = torch.softmax(logqs - torch.max(logqs), dim=0)
                chosen_idx = int(torch.multinomial(probs, num_samples=1).item())
                chosen_path = usable_paths[chosen_idx][: self.gamma + 1]

            if len(chosen_path) - 1 < self.gamma:
                return None

            token_ids = candidates[
                b,
                torch.tensor(
                    chosen_path[1 : self.gamma + 1],
                    dtype=torch.long,
                    device=self.device,
                ),
            ]
            selected_tokens.append(token_ids)
            if self.smc_draft_mode == "eagle3_tree_oracle":
                chosen_logq = node_logq[b, chosen_path[-1]]
            else:
                logqs = torch.stack(path_logqs)
                chosen_logq = logqs[chosen_idx] - torch.logsumexp(logqs, dim=0)
            selected_logq.append(chosen_logq)

            parent_best = torch.argmax(target_log_probs[b], dim=-1)
            hits = []
            for p in range(num_verify_tokens):
                children = []
                child = int(tree["retrive_next_token"][b, p].item())
                while child >= 0:
                    children.append(child)
                    child = int(tree["retrive_next_sibling"][b, child].item())
                if children:
                    child_tokens = candidates[
                        b,
                        torch.tensor(children, dtype=torch.long, device=self.device),
                    ]
                    hits.append((child_tokens == parent_best[p]).any().float())
            top1_edge_hit = (
                torch.stack(hits).mean() if hits else torch.zeros((), device=self.device)
            )

            path_counts.append(len(paths))
            exact_counts.append(len(exact_paths))
            oracle_target_scores.append(best_target_score)
            draft_target_scores.append(best_draft_target_score)
            oracle_logq_scores.append(node_logq[b, best_target_path[-1]])
            draft_logq_scores.append(best_draft_logq)
            top1_edge_hits.append(top1_edge_hit)

        path_tokens = torch.stack(selected_tokens, dim=0).contiguous()
        path_logq = torch.stack(selected_logq).to(torch.float32)
        oracle_target = torch.stack(oracle_target_scores)
        draft_target = torch.stack(draft_target_scores)
        oracle_logq = torch.stack(oracle_logq_scores)
        draft_logq = torch.stack(draft_logq_scores)
        hit_rate = torch.stack(top1_edge_hits)
        diagnostics = {
            "mode": self.smc_draft_mode,
            "bs": bs,
            "gamma": self.gamma,
            "topk": self.eagle_topk,
            "num_verify_tokens": num_verify_tokens,
            "mean_path_count": float(sum(path_counts) / max(len(path_counts), 1)),
            "mean_exact_path_count": float(sum(exact_counts) / max(len(exact_counts), 1)),
            "oracle_target_logp_mean": float(oracle_target.mean().item()),
            "draft_best_target_logp_mean": float(draft_target.mean().item()),
            "oracle_minus_draft_target_logp_mean": float(
                (oracle_target - draft_target).mean().item()
            ),
            "oracle_draft_logq_mean": float(oracle_logq.mean().item()),
            "draft_best_logq_mean": float(draft_logq.mean().item()),
            "target_top1_edge_hit_mean": float(hit_rate.mean().item()),
        }
        return path_tokens, path_logq, diagnostics

    def _forward_decode_eagle3_path(
        self,
        batch: ModelWorkerBatch,
        draft_input: SMCDraftInput,
        ctx: SMCDecodeContext,
        draft_fb,
        cache_locs: torch.Tensor,
        all_positions: torch.Tensor,
        all_seq_lens: torch.Tensor,
        current_stream,
        path_tokens: torch.Tensor,
        path_logq: torch.Tensor,
    ):
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma
        x0 = draft_input.verified_id
        all_tokens = [x0] + [
            path_tokens[:, i].contiguous() for i in range(path_tokens.shape[1])
        ]

        verify_forward_batch, can_run_cuda_graph = ctx.prepare_for_verify(
            self.req_to_token_pool,
            batch,
            self._target_worker,
            all_tokens,
            cache_locs,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        score_result = self._target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )

        score_logits = score_result.logits_output.next_token_logits
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]}, "
            f"expected {expected_rows}"
        )

        score_log_probs_all = torch.log_softmax(
            score_logits / self.smc_target_temperature, dim=-1
        ).reshape(bs, gamma + 1, -1)
        score_logprobs_stacked = score_log_probs_all[:, :gamma, :].gather(
            2, path_tokens.unsqueeze(2)
        ).squeeze(2)
        logprob_diff = score_logprobs_stacked.sum(dim=1) - path_logq

        bonus_logits = score_logits.reshape(bs, gamma + 1, -1)[:, -1, :]
        bonus_log_probs = torch.log_softmax(
            bonus_logits / self.smc_target_temperature, dim=-1
        )
        if self.smc_target_temperature > 0:
            bonus = torch.multinomial(bonus_log_probs.exp(), num_samples=1).squeeze(-1)
        else:
            bonus = torch.argmax(bonus_logits, dim=-1)

        output_token_ids = torch.cat([path_tokens, bonus.unsqueeze(1)], dim=1)
        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=self.device
        )

        h_all = score_result.logits_output.hidden_states
        if h_all is None:
            raise RuntimeError("EAGLE3 path verify must capture target hidden states.")
        target_h_steps = h_all.reshape(bs, gamma + 1, h_all.shape[-1]).to(
            self._eagle3_hidden_dtype
        )

        accepted_tokens = output_token_ids
        last_hidden = None
        last_logits = None
        for step in range(gamma + 1):
            tok = accepted_tokens[:, step].contiguous()
            step_seq_lens = all_seq_lens[:, step].contiguous()
            step_spec = EagleDraftInput(
                hidden_states=target_h_steps[:, step, :].contiguous(),
                verified_id=tok,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )
            self._populate_eagle_kv_metadata(
                step_spec, batch.req_pool_indices, step_seq_lens
            )
            draft_fb.input_ids = tok
            draft_fb.positions = all_positions[:, step].contiguous()
            draft_fb.out_cache_loc = cache_locs[:, step].contiguous()
            draft_fb.spec_info = step_spec
            draft_fb.capture_hidden_mode = CaptureHiddenMode.LAST
            draft_fb.seq_lens = step_seq_lens
            draft_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * (step + 1)
            draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + (step + 1)
            out = self.draft_runner.forward(draft_fb).logits_output
            last_hidden = out.hidden_states
            last_logits = out.next_token_logits

        assert last_hidden is not None and last_logits is not None
        next_hidden_state = last_hidden.contiguous().to(self._eagle3_hidden_dtype)

        nxt_scaled = last_logits / self.smc_draft_temperature
        nxt_probs = torch.softmax(nxt_scaled, dim=-1)
        next_topk_p, next_topk_index = torch.topk(nxt_probs, self.eagle_topk, dim=-1)
        nxt_log_probs = torch.log_softmax(nxt_scaled, dim=-1)
        if self.smc_draft_temperature > 0:
            nxt_idx = torch.multinomial(nxt_probs, num_samples=1).squeeze(-1)
        else:
            nxt_idx = torch.argmax(last_logits, dim=-1)
        nxt_x1_logprob = nxt_log_probs.gather(1, nxt_idx.unsqueeze(1)).squeeze(1)
        nxt_x1_target_id = (
            self.hot_token_id[nxt_idx] if self.hot_token_id is not None else nxt_idx
        )

        next_token_ids.record_stream(current_stream)
        accept_lens.record_stream(current_stream)
        bonus.record_stream(current_stream)
        logprob_diff.record_stream(current_stream)

        next_draft_input = SMCDraftInput(
            verified_id=bonus,
            logprob_diff=logprob_diff,
            num_tokens_per_req=self.speculative_num_draft_tokens,
            hidden_state=next_hidden_state,
            first_draft_token_id=nxt_x1_target_id,
            first_draft_logprob=nxt_x1_logprob,
            eagle_topk_p=next_topk_p,
            eagle_topk_index=next_topk_index,
        )

        return GenerationBatchResult(
            logits_output=score_result.logits_output,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            next_draft_input=next_draft_input,
            logprob_diff=logprob_diff,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _forward_decode_eagle3_chain(
        self,
        batch: ModelWorkerBatch,
        draft_input: SMCDraftInput,
        ctx: SMCDecodeContext,
        draft_fb,
        cache_locs: torch.Tensor,
        all_positions: torch.Tensor,
        all_seq_lens: torch.Tensor,
        current_stream,
    ):
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma
        assert gamma >= 1, "EAGLE3 decode requires gamma >= 1"
        if (
            draft_input.first_draft_token_id is None
            or draft_input.first_draft_logprob is None
            or draft_input.hidden_state is None
        ):
            raise RuntimeError("EAGLE3 decode missing first-token or hidden seed state.")

        x0 = draft_input.verified_id
        x1 = draft_input.first_draft_token_id
        x1_logprob = draft_input.first_draft_logprob
        current_hidden = draft_input.hidden_state.to(self._eagle3_hidden_dtype)

        all_tokens = [x0, x1]
        draft_logprobs = [x1_logprob]
        first_topk_p = torch.clamp(draft_input.eagle_topk_p.to(torch.float32), min=1e-30)
        first_topk_ids = draft_input.eagle_topk_index
        if self.hot_token_id is not None:
            first_topk_ids = self.hot_token_id[first_topk_ids]
        draft_topk_ids_steps = [first_topk_ids]
        draft_topk_logp_steps = [torch.log(first_topk_p)]
        current_ids = x1

        for step in range(gamma - 1):
            step_seq_lens = all_seq_lens[:, step].contiguous()
            step_spec = EagleDraftInput(
                hidden_states=current_hidden,
                verified_id=current_ids,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )
            self._populate_eagle_kv_metadata(
                step_spec, batch.req_pool_indices, step_seq_lens
            )
            draft_fb.input_ids = current_ids
            draft_fb.positions = all_positions[:, step].contiguous()
            draft_fb.out_cache_loc = cache_locs[:, step].contiguous()
            draft_fb.spec_info = step_spec
            draft_fb.capture_hidden_mode = CaptureHiddenMode.LAST
            draft_fb.seq_lens = step_seq_lens
            draft_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * (step + 1)
            draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + (step + 1)
            draft_out = self.draft_runner.forward(draft_fb)

            logits = draft_out.logits_output.next_token_logits
            new_hidden = draft_out.logits_output.hidden_states
            scaled = logits / self.smc_draft_temperature
            log_probs = torch.log_softmax(scaled, dim=-1)
            step_topk_logp, step_topk_ids = torch.topk(
                log_probs, draft_topk_ids_steps[0].shape[-1], dim=-1
            )
            if self.hot_token_id is not None:
                step_topk_ids = self.hot_token_id[step_topk_ids]
            draft_topk_ids_steps.append(step_topk_ids)
            draft_topk_logp_steps.append(step_topk_logp)
            draft_idx, token_logprob = self._eagle_sample_with_eps_mix(log_probs)
            next_token = (
                self.hot_token_id[draft_idx]
                if self.hot_token_id is not None
                else draft_idx
            )

            all_tokens.append(next_token)
            draft_logprobs.append(token_logprob)
            current_ids = next_token
            current_hidden = new_hidden

        draft_logprobs_stacked = torch.stack(draft_logprobs, dim=1)

        verify_forward_batch, can_run_cuda_graph = ctx.prepare_for_verify(
            self.req_to_token_pool,
            batch,
            self._target_worker,
            all_tokens,
            cache_locs,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        score_result = self._target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )

        score_logits = score_result.logits_output.next_token_logits
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]}, "
            f"expected {expected_rows}"
        )

        score_log_probs_all = torch.log_softmax(
            score_logits / self.smc_target_temperature, dim=-1
        ).reshape(bs, gamma + 1, -1)
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)
        score_logprobs_stacked = score_log_probs_all[:, :gamma, :].gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)
        logprob_diff = (score_logprobs_stacked - draft_logprobs_stacked).sum(dim=1)
        self._write_eagle_chain_diagnostics(
            target_log_probs=score_log_probs_all,
            target_tokens=target_tokens,
            target_logprobs=score_logprobs_stacked,
            draft_logprobs=draft_logprobs_stacked,
            draft_topk_ids_steps=draft_topk_ids_steps,
            draft_topk_logp_steps=draft_topk_logp_steps,
        )

        bonus_logits = score_logits.reshape(bs, gamma + 1, -1)[:, -1, :]
        bonus_log_probs = torch.log_softmax(
            bonus_logits / self.smc_target_temperature, dim=-1
        )
        if self.smc_target_temperature > 0:
            bonus = torch.multinomial(bonus_log_probs.exp(), num_samples=1).squeeze(-1)
        else:
            bonus = torch.argmax(bonus_logits, dim=-1)

        # Test B per-token trace: write x0, target_tokens, target_lp,
        # draft_lp, and bonus per cycle to a separate JSONL when
        # SMC_EAGLE_PER_TOKEN_TRACE is set. Side-channel buffer was
        # populated by _write_eagle_chain_diagnostics above.
        per_tok = getattr(self, "_last_chain_per_token", None)
        if (
            per_tok is not None
            and os.environ.get("SMC_EAGLE_PER_TOKEN_TRACE")
            and self.eagle_diag_path is not None
        ):
            tt_cpu, tlp_cpu, dlp_cpu = per_tok
            x0_cpu = all_tokens[0].to(torch.int32).cpu().tolist()
            bonus_cpu = bonus.to(torch.int32).cpu().tolist()
            trace_path = str(self.eagle_diag_path) + ".pertoken.jsonl"
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            with open(trace_path, "a") as f:
                f.write(json.dumps({
                    "cycle": int(self._eagle_diag_counter),
                    "bs": int(tt_cpu.shape[0]),
                    "gamma": int(tt_cpu.shape[1]),
                    "x0": x0_cpu,
                    "target_tokens": tt_cpu.tolist(),
                    "target_logprobs": tlp_cpu.tolist(),
                    "draft_logprobs": dlp_cpu.tolist(),
                    "bonus": bonus_cpu,
                }) + "\n")
            self._last_chain_per_token = None

        output_token_ids = torch.stack(all_tokens[1 : gamma + 1] + [bonus], dim=1)
        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=self.device
        )

        h_all = score_result.logits_output.hidden_states
        if h_all is None:
            raise RuntimeError("EAGLE3 verify must capture target hidden states.")
        target_h_steps = h_all.reshape(bs, gamma + 1, h_all.shape[-1]).to(
            self._eagle3_hidden_dtype
        )

        accepted_tokens = output_token_ids
        last_hidden = None
        last_logits = None
        for step in range(gamma + 1):
            tok = accepted_tokens[:, step].contiguous()
            step_seq_lens = all_seq_lens[:, step].contiguous()
            step_spec = EagleDraftInput(
                hidden_states=target_h_steps[:, step, :].contiguous(),
                verified_id=tok,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )
            self._populate_eagle_kv_metadata(
                step_spec, batch.req_pool_indices, step_seq_lens
            )
            draft_fb.input_ids = tok
            draft_fb.positions = all_positions[:, step].contiguous()
            draft_fb.out_cache_loc = cache_locs[:, step].contiguous()
            draft_fb.spec_info = step_spec
            draft_fb.capture_hidden_mode = CaptureHiddenMode.LAST
            draft_fb.seq_lens = step_seq_lens
            draft_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * (step + 1)
            draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + (step + 1)
            out = self.draft_runner.forward(draft_fb).logits_output
            last_hidden = out.hidden_states
            last_logits = out.next_token_logits

        assert last_hidden is not None and last_logits is not None
        next_hidden_state = last_hidden.contiguous().to(self._eagle3_hidden_dtype)

        nxt_scaled = last_logits / self.smc_draft_temperature
        nxt_probs = torch.softmax(nxt_scaled, dim=-1)
        next_topk_p, next_topk_index = torch.topk(nxt_probs, self.eagle_topk, dim=-1)
        nxt_log_probs = torch.log_softmax(nxt_scaled, dim=-1)
        if self.smc_draft_temperature > 0:
            nxt_idx = torch.multinomial(nxt_probs, num_samples=1).squeeze(-1)
        else:
            nxt_idx = torch.argmax(last_logits, dim=-1)
        nxt_x1_logprob = nxt_log_probs.gather(1, nxt_idx.unsqueeze(1)).squeeze(1)
        nxt_x1_target_id = (
            self.hot_token_id[nxt_idx] if self.hot_token_id is not None else nxt_idx
        )

        next_token_ids.record_stream(current_stream)
        accept_lens.record_stream(current_stream)
        bonus.record_stream(current_stream)
        logprob_diff.record_stream(current_stream)

        next_draft_input = SMCDraftInput(
            verified_id=bonus,
            logprob_diff=logprob_diff,
            num_tokens_per_req=self.speculative_num_draft_tokens,
            hidden_state=next_hidden_state,
            first_draft_token_id=nxt_x1_target_id,
            first_draft_logprob=nxt_x1_logprob,
            eagle_topk_p=next_topk_p,
            eagle_topk_index=next_topk_index,
        )

        return GenerationBatchResult(
            logits_output=score_result.logits_output,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            next_draft_input=next_draft_input,
            logprob_diff=logprob_diff,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _populate_eagle_kv_metadata(
        self,
        eagle_input,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Populate EAGLE draft attention metadata for eager Triton decode.

        Native EAGLE builds these fields before ``ForwardBatch.init_new``.
        The SMC worker mutates one reusable ``ForwardBatch`` per step, so the
        spike path fills the two tensors directly.
        """
        device = seq_lens.device
        lens_cpu = seq_lens.to("cpu", dtype=torch.int64).tolist()
        pools_cpu = req_pool_indices.to("cpu", dtype=torch.int64).tolist()
        offsets = [0]
        for length in lens_cpu:
            offsets.append(offsets[-1] + int(length))
        eagle_input.kv_indptr = torch.tensor(offsets, dtype=torch.int32, device=device)
        total = offsets[-1]
        kv_indices = torch.empty(total, dtype=torch.int32, device=device)
        write = 0
        for pool_idx, length in zip(pools_cpu, lens_cpu, strict=True):
            length = int(length)
            if length > 0:
                kv_indices[write : write + length] = self.req_to_token_pool.req_to_token[
                    int(pool_idx), :length
                ].to(torch.int32)
                write += length
        eagle_input.kv_indices = kv_indices

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
