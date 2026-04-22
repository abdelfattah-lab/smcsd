"""SMC Worker v2: standalone worker using SMCDecodeContext API.

Fully self-contained — no inheritance from SMCWorker.
Can replace SMCWorker entirely once the dedicated scheduler adopts v2.

Draft model performs gamma+1 autoregressive decode steps.
Score model performs one extend forward pass on the drafted tokens.
Computes logprob difference between the two models per request.
No rejection — all drafted tokens are accepted.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from smcsd.v2.info import SMCDecodeContext, SMCDraftInputV2
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

logger = logging.getLogger(__name__)


class SMCWorkerV2(BaseSpecWorker):
    """Standalone SMC worker using v2 API (SMCDecodeContext + SMCDraftInputV2)."""

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
        SMCDraftInputV2.ALLOC_LEN_PER_DECODE = self.speculative_num_draft_tokens

        # Override context length of draft model to match score model.
        # For EAGLE3, the draft head has a small max_position_embeddings (2048)
        # in its config, but it operates within the target's full context window.
        # We set the env var to suppress the validation error before constructing
        # the draft TpModelWorker.
        import os
        _eagle3_mode = getattr(server_args, "smc_draft_mode", "dense") == "eagle3"
        _prev_allow_overwrite = os.environ.get("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")
        _prev_dtype = server_args.dtype
        if _eagle3_mode:
            os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
            # Force draft dtype to match target's dtype — EAGLE3 head shares the
            # target's embedding, so mixing fp16 head + bf16 embedding causes
            # dtype mismatches in the midlayer's layernorm/cat/qkv_proj path.
            _torch_to_str = {
                torch.bfloat16: "bfloat16",
                torch.float16: "float16",
                torch.float32: "float32",
            }
            _target_torch_dtype = target_worker.model_runner.model_config.dtype
            server_args.dtype = _torch_to_str.get(
                _target_torch_dtype, "bfloat16"
            )
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

        # Restore env var and dtype after draft worker construction
        if _eagle3_mode:
            if _prev_allow_overwrite is None:
                os.environ.pop("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", None)
            else:
                os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = _prev_allow_overwrite
            server_args.dtype = _prev_dtype

        self.draft_runner = self._draft_worker.model_runner
        self.score_runner = self._target_worker.model_runner

        # EAGLE3 draft mode: share target embed/lm_head with draft head,
        # then disable the draft CUDA graph (EAGLE head runs eagerly).
        self.is_eagle3 = (getattr(server_args, "smc_draft_mode", "dense") == "eagle3")
        self.eagle_use_aux = False

        if self.is_eagle3:
            # Read checkpoint config to determine if aux hidden states are used
            eagle_cfg = getattr(
                self.draft_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux = eagle_cfg.get("use_aux_hidden_state", True)

            # Share target embedding (and lm_head if checkpoint requests it)
            embed, head = self._target_worker.model_runner.model.get_embed_and_head()
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_runner.model.set_embed(embed)

            # Cache the dtype of the fc projection layer (the layer that fuses
            # aux hidden states). This is the correct dtype to cast hidden states
            # to — not next(parameters()) which returns the shared bf16 embedding.
            fc_layer = getattr(self.draft_runner.model.model, "fc", None)
            if fc_layer is not None and hasattr(fc_layer, "weight"):
                self._eagle3_hidden_dtype = fc_layer.weight.dtype
            else:
                self._eagle3_hidden_dtype = next(
                    p for p in self.draft_runner.model.parameters()
                    if p is not embed
                ).dtype

            # EAGLE3 models often have a smaller "hot" draft vocab. The draft's
            # logits are indexed in draft-vocab space; hot_token_id[i] gives the
            # corresponding target-vocab token id. We need this mapping to:
            #   1) feed back the correct target-vocab id as input_ids for the
            #      next draft step (since embed_tokens is the target's table),
            #   2) look up target's logprob at the same actual token during
            #      verification, and
            #   3) commit the right token in the output.
            self.hot_token_id = getattr(
                self.draft_runner.model, "hot_token_id", None
            )
            if self.hot_token_id is not None:
                self.hot_token_id = self.hot_token_id.to(embed.device)

        # Multi-step draft attention backend
        from sglang.srt.speculative.draft_utils import DraftBackendFactory

        factory = DraftBackendFactory(
            server_args,
            self.draft_runner,
            topk=1,
            speculative_num_steps=self.gamma + 2,
        )
        self.draft_attn_backend = factory.create_decode_backend()

        # Restore cuda graph and capture for draft model.
        # For EAGLE3 we use a dedicated runner (below) instead of the
        # standard one because the base CudaGraphRunner does not bake
        # EagleDraftInput.hidden_states into its captured input set.
        server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self._draft_cg_runner = None
        if not backup_disable_cuda_graph:
            if self.is_eagle3:
                # EAGLE3: skip init_device_graphs() (which would try to
                # capture without hidden_states) and install our custom
                # lazy-capture runner. Gate via env var for easy A/B.
                if os.environ.get("SMC_EAGLE3_CG", "1") != "0":
                    from smcsd.model_executor.smc_eagle3_draft_cuda_graph_runner import (
                        SMCEagle3DraftCudaGraphRunner,
                    )
                    self._draft_cg_runner = SMCEagle3DraftCudaGraphRunner(self)
            else:
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
        bs = len(batch.seq_lens)

        if self.is_eagle3:
            # Target prefill — request aux hidden states for EAGLE3 head input
            chm = (
                CaptureHiddenMode.FULL if self.eagle_use_aux
                else CaptureHiddenMode.LAST
            )
            batch.capture_hidden_mode = chm
            score_result = self._target_worker.forward_batch_generation(batch)

            # target_h: (total_tokens, 3*hidden_dim) or (total_tokens, hidden_dim)
            # Cast to fc layer dtype (EAGLE3 fc weights may differ from target)
            target_h = score_result.logits_output.hidden_states.to(
                self._eagle3_hidden_dtype
            )

            # Draft prefill with EAGLE3 head using target hidden states.
            # EAGLE convention: at position p draft consumes embed(token_{p+1})
            # paired with target_h[p] — shift input_ids by 1 per request,
            # replacing the last token with the freshly-sampled next_token.
            from sglang.srt.speculative.eagle_info import EagleDraftInput
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
            # Draft prefill: capture LAST position per req so the returned
            # hidden_states is (bs, hidden_dim). This is the draft's hidden at
            # position L-1 (after seeing sampled next_token + target_h[L-1]),
            # which is the seed hidden for decode step 0. The last-position
            # next_token_logits is the draft's prediction of t_{L+1} = x1.
            draft_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            draft_fwd = ForwardBatch.init_new(draft_batch, self.draft_runner)
            draft_logits_out = self.draft_runner.forward(draft_fwd).logits_output
            h0 = draft_logits_out.hidden_states.contiguous()  # (bs, hidden_dim)

            # Pre-sample x1 from draft prefill's last-position logits.
            # This is the FIRST proposed draft token for the next decode cycle.
            # Matches official EAGLE3: decode step 0 uses draft's own
            # prefill-logit prediction, NOT the verified_id (= target's t_L).
            prefill_next_logits = draft_logits_out.next_token_logits  # (bs, draft_vocab)
            scaled = prefill_next_logits / self.smc_draft_temperature
            prefill_log_probs = torch.log_softmax(scaled, dim=-1)
            if self.smc_draft_temperature > 0:
                x1_idx = torch.multinomial(
                    prefill_log_probs.exp(), num_samples=1
                ).squeeze(-1)
            else:
                x1_idx = torch.argmax(prefill_next_logits, dim=-1)
            x1_logprob = prefill_log_probs.gather(
                1, x1_idx.unsqueeze(1)
            ).squeeze(1)
            if self.hot_token_id is not None:
                x1_target_id = self.hot_token_id[x1_idx]
            else:
                x1_target_id = x1_idx

            score_result.next_draft_input = SMCDraftInputV2(
                verified_id=score_result.next_token_ids,
                num_tokens_per_req=self.speculative_num_draft_tokens,
                target_hidden_state=h0,
                first_draft_token_id=x1_target_id,
                first_draft_logprob=x1_logprob,
            )
            score_result.accept_lens = torch.zeros(
                bs, dtype=torch.int32, device=self.device
            )
            return score_result

        # ── Dense path (unchanged) ──
        # Score model prefill
        score_result = self._target_worker.forward_batch_generation(batch)

        # Draft model prefill — samples the first token (x0)
        draft_batch = self._make_clean_batch(batch)
        draft_result = self._draft_worker.forward_batch_generation(draft_batch)

        # Use draft model's sampled token as verified_id
        score_result.next_token_ids = draft_result.next_token_ids

        # x0 KV is NOT written during prefill — first decode writes it.
        score_result.next_draft_input = SMCDraftInputV2(
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

        draft_input: SMCDraftInputV2 = batch.spec_info
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

        if self.is_eagle3:
            return self._forward_decode_eagle3(
                batch, draft_input, ctx,
                draft_fb, cache_locs, all_positions, all_seq_lens,
                current_stream,
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
            capture_hidden_mode=CaptureHiddenMode.NULL,
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
        score_log_probs = torch.log_softmax(score_logits, dim=-1)
        score_log_probs = score_log_probs.reshape(bs, gamma + 1, -1)
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)
        score_logprobs_stacked = score_log_probs[:, :gamma, :].gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)

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
        next_verified_id = bonus

        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=self.device
        )

        next_token_ids.record_stream(current_stream)
        accept_lens.record_stream(current_stream)
        next_verified_id.record_stream(current_stream)
        logprob_diff.record_stream(current_stream)

        next_draft_input = SMCDraftInputV2(
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
    #  EAGLE3 DECODE — matches the official SGLang EAGLE3 flow
    # ─────────────────────────────────────────────────────────

    def _forward_decode_eagle3(
        self,
        batch: ModelWorkerBatch,
        draft_input: SMCDraftInputV2,
        ctx: SMCDecodeContext,
        draft_fb: ForwardBatch,
        cache_locs: torch.Tensor,
        all_positions: torch.Tensor,
        all_seq_lens: torch.Tensor,
        current_stream,
    ):
        """Official-style EAGLE3 decode cycle.

        Per-cycle structure (L = orig_seq_lens, gamma = speculative_num_steps):

          x0   = verified_id  (already committed in prev cycle; target's next
                               token at position L).
          x1   = draft's own prefill/rewrite-sampled prediction of t_{L+1}
                 (pre-sampled from the previous cycle's last-position logits
                 and passed in as draft_input.first_draft_token_id).
          x2.. = produced here by gamma-1 draft forwards, each fusing the
                 previous step's (hidden, prev_token).

        The last rewrite step additionally samples x1 for the NEXT cycle from
        its last-position logits, carried forward on the next SMCDraftInputV2.
        """
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma
        assert gamma >= 1, "EAGLE3 decode requires gamma >= 1"

        # ---- 2. Seed from the pre-sampled x1 + h0 (no re-consumption) ----
        x0 = draft_input.verified_id
        assert draft_input.first_draft_token_id is not None, (
            "EAGLE3 decode requires first_draft_token_id from prefill/rewrite."
        )
        assert draft_input.first_draft_logprob is not None
        assert draft_input.target_hidden_state is not None

        x1 = draft_input.first_draft_token_id
        x1_logprob = draft_input.first_draft_logprob
        current_hidden = draft_input.target_hidden_state.to(
            self._eagle3_hidden_dtype
        )

        all_tokens = [x0, x1]
        draft_logprobs = [x1_logprob]
        current_ids = x1

        # ---- 3. Gamma-1 draft forwards → produce x2..x_gamma ----
        # Step k forwards at position (L + k), consuming (embed(x_{k+1}),
        # current_hidden). Output logits at position L+k predict x_{k+2}.
        #
        # NB: seq_lens passed to the attn backend must equal orig+k, NOT
        # orig+k+1 — with default attn backend (speculative_step_id=0) the
        # FA backend computes cache_seqlens = seq_lens + step_id + 1, so
        # seq_lens=orig+k gives the correct attention window (0..orig+k
        # inclusive). The off-by-one is tolerable for the dense 1B draft but
        # destroys the 1-layer EAGLE3 draft head's predictions.
        for step in range(gamma - 1):
            pos_step = all_positions[:, step].contiguous()
            cache_step = cache_locs[:, step].contiguous()
            sl_step = all_positions[:, step].contiguous()  # NOTE: using positions (= orig+step), not all_seq_lens (= orig+step+1), to avoid off-by-one in FA cache_seqlens

            if (
                self._draft_cg_runner is not None
                and self._draft_cg_runner.can_run(bs, int(current_hidden.shape[-1]))
            ):
                logits, new_hidden = self._draft_cg_runner.replay(
                    bs,
                    input_ids=current_ids,
                    hidden_states=current_hidden,
                    positions=pos_step,
                    seq_lens=sl_step,
                    out_cache_loc=cache_step,
                    req_pool_indices=batch.req_pool_indices,
                )
            else:
                draft_fb.input_ids = current_ids
                draft_fb.positions = pos_step
                draft_fb.out_cache_loc = cache_step
                draft_fb.seq_lens = sl_step
                draft_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * step
                draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + step
                draft_fb.spec_info = EagleDraftInput(
                    hidden_states=current_hidden,
                    verified_id=current_ids,
                    num_tokens_per_req=1,
                    num_tokens_for_logprob_per_req=1,
                )
                draft_fb.capture_hidden_mode = CaptureHiddenMode.LAST

                draft_out = self.draft_runner.forward(draft_fb)
                logits = draft_out.logits_output.next_token_logits  # (bs, draft_vocab)
                new_hidden = draft_out.logits_output.hidden_states

            scaled = logits / self.smc_draft_temperature
            log_probs = torch.log_softmax(scaled, dim=-1)
            if self.smc_draft_temperature > 0:
                draft_idx = torch.multinomial(
                    log_probs.exp(), num_samples=1
                ).squeeze(-1)
            else:
                draft_idx = torch.argmax(logits, dim=-1)

            token_logprob = log_probs.gather(
                1, draft_idx.unsqueeze(1)
            ).squeeze(1)

            if self.hot_token_id is not None:
                next_token = self.hot_token_id[draft_idx]
            else:
                next_token = draft_idx

            all_tokens.append(next_token)
            draft_logprobs.append(token_logprob)
            current_ids = next_token
            current_hidden = new_hidden

        # all_tokens now has [x0, x1, x2, ..., x_gamma] (length gamma+1)
        assert len(all_tokens) == gamma + 1

        draft_logprobs_stacked = torch.stack(draft_logprobs, dim=1)  # (bs, gamma)

        # ---- 4. Target verify ----
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

        score_logits = score_result.logits_output.next_token_logits  # (bs*(gamma+1), V)
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]}, "
            f"expected {expected_rows}"
        )

        # Match dense-path convention: compute target logprobs at temp=1 for
        # the SMC weight math (NOT smc_target_temperature). Dense's empirically
        # working accuracy is governed by this choice; eagle3 was diverging
        # by applying smc_target_temperature here. Bonus sampling below
        # continues to use smc_target_temperature (same as dense).
        score_log_probs_all = torch.log_softmax(
            score_logits, dim=-1
        ).reshape(bs, gamma + 1, -1)
        # x_{k+1} (= all_tokens[k+1]) is verified at position k+1 of the window.
        # Target log-prob of x_{k+1} is score_log_probs_all[:, k, :].
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)  # (bs, gamma)
        score_logprobs_stacked = score_log_probs_all[:, :gamma, :].gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)

        logprob_diff = (score_logprobs_stacked - draft_logprobs_stacked).sum(dim=1)

        # ---- 5. Bonus token from target's last-position logits ----
        bonus_logits = score_logits.reshape(bs, gamma + 1, -1)[:, -1, :]
        bonus_log_probs = torch.log_softmax(
            bonus_logits / self.smc_target_temperature, dim=-1
        )
        if self.smc_target_temperature > 0:
            bonus = torch.multinomial(
                bonus_log_probs.exp(), num_samples=1
            ).squeeze(-1)
        else:
            bonus = torch.argmax(bonus_logits, dim=-1)

        # ---- 6. Output: commit draft's proposals + target's bonus ----
        # (Same as dense path — SMC commits all gamma+1 tokens per cycle.)
        output_token_ids = torch.stack(
            all_tokens[1 : gamma + 1] + [bonus], dim=1
        )
        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=self.device
        )

        # ---- 7. Rewrite draft KV and sample next cycle's x1 ----
        # For each position L+step (step=0..gamma), feed the COMMITTED token
        # at that position paired with target verify aux hidden at p. The
        # LAST step's (hidden, next_token_logits) seeds the next cycle.
        h_all = score_result.logits_output.hidden_states
        assert h_all is not None, (
            "EAGLE3 verify must capture FULL target aux hidden states."
        )
        aux_dim = h_all.shape[-1]
        target_h_steps = h_all.reshape(bs, gamma + 1, aux_dim).to(
            self._eagle3_hidden_dtype
        )
        accepted_tokens = output_token_ids  # (bs, gamma+1) [x1, x2, ..., bonus]

        _rewrite_mode = os.environ.get("SMC_EAGLE3_REWRITE", "min")
        if _rewrite_mode == "fused":
            last_hidden, last_logits = self._rewrite_fused_extend(
                batch, ctx, accepted_tokens, cache_locs,
                all_positions, all_seq_lens, target_h_steps,
            )
        elif _rewrite_mode == "min":
            # Only run the TWO rewrite steps whose positions the draft phase
            # never wrote: L+gamma-1 (input=x_gamma) and L+gamma (input=bonus).
            # Positions L+0..L+gamma-2 keep draft-hidden-based KV from the
            # draft phase — slight inconsistency vs full rewrite, but saves
            # gamma-1 = 7 forwards per cycle.
            rewrite_fb = draft_fb
            last_hidden = None
            last_logits = None
            for step in (gamma - 1, gamma):
                tok = accepted_tokens[:, step].contiguous()
                pos_step = all_positions[:, step].contiguous()
                cache_step = cache_locs[:, step].contiguous()
                # Use positions (= orig+step) as seq_lens so FA's
                # cache_seqlens = seq_lens + 1 = orig+step+1 matches the
                # position we're writing.
                sl_step = all_positions[:, step].contiguous()
                hidden_step = target_h_steps[:, step, :].contiguous()

                if (
                    self._draft_cg_runner is not None
                    and self._draft_cg_runner.can_run(
                        bs, int(hidden_step.shape[-1])
                    )
                ):
                    last_logits, last_hidden = self._draft_cg_runner.replay(
                        bs,
                        input_ids=tok,
                        hidden_states=hidden_step,
                        positions=pos_step,
                        seq_lens=sl_step,
                        out_cache_loc=cache_step,
                        req_pool_indices=batch.req_pool_indices,
                    )
                else:
                    rewrite_fb.input_ids = tok
                    rewrite_fb.positions = pos_step
                    rewrite_fb.out_cache_loc = cache_step
                    rewrite_fb.seq_lens = sl_step
                    rewrite_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * step
                    rewrite_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + step
                    rewrite_fb.spec_info = EagleDraftInput(
                        hidden_states=hidden_step,
                        verified_id=tok,
                        num_tokens_per_req=1,
                        num_tokens_for_logprob_per_req=1,
                    )
                    rewrite_fb.capture_hidden_mode = CaptureHiddenMode.LAST
                    out = self.draft_runner.forward(rewrite_fb).logits_output
                    last_hidden = out.hidden_states
                    last_logits = out.next_token_logits
        else:
            rewrite_fb = draft_fb
            last_hidden = None
            last_logits = None
            for step in range(gamma + 1):
                tok = accepted_tokens[:, step].contiguous()
                pos_step = all_positions[:, step].contiguous()
                cache_step = cache_locs[:, step].contiguous()
                # Use positions (= orig+step) as seq_lens to match FA's
                # cache_seqlens = seq_lens + 1 convention.
                sl_step = all_positions[:, step].contiguous()
                hidden_step = target_h_steps[:, step, :].contiguous()

                if (
                    self._draft_cg_runner is not None
                    and self._draft_cg_runner.can_run(bs, int(hidden_step.shape[-1]))
                ):
                    last_logits, last_hidden = self._draft_cg_runner.replay(
                        bs,
                        input_ids=tok,
                        hidden_states=hidden_step,
                        positions=pos_step,
                        seq_lens=sl_step,
                        out_cache_loc=cache_step,
                        req_pool_indices=batch.req_pool_indices,
                    )
                else:
                    rewrite_fb.input_ids = tok
                    rewrite_fb.positions = pos_step
                    rewrite_fb.out_cache_loc = cache_step
                    rewrite_fb.seq_lens = sl_step
                    rewrite_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * step
                    rewrite_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + step
                    rewrite_fb.spec_info = EagleDraftInput(
                        hidden_states=hidden_step,
                        verified_id=tok,
                        num_tokens_per_req=1,
                        num_tokens_for_logprob_per_req=1,
                    )
                    rewrite_fb.capture_hidden_mode = CaptureHiddenMode.LAST

                    out = self.draft_runner.forward(rewrite_fb).logits_output
                    last_hidden = out.hidden_states
                    last_logits = out.next_token_logits

        assert last_hidden is not None and last_logits is not None

        next_target_hidden = last_hidden.contiguous().to(self._eagle3_hidden_dtype)

        # Sample the NEXT cycle's x1 from the last rewrite step's logits.
        nxt_scaled = last_logits / self.smc_draft_temperature
        nxt_log_probs = torch.log_softmax(nxt_scaled, dim=-1)
        if self.smc_draft_temperature > 0:
            nxt_idx = torch.multinomial(
                nxt_log_probs.exp(), num_samples=1
            ).squeeze(-1)
        else:
            nxt_idx = torch.argmax(last_logits, dim=-1)
        nxt_x1_logprob = nxt_log_probs.gather(
            1, nxt_idx.unsqueeze(1)
        ).squeeze(1)
        if self.hot_token_id is not None:
            nxt_x1_target_id = self.hot_token_id[nxt_idx]
        else:
            nxt_x1_target_id = nxt_idx

        next_verified_id = bonus

        next_token_ids.record_stream(current_stream)
        accept_lens.record_stream(current_stream)
        next_verified_id.record_stream(current_stream)
        logprob_diff.record_stream(current_stream)

        next_draft_input = SMCDraftInputV2(
            verified_id=next_verified_id,
            logprob_diff=logprob_diff,
            num_tokens_per_req=self.speculative_num_draft_tokens,
            target_hidden_state=next_target_hidden,
            first_draft_token_id=nxt_x1_target_id,
            first_draft_logprob=nxt_x1_logprob,
        )

        return GenerationBatchResult(
            logits_output=score_result.logits_output,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            next_draft_input=next_draft_input,
            logprob_diff=logprob_diff,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _rewrite_fused_extend(
        self,
        batch: ModelWorkerBatch,
        ctx: SMCDecodeContext,
        accepted_tokens: torch.Tensor,   # (bs, gamma+1) target-vocab
        cache_locs: torch.Tensor,         # (bs, gamma+1)
        all_positions: torch.Tensor,      # (bs, gamma+1)
        all_seq_lens: torch.Tensor,       # (bs, gamma+1)
        target_h_steps: torch.Tensor,     # (bs, gamma+1, aux_dim)
    ):
        """Single-forward EXTEND-mode rewrite.

        Replaces the gamma+1 sequential decode forwards with ONE extend
        forward that processes bs*(gamma+1) tokens in a single weight-read
        pass. Same KV output as the sequential loop; same last-position
        hidden and logits for next-cycle seeding.
        """
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        bs = len(ctx.orig_seq_lens)
        gamma_p1 = self.gamma + 1
        total_tokens = bs * gamma_p1
        aux_dim = target_h_steps.shape[-1]

        flat_input_ids = accepted_tokens.reshape(-1).contiguous()
        flat_positions = all_positions.reshape(-1).contiguous()
        flat_out_cache = cache_locs.reshape(-1).contiguous()
        flat_hidden = target_h_steps.reshape(total_tokens, aux_dim).contiguous()

        # After this extend each req's total length is orig + (gamma+1).
        new_seq_lens = ctx.orig_seq_lens + gamma_p1
        new_seq_lens_cpu = ctx.orig_seq_lens_cpu + gamma_p1
        new_seq_lens_sum = ctx.orig_seq_lens_sum + bs * gamma_p1
        orig_seq_lens_list = ctx.orig_seq_lens_cpu.tolist()

        spec_info = EagleDraftInput(
            hidden_states=flat_hidden,
            verified_id=flat_input_ids,
            num_tokens_per_req=gamma_p1,
            num_tokens_for_logprob_per_req=1,  # only the last position per req
        )

        rewrite_batch = dataclasses.replace(
            batch,
            forward_mode=ForwardMode.EXTEND,
            input_ids=flat_input_ids,
            out_cache_loc=flat_out_cache,
            seq_lens=new_seq_lens,
            seq_lens_cpu=new_seq_lens_cpu,
            seq_lens_sum=new_seq_lens_sum,
            extend_num_tokens=total_tokens,
            extend_seq_lens=[gamma_p1] * bs,
            extend_prefix_lens=orig_seq_lens_list,
            extend_logprob_start_lens=[self.gamma] * bs,  # only last pos
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            is_extend_in_batch=True,
            all_extend_in_batch=True,
            return_logprob=False,
        )

        rewrite_fb = ForwardBatch.init_new(rewrite_batch, self.draft_runner)
        # Positions are authoritative for extend rewrite — override what
        # init_new computed from seq_lens (which would assume a contiguous
        # suffix starting at orig_seq_lens).
        rewrite_fb.positions = flat_positions

        out = self.draft_runner.forward(rewrite_fb).logits_output
        # LAST mode → hidden_states: (bs, H_draft); last_logits: (bs, vocab)
        return out.hidden_states, out.next_token_logits

    def _forward_idle(self, batch: ModelWorkerBatch):
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.empty(0, dtype=torch.int64, device=self.device),
            accept_lens=torch.empty(0, dtype=torch.int32, device=self.device),
            next_draft_input=SMCDraftInputV2.create_idle_input(self.device),
        )

    def _make_clean_batch(self, batch: ModelWorkerBatch) -> ModelWorkerBatch:
        """Copy batch with no spec_info (for draft model)."""
        return dataclasses.replace(
            batch, spec_info=None, capture_hidden_mode=CaptureHiddenMode.NULL
        )
