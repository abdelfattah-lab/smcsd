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
from typing import Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
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
        self._eagle3_residual_alpha = getattr(server_args, "eagle3_residual_alpha", 0.0)

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

                # Build target-vocab → hot-vocab inverse map (t2d).
                # We need this to gather target log-probs on the SAME support as
                # the draft so that logprob_diff is a valid importance weight.
                # Entries not in the hot set stay at -1 (not expected to be hit
                # since all proposed tokens come from hot_token_id[draft_idx]).
                target_vocab_size = int(
                    self._target_worker.model_runner.model_config.vocab_size
                )
                self._t2d_map = torch.full(
                    (target_vocab_size,),
                    -1,
                    dtype=torch.int64,
                    device=embed.device,
                )
                hot_ids_int64 = self.hot_token_id.to(torch.int64)
                self._t2d_map[hot_ids_int64] = torch.arange(
                    hot_ids_int64.numel(),
                    dtype=torch.int64,
                    device=embed.device,
                )
            else:
                self._t2d_map = None

            # Configure the TARGET model to capture aux hidden states from
            # the 3 designated layers (low, mid, high).  The CUDA graph runner
            # also does this, but when graphs are disabled (e.g. FA3) we need
            # to do it explicitly here.
            target_model = self._target_worker.model_runner.model
            if hasattr(target_model, "set_eagle3_layers_to_capture"):
                eagle_aux_layers = None
                draft_hf = self.draft_runner.model_config.hf_config
                eagle_cfg = getattr(draft_hf, "eagle_config", {})
                if eagle_cfg.get("use_aux_hidden_state_layers"):
                    eagle_aux_layers = eagle_cfg["use_aux_hidden_state_layers"]
                target_model.set_eagle3_layers_to_capture(eagle_aux_layers)

            # EAGLE3 head runs eagerly — disable its CUDA graph.
            backup_disable_cuda_graph = True

        # Multi-step draft attention backend
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

    def sample_per_particle_x1(
        self,
        parent_log_probs: torch.Tensor,
        n_particles: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """EAGLE3 only: sample ``n_particles`` distinct x1 draws for ONE parent.

        Args:
            parent_log_probs: (draft_vocab,) log-softmax'ed draft prefill
                              logits for a single parent.
            n_particles: group fan-out.

        Returns:
            (target_ids, logprobs) both of shape (n_particles,). target_ids
            are already in TARGET-vocab space (mapped via hot_token_id when
            applicable) so they can be used directly as input_ids for the
            next draft step.
        """
        if self.smc_draft_temperature > 0:
            idx = torch.multinomial(
                parent_log_probs.exp(), num_samples=n_particles, replacement=True
            )
        else:
            top_idx = torch.argmax(parent_log_probs, dim=-1)
            idx = top_idx.expand(n_particles)
        logprobs = parent_log_probs.gather(0, idx)
        if self.hot_token_id is not None:
            target_ids = self.hot_token_id[idx]
        else:
            target_ids = idx
        return target_ids, logprobs

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

            # Return the draft's prefill last-position log-probs so the
            # scheduler can sample a DISTINCT x1 per particle (one row per
            # parent → fan-out in set_prefill_hidden via
            # sample_per_particle_x1). Broadcasting a single x1 across all N
            # particles would destroy particle diversity in the first decode
            # cycle — this is the critical correctness fix for small gamma.
            prefill_next_logits = draft_logits_out.next_token_logits  # (bs, draft_vocab)
            scaled = prefill_next_logits / self.smc_draft_temperature
            prefill_log_probs = torch.log_softmax(scaled, dim=-1)

            score_result.next_draft_input = SMCDraftInputV2(
                verified_id=score_result.next_token_ids,
                num_tokens_per_req=self.speculative_num_draft_tokens,
                target_hidden_state=h0,
                # Full per-parent logprob row in draft-vocab space.
                first_draft_logprobs=prefill_log_probs,
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

        # Anchor: the fc-projected target hidden state (4096-dim) from the
        # rewrite step. We blend this back into the draft's recurrent hidden
        # at each subsequent step so that target information doesn't fade.
        target_anchor = current_hidden
        residual_alpha = getattr(self, "_eagle3_residual_alpha", 0.0)

        all_tokens = [x0, x1]
        draft_logprobs = [x1_logprob]
        current_ids = x1

        # ---- 3. Gamma-1 draft forwards → produce x2..x_gamma ----
        for step in range(gamma - 1):
            draft_fb.input_ids = current_ids
            draft_fb.positions = all_positions[:, step].contiguous()
            draft_fb.out_cache_loc = cache_locs[:, step].contiguous()
            draft_fb.seq_lens = all_seq_lens[:, step].contiguous()
            draft_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * (step + 1)
            draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + (step + 1)

            draft_fb.spec_info = None
            self.draft_runner.attn_backend.init_forward_metadata(draft_fb)

            draft_fb.spec_info = EagleDraftInput(
                hidden_states=current_hidden,
                target_anchor=target_anchor,
                draft_step=step + 1,
                verified_id=current_ids,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )
            draft_fb.capture_hidden_mode = CaptureHiddenMode.LAST

            draft_out = self.draft_runner.forward(
                draft_fb, skip_attn_backend_init=True
            )
            logits = draft_out.logits_output.next_token_logits  # (bs, draft_vocab)
            new_hidden = draft_out.logits_output.hidden_states

            if residual_alpha > 0:
                new_hidden = (
                    (1 - residual_alpha) * new_hidden
                    + residual_alpha * target_anchor
                )

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

        # Match the dense path: compute target log-probs at T=1 so that
        # logprob_diff = log p_target(y) - log q_draft(y) is a well-formed
        # importance weight on the same scale as the dense SMC path.
        # (Bonus sampling below still uses smc_target_temperature — same as dense.)
        #
        # Additionally, if the EAGLE3 draft has a "hot" sub-vocabulary, project
        # the target logits onto the hot set before log-softmax so that `p` and
        # `q` share support. Otherwise `log p` is normalized over the full
        # vocabulary while `log q` is normalized only over hot tokens, which
        # biases the IS weight by log(sum_hot p) on every step.
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)  # (bs, gamma)
        if self.hot_token_id is not None:
            hot_logits = score_logits.index_select(
                1, self.hot_token_id.to(torch.int64)
            )
            hot_log_probs = torch.log_softmax(hot_logits, dim=-1).reshape(
                bs, gamma + 1, -1
            )
            draft_idx_tokens = self._t2d_map[target_tokens.to(torch.int64)]
            score_logprobs_stacked = hot_log_probs[:, :gamma, :].gather(
                2, draft_idx_tokens.unsqueeze(2)
            ).squeeze(2)
        else:
            score_log_probs_all = torch.log_softmax(
                score_logits, dim=-1
            ).reshape(bs, gamma + 1, -1)
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
        # at that position (shift-by-one convention: input at p = token_{p+1})
        # paired with target verify aux hidden at p. The LAST step's
        # (hidden, next_token_logits) seeds the next cycle.
        h_all = score_result.logits_output.hidden_states
        assert h_all is not None, (
            "EAGLE3 verify must capture FULL target aux hidden states."
        )
        aux_dim = h_all.shape[-1]
        target_h_steps = h_all.reshape(bs, gamma + 1, aux_dim).to(
            self._eagle3_hidden_dtype
        )

        # Rewrite draft KV with gamma+1 eager DECODE forwards (one per
        # committed position). An earlier attempt to collapse this into a
        # single DRAFT_EXTEND_V2 multi-token forward (mirroring
        # eagle_worker_v2._draft_extend_for_decode) ran but produced
        # gibberish outputs — almost certainly due to a mismatch between
        # our ForwardBatch setup and what FA3's DRAFT_EXTEND_V2 path
        # expects (positions / cache_seqlens / metadata). Left as a known
        # perf opportunity for a follow-up that ports the dedicated
        # EAGLE3 draft-extend CUDA graph runner.
        rewrite_fb = draft_fb
        accepted_tokens = output_token_ids  # [x1, x2, ..., x_gamma, bonus]
        last_hidden = None
        last_logits = None
        for step in range(gamma + 1):
            tok = accepted_tokens[:, step].contiguous()
            rewrite_fb.input_ids = tok
            rewrite_fb.positions = all_positions[:, step].contiguous()
            rewrite_fb.out_cache_loc = cache_locs[:, step].contiguous()
            rewrite_fb.seq_lens = all_seq_lens[:, step].contiguous()
            rewrite_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * (step + 1)
            rewrite_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + (step + 1)

            rewrite_fb.spec_info = None
            self.draft_runner.attn_backend.init_forward_metadata(rewrite_fb)

            rewrite_fb.spec_info = EagleDraftInput(
                hidden_states=target_h_steps[:, step, :].contiguous(),
                verified_id=tok,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )
            rewrite_fb.capture_hidden_mode = CaptureHiddenMode.LAST

            out = self.draft_runner.forward(
                rewrite_fb, skip_attn_backend_init=True
            ).logits_output
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
