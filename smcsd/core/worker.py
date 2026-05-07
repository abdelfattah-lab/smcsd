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
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

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
    """Standalone SMC worker using v2 API (SMCDecodeContext + SMCDraftInput)."""

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
        # Hybrid-only branch: only the dense-draft path is supported.
        # eagle3/dflash code paths have been stripped out — see
        # hybrid-models-on-main if you want those.
        self._dense_draft_hybrid_req_to_token_pool = None

        # Per-phase timing accumulators (env-gated). Records the time spent in
        # the draft AR loop, target verify forward, and "other" SMC bookkeeping
        # (resample, mamba commit, output build) in milliseconds, summed across
        # decode steps. Set SMCSD_TIMING=1 to enable; print summary every
        # SMCSD_TIMING_EVERY steps (default 50).
        self._timing_enabled = bool(os.environ.get("SMCSD_TIMING"))
        self._timing_every = int(os.environ.get("SMCSD_TIMING_EVERY", "50"))
        self._t_draft_ms = 0.0
        self._t_verify_ms = 0.0
        self._t_other_ms = 0.0
        self._t_steps = 0
        if self._timing_enabled:
            print(
                f"[SMC TIMING] enabled (every {self._timing_every} steps) "
                f"on tp_rank={tp_rank}",
                flush=True,
            )

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

        # Dense AR draft worker — no MTP-architecture rewrite, no shared
        # embed/lm_head with the target.
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

        # Hybrid Qwen3.5/3.6 drafts need an isolated MambaPool sized to the
        # draft's recurrent state shape (different from the target's).
        self._maybe_isolate_dense_hybrid_draft_state()

        # Multi-step draft attention backend.
        # DraftBackendFactory.create_decode_backend() returns a flat-attention
        # multi-step backend that doesn't implement the linear-attn forward
        # signature radix_linear_attention.py expects (mixed_qkv/a/b kwargs).
        # For hybrid (Mamba+attention) drafts, build a custom multi-step
        # backend whose per-step backends are HybridLinearAttnBackend
        # instances that delegate full-attn vs linear-attn per layer_id.
        draft_is_hybrid = (
            getattr(self.draft_runner, "hybrid_gdn_config", None) is not None
        )
        if draft_is_hybrid:
            from smcsd.core.hybrid_multistep_backend import (
                HybridLinearAttnMultiStepBackend,
            )
            self.draft_attn_backend = HybridLinearAttnMultiStepBackend(
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

        # Restore cuda graph and capture for draft model
        server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        if not backup_disable_cuda_graph:
            self.draft_runner.init_device_graphs()

    def _dense_hybrid_state_shape(self) -> Optional[Tuple[Tuple, Tuple]]:
        target_cfg = getattr(self.score_runner, "hybrid_gdn_config", None)
        draft_cfg = getattr(self.draft_runner, "hybrid_gdn_config", None)
        if target_cfg is None or draft_cfg is None:
            return None

        keys = (
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
        )
        target_shape = tuple(getattr(target_cfg, key, None) for key in keys)
        draft_shape = tuple(getattr(draft_cfg, key, None) for key in keys)
        return target_shape, draft_shape

    def _maybe_isolate_dense_hybrid_draft_state(self) -> None:
        """Give dense hybrid drafts their own recurrent state and KV layout.

        Dense SMC still shares request/token slot indices so target and draft KV
        caches stay aligned. For hybrid GDN drafts, the recurrent state shape is
        model-specific, and normal AR drafts need every full-attention layer in
        their KV pool rather than SGLang's one-layer MTP draft layout.
        """
        shapes = self._dense_hybrid_state_shape()
        target_shape, draft_shape = shapes or (None, None)
        from sglang.srt.layers.dp_attention import get_attention_tp_size
        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            HybridReqToTokenPool,
        )

        target_pool = self.req_to_token_pool
        draft_config = self.draft_runner.mambaish_config
        _smc_debug = bool(os.environ.get("SMCSD_HYBRID_DEBUG"))
        if _smc_debug:
            print(
                f"[SMC HYBRID] tp{self.tp_rank} isolation check: "
                f"target_has_mamba_pool={hasattr(target_pool, 'mamba_pool')} "
                f"draft_mambaish_config={draft_config is not None} "
                f"target_shape={target_shape} draft_shape={draft_shape}",
                flush=True,
            )
        if not hasattr(target_pool, "mamba_pool") or draft_config is None:
            if _smc_debug:
                print(
                    f"[SMC HYBRID] tp{self.tp_rank} isolation SKIPPED — "
                    f"draft uses target's pool",
                    flush=True,
                )
            return

        draft_pool = HybridReqToTokenPool(
            size=target_pool.size,
            mamba_size=target_pool.size,
            mamba_spec_state_size=target_pool.size,
            max_context_len=target_pool.max_context_len,
            device=self.draft_runner.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            cache_params=draft_config.mamba2_cache_params,
            mamba_layer_ids=[
                i
                for i in draft_config.mamba2_cache_params.layers
                if self.draft_runner.start_layer <= i < self.draft_runner.end_layer
            ],
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=None,
            enable_overlap_schedule=False,
            start_layer=self.draft_runner.start_layer,
        )
        # Share token block-table storage; isolate only the recurrent state pool.
        draft_pool.req_to_token = target_pool.req_to_token
        draft_pool.req_index_to_mamba_index_mapping.copy_(
            torch.arange(
                target_pool.size + 1,
                dtype=torch.int32,
                device=self.draft_runner.device,
            )
        )
        draft_pool.free_slots = []
        draft_pool.mamba_pool.free_slots = torch.empty(
            0, dtype=torch.int64, device=self.draft_runner.device
        )

        self.draft_runner.req_to_token_pool = draft_pool

        extra_args = {}
        if self.draft_runner.use_mla_backend:
            extra_args = {
                "kv_lora_rank": self.draft_runner.model_config.kv_lora_rank,
                "qk_rope_head_dim": self.draft_runner.model_config.qk_rope_head_dim,
            }
        self.draft_runner.token_to_kv_pool = HybridLinearKVPool(
            page_size=self.draft_runner.page_size,
            size=self.draft_runner.max_total_num_tokens,
            dtype=self.draft_runner.kv_cache_dtype,
            head_num=self.draft_runner.model_config.get_num_kv_heads(
                get_attention_tp_size()
            ),
            head_dim=self.draft_runner.model_config.head_dim,
            full_attention_layer_ids=[
                i
                for i in draft_config.full_attention_layer_ids
                if self.draft_runner.start_layer <= i < self.draft_runner.end_layer
            ],
            enable_kvcache_transpose=False,
            device=self.draft_runner.device,
            mamba_pool=draft_pool.mamba_pool,
            enable_memory_saver=self.server_args.enable_memory_saver,
            use_mla=self.draft_runner.use_mla_backend,
            start_layer=self.draft_runner.start_layer,
            **extra_args,
        )

        linear_backend = getattr(
            self.draft_runner.attn_backend, "linear_attn_backend", None
        )
        if linear_backend is not None:
            linear_backend.req_to_token_pool = draft_pool
            linear_backend.conv_states_shape = draft_pool.mamba_pool.mamba_cache.conv[
                0
            ].shape
            if hasattr(linear_backend, "verify_intermediate_state_indices"):
                linear_backend.verify_intermediate_state_indices = torch.arange(
                    draft_pool.size,
                    dtype=torch.int32,
                    device=self.draft_runner.device,
                )

        self._dense_draft_hybrid_req_to_token_pool = draft_pool
        # Backref so the SMC release helpers (_release_internal_req /
        # _release_smc_parent_req) can free the draft pool's mamba state
        # alongside the target's. Without this, freed req_pool_idx slots
        # get re-used by the next request while their draft Mamba state
        # carries over from the previous occupant — causes accuracy to
        # degrade monotonically across questions on hybrid+hybrid pairs.
        target_pool._smc_draft_hybrid_pool = draft_pool
        msg = (
            f"SMC dense mode isolated hybrid draft state/KV: "
            f"target={self.score_runner.model_config.model_path} "
            f"shape={target_shape} "
            f"draft={self.draft_runner.model_config.model_path} "
            f"shape={draft_shape} "
            f"full_attn_layers="
            f"{list(self.draft_runner.token_to_kv_pool.full_attention_layer_id_mapping.keys())}"
        )
        logger.warning(msg)
        if _smc_debug:
            print(f"[SMC HYBRID] tp{self.tp_rank} {msg}", flush=True)

    @staticmethod
    def _copy_hybrid_state_pairwise(pool, src_req_pool_indices, dst_req_pool_indices):
        if pool is None or not hasattr(pool, "mamba_pool"):
            return
        if src_req_pool_indices.numel() == 0:
            return
        mapping = pool.req_index_to_mamba_index_mapping
        src_mamba = mapping[src_req_pool_indices.to(torch.long)].to(torch.long)
        dst_mamba = mapping[dst_req_pool_indices.to(torch.long)].to(torch.long)
        pool.mamba_pool.copy_from(src_mamba, dst_mamba)

    def fanout_smc_parent_hybrid_state(self, parent_req, particle_reqs) -> None:
        """Copy hybrid recurrent state from the prefilled parent to particles."""
        if parent_req.req_pool_idx is None or not particle_reqs:
            return
        dst = torch.tensor(
            [req.req_pool_idx for req in particle_reqs],
            dtype=torch.long,
            device=self.device,
        )
        src = torch.full_like(dst, int(parent_req.req_pool_idx))
        self._copy_hybrid_state_pairwise(self.req_to_token_pool, src, dst)
        self._copy_hybrid_state_pairwise(
            self._dense_draft_hybrid_req_to_token_pool, src, dst
        )

    def copy_smc_resampled_hybrid_state(self, slot_state, plan) -> None:
        """Copy hybrid recurrent state after SMC resampling clones particles."""
        if isinstance(plan, list):
            dst_slots = []
            src_slots = []
            for job in plan:
                dst_slots.extend(job.dst_slots)
                src_slots.extend(job.src_slots)
            if not dst_slots:
                return
            dst_slots_t = torch.tensor(dst_slots, dtype=torch.long, device=self.device)
            src_slots_t = torch.tensor(src_slots, dtype=torch.long, device=self.device)
        else:
            if plan.n_jobs == 0:
                return
            dst_slots_t = plan.dst_slots.to(torch.long)
            src_slots_t = plan.src_slots.to(torch.long)

        dst_req_pool = slot_state.req_pool_indices[dst_slots_t]
        src_req_pool = slot_state.req_pool_indices[src_slots_t]
        self._copy_hybrid_state_pairwise(
            self.req_to_token_pool, src_req_pool, dst_req_pool
        )
        self._copy_hybrid_state_pairwise(
            self._dense_draft_hybrid_req_to_token_pool, src_req_pool, dst_req_pool
        )

    def _commit_target_mamba_state_after_verify(
        self,
        verify_forward_batch: ForwardBatch,
        accepted_steps: torch.Tensor,
    ) -> None:
        """Commit hybrid recurrent state produced during TARGET_VERIFY.

        Official SGLang speculative paths run hybrid/GDN target verification with
        deferred state updates, then scatter the accepted intermediate state back
        into the live mamba cache. Dense SMC also uses TARGET_VERIFY, so it must
        perform the same commit for Qwen3.5-style hybrid targets.
        """
        attn_backend = self._target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return
        if verify_forward_batch.forward_mode.is_idle():
            return

        attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps.to(dtype=torch.int64),
            mamba_track_indices=verify_forward_batch.mamba_track_indices,
            mamba_steps_to_track=None,
            model=self._target_worker.model_runner.model,
        )

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

        # Dense AR draft + target prefill.
        # Score model prefill
        score_result = self._target_worker.forward_batch_generation(batch)

        # Draft model prefill — samples the first token (x0)
        draft_batch = self._make_clean_batch(batch)
        draft_result = self._draft_worker.forward_batch_generation(draft_batch)

        # Use draft model's sampled token as verified_id
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

        # ---- Timing setup (env-gated, dense path only) ----
        if self._timing_enabled:
            ev_t0 = torch.cuda.Event(enable_timing=True)
            ev_draft_end = torch.cuda.Event(enable_timing=True)
            ev_verify_end = torch.cuda.Event(enable_timing=True)
            ev_other_end = torch.cuda.Event(enable_timing=True)
            ev_t0.record()

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

        if self._timing_enabled:
            ev_draft_end.record()

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
        if self.score_runner.hybrid_gdn_config is not None:
            accepted_steps = torch.full(
                (bs,), gamma, dtype=torch.int64, device=self.device
            )
            self._commit_target_mamba_state_after_verify(
                verify_forward_batch, accepted_steps
            )

        if self._timing_enabled:
            ev_verify_end.record()

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

        next_draft_input = SMCDraftInput(
            verified_id=next_verified_id,
            logprob_diff=logprob_diff,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )

        if self._timing_enabled:
            ev_other_end.record()
            ev_other_end.synchronize()
            self._t_draft_ms += ev_t0.elapsed_time(ev_draft_end)
            self._t_verify_ms += ev_draft_end.elapsed_time(ev_verify_end)
            self._t_other_ms += ev_verify_end.elapsed_time(ev_other_end)
            self._t_steps += 1
            if self._t_steps % self._timing_every == 0:
                tot = self._t_draft_ms + self._t_verify_ms + self._t_other_ms
                print(
                    f"[SMC TIMING] tp{self.tp_rank} steps={self._t_steps} "
                    f"draft={self._t_draft_ms:.0f}ms ({100 * self._t_draft_ms / max(tot, 1e-6):.1f}%) "
                    f"verify={self._t_verify_ms:.0f}ms ({100 * self._t_verify_ms / max(tot, 1e-6):.1f}%) "
                    f"other={self._t_other_ms:.0f}ms ({100 * self._t_other_ms / max(tot, 1e-6):.1f}%) "
                    f"avg/step={tot / self._t_steps:.1f}ms",
                    flush=True,
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
