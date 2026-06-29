"""SMC worker: dense-AR draft path.

Draft model performs gamma autoregressive decode steps.
Score model performs one extend forward pass on the drafted tokens.
Computes logprob difference between the two models per request.
No rejection — all drafted tokens are accepted.

Supports any (target, draft) pair where the draft can be loaded as a
standalone autoregressive LM. Hybrid (Mamba+attention) targets whose draft
has a different recurrent-state shape get an isolated draft Mamba pool via
``_maybe_isolate_dense_hybrid_draft_state``.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
from typing import Optional, Tuple

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

        # Draft-only quantization (e.g. SMC_DRAFT_QUANTIZATION=fp8): the
        # draft phase is weight-read-bound at decode, so quantized draft
        # weights cut its HBM traffic.  The sampler stays exact —
        # importance weights are computed under the same (quantized)
        # proposal q the tokens are drawn from, so this changes proposal
        # quality only, never introduces bias.  The target model is
        # untouched.  Swap self.server_args to the modified copy so the
        # draft ModelRunner sees a consistent view.
        draft_quant = os.environ.get("SMC_DRAFT_QUANTIZATION")
        if draft_quant:
            # Shallow copy (NOT dataclasses.replace, which would re-run
            # ServerArgs.__post_init__ on an already-derived instance).
            draft_args = copy.copy(self.server_args)
            draft_args.quantization = draft_quant
            self.server_args = draft_args

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
        self.device = server_args.device
        self._target_worker = target_worker  # score model

        self.gamma = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = self.gamma + 1
        self.smc_draft_temperature = server_args.smc_draft_temperature
        self.smc_target_temperature = max(
            float(server_args.smc_target_temperature), 1e-5
        )
        # Exponent alpha for the sequence-wise power target p^alpha in the SMC
        # importance weight.  Set by SMCEngine as a dynamic attribute on the
        # ServerArgs instance (keeps the vendored class unmodified); defaults
        # to 1.0 (plain p) for launches that don't go through SMCEngine.
        self.smc_power_alpha = float(getattr(server_args, "smc_power_alpha", 1.0))
        # Exponent-bridging (alpha-ramp) config: the effective exponent rises
        # from 1.0 to smc_power_alpha over the first smc_alpha_ramp_tokens
        # generated tokens, then holds (Power-SMC paper Sec. 5.3 / App. B).
        # 0 disables the ramp (fixed alpha). See _ramped_alpha().
        self.smc_alpha_ramp_tokens = int(
            getattr(server_args, "smc_alpha_ramp_tokens", 0)
        )
        if self.smc_alpha_ramp_tokens > 0:
            # Steps 1-2 of the ramp are wired here: the per-row exponent
            # alpha(t) flows into the proposal temperature, the per-token
            # weight, and the bonus draw (cycle-graph path). The exact
            # SMC-samplers boundary reweight (alpha(t)-alpha(t-1))*log p(y_{1:t})
            # that makes the FINAL target precisely p^alpha (Step 3) is NOT in
            # yet, so this is the *annealed-proposal* ramp: it bridges the
            # proposal/weight exponent (the main anti-early-collapse mechanism)
            # but the final target is approximate until the boundary term lands.
            # Only the cycle-graph decode path implements it.
            if not bool(int(os.environ.get("SMC_CYCLE_GRAPH", "0"))):
                raise NotImplementedError(
                    "alpha-ramp currently requires the cycle-graph decode path "
                    "(set SMC_CYCLE_GRAPH=1); the eager path is not wired."
                )
            logger.warning(
                "[SMC] alpha-ramp ON (T_ramp=%d, linear): annealed "
                "proposal+weight; exact boundary reweight (final-target "
                "preservation) not yet implemented.",
                self.smc_alpha_ramp_tokens,
            )
        # Debug-only: dump draft KV positions / cache-loc mapping for the first
        # few decode calls to confirm the prefill→step-0 position convention
        # before the deferred-bonus rework.  No behavior change when unset.
        self._smc_dbg_positions = bool(
            int(os.environ.get("SMC_DEBUG_POSITIONS", "0"))
        )
        self._smc_dbg_calls = 0
        # Deferred-bonus draft schedule: drop the per-step over-draft and fold
        # the deferred d_{gamma-1} write into the next step's leading 2-token
        # head (eager; CUDA-graph capture is a later step).  Off => legacy
        # gamma+1 single-token AR loop, byte-identical.
        self.smc_defer_bonus = bool(
            int(os.environ.get("SMC_DEFER_BONUS", "0"))
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

        # Deferred-bonus: pin the DRAFT backend's verify-block-size global to
        # the head's 2 tokens.  The vendored verify-metadata paths read this
        # backend global rather than the per-batch spec value (triton:
        # num_draft_tokens in capture/replay; FA3: speculative_num_draft_tokens
        # in eager/capture/replay) — on the draft backend the only verify
        # consumer is the 2-token head, so 2 is the correct value for this
        # instance.  The target's backend is a separate object and keeps
        # gamma+1.  Pinned at worker level (not in the head graph runner) so
        # the EAGER head path is covered too: FA3's eager verify metadata also
        # reads the global.  Done after init_device_graphs so the decode
        # graphs capture with stock state.
        # Hazard trail: if a future vendored path on the *draft* backend reads
        # this global expecting gamma+1, it would silently get 2 — today no
        # such path exists (decode ignores it; verify on the draft IS the
        # head).
        if self.smc_defer_bonus:
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )
            from sglang.srt.layers.attention.triton_backend import (
                TritonAttnBackend,
            )

            draft_ab = self.draft_runner.attn_backend
            if isinstance(draft_ab, TritonAttnBackend):
                draft_ab.num_draft_tokens = 2
            elif isinstance(draft_ab, FlashAttentionBackend):
                draft_ab.speculative_num_draft_tokens = 2
            else:
                raise RuntimeError(
                    "SMC_DEFER_BONUS supports triton and fa3 draft attention "
                    f"backends only; got {type(draft_ab).__name__} (hybrid "
                    "and MLA drafts are not supported by the deferred-bonus "
                    "head)."
                )

        # Deferred-bonus: a second, num_tokens_per_bs=2 TARGET_VERIFY graph
        # runner on the draft for the 2-token head (the primary draft runner is
        # decode-only and can't replay it).  Captured here, after the draft
        # model + its decode graphs are fully set up.  Eager fallback when None.
        # SMC_DEFER_BONUS_EAGER=1 skips capture so the head runs the eager path
        # — the A/B reference for graph-vs-eager equivalence on a fixed seed.
        self.draft_head_graph_runner = None
        defer_eager = bool(int(os.environ.get("SMC_DEFER_BONUS_EAGER", "0")))
        if (
            self.smc_defer_bonus
            and not backup_disable_cuda_graph
            and not defer_eager
        ):
            from smcsd.model_executor.smc_cuda_graph_runner import (
                SMCDraftHeadGraphRunner,
            )
            self.draft_head_graph_runner = SMCDraftHeadGraphRunner(
                self.draft_runner
            )

        # Whole-draft-phase CUDA graph (issue #14): capture all gamma+1 draft
        # forwards + in-graph Gumbel sampling in one graph per bs bucket,
        # replacing gamma+1 separate decode-graph replays + eager sampling
        # (~25% GPU idle from host dispatch at bs=32).  SMC_CYCLE_GRAPH=1
        # extends the capture through TARGET_VERIFY + weight diff + bonus —
        # one launch for the whole worker cycle.  Opt-in; the per-step path
        # stays as the fallback for oversized batches / long contexts.
        self.draft_phase_graph_runner = None
        self.cycle_graph_runner = None
        want_cycle = bool(int(os.environ.get("SMC_CYCLE_GRAPH", "0")))
        want_phase = bool(int(os.environ.get("SMC_DRAFT_PHASE_GRAPH", "0")))
        if want_cycle or want_phase:
            from sglang.srt.layers.attention.triton_backend import (
                TritonAttnBackend,
                TritonMultiStepDraftBackend,
            )

            flag = "SMC_CYCLE_GRAPH" if want_cycle else "SMC_DRAFT_PHASE_GRAPH"
            reasons = []
            if backup_disable_cuda_graph:
                reasons.append("cuda graph disabled")
            if self.smc_defer_bonus and not want_cycle:
                reasons.append(
                    "SMC_DEFER_BONUS=1 (phase graph captures the legacy "
                    "gamma+1 schedule; the deferred head is only captured "
                    "by the full-cycle runner, SMC_CYCLE_GRAPH=1)"
                )
            if not self.smc_draft_temperature > 0:
                reasons.append("greedy draft (temperature 0)")
            if not isinstance(self.draft_attn_backend, TritonMultiStepDraftBackend):
                reasons.append(
                    f"unsupported multi-step backend "
                    f"{type(self.draft_attn_backend).__name__} (triton only)"
                )
            if want_cycle and self.score_runner.hybrid_gdn_config is not None:
                reasons.append(
                    "hybrid target (cycle graph cannot capture the "
                    "post-verify mamba state commit)"
                )
            if want_cycle and not isinstance(
                self.score_runner.attn_backend, TritonAttnBackend
            ):
                reasons.append(
                    f"unsupported target backend "
                    f"{type(self.score_runner.attn_backend).__name__} "
                    "(triton only)"
                )
            if reasons:
                logger.warning("%s=1 ignored: %s", flag, "; ".join(reasons))
            elif want_cycle:
                from smcsd.model_executor.smc_draft_phase_graph_runner import (
                    SMCDeferredCycleGraphRunner,
                    SMCFullCycleGraphRunner,
                )

                # With SMC_DEFER_BONUS=1 the cycle capture uses the deferred
                # draft schedule (2-token head + gamma-1 singles): one fewer
                # draft forward per cycle.
                runner_cls = (
                    SMCDeferredCycleGraphRunner
                    if self.smc_defer_bonus
                    else SMCFullCycleGraphRunner
                )
                self.cycle_graph_runner = runner_cls(self)
            else:
                from smcsd.model_executor.smc_draft_phase_graph_runner import (
                    SMCDraftPhaseGraphRunner,
                )

                self.draft_phase_graph_runner = SMCDraftPhaseGraphRunner(self)

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

        Unlike vanilla speculative decoding, SMC accepts every drafted token
        (no rejection / rollback), so a separate draft pool is NOT needed for
        recovery. We share what we can: the request→token block-table
        (req_to_token), the req_pool_idx allocator, and the identity-mapped
        req_index→mamba_index mapping.

        What we cannot share for an asymmetric hybrid pair (e.g. Qwen3.5-9B
        target + Qwen3.5-2B draft):
          * MambaPool — recurrent state shape (num_heads, ssm_state_size, …)
            differs between target and draft, so each model needs its own
            buffers sized to its own config.
          * HybridLinearKVPool — head_dim / num_kv_heads / number of full-attn
            layers differ, so KV layout is model-specific. AR drafts also need
            every full-attn layer (vs SGLang's one-layer MTP draft layout).
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

    def _commit_target_mamba_state_after_verify(
        self,
        verify_forward_batch: ForwardBatch,
        accepted_steps: torch.Tensor,
    ) -> None:
        """Commit hybrid recurrent state produced during TARGET_VERIFY.

        Official SGLang speculative paths run hybrid/GDN target verification with
        deferred state updates, then scatter the accepted intermediate state back
        into the live mamba cache. The dense-AR SMC path also uses TARGET_VERIFY,
        so it must perform the same commit for hybrid (Mamba+attention) targets.
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

        # Score model prefill — authoritative prompt KV + score state.
        score_result = self._target_worker.forward_batch_generation(batch)

        # Draft model prefill — populate the draft's prompt KV; token discarded.
        draft_batch = self._make_clean_batch(batch)
        self._draft_worker.forward_batch_generation(draft_batch)

        # x0 seed from the TARGET's real first-token distribution.
        #
        # The SMC target's score/extend forward does not compute
        # next_token_logits (the logits processor gets no sample position and
        # returns an unwritten warmup buffer), so we recover the real
        # distribution from the prompt's hidden states: one target forward with
        # FULL hidden capture, then the lm_head on the last prompt token's
        # (post-final-norm) hidden state. KV writes are idempotent (same prompt
        # -> same out_cache_loc), so the score prefill above stays authoritative.
        # x0 is shared conditioning across the group's particles and carries no
        # per-particle weight; drawing it from the target power p_T^alpha makes
        # the seed model-correct and draft-independent.
        x0_batch = dataclasses.replace(
            batch, spec_info=None, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        x0_result = self._target_worker.forward_batch_generation(x0_batch)
        hidden = x0_result.logits_output.hidden_states
        assert hidden is not None, (
            "FULL hidden-state capture did not populate hidden_states for x0"
        )
        last_idx = torch.cumsum(batch.seq_lens.to(torch.int64), dim=0) - 1
        last_hidden = hidden[last_idx]  # last prompt token per request (post-norm)
        lm_head = self.score_runner.model.lm_head
        x0_logits = torch.matmul(
            last_hidden.to(lm_head.weight.dtype), lm_head.weight.T
        ).float()
        x0, _ = self._sample_target_power(x0_logits)
        score_result.next_token_ids = x0

        # x0 KV is NOT written during prefill — first decode writes it.
        score_result.next_draft_input = SMCDraftInput(
            verified_id=x0,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )
        score_result.accept_lens = torch.zeros(
            bs, dtype=torch.int32, device=self.device
        )
        return score_result

    # ── DECODE ──

    def _sample_draft_token(self, logits, need_logprob: bool = True):
        """Sample one draft token (+ its log-prob) at the draft temperature.

        Gumbel-max sampling: ``argmax(logits/T + g)`` with ``g ~ Gumbel(0,1)``
        draws exactly from ``softmax(logits/T)`` — the same distribution as
        the previous ``log_softmax → exp → multinomial`` chain, without the
        multi-kernel multinomial.  The Gumbel noise drives only the
        *selection*; the token's logprob comes from the noise-free scaled
        logits, ``logp = (logits/T)[idx] - logsumexp(logits/T)``, which is
        what the importance weight requires.  Used by both the legacy AR
        loop and the deferred-bonus head path.
        """
        if self.smc_draft_temperature > 0:
            scaled = logits / self.smc_draft_temperature
            gumbel = -torch.log(
                -torch.log(
                    torch.rand_like(scaled).clamp_min_(
                        torch.finfo(scaled.dtype).tiny
                    )
                )
            )
            idx = torch.argmax(scaled + gumbel, dim=-1)
        else:
            # Greedy: noise-free argmax (scale-invariant in the logits).
            scaled = logits
            idx = torch.argmax(logits, dim=-1)
        if not need_logprob:
            return idx, None
        chosen = scaled.gather(1, idx.unsqueeze(1)).squeeze(1)
        lp = chosen - torch.logsumexp(scaled, dim=-1)
        return idx, lp

    def _ramped_alpha(self, gen_lens):
        """Per-row effective exponent under the exponent-bridging schedule.

        ``gen_lens`` is a tensor of per-row generated-token counts (the prefix
        length already committed for each particle). With ramp horizon
        ``T = smc_alpha_ramp_tokens`` and final exponent ``a = smc_power_alpha``
        the linear schedule (Power-SMC paper App. B) is::

            alpha(t) = 1 + (a - 1) * min(t, T) / T      (T > 0)
            alpha(t) = a                                (T == 0, ramp off)

        Returns a float tensor shaped like ``gen_lens``. The caller uses
        ``alpha(t)`` for the proposal temperature (T_target / alpha(t)), the
        per-token weight, and the bonus draw; consecutive-step differences
        ``alpha(t) - alpha(t-1)`` drive the SMC-samplers boundary reweight.
        """
        a = self.smc_power_alpha
        T = self.smc_alpha_ramp_tokens
        if T <= 0:
            return torch.full_like(gen_lens, a, dtype=torch.float64)
        frac = (gen_lens.to(torch.float64) / T).clamp_(max=1.0)
        return 1.0 + (a - 1.0) * frac

    def _sample_target_power(self, logits):
        """Gumbel-max draw from the tempered-power target
        ``p_T^alpha ∝ softmax(alpha * logits / T_target)``.

        Returns ``(idx, logz)`` where ``idx`` is the sampled token and
        ``logz = logsumexp(alpha*logits/T) - alpha*logsumexp(logits/T)`` is the
        per-row log-normalizer of the power conditional (the bonus token's
        incremental importance weight under the joint-power target; 0 at
        alpha=1).  Shared by the per-block bonus draw and the prefill x0 draw so
        both come from one consistent target.
        """
        t = self.smc_target_temperature
        base = logits / t
        scaled = self.smc_power_alpha * base
        gumbel = -torch.log(
            -torch.log(
                torch.rand_like(scaled).clamp_min_(
                    torch.finfo(scaled.dtype).tiny
                )
            )
        )
        idx = torch.argmax(scaled + gumbel, dim=-1)
        logz = torch.logsumexp(scaled, dim=-1) - self.smc_power_alpha * (
            torch.logsumexp(base, dim=-1)
        )
        return idx, logz

    def _draft_ar_deferred(
        self, ctx, draft_input, draft_fb, cache_locs,
        all_positions, all_seq_lens, batch, bs, gamma,
    ):
        """Deferred-bonus draft AR (eager): head + gamma-1 single decodes, no
        over-draft.  Returns (all_tokens, draft_logprobs_stacked) with
        ``all_tokens = [verified_id, d_0, ..., d_{gamma-1}]`` (gamma+1 long, same
        shape the verify/bonus/weight code already consumes) and
        ``draft_logprobs_stacked`` of shape (bs, gamma).

        Every step runs the 2-token head ``[prev @ S-1, verified_id @ S]``
        (see ``prepare_for_draft_head``); d_0 is sampled from the S/bonus
        column, then gamma-1 singles.  On a group's FIRST decode step,
        ``prev`` is the last committed prompt token (seeded at
        ``allocate_slots``), so the S-1 write rewrites the prefill's draft
        KV byte-identically — no step-0 special case.  Per-row head
        selection is deliberately avoided: batches freely mix groups at
        different steps under continuous batching, and any batch-global
        step flag would mis-handle the joins (a -1 sentinel here used to
        reach the embedding as a token id and kill the scheduler).
        """
        x0 = draft_input.verified_id
        prev = draft_input.prev_last_draft_id
        assert prev is not None, (
            "deferred-bonus draft requires prev_last_draft_id "
            "(seeded at allocate_slots, carried by resample)"
        )

        all_tokens = [x0]
        draft_logprobs = []

        # 2-token head extend [prev @ S-1, verified_id @ S]; d_0 from the
        # second (S / bonus) column.
        head_fb = ctx.prepare_for_draft_head(
            prev, x0, cache_locs, self.req_to_token_pool, batch,
            self.draft_runner,
        )
        hgr = self.draft_head_graph_runner
        if hgr is not None and hgr.can_run(head_fb):
            # Graph path: replay the dedicated num_tokens_per_bs=2 head
            # runner.  replay() runs replay_prepare → attn metadata + buffer
            # copy itself, and returns a LogitsProcessorOutput directly.
            # replay() bypasses model_runner.forward (where _build_step_span_name
            # emits the trace span), so label it explicitly here.
            with torch.profiler.record_function(
                f"step[DECODE smc-head-graph bs={bs} toks={2 * bs}]"
            ):
                head_logits_full = hgr.replay(head_fb).next_token_logits
        else:
            # Eager fallback (no head graph captured, or bs beyond the
            # captured range).  The draft's *primary* graph runner is
            # decode-only and its can_run() keys on request count, so it
            # would wrongly replay a 1-token graph for this 2-token forward;
            # null it for just this call and init metadata eagerly.
            self.draft_runner.attn_backend.init_forward_metadata(head_fb)
            saved_gr = getattr(self.draft_runner, "graph_runner", None)
            self.draft_runner.graph_runner = None
            try:
                # Outer span labels the head; the inner forward emits a
                # step[TARGET_VERIFY ...] span (vendored naming), nested.
                with torch.profiler.record_function(
                    f"step[DECODE smc-head-eager bs={bs} toks={2 * bs}]"
                ):
                    head_logits_full = self.draft_runner.forward(
                        head_fb, skip_attn_backend_init=True
                    ).logits_output.next_token_logits
            finally:
                self.draft_runner.graph_runner = saved_gr
        # d_0 from the second (S / bonus) column of each req pair.
        head_logits = head_logits_full.reshape(bs, 2, -1)[:, 1, :]  # (bs, V)

        if self._smc_dbg_positions:
            used_graph = self.draft_head_graph_runner is not None
            n_nan = int(torch.isnan(head_logits).sum().item())
            n_inf = int(torch.isinf(head_logits).sum().item())
            print(
                f"[SMC_DBG] head graph={used_graph} "
                f"nan={n_nan} inf={n_inf} shape={tuple(head_logits.shape)}",
                flush=True,
            )

        d0, lp0 = self._sample_draft_token(head_logits)
        draft_logprobs.append(lp0)
        all_tokens.append(d0)
        current_ids = d0

        # gamma-1 single decodes: forward(d_{s-1}) @ S+s for s = 1..gamma-1.
        for step in range(1, gamma):
            draft_fb.input_ids = current_ids
            draft_fb.positions = all_positions[:, step].contiguous()
            draft_fb.out_cache_loc = cache_locs[:, step].contiguous()
            draft_fb.seq_lens = all_seq_lens[:, step].contiguous()
            draft_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * (step + 1)
            draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + (step + 1)
            out = self.draft_runner.forward(draft_fb)
            d, lp = self._sample_draft_token(out.logits_output.next_token_logits)
            draft_logprobs.append(lp)
            all_tokens.append(d)
            current_ids = d

        # No over-draft.  d_{gamma-1} (= all_tokens[gamma]) is kept + stashed as
        # next step's prev_last_draft_id by the caller.
        draft_logprobs_stacked = torch.stack(draft_logprobs, dim=1)  # (bs, gamma)
        return all_tokens, draft_logprobs_stacked

    def _forward_decode_cycle_graph(
        self,
        batch: ModelWorkerBatch,
        draft_input: SMCDraftInput,
        ctx: SMCDecodeContext,
    ) -> GenerationBatchResult:
        """Run one decode cycle through the full-cycle CUDA graph.

        Replaces prepare_for_draft + the draft AR loop + prepare_for_verify +
        the target verify forward + logprob extraction + bonus sampling with
        the staging copies, two metadata updates, and a single graph launch.
        Output tensors are views into the runner's persistent buffers; every
        consumer (scheduler write-back) is enqueued on the same stream before
        the next cycle's staging copies, so reuse is safe under both the
        sequential and overlapped event loops.
        """
        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma

        # Carried from the fused prepare kernel (the cycle's fresh pages);
        # legacy fallback re-reads the block table.
        cache_locs = ctx.cache_locs
        if cache_locs is None:
            from smcsd.common.verify import assign_smc_cache_locs_kernel

            r2t = self.req_to_token_pool.req_to_token
            out_cache_loc = torch.empty(
                bs * (gamma + 1), dtype=torch.int64, device=self.device
            )
            assign_smc_cache_locs_kernel[(bs,)](
                batch.req_pool_indices,
                r2t,
                ctx.orig_seq_lens,
                out_cache_loc,
                r2t.shape[1],
                gamma + 1,
            )
            cache_locs = out_cache_loc.reshape(bs, gamma + 1)

        replay_kwargs = {}
        if getattr(self.cycle_graph_runner, "deferred", False):
            replay_kwargs["prev_last_draft_id"] = draft_input.prev_last_draft_id
        # Exponent-bridging: per-row alpha(t) + matching proposal temperature.
        # Only sent when the ramp is on; otherwise the graph keeps its buffers at
        # the baked (power_alpha, draft_temperature) constants, so the fixed-alpha
        # path is byte-identical. temp(t) = draft_temp * power_alpha / alpha(t)
        # makes ramp-off exact for any configured --draft-temperature and flattens
        # the proposal early (alpha(t)->1) as the schedule intends.
        if self.smc_alpha_ramp_tokens > 0 and ctx.gen_lens is not None:
            alpha_vec = self._ramped_alpha(ctx.gen_lens)
            draft_temp_vec = self.smc_draft_temperature * (
                self.smc_power_alpha / alpha_vec
            )
            replay_kwargs["alpha_vec"] = alpha_vec.to(torch.float32)
            replay_kwargs["draft_temp_vec"] = draft_temp_vec.to(torch.float32)
        (
            tokens_out,
            _draft_logprobs,
            logprob_diff,
            bonus,
            bonus_logz,
            next_tokens,
        ) = self.cycle_graph_runner.replay(
            draft_input.verified_id,
            cache_locs,
            ctx,
            batch.req_pool_indices,
            **replay_kwargs,
        )

        next_token_ids = next_tokens.reshape(-1)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=self.device
        )
        # This step's last drafted token, deferred-bonus seed for next cycle.
        prev_last_draft_id = tokens_out[:, gamma]

        next_draft_input = SMCDraftInput(
            verified_id=bonus,
            prev_last_draft_id=prev_last_draft_id,
            logprob_diff=logprob_diff,
            bonus_logz=bonus_logz,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            next_draft_input=next_draft_input,
            logprob_diff=logprob_diff,
            can_run_cuda_graph=True,
        )

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

        # ---- 0. Full-cycle graph (issue #14, stage 3) ----
        # One launch covers draft AR + TARGET_VERIFY + weight diff + bonus,
        # skipping both ForwardBatch.init_new constructions and all per-phase
        # dispatch.  Falls through to the regular path when the batch exceeds
        # the captured bs/context caps.
        if self.cycle_graph_runner is not None and self.cycle_graph_runner.can_run(
            len(ctx.orig_seq_lens), ctx
        ):
            return self._forward_decode_cycle_graph(batch, draft_input, ctx)

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

        if self._smc_dbg_positions and self._smc_dbg_calls < 3:
            self._smc_dbg_calls += 1
            rp = int(batch.req_pool_indices[0].item())
            S = int(ctx.orig_seq_lens[0].item())
            new_S = int(ctx.new_seq_lens[0].item())
            vid = int(draft_input.verified_id[0].item())
            prev = (
                int(draft_input.prev_last_draft_id[0].item())
                if draft_input.prev_last_draft_id is not None
                else None
            )
            lo = max(0, S - 2)
            r2t = self.req_to_token_pool.req_to_token
            slots = r2t[rp, lo : S + gamma + 1].tolist()
            print(
                f"[SMC_DBG] decode#{self._smc_dbg_calls} bs={bs} gamma={gamma}\n"
                f"  req_pool_idx[0]={rp}  orig_seq_len[0]={S}  "
                f"new_seq_len[0]={new_S}  verified_id[0]={vid}  "
                f"prev_last_draft_id[0]={prev}\n"
                f"  all_positions[0]={all_positions[0].tolist()}\n"
                f"  cache_locs[0]={cache_locs[0].tolist()}\n"
                f"  req_to_token[{rp}, {lo}:{S + gamma + 1}]={slots}",
                flush=True,
            )

        # ---- 2. Dense draft AR ----
        if self.smc_defer_bonus and not draft_fb.forward_mode.is_idle():
            # Deferred-bonus schedule: head + gamma-1 singles, no over-draft.
            all_tokens, draft_logprobs_stacked = self._draft_ar_deferred(
                ctx, draft_input, draft_fb, cache_locs,
                all_positions, all_seq_lens, batch, bs, gamma,
            )
        elif (
            self.draft_phase_graph_runner is not None
            and not draft_fb.forward_mode.is_idle()
            and self.draft_phase_graph_runner.can_run(bs, ctx)
        ):
            # Whole-phase graph: one launch for all gamma+1 draft forwards +
            # in-graph Gumbel sampling (issue #14).  Returns the same
            # [x0, d_0..d_gamma] layout the verify/bonus code consumes.
            tokens_out, draft_logprobs_stacked = (
                self.draft_phase_graph_runner.replay(
                    draft_input.verified_id,
                    cache_locs,
                    ctx,
                    batch.req_pool_indices,
                )
            )
            all_tokens = [tokens_out[:, j] for j in range(gamma + 2)]
        else:
            # Legacy gamma+1 single-token AR loop (over-draft included).
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

                # Shared Gumbel-max + fused-logprob sampler; the over-draft
                # step (step == gamma) skips the logprob reduction since its
                # token never contributes to the importance weight.
                draft_idx, token_logprob = self._sample_draft_token(
                    logits, need_logprob=step < gamma
                )
                if step < gamma:
                    draft_logprobs.append(token_logprob)

                all_tokens.append(draft_idx)
                current_ids = draft_idx

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
            f"expected {expected_rows} (bs={bs}, gamma+1={gamma + 1}, "
            f"cuda_graph={can_run_cuda_graph})"
        )
        # Fused score-logprob extraction: we only need, per row, the logprob
        # of one already-chosen token under the tempered target
        # p_T = softmax(logits / T):
        #   logp = (logits/T)[token] - logsumexp(logits/T)
        # — one gather + one reduction instead of materializing the full
        # (bs*(gamma+1), vocab) log_softmax tensor.
        score_logits_3d = score_logits.reshape(bs, gamma + 1, -1)
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)
        verify_scaled = (
            score_logits_3d[:, :gamma, :] / self.smc_target_temperature
        )
        chosen_logits = verify_scaled.gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)
        score_logprobs_stacked = chosen_logits - torch.logsumexp(
            verify_scaled, dim=-1
        )

        # ---- 5. Logprob diff ----
        # Per-position (bs, gamma) importance-weight increment, NOT summed
        # over the block.  write_back_gpu masks out positions at/after
        # an EOS before summing, so a particle that terminates mid-block does
        # not accrue weight from the draft's post-EOS continuation tokens
        # (those are not part of the sequence — EOS is an absorbing state with
        # incremental weight 1).
        # Targets the (unnormalized) sequence-wise tempered-power distribution
        # p_{T_t}^alpha where p_{T_t}(x) = softmax(logits / T_t):
        #   log w = alpha * log p_{T_t}(x_t | x_{<t}) - log q(x_t | x_{<t}).
        logprob_diff = (
            self.smc_power_alpha * score_logprobs_stacked - draft_logprobs_stacked
        )

        # ---- 6. Bonus token ----
        # Sample from the same p_{T_t}^alpha distribution targeted above so the
        # bonus and per-step draws come from one consistent target.  ``bonus_logz``
        # is the bonus token's incremental importance weight log Z = logsumexp(
        # alpha*ℓ/T) - alpha*logsumexp(ℓ/T): the bonus is drawn from the locally
        # normalized power conditional p_T^alpha / Z, so under the joint-power
        # target its weight is Z, not 1 (identically 0 at alpha=1).  Accumulated
        # in write_back_gpu, gated by the same EOS/finish logic as logprob_diff.
        bonus, bonus_logz = self._sample_target_power(score_logits_3d[:, -1, :])

        # ---- 7. Output ----
        output_token_ids = torch.stack(
            all_tokens[1 : gamma + 1] + [bonus], dim=1
        )
        next_verified_id = bonus

        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=self.device
        )

        # This step's last *drafted* token d_{gamma-1} (= all_tokens[gamma];
        # all_tokens is [x0, d_0, ..., d_gamma], so index gamma is the last
        # kept draft token, index gamma+1 the discarded over-draft).  Deferred
        # into next step's leading 2-token draft forward.  Carried on
        # next_draft_input but NOT yet consumed by the draft loop — Step 2
        # wires the consumer and drops the over-draft.
        prev_last_draft_id = all_tokens[gamma]

        next_token_ids.record_stream(current_stream)
        accept_lens.record_stream(current_stream)
        next_verified_id.record_stream(current_stream)
        prev_last_draft_id.record_stream(current_stream)
        logprob_diff.record_stream(current_stream)
        bonus_logz.record_stream(current_stream)

        next_draft_input = SMCDraftInput(
            verified_id=next_verified_id,
            prev_last_draft_id=prev_last_draft_id,
            logprob_diff=logprob_diff,
            bonus_logz=bonus_logz,
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
