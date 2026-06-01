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
from copy import deepcopy
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

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
from sglang.srt.speculative.dflash_worker import DFlashWorker
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_accept_len_and_bonus,
    compute_dflash_sampling_accept_len_and_bonus,
    is_dflash_sampling_verify_available,
    parse_dflash_draft_config,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

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
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self.device = server_args.device
        self._target_worker = target_worker  # score model

        self.gamma = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = self.gamma + 1
        self.smc_draft_temperature = server_args.smc_draft_temperature
        self.smc_target_temperature = max(
            float(server_args.smc_target_temperature), 1e-5
        )
        self.smc_draft_mode = getattr(server_args, "smc_draft_mode", "dense")
        self.is_dflash = self.smc_draft_mode == "dflash"
        self.is_eagle3 = self.smc_draft_mode == "eagle3"
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
        self._t_accept_tokens = 0.0
        self._t_accept_reqs = 0
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
        SMCDraftInputV2.ALLOC_LEN_PER_DECODE = self.speculative_num_draft_tokens

        server_args.context_length = target_worker.model_runner.model_config.context_len
        self.score_runner = self._target_worker.model_runner
        if self.is_dflash:
            self._draft_worker = None
            self.draft_runner = None
            self.draft_attn_backend = None
            self.hot_token_id = None
            self._t2d_map = None
            self._init_dflash_direct()
            return

        # Override context length of draft model to match score model.
        # For EAGLE3, the draft head has a small max_position_embeddings (2048)
        # in its config, but it operates within the target's full context window.
        # We set the env var to suppress the validation error before constructing
        # the draft TpModelWorker.
        _eagle3_mode = self.is_eagle3
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
        # Do not capture cuda graph during TpModelWorker init —
        # we capture manually after the draft model is fully set up
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Create draft TpModelWorker — fully independent, no shared lm_head/embed.
        # Dense SMC uses a normal AR draft; EAGLE3 keeps upstream draft-model
        # config rewriting because its checkpoint is an EAGLE/MTP-style head.
        draft_worker_cls = TpModelWorker if self.is_eagle3 else SMCDenseDraftTpModelWorker
        self._draft_worker = draft_worker_cls(
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

        # EAGLE3 draft mode: share target embed/lm_head with draft head,
        # then disable the draft CUDA graph (EAGLE head runs eagerly).
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
        else:
            self._maybe_isolate_dense_hybrid_draft_state()

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

        # --- Hierarchical (nested) SMC: separate larger SCORE model ----------
        # In nested mode the eagle3 ``target_worker`` above is the small EAGLE
        # *base* (e.g. Qwen3-8B): it supplies the head's fused hidden states and
        # its own KV/rewrite. A separate, larger SCORE model (e.g. Qwen3-32B)
        # supplies the SMC importance-weight target ``p`` and the bonus token.
        # The EAGLE head only has to be a fast proposer of the 8B's
        # distribution; SMC reweights its proposals toward the 32B target.
        # Enabled via the SMCSD_SCORE_MODEL env var (path to the score model).
        # Requires the base and score models to share a tokenizer/vocab.
        self._score_worker = None
        self._score_runner = None
        self._has_score_model = False
        score_model_path = os.environ.get("SMCSD_SCORE_MODEL")
        if score_model_path and self.is_eagle3:
            score_server_args = deepcopy(server_args)
            # SMCDenseDraftTpModelWorker loads speculative_draft_model_path as a
            # plain autoregressive LM (no MTP/EAGLE rewrite) sharing the slot pool.
            score_server_args.speculative_draft_model_path = score_model_path
            score_server_args.disable_cuda_graph = backup_disable_cuda_graph
            self._score_worker = SMCDenseDraftTpModelWorker(
                server_args=score_server_args,
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
            self._score_runner = self._score_worker.model_runner
            self._has_score_model = True
            logging.getLogger(__name__).warning(
                "[SMC NESTED] hierarchical: EAGLE base=%s + head=%s, "
                "SCORE target=%s",
                server_args.model_path,
                server_args.speculative_draft_model_path,
                score_model_path,
            )

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
        if not hasattr(target_pool, "mamba_pool") or draft_config is None:
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
            linear_backend = self.draft_runner.attn_backend.linear_attn_backend
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
        logger.warning(
            "SMC dense mode isolated hybrid draft state/KV: target=%s shape=%s "
            "draft=%s shape=%s full_attn_layers=%s",
            self.score_runner.model_config.model_path,
            target_shape,
            self.draft_runner.model_config.model_path,
            draft_shape,
            self.draft_runner.token_to_kv_pool.full_attention_layer_id_mapping.keys(),
        )

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

    def _init_dflash_direct(self) -> None:
        """Initialize SGLang's native DFlash draft runner for SMC.

        The old SMC DFlash path loaded the checkpoint through Hugging Face and
        replayed Python-level DynamicCache updates per particle. Upstream SGLang
        gets its speed by keeping DFlash in a TpModelWorker, materializing target
        hidden states into the draft KV pool, and running a fixed TARGET_VERIFY
        block through the native attention backend. SMC keeps its own slot
        scheduler, but uses that same native draft runner here.
        """
        target_model = self._target_worker.model_runner.model

        dflash_server_args = deepcopy(self.server_args)
        dflash_server_args.speculative_algorithm = "DFLASH"

        self._native_dflash_worker = DFlashWorker(
            server_args=dflash_server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            dp_rank=self.dp_rank,
            moe_ep_rank=self.moe_ep_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
            nccl_port=self.nccl_port,
            target_worker=self._target_worker,
        )
        self._draft_worker = self._native_dflash_worker.draft_worker
        self.draft_runner = self._native_dflash_worker.draft_model_runner
        self.draft_attn_backend = self.draft_runner.attn_backend
        self.dflash_model = self._native_dflash_worker.draft_model
        self._dflash_block_size = int(self._native_dflash_worker.block_size)
        self._dflash_mask_token_id = int(self._native_dflash_worker._mask_token_id)
        self._dflash_device = self._native_dflash_worker.device
        embed_module = target_model.get_input_embeddings()
        self._dflash_dtype = getattr(embed_module, "weight", None).dtype

        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.draft_runner.model_config.hf_config
        )
        target_num_layers = int(
            getattr(
                self._target_worker.model_runner.model_config.hf_text_config,
                "num_hidden_layers",
            )
        )
        self._dflash_target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers,
            draft_num_layers=draft_config.require_num_layers(),
        )
        if hasattr(target_model, "set_dflash_layers_to_capture"):
            target_model.set_dflash_layers_to_capture(self._dflash_target_layer_ids)
        elif hasattr(target_model, "set_eagle3_layers_to_capture"):
            target_model.set_eagle3_layers_to_capture(self._dflash_target_layer_ids)
        else:
            raise ValueError(
                "Target model does not expose a DFlash/EAGLE hidden-state capture hook; "
                "DFlash needs selected target hidden states."
            )

        if self._dflash_block_size != self.speculative_num_draft_tokens:
            logger.warning(
                "SMC DFlash using block_size=%d while gamma+1=%d. "
                "Pass --gamma %d to match this draft checkpoint's native block size.",
                self._dflash_block_size,
                self.speculative_num_draft_tokens,
                self._dflash_block_size - 1,
            )

        logger.info(
            "Initialized native DFlash draft runner: path=%s layers=%s block_size=%d",
            self.server_args.speculative_draft_model_path,
            self._dflash_target_layer_ids,
            self._dflash_block_size,
        )

    @torch.inference_mode()
    def _materialize_dflash_hidden_to_native_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
    ) -> None:
        if target_hidden is None or target_hidden.numel() == 0:
            return
        native = self._native_dflash_worker
        target_hidden = target_hidden.to(device=native.device)
        positions = positions.to(device=native.device, dtype=torch.int64).reshape(-1)
        cache_locs = cache_locs.to(device=native.device, dtype=torch.int64).reshape(-1)
        if target_hidden.shape[0] != cache_locs.numel():
            raise RuntimeError(
                f"DFlash hidden/cache length mismatch: {target_hidden.shape[0]} vs {cache_locs.numel()}."
            )

        ctx_hidden = native.draft_model.project_target_hidden(target_hidden)
        if native._use_fused_kv_materialize and native._fused_kv_helper is not None:
            try:
                native._append_target_hidden_fused(ctx_hidden, positions, cache_locs)
                return
            except Exception as exc:
                logger.warning(
                    "SMC DFlash fused KV append failed; falling back to sequential path: %s",
                    exc,
                )
                native._use_fused_kv_materialize = False
                native._fused_kv_helper = None
        native._append_target_hidden_sequential(ctx_hidden, positions, cache_locs)

    def _dflash_prefill_positions(self, batch: ModelWorkerBatch) -> torch.Tensor:
        positions = []
        assert batch.extend_seq_lens is not None
        prefix_lens = batch.extend_prefix_lens or [0] * len(batch.extend_seq_lens)
        for prefix_len, extend_len in zip(prefix_lens, batch.extend_seq_lens):
            start = int(prefix_len)
            end = start + int(extend_len)
            positions.append(
                torch.arange(start, end, dtype=torch.int64, device=self.device)
            )
        if not positions:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return torch.cat(positions, dim=0)

    @torch.inference_mode()
    def _dflash_native_propose_batch(
        self,
        *,
        batch: ModelWorkerBatch,
        ctx: SMCDecodeContext,
        x0: torch.Tensor,
        cache_locs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        native = self._native_dflash_worker
        bs = int(x0.shape[0])
        draft_len = int(self.speculative_num_draft_tokens)
        gamma = draft_len - 1
        device = native.device

        native._ensure_draft_block_buffers(bs)
        block_ids = native._draft_block_ids_buf[:bs]
        positions_2d = native._draft_block_positions_buf[:bs]
        draft_tokens = native._draft_block_tokens_buf[:bs]
        seq_lens_cpu = native._draft_seq_lens_cpu_buf[:bs]
        assert block_ids is not None
        assert positions_2d is not None
        assert draft_tokens is not None
        assert seq_lens_cpu is not None

        block_ids[:, :draft_len].fill_(int(self._dflash_mask_token_id))
        block_ids[:, 0].copy_(x0.to(device=device, dtype=torch.long))

        target_model = self._target_worker.model_runner.model
        input_embeds = target_model.get_input_embeddings()(
            block_ids[:, :draft_len]
        ).reshape(bs * draft_len, -1)
        positions_2d[:, :draft_len].copy_(
            ctx.orig_seq_lens.to(device=device, dtype=torch.int64).unsqueeze(1)
            + torch.arange(draft_len, device=device, dtype=torch.int64).unsqueeze(0)
        )
        positions = positions_2d[:, :draft_len].reshape(-1)

        draft_prefix_lens = ctx.orig_seq_lens.to(device=device, dtype=torch.int32)
        seq_lens_cpu.copy_(ctx.orig_seq_lens_cpu.to(dtype=torch.int32))

        allocator = native.draft_model_runner.token_to_kv_pool_allocator
        token_to_kv_pool_state_backup = allocator.backup_state()
        block_cache_loc = None
        try:
            block_cache_loc = allocator.alloc(bs * draft_len)
            if block_cache_loc is None:
                raise RuntimeError(
                    f"SMC DFlash draft OOM when allocating {bs * draft_len} block tokens."
                )
            block_start = draft_prefix_lens
            block_end = draft_prefix_lens + draft_len
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                native.draft_model_runner.req_to_token_pool.req_to_token,
                block_start,
                block_end,
                block_cache_loc,
                bs,
            )

            draft_spec_info = native._draft_block_spec_info
            draft_spec_info.draft_token_num = draft_len
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                input_ids=block_ids[:, :draft_len].reshape(-1),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=draft_prefix_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=ctx.orig_seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                positions=positions,
                req_to_token_pool=native.draft_model_runner.req_to_token_pool,
                token_to_kv_pool=native.draft_model_runner.token_to_kv_pool,
                attn_backend=native.draft_model_runner.attn_backend,
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=draft_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            draft_logits_output = native.draft_model_runner.forward(
                forward_batch
            ).logits_output
        finally:
            allocator.restore_state(token_to_kv_pool_state_backup)
            # The native draft worker temporarily writes draft-block locations
            # into req_to_token. Restore SMC's target verify locations before
            # the target verify pass consumes them.
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.req_to_token_pool.req_to_token,
                ctx.orig_seq_lens.to(torch.int32),
                (ctx.orig_seq_lens + draft_len).to(torch.int32),
                cache_locs.reshape(-1),
                bs,
            )

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("SMC DFlash draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, draft_len, -1)
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("DFlash requires the target model to expose lm_head.")
        hidden_for_head = draft_hidden[:, 1:draft_len, :].reshape(
            bs * gamma, draft_hidden.shape[-1]
        )
        sample_proposals = (
            os.getenv("SMCSD_DFLASH_SAMPLE_PROPOSALS", "0") == "1"
            and self.smc_draft_temperature > 0
        )
        draft_logits = (
            self._dflash_project_target_head_logits(
                hidden_states=hidden_for_head,
                lm_head=lm_head,
            )
            if sample_proposals
            else None
        )
        if draft_logits is None:
            draft_next = native._greedy_sample_from_vocab_parallel_head(
                hidden_states=hidden_for_head,
                lm_head=lm_head,
            ).view(bs, gamma)
            draft_logprobs = torch.zeros(
                (bs, gamma), dtype=torch.float32, device=self.device
            )
            sample_proposals = False
        else:
            draft_logits = draft_logits.view(bs, gamma, -1)
            draft_log_probs = torch.log_softmax(
                draft_logits / self.smc_draft_temperature, dim=-1
            )
            draft_next = torch.multinomial(
                draft_log_probs.reshape(bs * gamma, -1).exp(),
                num_samples=1,
            ).view(bs, gamma)
            draft_logprobs = draft_log_probs.gather(
                2, draft_next.unsqueeze(2)
            ).squeeze(2)
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:draft_len].copy_(draft_next)
        return (
            draft_next.to(device=self.device, dtype=torch.long),
            draft_logprobs.to(device=self.device, dtype=torch.float32),
            sample_proposals,
        )

    def _dflash_project_target_head_logits(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
    ) -> Optional[torch.Tensor]:
        """Materialize target-vocab logits for DFlash proposal sampling.

        This fast path intentionally supports the common tp=1/no-added-vocab
        setup used by the Llama DFlash checkpoint. TP-safe greedy fallback stays
        available through SGLang's native helper when the head layout is more
        complex.
        """
        if int(getattr(self.server_args, "tp_size", 1)) != 1:
            return None
        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            return None

        shard = lm_head.shard_indices
        num_org = int(shard.num_org_elements)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        if num_org <= 0 or num_added != 0 or org_vocab_start != 0:
            return None

        weight = lm_head.weight[:num_org]
        hidden_states = hidden_states.to(weight.dtype)
        return torch.matmul(hidden_states, weight.T)

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

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if temperature > 0:
            scaled = logits / temperature
            log_probs = torch.log_softmax(scaled, dim=-1)
            token_ids = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
        else:
            log_probs = torch.log_softmax(logits, dim=-1)
            token_ids = torch.argmax(logits, dim=-1)
        token_logprobs = log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        return token_ids, token_logprobs

    def _prepare_dflash_cache_metadata(
        self,
        ctx: SMCDecodeContext,
        batch: ModelWorkerBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from smcsd.common.verify import assign_smc_cache_locs_kernel

        orig_seq_lens = ctx.orig_seq_lens
        bs = len(orig_seq_lens)
        device = orig_seq_lens.device
        gamma = ctx.gamma

        out_cache_loc = torch.empty(
            bs * (gamma + 1), dtype=torch.int64, device=device
        )
        assign_smc_cache_locs_kernel[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            orig_seq_lens,
            out_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            gamma + 1,
        )
        cache_locs = out_cache_loc.reshape(bs, gamma + 1)

        step_offsets = torch.arange(gamma + 1, device=device)
        all_positions = orig_seq_lens.unsqueeze(1) + step_offsets
        all_seq_lens = all_positions + 1
        return cache_locs, all_positions, all_seq_lens

    def _make_dflash_attention_mask(
        self,
        *,
        context_len: int,
        draft_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        # One DFlash block anchored at context_len. The block can see all target
        # context tokens and all positions inside its masked draft block.
        return torch.ones(
            (1, 1, draft_len, context_len + draft_len),
            dtype=torch.bool,
            device=device,
        )

    @torch.inference_mode()
    def _merge_dflash_caches(self, draft_caches: list[Any]) -> Any:
        from transformers import DynamicCache

        if not draft_caches or draft_caches[0] is None:
            return DynamicCache()
        if int(draft_caches[0].get_seq_length()) == 0:
            return DynamicCache()

        cache_data = []
        for layers in zip(*draft_caches, strict=True):
            keys = [layer[0] for layer in layers]
            values = [layer[1] for layer in layers]
            sliding_windows = [layer[2] for layer in layers]
            if keys[0] is None or values[0] is None:
                cache_data.append((keys[0], values[0]))
            elif sliding_windows[0] is None:
                cache_data.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
            else:
                cache_data.append(
                    (
                        torch.cat(keys, dim=0),
                        torch.cat(values, dim=0),
                        sliding_windows[0],
                    )
                )
        return DynamicCache(ddp_cache_data=cache_data)

    @torch.inference_mode()
    def _split_dflash_cache(self, draft_cache: Any, bs: int) -> list[Any]:
        from transformers import DynamicCache

        if draft_cache is None:
            return [DynamicCache() for _ in range(bs)]

        split_caches = []
        for i in range(bs):
            cache_data = []
            for key, value, sliding_window in draft_cache:
                if key is None or value is None:
                    cache_data.append((key, value))
                elif sliding_window is None:
                    cache_data.append(
                        (key[i : i + 1].contiguous(), value[i : i + 1].contiguous())
                    )
                else:
                    cache_data.append(
                        (
                            key[i : i + 1].contiguous(),
                            value[i : i + 1].contiguous(),
                            sliding_window,
                        )
                    )
            split_caches.append(DynamicCache(ddp_cache_data=cache_data))
        return split_caches

    @torch.inference_mode()
    def _dflash_propose_cached_batch(
        self,
        *,
        target_hidden_deltas: torch.Tensor,
        verified_ids: torch.Tensor,
        draft_caches: list[Any],
        position_start: int,
        gamma: int,
    ) -> Tuple[torch.Tensor, list[Any]]:
        draft_len = gamma + 1
        device = self._dflash_device
        bs = int(target_hidden_deltas.shape[0])
        draft_cache = self._merge_dflash_caches(draft_caches)
        cache_len = int(draft_cache.get_seq_length())
        delta_len = int(target_hidden_deltas.shape[1])

        noise_ids = torch.full(
            (bs, draft_len),
            int(self._dflash_mask_token_id),
            dtype=torch.long,
            device=device,
        )
        noise_ids[:, 0] = verified_ids.to(device=device, dtype=torch.long)
        noise_embedding = F.embedding(noise_ids, self._dflash_embed_weight)

        position_ids = torch.arange(
            cache_len,
            position_start + draft_len,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(bs, -1)
        attention_mask = self._make_dflash_attention_mask(
            context_len=cache_len + delta_len,
            draft_len=draft_len,
            device=device,
        ).expand(bs, -1, -1, -1)

        hidden = self.dflash_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden_deltas.to(
                device=device, dtype=self._dflash_dtype
            ),
            attention_mask=attention_mask,
            past_key_values=draft_cache,
            use_cache=True,
            is_causal=False,
        )
        draft_cache.crop(position_start)
        logits = torch.matmul(
            hidden[:, 1:, :].to(self._dflash_lm_head_weight.dtype),
            self._dflash_lm_head_weight.T,
        )
        return torch.argmax(logits, dim=-1), self._split_dflash_cache(draft_cache, bs)

    @torch.inference_mode()
    def _dflash_propose_one_cached(
        self,
        *,
        target_hidden_delta: torch.Tensor,
        verified_id: torch.Tensor,
        draft_cache: Any,
        position_start: int,
        gamma: int,
    ) -> Tuple[torch.Tensor, Any]:
        draft_len = gamma + 1
        device = self._dflash_device
        if draft_cache is None:
            from transformers import DynamicCache

            draft_cache = DynamicCache()
        cache_len = int(draft_cache.get_seq_length())
        delta_len = int(target_hidden_delta.shape[0])

        noise_ids = torch.full(
            (1, draft_len),
            int(self._dflash_mask_token_id),
            dtype=torch.long,
            device=device,
        )
        noise_ids[:, 0] = verified_id.to(device=device, dtype=torch.long)
        noise_embedding = F.embedding(noise_ids, self._dflash_embed_weight)

        position_ids = torch.arange(
            cache_len,
            position_start + draft_len,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        attention_mask = self._make_dflash_attention_mask(
            context_len=cache_len + delta_len,
            draft_len=draft_len,
            device=device,
        )

        hidden = self.dflash_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden_delta.unsqueeze(0).to(
                device=device, dtype=self._dflash_dtype
            ),
            attention_mask=attention_mask,
            past_key_values=draft_cache,
            use_cache=True,
            is_causal=False,
        )
        draft_cache.crop(position_start)
        logits = torch.matmul(
            hidden[:, 1:, :].to(self._dflash_lm_head_weight.dtype),
            self._dflash_lm_head_weight.T,
        ).squeeze(0)
        return torch.argmax(logits, dim=-1), draft_cache

    @torch.inference_mode()
    def _dflash_propose_batch(
        self,
        target_hidden_contexts: torch.Tensor,
        verified_ids: torch.Tensor,
        position_start: int,
        gamma: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        draft_len = gamma + 1
        device = self._dflash_device
        bs = int(target_hidden_contexts.shape[0])
        noise_ids = torch.full(
            (bs, draft_len),
            int(self._dflash_mask_token_id),
            dtype=torch.long,
            device=device,
        )
        noise_ids[:, 0] = verified_ids.to(device=device, dtype=torch.long)
        noise_embedding = F.embedding(noise_ids, self._dflash_embed_weight)

        position_ids = torch.arange(
            position_start + draft_len,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(bs, -1)
        attention_mask = self._make_dflash_attention_mask(
            context_len=int(target_hidden_contexts.shape[1]),
            draft_len=draft_len,
            device=device,
        ).expand(bs, -1, -1, -1)

        hidden = self.dflash_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden_contexts.to(
                device=device, dtype=self._dflash_dtype
            ),
            attention_mask=attention_mask,
            is_causal=False,
        )
        logits = torch.matmul(
            hidden[:, 1:, :].to(self._dflash_lm_head_weight.dtype),
            self._dflash_lm_head_weight.T,
        )

        sampled_ids = torch.argmax(logits, dim=-1)
        sampled_logprobs = torch.zeros(
            (bs, gamma), dtype=torch.float32, device=device
        )
        return sampled_ids.view(bs, gamma), sampled_logprobs

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

        if self.is_dflash:
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            score_result = self._target_worker.forward_batch_generation(batch)

            target_h = score_result.logits_output.hidden_states
            if target_h is None:
                raise RuntimeError("DFlash prefill requires target hidden states.")
            prefill_positions = self._dflash_prefill_positions(batch)
            self._materialize_dflash_hidden_to_native_kv(
                target_hidden=target_h,
                positions=prefill_positions,
                cache_locs=batch.out_cache_loc,
            )

            score_result.next_draft_input = SMCDraftInputV2(
                verified_id=score_result.next_token_ids,
                num_tokens_per_req=self.speculative_num_draft_tokens,
            )
            score_result.accept_lens = torch.zeros(
                bs, dtype=torch.int32, device=self.device
            )
            return score_result

        if self.is_eagle3:
            # Nested mode: snapshot the prompt batch BEFORE mutation so the
            # larger SCORE model can prefill its own KV over the same slots.
            # Without this the score model would verify with empty prompt
            # context and produce meaningless importance weights.
            score_prefill_batch = (
                self._make_clean_batch(batch) if self._has_score_model else None
            )

            # Target prefill — request aux hidden states for EAGLE3 head input
            chm = (
                CaptureHiddenMode.FULL if self.eagle_use_aux
                else CaptureHiddenMode.LAST
            )
            batch.capture_hidden_mode = chm
            score_result = self._target_worker.forward_batch_generation(batch)

            # Nested SCORE model prefill: build the score model's KV over the
            # prompt slots (same req_to_token slots; separate KV tensor).
            if score_prefill_batch is not None:
                self._score_worker.forward_batch_generation(score_prefill_batch)

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

        if self.is_dflash:
            cache_locs, all_positions, _ = self._prepare_dflash_cache_metadata(
                ctx, batch
            )
            return self._forward_decode_dflash(
                batch,
                draft_input,
                ctx,
                cache_locs,
                all_positions,
                current_stream,
            )

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

    def _record_decode_timing(
        self,
        mode: str,
        ev_t0: torch.cuda.Event,
        ev_draft_end: torch.cuda.Event,
        ev_verify_end: torch.cuda.Event,
        ev_other_end: torch.cuda.Event,
        accept_lens: torch.Tensor | None = None,
    ) -> None:
        if not self._timing_enabled:
            return

        ev_other_end.record()
        ev_other_end.synchronize()
        self._t_draft_ms += ev_t0.elapsed_time(ev_draft_end)
        self._t_verify_ms += ev_draft_end.elapsed_time(ev_verify_end)
        self._t_other_ms += ev_verify_end.elapsed_time(ev_other_end)
        self._t_steps += 1
        if accept_lens is not None and accept_lens.numel() > 0:
            self._t_accept_tokens += float(accept_lens.float().sum().item())
            self._t_accept_reqs += int(accept_lens.numel())

        if self._t_steps % self._timing_every == 0:
            tot = self._t_draft_ms + self._t_verify_ms + self._t_other_ms
            avg_accept = (
                self._t_accept_tokens / self._t_accept_reqs
                if self._t_accept_reqs > 0
                else float("nan")
            )
            print(
                f"[SMC TIMING] {mode} tp{self.tp_rank} steps={self._t_steps} "
                f"draft={self._t_draft_ms:.0f}ms ({100 * self._t_draft_ms / max(tot, 1e-6):.1f}%) "
                f"verify={self._t_verify_ms:.0f}ms ({100 * self._t_verify_ms / max(tot, 1e-6):.1f}%) "
                f"other={self._t_other_ms:.0f}ms ({100 * self._t_other_ms / max(tot, 1e-6):.1f}%) "
                f"avg/step={tot / self._t_steps:.1f}ms "
                f"avg_accept={avg_accept:.2f}",
                flush=True,
            )

    def _forward_decode_dflash(
        self,
        batch: ModelWorkerBatch,
        draft_input: SMCDraftInputV2,
        ctx: SMCDecodeContext,
        cache_locs: torch.Tensor,
        all_positions: torch.Tensor,
        current_stream,
    ):
        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma

        if self._timing_enabled:
            ev_t0 = torch.cuda.Event(enable_timing=True)
            ev_draft_end = torch.cuda.Event(enable_timing=True)
            ev_verify_end = torch.cuda.Event(enable_timing=True)
            ev_other_end = torch.cuda.Event(enable_timing=True)
            ev_t0.record()

        x0 = draft_input.verified_id.to(torch.long)
        proposed_tokens, draft_logprobs, used_sampled_proposals = (
            self._dflash_native_propose_batch(
                batch=batch,
                ctx=ctx,
                x0=x0,
                cache_locs=cache_locs,
            )
        )
        if self._timing_enabled:
            ev_draft_end.record()
        all_tokens = [x0] + [
            proposed_tokens[:, step].contiguous() for step in range(gamma)
        ]
        use_full_block_smc = os.getenv("SMCSD_DFLASH_FULL_BLOCK_SMC", "0") == "1"
        use_target_rewrite = (
            use_full_block_smc
            and os.getenv("SMCSD_DFLASH_TARGET_REWRITE", "1") != "0"
        )

        verify_forward_batch, can_run_cuda_graph = ctx.prepare_for_verify(
            self.req_to_token_pool,
            batch,
            self._target_worker,
            all_tokens,
            cache_locs,
            capture_hidden_mode=(
                CaptureHiddenMode.NULL if use_target_rewrite else CaptureHiddenMode.FULL
            ),
        )

        score_result = self._target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        if self._timing_enabled:
            ev_verify_end.record()

        score_logits = score_result.logits_output.next_token_logits
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]} rows, "
            f"expected {expected_rows} (bs={bs}, gamma+1={gamma + 1})"
        )

        if not use_full_block_smc:
            candidates = torch.cat(
                [x0.unsqueeze(1), proposed_tokens], dim=1
            ).contiguous()
            use_sampling_verify = (
                os.getenv("SMCSD_DFLASH_SAMPLING_VERIFY", "0") == "1"
                and is_dflash_sampling_verify_available()
                and self.smc_target_temperature > 1e-4
            )
            if use_sampling_verify:
                accepted_draft_lens, bonus = (
                    compute_dflash_sampling_accept_len_and_bonus(
                        candidates=candidates,
                        next_token_logits=score_logits,
                        sampling_info=batch.sampling_info,
                    )
                )
                target_predict = None
            else:
                target_predict = torch.argmax(score_logits, dim=-1).reshape(
                    bs, gamma + 1
                )
                accepted_draft_lens, bonus = compute_dflash_accept_len_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )
            accepted_draft_lens = accepted_draft_lens.to(
                device=self.device, dtype=torch.long
            )
            min_accept = max(0, int(os.getenv("SMCSD_DFLASH_MIN_ACCEPT", "0")))
            use_min_accept = min_accept > 0 and not use_sampling_verify
            if use_min_accept:
                min_accept = min(min_accept, gamma)
                committed_draft_lens = torch.maximum(
                    accepted_draft_lens,
                    torch.full_like(accepted_draft_lens, min_accept),
                )
                if target_predict is None:
                    target_predict = torch.argmax(score_logits, dim=-1).reshape(
                        bs, gamma + 1
                    )
                bonus = target_predict.gather(
                    1, committed_draft_lens.unsqueeze(1)
                ).squeeze(1)
            else:
                committed_draft_lens = accepted_draft_lens
            accept_lens = (committed_draft_lens + 1).to(torch.int32)

            output_token_ids = torch.zeros(
                (bs, gamma + 1), dtype=proposed_tokens.dtype, device=self.device
            )
            for i in range(bs):
                n_accept = int(committed_draft_lens[i].item())
                if n_accept > 0:
                    output_token_ids[i, :n_accept] = proposed_tokens[i, :n_accept]
                output_token_ids[i, n_accept] = bonus[i].to(proposed_tokens.dtype)

            next_token_ids = output_token_ids.reshape(-1)
            next_verified_id = bonus.to(dtype=torch.long)

            if used_sampled_proposals or use_min_accept:
                score_log_probs = torch.log_softmax(
                    score_logits.reshape(bs, gamma + 1, -1)[:, :gamma, :],
                    dim=-1,
                )
                committed_mask = (
                    torch.arange(gamma, device=self.device).unsqueeze(0)
                    < committed_draft_lens.unsqueeze(1)
                )
                score_logprobs = score_log_probs.gather(
                    2, proposed_tokens.unsqueeze(2)
                ).squeeze(2)
                if used_sampled_proposals:
                    logprob_terms = score_logprobs - draft_logprobs
                    weight_mask = committed_mask
                else:
                    extra_mask = (
                        torch.arange(gamma, device=self.device).unsqueeze(0)
                        >= accepted_draft_lens.unsqueeze(1)
                    ) & committed_mask
                    # The verified prefix already follows the target's greedy
                    # accept rule. Only forced post-mismatch tokens need SMC
                    # correction, otherwise long naturally accepted blocks get
                    # penalized purely for being long.
                    logprob_terms = score_logprobs
                    weight_mask = extra_mask
                logprob_diff = (logprob_terms * weight_mask.to(score_logprobs.dtype)).sum(
                    dim=1
                )
            else:
                logprob_diff = torch.zeros(
                    bs, dtype=torch.float32, device=self.device
                )

            h_all = score_result.logits_output.hidden_states
            if h_all is None:
                raise RuntimeError(
                    "DFlash verify must capture FULL target hidden states."
                )
            aux_dim = h_all.shape[-1]
            target_h_steps = h_all.reshape(bs, gamma + 1, aux_dim)
            commit_mask = (
                torch.arange(gamma + 1, device=self.device).unsqueeze(0)
                < accept_lens.to(dtype=torch.long).unsqueeze(1)
            )
            self._materialize_dflash_hidden_to_native_kv(
                target_hidden=target_h_steps[commit_mask].contiguous(),
                positions=all_positions[commit_mask].contiguous(),
                cache_locs=cache_locs[commit_mask].contiguous(),
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
                self._record_decode_timing(
                    "dflash",
                    ev_t0,
                    ev_draft_end,
                    ev_verify_end,
                    ev_other_end,
                    accept_lens,
                )

            return GenerationBatchResult(
                logits_output=score_result.logits_output,
                next_token_ids=next_token_ids,
                accept_lens=accept_lens,
                next_draft_input=next_draft_input,
                logprob_diff=logprob_diff,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        if use_target_rewrite:
            anchor_logits = score_logits.reshape(bs, gamma + 1, -1)[:, :gamma, :]
            if self.smc_target_temperature > 0:
                proposal_log_probs = torch.log_softmax(
                    anchor_logits / self.smc_target_temperature, dim=-1
                )
                rewritten_tokens = torch.multinomial(
                    proposal_log_probs.reshape(bs * gamma, -1).exp(),
                    num_samples=1,
                ).view(bs, gamma)
            else:
                proposal_log_probs = torch.log_softmax(anchor_logits, dim=-1)
                rewritten_tokens = torch.argmax(anchor_logits, dim=-1)
            proposal_logprobs = proposal_log_probs.gather(
                2, rewritten_tokens.unsqueeze(2)
            ).squeeze(2)

            all_tokens = [x0] + [
                rewritten_tokens[:, step].contiguous() for step in range(gamma)
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
            if self._timing_enabled:
                ev_verify_end.record()
            score_logits = score_result.logits_output.next_token_logits
            assert score_logits.shape[0] == expected_rows, (
                f"TARGET_VERIFY rewrite logits truncated: got {score_logits.shape[0]} rows, "
                f"expected {expected_rows} (bs={bs}, gamma+1={gamma + 1})"
            )
            proposed_tokens = rewritten_tokens.to(device=self.device, dtype=torch.long)
            draft_logprobs = proposal_logprobs.to(
                device=self.device, dtype=torch.float32
            )
            used_sampled_proposals = True

        # SMC semantics: do not reject or truncate DFlash's block.  Commit the
        # whole proposal and let particle weights/resampling correct low-quality
        # blocks, matching the dense/EAGLE SMC paths.
        score_log_probs = torch.log_softmax(
            score_logits.reshape(bs, gamma + 1, -1)[:, :gamma, :], dim=-1
        )
        score_logprobs = score_log_probs.gather(
            2, proposed_tokens.unsqueeze(2)
        ).squeeze(2)
        if used_sampled_proposals:
            logprob_diff = (score_logprobs - draft_logprobs).sum(dim=1)
        else:
            # Deterministic greedy proposal: q is a point mass on the selected
            # block, so log q contributes zero for the chosen trajectory.
            logprob_diff = score_logprobs.sum(dim=1)

        bonus_logits = score_logits.reshape(bs, gamma + 1, -1)[:, -1, :]
        bonus_log_probs = torch.log_softmax(
            bonus_logits / self.smc_target_temperature, dim=-1
        )
        bonus = torch.multinomial(bonus_log_probs.exp(), num_samples=1).squeeze(-1)

        output_token_ids = torch.cat(
            [proposed_tokens, bonus.to(proposed_tokens.dtype).unsqueeze(1)], dim=1
        )
        next_token_ids = output_token_ids.reshape(-1)
        next_verified_id = bonus.to(dtype=torch.long)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=self.device
        )

        h_all = score_result.logits_output.hidden_states
        if h_all is None:
            raise RuntimeError("DFlash verify must capture FULL target hidden states.")
        aux_dim = h_all.shape[-1]
        target_h_steps = h_all.reshape(bs, gamma + 1, aux_dim)
        self._materialize_dflash_hidden_to_native_kv(
            target_hidden=target_h_steps.reshape(bs * (gamma + 1), aux_dim).contiguous(),
            positions=all_positions.reshape(-1).contiguous(),
            cache_locs=cache_locs.reshape(-1).contiguous(),
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

        if self._timing_enabled:
            self._record_decode_timing(
                "dflash",
                ev_t0,
                ev_draft_end,
                ev_verify_end,
                ev_other_end,
                accept_lens,
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

        if self._timing_enabled:
            ev_t0 = torch.cuda.Event(enable_timing=True)
            ev_draft_end = torch.cuda.Event(enable_timing=True)
            ev_verify_end = torch.cuda.Event(enable_timing=True)
            ev_other_end = torch.cuda.Event(enable_timing=True)
            ev_t0.record()

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
        if self._timing_enabled:
            ev_draft_end.record()

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
        if self._timing_enabled:
            ev_verify_end.record()

        # ---- 4b. Hierarchical SMC score pass ----
        # The 8B verify above (score_result) supplies the aux hidden states used
        # by the head rewrite below. When a separate larger SCORE model is set
        # (nested mode), run it over the same gamma+1 block and use ITS logits as
        # the SMC importance-weight target p (and for the bonus). The block was
        # proposed by the EAGLE head (q); SMC reweights p_score / q_head. Base
        # and score models must share a vocabulary.
        if self._has_score_model:
            score_verify_fb, _ = ctx.prepare_for_verify(
                self.req_to_token_pool,
                batch,
                self._score_worker,
                all_tokens,
                cache_locs,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            score_big_result = self._score_worker.forward_batch_generation(
                model_worker_batch=None,
                forward_batch=score_verify_fb,
                is_verify=True,
                skip_attn_backend_init=True,
            )
            score_logits = score_big_result.logits_output.next_token_logits
            if os.environ.get("SMCSD_NESTED_DEBUG"):
                with torch.no_grad():
                    _tl = score_result.logits_output.next_token_logits
                    _linf = (
                        score_logits.float() - _tl.float()
                    ).abs().max().item()
                print(
                    f"[NESTED DBG] bs={bs} score-vs-base verify logits "
                    f"Linf={_linf:.4f}",
                    flush=True,
                )
        else:
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

        if self._timing_enabled:
            self._record_decode_timing(
                "eagle3",
                ev_t0,
                ev_draft_end,
                ev_verify_end,
                ev_other_end,
                accept_lens,
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
