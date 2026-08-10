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

import atexit
import copy
import dataclasses
import json
import logging
import os
import time
from typing import Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from smcsd.common.verify import SMCVerifyInput, assign_smc_cache_locs_kernel
from smcsd.core.info import SMCDecodeContext, SMCDraftInput
from smcsd.cross_tokenizer.bucketing import DEFAULT_PROXY_BUCKETS, bucket_proxy_sequences
from smcsd.cross_tokenizer.build import encode_text
from smcsd.cross_tokenizer.lineage import plan_cross_draft_lineage
from smcsd.cross_tokenizer.spans import decode_exact, shared_text_spans
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

logger = logging.getLogger(__name__)


class _NoPrefixDraftCache:
    """Minimal allocation facade for draft-only private KV pools.

    Cross-tokenizer draft state is not visible to the scheduler radix cache.
    `ScheduleBatch.prepare_for_extend` only needs the small cache surface used by
    `alloc_for_extend`, so this shim allocates directly from the draft worker's
    KV allocator and never inserts prefix-cache entries.
    """

    def __init__(self, req_to_token_pool, token_to_kv_pool_allocator):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = token_to_kv_pool_allocator.page_size

    def is_chunk_cache(self):
        return True

    def supports_swa(self):
        return False

    def supports_mamba(self):
        return False

    def evict(self, *_args, **_kwargs):
        return None

    def pretty_print(self):
        return None

    def available_and_evictable_str(self):
        """Surface a real OOM message on the draft pool.

        SGLang's ``alloc_token_slots`` formats this string when allocation fails.
        Without it the formatter itself raises ``AttributeError`` and masks the
        underlying draft-KV exhaustion (e.g. when a non-terminating or
        token-inflated cross-tokenizer run fills the private draft pool).  There
        is no prefix cache here, so nothing is evictable.
        """
        try:
            avail = self.token_to_kv_pool_allocator.available_size()
        except Exception:
            avail = "?"
        return (
            f"Draft KV pool: #available={avail} tokens, #evictable=0 "
            "(cross-tokenizer draft pool has no prefix cache). If this is an "
            "OOM, raise the draft KV budget or lower N/gamma/concurrency."
        )


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
        self.device = server_args.device
        self._target_worker = target_worker  # score model

        # SMC trace emission (SMC_TRACE_JSONL) is a rank-0-only diagnostic.
        # Both the req_state population and the worker dump are pure host-side
        # ops (.cpu() syncs + JSONL appends) with no collectives; running them
        # on every TP rank double-writes the trace file AND stalls the TP group
        # (non-zero ranks NCCL-spin while rank-0 does extra host syncs). Clear
        # the env on non-zero ranks so both trace paths in this process no-op.
        if tp_rank != 0:
            os.environ.pop("SMC_TRACE_JSONL", None)

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
        self.smc_cross_tokenizer = bool(
            getattr(server_args, "smc_cross_tokenizer", False)
        )
        self.cross_tokenizer_mapper = None
        # Debug-only: dump draft KV positions / cache-loc mapping for the first
        # few decode calls to confirm the prefill→step-0 position convention
        # before the deferred-bonus rework.  No behavior change when unset.
        self._smc_dbg_positions = bool(
            int(os.environ.get("SMC_DEBUG_POSITIONS", "0"))
        )
        self._smc_dbg_cross = bool(int(os.environ.get("SMC_DEBUG_CROSS", "0")))
        self._smc_dbg_cross_calls = 0
        # One-shot draft CUDA-graph diagnostics for both the conventional and
        # cross-tokenizer draft AR loops (SMC_DBG_DRAFT_GRAPH=1).
        self._smc_dbg_draft_graph = bool(
            int(os.environ.get("SMC_DBG_DRAFT_GRAPH", "0"))
        )
        self._smc_dbg_draft_graph_done = False
        self._smc_cross_stats_interval = int(
            os.environ.get("SMC_CROSS_STATS_INTERVAL", "0")
        )
        self._smc_cross_next_stats_at = max(self._smc_cross_stats_interval, 0)
        self._smc_cross_profile = bool(
            int(os.environ.get("SMC_CROSS_PROFILE", "0"))
        )
        self._smc_cross_profile_interval = max(
            1, int(os.environ.get("SMC_CROSS_PROFILE_INTERVAL", "50"))
        )
        self._smc_cross_profile_calls = 0
        self._smc_cross_profile_acc = {
            "draft_ms": 0.0,
            "map_ms": 0.0,
            "bucket_ms": 0.0,
            "target_ms": 0.0,
            "post_ms": 0.0,
        }
        # Memo for single-token (bonus) retokenization.  The per-step bonus
        # retok decodes one target token to text and re-encodes it to draft
        # ids; with bs particles that is ~2*bs HF tokenizer calls/step and was
        # the dominant cross-family decode overhead (post_ms).  The result is a
        # pure function of the target token id, so memoizing it is exact
        # (byte-identical) while collapsing the steady-state cost to a dict
        # lookup.  Disable with SMC_CROSS_BONUS_RETOK_CACHE=0.
        self._bonus_retok_cache: dict[int, list[int]] = {}
        self._bonus_retok_cache_enabled = bool(
            int(os.environ.get("SMC_CROSS_BONUS_RETOK_CACHE", "1"))
        )
        # Single-extend bonus-prefix commit: when a commit needs >1 suffix
        # position, write all of them in ONE padded linear (EXTEND-style)
        # draft forward instead of `max_suffix` separate decode forwards.
        # Accuracy-neutral (validated: same KV up to bf16 kernel noise, same
        # GSM8K distribution).  NOTE: default OFF and NOT recommended in the
        # common config — the per-position loop's decode forwards are CUDA-
        # graph-replayed (cheap), while this extend is eager (heavier setup),
        # so it is a net TPS *loss* there.  Clean profiling shows the commit is
        # only ~1.4 ms/step (fwd_positions/step≈1.1, almost always a single
        # position); the real costs are draft_ms (~10 ms) and target_ms
        # (~6 ms).  Retained (gated) for configs where the draft decode is NOT
        # graphed, where collapsing N eager decodes into one extend can help.
        self._smc_commit_extend = bool(
            int(os.environ.get("SMC_CROSS_COMMIT_EXTEND", "0"))
        )
        # Dual-write KV equivalence check for the extend commit: run the extend,
        # snapshot the suffix-slot KV, then run the legacy loop (overwriting the
        # same slots) and assert byte-identical KV.  The run keeps the legacy
        # (known-correct) values.  Diagnostic only (2x commit cost).
        self._smc_commit_verify = bool(
            int(os.environ.get("SMC_CROSS_COMMIT_VERIFY", "0"))
        )
        self._smc_commit_verify_calls = 0
        # G8: defer cross-tokenizer suffix KV writes to the next draft step.
        # Pending verify dual-runs the new writer and the old loop oracle, then
        # keeps the oracle values.  Pending suffix itself is off by default
        # until the verifier passes on the target benchmark matrix.
        self._smc_pending_suffix = bool(
            int(os.environ.get("SMC_CROSS_PENDING_SUFFIX", "0"))
        )
        self._smc_pending_verify = bool(
            int(os.environ.get("SMC_CROSS_PENDING_VERIFY", "0"))
        )
        self._smc_pending_eager = bool(
            int(os.environ.get("SMC_CROSS_PENDING_EAGER", "1"))
        )
        self._smc_pending_verify_calls = 0
        # Sub-breakdown of cross post_ms (non-identity branch): D2H sync, bonus
        # retok, lineage plan, suffix-commit draft forwards, history H2D.
        self._smc_cross_post_acc = {
            "d2h": 0.0,
            "retok": 0.0,
            "lineage": 0.0,
            "commit": 0.0,
            "h2d": 0.0,
        }
        self._commit_stats = {"calls": 0, "fwd_positions": 0, "row_steps": 0}
        self._pending_stats = {
            "calls": 0,
            "buckets": 0,
            "rows": 0,
            "tokens": 0,
        }
        self._proposal_stats = {
            "tokens": 0,
            "logdiff_sum": 0.0,
            "logdiff_sumsq": 0.0,
            "logdiff_min": float("inf"),
            "logdiff_max": float("-inf"),
            "target_logp_sum": 0.0,
            "draft_logq_sum": 0.0,
            "proxy_len_sum": 0.0,
            "proxy_len_max": 0.0,
        }
        self._smc_trace_jsonl = os.environ.get("SMC_TRACE_JSONL")
        self._smc_trace_limit = int(os.environ.get("SMC_TRACE_LIMIT", "0"))
        self._smc_trace_topk = max(0, int(os.environ.get("SMC_TRACE_TOPK", "0")))
        self._smc_trace_rows = 0
        self._smc_trace_decode_call = 0
        # Snapshot of cumulative heal/cache counters at the last profile print, so
        # we can report *interval* (steady-state) heal re-encode rate and cache
        # hit rate rather than a cold-start-muddied cumulative average.
        self._smc_cross_heal_last = {
            "groups": 0,
            "reencodes": 0,
            "hits": 0,
            "misses": 0,
        }
        # Mirror profiler for the conventional same-tokenizer decode path so its
        # draft/target/post phase times are directly comparable to the cross
        # profile above (SMC_SAME_PROFILE=1).
        self._smc_same_profile = bool(
            int(os.environ.get("SMC_SAME_PROFILE", "0"))
        )
        self._smc_same_profile_calls = 0
        self._smc_same_profile_acc = {
            "draft_ms": 0.0,
            "target_ms": 0.0,
            "post_ms": 0.0,
        }
        self._smc_dbg_calls = 0
        # Deferred-bonus draft schedule: drop the per-step over-draft and fold
        # the deferred d_{gamma-1} write into the next step's leading 2-token
        # head. ON BY DEFAULT; configured via server_args (SMCEngine's
        # defer_bonus=False to opt out). Unsupported draft backends downgrade
        # to the legacy gamma+1 schedule with a warning.
        # Resolution order: SMC_DEFER_BONUS env > SMCEngine kwarg > default ON.
        _defer_env = os.environ.get("SMC_DEFER_BONUS")
        self.smc_defer_bonus = (
            bool(int(_defer_env))
            if _defer_env is not None
            else bool(getattr(server_args, "smc_defer_bonus", True))
        )
        # Diagnostic: force the CPU mapping/lineage path even when the draft and
        # target vocabularies are identical, so the GPU identity passthrough can
        # be equivalence-checked against it at a fixed seed.
        self._smc_force_cpu_lineage = bool(
            int(os.environ.get("SMC_CROSS_FORCE_CPU_LINEAGE", "0"))
        )
        # Identity-vocab fast write-back: skip the per-step accept_lens /
        # draft_visible_lens device->host copies, whose host-mirror corrections
        # are constant under identity (target: -gamma_plus_1 + gamma_plus_1 = 0;
        # draft: -draft_gamma_plus_1 + gamma_plus_1 = -headroom).  Default on;
        # set SMC_CROSS_FAST_WRITEBACK=0 to force the syncing path for A/B.
        self._smc_fast_writeback = bool(
            int(os.environ.get("SMC_CROSS_FAST_WRITEBACK", "1"))
        )
        # G1: extend the sync-free write-back to the NON-identity cross path.
        # proxy_lens / draft_visible_lens are built from host lists, so the
        # write-back's host-mirror update needs no device->host copy.  Lossless
        # (identical int lengths) -> default ON; removes 2 D2H syncs/step (small
        # clean-tok/s win, exp10: 258.7->261.9, masked under SMC_SCHED_PROFILE).
        # Set SMC_CROSS_HOST_LENS=0 to A/B back to the syncing path.
        self._smc_cross_host_lens = bool(
            int(os.environ.get("SMC_CROSS_HOST_LENS", "1"))
        )
        # Correctness-first streaming mapper.  This stays opt-in until the
        # induced-proposal and finite-particle gates pass, but unlike the
        # mapper-only diagnostic it supports real no-bonus carry rounds.
        self._smc_boundary_carry = bool(
            int(os.environ.get("SMC_CROSS_BOUNDARY_CARRY", "0"))
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
        # embed/lm_head with the target.  Same-tokenizer SMC keeps sharing
        # the target request/KV pools.  Cross-tokenizer SMC needs independent
        # draft lineage because target and draft sequence lengths diverge.
        draft_req_pool = None if self.smc_cross_tokenizer else self.req_to_token_pool
        draft_kv_allocator = (
            None if self.smc_cross_tokenizer else self.token_to_kv_pool_allocator
        )
        # SGLang requires draft workers to receive a resolved memory pool
        # config, even when the actual draft pools are private. Passing None
        # for the pools below still makes ModelRunner allocate independent
        # draft req/KV pools; this config only carries sizing constraints.
        draft_memory_pool_config = target_worker.model_runner.memory_pool_config
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
            req_to_token_pool=draft_req_pool,
            token_to_kv_pool_allocator=draft_kv_allocator,
            memory_pool_config=draft_memory_pool_config,
        )
        self.draft_runner = self._draft_worker.model_runner
        self.draft_req_to_token_pool, self.draft_token_to_kv_pool_allocator = (
            self._draft_worker.get_memory_pool()
        )
        if self.smc_cross_tokenizer:
            self._maybe_refcount_cross_draft_allocator()
        self._draft_tree_cache = _NoPrefixDraftCache(
            self.draft_req_to_token_pool,
            self.draft_token_to_kv_pool_allocator,
        )

        if self.smc_cross_tokenizer:
            self.cross_tokenizer_mapper = self._init_cross_tokenizer_mapper()
            if self.tp_rank == 0:
                atexit.register(self._print_final_cross_mapper_stats)

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
        self._draft_is_hybrid = draft_is_hybrid
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
            elif self.draft_runner.hybrid_gdn_config is not None:
                # Hybrid (Mamba/GDN) draft: the 2-token head uses per-batch
                # linear-verify metadata (head_spec.draft_token_num=2 +
                # populate_linear_verify_metadata in prepare_for_draft_head).
                # The draft pool carries a depth-2 intermediate recurrent buffer
                # (see _maybe_isolate_dense_hybrid_draft_state), and the head's
                # S-position state is committed after the forward via
                # _commit_draft_mamba_state_after_head.
                #
                # The head is a 2-token verify on the FULL-ATTENTION
                # sub-backend too: pin its verify-block-size global to 2,
                # exactly like the non-hybrid branches above pin the primary.
                # The captured verify path reads this global, not the batch's
                # spec fields.  At gamma==1 the inherited default (gamma+1 == 2)
                # masked this; at gamma>1 the stale global made the captured
                # head verify kernels stride kv indices by gamma+1 per request
                # over a 2-token window -> illegal memory access at capture.
                full_ab = getattr(draft_ab, "full_attn_backend", None)
                if isinstance(full_ab, TritonAttnBackend):
                    full_ab.num_draft_tokens = 2
                elif isinstance(full_ab, FlashAttentionBackend):
                    full_ab.speculative_num_draft_tokens = 2
                # CUDA-graph verify cache fix: pin the shared linear backend's
                # verify query-start-loc cache to step-2 BEFORE the standalone
                # head runner captures (see _pin_draft_head_verify_qsl).
                self._pin_draft_head_verify_qsl()
            else:
                logger.warning(
                    "Deferred bonus supports triton, fa3, and hybrid-GDN draft "
                    "attention backends only; got %s (MLA drafts are not "
                    "supported by the deferred-bonus head).  Falling back to "
                    "the legacy gamma+1 draft schedule.",
                    type(draft_ab).__name__,
                )
                self.smc_defer_bonus = False

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

        self.draft_phase_graph_runner = None
        self.cycle_graph_runner = None
        _cycle_env = os.environ.get("SMC_CYCLE_GRAPH")
        want_cycle = (
            bool(int(_cycle_env))
            if _cycle_env is not None
            else bool(getattr(server_args, "smc_cycle_graph", True))
        )
        want_phase = bool(int(os.environ.get("SMC_DRAFT_PHASE_GRAPH", "0")))
        # Cycle and draft-phase graphs assume identical target/draft sequence
        # layouts. Cross-tokenizer decoding has independent KV lineages and
        # uses its validated eager multi-step path.
        if self.smc_cross_tokenizer:
            want_cycle = False
            want_phase = False
        if want_cycle or want_phase:
            from sglang.srt.layers.attention.triton_backend import (
                TritonAttnBackend,
                TritonMultiStepDraftBackend,
            )

            from smcsd.core.hybrid_multistep_backend import (
                HybridLinearAttnMultiStepBackend,
            )

            reasons = []
            if backup_disable_cuda_graph:
                reasons.append("cuda graph disabled")
            if not self.smc_draft_temperature > 0:
                reasons.append("greedy draft (temperature 0)")
            if not isinstance(
                self.draft_attn_backend,
                (TritonMultiStepDraftBackend, HybridLinearAttnMultiStepBackend),
            ):
                reasons.append(
                    f"unsupported multi-step backend "
                    f"{type(self.draft_attn_backend).__name__} "
                    "(triton or hybrid-GDN multi-step)"
                )

            target_ab = self.score_runner.attn_backend
            target_ok = isinstance(target_ab, TritonAttnBackend) or (
                self.score_runner.hybrid_gdn_config is not None
                and hasattr(target_ab, "update_mamba_state_after_mtp_verify")
            )
            if want_cycle and not target_ok:
                reasons.append(
                    f"unsupported target backend "
                    f"{type(target_ab).__name__} "
                    "(triton, or hybrid-GDN with in-graph mamba commit)"
                )
            if reasons:
                logger.warning(
                    "cycle/draft-phase graph disabled (falling back to the "
                    "per-step path): %s",
                    "; ".join(reasons),
                )
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
                # The cycle runner's multi-step backend init re-binds the
                # shared linear backend's verify query-start-loc cache back to
                # step-1 (decode layout).  Re-pin step-2 so the STANDALONE head
                # runner (used on the eager/cycle-fallback path) replays with
                # correct 2-token windows.  The cycle runner itself is immune —
                # its head uses a dedicated linear backend.
                if self.smc_defer_bonus:
                    self._pin_draft_head_verify_qsl()
            else:
                from smcsd.model_executor.smc_draft_phase_graph_runner import (
                    SMCDraftPhaseGraphRunner,
                )

                self.draft_phase_graph_runner = SMCDraftPhaseGraphRunner(self)

    def _pin_draft_head_verify_qsl(self) -> None:
        """(Deferred-bonus, hybrid draft) Pin the shared linear backend's
        cuda-graph VERIFY query-start-loc cache to a step-2 arange.

        The draft's linear (GDN) backend gets its cuda-graph state from the
        DECODE graph init (max_num_tokens == max_bs -> draft_token_num == 1),
        so its verify cache is a step-1 arange.  The deferred 2-token head is a
        verify forward: with step-1 windows the GDN kernels process only the
        first bs of the head's 2*bs tokens and leave the remaining outputs
        unwritten (stale graph-pool memory -> non-deterministic quality
        collapse).  Called before the standalone head runner captures, and
        again after the cycle runner's multi-step init (which re-binds the
        cache back to step-1).  The decode cache is a separate buffer and is
        untouched.  No-op for non-hybrid drafts or when cuda graphs are off.
        """
        if getattr(self.draft_runner, "hybrid_gdn_config", None) is None:
            return
        lin = getattr(self.draft_runner.attn_backend, "linear_attn_backend", None)
        cached = getattr(lin, "cached_cuda_graph_verify_query_start_loc", None)
        if cached is None:
            return
        # Cover the largest bs any runner may replay with (list length grows
        # across repeated init_cuda_graph_state calls; over-length is harmless,
        # under-length would slice out of range on fallback replays).
        n = max(cached.numel() - 1, len(lin.state_indices_list))
        lin.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0, 2 * n + 1, step=2, dtype=cached.dtype, device=cached.device
        )

    def _print_final_cross_mapper_stats(self) -> None:
        mapper = self.cross_tokenizer_mapper
        if mapper is not None and mapper.stats.calls:
            print(
                "[SMC_CROSS_FINAL] " + mapper.format_stats(),
                flush=True,
            )

    def _maybe_refcount_cross_draft_allocator(self) -> None:
        """Use SMC refcounts for private cross-tokenizer draft KV.

        Cross-tokenizer drafting keeps a separate draft req/KV pool because
        target and draft tokenizations diverge. The allocator still needs SMC
        refcount semantics so prefix fanout and resampling can share draft KV
        slots by block-table reference instead of cloning full KV tensors
        through CPU.
        """
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

        from smcsd.mem_cache.allocator import SMCRefCountedTokenAllocator

        allocator = self.draft_token_to_kv_pool_allocator
        if isinstance(allocator, SMCRefCountedTokenAllocator):
            return
        if type(allocator) is not TokenToKVPoolAllocator:
            return
        if allocator.page_size != 1:
            return

        refcounted = SMCRefCountedTokenAllocator(
            allocator.size,
            dtype=allocator.dtype,
            device=allocator.device,
            kvcache=allocator.get_kvcache(),
            need_sort=allocator.need_sort,
        )
        self.draft_token_to_kv_pool_allocator = refcounted
        self.draft_runner.token_to_kv_pool_allocator = refcounted
        self._draft_worker.token_to_kv_pool_allocator = refcounted

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
            # Deferred-bonus runs a 2-token verify-style head on the draft, so
            # it needs a depth-2 intermediate recurrent-state buffer (like the
            # target's verify buffer); plain full-cycle drafts need none.
            speculative_num_draft_tokens=(2 if self.smc_defer_bonus else None),
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

    def _commit_draft_mamba_state_after_head(
        self, head_forward_batch: ForwardBatch, bs: int
    ) -> None:
        """Commit the DRAFT's recurrent state after the deferred-bonus 2-token
        head.

        The head ``[prev @ S-1, verified @ S]`` runs on the draft as a
        verify-style forward with deferred state updates, so the live Mamba
        state is still at S-2 afterwards.  We scatter the index-1 (verified /
        S) intermediate state back into the live cache so the subsequent single
        decodes continue from S.  accepted_steps==1 selects the second (last)
        of the head's two positions, mirroring the target's accepted_steps==gamma
        for its gamma+1-token verify.
        """
        attn_backend = self.draft_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return
        if head_forward_batch.forward_mode.is_idle():
            return

        accepted_steps = torch.ones(bs, dtype=torch.int64, device=self.device)
        attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=getattr(
                head_forward_batch, "mamba_track_indices", None
            ),
            mamba_steps_to_track=None,
            model=self.draft_runner.model,
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

    def materialize_smc_parent_draft_prefix(self, req, particle_reqs=None) -> None:
        """Fan out the draft-side prefilled prefix for cross-tokenizer SMC."""
        if not self.smc_cross_tokenizer:
            return
        if particle_reqs is None:
            raise RuntimeError("cross-tokenizer draft fanout requires particle reqs.")
        parent_draft_req = getattr(req, "smc_draft_req", None)
        if parent_draft_req is None:
            raise RuntimeError("cross-tokenizer parent is missing draft prefill state.")

        draft_particle_reqs = []
        for particle_req in particle_reqs:
            draft_req = self._make_cross_draft_req(
                particle_req,
                list(parent_draft_req.origin_input_ids),
                suffix="_cross_draft_particle",
                output_ids=list(getattr(parent_draft_req, "output_ids", [])),
            )
            draft_particle_reqs.append(draft_req)

        if self.draft_req_to_token_pool.alloc(draft_particle_reqs) is None:
            raise RuntimeError("cross-tokenizer draft req_to_token_pool full.")

        from smcsd.mem_cache.allocator import copy_block_table

        shared_len = int(parent_draft_req.kv_committed_len)
        for particle_req, draft_req in zip(particle_reqs, draft_particle_reqs):
            copy_block_table(
                self.draft_req_to_token_pool,
                parent_draft_req.req_pool_idx,
                draft_req.req_pool_idx,
                shared_len,
                self.draft_token_to_kv_pool_allocator,
            )
            draft_req.kv_committed_len = shared_len
            draft_req.kv_allocated_len = shared_len
            draft_req.prefix_indices = self.draft_req_to_token_pool.req_to_token[
                draft_req.req_pool_idx, :shared_len
            ].to(dtype=torch.int64, copy=True)
            draft_req.cache_protected_len = shared_len

            particle_req.smc_draft_req = draft_req
            particle_req.smc_draft_req_pool_idx = int(draft_req.req_pool_idx)
            particle_req.smc_draft_origin_input_ids = list(draft_req.origin_input_ids)
            particle_req.smc_draft_kv_committed_len = shared_len
            particle_req.smc_draft_kv_allocated_len = shared_len
            particle_req.smc_draft_verified_id = int(req.smc_draft_verified_id)
            particle_req.smc_draft_output_ids = []

        from smcsd.common.utils import _release_internal_req

        _release_internal_req(
            parent_draft_req,
            req_to_token_pool=self.draft_req_to_token_pool,
            token_to_kv_pool_allocator=self.draft_token_to_kv_pool_allocator,
        )
        req.smc_draft_req = None

    def _init_cross_tokenizer_mapper(self):
        from transformers import AutoTokenizer

        from smcsd.cross_tokenizer.artifacts import load_artifact
        from smcsd.cross_tokenizer.mapper import TokenMapper

        target_tokenizer_path = (
            self.server_args.tokenizer_path or self.server_args.model_path
        )
        draft_tokenizer_path = getattr(
            self.server_args,
            "smc_draft_tokenizer_path",
            self.server_args.speculative_draft_model_path,
        )
        target_tokenizer = AutoTokenizer.from_pretrained(
            target_tokenizer_path,
            trust_remote_code=self.server_args.trust_remote_code,
        )
        draft_tokenizer = AutoTokenizer.from_pretrained(
            draft_tokenizer_path,
            trust_remote_code=self.server_args.trust_remote_code,
        )
        artifact_path = getattr(
            self.server_args, "smc_cross_tokenizer_artifact_path", None
        )
        artifact = load_artifact(artifact_path) if artifact_path else None
        mode = getattr(self.server_args, "smc_cross_tokenizer_mode", "hybrid")
        if mode == "hybrid" and artifact is None:
            raise ValueError(
                "hybrid cross-tokenizer mode requires a mapping artifact"
            )
        return TokenMapper(
            draft_tokenizer=draft_tokenizer,
            target_tokenizer=target_tokenizer,
            artifact=artifact,
            mode=mode,
        )

    def _make_cross_draft_req(
        self,
        source_req: Req,
        draft_input_ids: list[int],
        *,
        suffix: str,
        output_ids: Optional[list[int]] = None,
    ) -> Req:
        sampling_params = copy.copy(source_req.sampling_params)
        if isinstance(sampling_params.custom_params, dict):
            sampling_params.custom_params = dict(sampling_params.custom_params)

        draft_req = Req(
            rid=f"{source_req.rid}{suffix}",
            origin_input_text=source_req.origin_input_text,
            origin_input_ids=list(draft_input_ids),
            sampling_params=sampling_params,
            return_logprob=False,
            top_logprobs_num=0,
            dllm_config=None,
            token_ids_logprob=None,
            stream=False,
            origin_input_ids_unpadded=tuple(draft_input_ids),
            lora_id=source_req.lora_id,
            input_embeds=None,
            token_type_ids=None,
            session=None,
            custom_logit_processor=None,
            require_reasoning=False,
            return_hidden_states=False,
            return_routed_experts=False,
            eos_token_ids=None,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            disagg_mode=None,
            routed_dp_rank=None,
            disagg_prefill_dp_rank=None,
            vocab_size=self.draft_runner.model_config.vocab_size,
            priority=source_req.priority,
            metrics_collector=None,
            extra_key=source_req.extra_key,
            routing_key=source_req.routing_key,
            dimensions=getattr(source_req, "dimensions", None),
            http_worker_ipc=None,
            time_stats=None,
        )
        draft_req.output_ids = list(output_ids or [])
        draft_req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        draft_req.last_node = None
        draft_req.last_host_node = None
        draft_req.last_host_backup_node = None
        draft_req.host_hit_length = 0
        draft_req.cache_protected_len = 0
        draft_req.init_next_round_input(tree_cache=None)
        return draft_req

    def _draft_ids_for_visible_target(self, target_ids: list[int]) -> list[int]:
        text = self.cross_tokenizer_mapper.target_tokenizer.decode(
            target_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        try:
            return list(
                self.cross_tokenizer_mapper.draft_tokenizer.encode(
                    text, add_special_tokens=False
                )
            )
        except TypeError:
            return list(self.cross_tokenizer_mapper.draft_tokenizer.encode(text))

    def _bonus_draft_ids(self, bonus_cpu: list[int]) -> list[list[int]]:
        """Retokenize per-row bonus tokens (one target token each) into draft
        ids, memoized by token id.

        The bonus retok is a pure function of the single target token id —
        ``decode([id])`` then ``encode(text)`` — so caching it is exact.  This
        is the dominant cross-family ``post_ms`` cost: without the cache it is
        ~2*bs HF tokenizer calls every decode step.
        """
        if not self._bonus_retok_cache_enabled:
            return [
                self._draft_ids_for_visible_target([int(token_id)])
                for token_id in bonus_cpu
            ]
        cache = self._bonus_retok_cache
        out: list[list[int]] = []
        for token_id in bonus_cpu:
            tid = int(token_id)
            row = cache.get(tid)
            if row is None:
                row = self._draft_ids_for_visible_target([tid])
                cache[tid] = row
            out.append(row)
        return out

    # ── Main entry point ──

    def forward_batch_generation(self, batch):
        if isinstance(batch, ScheduleBatch):
            batch = batch.get_model_worker_batch()

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_extend(batch)
        else:
            return self._forward_decode(batch)

    # ── EXTEND (prefill) ──

    def _target_prefill_with_power_seed(self, batch: ModelWorkerBatch):
        """Prefill target KV and draw x0 from the configured power target.

        SMC shares x0 across a newly materialized particle group, so it has no
        per-particle proposal correction. It must therefore be target-distributed
        and independent of the draft. The target extend path does not expose a
        reliable final-position logit row, so project its captured final hidden
        state through the target's own tensor-parallel-aware LogitsProcessor.
        """
        score_batch = dataclasses.replace(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        score_result = self._target_worker.forward_batch_generation(score_batch)
        hidden = score_result.logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "FULL target hidden-state capture did not populate x0 states."
            )

        total_prompt_tokens = int(batch.seq_lens.sum().item())
        if hidden.shape[0] != total_prompt_tokens:
            raise RuntimeError(
                f"x0 hidden capture covers {hidden.shape[0]} tokens but the "
                f"batch has {total_prompt_tokens}; SMC requires full-prompt "
                "prefill (disable radix/prefix caching and chunked prefill)."
            )
        last_idx = torch.cumsum(batch.seq_lens.to(torch.int64), dim=0) - 1
        last_hidden = hidden[last_idx]
        score_result.logits_output.hidden_states = None

        logits_metadata = LogitsMetadata(
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        x0_logits = self.score_runner.model.logits_processor(
            None,
            last_hidden,
            self.score_runner.model.lm_head,
            logits_metadata,
        ).next_token_logits
        x0, _ = self._sample_target_power(x0_logits)
        score_result.next_token_ids = x0
        return score_result, x0

    def _forward_extend_cross_tokenizer(self, batch: ModelWorkerBatch):
        bs = len(batch.seq_lens)

        score_result, x0 = self._target_prefill_with_power_seed(batch)
        target_next = x0.detach().to("cpu").tolist()

        draft_reqs = []
        draft_verified_ids = []
        for req, next_id in zip(batch.reqs, target_next):
            visible_target_ids = list(req.origin_input_ids) + list(req.output_ids) + [
                int(next_id)
            ]
            draft_ids = self._draft_ids_for_visible_target(visible_target_ids)
            if not draft_ids:
                raise RuntimeError(
                    "cross-tokenizer retokenization produced an empty draft prefix."
                )
            # Match the SMC decode convention: seq_lens / KV contain the
            # committed prefix, while verified_id is the final visible token
            # whose KV will be written by the first decode step.
            committed_draft_ids = draft_ids[:-1]
            draft_verified_ids.append(int(draft_ids[-1]))
            draft_req = self._make_cross_draft_req(
                req,
                committed_draft_ids,
                suffix="_smc_cross_draft_parent",
            )
            req.smc_draft_req = draft_req
            req.smc_draft_origin_input_ids = list(committed_draft_ids)
            draft_reqs.append(draft_req)

        draft_batch = ScheduleBatch.init_new(
            draft_reqs,
            self.draft_req_to_token_pool,
            self.draft_token_to_kv_pool_allocator,
            self._draft_tree_cache,
            self.draft_runner.model_config,
            enable_overlap=False,
            spec_algorithm=None,
        )
        draft_batch.prepare_for_extend()
        draft_mwb = draft_batch.get_model_worker_batch()
        self._draft_worker.forward_batch_generation(draft_mwb)
        draft_verified = torch.tensor(
            draft_verified_ids, dtype=torch.int64, device=self.device
        )

        for req, draft_req, draft_next in zip(batch.reqs, draft_reqs, draft_verified_ids):
            req.smc_draft_req_pool_idx = int(draft_req.req_pool_idx)
            req.smc_draft_kv_committed_len = int(draft_req.kv_committed_len)
            req.smc_draft_kv_allocated_len = int(draft_req.kv_allocated_len)
            req.smc_draft_verified_id = int(draft_next)
            req.smc_draft_output_ids = []

        score_result.next_draft_input = SMCDraftInput(
            verified_id=draft_verified,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )
        score_result.accept_lens = torch.zeros(
            bs, dtype=torch.int32, device=self.device
        )
        return score_result

    def _forward_extend(self, batch: ModelWorkerBatch):
        if self.smc_cross_tokenizer:
            return self._forward_extend_cross_tokenizer(batch)

        bs = len(batch.seq_lens)

        score_result, x0 = self._target_prefill_with_power_seed(batch)

        # Populate draft prompt KV; its sampled token is deliberately discarded.
        draft_batch = self._make_clean_batch(batch)
        self._draft_worker.forward_batch_generation(draft_batch)

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

    def _run_cross_tokenizer_draft_ar(
        self,
        batch: ModelWorkerBatch,
        draft_input: SMCDraftInput,
    ):
        draft_ctx = draft_input.draft_decode_ctx
        if draft_ctx is None or draft_input.draft_verified_id is None:
            raise RuntimeError("cross-tokenizer decode is missing draft context.")

        draft_batch = dataclasses.replace(
            batch,
            input_ids=draft_input.draft_verified_id,
            req_pool_indices=batch.smc_draft_req_pool_indices,
            seq_lens=draft_ctx.new_seq_lens,
            seq_lens_cpu=draft_ctx.orig_seq_lens_cpu + self.speculative_num_draft_tokens,
            seq_lens_sum=int(
                (
                    draft_ctx.orig_seq_lens_cpu
                    + self.speculative_num_draft_tokens
                ).sum().item()
            ),
            spec_info=draft_input,
        )
        draft_fb, can_cuda_graph, cache_locs, all_positions, all_seq_lens = (
            draft_ctx.prepare_for_draft(
                draft_input.draft_verified_id,
                self.draft_req_to_token_pool,
                draft_batch,
                self.draft_runner.graph_runner
                if hasattr(self.draft_runner, "graph_runner")
                else None,
                self.draft_runner,
            )
        )

        if self._smc_pending_verify:
            self._verify_pending_cross_draft_suffix(
                batch,
                draft_ctx,
                draft_input.pending_draft_suffix_ids,
                draft_input.pending_draft_suffix_lens,
            )
        elif self._smc_pending_suffix:
            self._run_pending_cross_draft_suffix_heads(
                batch,
                draft_ctx,
                draft_input.pending_draft_suffix_ids,
                draft_input.pending_draft_suffix_lens,
            )

        bs = len(draft_ctx.orig_seq_lens)
        gamma = self.gamma
        use_multistep = self.draft_attn_backend is not None and not can_cuda_graph
        if use_multistep and not draft_fb.forward_mode.is_idle():
            draft_fb.spec_info = draft_input
            draft_fb.seq_lens = draft_ctx.orig_seq_lens
            draft_fb.seq_lens_cpu = draft_ctx.orig_seq_lens_cpu
            self.draft_attn_backend.init_forward_metadata(draft_fb)

        if self.smc_defer_bonus and not draft_fb.forward_mode.is_idle():
            deferred_input = dataclasses.replace(
                draft_input, verified_id=draft_input.draft_verified_id
            )
            all_tokens, draft_logprobs_stacked = self._draft_ar_deferred(
                draft_ctx,
                deferred_input,
                draft_fb,
                cache_locs,
                all_positions,
                all_seq_lens,
                draft_batch,
                bs,
                gamma,
                req_to_token_pool=self.draft_req_to_token_pool,
            )
            return all_tokens, draft_logprobs_stacked, True, cache_locs

        current_ids = draft_input.draft_verified_id
        all_tokens = [current_ids]
        draft_logprobs = []
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
                draft_fb.seq_lens_sum = draft_ctx.orig_seq_lens_sum + bs * (step + 1)
                draft_fb.seq_lens_cpu = draft_ctx.orig_seq_lens_cpu + (step + 1)
                draft_out = self.draft_runner.forward(draft_fb)

            if (
                self._smc_dbg_draft_graph
                and not self._smc_dbg_draft_graph_done
                and step == 0
            ):
                self._smc_dbg_draft_graph_done = True
                print(
                    "[SMC_DBG_DRAFT_GRAPH] path=cross "
                    f"bs={bs} gamma={gamma} can_cuda_graph={can_cuda_graph} "
                    f"use_multistep={use_multistep} "
                    f"forward_can_run_graph={draft_out.can_run_graph}",
                    flush=True,
                )

            next_id, token_logprob = self._sample_draft_token(
                draft_out.logits_output.next_token_logits
            )
            if step < gamma:
                draft_logprobs.append(token_logprob)
            all_tokens.append(next_id)
            current_ids = next_id

        return all_tokens, torch.stack(draft_logprobs, dim=1), False, cache_locs

    def _commit_cross_draft_suffix(
        self,
        batch: ModelWorkerBatch,
        draft_ctx: SMCDecodeContext,
        cache_locs: torch.Tensor,
        suffix_ids: list[list[int]],
        *,
        position_start: int,
    ) -> None:
        """Write extra draft-token prefix KVs created by target-bonus retokenization."""
        if not suffix_ids:
            return
        max_suffix = max((len(row) for row in suffix_ids), default=0)
        if max_suffix == 0:
            return
        if self._smc_cross_profile:
            n_rows = sum(1 for row in suffix_ids if row)
            self._commit_stats["calls"] += 1
            self._commit_stats["fwd_positions"] += max_suffix
            self._commit_stats["row_steps"] += n_rows
        if position_start + max_suffix > cache_locs.shape[1]:
            raise RuntimeError(
                "cross-tokenizer draft bonus needs "
                f"{position_start + max_suffix} draft KV slots, but only "
                f"{cache_locs.shape[1]} were allocated. Increase "
                "SMC_CROSS_DRAFT_BONUS_HEADROOM."
            )

        # The single-extend path only helps when the legacy loop would launch
        # more than one forward (max_suffix >= 2); for max_suffix == 1 the loop
        # is already a single forward, so keep it (identical cost, simplest).
        if self._smc_commit_extend and max_suffix >= 2:
            if self._smc_commit_verify:
                self._commit_verify_extend_vs_loop(
                    batch, draft_ctx, cache_locs, suffix_ids, position_start, max_suffix
                )
            else:
                self._commit_cross_draft_suffix_extend(
                    batch, draft_ctx, cache_locs, suffix_ids, position_start, max_suffix
                )
            return
        self._commit_cross_draft_suffix_loop(
            batch, draft_ctx, cache_locs, suffix_ids, position_start, max_suffix
        )

    def _commit_cross_draft_suffix_loop(
        self,
        batch: ModelWorkerBatch,
        draft_ctx: SMCDecodeContext,
        cache_locs: torch.Tensor,
        suffix_ids: list[list[int]],
        position_start: int,
        max_suffix: int,
    ) -> None:
        """Default per-position commit: one decode draft forward per suffix
        position over the rows that still have a token at that position.  These
        decode forwards are CUDA-graph-replayed when the shape is captured,
        which is why they are cheap and hard to beat with an eager extend."""
        device = self.device
        for step in range(max_suffix):
            rows = [row for row, ids in enumerate(suffix_ids) if step < len(ids)]
            if not rows:
                continue
            rows_t = torch.tensor(rows, dtype=torch.int64, device=device)
            input_ids = torch.tensor(
                [suffix_ids[row][step] for row in rows],
                dtype=torch.int64,
                device=device,
            )
            positions = draft_ctx.orig_seq_lens[rows_t] + position_start + step
            seq_lens = positions + 1
            seq_lens_cpu = draft_ctx.orig_seq_lens_cpu[rows] + position_start + step + 1
            suffix_batch = dataclasses.replace(
                batch,
                input_ids=input_ids,
                req_pool_indices=batch.smc_draft_req_pool_indices[rows_t],
                out_cache_loc=cache_locs[rows_t, position_start + step].contiguous(),
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=int(seq_lens_cpu.sum().item()),
                spec_info=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            suffix_fb = ForwardBatch.init_new(suffix_batch, self.draft_runner)
            self.draft_runner.forward(suffix_fb)

    def _commit_cross_draft_suffix_extend(
        self,
        batch: ModelWorkerBatch,
        draft_ctx: SMCDecodeContext,
        cache_locs: torch.Tensor,
        suffix_ids: list[list[int]],
        position_start: int,
        max_suffix: int,
    ) -> None:
        """LOSSLESS single-forward equivalent of the per-position commit loop.

        Writes ALL suffix positions for ALL non-empty rows in ONE padded linear
        (EXTEND-style) draft forward of width ``max_suffix`` — the same
        mechanism as the deferred-bonus draft head (``prepare_for_draft_head``).
        Real suffix tokens write their KV to exactly the slots the loop would
        use (``cache_locs[row, position_start + j]``) at the same positions with
        the same causal prefix, so the committed KV is identical; rows shorter
        than ``max_suffix`` are padded and their pad columns write to their own
        allocated (next-step-overwritten) headroom slots, never touching real
        KV.  This removes ``max_suffix - 1`` redundant draft weight-reloads +
        per-forward setups.
        """
        device = self.device
        rows = [r for r, ids in enumerate(suffix_ids) if ids]
        if not rows:
            return
        rows_t = torch.tensor(rows, dtype=torch.int64, device=device)

        # (n_rows, max_suffix) token grid, short rows padded with 0.
        token_grid = [
            suffix_ids[r] + [0] * (max_suffix - len(suffix_ids[r])) for r in rows
        ]
        input_ids = torch.tensor(
            token_grid, dtype=torch.int64, device=device
        ).reshape(-1)

        step_offsets = torch.arange(max_suffix, device=device)
        positions = (
            draft_ctx.orig_seq_lens[rows_t].unsqueeze(1) + position_start + step_offsets
        ).reshape(-1)
        # Real tokens land on the loop's slots; pad columns land on the row's
        # own headroom slots (allocated, overwritten next step, never read).
        col_idx = (position_start + step_offsets).to(torch.int64)
        out_cache_loc = cache_locs[rows_t][:, col_idx].reshape(-1).contiguous()

        prefix_lens = draft_ctx.orig_seq_lens[rows_t] + position_start
        # The CPU copy MUST match the GPU tensor exactly: the linear-verify
        # path builds kv_indptr from the GPU extend_prefix_lens but sizes the
        # kv_indices buffer from extend_prefix_lens_cpu.  draft_ctx's host
        # shadow orig_seq_lens_cpu can drift from orig_seq_lens (GPU) after
        # resampling, so derive the CPU lengths straight from the GPU tensor
        # (one cold-path D2H) to avoid an out-of-bounds attention launch.
        prefix_lens_cpu = prefix_lens.to("cpu")

        spec = SMCVerifyInput(
            draft_token_num=max_suffix,
            positions=positions,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=int(prefix_lens_cpu.sum().item()),
            seq_lens_cpu=prefix_lens_cpu,
            num_tokens_per_req=max_suffix,
        )
        commit_batch = copy.copy(batch)
        commit_batch.input_ids = input_ids
        commit_batch.req_pool_indices = batch.smc_draft_req_pool_indices[rows_t]
        commit_batch.out_cache_loc = out_cache_loc
        commit_batch.seq_lens = prefix_lens
        commit_batch.seq_lens_cpu = prefix_lens_cpu
        commit_batch.seq_lens_sum = spec.seq_lens_sum
        commit_batch.spec_info = spec
        commit_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        commit_batch.forward_mode = ForwardMode.TARGET_VERIFY

        forward_batch = ForwardBatch.init_new(commit_batch, self.draft_runner)
        spec.populate_linear_verify_metadata(forward_batch)
        # Eager: the draft's primary graph runner is decode-only; null it for
        # this multi-token forward and init metadata directly (mirrors the
        # deferred-bonus head's eager fallback).
        self.draft_runner.attn_backend.init_forward_metadata(forward_batch)
        saved_gr = getattr(self.draft_runner, "graph_runner", None)
        self.draft_runner.graph_runner = None
        try:
            self.draft_runner.forward(forward_batch, skip_attn_backend_init=True)
        finally:
            self.draft_runner.graph_runner = saved_gr

    def _read_draft_kv_at(self, slots: torch.Tensor):
        """Snapshot draft K/V buffers at the given cache slots, all layers."""
        pool = self.draft_runner.token_to_kv_pool
        n_layers = self.draft_runner.model_config.num_hidden_layers
        start = getattr(pool, "start_layer", 0)
        keys, vals = [], []
        for layer in range(start, start + n_layers):
            keys.append(pool.get_key_buffer(layer)[slots].detach().clone())
            vals.append(pool.get_value_buffer(layer)[slots].detach().clone())
        return keys, vals

    @staticmethod
    def _combine_pending_suffixes(
        *suffix_groups: list[list[int]],
    ) -> list[list[int]]:
        """Combine mutually exclusive lineage suffix groups into per-row state."""
        bs = max((len(group) for group in suffix_groups), default=0)
        pending = [[] for _ in range(bs)]
        for group in suffix_groups:
            if len(group) != bs:
                raise RuntimeError(
                    "cross-tokenizer pending suffix groups have mismatched batch sizes"
                )
            for row, ids in enumerate(group):
                if not ids:
                    continue
                if pending[row]:
                    raise RuntimeError(
                        "cross-tokenizer lineage produced overlapping suffix groups "
                        f"for row {row}"
                    )
                pending[row] = list(ids)
        return pending

    def _run_pending_cross_draft_suffix_heads(
        self,
        batch: ModelWorkerBatch,
        draft_ctx: SMCDecodeContext,
        pending_ids: torch.Tensor | None,
        pending_lens: torch.Tensor | None,
    ) -> None:
        """Materialize deferred cross-tokenizer draft suffix KVs.

        Each row's suffix occupies positions
        ``draft_ctx.orig_seq_lens[row] - lens[row] .. S-1``.  Rows are bucketed
        by exact suffix length so the linear forward never pads correctness-
        critical KV writes.
        """
        if pending_ids is None or pending_lens is None or pending_lens.numel() == 0:
            return
        lens_cpu = pending_lens.detach().to("cpu").to(torch.int64)
        lengths = [int(x) for x in lens_cpu.tolist()]
        exact_lengths = sorted({length for length in lengths if length > 0})
        if not exact_lengths:
            return

        use_linear = bool(int(os.environ.get("SMC_CROSS_PENDING_LINEAR", "0")))
        if not use_linear:
            # Byte-equivalent default: write only the suffix prefix that the
            # next 2-token draft head cannot cover.  The final suffix token is
            # exactly `prev_last_draft_id` at S-1 and is rewritten by the
            # existing deferred head before any read.  With legacy non-deferred
            # draft AR there is no head rewrite, so write all suffix tokens.
            self._commit_pending_cross_draft_suffix_loop(
                batch,
                draft_ctx,
                pending_ids,
                pending_lens,
                skip_last=self.smc_defer_bonus,
                record_pending_stats=True,
            )
            return

        device = self.device
        for suffix_len in exact_lengths:
            rows = [row for row, length in enumerate(lengths) if length == suffix_len]
            if not rows:
                continue
            rows_t = torch.tensor(rows, dtype=torch.int64, device=device)
            input_ids = pending_ids[rows_t, :suffix_len].to(torch.int64).reshape(-1)
            start_pos = draft_ctx.orig_seq_lens[rows_t] - suffix_len
            if bool((start_pos < 0).any().item()):
                raise RuntimeError("pending draft suffix length exceeds draft prefix")
            step_offsets = torch.arange(suffix_len, dtype=torch.int64, device=device)
            positions_2d = start_pos.unsqueeze(1) + step_offsets.unsqueeze(0)
            positions = positions_2d.reshape(-1)
            req_rows = batch.smc_draft_req_pool_indices[rows_t]
            out_cache_loc = self.draft_req_to_token_pool.req_to_token[
                req_rows.unsqueeze(1), positions_2d
            ].reshape(-1).contiguous()

            prefix_lens = start_pos
            prefix_lens_cpu = prefix_lens.to("cpu")
            spec = SMCVerifyInput(
                draft_token_num=suffix_len,
                positions=positions,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                seq_lens_sum=int(prefix_lens_cpu.sum().item()),
                seq_lens_cpu=prefix_lens_cpu,
                num_tokens_per_req=suffix_len,
            )
            pending_batch = copy.copy(batch)
            pending_batch.input_ids = input_ids
            pending_batch.req_pool_indices = req_rows
            pending_batch.out_cache_loc = out_cache_loc
            pending_batch.seq_lens = prefix_lens
            pending_batch.seq_lens_cpu = prefix_lens_cpu
            pending_batch.seq_lens_sum = spec.seq_lens_sum
            pending_batch.spec_info = spec
            pending_batch.capture_hidden_mode = CaptureHiddenMode.NULL
            pending_batch.forward_mode = ForwardMode.TARGET_VERIFY

            forward_batch = ForwardBatch.init_new(pending_batch, self.draft_runner)
            spec.populate_linear_verify_metadata(forward_batch)
            self.draft_runner.attn_backend.init_forward_metadata(forward_batch)
            saved_gr = getattr(self.draft_runner, "graph_runner", None)
            self.draft_runner.graph_runner = None
            try:
                self.draft_runner.forward(forward_batch, skip_attn_backend_init=True)
            finally:
                self.draft_runner.graph_runner = saved_gr

            if self._smc_cross_profile:
                self._pending_stats["calls"] += 1
                self._pending_stats["buckets"] += 1
                self._pending_stats["rows"] += len(rows)
                self._pending_stats["tokens"] += len(rows) * suffix_len

    def _commit_pending_cross_draft_suffix_loop(
        self,
        batch: ModelWorkerBatch,
        draft_ctx: SMCDecodeContext,
        pending_ids: torch.Tensor,
        pending_lens: torch.Tensor,
        *,
        skip_last: bool = False,
        record_pending_stats: bool = False,
    ) -> None:
        """Per-position oracle for pending suffix slots."""
        lens_cpu = pending_lens.detach().to("cpu").to(torch.int64)
        lengths = [int(x) for x in lens_cpu.tolist()]
        max_suffix = max(lengths, default=0)
        if max_suffix <= 0:
            return
        device = self.device
        req_rows_all = batch.smc_draft_req_pool_indices
        for step in range(max_suffix):
            rows = [
                row
                for row, length in enumerate(lengths)
                if step < length and not (skip_last and step == length - 1)
            ]
            if not rows:
                continue
            rows_t = torch.tensor(rows, dtype=torch.int64, device=device)
            step_t = pending_ids[rows_t, step].to(torch.int64)
            row_lens = pending_lens[rows_t].to(torch.int64)
            positions = draft_ctx.orig_seq_lens[rows_t] - row_lens + step
            seq_lens = positions + 1
            seq_lens_cpu = seq_lens.to("cpu")
            req_rows = req_rows_all[rows_t]
            out_cache_loc = self.draft_req_to_token_pool.req_to_token[
                req_rows, positions
            ].contiguous()
            suffix_batch = dataclasses.replace(
                batch,
                input_ids=step_t,
                req_pool_indices=req_rows,
                out_cache_loc=out_cache_loc,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=int(seq_lens_cpu.sum().item()),
                spec_info=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            suffix_fb = ForwardBatch.init_new(suffix_batch, self.draft_runner)
            self.draft_runner.forward(suffix_fb)
            if record_pending_stats and self._smc_cross_profile:
                self._pending_stats["calls"] += 1
                self._pending_stats["buckets"] += 1
                self._pending_stats["rows"] += len(rows)
                self._pending_stats["tokens"] += len(rows)

    def _verify_pending_cross_draft_suffix(
        self,
        batch: ModelWorkerBatch,
        draft_ctx: SMCDecodeContext,
        pending_ids: torch.Tensor | None,
        pending_lens: torch.Tensor | None,
    ) -> None:
        if pending_ids is None or pending_lens is None:
            return
        use_linear = bool(int(os.environ.get("SMC_CROSS_PENDING_LINEAR", "0")))
        skip_last = self.smc_defer_bonus and not use_linear
        lens_cpu = pending_lens.detach().to("cpu").to(torch.int64)
        lengths = [int(x) for x in lens_cpu.tolist()]
        real_slots = []
        real_steps = []
        for row, length in enumerate(lengths):
            if length <= 0:
                continue
            start = int(draft_ctx.orig_seq_lens[row].item()) - length
            req_row = int(batch.smc_draft_req_pool_indices[row].item())
            compare_len = max(0, length - 1) if skip_last else length
            for step in range(compare_len):
                pos = start + step
                real_slots.append(
                    int(self.draft_req_to_token_pool.req_to_token[req_row, pos].item())
                )
                real_steps.append(step)
        if not real_slots:
            self._commit_pending_cross_draft_suffix_loop(
                batch, draft_ctx, pending_ids, pending_lens
            )
            self._smc_pending_verify_calls += 1
            print(
                f"[SMC_PENDING_VERIFY] call={self._smc_pending_verify_calls} "
                "only deferred head-covered suffix slots; loop oracle kept",
                flush=True,
            )
            return

        slots = torch.tensor(real_slots, dtype=torch.int64, device=self.device)
        step0_mask = torch.tensor(
            [step == 0 for step in real_steps], dtype=torch.bool, device=self.device
        )
        self._run_pending_cross_draft_suffix_heads(
            batch, draft_ctx, pending_ids, pending_lens
        )
        k_new, v_new = self._read_draft_kv_at(slots)
        self._commit_pending_cross_draft_suffix_loop(
            batch, draft_ctx, pending_ids, pending_lens
        )
        k_old, v_old = self._read_draft_kv_at(slots)
        self._smc_pending_verify_calls += 1

        def _stats(new_list, old_list, row_mask):
            adiff = 0.0
            amax = 0.0
            rel = 0.0
            for a, b in zip(new_list, old_list):
                if row_mask is not None:
                    a = a[row_mask]
                    b = b[row_mask]
                if a.numel() == 0:
                    continue
                d = (a - b).abs()
                adiff = max(adiff, float(d.max()))
                amax = max(amax, float(b.abs().max()))
                denom = b.abs().clamp_min(1e-4)
                rel = max(rel, float((d / denom).max()))
            return adiff, amax, rel

        kd_all = _stats(k_new, k_old, None)
        vd_all = _stats(v_new, v_old, None)
        kd_j0 = _stats(k_new, k_old, step0_mask)
        print(
            f"[SMC_PENDING_VERIFY] call={self._smc_pending_verify_calls} "
            f"slots={len(real_slots)} max_suffix={max(lengths, default=0)} | "
            f"skip_last={int(skip_last)} | "
            f"K absdiff={kd_all[0]:.3e} maxval={kd_all[1]:.2f} relmax={kd_all[2]:.2e} | "
            f"V absdiff={vd_all[0]:.3e} maxval={vd_all[1]:.2f} relmax={vd_all[2]:.2e} | "
            f"K(j0only) absdiff={kd_j0[0]:.3e} relmax={kd_j0[2]:.2e}",
            flush=True,
        )

    def _commit_verify_extend_vs_loop(
        self, batch, draft_ctx, cache_locs, suffix_ids, position_start, max_suffix
    ) -> None:
        """Run extend, snapshot suffix-slot KV, then run the loop (overwriting
        the same slots) and report max |Δ|.  The run keeps the loop's values."""
        real_slots = [
            int(cache_locs[r, position_start + j].item())
            for r, ids in enumerate(suffix_ids)
            for j in range(len(ids))
        ]
        if not real_slots:
            return
        # Index (within real_slots) of each row's FIRST real token (j==0) so we
        # can isolate "prefix+self only" tokens from within-block (j>=1) tokens.
        j0_mask = torch.zeros(len(real_slots), dtype=torch.bool)
        pos = 0
        for _r, ids in enumerate(suffix_ids):
            if ids:
                j0_mask[pos] = True
                pos += len(ids)
        slots_t = torch.tensor(real_slots, dtype=torch.int64, device=self.device)
        self._commit_cross_draft_suffix_extend(
            batch, draft_ctx, cache_locs, suffix_ids, position_start, max_suffix
        )
        k_new, v_new = self._read_draft_kv_at(slots_t)
        self._commit_cross_draft_suffix_loop(
            batch, draft_ctx, cache_locs, suffix_ids, position_start, max_suffix
        )
        k_old, v_old = self._read_draft_kv_at(slots_t)
        self._smc_commit_verify_calls += 1
        if self._smc_commit_verify_calls <= 30:
            def _stats(new_list, old_list, row_mask):
                adiff = 0.0
                amax = 0.0
                rel = 0.0
                for a, b in zip(new_list, old_list):
                    if row_mask is not None:
                        a = a[row_mask]
                        b = b[row_mask]
                    if a.numel() == 0:
                        continue
                    d = (a - b).abs()
                    adiff = max(adiff, float(d.max()))
                    amax = max(amax, float(b.abs().max()))
                    denom = b.abs().clamp_min(1e-4)
                    rel = max(rel, float((d / denom).max()))
                return adiff, amax, rel

            jm = j0_mask.to(self.device)
            kd_all = _stats(k_new, k_old, None)
            vd_all = _stats(v_new, v_old, None)
            kd_j0 = _stats(k_new, k_old, jm)
            print(
                f"[SMC_COMMIT_VERIFY] call={self._smc_commit_verify_calls} "
                f"slots={len(real_slots)} max_suffix={max_suffix} | "
                f"K absdiff={kd_all[0]:.3e} maxval={kd_all[1]:.2f} relmax={kd_all[2]:.2e} | "
                f"V absdiff={vd_all[0]:.3e} maxval={vd_all[1]:.2f} relmax={vd_all[2]:.2e} | "
                f"K(j0only) absdiff={kd_j0[0]:.3e} relmax={kd_j0[2]:.2e}",
                flush=True,
            )

    def _forward_decode_cross_tokenizer(self, batch: ModelWorkerBatch):
        if batch.forward_mode.is_idle():
            return self._forward_idle(batch)

        draft_input: SMCDraftInput = batch.spec_info
        profile = self._smc_cross_profile

        def _profile_mark():
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            return time.perf_counter()

        t0 = _profile_mark()
        draft_tokens, draft_logprobs, draft_deferred, draft_cache_locs = (
            self._run_cross_tokenizer_draft_ar(batch, draft_input)
        )
        t1 = _profile_mark()
        ctx = draft_input.decode_ctx
        if ctx is None:
            raise RuntimeError("cross-tokenizer decode is missing target context.")

        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma
        device = self.device

        # The GPU identity passthrough is equivalence-checkable against the CPU
        # mapping/lineage path: forcing the CPU path on identity vocab must
        # produce byte-identical output at a fixed seed (SMC_CROSS_FORCE_CPU_LINEAGE=1).
        identity = self.cross_tokenizer_mapper._identity_vocab and not self._smc_force_cpu_lineage
        if identity:
            # Identity vocab (e.g. Llama draft forced through the cross path):
            # map_batch returns the draft block verbatim with proxy_len == gamma
            # (never overlength), so build the proxy tensors directly on-device.
            # This skips the D2H sync + CPU tokenizer mapping + H2D round-trip
            # that otherwise stalls the launch pipeline before every verify.
            mapped = None
            proxy_tokens = torch.stack(draft_tokens[1 : gamma + 1], dim=1).to(
                torch.int64
            )
            proxy_logq = draft_logprobs.to(torch.float32)
            bucket_size = gamma
            proxy_lens = torch.full((bs,), gamma, dtype=torch.int64, device=device)
            valid_mask = torch.ones((bs, gamma), dtype=torch.bool, device=device)
            emit_target_bonus = torch.ones(
                (bs,), dtype=torch.bool, device=device
            )
            emit_target_bonus_host = torch.ones((bs,), dtype=torch.bool)
            t2 = _profile_mark()
            t3 = t2
        else:
            draft_ids_cpu = torch.stack(
                draft_tokens[1 : gamma + 1], dim=1
            ).detach().cpu()
            draft_lp_cpu = draft_logprobs.detach().cpu()
            mapped = self.cross_tokenizer_mapper.map_batch(
                draft_ids_cpu.tolist(),
                draft_lp_cpu.tolist(),
                max_proxy_len=gamma,
                boundary_states=draft_input.mapping_boundary_states,
                flush=not self._smc_boundary_carry,
            )
            t2 = _profile_mark()
            if (
                self._smc_cross_stats_interval > 0
                and self.cross_tokenizer_mapper.stats.calls
                >= self._smc_cross_next_stats_at
            ):
                print(self.cross_tokenizer_mapper.format_stats(), flush=True)
                self._smc_cross_next_stats_at += self._smc_cross_stats_interval

            kept_ids = [
                row[:length] for row, length in zip(mapped.proxy_ids, mapped.proxy_lens)
            ]
            kept_logq = [
                row[:length]
                for row, length in zip(mapped.proxy_logq, mapped.proxy_lens)
            ]
            max_kept_len = max(mapped.proxy_lens, default=0)
            use_short_proxy_buckets = bool(
                int(os.environ.get("SMC_CROSS_ENABLE_SHORT_PROXY_BUCKETS", "0"))
            )
            if max_kept_len == 0 and use_short_proxy_buckets:
                bucket_size = 0
                bucket_ids = [[] for _ in range(bs)]
                bucket_logq = [[] for _ in range(bs)]
                bucket_valid = [[] for _ in range(bs)]
            else:
                if use_short_proxy_buckets:
                    buckets = tuple(
                        sorted(
                            {int(x) for x in DEFAULT_PROXY_BUCKETS if int(x) <= gamma}
                            | {gamma}
                        )
                    )
                else:
                    # SGLang's target CUDA graph runner is captured for the full
                    # gamma+1 verify shape.  Shorter proxy buckets are available
                    # for investigation via SMC_CROSS_ENABLE_SHORT_PROXY_BUCKETS,
                    # but the benchmark default keeps the graph hot path.
                    buckets = (gamma,)
                bucketed = bucket_proxy_sequences(
                    kept_ids,
                    kept_logq,
                    pad_token_id=0,
                    buckets=buckets,
                )
                bucket_size = bucketed.bucket_size
                bucket_ids = bucketed.input_ids
                bucket_logq = bucketed.logq
                bucket_valid = bucketed.valid_mask
            t3 = _profile_mark()

            proxy_tokens = torch.tensor(
                bucket_ids, dtype=torch.int64, device=device
            )
            proxy_logq = torch.tensor(
                bucket_logq, dtype=torch.float32, device=device
            )
            proxy_lens = torch.tensor(
                mapped.proxy_lens, dtype=torch.int64, device=device
            )
            valid_mask = torch.tensor(
                bucket_valid, dtype=torch.bool, device=device
            )
            # A target bonus may only follow a fully committed mapped event.
            # If an unsafe suffix remains, accepting a bonus now would place it
            # before that suffix and reverse byte order.
            emit_target_bonus_host = torch.tensor(
                [state.is_empty for state in mapped.boundary_states],
                dtype=torch.bool,
            )
            emit_target_bonus = emit_target_bonus_host.to(device)

        all_target_tokens = [draft_input.verified_id.to(torch.int64)]
        all_target_tokens.extend(
            proxy_tokens[:, step].contiguous() for step in range(bucket_size)
        )
        target_cache_flat = torch.empty(
            bs * (bucket_size + 1), dtype=torch.int64, device=device
        )
        assign_smc_cache_locs_kernel[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            ctx.orig_seq_lens,
            target_cache_flat,
            self.req_to_token_pool.req_to_token.shape[1],
            bucket_size + 1,
        )
        target_cache_locs = target_cache_flat.reshape(bs, bucket_size + 1)
        verify_forward_batch, can_run_cuda_graph = ctx.prepare_for_verify(
            self.req_to_token_pool,
            batch,
            self._target_worker,
            all_target_tokens,
            target_cache_locs,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            proxy_token_num=bucket_size,
        )

        score_result = self._target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        t4 = _profile_mark()
        score_logits = score_result.logits_output.next_token_logits
        score_log_probs = torch.log_softmax(
            score_logits / self.smc_target_temperature, dim=-1
        ).reshape(bs, bucket_size + 1, -1)
        if bucket_size > 0:
            score_logprobs = score_log_probs[:, :bucket_size, :].gather(
                2, proxy_tokens.unsqueeze(2)
            ).squeeze(2)
        else:
            score_logprobs = torch.empty((bs, 0), dtype=torch.float32, device=device)
        logprob_diff = self.smc_power_alpha * score_logprobs - proxy_logq
        logprob_diff = logprob_diff * valid_mask.to(logprob_diff.dtype)
        if profile:
            valid = valid_mask.to(torch.bool)
            n_valid = int(valid.sum().item())
            ps = self._proposal_stats
            if n_valid > 0:
                diffs = logprob_diff[valid].to(torch.float32)
                target_vals = score_logprobs[valid].to(torch.float32)
                draft_vals = proxy_logq[valid].to(torch.float32)
                ps["tokens"] += n_valid
                ps["logdiff_sum"] += float(diffs.sum().item())
                ps["logdiff_sumsq"] += float((diffs * diffs).sum().item())
                ps["logdiff_min"] = min(ps["logdiff_min"], float(diffs.min().item()))
                ps["logdiff_max"] = max(ps["logdiff_max"], float(diffs.max().item()))
                ps["target_logp_sum"] += float(target_vals.sum().item())
                ps["draft_logq_sum"] += float(draft_vals.sum().item())
            if proxy_lens.numel() > 0:
                proxy_lens_f = proxy_lens.to(torch.float32)
                ps["proxy_len_sum"] += float(proxy_lens_f.sum().item())
                ps["proxy_len_max"] = max(
                    ps["proxy_len_max"], float(proxy_lens_f.max().item())
                )

        bonus_rows = proxy_lens.clamp(max=gamma)
        bonus_logits = score_logits.reshape(bs, bucket_size + 1, -1)[
            torch.arange(bs, device=device), bonus_rows, :
        ]
        sampled_bonus, bonus_logz = self._sample_target_power(bonus_logits)
        if bucket_size > 0:
            last_proxy_col = (proxy_lens - 1).clamp(min=0, max=bucket_size - 1)
            last_proxy = proxy_tokens.gather(
                1, last_proxy_col.unsqueeze(1)
            ).squeeze(1)
            carry_verified = torch.where(
                proxy_lens > 0,
                last_proxy,
                draft_input.verified_id.to(torch.int64),
            )
        else:
            carry_verified = draft_input.verified_id.to(torch.int64)
        bonus = torch.where(
            emit_target_bonus,
            sampled_bonus.to(torch.int64),
            carry_verified,
        )
        output_token_ids = torch.cat([proxy_tokens, bonus.unsqueeze(1)], dim=1)

        if identity:
            # Identity lineage is fully GPU-computable.  Because the bonus is a
            # single draft token, plan_cross_draft_lineage reduces to
            # history = [verified] + proposal, next seed = bonus,
            # prev_last = last proposal token, with every suffix commit empty
            # (bonus_prefix == []).  Building it on-device removes the
            # post-verify D2H sync, the per-row bonus retok, and the Python
            # lineage loop, leaving the identity cross path sync-free like the
            # same-tokenizer path.
            draft_verified_seed = draft_input.draft_verified_id.to(torch.int64)
            draft_accepted_ids = torch.cat(
                [draft_verified_seed.unsqueeze(1), proxy_tokens], dim=1
            )
            draft_accepted_lens = torch.full(
                (bs,), gamma + 1, dtype=torch.int64, device=device
            )
            draft_visible_lens = draft_accepted_lens.clone()
            draft_verified_id = bonus.to(torch.int64)
            prev_last_draft_id = proxy_tokens[:, -1].to(torch.int64)
            proxy_lens_host = None
            draft_visible_lens_host = None
            pending_suffix_lens_host = torch.zeros(bs, dtype=torch.int64)
            pending_suffix_lens = torch.zeros(
                bs, dtype=torch.int64, device=device
            )
        else:
            # Single batched device->host copy for every host value this step
            # needs: the sampled bonus plus the previous-step draft lineage
            # seeds.  This replaces three separate .cpu() syncs (bonus,
            # draft_verified_id, prev_last_draft_id) that otherwise each stall
            # the launch pipeline on the cross-tokenizer hot path.
            _qm0 = _profile_mark()
            lineage_host = torch.stack(
                [
                    bonus.to(torch.int64),
                    draft_input.draft_verified_id.to(torch.int64),
                    draft_input.prev_last_draft_id.to(torch.int64),
                ],
                dim=0,
            ).cpu()
            bonus_cpu = lineage_host[0].tolist()
            current_draft_verified = lineage_host[1].tolist()
            previous_last_draft = lineage_host[2].tolist()
            _qm1 = _profile_mark()
            bonus_draft_ids = self._bonus_draft_ids(bonus_cpu)
            bonus_draft_ids = [
                ids if bool(emit) else []
                for ids, emit in zip(
                    bonus_draft_ids,
                    emit_target_bonus_host.tolist(),
                    strict=True,
                )
            ]
            bonus_draft_lens = [len(row) for row in bonus_draft_ids]
            _qm2 = _profile_mark()

            if self._smc_dbg_cross and self._smc_dbg_cross_calls < 5:
                self._smc_dbg_cross_calls += 1
                sample = min(bs, 8)
                print(
                    "[SMC_CROSS_DBG] "
                    f"decode={self._smc_dbg_cross_calls} bs={bs} "
                    f"raw_lens={mapped.raw_proxy_lens[:sample]} "
                    f"kept_lens={mapped.proxy_lens[:sample]} "
                    f"bucket={bucket_size} "
                    f"overlength={mapped.overlength_rows}/{bs} "
                    f"bonus_draft_lens={bonus_draft_lens[:sample]} "
                    f"bonus={bonus[:sample].detach().cpu().tolist()}",
                    flush=True,
                )

            # Build the draft-side lineage that spells the same visible text as
            # the target side.  `draft_seq_lens` excludes `draft_verified_id`,
            # so each row records the prefix delta up to, but not including, the
            # final draft token of the target bonus.
            proposal_draft_ids = draft_ids_cpu.tolist()
            lineage = plan_cross_draft_lineage(
                current_verified_ids=current_draft_verified,
                previous_last_draft_ids=previous_last_draft,
                proposal_draft_ids=proposal_draft_ids,
                bonus_draft_ids=bonus_draft_ids,
                overlength_mask=mapped.overlength_mask,
                draft_deferred=draft_deferred,
                emit_target_bonus=emit_target_bonus_host.tolist(),
            )
            draft_verified_id = torch.tensor(
                lineage.draft_verified_ids, dtype=torch.int64, device=device
            )
            prev_last_draft_id = torch.tensor(
                lineage.prev_last_draft_ids, dtype=torch.int64, device=device
            )
            _qm3 = _profile_mark()

            pending_suffixes = self._combine_pending_suffixes(
                lineage.normal_deferred_suffix,
                lineage.normal_legacy_suffix,
                lineage.overlength_suffix,
            )
            pending_suffix_lens_host = torch.tensor(
                [len(row) for row in pending_suffixes], dtype=torch.int64
            )
            pending_suffix_lens = torch.tensor(
                pending_suffix_lens_host.tolist(), dtype=torch.int64, device=device
            )

            if not (self._smc_pending_suffix or self._smc_pending_verify):
                self._commit_cross_draft_suffix(
                    batch,
                    draft_input.draft_decode_ctx,
                    draft_cache_locs,
                    lineage.normal_deferred_suffix,
                    position_start=gamma,
                )
                self._commit_cross_draft_suffix(
                    batch,
                    draft_input.draft_decode_ctx,
                    draft_cache_locs,
                    lineage.normal_legacy_suffix,
                    position_start=gamma + 1,
                )
                self._commit_cross_draft_suffix(
                    batch,
                    draft_input.draft_decode_ctx,
                    draft_cache_locs,
                    lineage.overlength_suffix,
                    position_start=1,
                )
            _qm4 = _profile_mark()

            max_history_len = max((len(row) for row in lineage.histories), default=0)
            if max_history_len:
                # One host->device copy of the padded (bs, max_history_len)
                # block instead of a per-row torch.tensor()+scatter loop (bs
                # separate H2D copies + kernel launches every decode step).
                padded_histories = [
                    row + [0] * (max_history_len - len(row))
                    for row in lineage.histories
                ]
                draft_accepted_ids = torch.tensor(
                    padded_histories, dtype=torch.int64, device=device
                )
            else:
                draft_accepted_ids = torch.zeros(
                    (bs, 0), dtype=torch.int64, device=device
                )
            draft_accepted_lens = torch.tensor(
                lineage.draft_visible_lens, dtype=torch.int64, device=device
            )
            draft_visible_lens = draft_accepted_lens.clone()
            if self._smc_cross_host_lens:
                # G1: keep CPU copies of the host lists used to build proxy_lens /
                # draft_visible_lens so write_back can update its host mirrors
                # without a device->host sync.  No GPU work, no copy-back.
                proxy_lens_host = torch.tensor(
                    mapped.proxy_lens, dtype=torch.int64
                )
                draft_visible_lens_host = torch.tensor(
                    lineage.draft_visible_lens, dtype=torch.int64
                )
            else:
                proxy_lens_host = None
                draft_visible_lens_host = None
            _qm5 = _profile_mark()
            if profile:
                pa = self._smc_cross_post_acc
                pa["d2h"] += (_qm1 - _qm0) * 1000.0
                pa["retok"] += (_qm2 - _qm1) * 1000.0
                pa["lineage"] += (_qm3 - _qm2) * 1000.0
                pa["commit"] += (_qm4 - _qm3) * 1000.0
                pa["h2d"] += (_qm5 - _qm4) * 1000.0
        next_draft_input = SMCDraftInput(
            verified_id=bonus,
            # d_0..d_{gamma-1} were mapped/accepted on the target side.
            # Legacy mode over-drafts one token, so d_gamma is the next seed.
            # Deferred mode writes only through d_{gamma-2}; d_{gamma-1}
            # becomes the next seed and is written by the next head.
            draft_verified_id=draft_verified_id,
            prev_last_draft_id=prev_last_draft_id,
            logprob_diff=logprob_diff,
            bonus_logz=bonus_logz,
            num_tokens_per_req=self.speculative_num_draft_tokens,
            proxy_valid_mask=valid_mask,
            proxy_lens=proxy_lens,
            emit_target_bonus=emit_target_bonus,
            emit_target_bonus_host=emit_target_bonus_host,
            draft_accepted_ids=draft_accepted_ids,
            draft_accepted_lens=draft_accepted_lens,
            draft_visible_lens=draft_visible_lens,
            identity_const_lens=bool(identity and self._smc_fast_writeback),
            proxy_lens_host=proxy_lens_host,
            draft_visible_lens_host=draft_visible_lens_host,
            fast_writeback_lens=bool(
                self._smc_cross_host_lens and not identity
            ),
            pending_draft_suffix_lens=pending_suffix_lens,
            pending_draft_suffix_lens_host=pending_suffix_lens_host,
            mapping_boundary_states=(
                mapped.boundary_states if mapped is not None else None
            ),
        )
        if not identity:
            self._maybe_dump_cross_trace(
                batch=batch,
                draft_input=draft_input,
                draft_ids_cpu=draft_ids_cpu.tolist(),
                draft_lp_cpu=draft_lp_cpu.tolist(),
                mapped=mapped,
                score_log_probs=score_log_probs,
                score_logprobs=score_logprobs,
                proxy_logq=proxy_logq,
                logprob_diff=logprob_diff,
                valid_mask=valid_mask,
                proxy_lens=proxy_lens,
                emit_target_bonus=emit_target_bonus,
                bonus=bonus,
                lineage=lineage,
            )
        t5 = _profile_mark()
        if profile:
            self._smc_cross_profile_calls += 1
            acc = self._smc_cross_profile_acc
            acc["draft_ms"] += (t1 - t0) * 1000.0
            acc["map_ms"] += (t2 - t1) * 1000.0
            acc["bucket_ms"] += (t3 - t2) * 1000.0
            acc["target_ms"] += (t4 - t3) * 1000.0
            acc["post_ms"] += (t5 - t4) * 1000.0
            calls = self._smc_cross_profile_calls
            if calls % self._smc_cross_profile_interval == 0:
                scale = 1.0 / calls
                print(
                    "[SMC_CROSS_PROFILE] "
                    f"calls={calls} "
                    f"draft_ms={acc['draft_ms'] * scale:.2f} "
                    f"map_ms={acc['map_ms'] * scale:.2f} "
                    f"bucket_ms={acc['bucket_ms'] * scale:.2f} "
                    f"target_ms={acc['target_ms'] * scale:.2f} "
                    f"post_ms={acc['post_ms'] * scale:.2f}",
                    flush=True,
                )
                pa = self._smc_cross_post_acc
                print(
                    "[SMC_CROSS_PROFILE]   post split: "
                    f"d2h={pa['d2h'] * scale:.2f} "
                    f"retok={pa['retok'] * scale:.2f} "
                    f"lineage={pa['lineage'] * scale:.2f} "
                    f"commit={pa['commit'] * scale:.2f} "
                    f"h2d={pa['h2d'] * scale:.2f}",
                    flush=True,
                )
                cs = self._commit_stats
                print(
                    "[SMC_CROSS_PROFILE]   commit stats: "
                    f"fwd_calls/step={cs['calls'] * scale:.2f} "
                    f"fwd_positions/step={cs['fwd_positions'] * scale:.2f} "
                    f"row_steps/step={cs['row_steps'] * scale:.2f} (bs-rows)",
                    flush=True,
                )
                ps = self._pending_stats
                print(
                    "[SMC_CROSS_PROFILE]   pending stats: "
                    f"calls/step={ps['calls'] * scale:.2f} "
                    f"buckets/step={ps['buckets'] * scale:.2f} "
                    f"rows/step={ps['rows'] * scale:.2f} "
                    f"tokens/step={ps['tokens'] * scale:.2f}",
                    flush=True,
                )
                qs = self._proposal_stats
                if qs["tokens"] > 0:
                    n_tok = float(qs["tokens"])
                    mean = qs["logdiff_sum"] / n_tok
                    var = max(qs["logdiff_sumsq"] / n_tok - mean * mean, 0.0)
                    print(
                        "[SMC_CROSS_PROFILE]   proposal stats: "
                        f"tokens/step={qs['tokens'] * scale:.2f} "
                        f"logdiff_mean={mean:.3f} "
                        f"logdiff_std={var ** 0.5:.3f} "
                        f"logdiff_min={qs['logdiff_min']:.3f} "
                        f"logdiff_max={qs['logdiff_max']:.3f} "
                        f"target_logp_mean={qs['target_logp_sum'] / n_tok:.3f} "
                        f"draft_logq_mean={qs['draft_logq_sum'] / n_tok:.3f} "
                        f"proxy_len_mean={qs['proxy_len_sum'] / calls:.2f} "
                        f"proxy_len_max={qs['proxy_len_max']:.0f}",
                        flush=True,
                    )
                mapper = self.cross_tokenizer_mapper
                if mapper is not None and getattr(mapper, "_heal_boundaries", False):
                    scale_iv = 1.0 / self._smc_cross_profile_interval
                    last = self._smc_cross_heal_last
                    cache = getattr(mapper, "cache", None)
                    cur = {
                        "groups": int(getattr(mapper, "_heal_groups", 0)),
                        "reencodes": int(getattr(mapper, "_heal_reencodes", 0)),
                        "hits": int(getattr(cache, "hits", 0)),
                        "misses": int(getattr(cache, "misses", 0)),
                    }
                    d_groups = cur["groups"] - last["groups"]
                    d_reenc = cur["reencodes"] - last["reencodes"]
                    d_hits = cur["hits"] - last["hits"]
                    d_misses = cur["misses"] - last["misses"]
                    # Interval heal-cache hit rate (groups served from cache).
                    heal_hit_rate = (
                        (d_groups - d_reenc) / d_groups if d_groups > 0 else 0.0
                    )
                    # Overall LFU (heal + fallback) hit rate this interval.
                    lookups = d_hits + d_misses
                    cache_hit_rate = d_hits / lookups if lookups > 0 else 0.0
                    print(
                        "[SMC_CROSS_PROFILE]   heal stats: "
                        f"groups/step={d_groups * scale_iv:.2f} "
                        f"reencodes/step={d_reenc * scale_iv:.3f} "
                        f"heal_hit_rate={heal_hit_rate:.3f} "
                        f"cache_hit_rate={cache_hit_rate:.3f} "
                        f"cache_size={len(getattr(cache, '_data', {}))}",
                        flush=True,
                    )
                    self._smc_cross_heal_last = cur

        return GenerationBatchResult(
            logits_output=score_result.logits_output,
            next_token_ids=output_token_ids.reshape(-1),
            accept_lens=(
                proxy_lens + emit_target_bonus.to(proxy_lens.dtype)
            ).to(torch.int32),
            next_draft_input=next_draft_input,
            logprob_diff=logprob_diff,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _maybe_dump_cross_trace(
        self,
        *,
        batch: ModelWorkerBatch,
        draft_input: SMCDraftInput,
        draft_ids_cpu,
        draft_lp_cpu,
        mapped,
        score_log_probs: torch.Tensor,
        score_logprobs: torch.Tensor,
        proxy_logq: torch.Tensor,
        logprob_diff: torch.Tensor,
        valid_mask: torch.Tensor,
        proxy_lens: torch.Tensor,
        emit_target_bonus: torch.Tensor,
        bonus: torch.Tensor,
        lineage,
    ) -> None:
        """Write opt-in cross-tokenizer SMC proposal traces.

        This is intentionally JSONL and env-gated so production/eval behavior is
        unchanged unless SMC_TRACE_JSONL is set.
        """
        if not self._smc_trace_jsonl:
            return
        if self._smc_trace_limit > 0 and self._smc_trace_rows >= self._smc_trace_limit:
            return
        try:
            req_pool = batch.req_pool_indices.detach().cpu().tolist()
            verified = draft_input.verified_id.detach().cpu().tolist()
            draft_verified = draft_input.draft_verified_id.detach().cpu().tolist()
            prev_last = draft_input.prev_last_draft_id.detach().cpu().tolist()
            target_lps = score_logprobs.detach().cpu().tolist()
            proxy_logq_cpu = proxy_logq.detach().cpu().tolist()
            logdiff_cpu = logprob_diff.detach().cpu().tolist()
            valid_cpu = valid_mask.detach().cpu().tolist()
            proxy_lens_cpu = proxy_lens.detach().cpu().tolist()
            emit_target_bonus_cpu = emit_target_bonus.detach().cpu().tolist()
            bonus_cpu = bonus.detach().cpu().tolist()
            active_slots = getattr(batch, "smc_trace_active_slots", None)
            target_prefixes = getattr(batch, "smc_trace_target_prefix_ids", None)
            draft_prefixes = getattr(batch, "smc_trace_draft_prefix_ids", None)
            topk_ids = topk_lps = None
            if self._smc_trace_topk > 0 and score_log_probs.numel() > 0:
                k = min(self._smc_trace_topk, score_log_probs.shape[-1])
                vals, ids = torch.topk(score_log_probs[:, :-1, :], k=k, dim=-1)
                topk_ids = ids.detach().cpu().tolist()
                topk_lps = vals.detach().cpu().tolist()

            records = []
            for row_idx, pool_idx in enumerate(req_pool):
                if self._smc_trace_limit > 0 and (
                    self._smc_trace_rows + len(records)
                ) >= self._smc_trace_limit:
                    break
                plen = int(proxy_lens_cpu[row_idx])
                valid_width = len(valid_cpu[row_idx])
                keep = [bool(x) for x in valid_cpu[row_idx]]
                proxy_ids = list(mapped.proxy_ids[row_idx])[:plen]
                req = batch.reqs[row_idx] if getattr(batch, "reqs", None) else None
                segments = [
                    {
                        "draft_start": int(seg.draft_start),
                        "draft_end": int(seg.draft_end),
                        "target_start": int(seg.target_start),
                        "target_end": int(seg.target_end),
                        "draft_logq": float(seg.draft_logq),
                    }
                    for seg in getattr(mapped, "segments", [[]])[row_idx]
                ]
                raw_segments = getattr(mapped, "segments", [[]])[row_idx]
                draft_ids = list(draft_ids_cpu[row_idx])
                draft_text = decode_exact(
                    self.cross_tokenizer_mapper.draft_tokenizer, draft_ids
                )
                canonical_target_ids = encode_text(
                    self.cross_tokenizer_mapper.target_tokenizer, draft_text
                )
                shared_spans = shared_text_spans(
                    draft_tokenizer=self.cross_tokenizer_mapper.draft_tokenizer,
                    target_tokenizer=self.cross_tokenizer_mapper.target_tokenizer,
                    draft_ids=draft_ids,
                    target_ids=proxy_ids,
                    segments=raw_segments,
                )
                record = {
                    "trace_kind": "cross_smc_step",
                    "trace_schema": 4,
                    "decode_call": int(self._smc_trace_decode_call),
                    "batch_row": int(row_idx),
                    "tp_rank": self.tp_rank,
                    "req_pool_idx": int(pool_idx),
                    "active_slot": (
                        int(active_slots[row_idx]) if active_slots is not None else None
                    ),
                    "rid": str(getattr(req, "rid", "")) if req is not None else None,
                    "particle_idx": (
                        int(getattr(req, "smc_particle_idx", -1))
                        if req is not None
                        else None
                    ),
                    "gamma": int(self.gamma),
                    "target_prefix_ids": (
                        list(target_prefixes[row_idx])
                        if target_prefixes is not None
                        else None
                    ),
                    "draft_prefix_ids": (
                        list(draft_prefixes[row_idx])
                        if draft_prefixes is not None
                        else None
                    ),
                    "verified_target_id": int(verified[row_idx]),
                    "draft_verified_id": int(draft_verified[row_idx]),
                    "prev_last_draft_id": int(prev_last[row_idx]),
                    "proposal_draft_ids": draft_ids,
                    "proposal_draft_logq": list(draft_lp_cpu[row_idx]),
                    "proxy_target_ids": proxy_ids,
                    "proposal_draft_text": draft_text,
                    "proxy_target_text": decode_exact(
                        self.cross_tokenizer_mapper.target_tokenizer, proxy_ids
                    ),
                    "canonical_target_ids": canonical_target_ids,
                    "proxy_is_canonical": proxy_ids == canonical_target_ids,
                    "raw_proxy_len": int(mapped.raw_proxy_lens[row_idx]),
                    "proxy_len": plen,
                    "emit_target_bonus": bool(
                        emit_target_bonus_cpu[row_idx]
                    ),
                    "overlength": bool(mapped.overlength_mask[row_idx]),
                    "empty_proxy": bool(mapped.empty_proxy_mask[row_idx]),
                    "valid_mask": keep,
                    "target_proxy_lps": target_lps[row_idx][:valid_width],
                    "proxy_logq_lps": proxy_logq_cpu[row_idx][:valid_width],
                    "logprob_diff": logdiff_cpu[row_idx][:valid_width],
                    "segments": segments,
                    "shared_spans": [span.as_dict() for span in shared_spans],
                    "bonus_target_id": int(bonus_cpu[row_idx]),
                    "lineage_draft_visible_len": int(
                        lineage.draft_visible_lens[row_idx]
                    ),
                    "lineage_history": list(lineage.histories[row_idx]),
                    "lineage_normal_deferred_suffix": list(
                        lineage.normal_deferred_suffix[row_idx]
                    ),
                    "lineage_normal_legacy_suffix": list(
                        lineage.normal_legacy_suffix[row_idx]
                    ),
                    "lineage_overlength_suffix": list(
                        lineage.overlength_suffix[row_idx]
                    ),
                    "mapper": {
                        **getattr(mapped, "row_stats", [{}])[row_idx],
                        "boundary_state": {
                            "draft_suffix_ids": list(
                                mapped.boundary_states[
                                    row_idx
                                ].draft_suffix_ids
                            ),
                            "target_suffix_ids": list(
                                mapped.boundary_states[
                                    row_idx
                                ].target_suffix_ids
                            ),
                        },
                    },
                    "runtime_config": {
                        "power_alpha": float(self.smc_power_alpha),
                        "draft_temperature": float(self.smc_draft_temperature),
                        "target_temperature": float(self.smc_target_temperature),
                        "cross_tokenizer_mode": getattr(
                            self.server_args, "smc_cross_tokenizer_mode", "hybrid"
                        ),
                        "boundary_carry": bool(self._smc_boundary_carry),
                    },
                }
                if topk_ids is not None and topk_lps is not None:
                    record["target_topk_ids"] = topk_ids[row_idx][:valid_width]
                    record["target_topk_lps"] = topk_lps[row_idx][:valid_width]
                records.append(record)

            if not records:
                return
            with open(self._smc_trace_jsonl, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._smc_trace_rows += len(records)
            self._smc_trace_decode_call += 1
        except Exception as exc:
            logger.warning("SMC cross trace dump failed: %s", exc)

    def _sample_draft_token(self, logits):
        """Sample one draft token + its log-prob at the draft temperature.

        Identical to the per-step sampling in the legacy AR loop, factored out
        so the deferred-bonus path reuses it verbatim.
        """
        if self.smc_draft_temperature <= 0:
            idx = torch.argmax(logits, dim=-1)
            # Greedy draft is deterministic, so the proposal probability of the
            # selected token is 1 and its log-prob contribution is zero.
            return idx, torch.zeros_like(idx, dtype=torch.float32)
        scaled = logits / self.smc_draft_temperature
        log_probs = torch.log_softmax(scaled, dim=-1)
        idx = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
        lp = log_probs.gather(1, idx.unsqueeze(1)).squeeze(1)
        return idx, lp

    def _sample_target_power(self, logits):
        """Draw from ``softmax(alpha * logits / target_temperature)``.

        Returns the sampled token and the local power-normalizer
        ``logsumexp(alpha*l/T) - alpha*logsumexp(l/T)``. The latter is the
        bonus token's importance-weight increment under the sequence-wise
        unnormalized power target and is exactly zero when ``alpha == 1``.
        """
        base = logits / self.smc_target_temperature
        scaled = self.smc_power_alpha * base
        tiny = torch.finfo(scaled.dtype).tiny
        gumbel = -torch.log(-torch.log(torch.rand_like(scaled).clamp_min_(tiny)))
        idx = torch.argmax(scaled + gumbel, dim=-1)
        logz = torch.logsumexp(scaled, dim=-1) - self.smc_power_alpha * (
            torch.logsumexp(base, dim=-1)
        )
        return idx, logz

    def _draft_ar_deferred(
        self, ctx, draft_input, draft_fb, cache_locs,
        all_positions, all_seq_lens, batch, bs, gamma, req_to_token_pool=None,
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
        if req_to_token_pool is None:
            req_to_token_pool = self.req_to_token_pool

        # 2-token head extend [prev @ S-1, verified_id @ S]; d_0 from the
        # second (S / bonus) column.
        head_fb = ctx.prepare_for_draft_head(
            prev, x0, cache_locs, req_to_token_pool, batch,
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
        # Hybrid draft: the verify-style head defers its recurrent-state update
        # (like the target verify), so the draft's live Mamba state is still at
        # S-2 after the head forward.  Commit the S-position (index 1 = verified
        # token) intermediate state so the gamma-1 single decodes below continue
        # from S rather than the stale S-2 state.  No-op for attention drafts.
        if self._draft_is_hybrid:
            self._commit_draft_mamba_state_after_head(head_fb, bs)

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
        """Run one same-tokenizer decode cycle through the full-cycle graph."""
        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma

        cache_locs = ctx.cache_locs
        if cache_locs is None:
            from smcsd.common.verify import assign_smc_cache_locs_kernel

            req_to_token = self.req_to_token_pool.req_to_token
            out_cache_loc = torch.empty(
                bs * (gamma + 1), dtype=torch.int64, device=self.device
            )
            assign_smc_cache_locs_kernel[(bs,)](
                batch.req_pool_indices,
                req_to_token,
                ctx.orig_seq_lens,
                out_cache_loc,
                req_to_token.shape[1],
                gamma + 1,
            )
            cache_locs = out_cache_loc.reshape(bs, gamma + 1)

        replay_kwargs = {}
        if getattr(self.cycle_graph_runner, "deferred", False):
            replay_kwargs["prev_last_draft_id"] = draft_input.prev_last_draft_id
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
        next_draft_input = SMCDraftInput(
            verified_id=bonus,
            prev_last_draft_id=tokens_out[:, gamma],
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
        if self.smc_cross_tokenizer:
            return self._forward_decode_cross_tokenizer(batch)

        if batch.forward_mode.is_idle():
            return self._forward_idle(batch)

        same_profile = self._smc_same_profile

        def _same_profile_mark():
            if same_profile and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            return time.perf_counter()

        sp_t0 = _same_profile_mark()

        current_stream = torch.get_device_module(self.device).current_stream()
        if batch.req_pool_indices is not None:
            batch.req_pool_indices.record_stream(current_stream)

        draft_input: SMCDraftInput = batch.spec_info
        ctx: SMCDecodeContext = draft_input.decode_ctx

        if draft_input.verified_id is not None:
            draft_input.verified_id.record_stream(current_stream)

        # Full-cycle graphs are valid only for the same-tokenizer path; their
        # initialization is explicitly disabled for cross-tokenizer workers.
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

                if (
                    self._smc_dbg_draft_graph
                    and not self._smc_dbg_draft_graph_done
                    and step == 0
                ):
                    self._smc_dbg_draft_graph_done = True
                    print(
                        "[SMC_DBG_DRAFT_GRAPH] path=same "
                        f"bs={bs} gamma={gamma} can_cuda_graph={can_cuda_graph} "
                        f"use_multistep={use_multistep} "
                        f"forward_can_run_graph={draft_out.can_run_graph}",
                        flush=True,
                    )

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

        sp_t1 = _same_profile_mark()

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

        sp_t2 = _same_profile_mark()

        # ---- 4. Extract score logprobs ----
        score_logits = score_result.logits_output.next_token_logits
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]} rows, "
            f"expected {expected_rows} (bs={bs}, gamma+1={gamma + 1}, "
            f"cuda_graph={can_run_cuda_graph})"
        )
        score_log_probs = torch.log_softmax(
            score_logits / self.smc_target_temperature, dim=-1
        )
        score_log_probs = score_log_probs.reshape(bs, gamma + 1, -1)
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)
        score_logprobs_stacked = score_log_probs[:, :gamma, :].gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)

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
        # bonus and per-step draws come from one consistent target.
        bonus_logits = score_logits.reshape(bs, gamma + 1, -1)[:, -1, :]
        bonus, bonus_logz = self._sample_target_power(bonus_logits)

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

        if same_profile:
            sp_t3 = _same_profile_mark()
            self._smc_same_profile_calls += 1
            acc = self._smc_same_profile_acc
            acc["draft_ms"] += (sp_t1 - sp_t0) * 1000.0
            acc["target_ms"] += (sp_t2 - sp_t1) * 1000.0
            acc["post_ms"] += (sp_t3 - sp_t2) * 1000.0
            calls = self._smc_same_profile_calls
            if calls % self._smc_cross_profile_interval == 0:
                scale = 1.0 / calls
                print(
                    "[SMC_SAME_PROFILE] "
                    f"calls={calls} "
                    f"draft_ms={acc['draft_ms'] * scale:.2f} "
                    f"target_ms={acc['target_ms'] * scale:.2f} "
                    f"post_ms={acc['post_ms'] * scale:.2f}",
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
