"""Hybrid-aware multi-step draft attention backend for SMC.

Upstream sglang's ``DraftBackendFactory.create_decode_backend()`` returns a
flat-attention multi-step backend (``FlashAttentionMultiStepBackend``) that
holds one ``FlashAttentionBackend(speculative_step_id=i)`` per draft AR step.
That works for pure-attention drafts but breaks for hybrid (Mamba+attention)
drafts because:

 1. ``radix_linear_attention.py`` calls ``forward_batch.attn_backend.forward(
    layer=, forward_batch=, mixed_qkv=, a=, b=)`` (linear-attn signature),
    which the flat full-attention backend doesn't implement → TypeError.
 2. There's no Mamba state plumbing — the Mamba layers in the draft model
    would never be reached.

This wrapper builds one ``HybridLinearAttnBackend`` per AR step, where each
step's ``full_attn_backend`` is a step-aware ``FlashAttentionBackend`` (so
positions/seq-lens are pre-baked per step) and the ``linear_attn_backend``
is *shared* across steps. Sharing the Mamba backend is correct because:

 * The forward kernel reads/writes the per-request Mamba state stored in
   the runner's ``req_to_token_pool.mamba2_layer_cache(layer_id)``. State
   evolution across γ+1 steps happens through the pool, not through the
   backend object.
 * ``Mamba2Metadata`` is just chunk-size + per-token bookkeeping, identical
   shape for every decode step in the SMC AR loop.

The SMC worker swaps ``forward_batch.attn_backend = self.attn_backends[step]``
for each AR step (matching the existing non-hybrid multi-step contract) and
invokes ``draft_runner.forward(draft_fb, skip_attn_backend_init=True)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.utils.common import is_blackwell

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class HybridLinearAttnMultiStepBackend:
    """Multi-step draft backend with one HybridLinearAttnBackend per AR step.

    Shape mirrors ``FlashAttentionMultiStepBackend`` (``self.attn_backends``
    list indexed by step), so the SMC worker can use it interchangeably.
    """

    def __init__(
        self,
        draft_model_runner: "ModelRunner",
        topk: int,
        speculative_num_steps: int,
        fa_impl_ver: int = 3,
    ) -> None:
        self.model_runner = draft_model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps

        runner_backend = draft_model_runner.attn_backend
        if not isinstance(runner_backend, HybridLinearAttnBackend):
            raise TypeError(
                "HybridLinearAttnMultiStepBackend requires the draft runner's "
                "attn_backend to be a HybridLinearAttnBackend; got "
                f"{type(runner_backend).__name__}. This backend is only valid "
                "for hybrid (Mamba+attention) draft models."
            )

        # Share the runner's Mamba2 backend across all step backends — Mamba
        # state lives in the req_to_token_pool, not on the backend object.
        shared_linear_backend = runner_backend.linear_attn_backend
        full_attn_layers = runner_backend.full_attn_layers

        # Blackwell falls back to triton for hybrid linear-attn drafts; mirror
        # the choice DraftBackendFactory makes for the non-hybrid case.
        # Triton step backends aren't step-aware the way FA3 is
        # (speculative_step_id); instead TritonMultiStepDraftBackend gives each
        # step a slice of a shared kv_indptr/kv_indices buffer and fills all
        # steps' indices in one generate_draft_decode_kv_indices launch. Embed
        # a real TritonMultiStepDraftBackend and reuse its per-step backends as
        # the full-attn sub-backends, so all index bookkeeping stays upstream.
        self._triton_multistep = None
        if is_blackwell():
            from sglang.srt.layers.attention.triton_backend import (
                TritonMultiStepDraftBackend,
            )

            self._triton_multistep = TritonMultiStepDraftBackend(
                draft_model_runner,
                topk=topk,
                speculative_num_steps=speculative_num_steps,
            )

            def make_full_step(i: int):
                return self._triton_multistep.attn_backends[i]

        else:
            def make_full_step(i: int) -> FlashAttentionBackend:
                return FlashAttentionBackend(
                    draft_model_runner,
                    speculative_step_id=i,
                    topk=topk,
                    speculative_num_steps=speculative_num_steps,
                    fa_impl_ver=fa_impl_ver,
                )

        # ``FlashAttentionMultiStepBackend`` builds ``speculative_num_steps - 1``
        # backends. We build the same count so the indexing matches the
        # non-hybrid contract.
        self.attn_backends: List[HybridLinearAttnBackend] = []
        for i in range(speculative_num_steps - 1):
            step_full = make_full_step(i)
            self.attn_backends.append(
                HybridLinearAttnBackend(
                    full_attn_backend=step_full,
                    linear_attn_backend=shared_linear_backend,
                    full_attn_layers=full_attn_layers,
                )
            )

    # ── Multi-step API (mirrors FlashAttentionMultiStepBackend) ──

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        # Set up per-step full-attention metadata once at the start of the
        # outer SMC step. Mamba metadata is shape-invariant across the AR
        # steps so we only need to compute it once via the shared backend.
        if self._triton_multistep is not None:
            self._triton_multistep.init_forward_metadata(forward_batch)
        else:
            for hb in self.attn_backends:
                hb.full_attn_backend.init_forward_metadata(forward_batch)
        # Single shared mamba init — all step backends point at the same
        # linear_attn_backend instance, so this populates them all.
        if self.attn_backends:
            self.attn_backends[0].linear_attn_backend.init_forward_metadata(
                forward_batch
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        if self._triton_multistep is not None:
            self._triton_multistep.init_cuda_graph_state(max_bs, max_num_tokens)
        else:
            for hb in self.attn_backends:
                hb.full_attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)
        if self.attn_backends:
            self.attn_backends[0].linear_attn_backend.init_cuda_graph_state(
                max_bs, max_num_tokens
            )

    def init_forward_metadata_capture_cuda_graph(
        self, forward_batch: "ForwardBatch"
    ) -> None:
        # Match FlashAttentionMultiStepBackend.init_forward_metadata_capture_cuda_graph:
        # call the per-step full-attn capture with explicit args.
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        assert forward_batch.spec_info is not None
        if self._triton_multistep is not None:
            self._triton_multistep.init_forward_metadata_capture_cuda_graph(
                forward_batch
            )
        else:
            for hb in self.attn_backends:
                hb.full_attn_backend.init_forward_metadata_capture_cuda_graph(
                    forward_batch.batch_size,
                    forward_batch.batch_size * self.topk,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    encoder_lens=forward_batch.encoder_lens,
                    forward_mode=ForwardMode.DECODE,
                    spec_info=forward_batch.spec_info,
                )
        if self.attn_backends:
            self.attn_backends[0].linear_attn_backend.init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: "ForwardBatch", bs: int
    ) -> None:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        assert forward_batch.spec_info is not None
        if self._triton_multistep is not None:
            self._triton_multistep.init_forward_metadata_replay_cuda_graph(
                forward_batch, bs
            )
        else:
            for hb in self.attn_backends:
                hb.full_attn_backend.init_forward_metadata_replay_cuda_graph(
                    bs,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    encoder_lens=forward_batch.encoder_lens,
                    forward_mode=ForwardMode.DECODE,
                    spec_info=forward_batch.spec_info,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                    out_cache_loc=forward_batch.out_cache_loc,
                )
        if self.attn_backends:
            self.attn_backends[0].linear_attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )
