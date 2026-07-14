"""One-CUDA-graph capture of the SMC draft phase — and optionally the full
decode cycle (draft + TARGET_VERIFY + weight diff + bonus).

Today the draft phase issues, per decode cycle: gamma+1 separate decode-graph
replays (each with its own replay_prepare) interleaved with eager sampling —
~0.74 ms of host dispatch per draft step that profiles as ~25% GPU idle at
bs=32 (see issue #14).  ``SMCDraftPhaseGraphRunner`` captures the whole phase
in ONE graph per batch-size bucket:

    for s in 0..gamma:
        logits = draft_model(input_ids, positions, attn_backends[s])
        token  = argmax(logits/T + Gumbel)          # graph-safe RNG
        logp   = (logits/T)[token] - logsumexp(logits/T)   # s < gamma only
        input_ids <- token; positions += 1

``SMCFullCycleGraphRunner`` extends the capture through the rest of the
worker cycle — the TARGET_VERIFY forward on the score model, the fused
score-logprob extraction, the per-position weight diff (alpha * score -
draft), and the Gumbel-max bonus draw — so one replay covers everything the
worker does between "batch prepared" and "GenerationBatchResult tensors
ready".  SMC can do this where EAGLE can't because batch composition is
static within a cycle: the draft loop and the verify pass see the same bs.

Design notes
------------
* Mirrors ``EAGLEDraftCudaGraphRunner`` (same ``CudaGraphRunner.capture``
  driver, same multi-step attention-backend capture/replay API).  SMC is the
  easy case: topk=1 chain drafting, static batch composition within a cycle,
  per-step seq_lens affine in the step index.
* Sampling runs in-graph.  ``torch.rand_like`` under graph capture uses the
  graph-safe Philox state, so every replay draws fresh randomness; Gumbel-max
  (#12) is what makes the sampler capturable — ``torch.multinomial`` is not.
* The multi-step backend's graph kv-indices buffer is allocated HERE, not via
  ``TritonMultiStepDraftBackend.init_cuda_graph_state``, because upstream
  sizes it as ``(steps, max_num_tokens * model_context_len)`` int64 — at 128k
  context that is unusable.  We size it by ``SMC_DRAFT_GRAPH_MAX_CONTEXT``
  (default 8192) and ``SMC_DRAFT_GRAPH_MAX_BS`` (default 32) instead, and
  ``can_run`` falls back to the per-step path whenever the live batch exceeds
  either cap.  6 steps x 32 bs x 8192 ctx x 8 B ~= 12.6 GB.
* The full-cycle runner REUSES the target attention backend's existing
  cuda-graph metadata state (allocated by the target's own graph runner) —
  calling ``init_cuda_graph_state`` again would rebind the buffers and orphan
  the target's captured verify graphs (see SMCDraftHeadGraphRunner's hazard
  note).  Sharing is safe: every replay path writes its metadata immediately
  before its own launch, and launches are sequential on one stream.
* Padding: unused rows write their KV to page 0 (the allocator's reserved
  dummy slot) and sample garbage that is sliced off, same as every other
  graph runner in sglang.
"""

from __future__ import annotations

import bisect
import logging
import os
import copy
from typing import TYPE_CHECKING

import torch

from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    _default_make_graph_key,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from smcsd.common.verify import SMCVerifyInput
from smcsd.core.info import SMCDecodeContext, SMCDraftInput

if TYPE_CHECKING:
    from smcsd.core.worker import SMCWorker

logger = logging.getLogger(__name__)


class SMCDraftPhaseGraphRunner:
    def __init__(self, smc_worker: "SMCWorker"):
        self.smc_worker = smc_worker
        self.model_runner = model_runner = smc_worker.draft_runner
        self.gamma = smc_worker.gamma
        self.num_steps = self.gamma + 1
        self.temperature = float(smc_worker.smc_draft_temperature)
        assert self.temperature > 0, "phase graph requires stochastic sampling"
        self.draft_attn_backend = smc_worker.draft_attn_backend

        self.graphs = {}
        self.output_buffers = {}
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()
        self.num_tokens_per_bs = 1

        # Batch sizes: reuse the server's capture list, capped by our own
        # memory-bound limit (see module docstring).
        graph_max_bs = int(os.environ.get("SMC_DRAFT_GRAPH_MAX_BS", "32"))
        capture_bs, _ = get_batch_sizes_to_capture(model_runner)
        self.capture_bs = [bs for bs in capture_bs if bs <= graph_max_bs]
        self.compile_bs = []  # no torch.compile interplay
        if not self.capture_bs:
            raise ValueError(
                f"SMC_DRAFT_GRAPH_MAX_BS={graph_max_bs} excludes every "
                f"capturable batch size {capture_bs[:5]}..."
            )
        self.max_bs = max(self.capture_bs)

        # Per-row context bound for the graph path (kv-indices buffer rows
        # must hold seq_len entries per request).
        self.max_context = min(
            int(os.environ.get("SMC_DRAFT_GRAPH_MAX_CONTEXT", "8192")),
            model_runner.model_config.context_len,
        )

        # ── Multi-step attention graph state (our sizing, see docstring) ──
        backend = self.draft_attn_backend
        device = model_runner.device
        from smcsd.core.hybrid_multistep_backend import (
            HybridLinearAttnMultiStepBackend,
        )

        if isinstance(backend, HybridLinearAttnMultiStepBackend):
            # Hybrid (Mamba/GDN) draft: the multi-step backend manages its own
            # per-step full-attention + shared linear-attention (recurrent) graph
            # state internally — no manual triton kv-indices buffers.  It serves
            # DECODE steps only (one token per sequence per step), so
            # max_num_tokens == max_bs.  The deferred runner's 2-token verify
            # HEAD does NOT run on this backend: it gets a dedicated
            # linear-backend instance with verify-layout (step-2) graph state —
            # see SMCDeferredCycleGraphRunner._init_extra_state.
            backend.init_cuda_graph_state(self.max_bs, self.max_bs)
            self.seq_len_fill_value = backend.attn_backends[
                0
            ].get_cuda_graph_seq_len_fill_value()
        else:
            # Triton multi-step draft.  The kernel grid in common_template covers
            # speculative_num_steps (= gamma+2) rows even though only gamma+1 step
            # backends exist.
            n_rows = backend.speculative_num_steps
            backend.cuda_graph_kv_indices = torch.zeros(
                (n_rows, self.max_bs * self.max_context),
                dtype=torch.int64,
                device=device,
            )
            backend.cuda_graph_num_kv_splits = torch.full(
                (self.max_bs,),
                backend.attn_backends[0].max_kv_splits,
                dtype=torch.int32,
                device=device,
            )
            for i, step_backend in enumerate(backend.attn_backends):
                step_backend.init_cuda_graph_state(
                    self.max_bs,
                    self.max_bs,
                    kv_indices_buf=backend.cuda_graph_kv_indices[i],
                    cuda_graph_num_kv_splits_buf=backend.cuda_graph_num_kv_splits,
                )
            self.seq_len_fill_value = backend.attn_backends[
                0
            ].get_cuda_graph_seq_len_fill_value()

        # Hybrid (Mamba/GDN) backends refresh replay metadata EAGERLY in
        # replay(): their linear-backend updates are not validated under
        # graph capture.  Triton backends capture the refresh in-graph
        # (_metadata_in_graph) — the validated fast path.  The full-cycle
        # subclass extends this flag for hybrid targets.
        self._eager_replay_metadata = isinstance(
            backend, HybridLinearAttnMultiStepBackend
        )

        # ── Graph input/output buffers ──
        with torch.device(device):
            self.input_ids = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64
            )
            # (gamma+1, max_bs): step-major cache locations.
            self.out_cache_loc = torch.zeros(
                (self.num_steps, self.max_bs), dtype=torch.int64
            )
            # Outputs: [x0, d_0..d_gamma] and per-position draft logprobs.
            self.tokens_out = torch.zeros(
                (self.max_bs, self.num_steps + 1), dtype=torch.int64
            )
            self.logprobs_out = torch.zeros(
                (self.max_bs, self.gamma), dtype=torch.float32
            )
            # MRoPE draft (Qwen3.5/VL): the model overrides the passed
            # `positions` with forward_batch.mrope_positions, so the captured
            # draft forward needs a persistent (3, bs) mrope buffer kept in sync
            # with `positions` each step (text tokens -> all 3 rows equal).
            self.mrope_positions = torch.zeros(
                (3, self.max_bs), dtype=torch.int64
            )
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64
        )
        self._draft_is_mrope = bool(
            getattr(self.model_runner.model, "is_mrope_enabled", False)
        )

        # Fused-sampler RNG state: one Philox seed per cycle, bumped by a
        # captured add_ at the start of the in-graph draft phase, so every
        # replay draws fresh noise (deterministic from random_seed).  Each
        # in-graph sampling launch gets a disjoint counter range via
        # row_offset = step * max_bs.  Kill switch: SMC_FUSED_SAMPLING=0
        # falls back to the torch Gumbel chain.
        self.use_fused_sampling = bool(
            int(os.environ.get("SMC_FUSED_SAMPLING", "1"))
        )
        with torch.device(model_runner.device):
            self.sample_seed = torch.randint(
                0, 2**31 - 1, (1,), dtype=torch.int64
            )
            self._steps_arange = torch.arange(self.num_steps)

        # ForwardBatch per bucket, kept so replay-time metadata regeneration
        # reads the very buffers the graph was captured against.
        self.fbs = {}

        self._init_extra_state()

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture SMC draft phase graph failed: {e}\n"
                f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )
        logger.info(
            "%s: captured %d buckets (max_bs=%d, max_context=%d, steps=%d)",
            type(self).__name__,
            len(self.capture_bs), self.max_bs, self.max_context, self.num_steps,
        )

    def _init_extra_state(self):
        """Hook for subclasses to allocate additional buffers."""

    # ── Capture ──

    def capture(self):
        CudaGraphRunner.capture(self)

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _setup_draft_capture(self, bs: int):
        """Slice buffers, build the draft ForwardBatch, and write the
        per-step attention metadata for capture.  Returns (fb, slices)."""
        input_ids = self.input_ids[:bs]
        req_pool_indices = self.req_pool_indices[:bs]
        positions = self.positions[:bs]
        seq_lens = self.seq_lens[:bs]
        seq_lens_cpu = self.seq_lens_cpu[:bs]
        out_cache_loc_steps = [
            self.out_cache_loc[s, :bs] for s in range(self.num_steps)
        ]

        spec_info = SMCDraftInput(
            verified_id=input_ids,
            num_tokens_per_req=self.num_steps,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(seq_lens.sum().item()),
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc_steps[0],
            return_logprob=False,
            positions=positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        if self._draft_is_mrope:
            # MRoPE model reads forward_batch.mrope_positions (not `positions`).
            # Point it at the persistent buffer; _draft_steps_in_graph keeps it
            # in sync with `positions` each step.
            fb.mrope_positions = self.mrope_positions[:, :bs]
        self.fbs[bs] = fb
        self.draft_attn_backend.init_forward_metadata_capture_cuda_graph(fb)
        return fb, input_ids, positions, out_cache_loc_steps

    def _metadata_in_graph(self, bs: int):
        """Attention-metadata refresh, captured INSIDE the cycle graph.

        Every op here is a device op over staged input buffers (seq_lens /
        positions / req_pool_indices) or persistent pool tensors
        (req_to_token, the backends' cuda-graph indptr/indices buffers), so
        recording it in the graph reproduces today's eager pre-launch
        metadata calls exactly — while removing ~15-30 host-dispatched ops
        from the replay critical path.  Must run before the model forwards
        that consume the metadata; the draft loop's in-graph positions
        increments happen after, exactly like the eager ordering.
        """
        if not self._eager_replay_metadata:
            self.draft_attn_backend.init_forward_metadata_replay_cuda_graph(
                self.fbs[bs], bs
            )

    def _sample_step_in_graph(self, logits, step: int, need_logp: bool):
        """One in-graph draft draw: fused kernel or the torch Gumbel chain.

        Returns (idx, logp-or-None).  ``step`` selects the fused sampler's
        Philox counter range (disjoint per launch within a cycle).
        """
        if self.use_fused_sampling:
            from smcsd.core.kernels.fused_sampling import fused_gumbel_sample

            idx, logp, _ = fused_gumbel_sample(
                logits,
                self.temperature,
                self.sample_seed,
                need_logp=need_logp,
                row_offset=step * self.max_bs,
            )
            return idx, logp
        tiny = torch.finfo(torch.float32).tiny
        scaled = logits / self.temperature
        gumbel = -torch.log(
            -torch.log(torch.rand_like(scaled).clamp_min_(tiny))
        )
        idx = torch.argmax(scaled + gumbel, dim=-1)
        if not need_logp:
            return idx, None
        chosen = scaled.gather(1, idx.unsqueeze(1)).squeeze(1)
        return idx, chosen - torch.logsumexp(scaled, dim=-1)

    def _draft_steps_in_graph(self, bs, forward, fb, input_ids, positions,
                              out_cache_loc_steps):
        """The captured draft loop: gamma+1 forwards + Gumbel sampling."""
        backends = self.draft_attn_backend.attn_backends
        tokens_out = self.tokens_out[:bs]
        logprobs_out = self.logprobs_out[:bs]
        if self.use_fused_sampling:
            self.sample_seed.add_(1)  # captured: fresh noise every replay
        for s in range(self.num_steps):
            fb.attn_backend = backends[s]
            fb.out_cache_loc = out_cache_loc_steps[s]
            if self._draft_is_mrope:
                # Keep mrope_positions in sync with the current `positions`
                # (text tokens: all 3 rows share the linear position).  copy_
                # broadcasts (1, bs) -> (3, bs); in-graph and fixed-shape.
                self.mrope_positions[:, :bs].copy_(positions.unsqueeze(0))
            # `forward` is the (patched) model.forward — returns a
            # LogitsProcessorOutput directly.
            logits = forward(input_ids, positions, fb).next_token_logits
            idx, logp = self._sample_step_in_graph(
                logits, s, need_logp=s < self.gamma
            )
            tokens_out[:, s + 1] = idx
            if s < self.gamma:
                logprobs_out[:, s] = logp
            input_ids.copy_(idx)
            positions.add_(1)
        return tokens_out, logprobs_out

    def capture_one_batch_size(self, num_seqs: int, forward, stream_idx: int = 0):
        graph = self._create_graph()
        stream = self.stream
        bs = num_seqs

        fb, input_ids, positions, ocl_steps = self._setup_draft_capture(bs)

        def run_once():
            set_is_extend_in_batch(False)
            self._metadata_in_graph(bs)
            return self._draft_steps_in_graph(
                bs, forward, fb, input_ids, positions, ocl_steps
            )

        self.deepep_adapter.capture(is_extend_in_batch=False)
        self._capture_init(run_once)
        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )
        set_global_graph_memory_pool(graph.pool())
        return graph, out

    # ── Replay ──

    def can_run(self, raw_bs: int, ctx: SMCDecodeContext) -> bool:
        if raw_bs > self.max_bs:
            return False
        # Per-step kv length peaks at orig_seq_len + gamma + 1; the kv-indices
        # rows are sized to max_context entries per request.
        max_len = int(ctx.orig_seq_lens_cpu.max().item()) + self.num_steps
        return max_len <= self.max_context

    def _stage_replay_inputs(
        self,
        verified_id: torch.Tensor,
        cache_locs: torch.Tensor,
        ctx: SMCDecodeContext,
        req_pool_indices: torch.Tensor,
    ):
        """Copy live inputs into the captured buffers; returns (raw_bs, bs)."""
        raw_bs = verified_id.shape[0]
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]

        if bs != raw_bs:
            self.seq_lens.fill_(self.seq_len_fill_value)
            self.seq_lens_cpu.fill_(self.seq_len_fill_value)
            self.positions.zero_()
            self.req_pool_indices.zero_()
            self.out_cache_loc.zero_()  # padded rows scribble page 0
            self.input_ids.zero_()

        self.input_ids[:raw_bs].copy_(verified_id)
        self.tokens_out[:raw_bs, 0].copy_(verified_id)
        self.req_pool_indices[:raw_bs].copy_(req_pool_indices)
        self.positions[:raw_bs].copy_(ctx.orig_seq_lens)
        self.seq_lens[:raw_bs].copy_(ctx.orig_seq_lens)
        self.seq_lens_cpu[:raw_bs].copy_(ctx.orig_seq_lens_cpu)
        # (bs, gamma+1) -> step-major buffer.
        self.out_cache_loc[:, :raw_bs].copy_(cache_locs.t())
        return raw_bs, bs

    def replay(
        self,
        verified_id: torch.Tensor,
        cache_locs: torch.Tensor,
        ctx: SMCDecodeContext,
        req_pool_indices: torch.Tensor,
    ):
        """Returns (tokens_out[:raw_bs], logprobs_out[:raw_bs]) where
        tokens_out columns are [x0, d_0, ..., d_gamma]."""
        raw_bs, bs = self._stage_replay_inputs(
            verified_id, cache_locs, ctx, req_pool_indices
        )
        # Attention metadata is captured in-graph for triton backends
        # (_metadata_in_graph); hybrid backends refresh eagerly here.
        if self._eager_replay_metadata:
            self.draft_attn_backend.init_forward_metadata_replay_cuda_graph(
                self.fbs[bs], bs
            )
        # capture() stores keys via _default_make_graph_key(bs, None, None),
        # which is the plain bs int.
        self.graphs[_default_make_graph_key(bs)].replay()
        return self.tokens_out[:raw_bs], self.logprobs_out[:raw_bs]


class SMCFullCycleGraphRunner(SMCDraftPhaseGraphRunner):
    """Draft phase + TARGET_VERIFY + weight diff + bonus, one graph.

    Replay returns every tensor the worker needs for its
    ``GenerationBatchResult``:

        tokens_out        (bs, gamma+2)  [x0, d_0..d_gamma]
        logprobs_out      (bs, gamma)    draft logprobs (diagnostic)
        logprob_diff_out  (bs, gamma)    alpha * score_logp - draft_logp
        bonus_out         (bs,)          Gumbel draw from p_T^alpha
        bonus_logz_out    (bs,)          bonus normalizer log Z (0 at alpha=1)
        next_tokens_out   (bs, gamma+1)  [d_0..d_{gamma-1}, bonus]

    The verify forward runs on the TARGET model inside the same graph —
    legal because a CUDA graph just records kernels, regardless of which
    module launches them.  Its attention metadata uses the target backend's
    EXISTING graph-state buffers (shared with the target's own verify
    graphs; see module docstring).
    """

    def _init_extra_state(self):
        worker = self.smc_worker
        self.target_runner = worker.score_runner
        self.target_backend = self.target_runner.attn_backend
        self.target_temperature = float(worker.smc_target_temperature)
        self.alpha = float(worker.smc_power_alpha)

        n_verify_tokens = self.max_bs * self.num_steps
        with torch.device(self.model_runner.device):
            self.verify_input_ids = torch.zeros(
                (n_verify_tokens,), dtype=torch.int64
            )
            self.verify_positions = torch.zeros(
                (n_verify_tokens,), dtype=torch.int64
            )
            self.verify_out_cache_loc = torch.zeros(
                (n_verify_tokens,), dtype=torch.int64
            )
            self.logprob_diff_out = torch.zeros(
                (self.max_bs, self.gamma), dtype=torch.float32
            )
            self.bonus_out = torch.zeros((self.max_bs,), dtype=torch.int64)
            # Per-particle bonus normalizer log Z (see SMCDraftInput.bonus_logz).
            self.bonus_logz_out = torch.zeros(
                (self.max_bs,), dtype=torch.float32
            )
            self.next_tokens_out = torch.zeros(
                (self.max_bs, self.num_steps), dtype=torch.int64
            )
            # MRoPE target (Qwen3.5/VL): the verify forward reads
            # forward_batch.mrope_positions; keep a persistent (3, n_verify_tokens)
            # buffer synced with verify_positions.
            self.verify_mrope_positions = torch.zeros(
                (3, n_verify_tokens), dtype=torch.int64
            )
            # Padded token buffer for the fused score-logprob pass (the
            # bonus row scores a dummy 0 token and is sliced off).
            self.score_tok_buf = torch.zeros(
                (self.max_bs, self.num_steps), dtype=torch.int64
            )
        self.verify_fbs = {}
        self._target_is_mrope = bool(
            getattr(self.target_runner.model, "is_mrope_enabled", False)
        )

        # Hybrid (Mamba/GDN) target: the post-verify recurrent-state commit must
        # run INSIDE the captured cycle (it was the sole SMC-side reason hybrid
        # targets were excluded from the cycle graph).  For SMC every drafted
        # token is accepted, so accepted_steps is the constant gamma — a
        # persistent buffer whose stable address the graph captures once and the
        # scatter reads on every replay.  mamba_track_indices is None for SMC
        # (no prefix-cache tracking), so the commit is just the fused
        # gather-scatter kernel: fixed-shape and capturable.  replay() refreshes
        # the target verify metadata (mamba_cache_indices) via
        # init_forward_metadata_replay_cuda_graph before graph.replay(), so the
        # captured scatter targets the current step's mamba slots.
        self._hybrid_commit = (
            worker.score_runner.hybrid_gdn_config is not None
            and hasattr(self.target_backend, "update_mamba_state_after_mtp_verify")
        )
        if self._hybrid_commit:
            with torch.device(self.model_runner.device):
                self._accepted_steps = torch.full(
                    (self.max_bs,), self.gamma, dtype=torch.int64
                )
        # Hybrid target metadata (mamba_cache_indices etc.) is refreshed
        # eagerly in replay(), like hybrid draft metadata.
        self._eager_replay_metadata = (
            self._eager_replay_metadata or self._hybrid_commit
        )

    def _setup_verify_capture(self, bs: int):
        n_tokens = bs * self.num_steps
        verify_input_ids = self.verify_input_ids[:n_tokens]
        verify_positions = self.verify_positions[:n_tokens]
        verify_ocl = self.verify_out_cache_loc[:n_tokens]
        seq_lens = self.seq_lens[:bs]
        seq_lens_cpu = self.seq_lens_cpu[:bs]
        req_pool_indices = self.req_pool_indices[:bs]

        verify_spec = SMCVerifyInput(
            draft_token_num=self.num_steps,
            positions=verify_positions,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=int(seq_lens.sum().item()),
            seq_lens_cpu=seq_lens_cpu,
            num_tokens_per_req=self.num_steps,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=verify_input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(seq_lens.sum().item()),
            req_to_token_pool=self.target_runner.req_to_token_pool,
            token_to_kv_pool=self.target_runner.token_to_kv_pool,
            attn_backend=self.target_backend,
            out_cache_loc=verify_ocl,
            return_logprob=False,
            positions=verify_positions,
            spec_algorithm=self.target_runner.spec_algorithm,
            spec_info=verify_spec,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        if self._target_is_mrope:
            fb.mrope_positions = self.verify_mrope_positions[:, :n_tokens]
        verify_spec.populate_linear_verify_metadata(fb)
        self.verify_fbs[bs] = (fb, verify_spec)
        # NOTE: reuses the target backend's existing cuda-graph metadata
        # buffers (allocated by the target's own graph runner) — do NOT call
        # init_cuda_graph_state here, it would orphan the target's graphs.
        self.target_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            n_tokens,
            req_pool_indices,
            seq_lens,
            None,
            ForwardMode.TARGET_VERIFY,
            verify_spec,
        )
        return fb, verify_input_ids, verify_positions

    def _metadata_in_graph(self, bs: int):
        """Draft multistep metadata + verify staging/metadata, in-graph.

        The verify positions / out-cache-locs are pure functions of the
        staged seq_lens and step-major out_cache_loc buffers, so they are
        derived here instead of being staged eagerly per replay.
        """
        super()._metadata_in_graph(bs)

        n_tokens = bs * self.num_steps
        # positions[r, e] = S_r + e
        self.verify_positions[:n_tokens].view(bs, self.num_steps).copy_(
            self.seq_lens[:bs].unsqueeze(1) + self._steps_arange
        )
        # step-major staged buffer -> request-major verify layout
        self.verify_out_cache_loc[:n_tokens].view(bs, self.num_steps).copy_(
            self.out_cache_loc[:, :bs].t()
        )
        # CROSS-REPO SEAM: seq_lens_sum=0 / seq_lens_cpu=None are
        # placeholders — the vendored triton backend's linear-verify replay
        # branch reads neither (verified byte-identical vs the eager path).
        # If a future submodule bump makes that branch read either host
        # value, this capture records garbage: None fails loudly, but the
        # 0 would be silent.  Re-verify on submodule bumps (an assert in
        # the vendored branch is queued for the next bump).
        if not self._eager_replay_metadata:
            _, verify_spec = self.verify_fbs[bs]
            self.target_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                self.req_pool_indices[:bs],
                self.seq_lens[:bs],
                0,      # seq_lens_sum: placeholder, see seam note above
                None,
                ForwardMode.TARGET_VERIFY,
                verify_spec,
                None,   # seq_lens_cpu: placeholder, see seam note above
            )

    def _verify_in_graph(self, bs, fb, verify_input_ids, verify_positions):
        gamma = self.gamma
        tokens_out = self.tokens_out[:bs]
        logprobs_out = self.logprobs_out[:bs]
        logprob_diff_out = self.logprob_diff_out[:bs]
        bonus_out = self.bonus_out[:bs]
        bonus_logz_out = self.bonus_logz_out[:bs]
        next_tokens_out = self.next_tokens_out[:bs]
        tiny = torch.finfo(torch.float32).tiny

        # Score input = [x0, d_0..d_{gamma-1}] per row, request-major.
        verify_input_ids.view(bs, self.num_steps).copy_(
            tokens_out[:, : self.num_steps]
        )
        if self._target_is_mrope:
            n_tokens = bs * self.num_steps
            self.verify_mrope_positions[:, :n_tokens].copy_(
                verify_positions.unsqueeze(0)
            )
        logits = self.target_runner.model.forward(
            verify_input_ids, verify_positions, fb
        ).next_token_logits

        # Hybrid target: commit the accepted recurrent state IN-GRAPH, right
        # after the verify forward that produced the intermediate states.
        # Reuses the eager commit (same guards / track-index handling) with the
        # persistent constant accepted_steps buffer; captured once, replayed
        # every step against the metadata replay() refreshes.
        if self._hybrid_commit:
            self.smc_worker._commit_target_mamba_state_after_verify(
                fb, self._accepted_steps[:bs]
            )

        logits3 = logits.view(bs, self.num_steps, -1)

        if self.use_fused_sampling:
            from smcsd.core.kernels.fused_sampling import (
                fused_chosen_logprob,
                fused_gumbel_sample,
            )

            # Score logprobs: one pass over ALL gamma+1 rows (contiguous),
            # the bonus row scored against a dummy token and sliced off —
            # cheaper than materializing a non-contiguous (bs*gamma, V) view.
            score_toks = self.score_tok_buf[:bs]
            score_toks[:, :gamma].copy_(tokens_out[:, 1 : gamma + 1])
            score_lp = fused_chosen_logprob(
                logits, score_toks.reshape(-1), self.target_temperature
            ).view(bs, self.num_steps)[:, :gamma]
            logprob_diff_out.copy_(self.alpha * score_lp - logprobs_out)

            # Bonus from p_T^alpha (strided row view: fixed offset + row
            # stride, inner-contiguous — supported by the kernel).  logz is
            # the bonus's incremental importance weight; exact 0 at alpha=1.
            b_idx, _, b_logz = fused_gumbel_sample(
                logits3[:, -1, :],
                self.target_temperature,
                self.sample_seed,
                alpha=self.alpha,
                need_logp=False,
                need_logz=True,
                row_offset=(self.num_steps + 1) * self.max_bs,
            )
            bonus_out.copy_(b_idx)
            bonus_logz_out.copy_(b_logz)
        else:
            # Torch reference chain.
            verify_scaled = logits3[:, :gamma, :] / self.target_temperature
            chosen = verify_scaled.gather(
                2, tokens_out[:, 1 : gamma + 1].unsqueeze(2)
            ).squeeze(2)
            score_logprobs = chosen - torch.logsumexp(verify_scaled, dim=-1)
            logprob_diff_out.copy_(self.alpha * score_logprobs - logprobs_out)

            # Bonus from the same p_T^alpha tempered-power target, Gumbel-max.
            bonus_base = logits3[:, -1, :] / self.target_temperature
            bonus_scaled = self.alpha * bonus_base
            bonus_gumbel = -torch.log(
                -torch.log(torch.rand_like(bonus_scaled).clamp_min_(tiny))
            )
            bonus_out.copy_(torch.argmax(bonus_scaled + bonus_gumbel, dim=-1))
            # Bonus normalizer log Z = logsumexp(alpha*ℓ/T) -
            # alpha*logsumexp(ℓ/T): the bonus's incremental importance weight
            # under the joint-power target (drawn from the locally normalized
            # p_T^alpha/Z).  0 at alpha=1.  Accumulated in write_back_gpu,
            # gated by the EOS/finish logic.
            bonus_logz_out.copy_(
                torch.logsumexp(bonus_scaled, dim=-1)
                - self.alpha * torch.logsumexp(bonus_base, dim=-1)
            )

        next_tokens_out[:, :gamma].copy_(tokens_out[:, 1 : gamma + 1])
        next_tokens_out[:, gamma].copy_(bonus_out)
        return (
            tokens_out,
            logprobs_out,
            logprob_diff_out,
            bonus_out,
            bonus_logz_out,
            next_tokens_out,
        )

    def capture_one_batch_size(self, num_seqs: int, forward, stream_idx: int = 0):
        graph = self._create_graph()
        stream = self.stream
        bs = num_seqs

        fb, input_ids, positions, ocl_steps = self._setup_draft_capture(bs)
        verify_fb, verify_input_ids, verify_positions = (
            self._setup_verify_capture(bs)
        )

        def run_once():
            set_is_extend_in_batch(False)
            self._metadata_in_graph(bs)
            self._draft_steps_in_graph(
                bs, forward, fb, input_ids, positions, ocl_steps
            )
            return self._verify_in_graph(
                bs, verify_fb, verify_input_ids, verify_positions
            )

        self.deepep_adapter.capture(is_extend_in_batch=False)
        self._capture_init(run_once)
        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )
        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def replay(
        self,
        verified_id: torch.Tensor,
        cache_locs: torch.Tensor,
        ctx: SMCDecodeContext,
        req_pool_indices: torch.Tensor,
    ):
        raw_bs, bs = self._stage_replay_inputs(
            verified_id, cache_locs, ctx, req_pool_indices
        )
        # Verify staging is captured in-graph; attention metadata is
        # in-graph for triton backends, eager here for hybrid ones.
        if self._eager_replay_metadata:
            self.draft_attn_backend.init_forward_metadata_replay_cuda_graph(
                self.fbs[bs], bs
            )
            _, verify_spec = self.verify_fbs[bs]
            seq_lens_sum = int(self.seq_lens_cpu[:bs].sum().item())
            self.target_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                self.req_pool_indices[:bs],
                self.seq_lens[:bs],
                seq_lens_sum,
                None,
                ForwardMode.TARGET_VERIFY,
                verify_spec,
                self.seq_lens_cpu[:bs],
            )
        self.graphs[_default_make_graph_key(bs)].replay()
        return (
            self.tokens_out[:raw_bs],
            self.logprobs_out[:raw_bs],
            self.logprob_diff_out[:raw_bs],
            self.bonus_out[:raw_bs],
            self.bonus_logz_out[:raw_bs],
            self.next_tokens_out[:raw_bs],
        )


class SMCDeferredCycleGraphRunner(SMCFullCycleGraphRunner):
    """Full-cycle graph with the deferred-bonus draft schedule.

    The in-graph draft phase becomes a 2-token head ``[prev_last_draft @
    S-1, verified @ S]`` followed by gamma-1 single-token decodes — gamma
    draft forwards per cycle instead of gamma+1.  The over-draft forward
    disappears; d_{gamma-1}'s draft-KV write is deferred into the NEXT
    cycle's head, exactly like the eager ``SMC_DEFER_BONUS`` path
    (``SMCWorker._draft_ar_deferred``), including the first-step property
    that ``prev`` is the last committed prompt token whose S-1 write
    rewrites the prefill's draft KV byte-identically.

    The head runs as a TARGET_VERIFY-mode forward on the DRAFT model inside
    the same capture, on the draft's PRIMARY attention backend — whose
    verify-block-size global the worker pins to 2 under SMC_DEFER_BONUS —
    reusing that backend's existing cuda-graph metadata buffers.  Same
    sharing argument as SMCDraftHeadGraphRunner and the target side of the
    cycle capture: every replay path writes its own metadata immediately
    before its own launch, and launches are sequential on one stream.

    ``tokens_out`` columns are ``[x0, d_0..d_{gamma-1}]`` (the final
    over-draft column stays zero); the worker reads
    ``prev_last_draft_id = tokens_out[:, gamma]`` identically either way.
    The verify pass, weight diff, and bonus draw are inherited unchanged —
    the verify input ``tokens_out[:, :gamma+1]`` carries the same token
    layout as the legacy schedule.
    """

    deferred = True

    def _init_extra_state(self):
        super()._init_extra_state()
        self.draft_primary_backend = self.model_runner.attn_backend
        n_head_tokens = 2 * self.max_bs
        with torch.device(self.model_runner.device):
            self.prev_input = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.head_input_ids = torch.zeros((n_head_tokens,), dtype=torch.int64)
            self.head_positions = torch.zeros((n_head_tokens,), dtype=torch.int64)
            self.head_out_cache_loc = torch.zeros(
                (n_head_tokens,), dtype=torch.int64
            )
            self.head_seq_lens = torch.zeros((self.max_bs,), dtype=torch.int64)
            # MRoPE draft: the 2-token head window needs its own (3, 2*bs)
            # mrope buffer (the singles reuse the base (3, bs) self.mrope_positions).
            self.head_mrope_positions = torch.zeros(
                (3, n_head_tokens), dtype=torch.int64
            )
        self.head_seq_lens_cpu = torch.zeros((self.max_bs,), dtype=torch.int64)
        self.head_fbs = {}
        # Hybrid (Mamba/GDN) draft: after the verify-style head, commit the
        # index-1 (verified / S) recurrent state IN-GRAPH so the gamma-1 singles
        # continue from S.  Constant accepted_steps=1 (persistent buffer).
        self._draft_head_commit = hasattr(
            self.draft_primary_backend, "update_mamba_state_after_mtp_verify"
        ) and (
            getattr(self.model_runner, "hybrid_gdn_config", None) is not None
        )
        if self._draft_head_commit:
            with torch.device(self.model_runner.device):
                self._head_accepted_steps = torch.ones(
                    self.max_bs, dtype=torch.int64
                )

        # Head attention backend.  For a hybrid (Mamba/GDN) draft the head gets
        # a DEDICATED linear-backend instance: the shared GDN backend has one
        # set of per-bs cuda-graph metadata buffers (query_start_loc_list /
        # state_indices_list), and inside ONE captured cycle the head needs
        # them in verify layout (2-token windows) while the gamma-1 singles
        # need decode layout (1-token windows) — one buffer can't hold both,
        # which is what previously limited deferred+cycle to gamma==1.
        # Recurrent STATE lives in the req_to_token_pool, not on the backend,
        # so a second backend instance over the same pool is safe (the
        # multi-step wrapper shares its linear backend for the same reason).
        # copy.copy carries over the isolation-time pool re-pointing
        # (req_to_token_pool / conv_states_shape /
        # verify_intermediate_state_indices); only the graph-state lists and
        # forward_metadata are given fresh instances, then graph state is
        # built with max_num_tokens=2*max_bs so draft_token_num == 2 and the
        # verify query-start-loc cache is a step-2 arange.  The FA3
        # full-attention side keeps per-mode metadata dicts, so it is shared.
        if self._draft_head_commit:
            from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
                HybridLinearAttnBackend,
            )

            shared_lin = self.draft_primary_backend.linear_attn_backend
            head_lin = copy.copy(shared_lin)
            head_lin.forward_metadata = None
            head_lin.state_indices_list = []
            head_lin.query_start_loc_list = []
            head_lin.retrieve_next_token_list = []
            head_lin.retrieve_next_sibling_list = []
            head_lin.retrieve_parent_token_list = []
            head_lin.init_cuda_graph_state(self.max_bs, 2 * self.max_bs)
            self.head_backend = HybridLinearAttnBackend(
                self.draft_primary_backend.full_attn_backend,
                head_lin,
                self.draft_primary_backend.full_attn_layers,
            )
        else:
            # Triton/FA3 drafts: the primary backend serves the head directly
            # (its verify-block-size global is pinned to 2 by the worker).
            self.head_backend = self.draft_primary_backend

    def _setup_head_capture(self, bs: int):
        """Build the head's ForwardBatch (TARGET_VERIFY on the draft model,
        2 tokens per request) and write its capture metadata."""
        n_tokens = 2 * bs
        head_input_ids = self.head_input_ids[:n_tokens]
        head_positions = self.head_positions[:n_tokens]
        head_ocl = self.head_out_cache_loc[:n_tokens]
        head_seq_lens = self.head_seq_lens[:bs]
        head_seq_lens_cpu = self.head_seq_lens_cpu[:bs]
        req_pool_indices = self.req_pool_indices[:bs]

        # Head prefix is S-1 (the deferred slot is rewritten, not appended).
        head_seq_lens.copy_(self.seq_lens[:bs] - 1)
        head_seq_lens_cpu.copy_(self.seq_lens_cpu[:bs] - 1)

        head_spec = SMCVerifyInput(
            draft_token_num=2,
            positions=head_positions,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=int(head_seq_lens.sum().item()),
            seq_lens_cpu=head_seq_lens_cpu,
            num_tokens_per_req=2,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=head_input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=head_seq_lens,
            seq_lens_cpu=head_seq_lens_cpu,
            seq_lens_sum=int(head_seq_lens.sum().item()),
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.head_backend,
            out_cache_loc=head_ocl,
            return_logprob=False,
            positions=head_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=head_spec,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        if self._draft_is_mrope:
            fb.mrope_positions = self.head_mrope_positions[:, :n_tokens]
        head_spec.populate_linear_verify_metadata(fb)
        self.head_fbs[bs] = (fb, head_spec)
        # head_backend: for hybrid drafts a DEDICATED backend whose linear
        # graph-state was built in _init_extra_state (verify layout, step-2
        # windows), so it never conflicts with the singles' decode metadata;
        # for triton/FA3 drafts it aliases the primary backend, reusing its
        # existing buffers (do NOT call init_cuda_graph_state — rebinding
        # hazard, see module docstring).
        self.head_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            n_tokens,
            req_pool_indices,
            head_seq_lens,
            None,
            ForwardMode.TARGET_VERIFY,
            head_spec,
        )
        return fb

    def _metadata_in_graph(self, bs: int):
        """Head-prefix lengths + head metadata, then the parent's verify/
        draft metadata — all captured in-graph.  The deferred S-1 cache
        slot (head_out_cache_loc column 0) is the one input still staged
        eagerly in replay(): it needs pad-row-safe zeros, and raw_bs is a
        host-side value."""
        torch.sub(self.seq_lens[:bs], 1, out=self.head_seq_lens[:bs])
        if not self._eager_replay_metadata:
            _, head_spec = self.head_fbs[bs]
            self.head_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                self.req_pool_indices[:bs],
                self.head_seq_lens[:bs],
                0,      # seq_lens_sum: unused by the linear-verify branch
                None,
                ForwardMode.TARGET_VERIFY,
                head_spec,
                None,   # seq_lens_cpu: unused by the linear-verify branch
            )
        super()._metadata_in_graph(bs)

    def _draft_steps_in_graph(self, bs, forward, fb, input_ids, positions,
                              out_cache_loc_steps):
        """Deferred captured draft phase: head + gamma-1 singles."""
        backends = self.draft_attn_backend.attn_backends
        tokens_out = self.tokens_out[:bs]
        logprobs_out = self.logprobs_out[:bs]
        if self.use_fused_sampling:
            self.sample_seed.add_(1)  # captured: fresh noise every replay

        head_fb, _ = self.head_fbs[bs]
        n_head = 2 * bs
        head_ids2 = self.head_input_ids[:n_head].view(bs, 2)
        head_pos2 = self.head_positions[:n_head].view(bs, 2)
        head_ocl2 = self.head_out_cache_loc[:n_head].view(bs, 2)
        seq_lens = self.seq_lens[:bs]

        # Head input staging — captured ops over the staged cycle buffers,
        # recomputed from live values on every replay.  Column 0 of
        # head_out_cache_loc (the deferred S-1 slot) is the one input staged
        # eagerly in replay(): it needs a live block-table read with
        # pad-row-safe zeros (page 0).
        head_ids2[:, 0] = self.prev_input[:bs]
        head_ids2[:, 1] = input_ids                  # x0 / verified_id
        head_pos2[:, 0] = seq_lens - 1
        head_pos2[:, 1] = seq_lens
        head_ocl2[:, 1] = out_cache_loc_steps[0]     # fresh slot @ S

        if self._draft_is_mrope:
            self.head_mrope_positions[:, :n_head].copy_(
                self.head_positions[:n_head].unsqueeze(0)
            )

        logits2 = forward(
            self.head_input_ids[:n_head],
            self.head_positions[:n_head],
            head_fb,
        ).next_token_logits

        # Hybrid draft: commit the S-position (index 1) recurrent state
        # IN-GRAPH, right after the head forward, so the gamma-1 singles (and
        # the next cycle) continue from S.  Runs on the DEDICATED head backend,
        # whose linear metadata is verify-layout and never clobbered by the
        # singles' decode metadata; the captured scatter reads the persistent
        # index buffers that replay() refreshes.  Constant accepted_steps=1.
        if self._draft_head_commit:
            self.head_backend.update_mamba_state_after_mtp_verify(
                accepted_steps=self._head_accepted_steps[:bs],
                mamba_track_indices=getattr(head_fb, "mamba_track_indices", None),
                mamba_steps_to_track=None,
                model=self.model_runner.model,
            )

        head_logits = logits2.view(bs, 2, -1)[:, 1, :]   # S / bonus column

        idx, logp = self._sample_step_in_graph(head_logits, 0, need_logp=True)
        tokens_out[:, 1] = idx
        logprobs_out[:, 0] = logp
        input_ids.copy_(idx)
        positions.add_(1)

        # gamma-1 singles: forward(d_{s-1}) @ S+s for s = 1..gamma-1.
        for s in range(1, self.gamma):
            fb.attn_backend = backends[s]
            fb.out_cache_loc = out_cache_loc_steps[s]
            if self._draft_is_mrope:
                self.mrope_positions[:, :bs].copy_(positions.unsqueeze(0))
            logits = forward(input_ids, positions, fb).next_token_logits
            idx, logp = self._sample_step_in_graph(logits, s, need_logp=True)
            tokens_out[:, s + 1] = idx
            logprobs_out[:, s] = logp
            input_ids.copy_(idx)
            positions.add_(1)
        return tokens_out, logprobs_out

    def capture_one_batch_size(self, num_seqs: int, forward, stream_idx: int = 0):
        graph = self._create_graph()
        stream = self.stream
        bs = num_seqs

        fb, input_ids, positions, ocl_steps = self._setup_draft_capture(bs)
        self._setup_head_capture(bs)
        verify_fb, verify_input_ids, verify_positions = (
            self._setup_verify_capture(bs)
        )

        def run_once():
            set_is_extend_in_batch(False)
            self._metadata_in_graph(bs)
            self._draft_steps_in_graph(
                bs, forward, fb, input_ids, positions, ocl_steps
            )
            return self._verify_in_graph(
                bs, verify_fb, verify_input_ids, verify_positions
            )

        self.deepep_adapter.capture(is_extend_in_batch=False)
        self._capture_init(run_once)
        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )
        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def replay(
        self,
        verified_id: torch.Tensor,
        cache_locs: torch.Tensor,
        ctx: SMCDecodeContext,
        req_pool_indices: torch.Tensor,
        prev_last_draft_id: torch.Tensor = None,
    ):
        assert prev_last_draft_id is not None, (
            "deferred cycle graph requires prev_last_draft_id "
            "(seeded at allocate_slots, carried by resample)"
        )
        raw_bs, bs = self._stage_replay_inputs(
            verified_id, cache_locs, ctx, req_pool_indices
        )
        if bs != raw_bs:
            self.prev_input.zero_()
            self.head_out_cache_loc.zero_()  # pad rows scribble page 0
        self.prev_input[:raw_bs].copy_(prev_last_draft_id)

        # Deferred S-1 slot: live block-table lookup (post-resample
        # correct), enqueued eagerly so pad rows keep page 0.  Everything
        # else — head prefix lens, head/draft/target metadata, verify
        # staging — is captured in-graph (_metadata_in_graph).
        r2t = self.model_runner.req_to_token_pool.req_to_token
        slot_sm1 = r2t[
            req_pool_indices.to(torch.int64),
            (ctx.orig_seq_lens - 1).to(torch.int64),
        ]
        self.head_out_cache_loc[: 2 * raw_bs].view(raw_bs, 2)[:, 0].copy_(
            slot_sm1
        )

        if self._eager_replay_metadata:
            # Hybrid (Mamba/GDN) backends: metadata replay-refresh runs EAGERLY
            # here (their linear-backend updates are not validated under graph
            # capture); triton backends do all of this inside the captured
            # graph via _metadata_in_graph.  Verify staging (positions /
            # cache-locs) is device-only and always captured in-graph.
            self.draft_attn_backend.init_forward_metadata_replay_cuda_graph(
                self.fbs[bs], bs
            )
            verify_fb, verify_spec = self.verify_fbs[bs]
            seq_lens_sum = int(self.seq_lens_cpu[:bs].sum().item())
            self.target_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                self.req_pool_indices[:bs],
                self.seq_lens[:bs],
                seq_lens_sum,
                None,
                ForwardMode.TARGET_VERIFY,
                verify_spec,
                self.seq_lens_cpu[:bs],
            )
            # Head metadata: for hybrid drafts this writes the DEDICATED head
            # backend's own persistent buffers (verify layout), so it cannot
            # clobber — nor be clobbered by — the singles' decode metadata
            # refresh above.  The in-graph draft commit (see
            # _draft_steps_in_graph) reads the buffers refreshed here.
            torch.sub(self.seq_lens[:bs], 1, out=self.head_seq_lens[:bs])
            self.head_seq_lens_cpu[:bs].copy_(self.seq_lens_cpu[:bs] - 1)
            head_sum = int(self.head_seq_lens_cpu[:bs].sum().item())
            _, head_spec = self.head_fbs[bs]
            self.head_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                self.req_pool_indices[:bs],
                self.head_seq_lens[:bs],
                head_sum,
                None,
                ForwardMode.TARGET_VERIFY,
                head_spec,
                self.head_seq_lens_cpu[:bs],
            )

        self.graphs[_default_make_graph_key(bs)].replay()

        return (
            self.tokens_out[:raw_bs],
            self.logprobs_out[:raw_bs],
            self.logprob_diff_out[:raw_bs],
            self.bonus_out[:raw_bs],
            self.bonus_logz_out[:raw_bs],
            self.next_tokens_out[:raw_bs],
        )
