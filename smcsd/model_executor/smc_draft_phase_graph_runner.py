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
        # The kernel grid in common_template covers speculative_num_steps
        # (= gamma+2) rows even though only gamma+1 step backends exist.
        backend = self.draft_attn_backend
        n_rows = backend.speculative_num_steps
        device = model_runner.device
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
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64
        )

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
        self.fbs[bs] = fb
        self.draft_attn_backend.init_forward_metadata_capture_cuda_graph(fb)
        return fb, input_ids, positions, out_cache_loc_steps

    def _draft_steps_in_graph(self, bs, forward, fb, input_ids, positions,
                              out_cache_loc_steps):
        """The captured draft loop: gamma+1 forwards + Gumbel sampling."""
        backends = self.draft_attn_backend.attn_backends
        tokens_out = self.tokens_out[:bs]
        logprobs_out = self.logprobs_out[:bs]
        tiny = torch.finfo(torch.float32).tiny
        for s in range(self.num_steps):
            fb.attn_backend = backends[s]
            fb.out_cache_loc = out_cache_loc_steps[s]
            # `forward` is the (patched) model.forward — returns a
            # LogitsProcessorOutput directly.
            logits = forward(input_ids, positions, fb).next_token_logits
            scaled = logits / self.temperature
            gumbel = -torch.log(
                -torch.log(torch.rand_like(scaled).clamp_min_(tiny))
            )
            idx = torch.argmax(scaled + gumbel, dim=-1)
            tokens_out[:, s + 1] = idx
            if s < self.gamma:
                chosen = scaled.gather(1, idx.unsqueeze(1)).squeeze(1)
                logprobs_out[:, s] = chosen - torch.logsumexp(scaled, dim=-1)
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
            self.next_tokens_out = torch.zeros(
                (self.max_bs, self.num_steps), dtype=torch.int64
            )
        self.verify_fbs = {}

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

    def _verify_in_graph(self, bs, fb, verify_input_ids, verify_positions):
        gamma = self.gamma
        tokens_out = self.tokens_out[:bs]
        logprobs_out = self.logprobs_out[:bs]
        logprob_diff_out = self.logprob_diff_out[:bs]
        bonus_out = self.bonus_out[:bs]
        next_tokens_out = self.next_tokens_out[:bs]
        tiny = torch.finfo(torch.float32).tiny

        # Score input = [x0, d_0..d_{gamma-1}] per row, request-major.
        verify_input_ids.view(bs, self.num_steps).copy_(
            tokens_out[:, : self.num_steps]
        )
        logits = self.target_runner.model.forward(
            verify_input_ids, verify_positions, fb
        ).next_token_logits
        logits3 = logits.view(bs, self.num_steps, -1)

        # Fused score logprobs under the tempered target p_T.
        verify_scaled = logits3[:, :gamma, :] / self.target_temperature
        chosen = verify_scaled.gather(
            2, tokens_out[:, 1 : gamma + 1].unsqueeze(2)
        ).squeeze(2)
        score_logprobs = chosen - torch.logsumexp(verify_scaled, dim=-1)
        logprob_diff_out.copy_(self.alpha * score_logprobs - logprobs_out)

        # Bonus from the same p_T^alpha tempered-power target, Gumbel-max.
        bonus_scaled = (
            self.alpha * logits3[:, -1, :] / self.target_temperature
        )
        bonus_gumbel = -torch.log(
            -torch.log(torch.rand_like(bonus_scaled).clamp_min_(tiny))
        )
        bonus_out.copy_(torch.argmax(bonus_scaled + bonus_gumbel, dim=-1))

        next_tokens_out[:, :gamma].copy_(tokens_out[:, 1 : gamma + 1])
        next_tokens_out[:, gamma].copy_(bonus_out)
        return (
            tokens_out,
            logprobs_out,
            logprob_diff_out,
            bonus_out,
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
        # Verify-side staging: positions [S..S+gamma] and request-major
        # cache locations per row.
        n_raw_tokens = raw_bs * self.num_steps
        self.verify_positions[:n_raw_tokens].view(raw_bs, self.num_steps).copy_(
            ctx.orig_seq_lens.unsqueeze(1)
            + torch.arange(self.num_steps, device=cache_locs.device)
        )
        self.verify_out_cache_loc[:n_raw_tokens].view(
            raw_bs, self.num_steps
        ).copy_(cache_locs)

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

        self.graphs[_default_make_graph_key(bs)].replay()
        return (
            self.tokens_out[:raw_bs],
            self.logprobs_out[:raw_bs],
            self.logprob_diff_out[:raw_bs],
            self.bonus_out[:raw_bs],
            self.next_tokens_out[:raw_bs],
        )
