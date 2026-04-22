"""CUDA graph runner for the EAGLE3 draft under SMC.

The standard ``draft_runner.init_device_graphs()`` path does not bake
``EagleDraftInput.hidden_states`` into the graph input set, so the SMC
EAGLE3 decode loop has been running the draft eagerly — paying full Python
launch overhead on every per-step forward.

This runner captures one graph per registered batch size, keyed on the
(bs, hidden_dim) combination actually observed. On first replay for a new
key it lazily allocates buffers + captures. Thereafter ``replay()`` copies
live tensors into persistent device buffers and issues a graph replay.

Scope limitations (intentional):
  * One-step graph. The captured region is a single ``draft_runner.forward(fb)``
    call. Multi-step capture (upstream style) is deferred.
  * Sampling stays outside the graph. Verified to be ~3% of cycle.
  * Draft-attention backend metadata is (re)initialised on every replay via
    the standard ``init_forward_metadata_replay_cuda_graph`` path, same as
    upstream ``EAGLEDraftCudaGraphRunner``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from sglang.srt.model_executor.cuda_graph_runner import (
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput

if TYPE_CHECKING:
    from smcsd.v2.worker import SMCWorkerV2

logger = logging.getLogger(__name__)


class SMCEagle3DraftCudaGraphRunner:
    """Per-step EAGLE3 draft CUDA-graph capture/replay for SMC."""

    def __init__(self, worker: "SMCWorkerV2"):
        self.worker = worker
        self.draft_runner = worker.draft_runner
        # Use the draft model_runner's DEFAULT attention backend (step_id=0,
        # topk=1) — matches what the eager EAGLE3 path uses. The multistep
        # backend (worker.draft_attn_backend) encodes different speculative
        # step_ids per backend, which would force one captured graph per step
        # and double-offset seq_lens under our convention.
        self.attn_backend = self.draft_runner.attn_backend
        self.device = worker.device

        # Capture bs schedule + max_bs from server_args
        self.capture_bs, _ = get_batch_sizes_to_capture(self.draft_runner)
        self.max_bs: int = max(self.capture_bs) if self.capture_bs else 0
        if self.max_bs == 0:
            raise RuntimeError(
                "SMCEagle3DraftCudaGraphRunner: capture_bs is empty; "
                "check --cuda-graph-max-bs is >0."
            )

        # seq_lens fill value for padded rows.
        self._seq_len_fill = self.attn_backend.get_cuda_graph_seq_len_fill_value()

        # Per-(bs, hidden_dim) captured graphs. Populated lazily.
        self._graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._buffers: Dict[Tuple[int, int], dict] = {}

        # Backend needs a one-time size init; only do it once.
        # num_tokens_per_bs = 1 for our single-step decode graph.
        self.attn_backend.init_cuda_graph_state(self.max_bs, self.max_bs)

        logger.info(
            "SMCEagle3DraftCudaGraphRunner: lazy capture — "
            "capture_bs=%s, hidden_dtype=%s, device=%s",
            self.capture_bs,
            worker._eagle3_hidden_dtype,
            self.device,
        )

    # ──────────────────────── public API ────────────────────────

    def can_run(self, bs: int, hidden_dim: int) -> bool:
        """Whether a replay exists (or can be captured) for this key.

        We lazily capture new keys so always return True unless the bs
        exceeds our max — but gate on max_bs so we don't unboundedly
        allocate under pathological inputs.
        """
        return bs <= self.max_bs

    def replay(
        self,
        bs: int,
        input_ids: torch.Tensor,          # (bs,)   int64
        hidden_states: torch.Tensor,      # (bs, H) fc-dtype
        positions: torch.Tensor,          # (bs,)   int64
        seq_lens: torch.Tensor,           # (bs,)   int32
        out_cache_loc: torch.Tensor,      # (bs,)   kv-dtype
        req_pool_indices: torch.Tensor,   # (bs,)   int64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one EAGLE3 draft step via graph replay.

        Returns (logits, new_hidden) as fresh *copies* of the captured output
        buffer views — callers should assume the returned tensors are only
        valid until the next ``replay()`` invocation.
        """
        hidden_dim = int(hidden_states.shape[-1])
        pad_bs = self._pick_capture_bs(bs)
        key = (pad_bs, hidden_dim)

        if key not in self._graphs:
            self._capture(pad_bs, hidden_dim, hidden_states.dtype, out_cache_loc.dtype)

        buf = self._buffers[key]

        # Copy inputs into persistent buffers. Zero-pad beyond raw_bs so the
        # attention backend sees stable (fill-value) seq_lens for dummy rows.
        if pad_bs != bs:
            buf["seq_lens"].fill_(self._seq_len_fill)
            buf["positions"].zero_()
            buf["input_ids"].zero_()
            buf["out_cache_loc"].zero_()
            buf["req_pool_indices"].zero_()

        buf["input_ids"][:bs].copy_(input_ids)
        buf["hidden_states"][:bs].copy_(hidden_states.to(buf["hidden_states"].dtype))
        buf["positions"][:bs].copy_(positions)
        buf["seq_lens"][:bs].copy_(seq_lens)
        buf["out_cache_loc"][:bs].copy_(out_cache_loc)
        buf["req_pool_indices"][:bs].copy_(req_pool_indices)

        # Attention metadata for this replay (bs may be padded up).
        fb = buf["forward_batch"]
        fb.batch_size = pad_bs
        fb.seq_lens_sum = (
            int(seq_lens.sum().item()) + (pad_bs - bs) * self._seq_len_fill
        )
        # seq_lens_cpu used by the backend for max_len.item() — keep it
        # stale-safe by copying the live seq_lens (cheap, sync-free on H2D).
        buf["seq_lens_cpu"][:bs].copy_(seq_lens, non_blocking=True)
        if pad_bs != bs:
            buf["seq_lens_cpu"][bs:pad_bs].fill_(self._seq_len_fill)

        self.attn_backend.init_forward_metadata_replay_cuda_graph(
            pad_bs,
            buf["req_pool_indices"],
            buf["seq_lens"],
            fb.seq_lens_sum,
            None,                                   # encoder_lens
            ForwardMode.DECODE,
            buf["spec_info"],
            buf["seq_lens_cpu"],
            buf["out_cache_loc"],
        )

        self._graphs[key].replay()

        logits = buf["logits"][:bs]
        new_hidden = buf["new_hidden"][:bs]
        return logits, new_hidden

    # ──────────────────────── capture path ────────────────────────

    def _pick_capture_bs(self, bs: int) -> int:
        """Round bs up to the smallest capture_bs that fits."""
        for cbs in self.capture_bs:
            if cbs >= bs:
                return cbs
        return self.max_bs

    def _capture(self, bs: int, hidden_dim: int, hidden_dtype, out_cache_dtype):
        logger.info(
            "SMCEagle3DraftCudaGraphRunner: capturing graph bs=%d hidden_dim=%d "
            "hidden_dtype=%s", bs, hidden_dim, hidden_dtype,
        )
        with torch.device(self.device):
            input_ids = torch.zeros((bs,), dtype=torch.int64)
            hidden_states = torch.zeros((bs, hidden_dim), dtype=hidden_dtype)
            positions = torch.zeros((bs,), dtype=torch.int64)
            seq_lens = torch.full((bs,), self._seq_len_fill, dtype=torch.int32)
            out_cache_loc = torch.zeros((bs,), dtype=out_cache_dtype)
            req_pool_indices = torch.zeros((bs,), dtype=torch.int64)
            extend_seq_lens = torch.ones((bs,), dtype=torch.int32)

        seq_lens_cpu = torch.full((bs,), self._seq_len_fill, dtype=torch.int32)

        spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=input_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=[self._seq_len_fill] * bs,
            req_to_token_pool=self.draft_runner.req_to_token_pool,
            token_to_kv_pool=self.draft_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(self._seq_len_fill) * bs,
            return_logprob=False,
            positions=positions,
            spec_algorithm=self.draft_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

        # Use the draft's default decode attn backend (step_id=0). This is
        # the same path `self.draft_runner.forward(...)` picks when no
        # backend is explicitly set, so CG semantics match the eager EAGLE3
        # decode loop on the branch.
        fb.attn_backend = self.attn_backend

        # Bake capture-side attention metadata for this bs.
        self.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            bs,                              # num_tokens (1 per req)
            req_pool_indices,
            seq_lens,
            None,                            # encoder_lens
            ForwardMode.DECODE,
            spec_info,
        )

        def _run_once():
            out = self.draft_runner.forward(fb, skip_attn_backend_init=True)
            return out.logits_output.next_token_logits, out.logits_output.hidden_states

        # Warmup (jit / autotune state must be resolved before capture).
        for _ in range(2):
            torch.cuda.synchronize()
            _run_once()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.current_stream()
        pool = get_global_graph_memory_pool()
        with model_capture_mode():
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                logits_out, hidden_out = _run_once()
        set_global_graph_memory_pool(graph.pool())

        self._graphs[(bs, hidden_dim)] = graph
        self._buffers[(bs, hidden_dim)] = {
            "input_ids": input_ids,
            "hidden_states": hidden_states,
            "positions": positions,
            "seq_lens": seq_lens,
            "seq_lens_cpu": seq_lens_cpu,
            "req_pool_indices": req_pool_indices,
            "out_cache_loc": out_cache_loc,
            "spec_info": spec_info,
            "forward_batch": fb,
            "logits": logits_out,
            "new_hidden": hidden_out,
        }
