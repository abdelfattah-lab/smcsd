"""SMC variant of CudaGraphRunner.

Overrides ``get_spec_info`` to return ``SMCVerifyInput`` during CUDA
graph capture so Triton autotune / graph capture sees the SMC-specific
attention path.  Non-SMC paths (draft worker, other spec algos) delegate
to the base class.
"""

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode


class SMCCudaGraphRunner(CudaGraphRunner):
    def get_spec_info(self, num_tokens: int):
        if (
            self.model_runner.spec_algorithm.is_smc()
            and not self.model_runner.is_draft_worker
        ):
            from smcsd.common.verify import SMCVerifyInput

            return SMCVerifyInput(
                draft_token_num=self.num_tokens_per_bs,
                positions=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                num_tokens_per_req=self.num_tokens_per_bs,
            )
        return super().get_spec_info(num_tokens)


class SMCDraftHeadGraphRunner(SMCCudaGraphRunner):
    """CUDA-graph runner for the deferred-bonus 2-token DRAFT head.

    The draft's primary graph runner is decode-only (``num_tokens_per_bs=1``;
    SMC drafts are deliberately excluded from the speculative capture path).
    This second runner captures ``num_tokens_per_bs=2`` TARGET_VERIFY graphs on
    the *draft* model so the head's 2-token extend can replay instead of running
    eager.

    Construction trick: the base ``__init__`` decides ``capture_forward_mode`` /
    ``num_tokens_per_bs`` from ``spec_algorithm`` + ``is_draft_worker`` and
    routes SMC drafts to the decode branch (and would ``raise`` on the draft
    verify branch).  We temporarily spoof ``is_draft_worker=False`` so the base
    takes the verify-capture branch (``TARGET_VERIFY``,
    ``num_tokens_per_bs=speculative_num_draft_tokens``) and pin that token count
    to 2 via the constructor arg.  ``get_spec_info`` (also gated on
    ``not is_draft_worker``) then yields ``SMCVerifyInput(draft_token_num=2)``
    during capture.  The flag is restored after construction; replay reads only
    ``num_tokens_per_bs`` (=2) and the captured graphs, not ``is_draft_worker``.
    """

    def __init__(self, draft_model_runner):
        mr = draft_model_runner
        saved = mr.is_draft_worker
        mr.is_draft_worker = False
        # NOTE: the draft backend's verify-block-size global (triton:
        # num_draft_tokens; FA3: speculative_num_draft_tokens) is pinned to 2
        # by SMCWorker.__init__ before this runner is constructed — the
        # vendored verify capture/replay paths read that global, not the dummy
        # batch's extend fields.
        #
        # Reuse the decode graph runner's already-allocated cuda-graph metadata
        # buffers instead of letting the base __init__ call
        # init_cuda_graph_state again: a second call REBINDS the backend's
        # buffers (new tensors), orphaning the decode graphs' baked pointers —
        # decode replay_prepare would then write fresh metadata into the new
        # buffers while the captured decode graphs keep reading the old ones
        # (stale/freed) → garbage attention → NaN logits.  Sharing is safe on
        # both supported backends:
        #  * triton — head and decode replays are strictly sequential on one
        #    stream, each preceded by its own replay_prepare write; and the
        #    head adds no size requirements (kv_indices scales with request
        #    count, max_bs * max_context, not query tokens; the
        #    max_num_tokens-scaled buffers attn_logits/lse/num_kv_splits are
        #    decode-kernel workspaces the linear-verify path never touches).
        #  * FA3 — even stronger isolation: per-mode metadata dicts
        #    (decode_cuda_graph_metadata vs target_verify_metadata) with
        #    separately-allocated, max_bs/context-scaled slabs.  The verify
        #    slabs already exist on the draft backend (the allocation gate
        #    speculative_num_draft_tokens > 0 passed at decode-graph init —
        #    gamma+1 inherited from shared server_args), and the per-bs
        #    cu_seqlens_q is created fresh at capture, never updated at
        #    replay.
        saved_init = mr.attn_backend.init_cuda_graph_state
        mr.attn_backend.init_cuda_graph_state = lambda *args, **kwargs: None
        try:
            super().__init__(mr, speculative_num_draft_tokens=2)
        finally:
            mr.attn_backend.init_cuda_graph_state = saved_init
            mr.is_draft_worker = saved
