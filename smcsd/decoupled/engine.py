"""Offline engine for decoupled SMC: target (verifier) on one GPU, draft
engine in a separate process on another GPU.

Drop-in replacement for ``SMCEngine``::

    engine = DecoupledSMCEngine(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        draft_model_path="meta-llama/Llama-3.2-1B-Instruct",
        n_particles=12, gamma=8,
        draft_gpu_id=1,
    )

The drafter process is spawned (and fully loaded) first; its ZMQ ipc:// base
name is handed to the scheduler subprocess via the ``SMCSD_DRAFT_IPC`` env
var.  Everything else (request flow, weighting, resampling, finalize) is the
verifier-side scheduler's job — see ``smcsd/decoupled/scheduler.py``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import uuid

from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.server_args import ServerArgs

from smcsd.decoupled.async_scheduler import run_async_smc_scheduler_process
from smcsd.decoupled.draft_server import run_smc_draft_server_process
from smcsd.decoupled.scheduler import DRAFT_IPC_ENV, run_decoupled_smc_scheduler_process
from smcsd.engine import SMCEngine

logger = logging.getLogger(__name__)


class DecoupledSMCEngine(SMCEngine):
    """SMCEngine variant whose draft model runs in a separate process/GPU."""

    scheduler_process_func = staticmethod(run_decoupled_smc_scheduler_process)

    def __init__(
        self,
        model_path: str,
        draft_model_path: str,
        *,
        n_particles: int = 4,
        gamma: int = 4,
        draft_temperature: float = 0.7,
        # Drafter placement / sizing
        draft_gpu_id: int = 1,
        draft_mem_fraction_static: float = 0.5,
        draft_cuda_graph: bool = True,
        **kwargs,
    ):
        # -- 1. Drafter ServerArgs (a plain model server for the draft model;
        #       the SMC refcounted allocator is installed by SMCTpModelWorker) --
        # Env override for quick A/B (cuda-graph vs multistep-only vs plain):
        #   SMCSD_DRAFT_CUDA_GRAPH=0/1
        _dcg = os.environ.get("SMCSD_DRAFT_CUDA_GRAPH")
        if _dcg is not None:
            draft_cuda_graph = _dcg not in ("0", "false", "False")
        user_max = kwargs.get("max_running_requests")
        draft_args_kwargs = dict(
            model_path=draft_model_path,
            skip_tokenizer_init=True,
            disable_radix_cache=True,
            page_size=1,
            # CUDA graphs on the drafter (the bottleneck stage): the AR loop
            # replays a captured standard-decode graph per step (mirrors the
            # colocated SMCWorker draft path).  The drafter has no speculative
            # algorithm, so it captures plain decode graphs.
            disable_cuda_graph=not draft_cuda_graph,
            # Upstream force-disables piecewise CUDA graph whenever a
            # speculative algorithm is set (colocated SMC therefore never
            # drafts through it).  This drafter has no spec algorithm in its
            # ServerArgs, so disable explicitly: the hand-mutated per-step AR
            # ForwardBatch is exactly the pattern piecewise replay mishandles
            # (observed as degraded drafts on ragged multi-group batches).
            disable_piecewise_cuda_graph=True,
            disable_overlap_schedule=True,
            mem_fraction_static=draft_mem_fraction_static,
            base_gpu_id=draft_gpu_id,
            tp_size=1,
            log_level=kwargs.get("log_level", "error"),
        )
        if user_max is not None:
            # Same expansion as the verifier: N particles + 1 transient parent.
            # Width-2 anchor tree (SMCSD_LAG1_ANCHOR_WIDTH>=2) adds N internal alt
            # draft slots per group (drafter-only); size the drafter's req pool +
            # CUDA-graph batch for them (the verifier still sees only N+1).
            _aw = max(int(os.environ.get("SMCSD_LAG1_ANCHOR_WIDTH", "1")), 1)
            per_group = (_aw * n_particles) + 1
            draft_args_kwargs["max_running_requests"] = user_max * per_group
        for key in ("trust_remote_code", "attention_backend", "random_seed", "context_length"):
            if key in kwargs:
                draft_args_kwargs[key] = kwargs[key]
        draft_server_args = ServerArgs(**draft_args_kwargs)
        _set_envs_and_config(draft_server_args)

        # -- 2. Spawn + await the drafter process --
        ipc_base = f"ipc:///tmp/smcsd_draft_{uuid.uuid4().hex[:12]}"
        os.environ[DRAFT_IPC_ENV] = ipc_base  # inherited by the scheduler proc

        ctx = mp.get_context("spawn")
        reader, writer = ctx.Pipe(duplex=False)
        self._draft_proc = ctx.Process(
            target=run_smc_draft_server_process,
            args=(
                draft_server_args,
                draft_gpu_id,
                gamma,
                draft_temperature,
                f"{ipc_base}_req",
                f"{ipc_base}_resp",
                writer,
            ),
            daemon=False,
        )
        self._draft_proc.start()
        logger.info(
            "DecoupledSMCEngine: waiting for draft engine (%s on GPU %d)...",
            draft_model_path, draft_gpu_id,
        )
        ready = reader.recv()  # raises EOFError if the drafter died during init
        assert ready.get("status") == "ready"
        logger.info("DecoupledSMCEngine: draft engine is ready.")

        # -- 3. Verifier engine (launches DecoupledSMCScheduler subprocess) --
        super().__init__(
            model_path,
            draft_model_path,
            n_particles=n_particles,
            gamma=gamma,
            draft_temperature=draft_temperature,
            **kwargs,
        )


class AsyncDecoupledSMCEngine(DecoupledSMCEngine):
    """Decoupled SMC engine with lag-1 exact-bonus draft/verify overlap.

    The drafter emits a gamma+1 anchor bet for run-ahead matching, but the
    verifier still commits the target-sampled bonus. Resampling is fixed at
    one verify cycle.
    """

    scheduler_process_func = staticmethod(run_async_smc_scheduler_process)

    def __init__(self, *args, **kwargs):
        previous = {
            "SMCSD_LAG1_BONUS": os.environ.get("SMCSD_LAG1_BONUS"),
            "SMCSD_LAG1_ANCHOR_WIDTH": os.environ.get("SMCSD_LAG1_ANCHOR_WIDTH"),
        }
        os.environ["SMCSD_LAG1_BONUS"] = "1"
        os.environ.setdefault("SMCSD_LAG1_ANCHOR_WIDTH", "2")
        try:
            super().__init__(*args, **kwargs)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
