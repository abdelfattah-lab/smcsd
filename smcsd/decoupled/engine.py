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

from smcsd.decoupled.draft_server import run_smc_draft_server_process
from smcsd.decoupled.pipeline_scheduler import run_pipelined_smc_scheduler_process
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
        **kwargs,
    ):
        # -- 1. Drafter ServerArgs (a plain model server for the draft model;
        #       the SMC refcounted allocator is installed by SMCTpModelWorker) --
        user_max = kwargs.get("max_running_requests")
        draft_args_kwargs = dict(
            model_path=draft_model_path,
            skip_tokenizer_init=True,
            disable_radix_cache=True,
            page_size=1,
            disable_cuda_graph=True,
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
            draft_args_kwargs["max_running_requests"] = user_max * (n_particles + 1)
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


class PipelinedDecoupledSMCEngine(DecoupledSMCEngine):
    """Decoupled SMC engine with cross-group draft/verify pipelining.

    Groups are split into ``SMCSD_PIPELINE_COHORTS`` cohorts (default 2); one
    cohort's draft round on the drafter GPU overlaps another cohort's verify
    on the target GPU.  Needs >= 2 concurrent requests to have anything to
    overlap — with a single group it degenerates to lockstep.
    """

    scheduler_process_func = staticmethod(run_pipelined_smc_scheduler_process)
