"""Standalone SMC draft engine process (the "drafter" role).

Hosts ONLY the draft model on its own GPU, with its own req_to_token pool,
refcounted KV allocator, and a per-slot mirror of the verifier's
``ScheduleBatchSMC`` membership (same slot ids, same seq_lens).  The verifier
drives every state transition over ZMQ (see ``io_struct.py``); this process
never makes scheduling, weighting, or resampling decisions.

Consistency contract (decoupled_serial): for every slot the mirror's seq_len equals
the verifier's, and the mirror's block table holds draft KV for exactly the
token sequence the verifier attributes to that slot.  Maintained by applying
the verifier's membership ops in FIFO order; ``DraftStepReq.seq_lens`` is
asserted against the mirror every round to fail fast on divergence.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import psutil
import torch
import zmq

import numpy as np
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req, ScheduleBatch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import configure_logger, kill_itself_when_parent_died
from sglang.srt.utils.network import get_zmq_socket
from sglang.utils import get_exception_traceback

from smcsd.common.utils import _release_internal_req, clone_req_for_smc_particle
from smcsd.core.info import SMCDecodeContext, SMCDraftInput
from smcsd.core.scheduler import _prepare_req_for_private_prefill
from smcsd.decoupled.io_struct import (
    DraftCloseGroup,
    DraftCommitResample,
    DraftMaterializeGroup,
    DraftPing,
    DraftPong,
    DraftPrefillReq,
    DraftPrefillResp,
    DraftPromoteAlt,
    DraftShutdown,
    DraftStepReq,
    DraftStepResp,
)
from smcsd.decoupled.kv_utils import truncate_block_table_allocations
from smcsd.managers.smc_tp_worker import SMCTpModelWorker
from smcsd.mem_cache.allocator import (
    SMCRefCountedTokenAllocator,
    copy_block_table,
)

logger = logging.getLogger(__name__)

EMPTY_SLOT = -1


class SMCDraftServer:
    """Drafter-side engine: draft model + slot-state mirror + ZMQ loop."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        gamma: int,
        draft_temperature: float,
        ipc_req_name: str,
        ipc_resp_name: str,
    ):
        self.server_args = server_args
        self.gamma = gamma
        self.draft_temperature = max(float(draft_temperature), 1e-5)
        # Lag-1 exact-bonus mode needs the gamma+1 anchor emitted as a bet; the
        # verifier still commits the target-sampled bonus.
        from sglang.srt.utils import get_bool_env_var

        self._lag1_bonus = get_bool_env_var("SMCSD_LAG1_BONUS", "false")
        self._emit_anchor = self._lag1_bonus
        # Default 0.1 (was: inherit draft temp): a greedier anchor "bet" x_g1
        # raises the bet-hit rate (x_g1==b) -> fewer run-ahead discards / catch-up
        # cycles -> higher tok/s, accuracy-neutral (the committed token is always
        # the exact target bonus b).  See docs/smc/lag1_optimization.md (Round 1).
        _anchor_t = os.environ.get("SMCSD_ANCHOR_TEMP", "0.1")
        self.anchor_temperature = max(float(_anchor_t), 0.05)
        # Width-W anchor tree (SMCSD_LAG1_ANCHOR_WIDTH): emit the top-W anchor
        # candidates at the bet position so the verifier can hedge the run-ahead
        # seed.  1 = today's single-bet behavior (no extra emit, wire-unchanged).
        self._anchor_width = max(int(os.environ.get("SMCSD_LAG1_ANCHOR_WIDTH", "1")), 1)
        # Optional draft-time decomposition (SMCSD_TIMING=1): the AR loop wall
        # time per window (the final .cpu() syncs, so this captures GPU time).
        self._timing = get_bool_env_var("SMCSD_TIMING", "false")
        self._t_draft = {"alloc": 0.0, "ar": 0.0, "out": 0.0}
        self._t_draft_n = 0

        # -- Draft model worker (own pools; SMCModelRunner installs the
        #    refcounted allocator since is_draft_worker=False here) --
        # Mirror the colocated SMCWorker draft path: suppress auto-capture
        # during worker init, build the multi-step decode attention backend,
        # then restore + manually capture device graphs.  CUDA graphs on the
        # drafter (the bottleneck) replay a captured standard-decode graph per
        # AR step; the multi-step backend is the fallback for uncaptured shapes.
        port_args = PortArgs.init_new(server_args)
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        self.tp_worker = SMCTpModelWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
            attn_cp_rank=0,
            moe_dp_rank=0,
            dp_rank=None,
            nccl_port=port_args.nccl_port,
        )
        self.model_runner = self.tp_worker.model_runner
        self.model_config = self.tp_worker.model_config

        self.draft_attn_backend = None
        try:
            from sglang.srt.speculative.draft_utils import DraftBackendFactory

            factory = DraftBackendFactory(
                server_args, self.model_runner, topk=1,
                speculative_num_steps=self.gamma + 2,
            )
            self.draft_attn_backend = factory.create_decode_backend()
        except Exception as exc:  # fallback to plain forward for uncaptured shapes
            logger.warning("SMCDraftServer: multistep draft backend unavailable: %s", exc)

        server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        if not backup_disable_cuda_graph:
            self.model_runner.init_device_graphs()
        self.draft_cuda_graph = not backup_disable_cuda_graph
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )
        if not isinstance(
            self.token_to_kv_pool_allocator, SMCRefCountedTokenAllocator
        ):
            raise RuntimeError(
                "SMCDraftServer requires the SMC refcounted allocator; got "
                f"{type(self.token_to_kv_pool_allocator).__name__}"
            )
        self.device = self.req_to_token_pool.device
        self._torch_profiler = None
        self._torch_profile_step_n = 0
        self._torch_profile_total_steps = 0
        self._torch_profile_stopped = False
        self._init_torch_profiler("draft")
        self.tree_cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=server_args.page_size,
            )
        )

        # -- Slot mirror (grows on demand; slot ids come from the verifier) --
        self.max_slots = 0
        self.req_pool_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.kv_allocated_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self._ensure_capacity(1024)

        self.slot_reqs: Dict[int, Req] = {}
        self.group_slots: Dict[str, List[int]] = {}
        self.pending_parents: Dict[str, Req] = {}
        # Width-2 anchor tree: each primary slot s gets an internal "alt" draft
        # slot _alt_of[s] (drafter-only; the verifier never sees it).  The alt
        # holds the c1-branch run-ahead so a c1-hit avoids the catch-up cycle.
        # alt ids live in a high range to never collide with verifier slot ids.
        self._alt_of: Dict[int, int] = {}
        self._group_alt_slots: Dict[str, List[int]] = {}
        self._ALT_BASE = 1 << 14  # 16384; verifier slot ids stay well below this

        # -- ZMQ (drafter binds; verifier connects) --
        self._zmq_context = zmq.Context(2)
        self.recv_from_verifier = get_zmq_socket(
            self._zmq_context, zmq.PULL, ipc_req_name, True
        )
        self.send_to_verifier = get_zmq_socket(
            self._zmq_context, zmq.PUSH, ipc_resp_name, True
        )

        logger.info(
            "SMCDraftServer ready: model=%s gpu=%d gamma=%d draft_temp=%.3f",
            server_args.model_path, gpu_id, gamma, self.draft_temperature,
        )

    # ────────────────────────────────────────────────────────
    #  Event loop
    # ────────────────────────────────────────────────────────

    def _init_torch_profiler(self, role: str) -> None:
        out_dir = os.environ.get("SMCSD_TORCH_PROFILE_DIR")
        if not out_dir:
            return
        roles = {
            part.strip()
            for part in os.environ.get("SMCSD_TORCH_PROFILE_ROLES", "target,draft").split(",")
            if part.strip()
        }
        if role not in roles:
            return
        wait = max(int(os.environ.get("SMCSD_TORCH_PROFILE_WAIT", "5")), 0)
        active = max(int(os.environ.get("SMCSD_TORCH_PROFILE_ACTIVE", "20")), 1)
        with_stack = os.environ.get("SMCSD_TORCH_PROFILE_WITH_STACK", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        role_dir = os.path.join(out_dir, role)
        os.makedirs(role_dir, exist_ok=True)
        self._torch_profile_total_steps = wait + active
        self._torch_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=0, active=active, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                role_dir, worker_name=f"{role}-{os.getpid()}"
            ),
            record_shapes=False,
            with_stack=with_stack,
        )
        self._torch_profiler.start()
        print(
            "[TORCH_PROFILE] "
            f"role={role} dir={role_dir} wait={wait} active={active} "
            f"with_stack={with_stack}",
            flush=True,
        )

    def _torch_record(self, name: str):
        if self._torch_profiler is None or self._torch_profile_stopped:
            return nullcontext()
        return torch.profiler.record_function(name)

    def _torch_profile_step(self) -> None:
        if self._torch_profiler is None or self._torch_profile_stopped:
            return
        self._torch_profiler.step()
        self._torch_profile_step_n += 1
        if self._torch_profile_step_n >= self._torch_profile_total_steps:
            self._torch_profiler.stop()
            self._torch_profile_stopped = True
            print("[TORCH_PROFILE] role=draft stopped", flush=True)

    def event_loop(self) -> None:
        while True:
            msg = self.recv_from_verifier.recv_pyobj()
            if isinstance(msg, DraftStepReq):
                with self._torch_record("draft_handle_step"):
                    resp = self._handle_step(msg)
                self._torch_profile_step()
                self.send_to_verifier.send_pyobj(resp)
            elif isinstance(msg, DraftCommitResample):
                self._handle_commit(msg)
            elif isinstance(msg, DraftPromoteAlt):
                self._handle_promote_alt(msg)
            elif isinstance(msg, DraftPrefillReq):
                self.send_to_verifier.send_pyobj(self._handle_prefill(msg))
            elif isinstance(msg, DraftMaterializeGroup):
                self._handle_materialize(msg)
            elif isinstance(msg, DraftCloseGroup):
                self._handle_close(msg)
            elif isinstance(msg, DraftPing):
                self.send_to_verifier.send_pyobj(
                    DraftPong(info={"model_path": self.server_args.model_path})
                )
            elif isinstance(msg, DraftShutdown):
                logger.info("SMCDraftServer: shutdown requested.")
                return
            else:
                raise RuntimeError(f"SMCDraftServer: unknown message {type(msg)}")

    # ────────────────────────────────────────────────────────
    #  Prefill + materialize (group bootstrap)
    # ────────────────────────────────────────────────────────

    def _handle_prefill(self, msg: DraftPrefillReq) -> DraftPrefillResp:
        reqs: List[Req] = []
        for group_id, ids, gsp in zip(msg.group_ids, msg.input_ids, msg.sampling):
            sp = SamplingParams(
                temperature=gsp.temperature,
                top_p=gsp.top_p,
                top_k=gsp.top_k,
                min_p=gsp.min_p,
                max_new_tokens=gsp.max_new_tokens,
            )
            sp.normalize(None)
            req = Req(
                rid=group_id,
                origin_input_text="",
                origin_input_ids=list(ids),
                sampling_params=sp,
                vocab_size=self.model_config.vocab_size,
                eos_token_ids=self.model_config.hf_eos_token_id,
            )
            req.tokenizer = None
            _prepare_req_for_private_prefill(req)
            reqs.append(req)

        batch = ScheduleBatch.init_new(
            reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            False,  # enable_overlap
            SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        result = self.tp_worker.forward_batch_generation(
            batch.get_model_worker_batch()
        )
        next_ids = result.next_token_ids.tolist()

        for group_id, ids, req in zip(msg.group_ids, msg.input_ids, reqs):
            req.kv_committed_len = len(ids)
            req.kv_allocated_len = len(ids)
            self.pending_parents[group_id] = req

        return DraftPrefillResp(group_ids=list(msg.group_ids), next_token_ids=next_ids)

    def _handle_materialize(self, msg: DraftMaterializeGroup) -> None:
        parent = self.pending_parents.pop(msg.group_id)
        shared = int(msg.shared_seq_len)
        if parent.kv_committed_len != shared:
            raise RuntimeError(
                f"Drafter/verifier prefix mismatch for group {msg.group_id}: "
                f"drafter committed {parent.kv_committed_len}, verifier says {shared}"
            )

        # Width-2: allocate N extra "alt" draft slots (one per primary), cloned
        # from the same parent prefix; they hold the c1-branch run-ahead.
        n = len(msg.slots)
        alt_slots = [self._ALT_BASE + s for s in msg.slots] if self._anchor_width >= 2 else []
        all_slots = list(msg.slots) + alt_slots
        particles = [
            clone_req_for_smc_particle(
                parent,
                particle_idx=i,
                temperature=self.draft_temperature,
                return_logprob=False,
            )
            for i in range(len(all_slots))
        ]
        if self.req_to_token_pool.alloc(particles) is None:
            raise RuntimeError("SMCDraftServer: req_to_token_pool full at materialize.")
        for particle in particles:
            copy_block_table(
                self.req_to_token_pool,
                parent.req_pool_idx,
                particle.req_pool_idx,
                shared,
                self.token_to_kv_pool_allocator,
            )
            particle.kv_committed_len = shared
            particle.kv_allocated_len = shared
        _release_internal_req(
            parent,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )

        self._ensure_capacity(max(all_slots) + 1)
        for slot, particle in zip(all_slots, particles):
            if slot in self.slot_reqs:
                raise RuntimeError(f"SMCDraftServer: slot {slot} already in use.")
            self.slot_reqs[slot] = particle
            self.req_pool_indices[slot] = particle.req_pool_idx
            self.seq_lens[slot] = shared
            self.kv_allocated_lens[slot] = shared
        self.group_slots[msg.group_id] = list(msg.slots)
        if alt_slots:
            for s, a in zip(msg.slots, alt_slots):
                self._alt_of[s] = a
            self._group_alt_slots[msg.group_id] = alt_slots

    # ────────────────────────────────────────────────────────
    #  Decode round (gamma+1 AR steps; last step only writes x_gamma's KV)
    # ────────────────────────────────────────────────────────

    def _handle_step(self, msg: DraftStepReq) -> DraftStepResp:
        active = torch.tensor(msg.slots, dtype=torch.int64, device=self.device)
        bs = len(msg.slots)
        gamma = self.gamma
        prof_alloc = prof_ar = prof_out = 0.0

        if isinstance(msg.rollback, list):
            # Mixed lag-1 StepReq: some rows advance normally while stale rows
            # rewind gamma+1 and re-draft in place from their committed bonus.
            if any(msg.rollback):
                self.seq_lens[active] -= torch.tensor(
                    msg.rollback, dtype=self.seq_lens.dtype, device=self.device
                )
        elif msg.rollback:
            raise RuntimeError(
                "Nonzero scalar draft rollback is no longer supported; "
                "use lag-1 per-row rollback lists."
            )

        seq_g = self.seq_lens[active]
        seq_cpu_t = seq_g.cpu()
        seq_cpu = seq_cpu_t.tolist()
        if seq_cpu != list(msg.seq_lens):
            raise RuntimeError(
                "Drafter/verifier seq_lens divergence: "
                f"mirror={seq_cpu} verifier={list(msg.seq_lens)} slots={msg.slots}"
            )
        self._truncate_step_kv_if_requested(active, msg.truncate_kv)

        # ── Width-2 anchor tree: fold an alt run-ahead row per branched slot into
        #    the SAME AR batch (one batched pass keeps the doubled draft ~free; a
        #    2nd pass would ~2x the launch cost).  Each alt is a clone of its
        #    primary's committed prefix [0, orig) seeded by c1 (msg.bet_alt).  The
        #    branch is resolved after the verifier's verify via DraftPromoteAlt. ──
        bs_primary = bs
        verified_ids_ext = list(msg.verified_ids)
        branched_pos: List[int] = []
        if msg.bet_alt is not None:
            branched_pos = [i for i, c in enumerate(msg.bet_alt) if c >= 0]
        if branched_pos:
            prim_branched = [msg.slots[i] for i in branched_pos]
            alt_ids = [self._alt_of[s] for s in prim_branched]
            # Re-sync each alt to its primary's committed prefix (frees the alt's
            # previous run-ahead); reuses the resample KV-clone primitive.
            self._handle_commit(
                DraftCommitResample(dst_slots=alt_ids, src_slots=prim_branched)
            )
            alt_t = torch.tensor(alt_ids, dtype=torch.int64, device=self.device)
            # Drop phantom slack inherited from the primary's kv_allocated so the
            # alt's own from_slot_gather allocates a fresh run-ahead suffix.
            self.kv_allocated_lens[alt_t] = self.seq_lens[alt_t]
            active = torch.cat([active, alt_t])
            verified_ids_ext += [int(msg.bet_alt[i]) for i in branched_pos]
            bs = int(active.numel())
            seq_g = self.seq_lens[active]
            seq_cpu_t = seq_g.cpu()

        tm = self._timing
        measure = tm
        t0 = time.perf_counter() if measure else 0.0
        ctx, new_kv_alloc = SMCDecodeContext.from_slot_gather(
            seq_lens=seq_g,
            kv_allocated_lens=self.kv_allocated_lens[active],
            req_pool_indices=self.req_pool_indices[active],
            gamma_plus_1=gamma + 1,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
            seq_lens_cpu=seq_cpu_t,
        )
        self.kv_allocated_lens[active] = new_kv_alloc
        self.seq_lens[active] = ctx.new_seq_lens
        if measure:
            prof_alloc = time.perf_counter() - t0
            if tm:
                self._t_draft["alloc"] += prof_alloc

        verified = torch.tensor(
            verified_ids_ext, dtype=torch.int32, device=self.device
        )
        draft_input = SMCDraftInput(
            verified_id=verified, num_tokens_per_req=gamma + 1, decode_ctx=ctx
        )
        batch = self._build_decode_batch(active, draft_input, ctx)

        graph_runner = getattr(self.model_runner, "graph_runner", None)
        draft_fb, can_cuda_graph, cache_locs, all_positions, all_seq_lens = (
            ctx.prepare_for_draft(
                verified,
                self.req_to_token_pool,
                batch,
                graph_runner,
                self.model_runner,
            )
        )

        # Draft AR (mirrors SMCWorker._forward_decode): replay the captured
        # standard-decode CUDA graph per step when the batch shape matches (the
        # fast path on the bottleneck drafter), else the multi-step attention
        # backend, else a plain per-step forward.
        use_multistep = self.draft_attn_backend is not None and not can_cuda_graph
        if use_multistep and not draft_fb.forward_mode.is_idle():
            draft_fb.spec_info = draft_input
            draft_fb.seq_lens = ctx.orig_seq_lens
            draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu
            self.draft_attn_backend.init_forward_metadata(draft_fb)

        # Lag-1 returns gamma+1 tokens: the final column is the drafter-known
        # anchor bet. Plain serial mode returns gamma because the verifier
        # samples the anchor from the target.
        n_emit = gamma + 1 if self._emit_anchor else gamma
        current_ids = verified
        tokens: List[torch.Tensor] = []
        logprobs: List[torch.Tensor] = []
        bet_topk_tokens: Optional[torch.Tensor] = None
        bet_topk_logprobs: Optional[torch.Tensor] = None
        t0 = time.perf_counter() if measure else 0.0
        for step in range(gamma + 1):
            draft_fb.input_ids = current_ids
            draft_fb.positions = all_positions[:, step].contiguous()
            draft_fb.out_cache_loc = cache_locs[:, step].contiguous()

            if use_multistep:
                draft_fb.attn_backend = self.draft_attn_backend.attn_backends[step]
                draft_out = self.model_runner.forward(
                    draft_fb, skip_attn_backend_init=True
                )
            else:
                draft_fb.seq_lens = all_seq_lens[:, step].contiguous()
                draft_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * (step + 1)
                draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + (step + 1)
                draft_out = self.model_runner.forward(draft_fb)

            logits = draft_out.logits_output.next_token_logits
            step_temp = self.draft_temperature
            if self._emit_anchor and step == gamma and self.anchor_temperature is not None:
                step_temp = self.anchor_temperature
            scaled_logits = logits / step_temp
            log_probs = torch.log_softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)

            if self._emit_anchor and step == gamma and self._anchor_width >= 2:
                # Width-W anchor tree: emit the top-W most-likely anchor candidates
                # (under the anchor-temp dist) so the verifier can hedge the bet.
                bt_lp, bt_idx = torch.topk(log_probs, self._anchor_width, dim=-1)
                bet_topk_tokens = bt_idx.to(torch.int64)
                bet_topk_logprobs = bt_lp

            if step < n_emit:
                token_logprob = log_probs.gather(
                    1, next_token.unsqueeze(1)
                ).squeeze(1)
                tokens.append(next_token)
                logprobs.append(token_logprob)
            current_ids = next_token

        if measure:
            torch.cuda.synchronize()
            prof_ar = time.perf_counter() - t0
            if tm:
                self._t_draft["ar"] += prof_ar
            t0 = time.perf_counter()
        tokens_all = torch.stack(tokens, dim=1)  # (bs_ext, n_emit)
        logprobs_all = torch.stack(logprobs, dim=1).to(torch.float32)
        tokens_np = tokens_all[:bs_primary].cpu().numpy()
        logprobs_np = logprobs_all[:bs_primary].cpu().numpy()
        if measure:
            prof_out = time.perf_counter() - t0
        if tm:
            self._t_draft["out"] += prof_out
            self._t_draft_n += 1
            if self._t_draft_n >= 200:
                per = {k: 1e3 * v / self._t_draft_n for k, v in self._t_draft.items()}
                print(
                    f"[DRAFT_TIMING] {self._t_draft_n} windows (per-window ms): "
                    f"alloc={per['alloc']:.2f} "
                    f"ar({gamma+1} steps,cuda_graph={not use_multistep})={per['ar']:.2f} "
                    f"out={per['out']:.2f} total={sum(per.values()):.2f}",
                    flush=True,
                )
                self._t_draft = {k: 0.0 for k in self._t_draft}
                self._t_draft_n = 0
        bet_topk_np = (
            bet_topk_tokens[:bs_primary].cpu().numpy()
            if bet_topk_tokens is not None
            else None
        )
        bet_topk_lp_np = (
            bet_topk_logprobs[:bs_primary].to(torch.float32).cpu().numpy()
            if bet_topk_logprobs is not None
            else None
        )
        # Width-2: split the alt run-ahead rows (appended after the primaries) back
        # to per-primary-slot arrays, filled only at branched positions.
        alt_tokens_np = alt_logprobs_np = alt_bet_topk_np = None
        if branched_pos:
            n_emit_eff = int(tokens_all.shape[1])
            alt_tokens_np = np.zeros((bs_primary, n_emit_eff), dtype=np.int64)
            alt_logprobs_np = np.zeros((bs_primary, n_emit_eff), dtype=np.float32)
            alt_tokens_np[branched_pos] = tokens_all[bs_primary:].cpu().numpy()
            alt_logprobs_np[branched_pos] = logprobs_all[bs_primary:].cpu().numpy()
            if bet_topk_tokens is not None:
                W = int(bet_topk_tokens.shape[1])
                alt_bet_topk_np = np.zeros((bs_primary, W), dtype=np.int64)
                alt_bet_topk_np[branched_pos] = (
                    bet_topk_tokens[bs_primary:].cpu().numpy()
                )
        return DraftStepResp(
            tokens=tokens_np,
            logprobs=logprobs_np,
            tag=msg.tag,
            epoch=msg.epoch,
            bet_topk=bet_topk_np,
            bet_topk_logprobs=bet_topk_lp_np,
            alt_tokens=alt_tokens_np,
            alt_logprobs=alt_logprobs_np,
            alt_bet_topk=alt_bet_topk_np,
        )

    def _truncate_step_kv_if_requested(self, active: torch.Tensor, truncate_kv) -> None:
        if isinstance(truncate_kv, list):
            if len(truncate_kv) != int(active.numel()):
                raise RuntimeError(
                    "DraftStepReq.truncate_kv length mismatch: "
                    f"{len(truncate_kv)} flags for {int(active.numel())} slots"
                )
            mask = torch.tensor(truncate_kv, dtype=torch.bool, device=self.device)
            if not bool(mask.any().item()):
                return
            active = active[mask]
        elif not truncate_kv:
            return

        new_alloc_lens = self.seq_lens[active].clone()
        old_alloc_lens = self.kv_allocated_lens[active].clone()
        truncate_block_table_allocations(
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.req_pool_indices[active],
            old_alloc_lens,
            new_alloc_lens,
        )
        self.kv_allocated_lens[active] = new_alloc_lens

    def _build_decode_batch(
        self,
        active: torch.Tensor,
        draft_input: SMCDraftInput,
        ctx: SMCDecodeContext,
    ) -> ModelWorkerBatch:
        """Mirror of ``ScheduleBatchSMC.build_model_worker_batch`` over the
        drafter's slot tensors (sampling_info is a placeholder — the AR loop
        samples explicitly)."""
        bs = active.numel()
        seq_lens = ctx.new_seq_lens
        seq_lens_cpu = getattr(ctx, "new_seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = self.seq_lens[active].detach().cpu()
        seq_lens_sum_attr = getattr(ctx, "new_seq_lens_sum", None)
        seq_lens_sum = (
            int(seq_lens_sum_attr)
            if seq_lens_sum_attr is not None
            else int(seq_lens_cpu.sum().item())
        )
        sampling_info = SamplingBatchInfo(
            temperatures=torch.ones(bs, 1, dtype=torch.float32, device=self.device),
            top_ps=torch.ones(bs, dtype=torch.float32, device=self.device),
            top_ks=torch.full((bs,), -1, dtype=torch.int32, device=self.device),
            min_ps=torch.zeros(bs, dtype=torch.float32, device=self.device),
            is_all_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=self.model_config.vocab_size,
        )
        return ModelWorkerBatch(
            forward_mode=ForwardMode.DECODE,
            input_ids=draft_input.verified_id,
            req_pool_indices=self.req_pool_indices[active],
            seq_lens=seq_lens,
            out_cache_loc=None,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=seq_lens_sum,
            return_logprob=False,
            top_logprobs_nums=[0] * bs,
            token_ids_logprobs=None,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=False,
            all_extend_in_batch=False,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=None,
            extend_num_tokens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            multimodal_inputs=[None] * bs,
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=None,
            sampling_info=sampling_info,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            spec_info=draft_input,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            reqs=[self.slot_reqs[s] for s in active.cpu().tolist()],
        )

    # ────────────────────────────────────────────────────────
    #  Commit (resample) + close
    # ────────────────────────────────────────────────────────

    def _handle_commit(self, msg: DraftCommitResample) -> None:
        if not msg.dst_slots:
            return
        for dst_slot, src_slot in zip(msg.dst_slots, msg.src_slots):
            if dst_slot == src_slot:
                continue
            dst_pool = int(self.req_pool_indices[dst_slot].item())
            src_pool = int(self.req_pool_indices[src_slot].item())
            if dst_pool == EMPTY_SLOT or src_pool == EMPTY_SLOT:
                raise RuntimeError(
                    "Drafter resample references an unmapped slot: "
                    f"dst={dst_slot}:{dst_pool} src={src_slot}:{src_pool}"
                )

            dst_alloc = int(self.kv_allocated_lens[dst_slot].item())
            src_seq = int(self.seq_lens[src_slot].item())
            if dst_alloc > 0:
                old = self.req_to_token_pool.req_to_token[
                    dst_pool, :dst_alloc
                ].clone()
                self.token_to_kv_pool_allocator.dec_ref_and_free(
                    old.to(torch.int64)
                )
            if src_seq > 0:
                copied = self.req_to_token_pool.req_to_token[
                    src_pool, :src_seq
                ].clone()
                self.token_to_kv_pool_allocator.inc_ref(copied.to(torch.int64))
                self.req_to_token_pool.write(
                    (dst_pool, slice(0, src_seq)), copied
                )

            self.seq_lens[dst_slot] = self.seq_lens[src_slot]
            self.kv_allocated_lens[dst_slot] = self.kv_allocated_lens[src_slot]

    def _handle_promote_alt(self, msg: DraftPromoteAlt) -> None:
        """Width-2 branch resolution: for slots whose c1 branch won (promote=True),
        swap the slot's draft state with its alt slot so the slot's lineage becomes
        the c1 run-ahead.  O(1) per slot (no KV copy); the displaced c0 run-ahead
        lives in the alt and is freed/re-cloned on the next branched cycle.  Slots
        with promote=False keep their primary (c0) run-ahead."""
        prom = [
            s
            for s, p in zip(msg.slots, msg.promote)
            if p and self._alt_of.get(s) is not None
        ]
        if not prom:
            return
        s_t = torch.tensor(prom, dtype=torch.int64, device=self.device)
        a_t = torch.tensor(
            [self._alt_of[s] for s in prom], dtype=torch.int64, device=self.device
        )
        for T in (self.req_pool_indices, self.seq_lens, self.kv_allocated_lens):
            tmp = T[s_t].clone()
            T[s_t] = T[a_t]
            T[a_t] = tmp
        for s in prom:
            a = self._alt_of[s]
            self.slot_reqs[s], self.slot_reqs[a] = self.slot_reqs[a], self.slot_reqs[s]

    def _handle_close(self, msg: DraftCloseGroup) -> None:
        parent = self.pending_parents.pop(msg.group_id, None)
        if parent is not None:
            _release_internal_req(
                parent,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        slots = self.group_slots.pop(msg.group_id, None)
        if slots is None:
            if parent is None:
                logger.warning(
                    "SMCDraftServer: close for unknown group %s (ignored).",
                    msg.group_id,
                )
            return
        # Width-2: free the group's internal alt draft slots alongside its primaries.
        alt_slots = self._group_alt_slots.pop(msg.group_id, [])
        for s in slots:
            self._alt_of.pop(s, None)
        for slot in list(slots) + list(alt_slots):
            req = self.slot_reqs.pop(slot)
            req.kv_allocated_len = int(self.kv_allocated_lens[slot].item())
            _release_internal_req(
                req,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )
            self.req_pool_indices[slot] = EMPTY_SLOT
            self.seq_lens[slot] = 0
            self.kv_allocated_lens[slot] = 0

    # ────────────────────────────────────────────────────────
    #  Misc
    # ────────────────────────────────────────────────────────

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self.max_slots:
            return
        new_size = max(needed, self.max_slots * 2, 1024)

        def grow(t: torch.Tensor, fill: int) -> torch.Tensor:
            out = torch.full(
                (new_size,), fill, dtype=t.dtype, device=self.device
            )
            if t.numel() > 0:
                out[: t.numel()] = t
            return out

        self.req_pool_indices = grow(self.req_pool_indices, EMPTY_SLOT)
        self.seq_lens = grow(self.seq_lens, 0)
        self.kv_allocated_lens = grow(self.kv_allocated_lens, 0)
        self.max_slots = new_size


def run_smc_draft_server_process(
    server_args: ServerArgs,
    gpu_id: int,
    gamma: int,
    draft_temperature: float,
    ipc_req_name: str,
    ipc_resp_name: str,
    pipe_writer,
):
    configure_logger(server_args, prefix=" DRAFT")
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    try:
        server = SMCDraftServer(
            server_args,
            gpu_id,
            gamma,
            draft_temperature,
            ipc_req_name,
            ipc_resp_name,
        )
        pipe_writer.send({"status": "ready"})
        server.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"SMCDraftServer hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
