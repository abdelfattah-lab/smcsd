"""Standalone SMC draft engine process (the "drafter" role).

Hosts ONLY the draft model on its own GPU, with its own req_to_token pool,
refcounted KV allocator, and a per-slot mirror of the verifier's
``ScheduleBatchSMC`` membership (same slot ids, same seq_lens).  The verifier
drives every state transition over ZMQ (see ``io_struct.py``); this process
never makes scheduling, weighting, or resampling decisions.

Consistency contract (lockstep): for every slot the mirror's seq_len equals
the verifier's, and the mirror's block table holds draft KV for exactly the
token sequence the verifier attributes to that slot.  Maintained by applying
the verifier's membership ops in FIFO order; ``DraftStepReq.seq_lens`` is
asserted against the mirror every round to fail fast on divergence.
"""

from __future__ import annotations

import logging
import signal
from typing import Dict, List, Optional

import psutil
import torch
import zmq

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
from smcsd.core.kernels.fused_resample_kv import batched_resample_kv
from smcsd.core.scheduler import _prepare_req_for_private_prefill
from smcsd.decoupled.io_struct import (
    DraftCloseGroup,
    DraftCommitResample,
    DraftMaterializeGroup,
    DraftPing,
    DraftPong,
    DraftPrefillReq,
    DraftPrefillResp,
    DraftShutdown,
    DraftStepReq,
    DraftStepResp,
)
from smcsd.managers.smc_tp_worker import SMCTpModelWorker
from smcsd.mem_cache.allocator import SMCRefCountedTokenAllocator, copy_block_table

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

        # -- Draft model worker (own pools; SMCModelRunner installs the
        #    refcounted allocator since is_draft_worker=False here) --
        port_args = PortArgs.init_new(server_args)
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

    def event_loop(self) -> None:
        while True:
            msg = self.recv_from_verifier.recv_pyobj()
            if isinstance(msg, DraftStepReq):
                self.send_to_verifier.send_pyobj(self._handle_step(msg))
            elif isinstance(msg, DraftCommitResample):
                self._handle_commit(msg)
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

        particles = [
            clone_req_for_smc_particle(
                parent,
                particle_idx=i,
                temperature=self.draft_temperature,
                return_logprob=False,
            )
            for i in range(len(msg.slots))
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

        self._ensure_capacity(max(msg.slots) + 1)
        for slot, particle in zip(msg.slots, particles):
            if slot in self.slot_reqs:
                raise RuntimeError(f"SMCDraftServer: slot {slot} already in use.")
            self.slot_reqs[slot] = particle
            self.req_pool_indices[slot] = particle.req_pool_idx
            self.seq_lens[slot] = shared
            self.kv_allocated_lens[slot] = shared
        self.group_slots[msg.group_id] = list(msg.slots)

    # ────────────────────────────────────────────────────────
    #  Decode round (gamma+1 AR steps; last step only writes x_gamma's KV)
    # ────────────────────────────────────────────────────────

    def _handle_step(self, msg: DraftStepReq) -> DraftStepResp:
        active = torch.tensor(msg.slots, dtype=torch.int64, device=self.device)
        bs = len(msg.slots)
        gamma = self.gamma

        seq_g = self.seq_lens[active]
        seq_cpu = seq_g.cpu().tolist()
        if seq_cpu != list(msg.seq_lens):
            raise RuntimeError(
                "Drafter/verifier seq_lens divergence: "
                f"mirror={seq_cpu} verifier={list(msg.seq_lens)} slots={msg.slots}"
            )

        ctx, new_kv_alloc = SMCDecodeContext.from_slot_gather(
            seq_lens=seq_g,
            kv_allocated_lens=self.kv_allocated_lens[active],
            req_pool_indices=self.req_pool_indices[active],
            gamma_plus_1=gamma + 1,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
        )
        self.kv_allocated_lens[active] = new_kv_alloc
        self.seq_lens[active] = ctx.new_seq_lens

        verified = torch.tensor(
            msg.verified_ids, dtype=torch.int32, device=self.device
        )
        draft_input = SMCDraftInput(
            verified_id=verified, num_tokens_per_req=gamma + 1, decode_ctx=ctx
        )
        batch = self._build_decode_batch(active, draft_input, ctx)

        draft_fb, _can_cuda_graph, cache_locs, all_positions, all_seq_lens = (
            ctx.prepare_for_draft(
                verified,
                self.req_to_token_pool,
                batch,
                None,  # no CUDA graph on the drafter prototype
                self.model_runner,
            )
        )

        # Simple per-step AR path (same as SMCWorker._forward_decode's
        # non-multistep branch): update seq metadata in place, full forward
        # with per-step attention metadata init.
        current_ids = verified
        tokens: List[torch.Tensor] = []
        logprobs: List[torch.Tensor] = []
        for step in range(gamma + 1):
            draft_fb.input_ids = current_ids
            draft_fb.positions = all_positions[:, step].contiguous()
            draft_fb.out_cache_loc = cache_locs[:, step].contiguous()
            draft_fb.seq_lens = all_seq_lens[:, step].contiguous()
            draft_fb.seq_lens_sum = ctx.orig_seq_lens_sum + bs * (step + 1)
            draft_fb.seq_lens_cpu = ctx.orig_seq_lens_cpu + (step + 1)
            draft_out = self.model_runner.forward(draft_fb)

            logits = draft_out.logits_output.next_token_logits
            scaled_logits = logits / self.draft_temperature
            log_probs = torch.log_softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)

            if step < gamma:
                token_logprob = log_probs.gather(
                    1, next_token.unsqueeze(1)
                ).squeeze(1)
                tokens.append(next_token)
                logprobs.append(token_logprob)
            current_ids = next_token

        tokens_np = torch.stack(tokens, dim=1).cpu().numpy()
        logprobs_np = (
            torch.stack(logprobs, dim=1).to(torch.float32).cpu().numpy()
        )
        return DraftStepResp(tokens=tokens_np, logprobs=logprobs_np, tag=msg.tag)

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
        seq_lens_cpu = seq_lens.cpu()
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
            seq_lens_sum=int(seq_lens_cpu.sum().item()),
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
        dst = torch.tensor(msg.dst_slots, dtype=torch.int64, device=self.device)
        src = torch.tensor(msg.src_slots, dtype=torch.int64, device=self.device)

        to_free = batched_resample_kv(
            self.req_to_token_pool.req_to_token,
            self.token_to_kv_pool_allocator.slot_ref_count,
            self.req_pool_indices[dst].to(torch.int32),
            self.req_pool_indices[src].to(torch.int32),
            self.kv_allocated_lens[dst].to(torch.int32),
            self.seq_lens[src].to(torch.int32),
        )
        if to_free.numel() > 0:
            self.token_to_kv_pool_allocator.free(to_free)

        self.seq_lens[dst] = self.seq_lens[src]
        self.kv_allocated_lens[dst] = self.kv_allocated_lens[src]

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
        for slot in slots:
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
