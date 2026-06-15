"""Verifier-side pieces of decoupled SMC: the draft-engine client and the
spec worker that replaces the local draft AR loop with an RPC.

``DecoupledSMCWorker`` mirrors ``smcsd.core.worker.SMCWorker`` exactly on the
verify / weight / bonus path; only the draft half changes:

- prefill: draft prompt prefill + x0 sampling happen on the drafter process;
- decode: gamma draft tokens + per-position draft logprobs arrive over ZMQ,
  while the target-side cache locations (already allocated by the scheduler's
  ``from_slot_gather``) are read directly with ``assign_smc_cache_locs_kernel``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar

import torch
import zmq

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.utils.network import get_zmq_socket

from smcsd.common.verify import assign_smc_cache_locs_kernel
from smcsd.core.info import SMCDraftInput
from smcsd.decoupled.io_struct import (
    DraftCloseGroup,
    DraftCommitResample,
    DraftMaterializeGroup,
    DraftPing,
    DraftPong,
    DraftPrefillReq,
    DraftPrefillResp,
    DraftStepReq,
    DraftStepResp,
    GroupSamplingParams,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Generous timeout: covers drafter model load (ping) and any prefill round.
DRAFT_RECV_TIMEOUT_MS = 600_000


class DraftEngineClient:
    """Blocking lockstep client for the drafter process (FIFO channel)."""

    def __init__(self, ipc_req_name: str, ipc_resp_name: str):
        self._zmq_context = zmq.Context(2)
        self.send_to_drafter = get_zmq_socket(
            self._zmq_context, zmq.PUSH, ipc_req_name, False
        )
        self.recv_from_drafter = get_zmq_socket(
            self._zmq_context, zmq.PULL, ipc_resp_name, False
        )
        self.recv_from_drafter.setsockopt(zmq.RCVTIMEO, DRAFT_RECV_TIMEOUT_MS)

    def _recv(self, expected: Type[T]) -> T:
        try:
            msg = self.recv_from_drafter.recv_pyobj()
        except zmq.error.Again:
            raise RuntimeError(
                f"Timed out waiting for {expected.__name__} from the draft "
                "engine — the drafter process is likely dead."
            )
        if not isinstance(msg, expected):
            raise RuntimeError(
                f"Expected {expected.__name__} from drafter, got {type(msg)}"
            )
        return msg

    def ping(self) -> DraftPong:
        self.send_to_drafter.send_pyobj(DraftPing())
        return self._recv(DraftPong)

    def prefill(
        self,
        group_ids: List[str],
        input_ids: List[List[int]],
        sampling: List[GroupSamplingParams],
    ) -> DraftPrefillResp:
        self.send_to_drafter.send_pyobj(
            DraftPrefillReq(
                group_ids=group_ids, input_ids=input_ids, sampling=sampling
            )
        )
        return self._recv(DraftPrefillResp)

    def send_step(
        self,
        slots: List[int],
        verified_ids: List[int],
        seq_lens: List[int],
        tag: int = 0,
        epoch: int = 0,
    ) -> None:
        """Fire a draft round without waiting for the reply (pipelined path)."""
        self.send_to_drafter.send_pyobj(
            DraftStepReq(
                slots=slots,
                verified_ids=verified_ids,
                seq_lens=seq_lens,
                tag=tag,
                epoch=epoch,
            )
        )

    def recv_step_resp(self, timeout_ms: Optional[int] = None) -> Optional[DraftStepResp]:
        """Receive the next StepResp.  With ``timeout_ms`` set, returns None on
        timeout instead of raising (used by the pipelined scheduler's poll)."""
        if timeout_ms is not None:
            if self.recv_from_drafter.poll(timeout_ms) == 0:
                return None
        return self._recv(DraftStepResp)

    def step(
        self, slots: List[int], verified_ids: List[int], seq_lens: List[int]
    ) -> DraftStepResp:
        self.send_step(slots, verified_ids, seq_lens)
        return self._recv(DraftStepResp)

    def send_materialize(
        self, group_id: str, slots: List[int], shared_seq_len: int
    ) -> None:
        self.send_to_drafter.send_pyobj(
            DraftMaterializeGroup(
                group_id=group_id, slots=slots, shared_seq_len=shared_seq_len
            )
        )

    def send_commit(self, dst_slots: List[int], src_slots: List[int]) -> None:
        self.send_to_drafter.send_pyobj(
            DraftCommitResample(dst_slots=dst_slots, src_slots=src_slots)
        )

    def send_close(self, group_id: str, slots: Optional[List[int]] = None) -> None:
        self.send_to_drafter.send_pyobj(
            DraftCloseGroup(group_id=group_id, slots=slots or [])
        )


@dataclass
class PendingDecodeStep:
    """In-flight draft round: snapshot taken by start_decode, consumed by
    finish_decode once the drafter's StepResp arrives."""

    batch: ModelWorkerBatch
    ctx: object  # SMCDecodeContext
    tag: int
    epoch: int = 0


class DecoupledSMCWorker(BaseSpecWorker):
    """SMC spec worker whose draft model lives in a separate process."""

    def __init__(
        self,
        server_args: ServerArgs,
        target_worker: TpModelWorker,
        draft_client: DraftEngineClient,
    ):
        self.server_args = server_args
        self.device = server_args.device
        self._target_worker = target_worker
        self._client = draft_client

        self.gamma = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = self.gamma + 1
        self.smc_target_temperature = max(
            float(server_args.smc_target_temperature), 1e-5
        )
        # No-bonus mode: the anchor is the drafter's own (gamma+1)-th token
        # (returned over the wire), weighted; no target bonus sampling.  Mirrors
        # SMCWorker.  The drafter applies the matching anchor temperature.
        from sglang.srt.utils import get_bool_env_var

        self.drop_bonus = get_bool_env_var("SMCSD_DROP_BONUS", "false")
        # Diagnostic (SMCSD_BONUS_AGREE_STATS=1): in no-bonus mode, measure how
        # often the drafter's anchor token (tokens[:, gamma]) equals the target
        # bonus — the "at-will bonus" match rate that bounds how much of the bonus
        # is already gotten for free vs would need a stall to reclaim.
        self._bonus_agree = get_bool_env_var("SMCSD_BONUS_AGREE_STATS", "false")
        self._ba_n = 0
        self._ba_match_sample = 0
        self._ba_match_argmax = 0

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        SMCDraftInput.ALLOC_LEN_PER_DECODE = self.speculative_num_draft_tokens

        # ScheduleBatchSMC sizes its history buffer from server_args.context_length;
        # resolve it from the target model config (colocated SMCWorker does the
        # same as a side effect of configuring the draft worker).
        if server_args.context_length is None:
            server_args.context_length = (
                target_worker.model_runner.model_config.context_len
            )

    # ── Properties (required by BaseSpecWorker / scheduler) ──

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return None  # the draft model lives in the drafter process

    @property
    def model_config(self):
        return self._target_worker.model_config

    @property
    def model_runner(self):
        return self._target_worker.model_runner

    def clear_cache_pool(self):
        pass

    def materialize_smc_parent_draft_prefix(self, req) -> None:
        """No-op: the drafter prefilled the prompt during _forward_extend."""
        pass

    # ── Main entry point ──

    def forward_batch_generation(self, batch):
        if isinstance(batch, ScheduleBatch):
            batch = batch.get_model_worker_batch()

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_extend(batch)
        else:
            return self._forward_decode(batch)

    # ── EXTEND (prefill) ──

    def _forward_extend(self, batch: ModelWorkerBatch):
        # Target (score) model prefill
        score_result = self._target_worker.forward_batch_generation(batch)

        # Draft prefill on the drafter process — returns x0 per group
        reqs = batch.reqs
        resp = self._client.prefill(
            group_ids=[req.rid for req in reqs],
            input_ids=[list(req.origin_input_ids) for req in reqs],
            sampling=[
                GroupSamplingParams(
                    temperature=req.sampling_params.temperature,
                    top_p=req.sampling_params.top_p,
                    top_k=req.sampling_params.top_k,
                    min_p=req.sampling_params.min_p,
                    max_new_tokens=req.sampling_params.max_new_tokens,
                )
                for req in reqs
            ],
        )
        x0 = torch.tensor(resp.next_token_ids, dtype=torch.int64, device=self.device)

        bs = len(batch.seq_lens)
        score_result.next_token_ids = x0
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
    # Split into start_decode (cache locs + fire the draft RPC) and
    # finish_decode (verify/weights/bonus) so the pipelined scheduler can
    # overlap one cohort's draft with another cohort's verify.  The lockstep
    # path is start + blocking recv + finish — behavior unchanged.

    def send_step_req(
        self,
        active_slots: List[int],
        verified_ids_cpu: List[int],
        seq_lens_cpu: List[int],
        tag: int = 0,
        epoch: int = 0,
    ) -> None:
        """Fire one draft round on the drafter (no slot_state dependency).

        The async scheduler uses this to prefetch the next window's StepReq from
        raw lists while it verifies the current window locally — the overlap.
        """
        self._client.send_step(
            slots=active_slots,
            verified_ids=verified_ids_cpu,
            seq_lens=seq_lens_cpu,
            tag=tag,
            epoch=epoch,
        )

    def start_decode(
        self, batch: ModelWorkerBatch, tag: int = 0, epoch: int = 0
    ) -> "PendingDecodeStep":
        draft_input: SMCDraftInput = batch.spec_info
        ctx = draft_input.decode_ctx
        active_slots = getattr(draft_input, "active_slots_cpu", None)
        if active_slots is None:
            raise RuntimeError(
                "DecoupledSMCWorker requires active_slots_cpu on the draft "
                "input (set by DecoupledSMCScheduler._prepare_decode_batch)."
            )
        self.send_step_req(
            active_slots,
            draft_input.verified_id.cpu().tolist(),
            ctx.orig_seq_lens_cpu.tolist(),
            tag=tag,
            epoch=epoch,
        )
        return PendingDecodeStep(batch=batch, ctx=ctx, tag=tag, epoch=epoch)

    def _forward_decode(self, batch: ModelWorkerBatch):
        if batch.forward_mode.is_idle():
            return self._forward_idle(batch)
        pending = self.start_decode(batch)
        resp = self._client.recv_step_resp()
        return self.finish_decode(pending, resp)

    def finish_decode(self, pending: "PendingDecodeStep", resp, drop_bonus=None):
        batch = pending.batch
        ctx = pending.ctx
        draft_input: SMCDraftInput = batch.spec_info
        bs = len(ctx.orig_seq_lens)
        gamma = self.gamma
        # Per-call anchor mode.  Defaults to the worker's mode, but the async
        # scheduler overrides it to False on the BARRIER window (drained, so the
        # target bonus is available before the next train's window-0 StepReq):
        # weight only gamma draft positions and seed the next anchor from the
        # exact target sample, even though the drafter ran no-bonus and emitted
        # gamma+1 columns (we slice to gamma).
        drop_bonus = self.drop_bonus if drop_bonus is None else drop_bonus

        # Target-side cache locations for x0..x_gamma (computed here, not at
        # send time, so the async path can prefetch the StepReq before the
        # verifier's target KV for this window is even touched).
        out_cache_loc = torch.empty(
            bs * (gamma + 1), dtype=torch.int64, device=ctx.orig_seq_lens.device
        )
        assign_smc_cache_locs_kernel[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            ctx.orig_seq_lens,
            out_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            gamma + 1,
        )
        cache_locs = out_cache_loc.reshape(bs, gamma + 1)
        if resp.tag != pending.tag:
            raise RuntimeError(
                f"Draft step reply tag mismatch: got {resp.tag}, "
                f"expected {pending.tag} (FIFO violated?)"
            )
        if resp.epoch != pending.epoch:
            raise RuntimeError(
                f"Draft step reply epoch mismatch: got {resp.epoch}, "
                f"expected {pending.epoch} (train fence violated?)"
            )
        device = ctx.orig_seq_lens.device
        tokens = torch.from_numpy(resp.tokens).to(device=device, dtype=torch.int64)
        draft_logprobs_stacked = torch.from_numpy(resp.logprobs).to(
            device=device, dtype=torch.float32
        )

        x0 = draft_input.verified_id.to(torch.int64)
        all_tokens = [x0] + [tokens[:, j] for j in range(gamma)]

        # ---- 1. Score verify (identical to colocated SMCWorker) ----
        verify_forward_batch, can_run_cuda_graph = ctx.prepare_for_verify(
            self.req_to_token_pool,
            batch,
            self._target_worker,
            all_tokens,
            cache_locs,
        )
        score_result = self._target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )

        # ---- 2. Extract score logprobs ----
        score_logits = score_result.logits_output.next_token_logits
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]} rows, "
            f"expected {expected_rows} (bs={bs}, gamma+1={gamma + 1}, "
            f"cuda_graph={can_run_cuda_graph})"
        )
        # Weight n_weighted columns: gamma (bonus — the anchor is the exact target
        # sample, weight 0) or gamma+1 (no-bonus — the anchor is the reweighted
        # (gamma+1)-th draft token).  `tokens` may carry gamma+1 columns even in
        # bonus mode (a no-bonus drafter that this call overrides), so slice it.
        n_weighted = gamma + 1 if drop_bonus else gamma
        weighted_tokens = tokens[:, :n_weighted]
        weighted_draft_lp = draft_logprobs_stacked[:, :n_weighted]
        score_log_probs = torch.log_softmax(score_logits, dim=-1)
        score_log_probs = score_log_probs.reshape(bs, gamma + 1, -1)
        score_logprobs_stacked = score_log_probs[:, :n_weighted, :].gather(
            2, weighted_tokens.unsqueeze(2)
        ).squeeze(2)

        # ---- 3. Logprob diff (SMC importance weight increment) ----
        logprob_diff = (score_logprobs_stacked - weighted_draft_lp).sum(dim=1)

        # ---- 4. Anchor + output ----
        if drop_bonus:
            # Anchor = the drafter's own (gamma+1)-th token; output = x1..x_{gamma+1}.
            next_verified_id = tokens[:, gamma]
            output_token_ids = tokens
            if self._bonus_agree:
                # Compare the drafter's anchor to the target bonus (both sampled,
                # and the argmax) — the at-will match rate.
                blogits = score_logits.reshape(bs, gamma + 1, -1)[:, gamma, :]
                blp = torch.log_softmax(blogits / self.smc_target_temperature, dim=-1)
                bsample = torch.multinomial(blp.exp(), num_samples=1).squeeze(-1)
                bargmax = blogits.argmax(dim=-1)
                anchor = tokens[:, gamma]
                self._ba_n += bs
                self._ba_match_sample += int((anchor == bsample).sum().item())
                self._ba_match_argmax += int((anchor == bargmax).sum().item())
                if self._ba_n >= 2000:
                    print(
                        f"[BONUS_AGREE] {self._ba_n} anchors: "
                        f"draft==target_sample={100*self._ba_match_sample/self._ba_n:.1f}% "
                        f"draft==target_argmax={100*self._ba_match_argmax/self._ba_n:.1f}%",
                        flush=True,
                    )
                    self._ba_n = self._ba_match_sample = self._ba_match_argmax = 0
        else:
            # Anchor = exact target sample after x_gamma; output = x1..x_gamma + bonus.
            bonus_logits = score_logits.reshape(bs, gamma + 1, -1)[:, gamma, :]
            bonus_log_probs = torch.log_softmax(
                bonus_logits / self.smc_target_temperature, dim=-1
            )
            next_verified_id = torch.multinomial(
                bonus_log_probs.exp(), num_samples=1
            ).squeeze(-1)
            output_token_ids = torch.cat(
                [tokens[:, :gamma], next_verified_id.unsqueeze(1)], dim=1
            )

        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full(
            (bs,), gamma + 1, dtype=torch.int32, device=device
        )

        next_draft_input = SMCDraftInput(
            verified_id=next_verified_id,
            logprob_diff=logprob_diff,
            num_tokens_per_req=self.speculative_num_draft_tokens,
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
