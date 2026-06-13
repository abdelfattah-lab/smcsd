"""Verifier-side scheduler for decoupled SMC.

``DecoupledSMCScheduler`` is ``SMCScheduler`` with the draft model moved to a
separate process: it wires up ``DecoupledSMCWorker`` + ``DraftEngineClient``
instead of the colocated ``SMCWorker``, and forwards every slot-membership
transition (materialize / resample / close) to the drafter so its KV mirror
stays consistent.  All weighting/resampling/finish logic is inherited.
"""

from __future__ import annotations

import logging
import os
import signal
from typing import Optional

import psutil
import torch

from sglang.srt.managers.scheduler import configure_scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import kill_itself_when_parent_died
from sglang.utils import get_exception_traceback

from smcsd.core.scheduler import SequenceGroup, SMCScheduler
from smcsd.decoupled.worker import DecoupledSMCWorker, DraftEngineClient

logger = logging.getLogger(__name__)

DRAFT_IPC_ENV = "SMCSD_DRAFT_IPC"


class DecoupledSMCScheduler(SMCScheduler):
    """SMC scheduler whose draft engine runs in a separate process."""

    # ── Worker override: RPC draft worker instead of colocated SMCWorker ──

    def maybe_init_draft_worker(self):
        ipc_base = os.environ.get(DRAFT_IPC_ENV)
        if not ipc_base:
            raise RuntimeError(
                f"DecoupledSMCScheduler requires the {DRAFT_IPC_ENV} env var "
                "(set by DecoupledSMCEngine)."
            )
        self._draft_client = DraftEngineClient(
            ipc_req_name=f"{ipc_base}_req",
            ipc_resp_name=f"{ipc_base}_resp",
        )
        pong = self._draft_client.ping()
        logger.info("Connected to draft engine: %s", pong.info)
        self.draft_worker = DecoupledSMCWorker(
            server_args=self.server_args,
            target_worker=self.tp_worker,
            draft_client=self._draft_client,
        )

    # ── Decode prep: expose the active slot ids to the worker RPC ──

    def _prepare_decode_batch(self):
        draft_input = self.slot_state.prepare_for_decode()
        if draft_input.decode_ctx is None:
            return None
        draft_input.active_slots_cpu = list(self.slot_state._active_slots_list)
        return self.slot_state.build_model_worker_batch(draft_input)

    # ── Group bootstrap: mirror materialize / early-finish to the drafter ──

    def _process_prefill_result(self, batch, result) -> None:
        # Same as SMCScheduler._process_prefill_result, plus a DraftCloseGroup
        # for parents that finish at prefill (the drafter holds their pending
        # parent KV from the prefill RPC).
        groups = self.prefill_groups
        self.prefill_groups = []
        if not groups:
            raise RuntimeError("Prefill result without active prefill group.")

        if result.copy_done is not None:
            result.copy_done.synchronize()

        next_token_ids = result.next_token_ids.tolist()
        assert len(next_token_ids) == len(batch.reqs) == len(groups)

        for group, req, next_token_id in zip(groups, batch.reqs, next_token_ids):
            assert req is group.parent_req

            req.output_ids.append(next_token_id)
            req.check_finished()

            if req.finished():
                self._draft_client.send_close(group.group_id)
                release_kv_cache(req, self.tree_cache)
                req.time_stats.set_completion_time()
                self.stream_output([req], False)
                continue

            error_msg = self._materialize_group(group)
            if error_msg is not None:
                self._abort_group(group, error_msg)
                continue

            self.running_groups.append(group)

    def _materialize_group(self, group: SequenceGroup) -> Optional[str]:
        error_msg = super()._materialize_group(group)
        if error_msg is None:
            # allocate_slots seeded every slot's seq_len with shared_seq_len;
            # read it back from there (parent_req's kv fields were reset when
            # the parent was released inside super()).
            slots = list(self.slot_state.group_slot_lists[group.group_id])
            shared_seq_len = int(self.slot_state.seq_lens[slots[0]].item())
            self._draft_client.send_materialize(
                group_id=group.group_id,
                slots=slots,
                shared_seq_len=shared_seq_len,
            )
        return error_msg

    def _abort_group(self, group: SequenceGroup, error_msg: str) -> None:
        self._draft_client.send_close(group.group_id)
        super()._abort_group(group, error_msg)

    # ── Decode result: mirror the resample plan to the drafter ──

    def _process_decode_result(self, result: GenerationBatchResult) -> None:
        if result.copy_done is not None:
            result.copy_done.synchronize()

        if result.logprob_diff is None:
            raise RuntimeError("SMCScheduler requires batched logprob_diff.")

        logprob_diff = (
            result.logprob_diff
            if torch.is_tensor(result.logprob_diff)
            else torch.as_tensor(
                result.logprob_diff, dtype=torch.float32, device=self.device
            )
        )

        next_draft = result.next_draft_input
        bonus_ids = next_draft.verified_id if next_draft is not None else None
        if bonus_ids is None:
            raise RuntimeError(
                "SMCScheduler: result missing next_draft_input.verified_id"
            )

        newly_finished = self.slot_state.process_batch_result(
            next_token_ids=result.next_token_ids,
            accept_lens=result.accept_lens,
            logprob_diff=logprob_diff,
            bonus_ids=bonus_ids,
            rebuild_active=False,
        )

        plan = self.coordinator.collect_resample_jobs_batch(self.slot_state)
        did_resample = plan.n_jobs > 0
        if did_resample:
            self.coordinator.dispatch_resample_batch(
                plan, self.slot_state, rebuild_active=False,
            )
            # Mirror the exact dst<-src plan on the drafter's KV state.
            self._draft_client.send_commit(
                dst_slots=plan.dst_slots.tolist(),
                src_slots=plan.src_slots.tolist(),
            )

        if newly_finished or did_resample:
            self.slot_state.rebuild_active_slots()

        self._drain_finished_groups()

    # ── Finalize: release the drafter's slots ──

    def _finalize_group(self, group: SequenceGroup) -> None:
        slots = list(self.slot_state.group_slot_lists.get(group.group_id, []))
        super()._finalize_group(group)
        self._draft_client.send_close(group.group_id, slots=slots)


def run_decoupled_smc_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    dp_rank = configure_scheduler(
        server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank
    )

    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    try:
        scheduler = DecoupledSMCScheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )
        pipe_writer.send(scheduler.get_init_info())
        scheduler.run_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DecoupledSMCScheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
