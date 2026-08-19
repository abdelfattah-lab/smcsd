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
from contextlib import nullcontext
from typing import List, Optional

import psutil
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import configure_scheduler_process
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import DynamicGradMode
from sglang.utils import get_exception_traceback

from smcsd.core.scheduler import SequenceGroup, SMCScheduler
from smcsd.decoupled.worker import DecoupledSMCWorker, DraftEngineClient

logger = logging.getLogger(__name__)

DRAFT_IPC_ENV = "SMCSD_DRAFT_IPC"


class DecoupledSMCScheduler(SMCScheduler):
    """SMC scheduler whose draft engine runs in a separate process."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._torch_profiler = None
        self._torch_profile_step_n = 0
        self._torch_profile_total_steps = 0
        self._torch_profile_stopped = False
        if type(self) is DecoupledSMCScheduler:
            self._init_torch_profiler("target")

    # ── Worker override: RPC draft worker instead of colocated SMCWorker ──

    def maybe_init_draft_worker(self):
        self.external_corpus_manager = None
        ipc_base = os.environ.get(DRAFT_IPC_ENV)
        if not ipc_base:
            raise RuntimeError(
                f"DecoupledSMCScheduler requires the {DRAFT_IPC_ENV} env var "
                "(set by DecoupledSMCEngine)."
            )
        self._draft_client = DraftEngineClient(
            ipc_req_name=f"{ipc_base}_req",
            ipc_resp_name=f"{ipc_base}_resp",
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        pong = self._draft_client.ping()
        logger.info(
            "Connected to draft engine (tp_rank=%d/%d, owner=%s): %s",
            self.tp_rank,
            self.tp_size,
            self.tp_rank == 0,
            pong.info,
        )
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

    def _process_batch_result(
        self,
        next_token_ids: torch.Tensor,
        logprob_diff: torch.Tensor,
        bonus_ids: torch.Tensor,
        *,
        rebuild_active: bool = True,
        active: Optional[torch.Tensor] = None,
    ) -> List[int]:
        if active is None:
            active = self.slot_state.active_slots
        prev_finished = self.slot_state.finished_mask[active].clone()
        self.slot_state.write_back_gpu(
            next_token_ids=next_token_ids,
            logprob_diff=logprob_diff,
            bonus_ids=bonus_ids,
            active=active,
        )
        newly_finished_t = self.slot_state.finished_mask[active] & ~prev_finished
        newly_finished = active[newly_finished_t].detach().cpu().tolist()
        if newly_finished and rebuild_active:
            self.slot_state.rebuild_active_slots()
        return newly_finished

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
            print("[TORCH_PROFILE] role=target stopped", flush=True)

    @DynamicGradMode()
    def _event_loop(self) -> None:
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self.cancel_bubble_timer()
                continue

            batch, batch_kind = self._get_next_batch()
            tracking_batch = self._make_runtime_tracking_batch(batch)
            self.cur_batch = tracking_batch
            self.running_batch = (
                tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
            )

            if batch is not None:
                if batch_kind == "decode":
                    with self._torch_record("target_exact_decode_cycle"):
                        result = self.run_batch(batch)
                        self._process_decode_result(result)
                    self._torch_profile_step()
                else:
                    result = self.run_batch(batch)
                    self._process_prefill_result(batch, result)
            else:
                pass

            self.last_batch = tracking_batch
            if hasattr(self, "waiting_queue"):
                self.waiting_queue = []

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

        newly_finished = self._process_batch_result(
            next_token_ids=result.next_token_ids,
            logprob_diff=logprob_diff,
            bonus_ids=bonus_ids,
            rebuild_active=False,
        )

        plan = self.coordinator.collect_resample_jobs_batch(self.slot_state)
        did_resample = plan.n_jobs_sync() > 0
        if did_resample:
            self.coordinator.dispatch_resample_batch(plan, self.slot_state)
            # Mirror the exact dst<-src plan on the drafter's KV state.
            self._draft_client.send_commit(
                dst_slots=plan.dst_slots.tolist(),
                src_slots=plan.src_slots.tolist(),
            )

        if newly_finished or did_resample:
            self.slot_state.rebuild_active_slots()

        self._drain_finished_groups(self.slot_state.finished_mask.detach().cpu())

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
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
    )

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
