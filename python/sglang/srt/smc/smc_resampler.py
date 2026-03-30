from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set

import torch

from sglang.srt.managers.schedule_batch import Req, SMCGroupSpan, ScheduleBatch
from sglang.srt.smc.smc_debug_utils import (
    append_smc_diag_record,
    append_smc_probe_record,
)
from sglang.srt.smc.smc_manager import SMCFinishedParticleSnapshot
from sglang.srt.smc.smc_utils import (
    effective_sample_size,
    multinomial_resample,
    normalize_log_weights,
    systematic_resample,
)

# Lazy imports resolved at first use
_SamplingBatchInfo = None
_build_smc_group_spans = None
_SMCDraftInput = None


def _ensure_lazy_imports():
    global _SamplingBatchInfo, _build_smc_group_spans, _SMCDraftInput
    if _SamplingBatchInfo is None:
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo as _SBI
        from sglang.srt.managers.schedule_batch import (
            build_smc_group_spans as _build,
        )
        from sglang.srt.smc.smc_info import SMCDraftInput as _SDI
        _SamplingBatchInfo = _SBI
        _build_smc_group_spans = _build
        _SMCDraftInput = _SDI


@dataclass
class PendingResample:
    group_id: str
    dst_reqs: List[Req] = field(default_factory=list)
    src_snapshots: List[dict] = field(default_factory=list)
    inc_ref: List[torch.Tensor] = field(default_factory=list)
    dec_ref: List[torch.Tensor] = field(default_factory=list)
    done_event: Optional[torch.cuda.Event] = None


@dataclass
class BatchSlot:
    """One side of the ping-pong double buffer.

    Each slot owns an independent ScheduleBatch and tracks which SMC groups
    are assigned to it.  All particles of a group reside in a single slot
    (atomic group placement).
    """

    slot_id: int  # 0 or 1
    batch: Optional[ScheduleBatch] = None
    saved_batch: Optional[ScheduleBatch] = None
    saved_result: object = None  # GenerationBatchResult
    group_ids: Set[str] = field(default_factory=set)
    particle_reqs_by_group: Dict[str, List[Req]] = field(default_factory=dict)
    groups_needing_resample: Set[str] = field(default_factory=set)
    needs_rebuild: bool = False

    def add_group(self, group_id: str, reqs: List[Req]) -> None:
        self.group_ids.add(group_id)
        self.particle_reqs_by_group[group_id] = reqs
        self.needs_rebuild = True

    def remove_group(self, group_id: str) -> None:
        self.group_ids.discard(group_id)
        self.particle_reqs_by_group.pop(group_id, None)
        self.groups_needing_resample.discard(group_id)
        self.needs_rebuild = True

    def particle_count(self) -> int:
        return sum(len(v) for v in self.particle_reqs_by_group.values())

    def clear(self) -> None:
        self.batch = None
        self.saved_batch = None
        self.saved_result = None
        self.group_ids.clear()
        self.particle_reqs_by_group.clear()
        self.groups_needing_resample.clear()
        self.needs_rebuild = False


class SMCResampler:
    def __init__(self, smc_manager, device):
        self.smc_manager = smc_manager
        self.device = device
        self.device_module = None
        self.resample_stream = None
        self.wait_for_running: Deque[str] = deque()
        self._wait_for_running_members: Set[str] = set()
        self.resampling_reqs: Dict[str, List[Req]] = {}
        self.pending_resamples: Dict[str, PendingResample] = {}
        self._groups_needing_resample: Set[str] = set()

        # ── Ping-pong double-buffer state ──
        self.slots = (BatchSlot(slot_id=0), BatchSlot(slot_id=1))
        self.forward_slot_idx: int = 0
        self.pending_groups: Deque[str] = deque()
        self._pending_group_ids: Set[str] = set()
        self.pingpong_active: bool = False

    def init_streams(self, enable_overlap: bool) -> None:
        self.device_module = torch.get_device_module(self.device)
        if not enable_overlap or str(self.device) == "cpu":
            self.resample_stream = None
            return
        self.resample_stream = self.device_module.Stream(priority=0)

    def clear(self) -> None:
        self.wait_for_running.clear()
        self._wait_for_running_members.clear()
        self.resampling_reqs.clear()
        self.pending_resamples.clear()
        self._groups_needing_resample.clear()

    def enqueue_group_for_running(self, group_id: Optional[str]) -> None:
        if group_id is None:
            return
        # Ping-pong path: uses pending_groups queue
        self.enqueue_new_group(group_id)
        # Legacy path: uses wait_for_running queue (skip in ping-pong mode
        # since _drain_wait_for_running is never called there)
        if not self.pingpong_active:
            if group_id not in self._wait_for_running_members:
                self.wait_for_running.append(group_id)
                self._wait_for_running_members.add(group_id)

    def get_stalled_req_count(self) -> int:
        return sum(len(reqs) for reqs in self.resampling_reqs.values())

    def should_delay_admission(
        self,
        running_req_count: int,
        group_size: int,
    ) -> bool:
        stalled_req_count = self.get_stalled_req_count()
        if stalled_req_count <= 0:
            return False
        current_gap = abs(running_req_count - stalled_req_count)
        next_gap = abs((running_req_count + group_size) - stalled_req_count)
        return next_gap > current_gap

    def step(self, scheduler) -> None:
        self._sync_completed_resamples(scheduler)
        self._launch_pending_resamples(scheduler)
        self._drain_wait_for_running(scheduler)

    def step_before_forward(self, scheduler) -> None:
        """Sync completed resamples and merge groups back into running_batch.

        Must run BEFORE get_next_batch_to_run() so that resumed particles
        are visible to batch selection.
        """
        self._sync_completed_resamples(scheduler)
        # Stall and launch resamples before selecting the next batch. The
        # resample path trims overallocated tails and rewrites req_to_token
        # entries, so doing it after the next decode launch can mutate KV
        # reservations that the in-flight forward is still consuming.
        self._launch_pending_resamples(scheduler)
        self._drain_wait_for_running(scheduler)

    def step_after_forward(self, scheduler) -> None:
        """Resample launches happen in step_before_forward.

        Keeping the launch there guarantees stalled groups are removed from
        running_batch before batch selection, while the queued resample-stream
        work still overlaps with the forward that follows.
        """

    def finish_pending_before_idle(self, scheduler) -> bool:
        """Drain pending resamples before the scheduler declares itself idle.

        Pending resamples temporarily rewrite req_to_token rows before
        _complete_resample() applies the matching allocator refcount updates.
        If the overlap loop runs its idle memory check in that window, the KV
        pool looks artificially short. When there is no forward to overlap
        against, block here, complete the pending work, and let the next loop
        iteration observe the resumed requests.
        """
        had_pending_work = bool(self.pending_resamples or self._groups_needing_resample)
        if not had_pending_work:
            return False

        self._launch_pending_resamples(scheduler)
        for group_id, pending in list(self.pending_resamples.items()):
            if pending.done_event is not None:
                pending.done_event.synchronize()
            self._complete_resample(group_id, pending, scheduler)
        self._drain_wait_for_running(scheduler)
        return True

    def on_batch_done(
        self,
        reqs: List[Req],
        logprob_diffs: torch.Tensor,
        group_spans: Optional[List[SMCGroupSpan]] = None,
    ) -> List[Req]:
        if not reqs:
            return []

        if not torch.is_tensor(logprob_diffs):
            logprob_diffs = torch.as_tensor(logprob_diffs, dtype=torch.float32)

        if group_spans is not None:
            return self._on_batch_done_group_spans(reqs, logprob_diffs, group_spans)
        atomic_group_spans = self._collect_atomic_group_spans(reqs)
        if atomic_group_spans is not None:
            return self._on_batch_done_atomic_groups(
                reqs,
                logprob_diffs,
                atomic_group_spans,
            )
        return self._on_batch_done_grouped(reqs, logprob_diffs)

    def _collect_atomic_group_spans(
        self,
        reqs: List[Req],
    ) -> Optional[List[tuple[object, int, int]]]:
        spans: List[tuple[object, int, int]] = []
        seen_group_ids: Set[str] = set()
        start = 0
        while start < len(reqs):
            group_id = reqs[start].smc_group_id
            if group_id is None or group_id in seen_group_ids:
                return None

            end = start + 1
            while end < len(reqs) and reqs[end].smc_group_id == group_id:
                end += 1

            group = self.smc_manager.get_group(group_id)
            if group is None or len(group.active_particle_indices()) != end - start:
                return None

            spans.append((group, start, end))
            seen_group_ids.add(group_id)
            start = end

        return spans

    def _on_batch_done_group_spans(
        self,
        reqs: List[Req],
        logprob_diffs: torch.Tensor,
        group_spans: List[SMCGroupSpan],
    ) -> List[Req]:
        finalized_reqs: List[Req] = []
        for span in group_spans:
            group = self.smc_manager.get_group(span.group_id)
            if group is None:
                continue
            fin = self._update_group(
                group,
                reqs[span.start : span.end],
                logprob_diffs[span.start : span.end],
            )
            if fin is not None:
                finalized_reqs.append(fin)
        return finalized_reqs

    def _on_batch_done_atomic_groups(
        self,
        reqs: List[Req],
        logprob_diffs: torch.Tensor,
        atomic_group_spans: List[tuple[object, int, int]],
    ) -> List[Req]:
        finalized_reqs: List[Req] = []
        for group, start, end in atomic_group_spans:
            fin = self._update_group(
                group,
                reqs[start:end],
                logprob_diffs[start:end],
            )
            if fin is not None:
                finalized_reqs.append(fin)
        return finalized_reqs

    def _on_batch_done_grouped(
        self,
        reqs: List[Req],
        logprob_diffs: torch.Tensor,
    ) -> List[Req]:
        grouped_reqs: Dict[str, List[tuple[int, Req]]] = {}
        for row, req in enumerate(reqs):
            group_id = req.smc_group_id
            if group_id is None or self.smc_manager.get_group(group_id) is None:
                continue
            grouped_reqs.setdefault(group_id, []).append((row, req))

        finalized_reqs: List[Req] = []
        for group_id, entries in grouped_reqs.items():
            group = self.smc_manager.get_group(group_id)
            if group is None:
                continue

            row_indices = torch.tensor(
                [row for row, _ in entries],
                dtype=torch.int64,
                device=logprob_diffs.device,
            )
            fin = self._update_group(
                group,
                [req for _, req in entries],
                logprob_diffs.index_select(0, row_indices),
            )
            if fin is not None:
                finalized_reqs.append(fin)

        return finalized_reqs

    def _update_group(
        self,
        group,
        reqs: List[Req],
        logprob_diffs: torch.Tensor,
    ) -> Optional[Req]:
        """100% CPU — stash log_weight diffs, update step_counts, mark resample.

        The GPU log_weight add is deferred to ``_launch_pending_resamples``
        (overlapped with the next forward) or ``_finalize_group``.
        """
        # Stash diffs for deferred GPU application (pure Python, no torch ops)
        n = len(reqs)
        if n == 1:
            idx = reqs[0].smc_particle_idx
            group.pending_diffs.append((idx, logprob_diffs))
            group.step_counts[idx] += 1
        else:
            pidxs = [req.smc_particle_idx for req in reqs]
            group.pending_diffs.append((pidxs, logprob_diffs))
            for p in pidxs:
                group.step_counts[p] += 1

        append_smc_diag_record(
            {
                "type": "group_update",
                "group_id": group.group_id,
                "particle_indices": [req.smc_particle_idx for req in reqs],
                "logprob_diffs": [float(x) for x in logprob_diffs.tolist()],
                "step_counts": list(group.step_counts),
            }
        )

        if not group.all_active_aligned():
            return None

        active_indices = group.active_particle_indices()
        if not active_indices:
            return self.smc_manager._finalize_group(group.group_id)

        group.resampled_at_step = group.step_counts[active_indices[0]]
        # Always mark aligned multi-particle groups for resampling. Finished
        # particles keep participating in the SMC population, so a group with
        # one live particle and finished siblings can still need resampling.
        if len(group.particle_reqs) > 1:
            self._groups_needing_resample.add(group.group_id)
        return None

    def _launch_pending_resamples(self, scheduler) -> None:
        group_ids = list(self._groups_needing_resample)
        self._groups_needing_resample.clear()

        for group_id in group_ids:
            if group_id in self.pending_resamples:
                continue

            group = self.smc_manager.get_group(group_id)
            if group is None:
                continue

            active_indices = group.active_particle_indices()
            resample_indices = sorted(group.particle_reqs)
            if len(resample_indices) <= 1:
                group.flush_pending_diffs()
                continue

            # Apply deferred log_weight updates before sampling ancestors
            group.flush_pending_diffs()
            group_log_weights = group.log_weights[resample_indices]
            normalized_weights = normalize_log_weights(
                group_log_weights, device=self.device
            )
            ess = effective_sample_size(normalized_weights, device=self.device)
            append_smc_diag_record(
                {
                    "type": "resample_check",
                    "group_id": group.group_id,
                    "resample_indices": list(resample_indices),
                    "active_indices": list(active_indices),
                    "log_weights": [float(x) for x in group_log_weights.tolist()],
                    "normalized_weights": [
                        float(x) for x in normalized_weights.tolist()
                    ],
                    "ess": float(ess),
                }
            )
            if (
                ess
                >= len(resample_indices)
                * self.smc_manager.server_args.smc_resample_threshold
            ):
                continue

            ancestors = self._sample_ancestors(normalized_weights)
            append_smc_diag_record(
                {
                    "type": "resample_choice",
                    "group_id": group.group_id,
                    "resample_indices": list(resample_indices),
                    "active_indices": list(active_indices),
                    "ancestors": list(ancestors),
                }
            )
            evictions = [
                (dst_idx, resample_indices[src_pos])
                for dst_idx, src_pos in zip(resample_indices, ancestors, strict=True)
                if dst_idx != resample_indices[src_pos]
            ]
            if not evictions:
                continue

            active_tensor = torch.tensor(
                resample_indices,
                dtype=torch.int64,
                device=group.log_weights.device,
            )
            group.log_weights[active_tensor] = 0.0

            stalled_reqs = [group.particle_reqs[idx] for idx in active_indices]
            self._stall_group_reqs(group_id, stalled_reqs, scheduler)

            pending = PendingResample(group_id=group_id)
            if self.resample_stream is None:
                self._prepare_pending_resample(group, evictions, scheduler, pending)
                self.pending_resamples[group_id] = pending
                self._complete_resample(group_id, pending, scheduler)
                continue

            with self.device_module.stream(self.resample_stream):
                self._prepare_pending_resample(group, evictions, scheduler, pending)
                pending.done_event = self.device_module.Event()
                pending.done_event.record()

            self.pending_resamples[group_id] = pending

    def _sync_completed_resamples(self, scheduler) -> None:
        for group_id, pending in list(self.pending_resamples.items()):
            if pending.done_event is not None and not pending.done_event.query():
                continue
            self._complete_resample(group_id, pending, scheduler)

    def _complete_resample(self, group_id: str, pending: PendingResample, scheduler) -> None:
        if pending.done_event is not None:
            scheduler.schedule_stream.wait_event(pending.done_event)

        for indices in pending.inc_ref:
            scheduler.token_to_kv_pool_allocator.inc_ref(indices)
        for indices in pending.dec_ref:
            scheduler.token_to_kv_pool_allocator.dec_ref_and_free(indices)

        for dst_req, snapshot in zip(
            pending.dst_reqs,
            pending.src_snapshots,
            strict=True,
        ):
            self._restore_req_state(dst_req, snapshot)

        self.resampling_reqs.pop(group_id, None)
        del self.pending_resamples[group_id]

        group = self.smc_manager.get_group(group_id)
        if group is not None:
            for dst_req, snapshot in zip(
                pending.dst_reqs,
                pending.src_snapshots,
                strict=True,
            ):
                particle_idx = dst_req.smc_particle_idx
                if snapshot["finished_reason"] is None:
                    group.finished_particles.pop(particle_idx, None)
                else:
                    group.finished_particles[particle_idx] = (
                        SMCFinishedParticleSnapshot(
                            output_ids=list(snapshot["output_ids"]),
                            finished_reason=copy.copy(snapshot["finished_reason"]),
                            finished_len=snapshot["finished_len"],
                        )
                    )

            if not group.active_particle_indices():
                finalized_req = self.smc_manager._finalize_group(group_id)
                if finalized_req is not None:
                    time_stats = getattr(finalized_req, "time_stats", None)
                    if time_stats is not None:
                        time_stats.set_completion_time()
                    scheduler.stream_output([finalized_req], False)
                return

        self.enqueue_group_for_running(group_id)

    def _drain_wait_for_running(self, scheduler) -> None:
        while self.wait_for_running:
            group_id = self.wait_for_running[0]
            group = self.smc_manager.get_group(group_id)
            if group is None:
                self._pop_wait_for_running_head()
                continue

            active_reqs = self.smc_manager.get_active_particle_reqs(group_id)
            if not active_reqs:
                self._pop_wait_for_running_head()
                continue

            group_size = len(active_reqs)
            if self._remaining_req_capacity(scheduler) < group_size:
                break

            running_smc_req_count = self._running_smc_req_count(scheduler)
            if self.should_delay_admission(
                running_req_count=running_smc_req_count,
                group_size=group_size,
            ):
                break

            resumed_batch = self.smc_manager._build_particle_batch(
                active_reqs,
                scheduler,
                use_future_map=self._running_batch_uses_future_indices(
                    scheduler.running_batch
                ),
            )
            self._pop_wait_for_running_head()

            if scheduler.running_batch.is_empty():
                scheduler.running_batch = resumed_batch
            else:
                scheduler.running_batch.merge_batch(resumed_batch)

    def _pop_wait_for_running_head(self) -> Optional[str]:
        if not self.wait_for_running:
            return None
        group_id = self.wait_for_running.popleft()
        self._wait_for_running_members.discard(group_id)
        return group_id

    def _remaining_req_capacity(self, scheduler) -> int:
        max_req_count = getattr(
            getattr(scheduler, "server_args", None),
            "pp_max_micro_batch_size",
            None,
        )
        if max_req_count is None:
            max_req_count = getattr(scheduler, "max_running_requests", None)
        if max_req_count is None:
            return 1 << 30

        pending_prefill_reqs = 0
        last_batch = getattr(scheduler, "last_batch", None)
        if self._has_pending_last_batch(scheduler) and last_batch is not None and last_batch.forward_mode.is_extend():
            if hasattr(last_batch, "batch_size"):
                pending_prefill_reqs = last_batch.batch_size()
            else:
                pending_prefill_reqs = len(last_batch.reqs)

        return max(
            max_req_count - len(scheduler.running_batch.reqs) - pending_prefill_reqs,
            0,
        )

    def _running_smc_req_count(self, scheduler) -> int:
        running_smc_req_count = scheduler.running_batch.count_smc_particle_reqs()
        last_batch = getattr(scheduler, "last_batch", None)
        if (
            self._has_pending_last_batch(scheduler)
            and last_batch is not None
            and last_batch.forward_mode.is_extend()
            and getattr(last_batch, "spec_algorithm", None) is not None
            and last_batch.spec_algorithm.is_smc()
        ):
            if hasattr(last_batch, "count_smc_particle_reqs"):
                running_smc_req_count += last_batch.count_smc_particle_reqs()
            else:
                running_smc_req_count += sum(
                    1 for req in last_batch.reqs if req.smc_group_id is not None
                )
        return running_smc_req_count

    def _has_pending_last_batch(self, scheduler) -> bool:
        result_queue = getattr(scheduler, "result_queue", None)
        return result_queue is not None and len(result_queue) > 0

    def _sample_ancestors(self, normalized_weights: torch.Tensor) -> List[int]:
        if self.smc_manager.server_args.smc_resample_method == "multinomial":
            return multinomial_resample(normalized_weights, device=self.device)
        return systematic_resample(normalized_weights, device=self.device)

    def _stall_group_reqs(
        self,
        group_id: str,
        stalled_reqs: List[Req],
        scheduler,
    ) -> None:
        self._trim_stale_overalloc(stalled_reqs, scheduler)

        group_span = scheduler.running_batch.get_smc_group_span(group_id)
        if group_span is not None and group_span.size == len(stalled_reqs):
            keep_indices = list(range(group_span.start)) + list(
                range(group_span.end, len(scheduler.running_batch.reqs))
            )
        else:
            stalled_req_ids = {id(req) for req in stalled_reqs}
            keep_indices = [
                idx
                for idx, req in enumerate(scheduler.running_batch.reqs)
                if id(req) not in stalled_req_ids
            ]
            if len(keep_indices) + len(stalled_reqs) != len(scheduler.running_batch.reqs):
                raise RuntimeError(
                    f"SMC group {group_id} could not be isolated from running_batch for resampling."
                )

        scheduler.running_batch.filter_batch(keep_indices=keep_indices)
        scheduler.running_batch.batch_is_full = False
        if not keep_indices:
            scheduler.running_batch = ScheduleBatch(reqs=[])

        self.resampling_reqs[group_id] = stalled_reqs

    def _trim_stale_overalloc(self, reqs: List[Req], scheduler) -> None:
        for req in reqs:
            allocated_len = int(req.kv_allocated_len)
            if allocated_len <= req.kv_committed_len:
                continue
            indices_to_free = scheduler.req_to_token_pool.req_to_token[
                req.req_pool_idx,
                req.kv_committed_len:allocated_len,
            ].to(dtype=torch.int64, copy=True)
            scheduler.token_to_kv_pool_allocator.dec_ref_and_free(indices_to_free)
            req.kv_allocated_len = req.kv_committed_len

    def _prepare_pending_resample(
        self,
        group,
        evictions: List[tuple[int, int]],
        scheduler,
        pending: PendingResample,
    ) -> None:
        append_smc_probe_record(
            {
                "type": "resample_launch",
                "group_id": group.group_id,
                "active_particles": len(group.active_particle_indices()),
                "num_evictions": len(evictions),
            }
        )
        req_to_token = scheduler.req_to_token_pool.req_to_token
        staged_snapshots: Dict[int, dict] = {}
        staged_copies: Dict[int, torch.Tensor] = {}
        staged_actions: List[tuple[Req, int, int]] = []

        for dst_idx, src_idx in evictions:
            dst_req = group.particle_reqs[dst_idx]
            src_req = group.particle_reqs[src_idx]
            src_len = src_req.kv_committed_len

            dst_allocated_len = int(dst_req.kv_allocated_len)
            if dst_allocated_len > 0:
                pending.dec_ref.append(
                    req_to_token[
                        dst_req.req_pool_idx, :dst_allocated_len
                    ].to(dtype=torch.int64, copy=True)
                )

            if src_idx not in staged_snapshots:
                staged_snapshots[src_idx] = self._snapshot_req_state(src_req)
                if src_len > 0:
                    staged_copies[src_idx] = req_to_token[
                        src_req.req_pool_idx, :src_len
                    ].to(dtype=torch.int64, copy=True)
                else:
                    staged_copies[src_idx] = torch.empty(
                        (0,),
                        dtype=torch.int64,
                        device=self.device,
                    )

            staged_actions.append((dst_req, src_idx, src_len))

        for dst_req, src_idx, src_len in staged_actions:
            copied_indices = staged_copies[src_idx]
            if src_len > 0:
                scheduler.req_to_token_pool.write(
                    (dst_req.req_pool_idx, slice(0, src_len)),
                    copied_indices.to(dtype=torch.int32),
                )
                pending.inc_ref.append(copied_indices)

            pending.dst_reqs.append(dst_req)
            pending.src_snapshots.append(staged_snapshots[src_idx])

    def _snapshot_req_state(self, req: Req) -> dict:
        seq_len = req.kv_committed_len
        if seq_len > 0:
            indices = self.smc_manager.req_to_token_pool.req_to_token[
                req.req_pool_idx, :seq_len
            ].to(dtype=torch.int64, copy=True)
        else:
            indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        return {
            "indices": indices,
            "output_ids": list(req.output_ids),
            "finished_reason": copy.copy(req.finished_reason),
            "finished_len": req.finished_len,
            "finished_output": req.finished_output,
            "to_finish": copy.copy(req.to_finish),
            "kv_committed_len": req.kv_committed_len,
            "kv_allocated_len": req.kv_allocated_len,
            "cache_protected_len": req.cache_protected_len,
            "logprob_start_len": req.logprob_start_len,
            "decoded_text": req.decoded_text,
            "surr_offset": req.surr_offset,
            "read_offset": req.read_offset,
            "surr_and_decode_ids": (
                list(req.surr_and_decode_ids)
                if getattr(req, "surr_and_decode_ids", None) is not None
                else None
            ),
            "cur_decode_ids_len": getattr(req, "cur_decode_ids_len", None),
        }

    def _restore_req_state(self, req: Req, snapshot: dict) -> None:
        indices = snapshot["indices"]
        if indices.numel() > 0:
            req.prefix_indices = indices.to(dtype=torch.int64, copy=True)
        else:
            req.prefix_indices = torch.empty((0,), dtype=torch.int64, device=indices.device)

        req.output_ids = list(snapshot["output_ids"])
        req.finished_reason = copy.copy(snapshot["finished_reason"])
        req.finished_len = snapshot["finished_len"]
        req.finished_output = snapshot["finished_output"]
        req.to_finish = copy.copy(snapshot["to_finish"])
        req.kv_committed_len = snapshot["kv_committed_len"]
        req.kv_allocated_len = snapshot.get(
            "kv_allocated_len",
            snapshot["kv_committed_len"],
        )
        req.cache_protected_len = snapshot["cache_protected_len"]
        req.logprob_start_len = snapshot["logprob_start_len"]
        req.decoded_text = snapshot.get(
            "decoded_text",
            getattr(req, "decoded_text", ""),
        )
        req.surr_offset = snapshot.get("surr_offset", getattr(req, "surr_offset", None))
        req.read_offset = snapshot.get("read_offset", getattr(req, "read_offset", None))
        surr_and_decode_ids = snapshot.get("surr_and_decode_ids", None)
        req.surr_and_decode_ids = (
            list(surr_and_decode_ids) if surr_and_decode_ids is not None else None
        )
        req.cur_decode_ids_len = snapshot.get(
            "cur_decode_ids_len",
            getattr(req, "cur_decode_ids_len", None),
        )

    def _running_batch_uses_future_indices(self, running_batch) -> bool:
        if running_batch is None or running_batch.is_empty():
            return False
        spec_info = getattr(running_batch, "spec_info", None)
        return getattr(spec_info, "future_indices", None) is not None

    # ================================================================
    # Ping-pong double-buffer methods
    # ================================================================

    @property
    def forward_slot(self) -> BatchSlot:
        return self.slots[self.forward_slot_idx]

    @property
    def process_slot(self) -> BatchSlot:
        return self.slots[1 - self.forward_slot_idx]

    def swap(self) -> None:
        """Swap forward and process slots."""
        self.forward_slot_idx = 1 - self.forward_slot_idx

    def total_particle_count(self) -> int:
        """Total particles across both slots (for admission control)."""
        return self.slots[0].particle_count() + self.slots[1].particle_count()

    def slot_for_group(self, group_id: str) -> Optional[BatchSlot]:
        """Return the slot containing this group, or None."""
        for slot in self.slots:
            if group_id in slot.group_ids:
                return slot
        return None

    # ── Dispatch: assign new groups to slots ──

    def enqueue_new_group(self, group_id: Optional[str]) -> None:
        """Called by prefill result processing after create_group()."""
        if group_id is None or group_id in self._pending_group_ids:
            return
        self.pending_groups.append(group_id)
        self._pending_group_ids.add(group_id)

    def dispatch_pending_groups(self, scheduler) -> None:
        """Assign new groups to batch slots atomically (all particles in one slot).

        Load-balances by assigning each group to the slot with fewer particles.
        """
        while self.pending_groups:
            group_id = self.pending_groups[0]
            group = self.smc_manager.get_group(group_id)
            if group is None:
                self._pop_pending_group()
                continue

            active_reqs = self.smc_manager.get_active_particle_reqs(group_id)
            if not active_reqs:
                self._pop_pending_group()
                continue

            if self._pingpong_remaining_capacity(scheduler) < len(active_reqs):
                break

            target_slot = self._pick_lighter_slot()
            target_slot.add_group(group_id, active_reqs)
            self._pop_pending_group()

    def _pick_lighter_slot(self) -> BatchSlot:
        count_0 = self.slots[0].particle_count()
        count_1 = self.slots[1].particle_count()
        return self.slots[0] if count_0 <= count_1 else self.slots[1]

    def _pop_pending_group(self) -> None:
        group_id = self.pending_groups.popleft()
        self._pending_group_ids.discard(group_id)

    def _pingpong_remaining_capacity(self, scheduler) -> int:
        max_req_count = getattr(
            getattr(scheduler, "server_args", None),
            "pp_max_micro_batch_size",
            None,
        )
        if max_req_count is None:
            max_req_count = getattr(scheduler, "max_running_requests", None)
        if max_req_count is None:
            return 1 << 30
        return max(max_req_count - self.total_particle_count(), 0)

    # ── Batch lifecycle: refresh and rebuild ──

    def prepare_forward_batch(self, slot: BatchSlot, scheduler) -> None:
        """Prepare a slot's batch for the next forward pass."""
        if not slot.group_ids:
            slot.batch = None
            return
        if slot.needs_rebuild:
            self._rebuild_slot_batch(slot, scheduler)
        else:
            self._refresh_slot_batch(slot, scheduler)

    def _rebuild_slot_batch(self, slot: BatchSlot, scheduler) -> None:
        """Full rebuild: collect all active reqs from slot's groups, build fresh batch."""
        all_reqs = []
        for group_id in sorted(slot.group_ids):
            reqs = self.smc_manager.get_active_particle_reqs(group_id)
            all_reqs.extend(reqs)
            slot.particle_reqs_by_group[group_id] = reqs

        if not all_reqs:
            slot.batch = None
            slot.needs_rebuild = False
            return

        slot.batch = self.smc_manager._build_particle_batch(
            all_reqs,
            scheduler,
            use_future_map=False,  # ping-pong never uses future indices
        )
        slot.needs_rebuild = False

    def _refresh_slot_batch(self, slot: BatchSlot, scheduler) -> None:
        """Lightweight refresh: update tensors from Req state without rebuilding.

        Uses tensor replacement (not in-place mutation) to avoid aliasing
        hazards with saved_batch copies.
        """
        _ensure_lazy_imports()
        batch = slot.batch
        if batch is None or not batch.reqs:
            return

        refreshed_seq_lens_cpu = torch.tensor(
            [int(req.kv_committed_len) for req in batch.reqs],
            dtype=torch.int64,
        )
        refreshed_verified_ids = torch.tensor(
            [
                req.output_ids[-1]
                if req.output_ids
                else req.origin_input_ids[-1]
                for req in batch.reqs
            ],
            dtype=torch.int32,
            device=self.device,
        )
        batch.seq_lens_cpu = refreshed_seq_lens_cpu
        batch.seq_lens = refreshed_seq_lens_cpu.to(device=self.device)
        batch.seq_lens_sum = int(refreshed_seq_lens_cpu.sum().item())
        batch.orig_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        batch.output_ids = refreshed_verified_ids
        if isinstance(batch.spec_info, _SMCDraftInput):
            batch.spec_info.verified_id = refreshed_verified_ids
            batch.spec_info.new_seq_lens = batch.seq_lens
        batch.sampling_info = _SamplingBatchInfo.from_schedule_batch(
            batch, scheduler.model_config.vocab_size
        )
        batch.smc_group_spans = _build_smc_group_spans(batch.reqs)

    # ── Inline resampling (slot-local, on schedule_stream) ──

    def launch_resamples_for_slot(self, slot: BatchSlot, scheduler) -> None:
        """Run resampling for all groups marked in this slot.

        All operations happen on the schedule_stream.  No events, no deferred
        completion.  Only touches this slot's req_to_token rows (all particles
        of a group are in the same slot).
        """
        group_ids = list(slot.groups_needing_resample)
        slot.groups_needing_resample.clear()

        for group_id in group_ids:
            group = self.smc_manager.get_group(group_id)
            if group is None:
                continue

            resample_indices = sorted(group.particle_reqs)
            if len(resample_indices) <= 1:
                group.flush_pending_diffs()
                continue

            group.flush_pending_diffs()
            group_log_weights = group.log_weights[resample_indices]
            normalized_weights = normalize_log_weights(
                group_log_weights, device=self.device
            )
            ess = effective_sample_size(normalized_weights, device=self.device)

            append_smc_diag_record(
                {
                    "type": "resample_check",
                    "group_id": group.group_id,
                    "resample_indices": list(resample_indices),
                    "active_indices": list(group.active_particle_indices()),
                    "log_weights": [float(x) for x in group_log_weights.tolist()],
                    "normalized_weights": [
                        float(x) for x in normalized_weights.tolist()
                    ],
                    "ess": float(ess),
                }
            )

            if (
                ess
                >= len(resample_indices)
                * self.smc_manager.server_args.smc_resample_threshold
            ):
                continue

            ancestors = self._sample_ancestors(normalized_weights)
            append_smc_diag_record(
                {
                    "type": "resample_choice",
                    "group_id": group.group_id,
                    "resample_indices": list(resample_indices),
                    "active_indices": list(group.active_particle_indices()),
                    "ancestors": list(ancestors),
                }
            )

            evictions = [
                (dst_idx, resample_indices[src_pos])
                for dst_idx, src_pos in zip(resample_indices, ancestors, strict=True)
                if dst_idx != resample_indices[src_pos]
            ]
            if not evictions:
                continue

            active_tensor = torch.tensor(
                resample_indices,
                dtype=torch.int64,
                device=group.log_weights.device,
            )
            group.log_weights[active_tensor] = 0.0

            # Trim overallocated KV tails before rewriting block table
            active_reqs = [group.particle_reqs[idx] for idx in group.active_particle_indices()]
            self._trim_stale_overalloc(active_reqs, scheduler)

            # Inline resample: prepare + complete in one shot on schedule_stream
            pending = PendingResample(group_id=group_id)
            self._prepare_pending_resample(group, evictions, scheduler, pending)

            # Apply ref counts in a single free group
            scheduler.token_to_kv_pool_allocator.free_group_begin()
            for indices in pending.inc_ref:
                scheduler.token_to_kv_pool_allocator.inc_ref(indices)
            for indices in pending.dec_ref:
                scheduler.token_to_kv_pool_allocator.dec_ref_and_free(indices)

            # Restore req state
            for dst_req, snapshot in zip(
                pending.dst_reqs, pending.src_snapshots, strict=True
            ):
                self._restore_req_state(dst_req, snapshot)

            # Update finished_particles bookkeeping
            for dst_req, snapshot in zip(
                pending.dst_reqs, pending.src_snapshots, strict=True
            ):
                particle_idx = dst_req.smc_particle_idx
                if snapshot["finished_reason"] is None:
                    group.finished_particles.pop(particle_idx, None)
                else:
                    group.finished_particles[particle_idx] = (
                        SMCFinishedParticleSnapshot(
                            output_ids=list(snapshot["output_ids"]),
                            finished_reason=copy.copy(snapshot["finished_reason"]),
                            finished_len=snapshot["finished_len"],
                        )
                    )

            # Check if all particles are done after resample
            if not group.active_particle_indices():
                finalized_req = self.smc_manager._finalize_group(group_id)
                if finalized_req is not None:
                    time_stats = getattr(finalized_req, "time_stats", None)
                    if time_stats is not None:
                        time_stats.set_completion_time()
                    scheduler.stream_output([finalized_req], False)
                slot.remove_group(group_id)

            # Resampling modified req states (output_ids, kv_committed_len,
            # finished_reason, etc.) via _restore_req_state.  Force a full
            # rebuild so the batch picks up the new particle set from
            # get_active_particle_reqs instead of relying on the stale
            # batch.reqs list that _refresh_slot_batch would use.
            if evictions:
                slot.needs_rebuild = True

            scheduler.token_to_kv_pool_allocator.free_group_end()

    # ── Idle drain for ping-pong ──

    def finish_all_pending_pingpong(self, scheduler) -> bool:
        """Drain any pending work across both slots before going idle."""
        had_work = False
        for slot in self.slots:
            if slot.groups_needing_resample:
                had_work = True
                self.launch_resamples_for_slot(slot, scheduler)
        return had_work

    def clear_pingpong(self) -> None:
        """Reset all ping-pong state (includes legacy queues)."""
        for slot in self.slots:
            slot.clear()
        self.forward_slot_idx = 0
        self.pending_groups.clear()
        self._pending_group_ids.clear()
        # Clear legacy queues that are unused in pingpong but may accumulate
        self.wait_for_running.clear()
        self._wait_for_running_members.clear()
        self.resampling_reqs.clear()
        self.pending_resamples.clear()
        self._groups_needing_resample.clear()
