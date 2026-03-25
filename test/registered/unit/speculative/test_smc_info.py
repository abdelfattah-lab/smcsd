"""Unit tests for SMC helper state and resampling utilities."""

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.schedule_batch import build_smc_group_spans
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.smc_manager import (
    SMCFinishedParticleSnapshot,
    SMCGroupState,
    SMCManager,
)
from sglang.srt.speculative.smc_info import SMCDraftInput, SMCScoreInput
from sglang.srt.speculative.smc_info import (
    SMCParticleState,
    SMCRequestState,
    _release_internal_req,
    _release_smc_parent_req,
    effective_sample_size,
    get_smc_reserved_kv_len,
    multinomial_resample,
    normalize_log_weights,
    resolve_smc_proposal_length,
    resolve_smc_seed_output_ids,
    set_smc_reserved_kv_len,
    systematic_resample,
    validate_smc_parent_req,
)
from sglang.srt.speculative.smc_scheduler import PendingResample, SMCScheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


def _make_particle(output_len: int, log_weight: float, finished: bool = False):
    target_req = MagicMock()
    target_req.output_ids = list(range(output_len))
    target_req.finished.return_value = finished
    target_req.finished_reason = None
    target_req.finished_len = None

    draft_req = MagicMock()
    draft_req.output_ids = list(range(output_len))
    draft_req.finished.return_value = finished

    return SMCParticleState(
        target_req=target_req,
        draft_req=draft_req,
        log_weight=log_weight,
    )


def _make_scheduler_req(
    *,
    group_id: str,
    particle_idx: int,
    req_pool_idx: int,
    output_ids: list[int],
    kv_indices: list[int],
    allocated_kv_indices: list[int] | None = None,
):
    allocated_kv_indices = (
        list(allocated_kv_indices)
        if allocated_kv_indices is not None
        else list(kv_indices)
    )
    return SimpleNamespace(
        smc_group_id=group_id,
        smc_particle_idx=particle_idx,
        req_pool_idx=req_pool_idx,
        origin_input_ids=[1, 2],
        output_ids=list(output_ids),
        kv_committed_len=len(kv_indices),
        kv_allocated_len=len(allocated_kv_indices),
        cache_protected_len=len(kv_indices),
        logprob_start_len=0,
        draft_prefix_materialized=True,
        prefix_indices=torch.tensor(kv_indices, dtype=torch.int64),
        finished_reason=None,
        finished_len=None,
        finished_output=None,
        to_finish=None,
        finished=lambda: False,
    )


class _FakeAllocator:
    def __init__(self):
        self.inc_calls = []
        self.dec_calls = []
        self.ops = []

    def inc_ref(self, indices):
        cloned = indices.clone()
        self.inc_calls.append(cloned)
        self.ops.append(("inc", cloned))

    def dec_ref_and_free(self, indices):
        cloned = indices.clone()
        self.dec_calls.append(cloned)
        self.ops.append(("dec", cloned))


class _FakeRunningBatch:
    def __init__(self, reqs, future_indices=None, batch_is_full=False):
        self.reqs = list(reqs)
        self.smc_group_spans = build_smc_group_spans(self.reqs)
        self.batch_is_full = batch_is_full
        self.spec_info = SimpleNamespace(future_indices=future_indices)

    def is_empty(self):
        return len(self.reqs) == 0

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self, keep_indices=None, **kwargs):
        keep_indices = keep_indices or []
        self.reqs = [self.reqs[i] for i in keep_indices]
        self.smc_group_spans = build_smc_group_spans(self.reqs)

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)
        self.smc_group_spans = build_smc_group_spans(self.reqs)

    def get_smc_group_span(self, group_id):
        if self.smc_group_spans is None:
            return None
        for span in self.smc_group_spans:
            if span.group_id == group_id:
                return span
        return None

    def count_smc_particle_reqs(self):
        if self.smc_group_spans is None:
            return sum(1 for req in self.reqs if req.smc_group_id is not None)
        return sum(span.size for span in self.smc_group_spans)


class _FakeOutputProcessor(SchedulerOutputProcessorMixin):
    pass


class TestSMCWeightHelpers(TestCase):
    def test_normalize_log_weights(self):
        normalized = normalize_log_weights([0.0, 0.0, 0.0])
        self.assertTrue(
            torch.allclose(normalized, torch.full((3,), 1.0 / 3.0, dtype=torch.float64))
        )

    def test_effective_sample_size(self):
        self.assertAlmostEqual(effective_sample_size([0.5, 0.5]), 2.0)
        self.assertAlmostEqual(effective_sample_size([1.0, 0.0]), 1.0)

    def test_systematic_resample_with_degenerate_weight(self):
        self.assertEqual(systematic_resample([1.0, 0.0, 0.0]), [0, 0, 0])

    def test_multinomial_resample_with_degenerate_weight(self):
        self.assertEqual(multinomial_resample([1.0, 0.0, 0.0]), [0, 0, 0])


class TestSMCManagerHelpers(TestCase):
    def test_group_queries_use_active_particles_only(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=0.5, smc_resample_method="systematic")
        )
        req0 = SimpleNamespace(
            smc_group_id="g1",
            smc_particle_idx=0,
            finished=lambda: False,
        )
        req1 = SimpleNamespace(
            smc_group_id="g1",
            smc_particle_idx=1,
            finished=lambda: False,
        )
        req2 = SimpleNamespace(
            smc_group_id="g1",
            smc_particle_idx=2,
            finished=lambda: False,
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1, 2: req2},
            log_weights=torch.zeros(3, dtype=torch.float64),
            step_counts=[3, 1, 5],
            finished_particles={
                2: SMCFinishedParticleSnapshot(
                    output_ids=[1, 2],
                    finished_reason=None,
                    finished_len=2,
                )
            },
        )

        self.assertEqual(manager.get_particle_lag(req0), 0)
        self.assertEqual(manager.get_particle_lag(req1), 2)
        self.assertEqual(manager.get_group_lag("g1"), 2)
        self.assertEqual(manager.get_active_particle_reqs("g1"), [req0, req1])
        self.assertEqual(
            manager.get_active_particle_reqs_in_collection("g1", [req1, req2]),
            [req1],
        )
        self.assertTrue(manager.all_active_members_present("g1", [req0, req1, req2]))
        self.assertFalse(manager.all_active_members_present("g1", [req0]))

    def test_release_internal_req_frees_reserved_tail_when_visible_len_shrinks(self):
        req = SimpleNamespace(
            req_pool_idx=0,
            kv_committed_len=1,
            kv_allocated_len=1,
            prefix_indices=torch.tensor([11], dtype=torch.int64),
        )
        set_smc_reserved_kv_len(req, 2)
        req_to_token = torch.tensor([[11, 99, 0]], dtype=torch.int32)
        allocator = _FakeAllocator()
        req_to_token_pool = SimpleNamespace(
            req_to_token=req_to_token,
            free=lambda target_req: setattr(target_req, "req_pool_idx", None),
        )

        _release_internal_req(req, req_to_token_pool, allocator)

        self.assertEqual(len(allocator.dec_calls), 1)
        self.assertTrue(
            torch.equal(allocator.dec_calls[0], torch.tensor([11, 99], dtype=torch.int64))
        )
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(req.kv_allocated_len, 0)
        self.assertEqual(get_smc_reserved_kv_len(req), 0)


class TestSMCReleaseHelpers(TestCase):
    @patch("sglang.srt.speculative.smc_info.get_global_server_args")
    def test_release_smc_parent_req_dec_refs_non_protected_committed_kv(
        self,
        mock_get_global_server_args,
    ):
        mock_get_global_server_args.return_value = SimpleNamespace(page_size=1)

        req = SimpleNamespace(
            req_pool_idx=0,
            cache_protected_len=2,
            last_node="node-1",
            pop_committed_kv_cache=lambda: 4,
            pop_overallocated_kv_cache=lambda: (4, 4),
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.int32),
            free=MagicMock(side_effect=lambda released_req: setattr(released_req, "req_pool_idx", None)),
        )
        allocator = _FakeAllocator()
        tree_cache = SimpleNamespace(dec_lock_ref=MagicMock())

        _release_smc_parent_req(
            req,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertEqual(req.req_pool_idx, None)
        self.assertEqual(len(allocator.dec_calls), 1)
        self.assertTrue(
            torch.equal(allocator.dec_calls[0], torch.tensor([13, 14], dtype=torch.int64))
        )
        req_to_token_pool.free.assert_called_once_with(req)
        tree_cache.dec_lock_ref.assert_called_once_with("node-1")


class TestSMCScheduler(TestCase):
    def test_build_smc_group_spans_invalidates_non_contiguous_group_layout(self):
        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[101],
        )
        req1 = _make_scheduler_req(
            group_id="g2",
            particle_idx=0,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        req2 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=2,
            output_ids=[30],
            kv_indices=[301],
        )

        self.assertIsNone(build_smc_group_spans([req0, req1, req2]))

    def test_should_delay_admission_is_disabled_without_stalled_bucket(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")

        self.assertFalse(scheduler.should_delay_admission(running_req_count=0, group_size=4))
        self.assertFalse(scheduler.should_delay_admission(running_req_count=8, group_size=4))

    def test_should_delay_admission_when_it_worsens_bucket_balance(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")
        scheduler.resampling_reqs["g1"] = [
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
        ]

        self.assertFalse(scheduler.should_delay_admission(running_req_count=0, group_size=4))
        self.assertFalse(scheduler.should_delay_admission(running_req_count=4, group_size=0))
        self.assertTrue(scheduler.should_delay_admission(running_req_count=4, group_size=4))
        self.assertTrue(scheduler.should_delay_admission(running_req_count=8, group_size=4))

    def test_complete_resample_waits_for_done_event_and_enqueues_group_for_running(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[101],
        )
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        order = []
        done_event = object()
        allocator = SimpleNamespace(
            inc_ref=lambda indices: order.append(("inc", indices.clone())),
            dec_ref_and_free=lambda indices: order.append(("dec", indices.clone())),
        )
        other_req = SimpleNamespace(rid="other")
        live_scheduler = SimpleNamespace(
            schedule_stream=SimpleNamespace(
                wait_event=lambda event: order.append(("wait", event))
            ),
            running_batch=_FakeRunningBatch([other_req]),
            token_to_kv_pool_allocator=allocator,
        )
        scheduler.resampling_reqs["g1"] = [req0, req1]

        pending = PendingResample(
            group_id="g1",
            dst_reqs=[req1],
            src_snapshots=[
                {
                    "indices": torch.tensor([101], dtype=torch.int64),
                    "output_ids": [10],
                    "finished_reason": None,
                    "finished_len": None,
                    "finished_output": None,
                    "to_finish": None,
                    "kv_committed_len": 1,
                    "cache_protected_len": 1,
                    "logprob_start_len": 0,
                    "draft_prefix_materialized": True,
                }
            ],
            inc_ref=[torch.tensor([101], dtype=torch.int64)],
            dec_ref=[torch.tensor([201], dtype=torch.int64)],
            done_event=done_event,
        )
        scheduler.pending_resamples["g1"] = pending

        scheduler._complete_resample("g1", pending, live_scheduler)

        self.assertEqual(order[0], ("wait", done_event))
        self.assertEqual([op for op, _ in order[1:3]], ["inc", "dec"])
        self.assertEqual(live_scheduler.running_batch.reqs, [other_req])
        self.assertEqual(list(scheduler.wait_for_running), ["g1"])
        self.assertEqual(scheduler._wait_for_running_members, {"g1"})
        self.assertNotIn("g1", scheduler.resampling_reqs)

    def test_finish_pending_before_idle_blocks_until_resamples_complete(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")

        done_event = MagicMock()
        pending = PendingResample(group_id="g1", done_event=done_event)
        scheduler.pending_resamples["g1"] = pending
        scheduler._launch_pending_resamples = MagicMock()
        scheduler._complete_resample = MagicMock(
            side_effect=lambda group_id, *_args: scheduler.pending_resamples.pop(
                group_id, None
            )
        )
        scheduler._drain_wait_for_running = MagicMock()
        live_scheduler = SimpleNamespace()

        self.assertTrue(scheduler.finish_pending_before_idle(live_scheduler))
        scheduler._launch_pending_resamples.assert_called_once_with(live_scheduler)
        done_event.synchronize.assert_called_once_with()
        scheduler._complete_resample.assert_called_once_with(
            "g1", pending, live_scheduler
        )
        scheduler._drain_wait_for_running.assert_called_once_with(live_scheduler)

    def test_event_loop_overlap_clears_last_batch_before_idle_continue(self):
        class _StopLoop(Exception):
            pass

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )

        copied_batch = SimpleNamespace(tag="copied-batch")
        decode_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: True,
                is_extend=lambda: False,
            ),
            spec_algorithm=SimpleNamespace(is_smc=lambda: True),
            copy=lambda: copied_batch,
        )
        smc_scheduler = SimpleNamespace()
        smc_scheduler.step_before_forward = MagicMock()
        smc_scheduler.step_after_forward = MagicMock()
        smc_scheduler.finish_pending_before_idle = MagicMock(
            side_effect=[False, True, False]
        )

        recv_count = 0

        def _recv_requests():
            nonlocal recv_count
            recv_count += 1
            if recv_count >= 4:
                raise _StopLoop()
            return []

        next_batches = deque([decode_batch, None, None])
        processed = []
        live_scheduler = SimpleNamespace(
            _engine_paused=False,
            last_batch=None,
            cur_batch=None,
            is_generation=False,
            smc_scheduler=smc_scheduler,
            recv_requests=_recv_requests,
            process_input_requests=lambda _reqs: None,
            get_next_batch_to_run=lambda: next_batches.popleft(),
            is_disable_overlap_for_batch=lambda _batch: False,
            run_batch=lambda _batch: "batch-result",
            cancel_bubble_timer=lambda: None,
            process_batch_result=lambda batch, result: processed.append((batch, result)),
            launch_batch_sample_if_needed=lambda _batch_result: None,
            self_check_during_idle=lambda: None,
            result_queue=deque(),
        )

        with self.assertRaises(_StopLoop):
            Scheduler.event_loop_overlap(live_scheduler)

        self.assertEqual(processed, [(copied_batch, "batch-result")])
        self.assertIsNone(live_scheduler.last_batch)

    def test_drain_wait_for_running_drops_stale_head_and_admits_next_group(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")

        req0 = _make_scheduler_req(
            group_id="g2",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[101],
        )
        req1 = _make_scheduler_req(
            group_id="g2",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        manager.groups["g2"] = SMCGroupState(
            group_id="g2",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        scheduler.enqueue_group_for_running("stale")
        scheduler.enqueue_group_for_running("g2")
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch([]),
            server_args=SimpleNamespace(pp_max_micro_batch_size=8),
        )
        rebuilt_batch = SimpleNamespace(reqs=[req0, req1], is_empty=lambda: False)
        manager._build_particle_batch = MagicMock(return_value=rebuilt_batch)

        scheduler._drain_wait_for_running(live_scheduler)

        self.assertIs(live_scheduler.running_batch, rebuilt_batch)
        self.assertFalse(scheduler.wait_for_running)
        self.assertFalse(scheduler._wait_for_running_members)
        manager._build_particle_batch.assert_called_once_with(
            [req0, req1],
            live_scheduler,
            use_future_map=False,
        )

    def test_drain_wait_for_running_respects_fifo_when_balance_blocks_head(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")

        blocked_req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[101],
        )
        blocked_req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        trailing_req0 = _make_scheduler_req(
            group_id="g2",
            particle_idx=0,
            req_pool_idx=2,
            output_ids=[30],
            kv_indices=[301],
        )
        trailing_req1 = _make_scheduler_req(
            group_id="g2",
            particle_idx=1,
            req_pool_idx=3,
            output_ids=[40],
            kv_indices=[401],
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: blocked_req0, 1: blocked_req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )
        manager.groups["g2"] = SMCGroupState(
            group_id="g2",
            parent_req=SimpleNamespace(),
            particle_reqs={0: trailing_req0, 1: trailing_req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        scheduler.resampling_reqs["stalled"] = [SimpleNamespace(), SimpleNamespace()]
        scheduler.enqueue_group_for_running("g1")
        scheduler.enqueue_group_for_running("g2")
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch(
                [
                    _make_scheduler_req(
                        group_id="running",
                        particle_idx=0,
                        req_pool_idx=4,
                        output_ids=[50],
                        kv_indices=[501],
                    ),
                    _make_scheduler_req(
                        group_id="running",
                        particle_idx=1,
                        req_pool_idx=5,
                        output_ids=[60],
                        kv_indices=[601],
                    ),
                ]
            ),
            server_args=SimpleNamespace(pp_max_micro_batch_size=8),
        )
        manager._build_particle_batch = MagicMock()

        scheduler._drain_wait_for_running(live_scheduler)

        self.assertEqual([req.smc_group_id for req in live_scheduler.running_batch.reqs], ["running", "running"])
        self.assertEqual(list(scheduler.wait_for_running), ["g1", "g2"])
        manager._build_particle_batch.assert_not_called()

    def test_running_smc_req_count_ignores_processed_last_batch_without_overlap_queue(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch(
                [
                    _make_scheduler_req(
                        group_id="running",
                        particle_idx=0,
                        req_pool_idx=0,
                        output_ids=[10],
                        kv_indices=[101],
                    )
                ]
            ),
            last_batch=SimpleNamespace(
                reqs=[
                    _make_scheduler_req(
                        group_id="last",
                        particle_idx=0,
                        req_pool_idx=1,
                        output_ids=[20],
                        kv_indices=[201],
                    )
                ],
                forward_mode=SimpleNamespace(is_extend=lambda: True),
                spec_algorithm=SimpleNamespace(is_smc=lambda: True),
                count_smc_particle_reqs=lambda: 1,
            ),
            result_queue=[],
        )

        self.assertEqual(scheduler._running_smc_req_count(live_scheduler), 1)
        self.assertEqual(scheduler._remaining_req_capacity(live_scheduler), 1 << 30)

    def test_running_smc_req_count_includes_pending_overlap_last_batch(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch(
                [
                    _make_scheduler_req(
                        group_id="running",
                        particle_idx=0,
                        req_pool_idx=0,
                        output_ids=[10],
                        kv_indices=[101],
                    )
                ]
            ),
            last_batch=SimpleNamespace(
                reqs=[
                    _make_scheduler_req(
                        group_id="last",
                        particle_idx=0,
                        req_pool_idx=1,
                        output_ids=[20],
                        kv_indices=[201],
                    )
                ],
                forward_mode=SimpleNamespace(is_extend=lambda: True),
                spec_algorithm=SimpleNamespace(is_smc=lambda: True),
                count_smc_particle_reqs=lambda: 1,
                batch_size=lambda: 1,
            ),
            result_queue=[object()],
            server_args=SimpleNamespace(pp_max_micro_batch_size=3),
        )

        self.assertEqual(scheduler._running_smc_req_count(live_scheduler), 2)
        self.assertEqual(scheduler._remaining_req_capacity(live_scheduler), 1)

    def test_on_batch_done_uses_atomic_group_fast_path_for_contiguous_groups(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=0.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[101],
        )
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        req2 = _make_scheduler_req(
            group_id="g2",
            particle_idx=0,
            req_pool_idx=2,
            output_ids=[30],
            kv_indices=[301],
        )
        req3 = _make_scheduler_req(
            group_id="g2",
            particle_idx=1,
            req_pool_idx=3,
            output_ids=[40],
            kv_indices=[401],
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )
        manager.groups["g2"] = SMCGroupState(
            group_id="g2",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req2, 1: req3},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        with patch.object(
            scheduler,
            "_on_batch_done_grouped",
            wraps=scheduler._on_batch_done_grouped,
        ) as mock_grouped:
            finalized = scheduler.on_batch_done(
                [req0, req1, req2, req3],
                torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
            )

        self.assertEqual(finalized, [])
        mock_grouped.assert_not_called()
        # log_weight updates are deferred; flush to verify correctness
        manager.groups["g1"].flush_pending_diffs()
        manager.groups["g2"].flush_pending_diffs()
        self.assertTrue(
            torch.allclose(
                manager.groups["g1"].log_weights,
                torch.tensor([0.1, 0.2], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.allclose(
                manager.groups["g2"].log_weights,
                torch.tensor([0.3, 0.4], dtype=torch.float64),
            )
        )
        self.assertEqual(manager.groups["g1"].step_counts, [1, 1])
        self.assertEqual(manager.groups["g2"].step_counts, [1, 1])
        # Always-resample: aligned groups with >1 active particle are marked
        self.assertEqual(scheduler._groups_needing_resample, {"g1", "g2"})

    @patch("sglang.srt.speculative.smc_scheduler.systematic_resample")
    def test_step_resamples_and_reinserts_group_with_matching_future_mode(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = [0, 0]

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")
        scheduler.init_streams(enable_overlap=False)

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10, 11],
            kv_indices=[101, 102],
        )
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        req_to_token = torch.tensor(
            [
                [101, 102, 0, 0],
                [201, 0, 0, 0],
                [301, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        allocator = _FakeAllocator()
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch(
                [req0, req1, SimpleNamespace(rid="other")],
                future_indices=SimpleNamespace(indices=torch.tensor([1])),
                batch_is_full=True,
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=req_to_token,
                write=lambda indices, values: req_to_token.__setitem__(indices, values),
            ),
            token_to_kv_pool_allocator=allocator,
        )
        manager.req_to_token_pool = live_scheduler.req_to_token_pool
        rebuilt_batch = SimpleNamespace(reqs=[req0, req1])
        manager._build_particle_batch = MagicMock(return_value=rebuilt_batch)

        finalized = scheduler.on_batch_done(
            [req0, req1],
            torch.tensor([9.0, 0.0], dtype=torch.float32),
        )
        self.assertEqual(finalized, [])

        scheduler.step(live_scheduler)

        self.assertEqual(live_scheduler.running_batch.reqs[0].rid, "other")
        self.assertEqual(live_scheduler.running_batch.reqs[1:], [req0, req1])
        self.assertEqual(req1.output_ids, req0.output_ids)
        self.assertTrue(
            torch.equal(
                req_to_token[req1.req_pool_idx, : req0.kv_committed_len],
                req_to_token[req0.req_pool_idx, : req0.kv_committed_len],
            )
        )
        self.assertEqual(len(allocator.dec_calls), 1)
        self.assertEqual(len(allocator.inc_calls), 1)
        self.assertTrue(
            torch.equal(allocator.dec_calls[0], torch.tensor([201], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(allocator.inc_calls[0], torch.tensor([101, 102], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                manager.groups["g1"].log_weights,
                torch.zeros(2, dtype=torch.float64),
            )
        )
        manager._build_particle_batch.assert_called_once_with(
            [req0, req1],
            live_scheduler,
            use_future_map=True,
        )

    @patch("sglang.srt.speculative.smc_scheduler.systematic_resample")
    def test_step_before_forward_stalls_group_before_next_launch(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = [0, 0]

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")

        class _FakePendingEvent:
            def record(self):
                return None

            def query(self):
                return False

        scheduler.resample_stream = object()
        scheduler.device_module = SimpleNamespace(
            stream=lambda _stream: nullcontext(),
            Event=lambda: _FakePendingEvent(),
        )

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10, 11],
            kv_indices=[101, 102],
        )
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        other_req = SimpleNamespace(rid="other", smc_group_id=None)
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        req_to_token = torch.tensor(
            [[101, 102, 0, 0], [201, 0, 0, 0], [301, 0, 0, 0]],
            dtype=torch.int32,
        )
        allocator = _FakeAllocator()
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch([req0, req1, other_req], batch_is_full=True),
            req_to_token_pool=SimpleNamespace(
                req_to_token=req_to_token,
                write=lambda indices, values: req_to_token.__setitem__(indices, values),
            ),
            token_to_kv_pool_allocator=allocator,
        )
        manager.req_to_token_pool = live_scheduler.req_to_token_pool
        manager._build_particle_batch = MagicMock()

        scheduler.on_batch_done(
            [req0, req1],
            torch.tensor([9.0, 0.0], dtype=torch.float32),
        )
        scheduler.step_before_forward(live_scheduler)

        self.assertEqual(live_scheduler.running_batch.reqs, [other_req])
        self.assertEqual(scheduler.resampling_reqs["g1"], [req0, req1])
        self.assertIn("g1", scheduler.pending_resamples)
        manager._build_particle_batch.assert_not_called()

    @patch("sglang.srt.speculative.smc_scheduler.systematic_resample")
    def test_step_skips_stall_when_resample_has_no_evictions(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = [0, 1]

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")
        scheduler.init_streams(enable_overlap=False)

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[101],
        )
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch([req0, req1]),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor(
                    [[101, 0, 0], [201, 0, 0]],
                    dtype=torch.int32,
                ),
                write=MagicMock(),
            ),
            token_to_kv_pool_allocator=_FakeAllocator(),
        )
        manager.req_to_token_pool = live_scheduler.req_to_token_pool
        manager._build_particle_batch = MagicMock()

        scheduler.on_batch_done(
            [req0, req1],
            torch.tensor([9.0, 0.0], dtype=torch.float32),
        )
        scheduler.step(live_scheduler)

        self.assertEqual(live_scheduler.running_batch.reqs, [req0, req1])
        self.assertFalse(scheduler.resampling_reqs)
        self.assertFalse(scheduler.pending_resamples)
        manager._build_particle_batch.assert_not_called()
        self.assertTrue(
            torch.equal(
                manager.groups["g1"].log_weights,
                torch.zeros(2, dtype=torch.float64),
            )
        )

    @patch("sglang.srt.speculative.smc_scheduler.systematic_resample")
    def test_step_skips_resample_when_ess_stays_above_threshold(
        self,
        mock_systematic_resample,
    ):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=0.1, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")
        scheduler.init_streams(enable_overlap=False)

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[101],
        )
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch([req0, req1]),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor(
                    [[101, 0, 0], [201, 0, 0]],
                    dtype=torch.int32,
                ),
                write=MagicMock(),
            ),
            token_to_kv_pool_allocator=_FakeAllocator(),
        )
        manager.req_to_token_pool = live_scheduler.req_to_token_pool

        scheduler.on_batch_done(
            [req0, req1],
            torch.tensor([9.0, 0.0], dtype=torch.float32),
        )
        scheduler.step(live_scheduler)

        mock_systematic_resample.assert_not_called()
        self.assertEqual(live_scheduler.running_batch.reqs, [req0, req1])
        self.assertFalse(scheduler.resampling_reqs)
        self.assertFalse(scheduler.pending_resamples)
        self.assertTrue(
            torch.equal(
                manager.groups["g1"].log_weights,
                torch.tensor([9.0, 0.0], dtype=torch.float64),
            )
        )

    @patch("sglang.srt.speculative.smc_scheduler.systematic_resample")
    def test_step_replaces_empty_running_batch_before_reinsert(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = [0, 0]

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")
        scheduler.init_streams(enable_overlap=False)

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[101],
        )
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        req_to_token = torch.tensor([[101, 0], [201, 0]], dtype=torch.int32)
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch([req0, req1], batch_is_full=True),
            req_to_token_pool=SimpleNamespace(
                req_to_token=req_to_token,
                write=lambda indices, values: req_to_token.__setitem__(indices, values),
            ),
            token_to_kv_pool_allocator=_FakeAllocator(),
        )
        manager.req_to_token_pool = live_scheduler.req_to_token_pool
        rebuilt_batch = SimpleNamespace(reqs=[req0, req1], is_empty=lambda: False)
        manager._build_particle_batch = MagicMock(return_value=rebuilt_batch)

        scheduler.on_batch_done(
            [req0, req1],
            torch.tensor([9.0, 0.0], dtype=torch.float32),
        )
        scheduler.step(live_scheduler)

        self.assertIs(live_scheduler.running_batch, rebuilt_batch)
        manager._build_particle_batch.assert_called_once_with(
            [req0, req1],
            live_scheduler,
            use_future_map=False,
        )

    @patch("sglang.srt.speculative.smc_scheduler.systematic_resample")
    def test_step_snapshots_resample_sources_before_destination_writes(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = [1, 0]

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")
        scheduler.init_streams(enable_overlap=False)

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10, 11],
            kv_indices=[101, 102],
        )
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20, 21],
            kv_indices=[201, 202],
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        req_to_token = torch.tensor(
            [[101, 102, 0], [201, 202, 0]],
            dtype=torch.int32,
        )
        allocator = _FakeAllocator()
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch([req0, req1]),
            req_to_token_pool=SimpleNamespace(
                req_to_token=req_to_token,
                write=lambda indices, values: req_to_token.__setitem__(indices, values),
            ),
            token_to_kv_pool_allocator=allocator,
        )
        manager.req_to_token_pool = live_scheduler.req_to_token_pool
        manager._build_particle_batch = MagicMock(
            return_value=SimpleNamespace(reqs=[req0, req1], is_empty=lambda: False)
        )

        scheduler.on_batch_done(
            [req0, req1],
            torch.tensor([9.0, 0.0], dtype=torch.float32),
        )
        scheduler.step(live_scheduler)

        self.assertEqual(req0.output_ids, [20, 21])
        self.assertEqual(req1.output_ids, [10, 11])
        self.assertTrue(
            torch.equal(
                req_to_token[0, :2],
                torch.tensor([201, 202], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                req_to_token[1, :2],
                torch.tensor([101, 102], dtype=torch.int32),
            )
        )
        self.assertEqual(len(allocator.dec_calls), 2)
        self.assertEqual(len(allocator.inc_calls), 2)
        self.assertEqual([op for op, _ in allocator.ops[:2]], ["inc", "inc"])

    @patch("sglang.srt.speculative.smc_scheduler.systematic_resample")
    def test_step_trims_stale_overalloc_before_reinsert(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = [0, 0]

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")
        scheduler.init_streams(enable_overlap=False)

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[101],
            allocated_kv_indices=[101, 111],
        )
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
            allocated_kv_indices=[201, 211],
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        req_to_token = torch.tensor([[101, 111, 0], [201, 211, 0]], dtype=torch.int32)
        allocator = _FakeAllocator()
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch([req0, req1]),
            req_to_token_pool=SimpleNamespace(
                req_to_token=req_to_token,
                write=lambda indices, values: req_to_token.__setitem__(indices, values),
            ),
            token_to_kv_pool_allocator=allocator,
        )
        manager.req_to_token_pool = live_scheduler.req_to_token_pool
        manager._build_particle_batch = MagicMock(return_value=SimpleNamespace(reqs=[req0, req1]))

        scheduler.on_batch_done(
            [req0, req1],
            torch.tensor([9.0, 0.0], dtype=torch.float32),
        )
        scheduler.step(live_scheduler)

        self.assertEqual(req0.kv_allocated_len, req0.kv_committed_len)
        self.assertEqual(req1.kv_allocated_len, req1.kv_committed_len)
        self.assertEqual(len(allocator.dec_calls), 3)
        self.assertTrue(
            torch.equal(allocator.dec_calls[0], torch.tensor([111], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(allocator.dec_calls[1], torch.tensor([211], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(allocator.dec_calls[2], torch.tensor([201], dtype=torch.int64))
        )

    @patch("sglang.srt.speculative.smc_scheduler.systematic_resample")
    def test_step_trims_hidden_reserved_tail_before_reinsert(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = [0, 0]

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCScheduler(manager, device="cpu")
        scheduler.init_streams(enable_overlap=False)

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[101],
        )
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        set_smc_reserved_kv_len(req0, 2)
        set_smc_reserved_kv_len(req1, 2)
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=SimpleNamespace(),
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.zeros(2, dtype=torch.float64),
            step_counts=[0, 0],
        )

        req_to_token = torch.tensor([[101, 111, 0], [201, 211, 0]], dtype=torch.int32)
        allocator = _FakeAllocator()
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch([req0, req1]),
            req_to_token_pool=SimpleNamespace(
                req_to_token=req_to_token,
                write=lambda indices, values: req_to_token.__setitem__(indices, values),
            ),
            token_to_kv_pool_allocator=allocator,
        )
        manager.req_to_token_pool = live_scheduler.req_to_token_pool
        manager._build_particle_batch = MagicMock(return_value=SimpleNamespace(reqs=[req0, req1]))

        scheduler.on_batch_done(
            [req0, req1],
            torch.tensor([9.0, 0.0], dtype=torch.float32),
        )
        scheduler.step(live_scheduler)

        self.assertEqual(req0.kv_allocated_len, req0.kv_committed_len)
        self.assertEqual(req1.kv_allocated_len, req1.kv_committed_len)
        self.assertEqual(get_smc_reserved_kv_len(req0), req0.kv_committed_len)
        self.assertEqual(get_smc_reserved_kv_len(req1), req1.kv_committed_len)
        self.assertEqual(len(allocator.dec_calls), 3)
        self.assertTrue(
            torch.equal(allocator.dec_calls[0], torch.tensor([111], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(allocator.dec_calls[1], torch.tensor([211], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(allocator.dec_calls[2], torch.tensor([201], dtype=torch.int64))
        )


class TestSMCScoreInput(TestCase):
    def _make_score_input(self, draft_token_num: int = 4) -> SMCScoreInput:
        return SMCScoreInput(
            draft_token=torch.tensor([17, 23, 29, 29], dtype=torch.int32),
            draft_lengths=torch.tensor([2], dtype=torch.int32),
            draft_logprobs=torch.tensor([0.5], dtype=torch.float32),
            positions=torch.tensor([3, 4, 5, 6], dtype=torch.int64),
            custom_mask=None,
            draft_token_num=draft_token_num,
            target_temperature=1.0,
        )

    @patch("sglang.srt.speculative.smc_info.assign_extend_cache_locs_func")
    def test_prepare_for_v2_verify_uses_graph_runner_when_available(
        self,
        mock_assign_extend_cache_locs,
    ):
        with patch("sglang.srt.speculative.smc_info.ForwardBatch.init_new") as mock_init_forward_batch:
            score_input = self._make_score_input()
            req = SimpleNamespace(req_pool_idx=3, kv_allocated_len=5, rid="r-1")
            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                req_pool_indices=torch.tensor([3], dtype=torch.int64),
                seq_lens=torch.tensor([4], dtype=torch.int64),
                input_ids=None,
                reqs=[req],
                capture_hidden_mode=None,
            )
            req_to_token_pool = SimpleNamespace(
                req_to_token=torch.zeros((8, 32), dtype=torch.int32),
                write=MagicMock(),
            )
            graph_runner = MagicMock()
            graph_runner.can_run.return_value = True
            attn_backend = MagicMock()
            mock_assign_extend_cache_locs.return_value = torch.tensor(
                [11, 12, 13, 14],
                dtype=torch.int64,
            )
            target_worker = SimpleNamespace(
                model_runner=SimpleNamespace(
                    token_to_kv_pool_allocator=MagicMock(),
                    graph_runner=graph_runner,
                    attn_backend=attn_backend,
                )
            )
            fake_forward_batch = SimpleNamespace(
                req_pool_indices=batch.req_pool_indices,
                seq_lens=batch.seq_lens.to(dtype=torch.int32),
            )
            mock_init_forward_batch.return_value = fake_forward_batch

            verify_forward_batch, can_run_cuda_graph = score_input.prepare_for_v2_verify(
                req_to_token_pool,
                batch,
                target_worker,
            )

            self.assertIs(verify_forward_batch, fake_forward_batch)
            self.assertTrue(can_run_cuda_graph)
            self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)
            self.assertEqual(batch.capture_hidden_mode, CaptureHiddenMode.NULL)
            self.assertTrue(torch.equal(batch.input_ids, score_input.draft_token))
            self.assertTrue(
                torch.equal(
                    batch.out_cache_loc,
                    torch.tensor([11, 12, 13, 14], dtype=torch.int64),
                )
            )
            self.assertTrue(
                torch.equal(
                    fake_forward_batch.extend_prefix_lens,
                    torch.tensor([4], dtype=torch.int32),
                )
            )
            self.assertTrue(
                torch.equal(
                    fake_forward_batch.extend_seq_lens,
                    torch.tensor([4], dtype=torch.int32),
                )
            )
            graph_runner.replay_prepare.assert_called_once_with(fake_forward_batch)
            attn_backend.init_forward_metadata.assert_not_called()
            self.assertFalse(verify_forward_batch.disable_graph_runner)

    @patch("sglang.srt.speculative.smc_info.assign_extend_cache_locs_func")
    @patch("sglang.srt.speculative.smc_info.ForwardBatch.init_new")
    def test_prepare_for_v2_verify_falls_back_to_attn_backend_without_graph(
        self, mock_init_forward_batch, mock_assign_extend_cache_locs
    ):
        score_input = self._make_score_input()
        req = SimpleNamespace(req_pool_idx=2, kv_allocated_len=8, rid="r-2")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.tensor([2], dtype=torch.int64),
            seq_lens=torch.tensor([4], dtype=torch.int64),
            input_ids=None,
            reqs=[req],
            capture_hidden_mode=None,
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((8, 32), dtype=torch.int32),
            write=MagicMock(),
        )
        graph_runner = MagicMock()
        graph_runner.can_run.return_value = False
        attn_backend = MagicMock()
        mock_assign_extend_cache_locs.return_value = torch.tensor(
            [11, 12, 13, 14],
            dtype=torch.int64,
        )
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool_allocator=MagicMock(),
                graph_runner=graph_runner,
                attn_backend=attn_backend,
            )
        )
        fake_forward_batch = SimpleNamespace(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens.to(dtype=torch.int32),
        )
        mock_init_forward_batch.return_value = fake_forward_batch

        verify_forward_batch, can_run_cuda_graph = score_input.prepare_for_v2_verify(
            req_to_token_pool,
            batch,
            target_worker,
        )

        self.assertIs(verify_forward_batch, fake_forward_batch)
        self.assertFalse(can_run_cuda_graph)
        attn_backend.init_forward_metadata.assert_called_once_with(fake_forward_batch)
        graph_runner.replay_prepare.assert_not_called()
        self.assertTrue(verify_forward_batch.disable_graph_runner)

    @patch("sglang.srt.speculative.smc_info.assign_extend_cache_locs_func")
    @patch("sglang.srt.speculative.smc_info.ForwardBatch.init_new")
    def test_prepare_for_v2_verify_disables_graph_runner_on_temperature_mismatch(
        self, mock_init_forward_batch, mock_assign_extend_cache_locs
    ):
        score_input = self._make_score_input()
        score_input.target_temperature = 0.8
        req = SimpleNamespace(req_pool_idx=2, kv_allocated_len=8, rid="r-3")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.tensor([2], dtype=torch.int64),
            seq_lens=torch.tensor([4], dtype=torch.int64),
            input_ids=None,
            reqs=[req],
            capture_hidden_mode=None,
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((8, 32), dtype=torch.int32),
            write=MagicMock(),
        )
        graph_runner = MagicMock()
        attn_backend = MagicMock()
        mock_assign_extend_cache_locs.return_value = torch.tensor(
            [11, 12, 13, 14],
            dtype=torch.int64,
        )
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                server_args=SimpleNamespace(smc_target_temperature=1.0),
                token_to_kv_pool_allocator=MagicMock(),
                graph_runner=graph_runner,
                attn_backend=attn_backend,
            )
        )
        fake_forward_batch = SimpleNamespace(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens.to(dtype=torch.int32),
        )
        mock_init_forward_batch.return_value = fake_forward_batch

        verify_forward_batch, can_run_cuda_graph = score_input.prepare_for_v2_verify(
            req_to_token_pool,
            batch,
            target_worker,
        )

        self.assertIs(verify_forward_batch, fake_forward_batch)
        self.assertFalse(can_run_cuda_graph)
        graph_runner.can_run.assert_not_called()
        graph_runner.replay_prepare.assert_not_called()
        attn_backend.init_forward_metadata.assert_called_once_with(fake_forward_batch)
        self.assertTrue(verify_forward_batch.disable_graph_runner)

    def test_score_input_uses_linear_target_verify(self):
        score_input = self._make_score_input()
        self.assertTrue(score_input.use_linear_target_verify())

    def test_score_input_can_disable_linear_target_verify(self):
        score_input = self._make_score_input()
        score_input.linear_target_verify = False
        self.assertFalse(score_input.use_linear_target_verify())

    def test_sample_returns_batched_logprob_diffs(self):
        score_input = self._make_score_input()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.tensor([4], dtype=torch.int64),
        )
        logits_output = SimpleNamespace(
            next_token_logits=torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0],
                ],
                dtype=torch.float32,
            )
        )

        (
            accept_lengths,
            committed_seq_lens,
            next_last_token_ids,
            logprob_diffs,
        ) = score_input.sample(batch, logits_output)

        self.assertTrue(torch.equal(accept_lengths, torch.tensor([2], dtype=torch.int32)))
        self.assertTrue(torch.equal(committed_seq_lens, torch.tensor([7], dtype=torch.int64)))
        self.assertTrue(torch.equal(next_last_token_ids, torch.tensor([29], dtype=torch.int32)))
        self.assertEqual(logprob_diffs.shape, (1,))
        self.assertEqual(logprob_diffs.dtype, torch.float32)
        self.assertAlmostEqual(float(logprob_diffs[0].item()), 8.5)


class TestSMCVerifyGraphGate(TestCase):
    def test_model_runner_forward_respects_disable_graph_runner(self):
        graph_runner = SimpleNamespace(
            can_run=MagicMock(return_value=True),
            replay=MagicMock(return_value="graph"),
        )
        fake_self = SimpleNamespace(
            device="cuda",
            graph_runner=graph_runner,
            forward_decode=MagicMock(),
            forward_extend=MagicMock(return_value=("extend", False)),
            forward_idle=MagicMock(),
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            disable_graph_runner=True,
            global_num_tokens_cpu=None,
            num_token_non_padded=None,
            global_num_tokens_gpu=None,
            out_cache_loc_swa=None,
            prepare_attn_tp_scatter_input=MagicMock(),
        )

        out = ModelRunner._forward_raw(
            fake_self,
            forward_batch,
            skip_attn_backend_init=True,
            pp_proxy_tensors=None,
        )

        graph_runner.can_run.assert_not_called()
        graph_runner.replay.assert_not_called()
        fake_self.forward_extend.assert_called_once_with(
            forward_batch,
            skip_attn_backend_init=True,
            pp_proxy_tensors=None,
        )
        self.assertEqual(out.logits_output, "extend")
        self.assertFalse(out.can_run_graph)


class TestSMCDraftInput(TestCase):
    def test_prepare_for_v2_draft_replays_anchor_at_visible_position(self):
        @dataclass
        class _DraftBatch:
            seq_lens: torch.Tensor
            seq_lens_cpu: torch.Tensor
            req_pool_indices: torch.Tensor
            input_ids: torch.Tensor
            out_cache_loc: torch.Tensor | None = None
            sampling_info: object | None = None
            spec_info: object | None = None
            capture_hidden_mode: object | None = None
            return_logprob: bool = False
            forward_mode: object | None = None
            seq_lens_sum: int | None = None

        class _FakeDraftCacheKernel:
            def __init__(self):
                self.calls = []

            def __getitem__(self, grid):
                def _launch(
                    req_pool_indices,
                    req_to_token,
                    seq_lens,
                    out_cache_loc,
                    pool_len,
                    topk,
                    speculative_num_steps,
                ):
                    self.calls.append(
                        {
                            "grid": grid,
                            "req_pool_indices": req_pool_indices.clone(),
                            "seq_lens": seq_lens.clone(),
                            "pool_len": pool_len,
                            "topk": topk,
                            "speculative_num_steps": speculative_num_steps,
                        }
                    )
                    out_cache_loc.copy_(
                        torch.arange(
                            900,
                            900 + out_cache_loc.numel(),
                            dtype=torch.int64,
                            device=out_cache_loc.device,
                        )
                    )

                return _launch

        fake_kernel = _FakeDraftCacheKernel()
        batch = _DraftBatch(
            seq_lens=torch.tensor([6, 8], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([6, 8], dtype=torch.int64),
            req_pool_indices=torch.tensor([1, 3], dtype=torch.int64),
            input_ids=torch.tensor([17, 19], dtype=torch.int32),
        )
        draft_input = SMCDraftInput(
            last_token_ids=torch.tensor([17, 19], dtype=torch.int32),
            new_seq_lens=batch.seq_lens,
        )
        cuda_graph_runner = MagicMock()
        cuda_graph_runner.can_run.return_value = True
        forward_batch = MagicMock()

        with patch(
            "sglang.srt.speculative.smc_info.assign_draft_cache_locs_page_size_1",
            new=fake_kernel,
        ), patch(
            "sglang.srt.speculative.smc_info.ForwardBatch.init_new",
            return_value=forward_batch,
        ) as mock_init_new:
            returned_forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.zeros((4, 32), dtype=torch.int32)
                ),
                batch=batch,
                cuda_graph_runner=cuda_graph_runner,
                draft_model_runner=MagicMock(),
                gamma=3,
                draft_sampling_info=MagicMock(),
            )

        self.assertIs(returned_forward_batch, forward_batch)
        self.assertTrue(can_cuda_graph)
        self.assertTrue(
            torch.equal(
                draft_input.seq_lens_steps,
                torch.tensor([[6, 8], [7, 9], [8, 10]], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                draft_input.seq_lens_cpu_steps,
                torch.tensor([[6, 8], [7, 9], [8, 10]], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                draft_input.seq_lens_sum_steps,
                torch.tensor([14, 16, 18], dtype=torch.int64),
            )
        )
        self.assertTrue(torch.equal(draft_input.positions, torch.tensor([5, 7])))
        self.assertEqual(len(fake_kernel.calls), 1)
        self.assertEqual(fake_kernel.calls[0]["grid"], (2,))
        self.assertTrue(
            torch.equal(
                fake_kernel.calls[0]["req_pool_indices"],
                torch.tensor([1, 3], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                fake_kernel.calls[0]["seq_lens"],
                torch.tensor([5, 7], dtype=torch.int64),
            )
        )
        self.assertEqual(fake_kernel.calls[0]["pool_len"], 32)
        self.assertEqual(fake_kernel.calls[0]["topk"], 1)
        self.assertEqual(fake_kernel.calls[0]["speculative_num_steps"], 3)
        self.assertTrue(
            torch.equal(
                draft_input.out_cache_loc_steps,
                torch.tensor([[900, 903], [901, 904], [902, 905]], dtype=torch.int64),
            )
        )

        draft_batch = mock_init_new.call_args.args[0]
        self.assertEqual(draft_batch.forward_mode, ForwardMode.DECODE)
        self.assertTrue(
            torch.equal(draft_batch.input_ids, torch.tensor([17, 19], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(draft_batch.seq_lens, torch.tensor([6, 8], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                draft_batch.seq_lens_cpu, torch.tensor([6, 8], dtype=torch.int64)
            )
        )
        self.assertEqual(draft_batch.seq_lens_sum, 14)
        self.assertTrue(
            torch.equal(
                draft_batch.out_cache_loc,
                torch.tensor([900, 903], dtype=torch.int64),
            )
        )

    def test_filter_batch_with_future_indices_only_updates_future_map_view(self):
        future_indices = SimpleNamespace(indices=torch.tensor([4, 7], dtype=torch.int64))
        draft_input = SMCDraftInput(
            last_token_ids=torch.tensor([10], dtype=torch.int32),
            new_seq_lens=torch.tensor([20], dtype=torch.int64),
            future_indices=future_indices,
            committed_seq_lens_cpu=torch.tensor([20, 21], dtype=torch.int64),
        )

        draft_input.filter_batch(torch.tensor([1], dtype=torch.int64))

        self.assertTrue(
            torch.equal(
                future_indices.indices,
                torch.tensor([7], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                draft_input.last_token_ids,
                torch.tensor([10], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                draft_input.new_seq_lens,
                torch.tensor([20], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                draft_input.committed_seq_lens_cpu,
                torch.tensor([21], dtype=torch.int64),
            )
        )

    def test_merge_batch_with_future_indices_preserves_cached_seq_lens_cpu(self):
        lhs = SMCDraftInput(
            last_token_ids=torch.tensor([10], dtype=torch.int32),
            new_seq_lens=torch.tensor([20], dtype=torch.int64),
            future_indices=SimpleNamespace(indices=torch.tensor([4], dtype=torch.int64)),
            committed_seq_lens_cpu=torch.tensor([20], dtype=torch.int64),
        )
        rhs = SMCDraftInput(
            last_token_ids=torch.tensor([11], dtype=torch.int32),
            new_seq_lens=torch.tensor([21], dtype=torch.int64),
            future_indices=SimpleNamespace(indices=torch.tensor([7], dtype=torch.int64)),
            committed_seq_lens_cpu=torch.tensor([21], dtype=torch.int64),
        )

        lhs.merge_batch(rhs)

        self.assertTrue(
            torch.equal(lhs.future_indices.indices, torch.tensor([4, 7], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                lhs.committed_seq_lens_cpu,
                torch.tensor([20, 21], dtype=torch.int64),
            )
        )

    def test_filter_batch_updates_cached_seq_lens_cpu_without_future_indices(self):
        draft_input = SMCDraftInput(
            last_token_ids=torch.tensor([10, 11], dtype=torch.int32),
            new_seq_lens=torch.tensor([20, 21], dtype=torch.int64),
            committed_seq_lens_cpu=torch.tensor([20, 21], dtype=torch.int64),
        )

        draft_input.filter_batch(torch.tensor([1], dtype=torch.int64))

        self.assertTrue(
            torch.equal(draft_input.last_token_ids, torch.tensor([11], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(draft_input.new_seq_lens, torch.tensor([21], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                draft_input.committed_seq_lens_cpu,
                torch.tensor([21], dtype=torch.int64),
            )
        )

    def test_merge_batch_updates_cached_seq_lens_cpu_without_future_indices(self):
        lhs = SMCDraftInput(
            last_token_ids=torch.tensor([10], dtype=torch.int32),
            new_seq_lens=torch.tensor([20], dtype=torch.int64),
            committed_seq_lens_cpu=torch.tensor([20], dtype=torch.int64),
        )
        rhs = SMCDraftInput(
            last_token_ids=torch.tensor([11], dtype=torch.int32),
            new_seq_lens=torch.tensor([21], dtype=torch.int64),
            committed_seq_lens_cpu=torch.tensor([21], dtype=torch.int64),
        )

        lhs.merge_batch(rhs)

        self.assertTrue(
            torch.equal(lhs.last_token_ids, torch.tensor([10, 11], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(lhs.new_seq_lens, torch.tensor([20, 21], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                lhs.committed_seq_lens_cpu,
                torch.tensor([20, 21], dtype=torch.int64),
            )
        )

    @patch("sglang.srt.speculative.smc_info.get_global_server_args")
    @patch("sglang.srt.speculative.smc_info.alloc_token_slots")
    @patch("sglang.srt.speculative.smc_info.assign_req_to_token_pool_func")
    def test_prepare_for_decode_updates_allocated_lens_and_slot_assignments(
        self,
        mock_assign_req_to_token_pool,
        mock_alloc_token_slots,
        mock_get_global_server_args,
    ):
        mock_get_global_server_args.return_value = SimpleNamespace(
            smc_gamma=3,
            speculative_num_draft_tokens=4,
        )
        mock_alloc_token_slots.return_value = torch.tensor(
            [101, 102, 103, 104, 105, 106, 107, 108],
            dtype=torch.int64,
        )

        def fake_assign(req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, batch_size):
            cursor = 0
            for row in range(batch_size):
                start = int(start_offset[row].item())
                end = int(end_offset[row].item())
                req_to_token[int(req_pool_indices[row].item()), start:end] = out_cache_loc[
                    cursor : cursor + (end - start)
                ].to(dtype=torch.int32)
                cursor += end - start

        mock_assign_req_to_token_pool.side_effect = fake_assign

        reqs = [
            SimpleNamespace(req_pool_idx=3, kv_allocated_len=5, decode_batch_idx=0),
            SimpleNamespace(req_pool_idx=4, kv_allocated_len=7, decode_batch_idx=2),
        ]
        req_to_token = torch.zeros((8, 32), dtype=torch.int32)
        batch = SimpleNamespace(
            reqs=reqs,
            seq_lens=torch.tensor([6, 8], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([99, 101], dtype=torch.int64),
            seq_lens_sum=200,
            req_pool_indices=torch.tensor([3, 4], dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            tree_cache=MagicMock(),
            device=torch.device("cpu"),
            maybe_evict_swa=MagicMock(),
            batch_size=lambda: 2,
        )

        draft_input = SMCDraftInput(
            last_token_ids=torch.tensor([7, 9], dtype=torch.int32),
            new_seq_lens=torch.tensor([6, 8], dtype=torch.int64),
        )
        draft_input.prepare_for_decode(batch)

        self.assertEqual(reqs[0].kv_allocated_len, 9)
        self.assertEqual(reqs[1].kv_allocated_len, 11)
        self.assertEqual(reqs[0].decode_batch_idx, 1)
        self.assertEqual(reqs[1].decode_batch_idx, 3)
        self.assertTrue(
            torch.equal(batch.seq_lens_cpu, torch.tensor([6, 8], dtype=torch.int64))
        )
        self.assertEqual(batch.seq_lens_sum, 14)
        self.assertTrue(
            torch.equal(
                req_to_token[3, 5:9],
                torch.tensor([101, 102, 103, 104], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                req_to_token[4, 7:11],
                torch.tensor([105, 106, 107, 108], dtype=torch.int32),
            )
        )

    @patch("sglang.srt.speculative.smc_info.get_global_server_args")
    @patch("sglang.srt.speculative.smc_info.alloc_token_slots")
    @patch("sglang.srt.speculative.smc_info.assign_req_to_token_pool_func")
    def test_prepare_for_decode_reuses_reserved_len_after_visible_shrink(
        self,
        mock_assign_req_to_token_pool,
        mock_alloc_token_slots,
        mock_get_global_server_args,
    ):
        mock_get_global_server_args.return_value = SimpleNamespace(
            smc_gamma=3,
            speculative_num_draft_tokens=4,
        )

        req = SimpleNamespace(req_pool_idx=3, kv_allocated_len=5, decode_batch_idx=0)
        set_smc_reserved_kv_len(req, 9)
        req_to_token = torch.zeros((8, 32), dtype=torch.int32)
        batch = SimpleNamespace(
            reqs=[req],
            seq_lens=torch.tensor([6], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([99], dtype=torch.int64),
            seq_lens_sum=99,
            req_pool_indices=torch.tensor([3], dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            tree_cache=MagicMock(),
            device=torch.device("cpu"),
            maybe_evict_swa=MagicMock(),
            batch_size=lambda: 1,
        )

        draft_input = SMCDraftInput(
            last_token_ids=torch.tensor([7], dtype=torch.int32),
            new_seq_lens=torch.tensor([6], dtype=torch.int64),
        )
        draft_input.prepare_for_decode(batch)

        mock_alloc_token_slots.assert_not_called()
        mock_assign_req_to_token_pool.assert_not_called()
        self.assertEqual(req.kv_allocated_len, 9)
        self.assertEqual(get_smc_reserved_kv_len(req), 9)
        self.assertEqual(req.decode_batch_idx, 1)

    @patch("sglang.srt.speculative.smc_info.get_global_server_args")
    @patch("sglang.srt.speculative.smc_info.alloc_token_slots")
    @patch("sglang.srt.speculative.smc_info.assign_req_to_token_pool_func")
    def test_prepare_for_decode_uses_cached_seq_lens_cpu_without_verify_sync(
        self,
        mock_assign_req_to_token_pool,
        mock_alloc_token_slots,
        mock_get_global_server_args,
    ):
        mock_get_global_server_args.return_value = SimpleNamespace(
            smc_gamma=3,
            speculative_num_draft_tokens=4,
        )

        req = SimpleNamespace(req_pool_idx=3, kv_allocated_len=5, decode_batch_idx=0)
        set_smc_reserved_kv_len(req, 9)
        verify_done = MagicMock()
        batch_seq_lens = MagicMock()
        batch_seq_lens.cpu.side_effect = AssertionError(
            "prepare_for_decode should reuse cached CPU seq_lens"
        )
        batch = SimpleNamespace(
            reqs=[req],
            seq_lens=batch_seq_lens,
            seq_lens_cpu=torch.tensor([99], dtype=torch.int64),
            seq_lens_sum=99,
            req_pool_indices=torch.tensor([3], dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((8, 32), dtype=torch.int32)),
            tree_cache=MagicMock(),
            device=torch.device("cpu"),
            maybe_evict_swa=MagicMock(),
            batch_size=lambda: 1,
        )

        draft_input = SMCDraftInput(
            last_token_ids=torch.tensor([7], dtype=torch.int32),
            new_seq_lens=torch.tensor([6], dtype=torch.int64),
            verify_done=verify_done,
            committed_seq_lens_cpu=torch.tensor([6], dtype=torch.int64),
        )
        draft_input.prepare_for_decode(batch)

        verify_done.synchronize.assert_not_called()
        mock_alloc_token_slots.assert_not_called()
        mock_assign_req_to_token_pool.assert_not_called()
        self.assertTrue(
            torch.equal(batch.seq_lens_cpu, torch.tensor([6], dtype=torch.int64))
        )
        self.assertEqual(batch.seq_lens_sum, 6)
        self.assertEqual(req.kv_allocated_len, 9)
        self.assertEqual(req.decode_batch_idx, 1)


class TestSMCRequestState(TestCase):
    def test_get_best_particle_prefers_higher_weight(self):
        short_heavier = _make_particle(output_len=2, log_weight=10.0)
        long_lighter = _make_particle(output_len=3, log_weight=1.0)
        state = SMCRequestState(
            parent_req_id="parent",
            particles=[short_heavier, long_lighter],
            n_particles=2,
            gamma=2,
            resample_threshold=0.5,
            resample_method="systematic",
        )

        best = state.get_best_particle()
        self.assertIs(best, short_heavier)
        self.assertEqual(state.best_particle_idx, 0)

    def test_active_particles_excludes_finished(self):
        active = _make_particle(output_len=2, log_weight=0.0, finished=False)
        finished = _make_particle(output_len=3, log_weight=1.0, finished=True)
        state = SMCRequestState(
            parent_req_id="parent",
            particles=[active, finished],
            n_particles=2,
            gamma=2,
            resample_threshold=0.5,
            resample_method="systematic",
        )

        self.assertEqual(state.active_particles(), [active])
        self.assertFalse(state.is_terminal())


class TestValidateSMCParentReq(TestCase):
    def test_validate_rejects_stop_strings_and_hidden_states(self):
        req = MagicMock()
        req.grammar = None
        req.return_logprob = False
        req.return_hidden_states = True
        req.return_routed_experts = False
        req.sampling_params.stop_strs = []
        req.sampling_params.stop_regex_strs = []
        self.assertIn("return_hidden_states", validate_smc_parent_req(req))

        req.return_hidden_states = False
        req.sampling_params.stop_strs = ["stop"]
        self.assertIn("stop strings", validate_smc_parent_req(req))


class TestResolveSMCSeedOutputIds(TestCase):
    def test_uses_overlap_token_when_parent_lags_by_one(self):
        req = MagicMock()
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = []

        seeded = resolve_smc_seed_output_ids(
            req,
            overlap_last_token_id=99,
            overlap_new_seq_len=4,
        )

        self.assertEqual(seeded, [99])

    def test_accepts_already_synchronized_parent(self):
        req = MagicMock()
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [99]

        seeded = resolve_smc_seed_output_ids(
            req,
            overlap_last_token_id=99,
            overlap_new_seq_len=4,
        )

        self.assertEqual(seeded, [99])

    def test_rejects_large_overlap_gap(self):
        req = MagicMock()
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = []

        with self.assertRaisesRegex(ValueError, "at most one missing output token"):
            resolve_smc_seed_output_ids(
                req,
                overlap_last_token_id=99,
                overlap_new_seq_len=5,
            )


class TestResolveSMCProposalLength(TestCase):
    def _make_req(
        self,
        *,
        output_len: int = 0,
        max_new_tokens: int = 16,
        ignore_eos: bool = False,
        stop_token_ids=None,
        eos_token_ids=None,
        tokenizer_eos_id=None,
        additional_stop_token_ids=None,
        vocab_size: int = 32000,
        to_finish=None,
    ):
        req = MagicMock()
        req.finished.return_value = False
        req.output_ids = [1] * output_len
        req.sampling_params.max_new_tokens = max_new_tokens
        req.sampling_params.ignore_eos = ignore_eos
        req.sampling_params.stop_token_ids = stop_token_ids
        req.eos_token_ids = eos_token_ids
        req.tokenizer = (
            None
            if tokenizer_eos_id is None and additional_stop_token_ids is None
            else SimpleNamespace(
                eos_token_id=tokenizer_eos_id,
                additional_stop_token_ids=additional_stop_token_ids,
            )
        )
        req.vocab_size = vocab_size
        req.to_finish = to_finish
        return req

    def test_truncates_at_eos_without_changing_fixed_width_contract(self):
        req = self._make_req(
            eos_token_ids={99},
            tokenizer_eos_id=100,
            additional_stop_token_ids=[101],
        )

        proposal_len, proposal_finished = resolve_smc_proposal_length(
            req, [7, 99, 88, 77]
        )

        self.assertEqual(proposal_len, 2)
        self.assertTrue(proposal_finished)

    def test_ignore_eos_keeps_full_proposal_length(self):
        req = self._make_req(ignore_eos=True, eos_token_ids={99})

        proposal_len, proposal_finished = resolve_smc_proposal_length(
            req, [7, 99, 88, 77]
        )

        self.assertEqual(proposal_len, 4)
        self.assertFalse(proposal_finished)

    def test_max_new_tokens_is_counted_in_sample_phase(self):
        req = self._make_req(output_len=3, max_new_tokens=5)

        proposal_len, proposal_finished = resolve_smc_proposal_length(req, [7, 8, 9])

        self.assertEqual(proposal_len, 2)
        self.assertTrue(proposal_finished)


class TestSMCDraftCudaGraphSamplingSupport(TestCase):
    """Tests for SMCDraftCudaGraphRunner._supports_sampling_info."""

    def _make_sampling_info(self, **overrides):
        defaults = dict(
            grammars=None,
            has_custom_logit_processor=False,
            logit_bias=None,
            penalizer_orchestrator=SimpleNamespace(is_required=False),
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _check(self, sampling_info):
        from sglang.srt.speculative.smc_draft_cuda_graph_runner import (
            SMCDraftCudaGraphRunner,
        )

        return SMCDraftCudaGraphRunner._supports_sampling_info(None, sampling_info)

    def test_supports_standard_sampling(self):
        self.assertTrue(self._check(self._make_sampling_info()))

    def test_rejects_grammars(self):
        self.assertFalse(self._check(self._make_sampling_info(grammars=[object()])))

    def test_rejects_custom_logit_processor(self):
        self.assertFalse(
            self._check(self._make_sampling_info(has_custom_logit_processor=True))
        )

    def test_rejects_logit_bias(self):
        self.assertFalse(
            self._check(self._make_sampling_info(logit_bias=torch.zeros(10)))
        )

    def test_rejects_required_penalizer(self):
        self.assertFalse(
            self._check(
                self._make_sampling_info(
                    penalizer_orchestrator=SimpleNamespace(is_required=True)
                )
            )
        )

    def test_accepts_none_penalizer_orchestrator(self):
        """Overlap path may pass sampling_info with penalizer_orchestrator=None."""
        self.assertTrue(
            self._check(self._make_sampling_info(penalizer_orchestrator=None))
        )


class TestSMCPrefillOutputProcessor(TestCase):
    @patch("sglang.srt.managers.scheduler_output_processor_mixin._release_smc_parent_req")
    def test_process_batch_result_prefill_enqueues_new_smc_group_for_running(
        self,
        mock_release_parent,
    ):
        req = SimpleNamespace(
            rid="parent-1",
            output_ids=[],
            finished=lambda: False,
            is_retracted=False,
            is_chunked=0,
            smc_state=None,
            smc_particle_idx=None,
            return_logprob=False,
            return_hidden_states=False,
            multimodal_inputs=None,
            grammar=None,
            finished_reason=None,
            finished_len=None,
            time_stats=SimpleNamespace(
                set_prefill_finished_time=lambda: None,
                set_completion_time=lambda: None,
                set_last_chunked_prefill_finish_time=lambda: None,
            ),
            check_finished=lambda: None,
        )
        batch = SimpleNamespace(
            reqs=[req],
            spec_algorithm=SimpleNamespace(is_smc=lambda: True),
            return_logprob=False,
            decoding_reqs=None,
            prefill_stats=None,
            dp_cooperation_info=None,
            filter_batch=MagicMock(),
        )
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([41], dtype=torch.int32),
            can_run_cuda_graph=False,
        )

        processor = _FakeOutputProcessor()
        processor.is_generation = True
        processor.enable_metrics = False
        processor.smc_manager = SimpleNamespace(create_group=MagicMock(return_value=None))
        processor.smc_scheduler = SimpleNamespace(
            enqueue_group_for_running=MagicMock()
        )
        processor.req_to_token_pool = MagicMock()
        processor.token_to_kv_pool_allocator = MagicMock()
        processor.tree_cache = MagicMock()
        processor.maybe_collect_routed_experts = MagicMock()
        processor.maybe_collect_customized_info = MagicMock()
        processor.stream_output = MagicMock()
        processor.report_prefill_stats = MagicMock()

        processor.process_batch_result_prefill(batch, result)

        self.assertEqual(req.output_ids, [41])
        processor.smc_manager.create_group.assert_called_once_with(req, processor)
        mock_release_parent.assert_called_once_with(
            req,
            tree_cache=processor.tree_cache,
            req_to_token_pool=processor.req_to_token_pool,
            token_to_kv_pool_allocator=processor.token_to_kv_pool_allocator,
        )
        processor.smc_scheduler.enqueue_group_for_running.assert_called_once_with(
            "parent-1"
        )
        batch.filter_batch.assert_called_once_with(keep_indices=[])


class TestSMCDecodeOutputProcessor(TestCase):
    def test_process_batch_result_decode_does_not_double_increment_committed_kv(self):
        req = SimpleNamespace(
            rid="r-1",
            output_ids=[17],
            origin_input_ids=[1, 2, 3],
            kv_committed_len=3,
            kv_allocated_len=8,
            draft_prefix_materialized=True,
            finished=lambda: False,
            is_retracted=False,
            check_finished=lambda new_tokens: None,
            time_stats=SimpleNamespace(
                set_last_decode_finish_time=lambda: None,
                set_completion_time=lambda: None,
            ),
            spec_verify_ct=0,
            spec_accepted_tokens=0,
            update_spec_acceptance_histogram=lambda accepted: None,
            return_logprob=False,
            multimodal_inputs=None,
            grammar=None,
            finished_reason=None,
            finished_len=None,
            smc_group_id="g1",
            smc_particle_idx=0,
            mamba_ping_pong_track_buffer=None,
        )
        batch = SimpleNamespace(
            reqs=[req],
            smc_group_spans=build_smc_group_spans([req]),
            spec_algorithm=SimpleNamespace(
                is_none=lambda: False,
                is_smc=lambda: True,
            ),
            is_spec_v2=False,
            return_logprob=False,
            batch_size=lambda: 1,
            seq_lens=torch.tensor([4], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            seq_lens_sum=4,
            orig_seq_lens=torch.tensor([4], dtype=torch.int32),
            output_ids=torch.tensor([17], dtype=torch.int32),
            spec_info=SMCDraftInput(
                last_token_ids=torch.tensor([17], dtype=torch.int32),
                new_seq_lens=torch.tensor([4], dtype=torch.int64),
                committed_seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            ),
        )
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([41, 43, 0, 0, 0], dtype=torch.int32),
            accept_lens=torch.tensor([2], dtype=torch.int32),
            smc_logprob_diffs=torch.tensor([0.75], dtype=torch.float32),
            can_run_cuda_graph=False,
        )

        processor = _FakeOutputProcessor()
        processor.enable_overlap = False
        processor.enable_metrics = False
        processor.device = "cpu"
        processor.num_generated_tokens = 0
        processor.forward_ct_decode = 0
        processor.draft_worker = SimpleNamespace(speculative_num_draft_tokens=5)
        processor.server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False
        )
        processor.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=MagicMock(),
            free_group_end=MagicMock(),
        )
        processor.req_to_token_pool = MagicMock()
        processor.tree_cache = MagicMock()
        processor.smc_manager = SimpleNamespace(on_particle_finished=MagicMock())
        processor.smc_scheduler = SimpleNamespace(on_batch_done=MagicMock(return_value=[]))
        processor.stream_output = MagicMock()
        processor.report_decode_stats = MagicMock()
        processor.update_spec_metrics = MagicMock()
        processor.maybe_collect_customized_info = MagicMock()
        processor.maybe_collect_routed_experts = MagicMock()

        processor.process_batch_result_decode(batch, result)

        self.assertEqual(req.output_ids, [17, 41, 43])
        self.assertEqual(req.kv_committed_len, 5)
        self.assertEqual(req.kv_allocated_len, 8)
        self.assertTrue(req.draft_prefix_materialized)
        self.assertTrue(torch.equal(batch.seq_lens, torch.tensor([6], dtype=torch.int64)))
        self.assertTrue(
            torch.equal(batch.seq_lens_cpu, torch.tensor([6], dtype=torch.int64))
        )
        self.assertEqual(batch.seq_lens_sum, 6)
        self.assertTrue(torch.equal(batch.orig_seq_lens, torch.tensor([6], dtype=torch.int32)))
        self.assertTrue(torch.equal(batch.output_ids, torch.tensor([43], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(batch.spec_info.last_token_ids, torch.tensor([43], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(batch.spec_info.new_seq_lens, torch.tensor([6], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                batch.spec_info.committed_seq_lens_cpu,
                torch.tensor([6], dtype=torch.int64),
            )
        )
        self.assertEqual(req.spec_verify_ct, 1)
        self.assertEqual(req.spec_accepted_tokens, 1)
        processor.smc_scheduler.on_batch_done.assert_called_once()
        processor.token_to_kv_pool_allocator.free_group_begin.assert_called_once()
        processor.token_to_kv_pool_allocator.free_group_end.assert_called_once()
        processor.update_spec_metrics.assert_not_called()

    def test_process_batch_result_decode_passes_full_smc_batch_when_no_rows_are_skipped(self):
        req0 = SimpleNamespace(
            rid="r-1",
            output_ids=[17],
            origin_input_ids=[1, 2, 3],
            kv_committed_len=3,
            kv_allocated_len=8,
            finished=lambda: False,
            is_retracted=False,
            check_finished=lambda new_tokens: None,
            time_stats=SimpleNamespace(
                set_last_decode_finish_time=lambda: None,
                set_completion_time=lambda: None,
            ),
            spec_verify_ct=0,
            spec_accepted_tokens=0,
            update_spec_acceptance_histogram=lambda accepted: None,
            return_logprob=False,
            multimodal_inputs=None,
            grammar=None,
            finished_reason=None,
            finished_len=None,
            smc_group_id="g1",
            smc_particle_idx=0,
            mamba_ping_pong_track_buffer=None,
        )
        req1 = SimpleNamespace(
            rid="r-2",
            output_ids=[27],
            origin_input_ids=[1, 2, 3],
            kv_committed_len=3,
            kv_allocated_len=8,
            finished=lambda: False,
            is_retracted=False,
            check_finished=lambda new_tokens: None,
            time_stats=SimpleNamespace(
                set_last_decode_finish_time=lambda: None,
                set_completion_time=lambda: None,
            ),
            spec_verify_ct=0,
            spec_accepted_tokens=0,
            update_spec_acceptance_histogram=lambda accepted: None,
            return_logprob=False,
            multimodal_inputs=None,
            grammar=None,
            finished_reason=None,
            finished_len=None,
            smc_group_id="g1",
            smc_particle_idx=1,
            mamba_ping_pong_track_buffer=None,
        )
        batch = SimpleNamespace(
            reqs=[req0, req1],
            smc_group_spans=build_smc_group_spans([req0, req1]),
            spec_algorithm=SimpleNamespace(
                is_none=lambda: False,
                is_smc=lambda: True,
            ),
            is_spec_v2=True,
            return_logprob=False,
            batch_size=lambda: 2,
        )
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([41, 51, 0, 0, 0], dtype=torch.int32),
            accept_lens=torch.tensor([1, 1], dtype=torch.int32),
            smc_logprob_diffs=torch.tensor([0.25, 0.75], dtype=torch.float32),
            can_run_cuda_graph=False,
        )

        processor = _FakeOutputProcessor()
        processor.enable_overlap = True
        processor.enable_metrics = False
        processor.device = "cpu"
        processor.num_generated_tokens = 0
        processor.forward_ct_decode = 0
        processor.draft_worker = SimpleNamespace(speculative_num_draft_tokens=5)
        processor.server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False
        )
        processor.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=MagicMock(),
            free_group_end=MagicMock(),
        )
        processor.req_to_token_pool = MagicMock()
        processor.tree_cache = MagicMock()
        processor.smc_manager = SimpleNamespace(on_particle_finished=MagicMock())
        processor.smc_scheduler = SimpleNamespace(on_batch_done=MagicMock(return_value=[]))
        processor.stream_output = MagicMock()
        processor.report_decode_stats = MagicMock()
        processor.update_spec_metrics = MagicMock()
        processor.maybe_collect_customized_info = MagicMock()
        processor.maybe_collect_routed_experts = MagicMock()
        processor._resolve_spec_overlap_token_ids = MagicMock(return_value=[[41], [51]])

        processor.process_batch_result_decode(batch, result)

        called_reqs, called_diffs = processor.smc_scheduler.on_batch_done.call_args.args
        self.assertIs(called_reqs, batch.reqs)
        self.assertIs(called_diffs, result.smc_logprob_diffs)
        self.assertIs(
            processor.smc_scheduler.on_batch_done.call_args.kwargs["group_spans"],
            batch.smc_group_spans,
        )

    def test_process_batch_result_decode_does_not_pass_group_spans_when_rows_are_skipped(
        self,
    ):
        req0 = SimpleNamespace(
            rid="r-1",
            output_ids=[17],
            origin_input_ids=[1, 2, 3],
            kv_committed_len=3,
            kv_allocated_len=5,
            req_pool_idx=0,
            prefix_indices=torch.tensor([11, 12, 13], dtype=torch.int64),
            finished=lambda: True,
            is_retracted=False,
            check_finished=lambda new_tokens: None,
            time_stats=SimpleNamespace(
                set_last_decode_finish_time=lambda: None,
                set_completion_time=lambda: None,
            ),
            spec_verify_ct=0,
            spec_accepted_tokens=0,
            update_spec_acceptance_histogram=lambda accepted: None,
            return_logprob=False,
            multimodal_inputs=None,
            grammar=None,
            finished_reason=None,
            finished_len=None,
            smc_group_id="g1",
            smc_particle_idx=0,
            mamba_ping_pong_track_buffer=None,
        )
        req1 = SimpleNamespace(
            rid="r-2",
            output_ids=[27],
            origin_input_ids=[1, 2, 3],
            kv_committed_len=3,
            kv_allocated_len=8,
            req_pool_idx=1,
            prefix_indices=torch.tensor([21, 22, 23], dtype=torch.int64),
            finished=lambda: False,
            is_retracted=False,
            check_finished=lambda new_tokens: None,
            time_stats=SimpleNamespace(
                set_last_decode_finish_time=lambda: None,
                set_completion_time=lambda: None,
            ),
            spec_verify_ct=0,
            spec_accepted_tokens=0,
            update_spec_acceptance_histogram=lambda accepted: None,
            return_logprob=False,
            multimodal_inputs=None,
            grammar=None,
            finished_reason=None,
            finished_len=None,
            smc_group_id="g1",
            smc_particle_idx=1,
            mamba_ping_pong_track_buffer=None,
        )
        batch = SimpleNamespace(
            reqs=[req0, req1],
            smc_group_spans=build_smc_group_spans([req0, req1]),
            spec_algorithm=SimpleNamespace(
                is_none=lambda: False,
                is_smc=lambda: True,
            ),
            is_spec_v2=True,
            return_logprob=False,
            batch_size=lambda: 2,
        )
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([0, 51, 0, 0, 0], dtype=torch.int32),
            accept_lens=torch.tensor([0, 1], dtype=torch.int32),
            smc_logprob_diffs=torch.tensor([0.25, 0.75], dtype=torch.float32),
            can_run_cuda_graph=False,
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor(
                [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25]], dtype=torch.int32
            ),
            free=MagicMock(side_effect=lambda released_req: setattr(released_req, "req_pool_idx", None)),
        )
        allocator = _FakeAllocator()

        processor = _FakeOutputProcessor()
        processor.enable_overlap = True
        processor.enable_metrics = False
        processor.device = "cpu"
        processor.num_generated_tokens = 0
        processor.forward_ct_decode = 0
        processor.draft_worker = SimpleNamespace(speculative_num_draft_tokens=5)
        processor.server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False
        )
        processor.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=MagicMock(),
            free_group_end=MagicMock(),
            dec_ref_and_free=allocator.dec_ref_and_free,
        )
        processor.req_to_token_pool = req_to_token_pool
        processor.tree_cache = MagicMock()
        processor.smc_manager = SimpleNamespace(on_particle_finished=MagicMock())
        processor.smc_scheduler = SimpleNamespace(on_batch_done=MagicMock(return_value=[]))
        processor.stream_output = MagicMock()
        processor.report_decode_stats = MagicMock()
        processor.update_spec_metrics = MagicMock()
        processor.maybe_collect_customized_info = MagicMock()
        processor.maybe_collect_routed_experts = MagicMock()
        processor._resolve_spec_overlap_token_ids = MagicMock(return_value=[[], [51]])

        processor.process_batch_result_decode(batch, result)

        called_reqs, called_diffs = processor.smc_scheduler.on_batch_done.call_args.args
        self.assertEqual(called_reqs, [req1])
        self.assertTrue(
            torch.equal(called_diffs, torch.tensor([0.75], dtype=torch.float32))
        )
        self.assertIsNone(
            processor.smc_scheduler.on_batch_done.call_args.kwargs["group_spans"]
        )

    def test_process_batch_result_decode_releases_already_finished_smc_req_in_overlap(self):
        req = SimpleNamespace(
            rid="r-2",
            output_ids=[17],
            origin_input_ids=[1, 2, 3],
            kv_committed_len=3,
            kv_allocated_len=5,
            req_pool_idx=0,
            prefix_indices=torch.tensor([11, 12, 13], dtype=torch.int64),
            finished=lambda: True,
            is_retracted=False,
            check_finished=lambda new_tokens: None,
            time_stats=SimpleNamespace(
                set_last_decode_finish_time=lambda: None,
                set_completion_time=lambda: None,
            ),
            spec_verify_ct=0,
            spec_accepted_tokens=0,
            update_spec_acceptance_histogram=lambda accepted: None,
            return_logprob=False,
            multimodal_inputs=None,
            grammar=None,
            finished_reason=None,
            finished_len=None,
            smc_group_id="g1",
            smc_particle_idx=0,
            mamba_ping_pong_track_buffer=None,
        )
        batch = SimpleNamespace(
            reqs=[req],
            spec_algorithm=SimpleNamespace(
                is_none=lambda: False,
                is_smc=lambda: True,
            ),
            is_spec_v2=True,
            return_logprob=False,
            batch_size=lambda: 1,
        )
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32),
            accept_lens=torch.tensor([0], dtype=torch.int32),
            smc_logprob_diffs=torch.tensor([0.0], dtype=torch.float32),
            can_run_cuda_graph=False,
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.int32),
            free=MagicMock(side_effect=lambda released_req: setattr(released_req, "req_pool_idx", None)),
        )
        allocator = _FakeAllocator()

        processor = _FakeOutputProcessor()
        processor.enable_overlap = True
        processor.enable_metrics = False
        processor.device = "cpu"
        processor.num_generated_tokens = 0
        processor.forward_ct_decode = 0
        processor.draft_worker = SimpleNamespace(speculative_num_draft_tokens=5)
        processor.server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False
        )
        processor.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=MagicMock(),
            free_group_end=MagicMock(),
            dec_ref_and_free=allocator.dec_ref_and_free,
        )
        processor.req_to_token_pool = req_to_token_pool
        processor.tree_cache = MagicMock()
        processor.smc_manager = SimpleNamespace(on_particle_finished=MagicMock())
        processor.smc_scheduler = SimpleNamespace(on_batch_done=MagicMock(return_value=[]))
        processor.stream_output = MagicMock()
        processor.report_decode_stats = MagicMock()
        processor.update_spec_metrics = MagicMock()
        processor.maybe_collect_customized_info = MagicMock()
        processor.maybe_collect_routed_experts = MagicMock()
        processor._resolve_spec_overlap_token_ids = MagicMock(return_value=[[]])

        processor.process_batch_result_decode(batch, result)

        self.assertEqual(len(allocator.dec_calls), 1)
        self.assertTrue(
            torch.equal(
                allocator.dec_calls[0],
                torch.tensor([11, 12, 13, 14, 15], dtype=torch.int64),
            )
        )
        self.assertIsNone(req.req_pool_idx)
        processor.smc_manager.on_particle_finished.assert_called_once_with(req)
