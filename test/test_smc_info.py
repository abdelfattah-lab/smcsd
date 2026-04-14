"""Unit tests for SMC helper state and resampling utilities."""

from collections import deque
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
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.smc.smc_manager import (
    SMCFinishedParticleSnapshot,
    SMCGroupState,
    SMCManager,
)
from sglang.srt.smc.smc_info import SMCDraftInput, SMCVerifyInput
from sglang.srt.smc.smc_utils import (
    _release_internal_req,
    _release_smc_parent_req,
    effective_sample_size,
    multinomial_resample,
    normalize_log_weights,
    systematic_resample,
    validate_smc_parent_req,
)
from sglang.srt.smc.smc_resampler import SMCResampler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")

def _make_scheduler_req(
    *,
    group_id: str,
    particle_idx: int,
    req_pool_idx: int,
    output_ids: list[int],
    kv_indices: list[int],
    allocated_kv_indices: list[int] | None = None,
    decoded_text: str = "",
    surr_offset: int | None = None,
    read_offset: int | None = None,
    surr_and_decode_ids: list[int] | None = None,
    cur_decode_ids_len: int | None = None,
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
        prefix_indices=torch.tensor(kv_indices, dtype=torch.int64),
        finished_reason=None,
        finished_len=None,
        finished_output=None,
        to_finish=None,
        decoded_text=decoded_text,
        surr_offset=surr_offset,
        read_offset=read_offset,
        surr_and_decode_ids=(
            list(surr_and_decode_ids) if surr_and_decode_ids is not None else None
        ),
        cur_decode_ids_len=cur_decode_ids_len,
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

    def test_finalize_group_tiebreak_uses_visible_finished_length(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=0.5, smc_resample_method="systematic")
        )
        manager.req_to_token_pool = SimpleNamespace(
            free=lambda released_req: setattr(released_req, "req_pool_idx", None)
        )
        manager.token_to_kv_pool_allocator = SimpleNamespace(
            dec_ref_and_free=lambda _indices: None
        )

        finish_reason = SimpleNamespace(type="stop")
        req0 = SimpleNamespace(
            smc_particle_idx=0,
            req_pool_idx=None,
            output_ids=[11, 99, 100, 101],
            finished_reason=finish_reason,
            finished_len=2,
        )
        req1 = SimpleNamespace(
            smc_particle_idx=1,
            req_pool_idx=None,
            output_ids=[11, 12, 99],
            finished_reason=finish_reason,
            finished_len=3,
        )
        parent_req = SimpleNamespace(
            output_ids=[],
            finished_reason=None,
            finished_len=None,
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=parent_req,
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.tensor([0.0, 0.0], dtype=torch.float64),
            step_counts=[1, 1],
            finished_particles={
                0: SMCFinishedParticleSnapshot(
                    output_ids=list(req0.output_ids),
                    finished_reason=finish_reason,
                    finished_len=req0.finished_len,
                ),
                1: SMCFinishedParticleSnapshot(
                    output_ids=list(req1.output_ids),
                    finished_reason=finish_reason,
                    finished_len=req1.finished_len,
                ),
            },
        )

        finalized = manager._finalize_group("g1")

        self.assertIs(finalized, parent_req)
        self.assertEqual(parent_req.output_ids, [11, 12, 99])
        self.assertEqual(parent_req.finished_len, 3)

    def test_release_internal_req_frees_reserved_tail_when_visible_len_shrinks(self):
        req = SimpleNamespace(
            req_pool_idx=0,
            kv_committed_len=1,
            kv_allocated_len=2,
            prefix_indices=torch.tensor([11], dtype=torch.int64),
        )
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


class TestSMCReleaseHelpers(TestCase):
    @patch("sglang.srt.smc.smc_utils.get_global_server_args")
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


class TestSMCResampler(TestCase):
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

    def test_on_batch_done_uses_atomic_group_fast_path_for_contiguous_groups(self):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=0.0, smc_resample_method="systematic")
        )
        scheduler = SMCResampler(manager, device="cpu")

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
        self.assertEqual(scheduler._groups_needing_resample, {"g1", "g2"})

    @patch("sglang.srt.smc.smc_resampler.systematic_resample")
    def test_step_before_forward_resamples_inline_and_resumes(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = torch.tensor([0, 0])

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCResampler(manager, device="cpu")

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
        resumed_batch = SimpleNamespace(reqs=[req0, req1], is_empty=lambda: False)
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch([req0, req1, other_req], batch_is_full=True),
            req_to_token_pool=SimpleNamespace(
                req_to_token=req_to_token,
                write=lambda indices, values: req_to_token.__setitem__(indices, values),
            ),
            token_to_kv_pool_allocator=allocator,
        )
        manager.req_to_token_pool = live_scheduler.req_to_token_pool
        manager._build_particle_batch = MagicMock(return_value=resumed_batch)

        scheduler.on_batch_done(
            [req0, req1],
            torch.tensor([9.0, 0.0], dtype=torch.float32),
        )
        scheduler.step_before_forward(live_scheduler)

        manager._build_particle_batch.assert_called_once()
        self.assertIn(other_req, live_scheduler.running_batch.reqs)

    @patch("sglang.srt.smc.smc_resampler.systematic_resample")
    def test_step_skips_stall_when_resample_has_no_evictions(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = torch.tensor([0, 1])

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCResampler(manager, device="cpu")

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
        scheduler.step_before_forward(live_scheduler)

        self.assertEqual(live_scheduler.running_batch.reqs, [req0, req1])
        manager._build_particle_batch.assert_not_called()
        self.assertTrue(
            torch.equal(
                manager.groups["g1"].log_weights,
                torch.tensor([9.0, 0.0], dtype=torch.float64),
            )
        )

    @patch("sglang.srt.smc.smc_resampler.systematic_resample")
    def test_step_skips_resample_when_ess_stays_above_threshold(
        self,
        mock_systematic_resample,
    ):
        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=0.1, smc_resample_method="systematic")
        )
        scheduler = SMCResampler(manager, device="cpu")

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
        scheduler.step_before_forward(live_scheduler)

        mock_systematic_resample.assert_not_called()
        self.assertEqual(live_scheduler.running_batch.reqs, [req0, req1])
        self.assertTrue(
            torch.equal(
                manager.groups["g1"].log_weights,
                torch.tensor([9.0, 0.0], dtype=torch.float64),
            )
        )

    @patch("sglang.srt.smc.smc_resampler.systematic_resample")
    def test_step_replaces_empty_running_batch_before_reinsert(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = torch.tensor([0, 0])

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCResampler(manager, device="cpu")

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
        scheduler.step_before_forward(live_scheduler)

        self.assertIs(live_scheduler.running_batch, rebuilt_batch)
        manager._build_particle_batch.assert_called_once_with(
            [req0, req1],
            live_scheduler,
            use_future_map=False,
        )

    @patch("sglang.srt.smc.smc_resampler.systematic_resample")
    def test_step_snapshots_resample_sources_before_destination_writes(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = torch.tensor([1, 0])

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCResampler(manager, device="cpu")

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10, 11],
            kv_indices=[101, 102],
            decoded_text="req0-text",
            surr_offset=4,
            read_offset=7,
            surr_and_decode_ids=[1, 2, 10, 11],
            cur_decode_ids_len=2,
        )
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20, 21],
            kv_indices=[201, 202],
            decoded_text="req1-text",
            surr_offset=5,
            read_offset=8,
            surr_and_decode_ids=[3, 4, 20, 21],
            cur_decode_ids_len=2,
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
        scheduler.step_before_forward(live_scheduler)

        self.assertEqual(req0.output_ids, [20, 21])
        self.assertEqual(req1.output_ids, [10, 11])
        self.assertEqual(req0.decoded_text, "req1-text")
        self.assertEqual(req1.decoded_text, "req0-text")
        self.assertEqual(req0.surr_offset, 5)
        self.assertEqual(req1.surr_offset, 4)
        self.assertEqual(req0.read_offset, 8)
        self.assertEqual(req1.read_offset, 7)
        self.assertEqual(req0.surr_and_decode_ids, [3, 4, 20, 21])
        self.assertEqual(req1.surr_and_decode_ids, [1, 2, 10, 11])
        self.assertEqual(req0.cur_decode_ids_len, 2)
        self.assertEqual(req1.cur_decode_ids_len, 2)
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

    @patch("sglang.srt.smc.smc_resampler.systematic_resample")
    def test_step_can_finalize_group_when_resample_clones_finished_particle(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = torch.tensor([1, 1])

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCResampler(manager, device="cpu")

        req0 = _make_scheduler_req(
            group_id="g1",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[101],
        )
        finish_reason = SimpleNamespace(type="stop")
        req1 = _make_scheduler_req(
            group_id="g1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[20],
            kv_indices=[201],
        )
        req1.finished_reason = finish_reason
        req1.finished_len = 1
        req1.finished = lambda: True
        parent_req = SimpleNamespace(
            output_ids=[],
            finished_reason=None,
            finished_len=None,
            time_stats=SimpleNamespace(set_completion_time=MagicMock()),
        )
        manager.groups["g1"] = SMCGroupState(
            group_id="g1",
            parent_req=parent_req,
            particle_reqs={0: req0, 1: req1},
            log_weights=torch.tensor([0.0, 4.0], dtype=torch.float64),
            step_counts=[0, 0],
            finished_particles={
                1: SMCFinishedParticleSnapshot(
                    output_ids=[20],
                    finished_reason=finish_reason,
                    finished_len=1,
                )
            },
        )

        req_to_token = torch.tensor([[101, 0, 0], [201, 0, 0]], dtype=torch.int32)
        allocator = _FakeAllocator()
        req_to_token_pool = SimpleNamespace(
            req_to_token=req_to_token,
            write=lambda indices, values: req_to_token.__setitem__(indices, values),
            free=MagicMock(
                side_effect=lambda released_req: setattr(
                    released_req, "req_pool_idx", None
                )
            ),
        )
        live_scheduler = SimpleNamespace(
            running_batch=_FakeRunningBatch([req0]),
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            stream_output=MagicMock(),
        )
        manager.req_to_token_pool = req_to_token_pool
        manager.token_to_kv_pool_allocator = allocator

        scheduler.on_batch_done(
            [req0],
            torch.tensor([0.0], dtype=torch.float32),
        )
        scheduler.step_before_forward(live_scheduler)

        self.assertIsNone(manager.get_group("g1"))
        self.assertEqual(parent_req.output_ids, [20])
        self.assertEqual(parent_req.finished_reason.type, finish_reason.type)
        parent_req.time_stats.set_completion_time.assert_called_once_with()
        live_scheduler.stream_output.assert_called_once_with([parent_req], False)

    @patch("sglang.srt.smc.smc_resampler.systematic_resample")
    def test_step_trims_stale_overalloc_before_reinsert(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = torch.tensor([0, 0])

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCResampler(manager, device="cpu")

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
        scheduler.step_before_forward(live_scheduler)

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

    @patch("sglang.srt.smc.smc_resampler.systematic_resample")
    def test_step_trims_hidden_reserved_tail_before_reinsert(
        self,
        mock_systematic_resample,
    ):
        mock_systematic_resample.return_value = torch.tensor([0, 0])

        manager = SMCManager(
            SimpleNamespace(smc_resample_threshold=1.0, smc_resample_method="systematic")
        )
        scheduler = SMCResampler(manager, device="cpu")

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
        req0.kv_allocated_len = 2
        req1.kv_allocated_len = 2
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
        scheduler.step_before_forward(live_scheduler)

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
        smc_resampler = SimpleNamespace()
        smc_resampler.step_before_forward = MagicMock()

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
            smc_resampler=smc_resampler,
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


class TestGenerationBatchResult(TestCase):
    def test_copy_to_cpu_moves_logprob_diff(self):
        copied_diffs = object()
        logprob_diff = MagicMock()
        logprob_diff.to.return_value = copied_diffs
        copy_done = SimpleNamespace(record=MagicMock())
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([1], dtype=torch.int32),
            copy_done=copy_done,
            logprob_diff=logprob_diff,
        )

        result.copy_to_cpu(return_logprob=False)

        logprob_diff.to.assert_called_once_with("cpu", non_blocking=True)
        self.assertIs(result.logprob_diff, copied_diffs)
        copy_done.record.assert_called_once()


class TestSMCPrefillOutputProcessor(TestCase):
    @patch("sglang.srt.managers.scheduler_output_processor_mixin._release_smc_parent_req")
    def test_process_batch_result_prefill_enqueues_new_smc_group_for_running(
        self,
        mock_release_parent,
    ):
        call_order = []
        req = SimpleNamespace(
            rid="parent-1",
            output_ids=[],
            finished=lambda: False,
            is_retracted=False,
            is_chunked=0,
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
        processor.model_worker = SimpleNamespace(
            materialize_smc_parent_draft_prefix=MagicMock(
                side_effect=lambda target_req: call_order.append(
                    ("materialize", target_req.rid)
                )
            )
        )
        processor.smc_manager = SimpleNamespace(
            create_group=MagicMock(
                side_effect=lambda target_req, scheduler: call_order.append(
                    ("create_group", target_req.rid)
                )
            )
        )
        processor.smc_resampler = SimpleNamespace(
            enqueue_group_for_running=MagicMock(
                side_effect=lambda group_id: call_order.append(("enqueue", group_id))
            )
        )
        processor.req_to_token_pool = MagicMock()
        processor.token_to_kv_pool_allocator = MagicMock()
        processor.tree_cache = MagicMock()
        processor.maybe_collect_routed_experts = MagicMock()
        processor.maybe_collect_customized_info = MagicMock()
        processor.stream_output = MagicMock()
        processor.report_prefill_stats = MagicMock()

        mock_release_parent.side_effect = (
            lambda *args, **kwargs: call_order.append(("release_parent", req.rid))
        )

        processor.process_batch_result_prefill(batch, result)

        self.assertEqual(req.output_ids, [41])
        processor.model_worker.materialize_smc_parent_draft_prefix.assert_called_once_with(
            req
        )
        processor.smc_manager.create_group.assert_called_once_with(req, processor)
        mock_release_parent.assert_called_once_with(
            req,
            tree_cache=processor.tree_cache,
            req_to_token_pool=processor.req_to_token_pool,
            token_to_kv_pool_allocator=processor.token_to_kv_pool_allocator,
        )
        processor.smc_resampler.enqueue_group_for_running.assert_called_once_with(
            "parent-1"
        )
        self.assertEqual(
            call_order,
            [
                ("materialize", "parent-1"),
                ("create_group", "parent-1"),
                ("release_parent", "parent-1"),
                ("enqueue", "parent-1"),
            ],
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
            seq_lens=torch.tensor([3], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([3], dtype=torch.int64),
            seq_lens_sum=3,
            orig_seq_lens=torch.tensor([3], dtype=torch.int32),
            output_ids=torch.tensor([17], dtype=torch.int32),
            spec_info=SMCDraftInput(
                verified_id=torch.tensor([17], dtype=torch.int32),
                new_seq_lens=torch.tensor([3], dtype=torch.int64),
            ),
        )
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([41, 43, 0, 0, 0], dtype=torch.int32),
            accept_lens=torch.tensor([2], dtype=torch.int32),
            logprob_diff=torch.tensor([0.75], dtype=torch.float32),
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
        processor.smc_resampler = SimpleNamespace(on_batch_done=MagicMock(return_value=[]))
        processor.stream_output = MagicMock()
        processor.report_decode_stats = MagicMock()
        processor.update_spec_metrics = MagicMock()
        processor.maybe_collect_customized_info = MagicMock()
        processor.maybe_collect_routed_experts = MagicMock()

        processor.process_batch_result_decode(batch, result)

        self.assertEqual(req.output_ids, [17, 41, 43])
        self.assertEqual(req.kv_committed_len, 5)
        self.assertEqual(req.kv_allocated_len, 8)
        self.assertTrue(torch.equal(batch.seq_lens, torch.tensor([5], dtype=torch.int64)))
        self.assertTrue(
            torch.equal(batch.seq_lens_cpu, torch.tensor([5], dtype=torch.int64))
        )
        self.assertEqual(batch.seq_lens_sum, 5)
        self.assertTrue(torch.equal(batch.orig_seq_lens, torch.tensor([5], dtype=torch.int32)))
        self.assertTrue(torch.equal(batch.output_ids, torch.tensor([43], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(batch.spec_info.verified_id, torch.tensor([43], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(batch.spec_info.new_seq_lens, torch.tensor([5], dtype=torch.int64))
        )
        self.assertEqual(req.spec_verify_ct, 1)
        self.assertEqual(req.spec_accepted_tokens, 1)
        processor.smc_resampler.on_batch_done.assert_called_once()
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
            logprob_diff=torch.tensor([0.25, 0.75], dtype=torch.float32),
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
        processor.smc_resampler = SimpleNamespace(on_batch_done=MagicMock(return_value=[]))
        processor.stream_output = MagicMock()
        processor.report_decode_stats = MagicMock()
        processor.update_spec_metrics = MagicMock()
        processor.maybe_collect_customized_info = MagicMock()
        processor.maybe_collect_routed_experts = MagicMock()
        processor._resolve_spec_overlap_token_ids = MagicMock(return_value=[[41], [51]])

        processor.process_batch_result_decode(batch, result)

        called_reqs, called_diffs = processor.smc_resampler.on_batch_done.call_args.args
        self.assertIs(called_reqs, batch.reqs)
        self.assertIs(called_diffs, result.logprob_diff)
        self.assertIs(
            processor.smc_resampler.on_batch_done.call_args.kwargs["group_spans"],
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
            logprob_diff=torch.tensor([0.25, 0.75], dtype=torch.float32),
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
        processor.smc_resampler = SimpleNamespace(on_batch_done=MagicMock(return_value=[]))
        processor.stream_output = MagicMock()
        processor.report_decode_stats = MagicMock()
        processor.update_spec_metrics = MagicMock()
        processor.maybe_collect_customized_info = MagicMock()
        processor.maybe_collect_routed_experts = MagicMock()
        processor._resolve_spec_overlap_token_ids = MagicMock(return_value=[[], [51]])

        processor.process_batch_result_decode(batch, result)

        called_reqs, called_diffs = processor.smc_resampler.on_batch_done.call_args.args
        self.assertEqual(called_reqs, [req1])
        self.assertTrue(
            torch.equal(called_diffs, torch.tensor([0.75], dtype=torch.float32))
        )
        self.assertIsNone(
            processor.smc_resampler.on_batch_done.call_args.kwargs["group_spans"]
        )

    def test_process_batch_result_decode_keeps_already_finished_smc_req_in_overlap(self):
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
            logprob_diff=torch.tensor([0.0], dtype=torch.float32),
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
        processor.smc_resampler = SimpleNamespace(on_batch_done=MagicMock(return_value=[]))
        processor.stream_output = MagicMock()
        processor.report_decode_stats = MagicMock()
        processor.update_spec_metrics = MagicMock()
        processor.maybe_collect_customized_info = MagicMock()
        processor.maybe_collect_routed_experts = MagicMock()
        processor._resolve_spec_overlap_token_ids = MagicMock(return_value=[[]])

        processor.process_batch_result_decode(batch, result)

        self.assertEqual(len(allocator.dec_calls), 0)
        self.assertEqual(req.req_pool_idx, 0)
        processor.smc_manager.on_particle_finished.assert_called_once_with(req)
