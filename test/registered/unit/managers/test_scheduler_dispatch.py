from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock

from sglang.srt.managers.scheduler import dispatch_event_loop
from sglang.srt.managers.schedule_batch import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


def _make_scheduler(*, enable_overlap: bool, is_smc: bool, smc_resampling_overlap: bool = False):
    spec_algorithm = SimpleNamespace(is_smc=lambda: is_smc)
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            pp_size=1,
            smc_resampling_overlap=smc_resampling_overlap,
        ),
        disaggregation_mode=DisaggregationMode.NULL,
        enable_pdmux=False,
        enable_overlap=enable_overlap,
        spec_algorithm=spec_algorithm,
        event_loop_pdmux=MagicMock(),
        event_loop_pp=MagicMock(),
        event_loop_overlap=MagicMock(),
        event_loop_overlap_smc=MagicMock(),
        event_loop_normal=MagicMock(),
        event_loop_normal_smc=MagicMock(),
    )
    return scheduler


class TestSchedulerDispatch(TestCase):
    def test_dispatches_smc_overlap_loop(self):
        scheduler = _make_scheduler(
            enable_overlap=True,
            is_smc=True,
            smc_resampling_overlap=True,
        )

        dispatch_event_loop(scheduler)

        scheduler.event_loop_overlap_smc.assert_called_once_with()
        scheduler.event_loop_normal_smc.assert_not_called()

    def test_dispatches_smc_normal_loop(self):
        scheduler = _make_scheduler(
            enable_overlap=False,
            is_smc=True,
            smc_resampling_overlap=False,
        )

        dispatch_event_loop(scheduler)

        scheduler.event_loop_normal_smc.assert_called_once_with()
        scheduler.event_loop_overlap_smc.assert_not_called()

    def test_dispatches_generic_overlap_loop_for_non_smc(self):
        scheduler = _make_scheduler(enable_overlap=True, is_smc=False)

        dispatch_event_loop(scheduler)

        scheduler.event_loop_overlap.assert_called_once_with()
        scheduler.event_loop_overlap_smc.assert_not_called()

    def test_dispatches_generic_normal_loop_for_non_smc(self):
        scheduler = _make_scheduler(enable_overlap=False, is_smc=False)

        dispatch_event_loop(scheduler)

        scheduler.event_loop_normal.assert_called_once_with()
        scheduler.event_loop_normal_smc.assert_not_called()
