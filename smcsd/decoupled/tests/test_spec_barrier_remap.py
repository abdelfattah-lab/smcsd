"""Unit tests for the SBP A1 source-row gather (`_spec_a1_source_rows`).

After a resample barrier the carried spec window (drafted over A0 =
spec.active_list_T) is scored over the post-rebuild active set A1.  Each A1 slot
holds the spec window of the ancestor it adopted — always a survivor in A0 — so
its columns are gathered from that ancestor's A0 row.  These tests pin that index
map without a GPU: survivors gather their own row, retired-onto-survivor slots
gather the survivor's row, revived finished slots gather their adopting
survivor's row, and slots that retired onto a pre-train-finished particle simply
aren't in A1 (so the out-of-A0 ancestor is never looked up — no KeyError).

Run: .venv/bin/python -m unittest smcsd.decoupled.tests.test_spec_barrier_remap
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from smcsd.decoupled.async_scheduler import AsyncDecoupledSMCScheduler, SpecState


def _make_spec(active_list_T, ancestor):
    """Minimal SpecState whose only gather-touched fields are populated."""
    return SpecState(
        pending=SimpleNamespace(batch=SimpleNamespace(spec_info=SimpleNamespace(
            verified_id=torch.zeros(len(active_list_T), dtype=torch.int64)))),
        active_list_T=list(active_list_T),
        active_t_T=torch.tensor(active_list_T, dtype=torch.int64),
        tag=7,
        epoch=3,
        ancestor=ancestor,
    )


def _rows(spec, a1_list):
    """Call the unbound method (it uses only spec + a1_list, not self)."""
    return AsyncDecoupledSMCScheduler._spec_a1_source_rows(
        SimpleNamespace(), spec, a1_list
    )


class TestSpecA1SourceRows(unittest.TestCase):
    def test_no_resample_identity(self):
        """ancestor=None (no resample) → A1 ⊆ A0; each A1 slot gathers its own
        A0 row."""
        spec = _make_spec([10, 11, 12], ancestor=None)
        np.testing.assert_array_equal(_rows(spec, [10, 11, 12]), [0, 1, 2])
        # a slot that finished and was dropped just isn't in A1
        np.testing.assert_array_equal(_rows(spec, [10, 12]), [0, 2])

    def test_retired_onto_survivor(self):
        """A0=[10,11,12,13]; slots 12,13 retired onto survivor 10.  All survive
        the rebuild → A1=A0; 12 and 13 gather survivor 10's row (0)."""
        ancestor = np.arange(64, dtype=np.int64)
        ancestor[12] = 10
        ancestor[13] = 10
        spec = _make_spec([10, 11, 12, 13], ancestor)
        np.testing.assert_array_equal(_rows(spec, [10, 11, 12, 13]), [0, 1, 0, 0])

    def test_revived_finished_slot_gathers_its_survivor(self):
        """A slot (50) that finished before this train is NOT in A0, but the
        resample revived it onto active survivor 10 (ancestor[50]=10).  In A1 it
        gathers survivor 10's row — ancestor[50]=10 ∈ A0, so well-defined."""
        ancestor = np.arange(64, dtype=np.int64)
        ancestor[50] = 10  # revived: finished slot 50 overwritten by survivor 10
        spec = _make_spec([10, 11], ancestor)
        np.testing.assert_array_equal(_rows(spec, [10, 11, 50]), [0, 1, 0])

    def test_retired_onto_finished_slot_not_in_a1(self):
        """A0=[10,11]; slot 11 retired onto a particle (124) that finished before
        this train (ancestor[11]=124 ∉ A0).  Slot 11 itself becomes finished and
        is dropped → not in A1, so 124 is never looked up (no KeyError)."""
        ancestor = np.arange(200, dtype=np.int64)
        ancestor[11] = 124
        spec = _make_spec([10, 11], ancestor)
        np.testing.assert_array_equal(_rows(spec, [10]), [0])  # only survivor 10


if __name__ == "__main__":
    unittest.main()
