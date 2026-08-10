from types import SimpleNamespace
from unittest.mock import patch

import torch

from smcsd.core.info import SMCDecodeContext


def test_decode_context_retains_fresh_cache_locations_for_cycle_graph():
    gamma_plus_1 = 3
    allocated = torch.tensor([41, 42, 43, 51, 52, 53], dtype=torch.int64)
    pool = SimpleNamespace(req_to_token=torch.zeros((2, 32), dtype=torch.int64))

    with (
        patch("smcsd.core.info.alloc_token_slots", return_value=allocated),
        patch(
            "sglang.srt.speculative.spec_utils.assign_req_to_token_pool_func"
        ),
    ):
        context, next_allocated_lens = SMCDecodeContext.from_slot_gather(
            seq_lens=torch.tensor([5, 7], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5, 7], dtype=torch.int64),
            kv_allocated_lens=torch.tensor([5, 7], dtype=torch.int64),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            gamma_plus_1=gamma_plus_1,
            req_to_token_pool=pool,
            tree_cache=None,
        )

    assert torch.equal(context.cache_locs, allocated.reshape(2, gamma_plus_1))
    assert torch.equal(next_allocated_lens, torch.tensor([8, 10]))
