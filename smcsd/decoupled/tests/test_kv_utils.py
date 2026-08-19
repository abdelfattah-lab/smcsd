import unittest

import torch

from smcsd.decoupled.kv_utils import truncate_block_table_allocations


class _FakeReqToTokenPool:
    def __init__(self, rows):
        self.req_to_token = torch.tensor(rows, dtype=torch.int32)


class _FakeAllocator:
    def __init__(self, size=256):
        self.dec_calls = []
        self.free_calls = []
        self.slot_ref_count = torch.zeros(size, dtype=torch.int32)

    def dec_ref_and_free(self, indices):
        self.dec_calls.append(indices.clone())
        idx = indices.to(torch.int64)
        self.slot_ref_count[idx] -= 1
        to_free = idx[self.slot_ref_count[idx] == 0]
        if to_free.numel() > 0:
            self.free(to_free)

    def free(self, indices):
        self.free_calls.append(indices.clone())


class TestKVAllocationTruncateHelper(unittest.TestCase):
    def test_truncate_helper_drops_only_suffix_ownership_per_row(self):
        req_to_token_pool = _FakeReqToTokenPool(
            [
                [10, 11, 30, 31, 0, 0],
                [10, 11, 30, 31, 0, 0],
                [10, 11, 30, 31, 0, 0],
            ]
        )
        allocator = _FakeAllocator()
        allocator.slot_ref_count[torch.tensor([10, 11, 30, 31])] = 3

        truncate_block_table_allocations(
            req_to_token_pool,
            allocator,
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([4, 4], dtype=torch.int32),
            torch.tensor([2, 2], dtype=torch.int32),
        )

        self.assertEqual(
            [call.tolist() for call in allocator.dec_calls],
            [[30, 31], [30, 31]],
        )

    def test_truncate_helper_keeps_survivor_shared_suffix_alive(self):
        req_to_token_pool = _FakeReqToTokenPool(
            [
                [10, 11, 30, 31, 0, 0],
                [10, 11, 30, 31, 0, 0],
                [10, 11, 30, 31, 0, 0],
            ]
        )
        allocator = _FakeAllocator()
        allocator.slot_ref_count[torch.tensor([10, 11, 30, 31])] = 3

        truncate_block_table_allocations(
            req_to_token_pool,
            allocator,
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([4, 4], dtype=torch.int32),
            torch.tensor([2, 2], dtype=torch.int32),
        )

        ref = allocator.slot_ref_count
        self.assertEqual(int(ref[10].item()), 3)
        self.assertEqual(int(ref[11].item()), 3)
        self.assertEqual(int(ref[30].item()), 1)
        self.assertEqual(int(ref[31].item()), 1)
        self.assertEqual([call.tolist() for call in allocator.free_calls], [])

    def test_truncate_helper_rejects_growth(self):
        req_to_token_pool = _FakeReqToTokenPool([[10, 11, 0, 0]])
        allocator = _FakeAllocator()

        with self.assertRaises(ValueError):
            truncate_block_table_allocations(
                req_to_token_pool,
                allocator,
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32),
                torch.tensor([3], dtype=torch.int32),
            )


if __name__ == "__main__":
    unittest.main()
