import unittest

import torch

from smcsd.mem_cache.allocator import copy_block_table


class FakeReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.zeros((3, 8), dtype=torch.int32)

    def write(self, indices, values):
        self.req_to_token[indices] = values


class NonRefcountedAllocator:
    def __init__(self):
        self._next = 100
        self.payload = {}

    def alloc(self, need_size):
        out = torch.arange(self._next, self._next + need_size, dtype=torch.int64)
        self._next += need_size
        return out

    def get_cpu_copy(self, indices, mamba_indices=None):
        return torch.stack([self.payload[int(i)] for i in indices.tolist()])

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        for idx, value in zip(indices.tolist(), kv_cache_cpu):
            self.payload[int(idx)] = value.clone()


class CopyBlockTableTest(unittest.TestCase):
    def test_non_refcounted_allocator_clones_kv_payload(self):
        pool = FakeReqToTokenPool()
        allocator = NonRefcountedAllocator()
        src_indices = torch.tensor([1, 2, 3], dtype=torch.int32)
        pool.req_to_token[1, :3] = src_indices
        for idx in src_indices.tolist():
            allocator.payload[idx] = torch.tensor([idx, idx + 10])

        copy_block_table(
            pool,
            src_req_pool_idx=1,
            dst_req_pool_idx=2,
            seq_len=3,
            token_to_kv_pool_allocator=allocator,
        )

        dst_indices = pool.req_to_token[2, :3].tolist()
        self.assertEqual(dst_indices, [100, 101, 102])
        self.assertEqual(pool.req_to_token[1, :3].tolist(), [1, 2, 3])
        for src_idx, dst_idx in zip(src_indices.tolist(), dst_indices):
            self.assertTrue(
                torch.equal(allocator.payload[src_idx], allocator.payload[dst_idx])
            )
            self.assertIsNot(allocator.payload[src_idx], allocator.payload[dst_idx])


if __name__ == "__main__":
    unittest.main()
