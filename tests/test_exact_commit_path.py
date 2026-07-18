"""CUDA integration test for the exact-mode commit path.

Exercises ``ScheduleBatchSMC.write_back_exact`` + ``collapse_exact`` and the
one-hot plan's dispatch through the REAL ``batched_resample_kv`` Triton
kernel, on a hand-built slot state that mimics the post-``prepare_for_decode``
snapshot of a cycle (seq lens advanced by gamma+1, fresh pages in the block
table at refcount 1, shared prefix pages at refcount N).

Asserts the invariants the design doc's collapse ordering guarantees
(docs/smc/unified-exact-smc.md):

* token history/counts advance by exactly ``accept_len + 1`` per slot;
* ``seq_lens == kv_allocated_lens == L + accept_len + 1`` on every slot
  (the invariant ``fused_prepare_decode`` requires next cycle);
* losers' block tables equal the winner's over the committed span;
* refcounts: shared prefix stays at N, winner's committed pages rise to N,
  the winner's rejected tail and all loser pages drop to 0 (tail freed
  directly, loser pages captured in the freed buffer);
* EOS inside the committed block finishes the whole group with the right
  ``finished_len``; a later EOS in the *rejected* span does not.

Requires CUDA (Triton kernel); skipped otherwise.
"""

import unittest
from types import SimpleNamespace

import torch

from smcsd.core.req_state import ScheduleBatchSMC


class _FakeReqToTokenPool:
    def __init__(self, n_rows, width, device):
        self.req_to_token = torch.zeros(
            (n_rows, width), dtype=torch.int32, device=device
        )
        self.device = device


class _FakeRefCountAllocator:
    """Functionally mirrors SMCRefCountedTokenAllocator's refcount API."""

    def __init__(self, size, device):
        self.page_size = 1
        self.size = size
        self.slot_ref_count = torch.zeros(
            size + 1, dtype=torch.int32, device=device
        )
        self.freed = []

    def dec_ref_and_free(self, indices):
        if indices.numel() == 0:
            return
        self.slot_ref_count[indices] -= 1
        to_free = indices[self.slot_ref_count[indices] == 0]
        if to_free.numel() > 0:
            self.freed.append(to_free.clone())


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA (Triton kernel)")
class TestExactCommitPath(unittest.TestCase):
    N = 4
    GAMMA = 3          # stride = 4
    L = 6              # shared prefix length before this cycle
    G = 2              # two groups
    EOS = 99

    def _build_state(self):
        device = "cuda"
        N, G, gp1, L = self.N, self.G, self.GAMMA + 1, self.L
        bs = G * N
        pool = _FakeReqToTokenPool(bs + 2, 64, device)
        alloc = _FakeRefCountAllocator(4096, device)

        st = ScheduleBatchSMC(
            max_num_reqs=bs,
            device=torch.device(device),
            gamma_plus_1=gp1,
            vocab_size=128,
            max_output_len=64,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=alloc,
            tree_cache=None,
            model_config=SimpleNamespace(),
            enable_overlap=False,
            n_particles=N,
        )

        # Hand-wire two groups: slots [0..N) and [N..2N), pool row == slot.
        slots = list(range(bs))
        st.group_slot_lists = {
            "g0": slots[:N],
            "g1": slots[N:],
        }
        st.group_mode = {"g0": "smc", "g1": "smc"}
        st.group_id_to_row = {"g0": 0, "g1": 1}
        st.row_to_group_id = {0: "g0", 1: "g1"}
        st.group_to_slots[0, :N] = torch.tensor(
            slots[:N], dtype=torch.int32, device=device
        )
        st.group_to_slots[1, :N] = torch.tensor(
            slots[N:], dtype=torch.int32, device=device
        )
        st.row_in_use[:2] = True
        st.rebuild_active_slots()
        idx = torch.arange(bs, device=device)
        st.req_pool_indices[idx] = idx
        # Post-prepare state: seq advanced to L + gp1, invariant kv == seq.
        st.seq_lens[idx] = L + gp1
        st.kv_allocated_lens[idx] = L + gp1
        st.seq_lens_host[torch.arange(bs)] = L + gp1
        st.token_counts[idx] = 3          # 3 tokens generated so far
        st.max_new_tokens_t[idx] = 32
        st.eos_token_ids_t[idx, 0] = self.EOS

        # Block tables: per group, shared prefix pages (refcount N) +
        # per-slot fresh pages (refcount 1).  Page ids are disjoint.
        page = 1  # avoid page 0 (kernel-safe but keeps asserts unambiguous)
        self.prefix_pages = {}
        self.fresh_pages = {}
        for g in range(G):
            pp = torch.arange(page, page + L, device=device)
            page += L
            self.prefix_pages[g] = pp
            for i in range(N):
                s = g * N + i
                fp = torch.arange(page, page + gp1, device=device)
                page += gp1
                self.fresh_pages[s] = fp
                pool.req_to_token[s, :L] = pp.to(torch.int32)
                pool.req_to_token[s, L : L + gp1] = fp.to(torch.int32)
                alloc.slot_ref_count[fp] = 1
            alloc.slot_ref_count[pp] = N
        return st, pool, alloc

    def test_commit_collapse_rollback(self):
        from smcsd.core.scheduler import SMCCoordinator

        device = "cuda"
        N, G, gamma, gp1, L = self.N, self.G, self.GAMMA, self.GAMMA + 1, self.L
        st, pool, alloc = self._build_state()

        # Group 0: accept_len=1 (commit 2 tokens: [10, 11]); EOS appears at
        # committed position 1 => the group finishes with finished_len
        # 3 (prior) + 2 = 5.  Group 1: accept_len=3 = gamma (full accept,
        # commit 4 tokens, no EOS), winner chain 2.
        accept_len = torch.tensor([1, 3], dtype=torch.int64, device=device)
        tokens = torch.tensor(
            [[10, self.EOS, 0, 0], [20, 21, 22, 23]],
            dtype=torch.int64,
            device=device,
        )
        winner = torch.tensor([0, 2], dtype=torch.int64, device=device)
        verified_next = tokens.gather(1, accept_len.view(2, 1)).squeeze(1)
        verified_next = verified_next.repeat_interleave(N)

        st.write_back_exact(tokens, accept_len, verified_next)
        plan = st.collapse_exact(accept_len, winner)

        coord = SMCCoordinator(
            device=device, resample_threshold=0.5, resample_method="systematic"
        )
        coord.dispatch_resample_batch(plan, st)
        st.rollback_seq_lens_host(accept_len.cpu(), st.active_slots_cpu)
        torch.cuda.synchronize()

        # -- lengths: rolled back everywhere, invariant restored --
        new_seq = torch.tensor(
            [L + 2] * N + [L + 4] * N, dtype=torch.int64, device=device
        )
        idx = torch.arange(G * N, device=device)
        self.assertTrue(torch.equal(st.seq_lens[idx], new_seq))
        self.assertTrue(torch.equal(st.kv_allocated_lens[idx], new_seq))
        self.assertTrue(
            torch.equal(st.seq_lens_host[: G * N], new_seq.cpu())
        )

        # -- token history: committed block on every slot --
        for g, (a, row) in enumerate([(1, [10, self.EOS]), (3, [20, 21, 22, 23])]):
            for i in range(N):
                s = g * N + i
                self.assertEqual(int(st.token_counts[s]), 3 + a + 1)
                got = st.all_token_ids[s, 3 : 3 + a + 1].tolist()
                self.assertEqual(got, row)
                self.assertEqual(int(st.verified_ids[s]), row[-1])

        # -- finish state: group 0 finished via EOS at committed pos 1 --
        for i in range(N):
            self.assertTrue(bool(st.finished_mask[i]))
            self.assertEqual(int(st.finished_len[i]), 5)
            self.assertEqual(int(st.finish_reason_code[i]), 2)
            self.assertEqual(int(st.matched_eos_token[i]), self.EOS)
            self.assertFalse(bool(st.finished_mask[N + i]))

        # -- block tables: losers mirror the winner over the committed span --
        for g in range(G):
            w = g * N + int(winner[g])
            span = int(new_seq[g * N])
            for i in range(N):
                s = g * N + i
                self.assertTrue(
                    torch.equal(
                        pool.req_to_token[s, :span],
                        pool.req_to_token[w, :span],
                    )
                )

        # -- refcounts --
        rc = alloc.slot_ref_count
        for g in range(G):
            w = g * N + int(winner[g])
            a = int(accept_len[g])
            # shared prefix: back to N owners
            self.assertTrue((rc[self.prefix_pages[g]] == N).all())
            # winner's committed fresh pages: N owners
            committed_fresh = self.fresh_pages[w][: a + 1]
            self.assertTrue((rc[committed_fresh] == N).all())
            # winner's rejected tail: freed (refcount 0)
            tail = self.fresh_pages[w][a + 1 :]
            self.assertTrue((rc[tail] == 0).all())
            # every loser's fresh pages: refcount 0, captured in freed_buf
            for i in range(N):
                s = g * N + i
                if s == w:
                    continue
                self.assertTrue((rc[self.fresh_pages[s]] == 0).all())

        # freed-buffer capture: losers' fresh pages ((N-1)*gp1 per group)
        # PLUS the winner tails (gamma - a per group) — both flow through
        # the same phase buffer now.  (No snapshot_to_host in this test, so
        # the phase never flipped.)
        expect = set()
        n_expected = 0
        for g in range(G):
            w = g * N + int(winner[g])
            a = int(accept_len[g])
            expect.update(self.fresh_pages[w][a + 1 :].tolist())
            n_expected += gp1 - (a + 1)
            for i in range(N):
                s = g * N + i
                if s != w:
                    expect.update(self.fresh_pages[s].tolist())
                    n_expected += gp1
        phase = st._snap_phase
        n_freed = int(st.kv_freed_counter[phase].item())
        self.assertEqual(n_freed, n_expected)
        freed = set(st.kv_freed_buf[phase, :n_freed].tolist())
        self.assertEqual(freed, expect)
        # Nothing goes through the allocator's immediate-free path anymore.
        self.assertEqual(alloc.freed, [])

    def test_mixed_partition_isolation(self):
        """Mixed cycle: SMC write-back on group 0's rows and exact commit
        on group 1's rows compose without cross-talk, and the ESS collect
        gate skips exact rows even when their weights look skewed."""
        from smcsd.core.scheduler import SMCCoordinator

        device = "cuda"
        N, gamma, gp1, L = self.N, self.GAMMA, self.GAMMA + 1, self.L
        st, pool, alloc = self._build_state()
        st.group_mode["g1"] = "exact"
        st.row_exact[1] = True
        st._mode_version += 1

        exact_gpos, exact_rows, smc_rows, exact_rows_cpu = (
            st.exact_partition()
        )
        self.assertEqual(exact_gpos.tolist(), [1])
        self.assertEqual(exact_rows.tolist(), list(range(N, 2 * N)))
        self.assertEqual(smc_rows.tolist(), list(range(N)))

        # ── SMC partition write-back (group 0) ──
        smc_tokens = torch.arange(
            40, 40 + N * gp1, dtype=torch.int64, device=device
        )
        lpd = torch.full((N, gamma), -0.5, device=device)
        bonus = torch.full((N,), 7, dtype=torch.int64, device=device)
        st.write_back_gpu(
            next_token_ids=smc_tokens,
            logprob_diff=lpd,
            bonus_ids=bonus,
            rows=smc_rows,
        )

        # ── Exact partition commit (group 1) ──
        accept_len = torch.tensor([2], dtype=torch.int64, device=device)
        tokens = torch.tensor(
            [[20, 21, 22, 0]], dtype=torch.int64, device=device
        )
        winner = torch.tensor([3], dtype=torch.int64, device=device)
        vnext = torch.full((N,), 22, dtype=torch.int64, device=device)
        st.write_back_exact(tokens, accept_len, vnext, rows=exact_rows)
        plan = st.collapse_exact(accept_len, winner, rows=exact_rows)
        coord = SMCCoordinator(
            device=device, resample_threshold=0.5,
            resample_method="systematic",
        )
        coord.dispatch_resample_batch(plan, st)
        st.rollback_seq_lens_host(
            accept_len.cpu(), st.active_slots_cpu[exact_rows_cpu]
        )
        torch.cuda.synchronize()

        # SMC group: weights accrued, full-stride commit, lengths untouched.
        for i in range(N):
            self.assertAlmostEqual(
                float(st.log_weights[i]), -0.5 * gamma, places=6
            )
            self.assertEqual(int(st.token_counts[i]), 3 + gp1)
            self.assertEqual(int(st.seq_lens[i]), L + gp1)
            self.assertEqual(int(st.verified_ids[i]), 7)
        # Exact group: zero weights, variable commit, rolled-back lengths.
        w_slot = N + int(winner[0])
        for i in range(N):
            s = N + i
            self.assertEqual(float(st.log_weights[s]), 0.0)
            self.assertEqual(int(st.token_counts[s]), 3 + 3)
            self.assertEqual(int(st.seq_lens[s]), L + 3)
            self.assertEqual(int(st.kv_allocated_lens[s]), L + 3)
            self.assertTrue(
                torch.equal(
                    pool.req_to_token[s, : L + 3],
                    pool.req_to_token[w_slot, : L + 3],
                )
            )
        self.assertTrue(
            torch.equal(st.seq_lens_host[N : 2 * N],
                        torch.full((N,), L + 3, dtype=torch.int64))
        )

        # ── ESS gate: skew BOTH groups' interval weights; only the SMC
        # group's row may resample. ──
        for i in range(N):
            st.interval_weights[i] = -50.0 * i
            st.interval_weights[N + i] = -50.0 * i
        plan2 = coord.collect_resample_jobs_batch(st)
        torch.cuda.synchronize()
        mask = plan2.resample_mask.tolist()
        self.assertTrue(mask[0], "SMC row should resample on skewed ESS")
        self.assertFalse(mask[1], "exact row must be gated out of ESS")

    def test_force_collapse_group(self):
        """SMC→exact entry: posterior collapse picks the dominant ancestor,
        zeroes weights, and mirrors its table onto the whole group."""
        from smcsd.core.scheduler import SMCCoordinator

        device = "cuda"
        N, gp1, L = self.N, self.GAMMA + 1, self.L
        st, pool, alloc = self._build_state()

        pick = 2
        for i in range(N):
            st.log_weights[i] = 0.0 if i == pick else -1e6
        plan = st.force_collapse_group("g0")
        coord = SMCCoordinator(
            device=device, resample_threshold=0.5,
            resample_method="systematic",
        )
        coord.dispatch_resample_batch(plan, st)
        torch.cuda.synchronize()

        for i in range(N):
            self.assertEqual(float(st.log_weights[i]), 0.0)
            self.assertEqual(float(st.interval_weights[i]), 0.0)
            self.assertTrue(
                torch.equal(
                    pool.req_to_token[i, : L + gp1],
                    pool.req_to_token[pick, : L + gp1],
                )
            )
        # Ancestor's pages now owned by the whole group; other chains'
        # fresh pages are unreachable (refcount 0).
        rc = alloc.slot_ref_count
        self.assertTrue((rc[self.fresh_pages[pick]] == N).all())
        for i in range(N):
            if i != pick:
                self.assertTrue((rc[self.fresh_pages[i]] == 0).all())
        self.assertTrue((rc[self.prefix_pages[0]] == N).all())
        # Group 1 untouched.
        for i in range(N, 2 * N):
            self.assertTrue((rc[self.fresh_pages[i]] == 1).all())


if __name__ == "__main__":
    unittest.main()
