"""Unit tests for SMC helper state and resampling utilities."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.smc_manager import SMCFinishedParticleSnapshot, SMCGroupState, SMCManager
from sglang.srt.speculative.smc_info import SMCDraftInput, SMCScoreInput
from sglang.srt.speculative.smc_info import (
    SMCParticleState,
    SMCRequestState,
    effective_sample_size,
    multinomial_resample,
    normalize_log_weights,
    resolve_smc_proposal_length,
    resolve_smc_seed_output_ids,
    systematic_resample,
    validate_smc_parent_req,
)
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
            step_counts={0: 3, 1: 1, 2: 5},
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


class TestSMCScoreInput(TestCase):
    def _make_score_input(self, draft_token_num: int = 4) -> SMCScoreInput:
        return SMCScoreInput(
            draft_token=torch.tensor([17, 23, 29, 29], dtype=torch.int64),
            draft_lengths=torch.tensor([2], dtype=torch.int32),
            draft_logprobs=torch.tensor([0.5], dtype=torch.float32),
            verify_out_cache_loc=torch.tensor(
                [[11], [12], [13], [14]], dtype=torch.int64
            ),
            positions=torch.tensor([3, 4, 5, 6], dtype=torch.int64),
            custom_mask=None,
            draft_token_num=draft_token_num,
            target_temperature=1.0,
        )

    def test_prepare_for_v2_verify_uses_graph_runner_when_available(
        self,
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

    @patch("sglang.srt.speculative.smc_info.ForwardBatch.init_new")
    def test_prepare_for_v2_verify_falls_back_to_attn_backend_without_graph(
        self, mock_init_forward_batch
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

    def test_score_input_uses_linear_target_verify(self):
        score_input = self._make_score_input()
        self.assertTrue(score_input.use_linear_target_verify())

    def test_sample_returns_batched_logprob_diffs(self):
        score_input = self._make_score_input()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.tensor([4], dtype=torch.int64),
        )
        logits_output = SimpleNamespace(
            next_token_logits=torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0],
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
        self.assertTrue(torch.equal(committed_seq_lens, torch.tensor([7], dtype=torch.int32)))
        self.assertTrue(torch.equal(next_last_token_ids, torch.tensor([29], dtype=torch.int64)))
        self.assertEqual(logprob_diffs.shape, (1,))
        self.assertEqual(logprob_diffs.dtype, torch.float32)


class TestSMCDraftInput(TestCase):
    @patch("sglang.srt.speculative.smc_info.get_global_server_args")
    @patch("sglang.srt.speculative.smc_info.alloc_token_slots")
    @patch("sglang.srt.speculative.smc_info.assign_req_to_token_pool_func")
    def test_prepare_for_decode_prepares_proposal_and_verify_cache_locs(
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
            req_pool_indices=torch.tensor([3, 4], dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            tree_cache=MagicMock(),
            device=torch.device("cpu"),
            maybe_evict_swa=MagicMock(),
            batch_size=lambda: 2,
        )

        draft_input = SMCDraftInput(
            last_token_ids=torch.tensor([7, 9], dtype=torch.int64),
            new_seq_lens=torch.tensor([6, 8], dtype=torch.int32),
        )
        draft_input.prepare_for_decode(batch)

        self.assertEqual(reqs[0].kv_allocated_len, 9)
        self.assertEqual(reqs[1].kv_allocated_len, 11)
        self.assertEqual(reqs[0].decode_batch_idx, 1)
        self.assertEqual(reqs[1].decode_batch_idx, 3)
        self.assertTrue(
            torch.equal(
                draft_input.verify_out_cache_loc,
                torch.tensor(
                    [[101, 105], [102, 106], [103, 107], [104, 108]],
                    dtype=torch.int64,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                draft_input.proposal_out_cache_loc,
                torch.tensor(
                    [[101, 105], [102, 106], [103, 107]],
                    dtype=torch.int64,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                draft_input.proposal_seq_lens_steps,
                torch.tensor([[5, 7], [6, 8], [7, 9]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                draft_input.proposal_seq_lens_cpu_steps,
                torch.tensor([[5, 7], [6, 8], [7, 9]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                draft_input.proposal_seq_lens_sum_steps,
                torch.tensor([12, 14, 16], dtype=torch.int64),
            )
        )


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
