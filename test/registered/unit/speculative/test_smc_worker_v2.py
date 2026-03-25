from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.smc_info import SMCDraftInput, set_smc_reserved_kv_len
from sglang.srt.speculative.smc_worker_v2 import SMCDraftWorker, SMCWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class TestSMCWorkerV2(TestCase):
    @patch("sglang.srt.speculative.smc_worker_v2.DraftBackendFactory")
    @patch("sglang.srt.speculative.smc_worker_v2.StandaloneDraftWorker.init_attention_backend")
    def test_smc_draft_worker_does_not_capture_generic_draft_graphs(
        self,
        mock_standalone_init_attention_backend,
        mock_draft_backend_factory,
    ):
        fake_backend = MagicMock()
        mock_draft_backend_factory.return_value.create_decode_backend.return_value = (
            fake_backend
        )
        worker = object.__new__(SMCDraftWorker)
        worker.draft_runner = SimpleNamespace(init_device_graphs=MagicMock())
        worker.server_args = SimpleNamespace(smc_gamma=3)

        SMCDraftWorker.init_attention_backend(worker)

        worker.draft_runner.init_device_graphs.assert_not_called()
        mock_standalone_init_attention_backend.assert_called_once_with()
        self.assertIs(worker.smc_draft_attn_backend, fake_backend)

    @patch("sglang.srt.speculative.smc_worker_v2.SamplingBatchInfo.from_schedule_batch")
    @patch("sglang.srt.speculative.smc_worker_v2.ScheduleBatch.init_new")
    def test_make_decode_batch_reuses_reserved_slots(
        self,
        mock_init_new,
        mock_sampling_info_from_schedule_batch,
    ):
        class _FakeAllocator:
            def __init__(self):
                self.dec_calls = []

            def dec_ref_and_free(self, indices):
                self.dec_calls.append(indices.clone())

        class _FakeBatch(SimpleNamespace):
            def prepare_for_decode(self):
                locs = self.seq_lens.clone()
                self.req_to_token_pool.write(
                    (self.req_pool_indices, locs),
                    torch.tensor([77], dtype=torch.int32),
                )
                self.out_cache_loc = torch.tensor([77], dtype=torch.int64)
                self.input_ids = self.output_ids
                self.output_ids = None
                self.seq_lens = self.seq_lens + 1
                self.seq_lens_cpu = self.seq_lens_cpu + 1
                self.orig_seq_lens = self.orig_seq_lens + 1
                self.seq_lens_sum += len(self.reqs)
                for req in self.reqs:
                    req.decode_batch_idx += 1
                    req.kv_committed_len += 1
                    req.kv_allocated_len += 1

        req_to_token = torch.tensor([[11, 12, 13, 99, 100]], dtype=torch.int32)
        allocator = _FakeAllocator()
        req_to_token_pool = SimpleNamespace(
            req_to_token=req_to_token,
            write=lambda target, values: req_to_token.__setitem__(target, values),
        )
        mock_init_new.side_effect = lambda **kwargs: _FakeBatch(
            reqs=kwargs["reqs"],
            req_to_token_pool=req_to_token_pool,
        )
        mock_sampling_info_from_schedule_batch.return_value = MagicMock()

        req = SimpleNamespace(
            req_pool_idx=0,
            output_ids=[9],
            origin_input_ids=[1, 2],
            kv_committed_len=3,
            kv_allocated_len=3,
            decode_batch_idx=0,
        )
        set_smc_reserved_kv_len(req, 5)

        fake_self = SimpleNamespace(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            _internal_tree_cache=object(),
            device="cpu",
        )
        model_config = SimpleNamespace(vocab_size=32000)

        batch = SMCWorkerV2._make_decode_batch(fake_self, [req], model_config)

        self.assertTrue(torch.equal(batch.out_cache_loc, torch.tensor([99], dtype=torch.int64)))
        self.assertTrue(torch.equal(req_to_token[0, :5], torch.tensor([11, 12, 13, 99, 100], dtype=torch.int32)))
        self.assertTrue(torch.equal(batch.orig_seq_lens, torch.tensor([4], dtype=torch.int32)))
        self.assertEqual(req.kv_committed_len, 4)
        self.assertEqual(req.kv_allocated_len, 5)
        self.assertEqual(len(allocator.dec_calls), 1)
        self.assertTrue(torch.equal(allocator.dec_calls[0], torch.tensor([77], dtype=torch.int64)))

    @patch("sglang.srt.speculative.smc_worker_v2.SamplingBatchInfo.from_schedule_batch")
    @patch("sglang.srt.speculative.smc_worker_v2.ScheduleBatch.init_new")
    def test_make_decode_batch_keeps_fresh_slot_without_reserved_tail(
        self,
        mock_init_new,
        mock_sampling_info_from_schedule_batch,
    ):
        class _FakeAllocator:
            def __init__(self):
                self.dec_calls = []

            def dec_ref_and_free(self, indices):
                self.dec_calls.append(indices.clone())

        class _FakeBatch(SimpleNamespace):
            def prepare_for_decode(self):
                locs = self.seq_lens.clone()
                self.req_to_token_pool.write(
                    (self.req_pool_indices, locs),
                    torch.tensor([77], dtype=torch.int32),
                )
                self.out_cache_loc = torch.tensor([77], dtype=torch.int64)
                self.input_ids = self.output_ids
                self.output_ids = None
                self.seq_lens = self.seq_lens + 1
                self.seq_lens_cpu = self.seq_lens_cpu + 1
                self.orig_seq_lens = self.orig_seq_lens + 1
                self.seq_lens_sum += len(self.reqs)
                for req in self.reqs:
                    req.decode_batch_idx += 1
                    req.kv_committed_len += 1
                    req.kv_allocated_len += 1

        req_to_token = torch.tensor([[11, 12, 13, 0, 0]], dtype=torch.int32)
        allocator = _FakeAllocator()
        req_to_token_pool = SimpleNamespace(
            req_to_token=req_to_token,
            write=lambda target, values: req_to_token.__setitem__(target, values),
        )
        mock_init_new.side_effect = lambda **kwargs: _FakeBatch(
            reqs=kwargs["reqs"],
            req_to_token_pool=req_to_token_pool,
        )
        mock_sampling_info_from_schedule_batch.return_value = MagicMock()

        req = SimpleNamespace(
            req_pool_idx=0,
            output_ids=[9],
            origin_input_ids=[1, 2],
            kv_committed_len=3,
            kv_allocated_len=3,
            decode_batch_idx=0,
        )
        set_smc_reserved_kv_len(req, 3)

        fake_self = SimpleNamespace(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            _internal_tree_cache=object(),
            device="cpu",
        )
        model_config = SimpleNamespace(vocab_size=32000)

        batch = SMCWorkerV2._make_decode_batch(fake_self, [req], model_config)

        self.assertTrue(torch.equal(batch.out_cache_loc, torch.tensor([77], dtype=torch.int64)))
        self.assertTrue(
            torch.equal(
                req_to_token[0, :5],
                torch.tensor([11, 12, 13, 77, 0], dtype=torch.int32),
            )
        )
        self.assertTrue(torch.equal(batch.orig_seq_lens, torch.tensor([4], dtype=torch.int32)))
        self.assertEqual(req.kv_committed_len, 4)
        self.assertEqual(req.kv_allocated_len, 4)
        self.assertEqual(allocator.dec_calls, [])

    @patch(
        "sglang.srt.speculative.smc_worker_v2.speculative_moe_a2a_backend_context",
        return_value=nullcontext(),
    )
    @patch(
        "sglang.srt.speculative.smc_worker_v2.speculative_moe_backend_context",
        return_value=nullcontext(),
    )
    def test_forward_batch_generation_prepares_non_overlap_smc_batch(
        self,
        _mock_moe_backend_context,
        _mock_moe_a2a_backend_context,
    ):
        sampling_info = MagicMock()
        copied_sampling_info = MagicMock()
        sampling_info.copy_for_forward.return_value = copied_sampling_info
        draft_input = SMCDraftInput(
            last_token_ids=torch.tensor([7], dtype=torch.int32),
            new_seq_lens=torch.tensor([4], dtype=torch.int64),
            committed_seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
        )
        draft_input.prepare_for_decode = MagicMock()
        model_worker_batch = SimpleNamespace(
            seq_lens=torch.tensor([4], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            sampling_info=sampling_info,
        )
        batch = SimpleNamespace(
            spec_info=draft_input,
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            is_extend_in_batch=False,
            reqs=[SimpleNamespace(origin_input_ids=[1, 2, 3], output_ids=[7])],
            seq_lens=torch.tensor([4], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            get_model_worker_batch=MagicMock(return_value=model_worker_batch),
            sampling_info=sampling_info,
        )
        fake_self = SimpleNamespace(
            device="cpu",
            server_args=SimpleNamespace(speculative_num_draft_tokens=5),
            draft_worker=SimpleNamespace(
                draft_tp_context=lambda _group: nullcontext(),
                draft_runner=SimpleNamespace(tp_group=object()),
            ),
            _ensure_draft_prefix_filled=MagicMock(),
            _can_use_fused_draft_cuda_graph=MagicMock(return_value=False),
            _run_stepwise_draft_reqs=MagicMock(
                return_value=(
                    torch.tensor([[8]], dtype=torch.int32),
                    torch.tensor([0.25], dtype=torch.float32),
                    torch.tensor([1], dtype=torch.int32),
                    False,
                )
            ),
            _run_score_batch=MagicMock(
                return_value=(
                    torch.tensor([1], dtype=torch.int32),
                    torch.tensor([5], dtype=torch.int64),
                    torch.tensor([8], dtype=torch.int32),
                    torch.tensor([0.1], dtype=torch.float32),
                    False,
                )
            ),
            _build_empty_decode_result=MagicMock(),
            _empty_logits_output=MagicMock(),
        )
        fake_self._get_forward_model_worker_batch = (
            lambda batch_obj, is_overlap_batch: SMCWorkerV2._get_forward_model_worker_batch(
                fake_self,
                batch_obj,
                is_overlap_batch,
            )
        )

        result = SMCWorkerV2.forward_batch_generation(fake_self, batch)

        draft_input.prepare_for_decode.assert_called_once_with(batch)
        sampling_info.copy_for_forward.assert_called_once_with()
        self.assertIs(batch.sampling_info, sampling_info)
        self.assertIs(model_worker_batch.sampling_info, copied_sampling_info)
        fake_self._can_use_fused_draft_cuda_graph.assert_called_once_with(
            batch.reqs,
            copied_sampling_info,
        )
        fake_self._run_stepwise_draft_reqs.assert_called_once()
        _, score_kwargs = fake_self._run_score_batch.call_args
        self.assertIs(score_kwargs["base_model_worker_batch"], model_worker_batch)
        self.assertTrue(
            torch.equal(
                score_kwargs["draft_committed_lens"],
                torch.tensor([3], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                score_kwargs["anchor_token_ids"],
                torch.tensor([7], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                score_kwargs["draft_tokens"],
                torch.tensor([[8]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                score_kwargs["draft_logprobs"],
                torch.tensor([0.25], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                score_kwargs["draft_lengths"],
                torch.tensor([1], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(result.accept_lens, torch.tensor([1], dtype=torch.int32))
        )

    def test_make_score_model_worker_batch_uses_parent_target_temperature(self):
        fake_self = SimpleNamespace(
            server_args=SimpleNamespace(
                attention_backend="triton",
                smc_target_temperature=1.0,
                speculative_num_draft_tokens=5,
            ),
        )
        base_model_worker_batch = ModelWorkerBatch(
            forward_mode=ForwardMode.DECODE,
            input_ids=None,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([4], dtype=torch.int64),
            out_cache_loc=None,
            seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            seq_lens_sum=4,
            return_logprob=True,
            top_logprobs_nums=[0],
            token_ids_logprobs=[None],
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=False,
            all_extend_in_batch=False,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=None,
            extend_num_tokens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            multimodal_inputs=None,
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=None,
            sampling_info=SimpleNamespace(
                temperatures=torch.tensor([1.0], dtype=torch.float32),
            ),
            reqs=[
                SimpleNamespace(
                    smc_parent=SimpleNamespace(
                        sampling_params=SimpleNamespace(temperature=0.8)
                    ),
                    sampling_params=SimpleNamespace(temperature=1.0),
                )
            ],
        )

        score_batch = SMCWorkerV2._make_score_model_worker_batch(
            fake_self,
            base_model_worker_batch=base_model_worker_batch,
            draft_committed_lens=torch.tensor([3], dtype=torch.int64),
            anchor_token_ids=torch.tensor([7], dtype=torch.int32),
            draft_tokens=torch.tensor([[8]], dtype=torch.int32),
            draft_logprobs=torch.tensor([0.1], dtype=torch.float32),
            draft_lengths=torch.tensor([1], dtype=torch.int32),
        )

        self.assertAlmostEqual(score_batch.spec_info.target_temperature, 0.8)

    @patch("sglang.srt.speculative.smc_worker_v2.SMCDraftInput")
    def test_run_fused_draft_reqs_falls_back_when_replay_cannot_run(
        self,
        mock_draft_input_cls,
    ):
        fallback_result = (
            torch.tensor([[1, 2]], dtype=torch.int32),
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([2], dtype=torch.int32),
            False,
        )
        fake_self = SimpleNamespace(
            draft_worker=SimpleNamespace(
                smc_draft_cuda_graph_runner=MagicMock(),
                draft_runner=MagicMock(),
            ),
            req_to_token_pool=MagicMock(),
            smc_gamma=4,
            device="cpu",
            _run_stepwise_draft_reqs=MagicMock(return_value=fallback_result),
        )
        draft_input = MagicMock()
        draft_input.prepare_for_v2_draft.return_value = (MagicMock(), False)
        mock_draft_input_cls.return_value = draft_input

        reqs = [SimpleNamespace(rid="r0")]
        model_worker_batch = SimpleNamespace(seq_lens=torch.tensor([3], dtype=torch.int64))
        last_token_ids = torch.tensor([7], dtype=torch.int32)
        visible_seq_lens = torch.tensor([4], dtype=torch.int64)
        draft_committed_lens = torch.tensor([3], dtype=torch.int64)
        draft_sampling_info = MagicMock()

        result = SMCWorkerV2._run_fused_draft_reqs(
            fake_self,
            reqs,
            model_worker_batch,
            last_token_ids,
            draft_sampling_info,
            visible_seq_lens,
            draft_committed_lens,
        )

        self.assertIs(result, fallback_result)
        fake_self.draft_worker.smc_draft_cuda_graph_runner.replay.assert_not_called()

        args = fake_self._run_stepwise_draft_reqs.call_args.args
        self.assertIs(args[0], reqs)
        self.assertTrue(torch.equal(args[1], visible_seq_lens))
        self.assertTrue(torch.equal(args[2], draft_committed_lens))
        self.assertTrue(torch.equal(args[3], last_token_ids))

    def test_run_stepwise_draft_reqs_frees_temporary_allocations_on_restore(self):
        class _FakeAllocator:
            def __init__(self):
                self.dec_calls = []

            def dec_ref_and_free(self, indices):
                self.dec_calls.append(indices.clone())

        req_to_token = torch.tensor([[11, 12, 13, 99]], dtype=torch.int32)
        allocator = _FakeAllocator()
        req = SimpleNamespace(
            rid="r0",
            req_pool_idx=0,
            origin_input_ids=[1, 2],
            output_ids=[7, 8, 9],
            kv_committed_len=3,
            kv_allocated_len=3,
            draft_prefix_materialized=True,
            finished_reason=None,
            finished_len=None,
            finished_output=None,
            to_finish=None,
            decode_batch_idx=0,
            sampling_params=SimpleNamespace(
                max_new_tokens=8,
                ignore_eos=True,
                stop_token_ids=[],
            ),
            vocab_size=32000,
            eos_token_ids=None,
            tokenizer=None,
        )

        def fake_run_decode_batch(step_reqs, worker):
            step_reqs[0].kv_allocated_len += 1
            return SimpleNamespace(can_run_cuda_graph=False)

        fake_self = SimpleNamespace(
            smc_gamma=1,
            device="cpu",
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=allocator,
            draft_worker=SimpleNamespace(draft_worker=object()),
            _run_decode_batch=fake_run_decode_batch,
            _fill_draft_step_outputs=lambda result: ([17], [0.25]),
        )

        result = SMCWorkerV2._run_stepwise_draft_reqs(
            fake_self,
            [req],
            torch.tensor([5], dtype=torch.int64),
            torch.tensor([4], dtype=torch.int64),
            torch.tensor([9], dtype=torch.int32),
        )

        self.assertTrue(torch.equal(result[0], torch.tensor([[17]], dtype=torch.int32)))
        self.assertEqual(req.kv_committed_len, 3)
        self.assertEqual(req.kv_allocated_len, 3)
        self.assertFalse(req.draft_prefix_materialized)
        self.assertEqual(len(allocator.dec_calls), 1)
        self.assertTrue(torch.equal(allocator.dec_calls[0], torch.tensor([99])))

    def test_run_stepwise_draft_reqs_restores_overwritten_reserved_slots(self):
        class _FakeAllocator:
            def __init__(self):
                self.dec_calls = []

            def dec_ref_and_free(self, indices):
                self.dec_calls.append(indices.clone())

        req_to_token = torch.tensor([[11, 12, 13, 99, 0]], dtype=torch.int32)
        allocator = _FakeAllocator()
        req = SimpleNamespace(
            rid="r0",
            req_pool_idx=0,
            origin_input_ids=[1, 2],
            output_ids=[7, 8, 9],
            kv_committed_len=3,
            kv_allocated_len=4,
            draft_prefix_materialized=True,
            finished_reason=None,
            finished_len=None,
            finished_output=None,
            to_finish=None,
            decode_batch_idx=0,
            sampling_params=SimpleNamespace(
                max_new_tokens=8,
                ignore_eos=True,
                stop_token_ids=[],
            ),
            vocab_size=32000,
            eos_token_ids=None,
            tokenizer=None,
        )
        set_smc_reserved_kv_len(req, 4)

        def fake_run_decode_batch(step_reqs, worker):
            req_to_token[0, 3] = 77
            req_to_token[0, 4] = 88
            step_reqs[0].kv_allocated_len += 1
            return SimpleNamespace(can_run_cuda_graph=False)

        fake_self = SimpleNamespace(
            smc_gamma=1,
            device="cpu",
            req_to_token_pool=SimpleNamespace(
                req_to_token=req_to_token,
                write=lambda target, values: req_to_token.__setitem__(target, values),
            ),
            token_to_kv_pool_allocator=allocator,
            draft_worker=SimpleNamespace(draft_worker=object()),
            _run_decode_batch=fake_run_decode_batch,
            _fill_draft_step_outputs=lambda result: ([17], [0.25]),
        )

        SMCWorkerV2._run_stepwise_draft_reqs(
            fake_self,
            [req],
            torch.tensor([5], dtype=torch.int64),
            torch.tensor([4], dtype=torch.int64),
            torch.tensor([9], dtype=torch.int32),
        )

        self.assertTrue(
            torch.equal(req_to_token[0, :4], torch.tensor([11, 12, 13, 99], dtype=torch.int32))
        )
        self.assertEqual(req.kv_committed_len, 3)
        self.assertEqual(req.kv_allocated_len, 4)
        self.assertFalse(req.draft_prefix_materialized)
        self.assertEqual(len(allocator.dec_calls), 1)
        self.assertTrue(
            torch.equal(allocator.dec_calls[0], torch.tensor([77, 88], dtype=torch.int64))
        )

    def test_run_stepwise_draft_reqs_ignores_dummy_zero_slots_on_restore(self):
        class _FakeAllocator:
            def __init__(self):
                self.dec_calls = []

            def dec_ref_and_free(self, indices):
                self.dec_calls.append(indices.clone())

        req_to_token = torch.tensor([[11, 12, 13, 0]], dtype=torch.int32)
        allocator = _FakeAllocator()
        req = SimpleNamespace(
            rid="r0",
            req_pool_idx=0,
            origin_input_ids=[1, 2],
            output_ids=[7, 8, 9],
            kv_committed_len=3,
            kv_allocated_len=3,
            draft_prefix_materialized=True,
            finished_reason=None,
            finished_len=None,
            finished_output=None,
            to_finish=None,
            decode_batch_idx=0,
            sampling_params=SimpleNamespace(
                max_new_tokens=8,
                ignore_eos=True,
                stop_token_ids=[],
            ),
            vocab_size=32000,
            eos_token_ids=None,
            tokenizer=None,
        )

        def fake_run_decode_batch(step_reqs, worker):
            step_reqs[0].kv_allocated_len += 1
            return SimpleNamespace(can_run_cuda_graph=False)

        fake_self = SimpleNamespace(
            smc_gamma=1,
            device="cpu",
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=allocator,
            draft_worker=SimpleNamespace(draft_worker=object()),
            _run_decode_batch=fake_run_decode_batch,
            _fill_draft_step_outputs=lambda result: ([17], [0.25]),
        )

        SMCWorkerV2._run_stepwise_draft_reqs(
            fake_self,
            [req],
            torch.tensor([5], dtype=torch.int64),
            torch.tensor([4], dtype=torch.int64),
            torch.tensor([9], dtype=torch.int32),
        )

        self.assertEqual(req.kv_committed_len, 3)
        self.assertEqual(req.kv_allocated_len, 3)
        self.assertFalse(req.draft_prefix_materialized)
        self.assertEqual(allocator.dec_calls, [])

    @patch("sglang.srt.speculative.smc_worker_v2.SMCDraftInput")
    def test_run_fused_draft_reqs_keeps_prefix_materialized_when_replay_runs(
        self,
        mock_draft_input_cls,
    ):
        runner = MagicMock()
        runner.replay.return_value = (
            torch.tensor([[11, 13]], dtype=torch.int32),
            torch.tensor([[0.2, 0.3]], dtype=torch.float32),
        )
        fake_self = SimpleNamespace(
            draft_worker=SimpleNamespace(
                smc_draft_cuda_graph_runner=runner,
                draft_runner=MagicMock(),
            ),
            req_to_token_pool=MagicMock(),
            smc_gamma=2,
            device="cpu",
            _run_stepwise_draft_reqs=MagicMock(),
        )
        draft_input = MagicMock()
        draft_input.prepare_for_v2_draft.return_value = (MagicMock(), True)
        mock_draft_input_cls.return_value = draft_input

        req = SimpleNamespace(
            rid="r0",
            origin_input_ids=[1, 2],
            output_ids=[7, 8],
            sampling_params=SimpleNamespace(
                max_new_tokens=8,
                ignore_eos=True,
                stop_token_ids=[],
            ),
            vocab_size=32000,
            eos_token_ids=None,
            tokenizer=None,
            draft_prefix_materialized=False,
        )
        result = SMCWorkerV2._run_fused_draft_reqs(
            fake_self,
            [req],
            SimpleNamespace(seq_lens=torch.tensor([3], dtype=torch.int64)),
            torch.tensor([7], dtype=torch.int32),
            MagicMock(),
            torch.tensor([4], dtype=torch.int64),
            torch.tensor([3], dtype=torch.int64),
        )

        self.assertTrue(
            torch.equal(result[0], torch.tensor([[11, 13]], dtype=torch.int32))
        )
        self.assertTrue(torch.equal(result[1], torch.tensor([0.5], dtype=torch.float32)))
        self.assertTrue(torch.equal(result[2], torch.tensor([2], dtype=torch.int32)))
        self.assertTrue(req.draft_prefix_materialized)
        fake_self._run_stepwise_draft_reqs.assert_not_called()

    @patch("sglang.srt.speculative.smc_worker_v2.SMCDraftInput")
    def test_run_fused_draft_reqs_truncates_to_committed_prefix(
        self,
        mock_draft_input_cls,
    ):
        runner = MagicMock()
        runner.replay.return_value = (
            torch.tensor([[11, 13]], dtype=torch.int32),
            torch.tensor([[0.2, 0.3]], dtype=torch.float32),
        )
        fake_self = SimpleNamespace(
            draft_worker=SimpleNamespace(
                smc_draft_cuda_graph_runner=runner,
                draft_runner=MagicMock(),
            ),
            req_to_token_pool=MagicMock(),
            smc_gamma=2,
            device="cpu",
            _run_stepwise_draft_reqs=MagicMock(),
        )
        draft_input = MagicMock()
        draft_input.prepare_for_v2_draft.return_value = (MagicMock(), True)
        mock_draft_input_cls.return_value = draft_input

        req = SimpleNamespace(
            rid="r0",
            origin_input_ids=[1, 2],
            output_ids=[7, 8, 9],
            sampling_params=SimpleNamespace(
                max_new_tokens=4,
                ignore_eos=True,
                stop_token_ids=[],
            ),
            vocab_size=32000,
            eos_token_ids=None,
            tokenizer=None,
            draft_prefix_materialized=False,
        )
        result = SMCWorkerV2._run_fused_draft_reqs(
            fake_self,
            [req],
            SimpleNamespace(seq_lens=torch.tensor([5], dtype=torch.int64)),
            torch.tensor([9], dtype=torch.int32),
            MagicMock(),
            torch.tensor([5], dtype=torch.int64),
            torch.tensor([4], dtype=torch.int64),
        )

        self.assertTrue(
            torch.equal(result[0], torch.tensor([[11, 13]], dtype=torch.int32))
        )
        self.assertTrue(torch.equal(result[1], torch.tensor([0.2], dtype=torch.float32)))
        self.assertTrue(torch.equal(result[2], torch.tensor([1], dtype=torch.int32)))
        self.assertTrue(req.draft_prefix_materialized)

    def test_run_score_batch_logprob_only_skips_sampling(self):
        score_input = MagicMock()
        score_input.prepare_for_v2_verify.return_value = (MagicMock(), True)
        score_input.compute_logprob_diffs.return_value = torch.tensor(
            [0.25], dtype=torch.float32
        )
        score_input.sample = MagicMock()
        model_worker_batch = SimpleNamespace(spec_info=score_input)
        target_worker = MagicMock()
        target_worker.forward_batch_generation.return_value = SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=torch.tensor([0.0]))
        )
        fake_self = SimpleNamespace(
            _make_score_model_worker_batch=MagicMock(return_value=model_worker_batch),
            req_to_token_pool=MagicMock(),
            target_worker=target_worker,
        )

        diffs, can_run_cuda_graph = SMCWorkerV2._run_score_batch(
            fake_self,
            base_model_worker_batch=MagicMock(),
            draft_committed_lens=torch.tensor([3], dtype=torch.int64),
            anchor_token_ids=torch.tensor([7], dtype=torch.int32),
            draft_tokens=torch.tensor([[8]], dtype=torch.int32),
            draft_logprobs=torch.tensor([0.0], dtype=torch.float32),
            draft_lengths=torch.tensor([1], dtype=torch.int32),
            logprob_only=True,
        )

        self.assertTrue(torch.equal(diffs, torch.tensor([0.25], dtype=torch.float32)))
        self.assertTrue(can_run_cuda_graph)
        score_input.compute_logprob_diffs.assert_called_once()
        score_input.sample.assert_not_called()
