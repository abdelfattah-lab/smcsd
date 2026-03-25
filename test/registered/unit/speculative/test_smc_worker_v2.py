from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative.smc_info import set_smc_reserved_kv_len
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
        self.assertEqual(req.kv_committed_len, 4)
        self.assertEqual(req.kv_allocated_len, 5)
        self.assertEqual(len(allocator.dec_calls), 1)
        self.assertTrue(torch.equal(allocator.dec_calls[0], torch.tensor([77], dtype=torch.int64)))

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
        self.assertEqual(len(allocator.dec_calls), 1)
        self.assertTrue(
            torch.equal(allocator.dec_calls[0], torch.tensor([77, 88], dtype=torch.int64))
        )
