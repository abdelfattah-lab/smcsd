from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.triton_backend import TritonMultiStepDraftBackend
from sglang.srt.speculative.smc_info import SMCDraftInput
from sglang.srt.speculative.smc_worker_v2 import SMCDraftWorker, SMCWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class TestSMCDraftWorker(TestCase):
    def test_draft_uses_live_committed_prefix_lengths(self):
        worker = SMCDraftWorker.__new__(SMCDraftWorker)
        observed = {}

        def run_draft(reqs, model_worker_batch, last_token_ids, draft_sampling_info):
            observed["seq_lens"] = model_worker_batch.seq_lens.clone()
            observed["last_token_ids"] = last_token_ids.clone()
            observed["sampling_info"] = draft_sampling_info
            return (
                torch.tensor([[31, 32, 33, 34], [41, 42, 43, 44]], dtype=torch.int32),
                torch.tensor([0.1, 0.2], dtype=torch.float32),
                torch.tensor([4, 4], dtype=torch.int32),
                False,
            )

        worker._smc_outer_worker = SimpleNamespace(
            device=torch.device("cpu"),
            server_args=SimpleNamespace(
                speculative_num_draft_tokens=5,
                smc_target_temperature=1.0,
                attention_backend="triton",
            ),
            smc_gamma=4,
            _run_eagle_style_draft_reqs=run_draft,
        )

        model_worker_batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    origin_input_ids=[1, 2, 3],
                    output_ids=[17],
                    origin_input_text="prompt a",
                    smc_group_id="g1",
                    smc_particle_idx=0,
                ),
                SimpleNamespace(
                    origin_input_ids=[4, 5, 6, 7],
                    output_ids=[19],
                    origin_input_text="prompt b",
                    smc_group_id="g2",
                    smc_particle_idx=0,
                ),
            ],
            seq_lens=torch.tensor([5, 7], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5, 7], dtype=torch.int64),
            spec_info=SimpleNamespace(
                last_token_ids=torch.tensor([17, 19], dtype=torch.int32),
            ),
            sampling_info=MagicMock(),
        )

        verify_input = worker.draft(model_worker_batch)

        self.assertTrue(
            torch.equal(
                observed["seq_lens"],
                torch.tensor([5, 7], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                verify_input.positions,
                torch.tensor([5, 6, 7, 8, 9, 7, 8, 9, 10, 11], dtype=torch.int64),
            )
        )

    def test_materialize_parent_draft_prefix_uses_shared_prefix_len(self):
        worker = SMCWorkerV2.__new__(SMCWorkerV2)
        worker._draft_worker = SimpleNamespace(
            draft_tp_context=lambda tp_group: nullcontext(),
            draft_runner=SimpleNamespace(tp_group=object()),
            draft_worker=SimpleNamespace(),
        )
        worker._run_draft_prefix_fill_batch = MagicMock()

        req = SimpleNamespace(
            rid="parent-1",
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[9],
            kv_committed_len=6,
        )

        worker.materialize_smc_parent_draft_prefix(req)

        worker._run_draft_prefix_fill_batch.assert_called_once_with(
            [req],
            [4],
        )


class TestSMCWorkerV2(TestCase):
    def test_triton_smc_graph_replay_uses_raw_last_step_seq_lens_for_kv_splits(self):
        class _FakeKernel:
            def __init__(self):
                self.calls = []

            def __getitem__(self, grid):
                def _launch(
                    req_pool_indices,
                    req_to_token,
                    base_seq_lens,
                    cuda_graph_kv_indices,
                    kv_indptr,
                    raw_bs,
                    pool_len,
                    kv_indices_stride,
                    kv_indptr_stride,
                    bs_upper,
                    num_steps_upper,
                ):
                    self.calls.append(
                        {
                            "grid": grid,
                            "base_seq_lens": base_seq_lens.clone(),
                            "raw_bs": raw_bs,
                            "pool_len": pool_len,
                        }
                    )

                return _launch

        fake_kernel = _FakeKernel()
        fake_last_backend = SimpleNamespace(
            cuda_graph_num_kv_splits=torch.zeros((8,), dtype=torch.int32),
            get_num_kv_splits=MagicMock(),
        )
        fake_backend = SimpleNamespace(
            speculative_num_steps=5,
            generate_smc_draft_decode_kv_indices=fake_kernel,
            req_to_token=torch.zeros((4, 32), dtype=torch.int32),
            cuda_graph_kv_indices=torch.zeros((5, 64), dtype=torch.int64),
            kv_indptr=torch.zeros((5, 8), dtype=torch.int32),
            pool_len=32,
            attn_backends=[object(), object(), object(), object(), fake_last_backend],
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 3], dtype=torch.int64),
            seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        )

        TritonMultiStepDraftBackend.init_smc_forward_metadata_replay_cuda_graph(
            fake_backend,
            forward_batch=forward_batch,
            bs=2,
            raw_bs=2,
        )

        self.assertEqual(len(fake_kernel.calls), 1)
        self.assertEqual(fake_kernel.calls[0]["grid"], (4, 2))
        self.assertTrue(
            torch.equal(
                fake_kernel.calls[0]["base_seq_lens"],
                torch.tensor([5, 7], dtype=torch.int32),
            )
        )
        fake_last_backend.get_num_kv_splits.assert_called_once()
        split_arg = fake_last_backend.get_num_kv_splits.call_args.args[1]
        self.assertTrue(
            torch.equal(split_arg, torch.tensor([8, 10], dtype=torch.int32))
        )

    def test_run_eagle_style_draft_reqs_uses_draft_forward_when_prepare_rejects_graph(
        self,
    ):
        @dataclass
        class _Batch:
            reqs: list
            seq_lens: torch.Tensor
            seq_lens_cpu: torch.Tensor
            seq_lens_sum: int
            sampling_info: object

        worker = SMCWorkerV2.__new__(SMCWorkerV2)
        worker.device = torch.device("cpu")
        worker.smc_gamma = 4
        worker.req_to_token_pool = SimpleNamespace(req_to_token=torch.zeros((4, 16)))
        fake_forward_batch = MagicMock()
        fake_forward_batch.forward_mode.is_idle.return_value = False
        fake_token_matrix = torch.tensor(
            [[31, 32, 33, 34], [41, 42, 43, 44]], dtype=torch.int32
        )
        fake_logprob_matrix = torch.tensor(
            [[0.01, 0.02, 0.03, 0.04], [0.11, 0.12, 0.13, 0.14]],
            dtype=torch.float32,
        )
        worker._draft_worker = SimpleNamespace(
            smc_draft_cuda_graph_runner=MagicMock(),
            draft_runner=MagicMock(),
            smc_draft_attn_backend=SimpleNamespace(init_forward_metadata=MagicMock()),
            draft_forward=MagicMock(return_value=(fake_token_matrix, fake_logprob_matrix)),
        )

        reqs = [SimpleNamespace(), SimpleNamespace()]
        committed_seq_lens = torch.tensor([5, 7], dtype=torch.int64)
        model_worker_batch = _Batch(
            reqs=reqs,
            seq_lens=committed_seq_lens,
            seq_lens_cpu=committed_seq_lens.cpu(),
            seq_lens_sum=int(committed_seq_lens.sum().item()),
            sampling_info=MagicMock(),
        )
        last_token_ids = torch.tensor([17, 19], dtype=torch.int32)
        draft_sampling_info = MagicMock()

        observed = {}

        def _prepare(
            self_input,
            req_to_token_pool,
            batch,
            cuda_graph_runner,
            draft_model_runner,
            gamma,
            draft_sampling_info,
        ):
            observed["seq_lens"] = batch.seq_lens.clone()
            observed["seq_lens_cpu"] = batch.seq_lens_cpu.clone()
            observed["seq_lens_sum"] = batch.seq_lens_sum
            observed["new_seq_lens"] = self_input.new_seq_lens.clone()
            return fake_forward_batch, False

        with patch.object(
            SMCDraftInput,
            "prepare_for_v2_draft",
            autospec=True,
            side_effect=_prepare,
        ):
            result = worker._run_eagle_style_draft_reqs(
                reqs,
                model_worker_batch,
                last_token_ids,
                draft_sampling_info,
            )

        token_matrix, draft_logprobs, draft_lengths, can_cuda_graph = result
        self.assertTrue(torch.equal(token_matrix, fake_token_matrix))
        self.assertTrue(
            torch.allclose(
                draft_logprobs,
                torch.tensor([0.10, 0.50], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(draft_lengths, torch.tensor([4, 4], dtype=torch.int32))
        )
        self.assertFalse(can_cuda_graph)
        self.assertTrue(
            torch.equal(observed["seq_lens"], torch.tensor([5, 7], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                observed["seq_lens_cpu"], torch.tensor([5, 7], dtype=torch.int64)
            )
        )
        self.assertEqual(observed["seq_lens_sum"], 12)
        self.assertTrue(
            torch.equal(observed["new_seq_lens"], torch.tensor([5, 7], dtype=torch.int64))
        )
        worker.draft_worker.smc_draft_attn_backend.init_forward_metadata.assert_called_once_with(
            fake_forward_batch
        )
        worker.draft_worker.draft_forward.assert_called_once_with(fake_forward_batch)

    def test_draft_extend_for_decode_falls_back_without_cuda_graph(self):
        worker = SMCDraftWorker.__new__(SMCDraftWorker)
        worker.device = torch.device("cpu")
        worker.speculative_num_steps = 4
        worker.speculative_num_draft_tokens = 5
        worker.topk = 1
        worker.plan_stream = None
        worker.plan_stream_ctx = nullcontext()
        worker.cuda_graph_runner_for_draft_extend = MagicMock()
        worker.cuda_graph_runner_for_draft_extend.can_run.return_value = False

        logits_output = SimpleNamespace(
            next_token_logits=torch.arange(40, dtype=torch.float32).reshape(10, 4),
            hidden_states=torch.arange(30, dtype=torch.float32).reshape(10, 3),
        )
        worker.draft_runner = MagicMock()
        worker.draft_runner.forward.return_value = SimpleNamespace(
            logits_output=logits_output
        )

        batch = SimpleNamespace(seq_lens=torch.tensor([6, 8], dtype=torch.int64))
        batch_result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=torch.zeros((2, 3))),
            accept_lens=torch.tensor([2, 4], dtype=torch.int32),
            next_token_ids=torch.tensor([17, 19], dtype=torch.int32),
            next_draft_input=SimpleNamespace(),
        )
        fake_forward_batch = SimpleNamespace(spec_info=SimpleNamespace(accept_length=None))

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.EagleDraftInput.prepare_for_extend_to_fill_draft_kvcache",
            return_value=fake_forward_batch,
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2.fast_topk",
            return_value=(
                torch.tensor([[0.9], [0.8]], dtype=torch.float32),
                torch.tensor([[3], [2]], dtype=torch.int64),
            ),
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2.maybe_detect_nan"
        ):
            worker._draft_extend_for_decode(batch, batch_result)

        worker.cuda_graph_runner_for_draft_extend.replay.assert_not_called()
        worker.draft_runner.forward.assert_called_once_with(
            fake_forward_batch,
            skip_attn_backend_init=True,
        )
        self.assertTrue(
            torch.equal(
                batch_result.next_draft_input.topk_p,
                torch.tensor([[0.9], [0.8]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                batch_result.next_draft_input.topk_index,
                torch.tensor([[3], [2]], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                batch_result.next_draft_input.hidden_states,
                torch.tensor([[3.0, 4.0, 5.0], [24.0, 25.0, 26.0]], dtype=torch.float32),
            )
        )
