from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative.smc_info import SMCDraftInput
from sglang.srt.speculative.smc_worker_v2 import SMCDraftWorker, SMCWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class TestSMCDraftWorker(TestCase):
    def test_draft_uses_committed_prefix_lengths(self):
        worker = SMCDraftWorker.__new__(SMCDraftWorker)
        observed = {}

        def record_prefix_fill(reqs, committed_lens):
            observed["fill_lens"] = list(committed_lens)

        def run_stepwise(reqs, visible_seq_lens, draft_committed_lens, last_token_ids):
            observed["visible_seq_lens"] = visible_seq_lens.clone()
            observed["draft_committed_lens"] = draft_committed_lens.clone()
            observed["last_token_ids"] = last_token_ids.clone()
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
            _ensure_draft_prefix_filled=record_prefix_fill,
            _can_use_fused_draft_cuda_graph=lambda reqs, sampling_info: False,
            _run_stepwise_draft_reqs=run_stepwise,
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
            seq_lens=torch.tensor([6, 8], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([6, 8], dtype=torch.int64),
            spec_info=SimpleNamespace(
                last_token_ids=torch.tensor([17, 19], dtype=torch.int32),
            ),
            sampling_info=MagicMock(),
        )

        verify_input = worker.draft(model_worker_batch)

        self.assertEqual(observed["fill_lens"], [5, 7])
        self.assertTrue(
            torch.equal(
                observed["draft_committed_lens"],
                torch.tensor([5, 7], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                verify_input.positions,
                torch.tensor([5, 6, 7, 8, 9, 7, 8, 9, 10, 11], dtype=torch.int64),
            )
        )


class TestSMCWorkerV2(TestCase):
    def test_can_use_fused_draft_cuda_graph_requires_runner_support(self):
        worker = SMCWorkerV2.__new__(SMCWorkerV2)
        sampling_info = MagicMock()
        reqs = [SimpleNamespace()]

        worker._draft_worker = SimpleNamespace(smc_draft_cuda_graph_runner=None)
        self.assertFalse(
            worker._can_use_fused_draft_cuda_graph(reqs, sampling_info)
        )

        runner = MagicMock()
        runner.supports_replay.return_value = False
        worker._draft_worker = SimpleNamespace(smc_draft_cuda_graph_runner=runner)
        self.assertFalse(
            worker._can_use_fused_draft_cuda_graph(reqs, sampling_info)
        )
        runner.supports_replay.assert_called_once_with(reqs, sampling_info)

        runner.supports_replay.reset_mock()
        runner.supports_replay.return_value = True
        self.assertTrue(
            worker._can_use_fused_draft_cuda_graph(reqs, sampling_info)
        )
        runner.supports_replay.assert_called_once_with(reqs, sampling_info)

    def test_run_fused_draft_reqs_falls_back_to_stepwise_when_prepare_rejects_graph(
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
        worker._draft_worker = SimpleNamespace(
            smc_draft_cuda_graph_runner=MagicMock(),
            draft_runner=MagicMock(),
        )

        reqs = [SimpleNamespace(), SimpleNamespace()]
        visible_seq_lens = torch.tensor([6, 8], dtype=torch.int64)
        draft_committed_lens = torch.tensor([5, 7], dtype=torch.int64)
        model_worker_batch = _Batch(
            reqs=reqs,
            seq_lens=visible_seq_lens,
            seq_lens_cpu=visible_seq_lens.cpu(),
            seq_lens_sum=int(visible_seq_lens.sum().item()),
            sampling_info=MagicMock(),
        )
        last_token_ids = torch.tensor([17, 19], dtype=torch.int32)
        draft_sampling_info = MagicMock()

        stepwise_result = ("tokens", "logprobs", "lengths", False)
        worker._run_stepwise_draft_reqs = MagicMock(return_value=stepwise_result)

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
            return MagicMock(), False

        with patch.object(
            SMCDraftInput,
            "prepare_for_v2_draft",
            autospec=True,
            side_effect=_prepare,
        ):
            result = worker._run_fused_draft_reqs(
                reqs,
                model_worker_batch,
                last_token_ids,
                draft_sampling_info,
                visible_seq_lens,
                draft_committed_lens,
            )

        self.assertEqual(result, stepwise_result)
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
            torch.equal(observed["new_seq_lens"], torch.tensor([6, 8], dtype=torch.int64))
        )
        worker._run_stepwise_draft_reqs.assert_called_once_with(
            reqs,
            visible_seq_lens,
            draft_committed_lens,
            last_token_ids,
        )

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
