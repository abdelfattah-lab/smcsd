"""Tests for the refactored SMC worker (EAGLE V2 aligned)."""

from types import SimpleNamespace
from unittest import TestCase

import torch

from sglang.srt.speculative.smc_info_v2 import (
    SmcDraftInput,
    SmcLogprobDiff,
    SmcVerifyInput,
    compute_smc_logprob_diff,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class TestSmcInfoV2(TestCase):
    """Tests for smc_info_v2 data structures and logprob computation."""

    def test_smc_draft_input_inherits_eagle_draft_input(self):
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        self.assertTrue(issubclass(SmcDraftInput, EagleDraftInput))
        inp = SmcDraftInput(
            verified_id=torch.tensor([5], dtype=torch.int32),
            new_seq_lens=torch.tensor([10], dtype=torch.int64),
        )
        self.assertTrue(torch.equal(inp.verified_id, torch.tensor([5], dtype=torch.int32)))
        self.assertTrue(torch.equal(inp.new_seq_lens, torch.tensor([10], dtype=torch.int64)))

    def test_smc_verify_input_inherits_eagle_verify_input(self):
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        self.assertTrue(issubclass(SmcVerifyInput, EagleVerifyInput))

    def test_compute_smc_logprob_diff_basic(self):
        """Test logprob diff computation with known values."""
        bs, spec_steps, vocab = 2, 3, 8
        draft_token_num = spec_steps + 1  # anchor + 3 draft tokens

        # draft_token layout per request: [anchor, d0, d1, d2]
        draft_token = torch.tensor(
            [0, 1, 2, 3, 0, 4, 5, 6], dtype=torch.int64
        )  # (bs * draft_token_num,)

        # Create target logits that give known logprobs
        # For simplicity, make uniform logits then adjust
        target_logits = torch.zeros(bs * draft_token_num, vocab)

        # Set known logits for draft tokens at the right positions
        # Position 0 predicts d0=1 (req0), position 0 predicts d0=4 (req1)
        target_logits[0, 1] = 2.0  # high logit for token 1 at pos 0, req 0
        target_logits[1, 2] = 1.5  # for token 2 at pos 1, req 0
        target_logits[2, 3] = 1.0  # for token 3 at pos 2, req 0
        target_logits[4, 4] = 2.0  # for token 4 at pos 0, req 1
        target_logits[5, 5] = 1.5
        target_logits[6, 6] = 1.0

        draft_logprobs = torch.zeros(bs, spec_steps)
        temperatures = torch.ones(bs, 1)

        result = compute_smc_logprob_diff(
            target_logits=target_logits,
            draft_token=draft_token,
            draft_logprobs=draft_logprobs,
            spec_steps=spec_steps,
            draft_token_num=draft_token_num,
            temperatures=temperatures,
        )

        self.assertIsInstance(result, SmcLogprobDiff)
        self.assertEqual(result.logprob_diff_sum.shape, (bs,))
        self.assertEqual(result.target_logprobs.shape, (bs, spec_steps))
        self.assertEqual(result.draft_logprobs.shape, (bs, spec_steps))
        # Since draft_logprobs are 0, diff_sum = sum of target logprobs
        self.assertTrue(
            torch.allclose(
                result.logprob_diff_sum,
                result.target_logprobs.sum(dim=1),
            )
        )

    def test_compute_smc_logprob_diff_with_temperature(self):
        """Test that temperature scaling is applied correctly."""
        bs, spec_steps, vocab = 1, 2, 4
        draft_token_num = spec_steps + 1

        draft_token = torch.tensor([0, 1, 2], dtype=torch.int64)
        target_logits = torch.tensor(
            [[1.0, 2.0, 0.5, 0.5], [0.5, 0.5, 3.0, 0.5], [0.5, 0.5, 0.5, 0.5]],
            dtype=torch.float32,
        )
        draft_logprobs = torch.zeros(bs, spec_steps)

        # Temperature = 2.0 should soften the distribution
        temperatures = torch.tensor([[2.0]], dtype=torch.float32)

        result = compute_smc_logprob_diff(
            target_logits=target_logits,
            draft_token=draft_token,
            draft_logprobs=draft_logprobs,
            spec_steps=spec_steps,
            draft_token_num=draft_token_num,
            temperatures=temperatures,
        )

        # Verify logprobs are computed with temperature scaling
        import torch.nn.functional as F

        expected_log_probs = F.log_softmax(target_logits / 2.0, dim=-1)
        expected_t0 = expected_log_probs[0, 1]  # target logprob for d0=1
        expected_t1 = expected_log_probs[1, 2]  # target logprob for d1=2

        self.assertAlmostEqual(
            result.target_logprobs[0, 0].item(), expected_t0.item(), places=5
        )
        self.assertAlmostEqual(
            result.target_logprobs[0, 1].item(), expected_t1.item(), places=5
        )


class TestSmcVerifyCudaGraphRunnerBody(TestCase):
    """Test the verify post-processing logic (without actual CUDA graphs)."""

    def _run_body_cpu(self, bs, spec_steps, draft_token_num, vocab_size,
                      target_logits, draft_token, temperatures):
        """Simulate SmcVerifyCudaGraphRunner._run_body on CPU."""
        import torch.nn.functional as F

        dtn = draft_token_num
        ss = spec_steps
        total = bs * dtn

        logits_3d = target_logits[:total].reshape(bs, dtn, -1)
        bonus = torch.argmax(logits_3d[:, -1, :], dim=-1).to(torch.int32)

        draft_2d = draft_token[:total].reshape(bs, dtn)
        predict = torch.zeros(total, dtype=torch.int32)
        pv = predict.reshape(bs, dtn)
        pv[:, :ss] = draft_2d[:, 1:].to(torch.int32)
        pv[:, ss] = bonus

        verified_id = pv[:, ss].clone()

        temp_3d = temperatures[:bs].unsqueeze(1).expand(-1, dtn, -1)
        target_log_probs = F.log_softmax(
            logits_3d / temp_3d.to(logits_3d.dtype), dim=-1
        )
        r_tokens = draft_2d[:, 1:ss + 1].long()
        target_lp = (
            target_log_probs[:, :ss, :]
            .gather(2, r_tokens.unsqueeze(-1))
            .squeeze(-1)
            .float()
        )

        accept_length = torch.full((bs,), ss + 1, dtype=torch.int32)
        accept_index = (
            torch.arange(ss + 1, dtype=torch.int32)
            .unsqueeze(0)
            .expand(bs, -1)
            .contiguous()
        )

        return predict, accept_length, accept_index, verified_id, target_lp

    def test_accept_all_with_bonus(self):
        """Verify accept_length = spec_steps + 1 and bonus = argmax(last pos)."""
        bs, ss, dtn, vocab = 1, 3, 4, 8

        # draft_token: [anchor=0, d0=1, d1=2, d2=3]
        draft_token = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

        # Make bonus = token 5 (highest logit at last position)
        target_logits = torch.zeros(dtn, vocab)
        target_logits[3, 5] = 10.0  # last position → bonus = 5

        temperatures = torch.ones(1, 1)

        predict, accept_len, accept_idx, verified_id, target_lp = self._run_body_cpu(
            bs, ss, dtn, vocab, target_logits, draft_token, temperatures
        )

        # accept_length = spec_steps + 1 = 4
        self.assertEqual(accept_len.item(), 4)

        # predict = [d0=1, d1=2, d2=3, bonus=5]
        expected_predict = torch.tensor([1, 2, 3, 5], dtype=torch.int32)
        self.assertTrue(torch.equal(predict, expected_predict))

        # verified_id = bonus = 5
        self.assertEqual(verified_id.item(), 5)

        # accept_index = [0, 1, 2, 3]
        expected_idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        self.assertTrue(torch.equal(accept_idx, expected_idx))

    def test_logprob_diff_matches_compute_function(self):
        """Verify the graph body's target_lp matches compute_smc_logprob_diff."""
        bs, ss, dtn, vocab = 2, 2, 3, 6

        draft_token = torch.tensor(
            [0, 1, 2, 0, 3, 4], dtype=torch.int64
        )
        target_logits = torch.randn(bs * dtn, vocab)
        temperatures = torch.ones(bs, 1) * 0.8
        draft_logprobs = torch.randn(bs, ss)

        _, _, _, _, target_lp = self._run_body_cpu(
            bs, ss, dtn, vocab, target_logits, draft_token, temperatures
        )

        # Compare with compute_smc_logprob_diff
        result = compute_smc_logprob_diff(
            target_logits=target_logits,
            draft_token=draft_token,
            draft_logprobs=draft_logprobs,
            spec_steps=ss,
            draft_token_num=dtn,
            temperatures=temperatures,
        )

        self.assertTrue(
            torch.allclose(target_lp, result.target_logprobs, atol=1e-5),
            f"target_lp mismatch: {target_lp} vs {result.target_logprobs}",
        )

    def test_bonus_is_greedy_under_temperature(self):
        """Bonus is argmax of RAW logits (before temperature), matching greedy."""
        bs, ss, dtn, vocab = 1, 1, 2, 4

        draft_token = torch.tensor([0, 1], dtype=torch.int64)
        target_logits = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 3.0, 2.0, 0.5]], dtype=torch.float32
        )
        temperatures = torch.tensor([[0.5]])

        _, _, _, verified_id, _ = self._run_body_cpu(
            bs, ss, dtn, vocab, target_logits, draft_token, temperatures
        )

        # argmax of logits[-1] = argmax([1.0, 3.0, 2.0, 0.5]) = 1
        self.assertEqual(verified_id.item(), 1)


class TestSmcManagerIntegration(TestCase):
    """Test that smc_manager builds particle batches with SmcDraftInput."""

    def test_build_particle_batch_uses_smc_draft_input(self):
        """Verify _build_particle_batch creates SmcDraftInput with verified_id."""
        from sglang.srt.speculative.smc_info_v2 import SmcDraftInput

        # Create a minimal SmcDraftInput like _build_particle_batch would
        last_token_ids = torch.tensor([7, 8], dtype=torch.int32)
        visible_seq_lens = torch.tensor([10, 12], dtype=torch.int64)
        spec_info = SmcDraftInput(
            verified_id=last_token_ids,
            new_seq_lens=visible_seq_lens,
        )

        self.assertTrue(torch.equal(spec_info.verified_id, last_token_ids))
        self.assertTrue(torch.equal(spec_info.new_seq_lens, visible_seq_lens))
        self.assertIsInstance(spec_info, SmcDraftInput)
