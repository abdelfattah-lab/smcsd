"""CPU regression tests for cross-tokenizer visible-prefix weighting."""

from __future__ import annotations

import unittest

import torch

from smcsd.core.req_state import bonus_weight_mask, cross_weight_cutoff


class TestCrossWeightCutoff(unittest.TestCase):
    def _cutoff(
        self,
        *,
        proxy_len: int,
        first_eos: int,
        eos_hit: bool,
        length_hit: bool,
        max_tokens: int,
        prior_tokens: int,
        width: int = 8,
    ) -> int:
        return int(
            cross_weight_cutoff(
                proxy_lens=torch.tensor([proxy_len]),
                n_weight_cols=width,
                first_eos=torch.tensor([first_eos]),
                eos_hit=torch.tensor([eos_hit]),
                length_hit=torch.tensor([length_hit]),
                max_tokens=torch.tensor([max_tokens]),
                prior_token_counts=torch.tensor([prior_tokens]),
            ).item()
        )

    def test_default_uses_all_visible_proxy_tokens(self):
        self.assertEqual(
            self._cutoff(
                proxy_len=3,
                first_eos=3,
                eos_hit=False,
                length_hit=False,
                max_tokens=100,
                prior_tokens=0,
            ),
            2,
        )

    def test_zero_proxy_carry_round_has_no_weight_columns(self):
        self.assertEqual(
            self._cutoff(
                proxy_len=0,
                first_eos=0,
                eos_hit=False,
                length_hit=False,
                max_tokens=100,
                prior_tokens=0,
            ),
            -1,
        )

    def test_eos_in_proxy_uses_eos_column(self):
        self.assertEqual(
            self._cutoff(
                proxy_len=5,
                first_eos=1,
                eos_hit=True,
                length_hit=False,
                max_tokens=100,
                prior_tokens=0,
            ),
            1,
        )

    def test_eos_in_bonus_keeps_entire_proxy(self):
        self.assertEqual(
            self._cutoff(
                proxy_len=4,
                first_eos=4,
                eos_hit=True,
                length_hit=False,
                max_tokens=100,
                prior_tokens=0,
            ),
            3,
        )

    def test_length_before_eos_uses_earlier_cutoff(self):
        self.assertEqual(
            self._cutoff(
                proxy_len=5,
                first_eos=3,
                eos_hit=True,
                length_hit=True,
                max_tokens=12,
                prior_tokens=10,
            ),
            1,
        )

    def test_length_cap_before_first_proxy_has_no_weight_columns(self):
        self.assertEqual(
            self._cutoff(
                proxy_len=5,
                first_eos=6,
                eos_hit=False,
                length_hit=True,
                max_tokens=10,
                prior_tokens=10,
            ),
            -1,
        )

    def test_eos_before_length_uses_earlier_cutoff(self):
        self.assertEqual(
            self._cutoff(
                proxy_len=5,
                first_eos=1,
                eos_hit=True,
                length_hit=True,
                max_tokens=15,
                prior_tokens=10,
            ),
            1,
        )


class TestBonusWeightMask(unittest.TestCase):
    def _mask(
        self,
        *,
        bonus_position: int = 3,
        first_eos: int = 4,
        eos_hit: bool = False,
        max_tokens: int = 100,
        prior_tokens: int = 0,
        prev_finished: bool = False,
        emit_bonus: bool = True,
    ) -> bool:
        return bool(
            bonus_weight_mask(
                bonus_positions=torch.tensor([bonus_position]),
                first_eos=torch.tensor([first_eos]),
                eos_hit=torch.tensor([eos_hit]),
                max_tokens=torch.tensor([max_tokens]),
                prior_token_counts=torch.tensor([prior_tokens]),
                prev_finished=torch.tensor([prev_finished]),
                emit_bonus=torch.tensor([emit_bonus]),
            ).item()
        )

    def test_normal_bonus_is_weighted(self):
        self.assertTrue(self._mask())

    def test_eos_in_proposal_suppresses_bonus(self):
        self.assertFalse(self._mask(first_eos=1, eos_hit=True))

    def test_eos_bonus_is_still_weighted(self):
        self.assertTrue(self._mask(first_eos=3, eos_hit=True))

    def test_length_cap_before_bonus_suppresses_it(self):
        self.assertFalse(self._mask(max_tokens=3))
        self.assertTrue(self._mask(max_tokens=4))

    def test_carry_and_finished_rows_suppress_bonus(self):
        self.assertFalse(self._mask(emit_bonus=False))
        self.assertFalse(self._mask(prev_finished=True))


if __name__ == "__main__":
    unittest.main()
