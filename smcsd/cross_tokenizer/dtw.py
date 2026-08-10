"""Bounded text-span alignment for draft-to-target token sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from smcsd.cross_tokenizer.canonicalize import canonicalize_join


@dataclass(frozen=True)
class AlignedSegment:
    draft_start: int
    draft_end: int
    target_start: int
    target_end: int

    @property
    def draft_len(self) -> int:
        return self.draft_end - self.draft_start

    @property
    def target_len(self) -> int:
        return self.target_end - self.target_start


def align_token_texts(
    draft_pieces: Sequence[str],
    target_pieces: Sequence[str],
    *,
    max_span: int = 8,
    canonicalize: Callable[[Sequence[str]], str] = canonicalize_join,
) -> List[AlignedSegment]:
    """Align token-piece sequences into text-preserving spans.

    The common case is handled by a linear two-buffer walk.  If a local
    tokenizer edge case prevents a flush within ``max_span`` on either side, a
    bounded dynamic program tries to recover.  If that still fails, the whole
    block is returned as one segment so probability mass remains conserved.
    """

    greedy = _greedy_segments(draft_pieces, target_pieces, max_span, canonicalize)
    if greedy is not None:
        return greedy

    dp = bounded_span_dp(draft_pieces, target_pieces, max_span=max_span)
    if dp:
        return dp

    if not draft_pieces and not target_pieces:
        return []
    return [AlignedSegment(0, len(draft_pieces), 0, len(target_pieces))]


def _greedy_segments(
    draft_pieces: Sequence[str],
    target_pieces: Sequence[str],
    max_span: int,
    canonicalize: Callable[[Sequence[str]], str],
) -> Optional[List[AlignedSegment]]:
    i = j = 0
    segments: List[AlignedSegment] = []

    while i < len(draft_pieces) or j < len(target_pieces):
        ds = i
        ts = j
        dbuf: List[str] = []
        tbuf: List[str] = []

        while True:
            dtext = canonicalize(dbuf)
            ttext = canonicalize(tbuf)
            if dbuf and tbuf and dtext == ttext:
                segments.append(AlignedSegment(ds, i, ts, j))
                break

            if len(dbuf) >= max_span or len(tbuf) >= max_span:
                return None

            # Extend the shorter canonical buffer; ties extend both sides if
            # possible to avoid a long run of empty fragments.
            if (len(dtext) <= len(ttext) or not tbuf) and i < len(draft_pieces):
                dbuf.append(draft_pieces[i])
                i += 1
            elif j < len(target_pieces):
                tbuf.append(target_pieces[j])
                j += 1
            else:
                return None

    return segments


def bounded_span_dp(
    draft_pieces: Sequence[str],
    target_pieces: Sequence[str],
    *,
    max_span: int = 8,
    gap_penalty: float = -1.5,
    match_reward: float = 3.0,
) -> List[AlignedSegment]:
    """DP alignment over equal decoded spans within a bounded span length."""

    n = len(draft_pieces)
    m = len(target_pieces)
    neg = -1e18
    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    back: List[List[Optional[Tuple[int, int, bool]]]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] <= neg / 2:
                continue
            if i < n and dp[i][j] + gap_penalty > dp[i + 1][j]:
                dp[i + 1][j] = dp[i][j] + gap_penalty
                back[i + 1][j] = (i, j, False)
            if j < m and dp[i][j] + gap_penalty > dp[i][j + 1]:
                dp[i][j + 1] = dp[i][j] + gap_penalty
                back[i][j + 1] = (i, j, False)

            for di in range(1, min(max_span, n - i) + 1):
                dtext = canonicalize_join(draft_pieces[i : i + di])
                for tj in range(1, min(max_span, m - j) + 1):
                    ttext = canonicalize_join(target_pieces[j : j + tj])
                    if dtext != ttext:
                        continue
                    score = dp[i][j] + match_reward * (di + tj)
                    if score > dp[i + di][j + tj]:
                        dp[i + di][j + tj] = score
                        back[i + di][j + tj] = (i, j, True)

    if dp[n][m] <= neg / 2:
        return []

    segments: List[AlignedSegment] = []
    i, j = n, m
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            return []
        pi, pj, is_match = prev
        if is_match:
            segments.append(AlignedSegment(pi, i, pj, j))
        i, j = pi, pj
    segments.reverse()
    return segments
