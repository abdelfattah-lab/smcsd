"""Tokenization-neutral diagnostics for cross-tokenizer proposal blocks.

These helpers do not assert that a canonical token path is the marginal
probability of a byte string.  They only identify byte-identical, monotonic
spans whose *native paths* can be scored and compared reproducibly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class SharedTextSpan:
    """One aligned draft/target token span with an identical decoded byte string."""

    draft_start: int
    draft_end: int
    target_start: int
    target_end: int
    text: str
    byte_length: int
    byte_identical: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "draft_start": self.draft_start,
            "draft_end": self.draft_end,
            "target_start": self.target_start,
            "target_end": self.target_end,
            "text": self.text,
            "byte_length": self.byte_length,
            "byte_identical": self.byte_identical,
        }


def decode_exact(tokenizer: Any, ids: Sequence[int]) -> str:
    """Decode without tokenizer cleanup that could hide a byte mismatch."""

    try:
        return tokenizer.decode(
            list(ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(list(ids))


def shared_text_spans(
    *,
    draft_tokenizer: Any,
    target_tokenizer: Any,
    draft_ids: Sequence[int],
    target_ids: Sequence[int],
    segments: Iterable[Any],
) -> list[SharedTextSpan]:
    """Describe mapped segments and mark whether their UTF-8 bytes agree.

    A false ``byte_identical`` result is a diagnostic signal: that span must not
    be used for a shared-string training loss or a likelihood-calibration claim.
    """

    spans: list[SharedTextSpan] = []
    for segment in segments:
        if isinstance(segment, dict):
            draft_start = int(segment["draft_start"])
            draft_end = int(segment["draft_end"])
            target_start = int(segment["target_start"])
            target_end = int(segment["target_end"])
        else:
            draft_start = int(segment.draft_start)
            draft_end = int(segment.draft_end)
            target_start = int(segment.target_start)
            target_end = int(segment.target_end)
        draft_text = decode_exact(draft_tokenizer, draft_ids[draft_start:draft_end])
        target_text = decode_exact(target_tokenizer, target_ids[target_start:target_end])
        draft_bytes = draft_text.encode("utf-8", errors="surrogatepass")
        target_bytes = target_text.encode("utf-8", errors="surrogatepass")
        spans.append(
            SharedTextSpan(
                draft_start=draft_start,
                draft_end=draft_end,
                target_start=target_start,
                target_end=target_end,
                text=draft_text if draft_bytes == target_bytes else "",
                byte_length=len(draft_bytes),
                byte_identical=draft_bytes == target_bytes,
            )
        )
    return spans
