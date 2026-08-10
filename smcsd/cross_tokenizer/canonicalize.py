"""Canonical string normalization for cross-tokenizer mapping.

The mapper compares decoded token fragments from different tokenizer families.
Those fragments often use different visible conventions for the same bytes
(SentencePiece underline, GPT byte-level whitespace markers, escaped newlines,
byte fallback tokens).  The rules here are deliberately small and reversible
enough for equality checks; they are not a text normalizer for model inputs.
"""

from __future__ import annotations

import re

_BYTE_FALLBACK_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def decode_byte_fallback(text: str) -> str:
    """Replace SentencePiece byte fallback markers with literal bytes."""

    def _replace(match: re.Match[str]) -> str:
        value = int(match.group(1), 16)
        return bytes([value]).decode("utf-8", errors="replace")

    return _BYTE_FALLBACK_RE.sub(_replace, text)


def canonicalize_piece(text: str) -> str:
    """Return a canonical form suitable for tokenizer-fragment equality."""

    if text is None:
        return ""

    out = str(text)
    out = decode_byte_fallback(out)

    # Common visible whitespace conventions.
    out = out.replace("\\n", "\n")
    out = out.replace("Ċ", "\n")
    out = out.replace("▁", " ")
    out = out.replace("Ġ", " ")
    out = out.replace("ā", " ")

    return out


def canonicalize_join(pieces: list[str] | tuple[str, ...]) -> str:
    """Canonicalize a sequence after concatenating its token fragments."""

    return canonicalize_piece("".join(pieces))


def is_special_fragment(text: str) -> bool:
    """Heuristic for model control tokens that should be explicitly mapped."""

    if not text:
        return False
    return (
        (text.startswith("<|") and text.endswith("|>"))
        or (text.startswith("<") and text.endswith(">"))
        or text in {"<s>", "</s>", "<pad>", "[PAD]", "[BOS]", "[EOS]"}
    )
