"""Build cross-tokenizer mapping artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from smcsd.cross_tokenizer.artifacts import (
    MappingArtifact,
    MappingMetadata,
    MappingStats,
    save_artifact,
    tokenizer_fingerprint,
)
from smcsd.cross_tokenizer.canonicalize import canonicalize_piece, is_special_fragment


def decode_one(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([int(token_id)])


def encode_text(tokenizer: Any, text: str) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def iter_vocab_ids(tokenizer: Any) -> Iterable[int]:
    vocab = tokenizer.get_vocab()
    return sorted(int(idx) for idx in vocab.values())


def build_artifact(
    *,
    draft_tokenizer: Any,
    target_tokenizer: Any,
    draft_tokenizer_name: str,
    target_tokenizer_name: str,
    max_single_token_span: int = 4,
    max_ngram: int = 8,
    top_k: int = 4,
    corpus_texts: Optional[Iterable[str]] = None,
    max_ngram_entries: int = 0,
    max_ngram_target_span: int = 16,
) -> MappingArtifact:
    """Build a deterministic single-token mapping artifact.

    Corpus-derived many-to-one n-grams are intentionally not inferred here;
    callers can augment ``artifact.ngram_map`` from aligned corpora.  The
    single-token pass is still valuable: exact matches and 1-to-many splits are
    the hot majority for many BPE pairs.
    """

    draft_vocab = list(iter_vocab_ids(draft_tokenizer))
    target_vocab = list(iter_vocab_ids(target_tokenizer))
    draft_special_ids = {
        int(token_id) for token_id in getattr(draft_tokenizer, "all_special_ids", [])
    }
    target_special_ids = {
        int(token_id) for token_id in getattr(target_tokenizer, "all_special_ids", [])
    }
    target_index: Dict[str, List[int]] = defaultdict(list)
    for tid in target_vocab:
        text = canonicalize_piece(decode_one(target_tokenizer, tid))
        target_index[text].append(tid)

    single_map: Dict[str, List[int]] = {}
    special_token_map: Dict[str, int] = {}
    canonical_vocab_index: Dict[str, Dict[str, List[int]]] = {
        "draft": defaultdict(list),
        "target": defaultdict(list),
    }
    stats = MappingStats(
        draft_vocab_size=len(draft_vocab),
        target_vocab_size=len(target_vocab),
    )

    for tid in target_vocab:
        canonical_vocab_index["target"][
            canonicalize_piece(decode_one(target_tokenizer, tid))
        ].append(tid)

    for did in draft_vocab:
        raw_text = decode_one(draft_tokenizer, did)
        text = canonicalize_piece(raw_text)
        canonical_vocab_index["draft"][text].append(did)

        if is_special_fragment(raw_text) or is_special_fragment(text):
            # A tokenizer-specific control fragment must never be mapped to a
            # normal target-vocabulary token merely because its decoded surface
            # string happens to match. Runtime EOS/control handling owns those
            # events. Persist a special mapping only when both ids are genuine
            # special tokens; this makes the artifact safe to validate and
            # prevents scoring literal control text as natural language.
            matches = [
                target_id
                for target_id in target_index.get(text, [])
                if int(target_id) in target_special_ids
            ]
            if int(did) in draft_special_ids and len(matches) == 1:
                special_token_map[str(did)] = int(matches[0])
            else:
                stats.blocked_specials += 1
            continue

        exact = target_index.get(text, [])
        if exact:
            single_map[str(did)] = [int(exact[0])]
            stats.exact_matches += 1
            continue

        target_ids = encode_text(target_tokenizer, raw_text)
        if 0 < len(target_ids) <= max_single_token_span:
            single_map[str(did)] = [int(x) for x in target_ids]
            if len(target_ids) == 1:
                stats.exact_matches += 1
            else:
                stats.one_to_many += 1
        else:
            stats.unmapped += 1

    metadata = MappingMetadata(
        draft_tokenizer=draft_tokenizer_name,
        target_tokenizer=target_tokenizer_name,
        draft_fingerprint=tokenizer_fingerprint(draft_tokenizer),
        target_fingerprint=tokenizer_fingerprint(target_tokenizer),
        max_single_token_span=max_single_token_span,
        max_ngram=max_ngram,
        top_k=top_k,
    )

    # Convert defaultdicts to plain dicts for stable JSON serialization.
    canonical_plain = {
        side: {key: list(ids) for key, ids in values.items()}
        for side, values in canonical_vocab_index.items()
    }

    artifact = MappingArtifact(
        metadata=metadata,
        stats=stats,
        single_map=single_map,
        special_token_map=special_token_map,
        ngram_map={},
        canonical_vocab_index=canonical_plain,
    )
    if corpus_texts is not None and max_ngram_entries > 0 and max_ngram > 1:
        _augment_ngram_map(
            artifact=artifact,
            draft_tokenizer=draft_tokenizer,
            target_tokenizer=target_tokenizer,
            corpus_texts=corpus_texts,
            max_ngram=max_ngram,
            max_entries=max_ngram_entries,
            max_target_span=max_ngram_target_span,
        )
    return artifact


def _augment_ngram_map(
    *,
    artifact: MappingArtifact,
    draft_tokenizer: Any,
    target_tokenizer: Any,
    corpus_texts: Iterable[str],
    max_ngram: int,
    max_entries: int,
    max_target_span: int,
) -> None:
    """Add frequent corpus n-grams as direct draft->target mappings."""
    from smcsd.cross_tokenizer.artifacts import encode_ngram_key

    counts: Counter[Tuple[int, ...]] = Counter()
    for text in corpus_texts:
        draft_ids = encode_text(draft_tokenizer, text)
        for start in range(len(draft_ids)):
            for length in range(2, max_ngram + 1):
                end = start + length
                if end > len(draft_ids):
                    break
                counts[tuple(int(x) for x in draft_ids[start:end])] += 1

    added = 0
    for draft_ngram, _ in counts.most_common(max_entries * 4):
        if added >= max_entries:
            break
        key = encode_ngram_key(draft_ngram)
        if key in artifact.ngram_map:
            continue
        text = TokenMapperDecode.decode_many(draft_tokenizer, draft_ngram)
        target_ids = encode_text(target_tokenizer, text)
        if not target_ids or len(target_ids) > max_target_span:
            continue
        artifact.ngram_map[key] = [int(x) for x in target_ids]
        added += 1
    artifact.stats.ngram_entries = len(artifact.ngram_map)


class TokenMapperDecode:
    @staticmethod
    def decode_many(tokenizer: Any, ids: Iterable[int]) -> str:
        try:
            return tokenizer.decode(
                list(ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(list(ids))


def _load_corpus_texts(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []
    if args.corpus_file:
        path = Path(args.corpus_file)
        texts.extend(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if args.gsm8k_num_questions > 0:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split=args.gsm8k_split)
        for sample in dataset.select(range(min(args.gsm8k_num_questions, len(dataset)))):
            texts.append(sample["question"])
            texts.append(sample["answer"])
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-tokenizer", required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-single-token-span", type=int, default=4)
    parser.add_argument("--max-ngram", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--corpus-file", default=None)
    parser.add_argument("--gsm8k-split", default="test")
    parser.add_argument("--gsm8k-num-questions", type=int, default=0)
    parser.add_argument("--max-ngram-entries", type=int, default=0)
    parser.add_argument("--max-ngram-target-span", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    draft = AutoTokenizer.from_pretrained(
        args.draft_tokenizer, trust_remote_code=args.trust_remote_code
    )
    target = AutoTokenizer.from_pretrained(
        args.target_tokenizer, trust_remote_code=args.trust_remote_code
    )
    corpus_texts = _load_corpus_texts(args)
    artifact = build_artifact(
        draft_tokenizer=draft,
        target_tokenizer=target,
        draft_tokenizer_name=args.draft_tokenizer,
        target_tokenizer_name=args.target_tokenizer,
        max_single_token_span=args.max_single_token_span,
        max_ngram=args.max_ngram,
        top_k=args.top_k,
        corpus_texts=corpus_texts,
        max_ngram_entries=args.max_ngram_entries,
        max_ngram_target_span=args.max_ngram_target_span,
    )
    save_artifact(artifact, args.output)


if __name__ == "__main__":
    main()
