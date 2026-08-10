"""Proxy target length bucketing for cross-tokenizer verify."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


DEFAULT_PROXY_BUCKETS = (4, 8, 12, 16)


@dataclass
class BucketedProxy:
    input_ids: List[List[int]]
    logq: List[List[float]]
    valid_mask: List[List[bool]]
    lengths: List[int]
    bucket_size: int


def choose_bucket(length: int, buckets: Sequence[int] = DEFAULT_PROXY_BUCKETS) -> int:
    for bucket in sorted(int(x) for x in buckets):
        if length <= bucket:
            return bucket
    raise ValueError(f"Proxy target length {length} exceeds max bucket {max(buckets)}.")


def bucket_proxy_sequences(
    target_ids: Sequence[Sequence[int]],
    target_logq: Sequence[Sequence[float]],
    *,
    pad_token_id: int = 0,
    buckets: Sequence[int] = DEFAULT_PROXY_BUCKETS,
) -> BucketedProxy:
    if len(target_ids) != len(target_logq):
        raise ValueError("target_ids and target_logq must have the same batch size.")
    lengths = [len(row) for row in target_ids]
    bucket = choose_bucket(max(lengths, default=0), buckets)
    padded_ids: List[List[int]] = []
    padded_logq: List[List[float]] = []
    valid_mask: List[List[bool]] = []
    for ids, logq in zip(target_ids, target_logq):
        if len(ids) != len(logq):
            raise ValueError("Each target_id row must align with target_logq.")
        pad = bucket - len(ids)
        padded_ids.append([int(x) for x in ids] + [int(pad_token_id)] * pad)
        padded_logq.append([float(x) for x in logq] + [0.0] * pad)
        valid_mask.append([True] * len(ids) + [False] * pad)
    return BucketedProxy(
        input_ids=padded_ids,
        logq=padded_logq,
        valid_mask=valid_mask,
        lengths=lengths,
        bucket_size=bucket,
    )
