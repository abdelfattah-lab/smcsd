"""Runtime token mapping between draft and target tokenizers."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

from smcsd.cross_tokenizer.artifacts import MappingArtifact, decode_ngram_key
from smcsd.cross_tokenizer.build import decode_one, encode_text
from smcsd.cross_tokenizer.canonicalize import canonicalize_piece
from smcsd.cross_tokenizer.dtw import AlignedSegment, align_token_texts


MapperMode = Literal["live", "hybrid"]


@dataclass
class BoundaryState:
    """Per-particle tokenizer-boundary carry state.

    The validated default commits full mapped blocks. The opt-in boundary-carry
    path stores an unsettled suffix here so it can be remapped with the next
    proposal block.
    """

    draft_suffix_ids: List[int] = field(default_factory=list)
    draft_suffix_logprobs: List[float] = field(default_factory=list)
    target_suffix_ids: List[int] = field(default_factory=list)
    target_suffix_logq: List[float] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (
            self.draft_suffix_ids
            or self.draft_suffix_logprobs
            or self.target_suffix_ids
            or self.target_suffix_logq
        )

    def clone(self) -> "BoundaryState":
        return BoundaryState(
            draft_suffix_ids=list(self.draft_suffix_ids),
            draft_suffix_logprobs=list(self.draft_suffix_logprobs),
            target_suffix_ids=list(self.target_suffix_ids),
            target_suffix_logq=list(self.target_suffix_logq),
        )


@dataclass(frozen=True)
class MappingSegment:
    draft_start: int
    draft_end: int
    target_start: int
    target_end: int
    draft_logq: float


@dataclass
class MappingResult:
    target_ids: List[int]
    target_logq: List[float]
    segments: List[MappingSegment]
    valid_mask: List[bool]
    boundary_state: BoundaryState
    used_live_fallback: bool = False
    trie_hits: int = 0
    single_hits: int = 0
    cache_hits: int = 0
    dtw_fallbacks: int = 0
    raw_target_len: Optional[int] = None

    @property
    def proxy_len(self) -> int:
        return len(self.target_ids)

    @property
    def raw_proxy_len(self) -> int:
        return (
            int(self.raw_target_len)
            if self.raw_target_len is not None
            else len(self.target_ids)
        )


@dataclass
class MappingBatchResult:
    proxy_ids: List[List[int]]
    proxy_logq: List[List[float]]
    proxy_lens: List[int]
    raw_proxy_lens: List[int]
    overlength_mask: List[bool]
    empty_proxy_mask: List[bool]
    overlength_rows: int
    segments: List[List[MappingSegment]]
    row_stats: List[Dict[str, int]]
    boundary_states: List[BoundaryState]


@dataclass
class MapperRuntimeStats:
    calls: int = 0
    draft_tokens: int = 0
    target_tokens: int = 0
    trie_hits: int = 0
    single_hits: int = 0
    cache_hits: int = 0
    fallback_hits: int = 0
    dtw_fallbacks: int = 0
    overlength_rows: int = 0
    empty_proxy_rows: int = 0
    carried_rows: int = 0
    carried_draft_tokens: int = 0
    carried_target_tokens: int = 0

    def snapshot(self) -> Dict[str, float]:
        calls = max(self.calls, 1)
        return {
            "calls": self.calls,
            "draft_tokens": self.draft_tokens,
            "target_tokens": self.target_tokens,
            "trie_hits": self.trie_hits,
            "single_hits": self.single_hits,
            "cache_hits": self.cache_hits,
            "fallback_hits": self.fallback_hits,
            "dtw_fallbacks": self.dtw_fallbacks,
            "overlength_rows": self.overlength_rows,
            "empty_proxy_rows": self.empty_proxy_rows,
            "carried_rows": self.carried_rows,
            "carried_draft_tokens": self.carried_draft_tokens,
            "carried_target_tokens": self.carried_target_tokens,
            "avg_target_tokens_per_call": self.target_tokens / calls,
        }


class _LFUCache:
    def __init__(self, capacity: int = 4096):
        self.capacity = max(int(capacity), 0)
        self._data: OrderedDict[Tuple[int, ...], Tuple[List[int], int]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Tuple[int, ...]) -> Optional[List[int]]:
        item = self._data.get(key)
        if item is None:
            self.misses += 1
            return None
        self.hits += 1
        value, count = item
        self._data[key] = (value, count + 1)
        self._data.move_to_end(key)
        return list(value)

    def put(self, key: Tuple[int, ...], value: List[int]) -> None:
        if self.capacity <= 0:
            return
        if key in self._data:
            _, count = self._data[key]
            self._data[key] = (list(value), count + 1)
            self._data.move_to_end(key)
            return
        if len(self._data) >= self.capacity:
            victim = min(self._data.items(), key=lambda kv: (kv[1][1],))
            self._data.pop(victim[0], None)
        self._data[key] = (list(value), 1)


class TokenMapper:
    """Map draft-token proposals into target-token proposals."""

    def __init__(
        self,
        *,
        draft_tokenizer: Any,
        target_tokenizer: Any,
        artifact: Optional[MappingArtifact] = None,
        mode: MapperMode = "hybrid",
        max_alignment_span: int = 8,
        fallback_cache_size: int = 4096,
    ) -> None:
        if mode not in ("live", "hybrid"):
            raise ValueError(f"Unsupported mapper mode: {mode!r}")
        self.draft_tokenizer = draft_tokenizer
        self.target_tokenizer = target_tokenizer
        self.artifact = artifact
        self.mode: MapperMode = mode
        self.max_alignment_span = max_alignment_span
        self.cache = _LFUCache(fallback_cache_size)
        # Safe-seam decision is a pure function of the right token id (finite
        # target vocab), but healing queries it for every segment boundary of
        # every particle every cycle.  Memoize id -> bool so the decode_one +
        # canonicalize collapses to a dict hit after warmup.
        self._safe_seam_cache: Dict[int, bool] = {}
        # G3: dense safe-seam bitmap over the target vocab.  _is_safe_seam is a
        # pure function of the right token id; a precomputed bytearray turns the
        # per-seam check into an O(1) index with no dict/decode after a one-time
        # build.  Byte-identical to the lazy rule -> default ON; built lazily on
        # first use.  Set SMC_CROSS_SEAM_BITMAP=0 to fall back to the lazy dict.
        self._seam_bitmap_enabled = bool(
            int(os.environ.get("SMC_CROSS_SEAM_BITMAP", "1"))
        )
        self._seam_safe_arr: Optional[bytearray] = None
        self.stats = MapperRuntimeStats()
        try:
            self._identity_vocab = draft_tokenizer.get_vocab() == target_tokenizer.get_vocab()
        except Exception:
            self._identity_vocab = False

        # Diagnostic (default off): bypass the identity-vocab mapping shortcut so
        # the real hybrid/live mapping path runs even when draft==target vocab.
        # Pairs with SMC_CROSS_FORCE_CPU_LINEAGE (worker side) to exercise the
        # full cross path on an identity pair, isolating cross-machinery overhead
        # at a fixed model quality.
        self._force_real_map = bool(
            int(os.environ.get("SMC_CROSS_FORCE_REAL_MAP", "0"))
        )

        if self.mode == "hybrid":
            if self.artifact is None:
                raise ValueError("hybrid mapper mode requires a MappingArtifact.")
            self.artifact.validate_for(draft_tokenizer, target_tokenizer)

        # Lossless hybrid fast path (default on; SMC_CROSS_FAST_MAP=0 to disable):
        # replace the per-probe string-key construction (" ".join(str(x) ...)) in
        # get_ngram/get_single with int-keyed lookups, and short-circuit the
        # n-gram probe when the current draft id starts no n-gram at all.  Proven
        # byte-identical to the slow path (10k-sequence equivalence test); only
        # the lookup cost differs (steady-state map_ms 0.93 -> 0.44 ms/cycle on
        # qwen2.5-0.5b -> llama3.1-8b hybrid).
        self._fast_map = bool(int(os.environ.get("SMC_CROSS_FAST_MAP", "1")))

        # Boundary healing (default ON; SMC_CROSS_HEAL_BOUNDARIES=0 to disable):
        # hybrid stitches per-segment maps that were each tokenized in isolation,
        # so the proxy is a *valid but non-canonical* target tokenization at the
        # seams (wrong BPE merges across fragment joins -- e.g. the digits of a
        # multi-digit number stay split).  The target then scores an off-manifold
        # sequence, miscalibrating the SMC importance weight AND feeding the bonus
        # head a non-canonical context.  Measured on GSM8K (Qwen3-0.6B ->
        # Llama-3.1-8B): healing lifts accuracy 25% -> 52% (~2x, matching live)
        # and restores particle ESS from 4.3 to 5.5 / 12, at ~1 ms/step.
        # Healing re-encodes only the spans *between safe seams* -- a seam is safe
        # when the right token begins with a leading-space/newline marker, where
        # target BPE provably never merges across.  Unsafe (word-internal) runs
        # are re-tokenized as one string so internal merges become canonical,
        # while safe single-token segments pass through with zero work.  The
        # identity-vocab shortcut in map_tokens returns before this, so an
        # identity (same-tokenizer) pair never pays for healing.
        self._heal_boundaries = bool(
            int(os.environ.get("SMC_CROSS_HEAL_BOUNDARIES", "1"))
        )
        self._heal_groups = 0
        # Heal groups that missed the cache and paid a live decode->encode.
        self._heal_reencodes = 0
        self._alignment_repairs = 0
        # Optional online n-gram promotion. A healed run is a pure draft-tuple->canon
        # function and the heal path already collapses it to ONE segment with the
        # summed draft logq -- identical to what an n-gram hit produces. So once
        # a healed run recurs >= MIN times we insert it into the n-gram trie; the
        # next occurrence is matched up-front (no per-token single-map, no seam
        # scan, no re-encode) with a byte-identical MappingResult. Keep it off by
        # default so production runs use a frozen, reproducible artifact.
        self._online_ngram = bool(int(os.environ.get("SMC_CROSS_ONLINE_NGRAM", "0")))
        self._online_ngram_min = max(
            2, int(os.environ.get("SMC_CROSS_ONLINE_NGRAM_MIN", "2"))
        )
        self._online_ngram_max_len = int(
            os.environ.get("SMC_CROSS_ONLINE_NGRAM_MAXLEN", "8")
        )
        self._online_ngram_max_entries = int(
            os.environ.get("SMC_CROSS_ONLINE_NGRAM_MAX", "50000")
        )
        self._heal_run_counts: Dict[Tuple[int, ...], int] = {}
        self._online_ngram_promotions = 0
        self._artifact_ngram_base = 0

        # Special-token id remap (default ON; SMC_CROSS_MAP_SPECIALS=0 to disable):
        # the artifact's single_map sends a draft control token (e.g. Qwen
        # <|im_end|>) to the *literal text* of its surface form ("<|im_end|>" ->
        # ['<', '|', 'im', '_', 'end', '|', '>']) because the target tokenizer
        # has no such token.  The target then scores junk text instead of its own
        # EOS, so the particle that correctly *finished* gets a large negative
        # weight and the sequence never terminates on the draft's stop signal.
        # We instead map draft EOS-family ids -> [target EOS id] so a draft stop
        # becomes a real target stop.  This removed fallback/DTW cases in the
        # current Qwen/Llama GSM8K runs and is now the default; set
        # SMC_CROSS_MAP_SPECIALS=0 for A/B or if a draft's stop timing is harmful.
        self._map_specials = bool(int(os.environ.get("SMC_CROSS_MAP_SPECIALS", "1")))
        self._special_id_map: Dict[int, List[int]] = {}
        if self._map_specials:
            if self.artifact is not None:
                self._special_id_map.update(
                    {
                        int(draft_id): [int(target_id)]
                        for draft_id, target_id in self.artifact.special_token_map.items()
                    }
                )
            # EOS-family semantics take priority over equal-surface artifact
            # mappings: a draft stop must become the target's configured EOS.
            self._special_id_map.update(
                self._build_special_id_map(draft_tokenizer, target_tokenizer)
            )

        self._single_map_int: Optional[Dict[int, List[int]]] = None
        self._ngram_map_int: Optional[Dict[Tuple[int, ...], List[int]]] = None
        self._ngram_first: Optional[Set[int]] = None
        self._max_ngram_len_fast = 0
        if self._fast_map and self.artifact is not None:
            self._single_map_int = {
                int(k): list(v) for k, v in self.artifact.single_map.items()
            }
            self._ngram_map_int = {}
            self._ngram_first = set()
            for key_str, value in self.artifact.ngram_map.items():
                key = decode_ngram_key(key_str)
                if not key:
                    continue
                self._ngram_map_int[key] = list(value)
                self._ngram_first.add(key[0])
            self._max_ngram_len_fast = self.artifact.max_ngram_len()
            self._artifact_ngram_base = len(self._ngram_map_int)

    @staticmethod
    def _build_special_id_map(draft_tok: Any, target_tok: Any) -> Dict[int, List[int]]:
        """Map draft EOS-family ids -> [target EOS id].

        Robust to multiple draft stop tokens (e.g. Qwen ships both
        ``<|im_end|>`` and ``<|endoftext|>``); both should terminate the target.
        """
        t_eos = getattr(target_tok, "eos_token_id", None)
        if t_eos is None:
            return {}
        d_eos_ids: Set[int] = set()
        d_eos = getattr(draft_tok, "eos_token_id", None)
        if d_eos is not None:
            d_eos_ids.add(int(d_eos))
        # Sweep common stop-token surface forms the draft may emit.
        for name in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"):
            try:
                tid = draft_tok.convert_tokens_to_ids(name)
            except Exception:
                tid = None
            if tid is not None and tid >= 0 and tid != getattr(
                draft_tok, "unk_token_id", -1
            ):
                d_eos_ids.add(int(tid))
        return {d: [int(t_eos)] for d in d_eos_ids}

    def map_tokens(
        self,
        draft_ids: Sequence[int],
        draft_logprobs: Sequence[float],
        boundary_state: Optional[BoundaryState] = None,
        *,
        flush: bool = True,
        max_target_len: Optional[int] = None,
    ) -> MappingResult:
        """Map one particle's draft block into target tokens.

        ``draft_logprobs`` must align with ``draft_ids`` and represent the
        autoregressive log probability of each draft token. The returned
        ``target_logq`` conserves the sampled draft-path probability by
        charging each aligned segment's total draft log probability to the
        first target token in that segment. This makes every visible target
        prefix include the proposal charge required to emit it when decoding
        stops inside a one-to-many mapped segment.
        """

        boundary_state = boundary_state or BoundaryState()
        all_draft_ids = list(boundary_state.draft_suffix_ids) + [
            int(x) for x in draft_ids
        ]
        all_logq = list(boundary_state.draft_suffix_logprobs) + [
            _as_float(x) for x in draft_logprobs
        ]
        if len(all_draft_ids) != len(all_logq):
            raise ValueError("draft_ids and draft_logprobs must have the same length.")

        if self._identity_vocab and not self._force_real_map:
            result = MappingResult(
                target_ids=list(all_draft_ids),
                target_logq=list(all_logq),
                segments=[
                    MappingSegment(i, i + 1, i, i + 1, float(all_logq[i]))
                    for i in range(len(all_draft_ids))
                ],
                valid_mask=[True] * len(all_draft_ids),
                boundary_state=BoundaryState(),
                single_hits=len(all_draft_ids),
            )
            result = self._partition_for_carry(
                result,
                all_draft_ids,
                all_logq,
                flush=flush,
                max_target_len=max_target_len,
            )
            self._record_result(result, len(all_draft_ids))
            return result

        if self.mode == "live":
            result = self._map_live(all_draft_ids, all_logq)
        else:
            result = self._map_hybrid(all_draft_ids, all_logq)
        result = self._partition_for_carry(
            result,
            all_draft_ids,
            all_logq,
            flush=flush,
            max_target_len=max_target_len,
        )
        self._record_result(result, len(all_draft_ids))
        return result

    def map_batch(
        self,
        draft_ids: Sequence[Sequence[int]],
        draft_logprobs: Sequence[Sequence[float]],
        *,
        max_proxy_len: int,
        boundary_states: Optional[Sequence[Optional[BoundaryState]]] = None,
        flush: bool = True,
    ) -> MappingBatchResult:
        """Map and pad a batch of draft blocks into rectangular target proxies."""
        if len(draft_ids) != len(draft_logprobs):
            raise ValueError("draft_ids and draft_logprobs batch sizes must match.")
        if boundary_states is None:
            boundary_states = [None] * len(draft_ids)
        if len(boundary_states) != len(draft_ids):
            raise ValueError("boundary_states batch size must match draft_ids.")

        proxy_ids: List[List[int]] = []
        proxy_logq: List[List[float]] = []
        proxy_lens: List[int] = []
        raw_proxy_lens: List[int] = []
        overlength_mask: List[bool] = []
        empty_proxy_mask: List[bool] = []
        segments: List[List[MappingSegment]] = []
        row_stats: List[Dict[str, int]] = []
        next_boundary_states: List[BoundaryState] = []
        overlength_rows = 0

        for row_ids, row_logprobs, row_boundary in zip(
            draft_ids, draft_logprobs, boundary_states
        ):
            result = self.map_tokens(
                row_ids,
                row_logprobs,
                row_boundary,
                flush=flush,
                max_target_len=max_proxy_len if not flush else None,
            )
            row_stat = {
                "trie_hits": int(result.trie_hits),
                "single_hits": int(result.single_hits),
                "cache_hits": int(result.cache_hits),
                "dtw_fallbacks": int(result.dtw_fallbacks),
            }
            if not result.boundary_state.is_empty:
                row_stat.update(
                    {
                        "carried_draft_tokens": len(
                            result.boundary_state.draft_suffix_ids
                        ),
                        "carried_target_tokens": len(
                            result.boundary_state.target_suffix_ids
                        ),
                    }
                )
            row_stats.append(row_stat)
            raw_proxy_lens.append(result.raw_proxy_len)
            next_boundary_states.append(result.boundary_state.clone())
            had_draft_event = bool(row_ids) or bool(
                row_boundary and row_boundary.draft_suffix_ids
            )
            empty_flushed = (
                had_draft_event
                and result.proxy_len == 0
                and result.boundary_state.is_empty
            )
            discard_proxy = result.proxy_len > max_proxy_len or empty_flushed
            empty_proxy_mask.append(empty_flushed)
            if discard_proxy:
                ids: List[int] = []
                logq: List[float] = []
                overlength_mask.append(True)
                overlength_rows += 1
                if empty_flushed:
                    self.stats.empty_proxy_rows += 1
            else:
                ids = result.target_ids
                logq = result.target_logq
                overlength_mask.append(False)
            segments.append([] if discard_proxy else result.segments)
            proxy_lens.append(len(ids))
            proxy_ids.append(ids + [0] * (max_proxy_len - len(ids)))
            proxy_logq.append(logq + [0.0] * (max_proxy_len - len(logq)))

        if overlength_rows:
            self.record_overlength(overlength_rows)
        return MappingBatchResult(
            proxy_ids=proxy_ids,
            proxy_logq=proxy_logq,
            proxy_lens=proxy_lens,
            raw_proxy_lens=raw_proxy_lens,
            overlength_mask=overlength_mask,
            empty_proxy_mask=empty_proxy_mask,
            overlength_rows=overlength_rows,
            segments=segments,
            row_stats=row_stats,
            boundary_states=next_boundary_states,
        )

    def record_overlength(self, count: int = 1) -> None:
        self.stats.overlength_rows += int(count)

    def format_stats(self) -> str:
        snapshot = self.stats.snapshot()
        ordered = (
            "calls",
            "draft_tokens",
            "target_tokens",
            "trie_hits",
            "single_hits",
            "cache_hits",
            "fallback_hits",
            "dtw_fallbacks",
            "overlength_rows",
            "empty_proxy_rows",
            "carried_rows",
            "carried_draft_tokens",
            "carried_target_tokens",
            "avg_target_tokens_per_call",
        )
        parts = []
        for key in ordered:
            value = snapshot[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value}")
        parts.append(f"heal_groups={self._heal_groups}")
        parts.append(f"heal_reencodes={self._heal_reencodes}")
        parts.append(f"alignment_repairs={self._alignment_repairs}")
        parts.append(f"online_ngram_promotions={self._online_ngram_promotions}")
        ngram_entries = (
            len(self._ngram_map_int) if self._ngram_map_int is not None else 0
        )
        parts.append(f"ngram_entries={ngram_entries}")
        return "cross_tokenizer_mapper_stats " + " ".join(parts)

    def _record_result(self, result: MappingResult, draft_len: int) -> None:
        self.stats.calls += 1
        self.stats.draft_tokens += int(draft_len)
        self.stats.target_tokens += int(result.proxy_len)
        self.stats.trie_hits += int(result.trie_hits)
        self.stats.single_hits += int(result.single_hits)
        self.stats.cache_hits += int(result.cache_hits)
        self.stats.fallback_hits += int(result.dtw_fallbacks)
        self.stats.dtw_fallbacks += int(result.dtw_fallbacks)
        if not result.boundary_state.is_empty:
            self.stats.carried_rows += 1
            self.stats.carried_draft_tokens += len(
                result.boundary_state.draft_suffix_ids
            )
            self.stats.carried_target_tokens += len(
                result.boundary_state.target_suffix_ids
            )

    def _partition_for_carry(
        self,
        result: MappingResult,
        draft_ids: Sequence[int],
        logq: Sequence[float],
        *,
        flush: bool,
        max_target_len: Optional[int],
    ) -> MappingResult:
        """Split a mapped stream at a tokenizer-stable segment boundary.

        This is a mapper-level primitive: callers may withhold the returned
        ``boundary_state`` and prepend it to a later block. Runtime callers must
        not emit a target bonus *before* that suffix is flushed, because doing so
        would reverse the byte order. The worker's opt-in boundary-carry path
        suppresses that bonus; the validated default still uses ``flush=True``.
        """

        raw_target_len = len(result.target_ids)
        result.raw_target_len = raw_target_len
        limit = raw_target_len if max_target_len is None else max(
            int(max_target_len), 0
        )
        if flush and raw_target_len <= limit:
            return result

        # A shared tokenizer has no cross-tokenizer seam ambiguity. It can split
        # directly at a token boundary when a caller imposes a target limit.
        if self._identity_vocab and not self._force_real_map:
            target_end = min(raw_target_len, limit)
            draft_end = min(len(draft_ids), target_end)
        else:
            candidates: List[Tuple[int, int]] = [(0, 0)]
            for index in range(1, len(result.segments)):
                segment = result.segments[index]
                target_boundary = int(segment.target_start)
                if target_boundary <= 0 or target_boundary >= raw_target_len:
                    continue
                if self._is_safe_seam(
                    result.target_ids[target_boundary - 1],
                    result.target_ids[target_boundary],
                ):
                    candidates.append(
                        (int(segment.draft_start), target_boundary)
                    )

            target_eos = getattr(self.target_tokenizer, "eos_token_id", None)
            eos_ids = (
                {int(x) for x in target_eos}
                if isinstance(target_eos, (list, tuple, set))
                else ({int(target_eos)} if target_eos is not None else set())
            )
            ends_in_eos = bool(
                result.target_ids and int(result.target_ids[-1]) in eos_ids
            )
            if ends_in_eos:
                candidates.append((len(draft_ids), raw_target_len))

            eligible = [
                (draft_end, target_end)
                for draft_end, target_end in candidates
                if target_end <= limit
            ]
            draft_end, target_end = max(eligible, key=lambda pair: pair[1])

        if target_end == raw_target_len and draft_end == len(draft_ids):
            result.boundary_state = BoundaryState()
            return result

        committed_segments = [
            segment
            for segment in result.segments
            if segment.draft_end <= draft_end
            and segment.target_end <= target_end
        ]
        boundary_state = BoundaryState(
            draft_suffix_ids=[int(x) for x in draft_ids[draft_end:]],
            draft_suffix_logprobs=[_as_float(x) for x in logq[draft_end:]],
            target_suffix_ids=[
                int(x) for x in result.target_ids[target_end:]
            ],
            target_suffix_logq=[
                _as_float(x) for x in result.target_logq[target_end:]
            ],
        )
        return MappingResult(
            target_ids=[int(x) for x in result.target_ids[:target_end]],
            target_logq=[
                _as_float(x) for x in result.target_logq[:target_end]
            ],
            segments=committed_segments,
            valid_mask=[True] * target_end,
            boundary_state=boundary_state,
            used_live_fallback=result.used_live_fallback,
            trie_hits=result.trie_hits,
            single_hits=result.single_hits,
            cache_hits=result.cache_hits,
            dtw_fallbacks=result.dtw_fallbacks,
            raw_target_len=raw_target_len,
        )

    def _map_live(self, draft_ids: List[int], logq: List[float]) -> MappingResult:
        # Split the block at special/control ids so they map to the target EOS
        # rather than being decoded into junk surface text and re-encoded.
        if self._special_id_map and any(
            d in self._special_id_map for d in draft_ids
        ):
            target_ids: List[int] = []
            segments: List[MappingSegment] = []
            i = 0
            while i < len(draft_ids):
                if draft_ids[i] in self._special_id_map:
                    sp = self._special_id_map[draft_ids[i]]
                    start = len(target_ids)
                    target_ids.extend(sp)
                    segments.append(
                        MappingSegment(i, i + 1, start, len(target_ids), float(logq[i]))
                    )
                    i += 1
                    continue
                j = i
                while j < len(draft_ids) and draft_ids[j] not in self._special_id_map:
                    j += 1
                sub_ids, sub_segs = self._live_map_run(draft_ids[i:j], logq[i:j], i, len(target_ids))
                target_ids.extend(sub_ids)
                segments.extend(sub_segs)
                i = j
            segments = self._merge_zero_target_segments(segments)
            target_logq = self._terminal_charge(len(target_ids), segments)
            return MappingResult(
                target_ids=target_ids,
                target_logq=target_logq,
                segments=segments,
                valid_mask=[True] * len(target_ids),
                boundary_state=BoundaryState(),
                used_live_fallback=True,
                dtw_fallbacks=1,
            )

        sub_ids, segments = self._live_map_run(draft_ids, logq, 0, 0)
        segments = self._merge_zero_target_segments(segments)
        target_logq = self._terminal_charge(len(sub_ids), segments)
        return MappingResult(
            target_ids=sub_ids,
            target_logq=target_logq,
            segments=segments,
            valid_mask=[True] * len(sub_ids),
            boundary_state=BoundaryState(),
            used_live_fallback=True,
            dtw_fallbacks=1,
        )

    def _live_map_run(
        self,
        draft_ids: Sequence[int],
        logq: Sequence[float],
        draft_offset: int,
        target_offset: int,
    ) -> Tuple[List[int], List[MappingSegment]]:
        """Live-map one special-free run; offsets shift segment coords to the block."""
        if not draft_ids:
            return [], []
        draft_text = self._decode_many(self.draft_tokenizer, draft_ids)
        target_ids = encode_text(self.target_tokenizer, draft_text)
        target_pieces = [
            decode_one(self.target_tokenizer, target_id) for target_id in target_ids
        ]
        draft_pieces = [
            decode_one(self.draft_tokenizer, draft_id) for draft_id in draft_ids
        ]
        aligned = align_token_texts(
            draft_pieces, target_pieces, max_span=self.max_alignment_span
        )
        draft_cursor = target_cursor = 0
        contiguous = True
        for segment in aligned:
            if (
                segment.draft_start != draft_cursor
                or segment.target_start != target_cursor
            ):
                contiguous = False
                break
            draft_cursor = segment.draft_end
            target_cursor = segment.target_end
        contiguous = (
            contiguous
            and draft_cursor == len(draft_ids)
            and target_cursor == len(target_ids)
        )
        if not contiguous:
            self._alignment_repairs += 1
            return target_ids, [
                MappingSegment(
                    draft_start=draft_offset,
                    draft_end=draft_offset + len(draft_ids),
                    target_start=target_offset,
                    target_end=target_offset + len(target_ids),
                    draft_logq=float(sum(logq)),
                )
            ]
        segments: List[MappingSegment] = []
        for seg in aligned:
            draft_logq = sum(logq[seg.draft_start : seg.draft_end])
            segments.append(
                MappingSegment(
                    draft_start=seg.draft_start + draft_offset,
                    draft_end=seg.draft_end + draft_offset,
                    target_start=seg.target_start + target_offset,
                    target_end=seg.target_end + target_offset,
                    draft_logq=float(draft_logq),
                )
            )
        return target_ids, segments

    def _map_hybrid(self, draft_ids: List[int], logq: List[float]) -> MappingResult:
        target_ids: List[int] = []
        aligned: List[AlignedSegment] = []
        protected_draft_positions: Set[int] = set()
        trie_hits = single_hits = cache_hits = dtw_fallbacks = 0
        i = 0
        while i < len(draft_ids):
            # Special/control tokens take priority over the artifact map so a
            # draft stop maps to a real target stop (not its literal surface
            # text).  Mark the segment as protected so healing never re-decodes
            # it back into junk text.
            sp = self._special_id_map.get(draft_ids[i])
            if sp is not None:
                start = len(target_ids)
                target_ids.extend(sp)
                aligned.append(AlignedSegment(i, i + 1, start, len(target_ids)))
                protected_draft_positions.add(i)
                single_hits += 1
                i += 1
                continue

            match_len, mapped = self._longest_ngram(draft_ids, i)
            if mapped:
                start = len(target_ids)
                target_ids.extend(mapped)
                aligned.append(AlignedSegment(i, i + match_len, start, len(target_ids)))
                trie_hits += 1
                i += match_len
                continue

            if self._single_map_int is not None:
                single = self._single_map_int.get(draft_ids[i])
            else:
                single = self.artifact.get_single(draft_ids[i]) if self.artifact else None
            if single:
                start = len(target_ids)
                target_ids.extend(single)
                aligned.append(AlignedSegment(i, i + 1, start, len(target_ids)))
                single_hits += 1
                i += 1
                continue

            fallback_ids, cache_hit = self._fallback_map_tuple((draft_ids[i],))
            start = len(target_ids)
            target_ids.extend(fallback_ids)
            aligned.append(AlignedSegment(i, i + 1, start, len(target_ids)))
            cache_hits += int(cache_hit)
            dtw_fallbacks += 1
            i += 1

        segments = self._segments_from_alignment(aligned, logq)
        segments = self._merge_zero_target_segments(segments)
        protect = {
            index
            for index, segment in enumerate(segments)
            if any(
                segment.draft_start <= position < segment.draft_end
                for position in protected_draft_positions
            )
        }
        if self._heal_boundaries:
            target_ids, segments = self._heal_segments(
                draft_ids, target_ids, segments, protect=protect
            )
        target_logq = self._terminal_charge(len(target_ids), segments)
        return MappingResult(
            target_ids=target_ids,
            target_logq=target_logq,
            segments=segments,
            valid_mask=[True] * len(target_ids),
            boundary_state=BoundaryState(),
            trie_hits=trie_hits,
            single_hits=single_hits,
            cache_hits=cache_hits,
            dtw_fallbacks=dtw_fallbacks,
        )

    def _build_seam_bitmap(self) -> bytearray:
        """G3: materialize right-edge seam safety for the whole target vocab.

        Byte-identical to the lazy ``_is_safe_seam`` rule (same
        ``decode_one``+``canonicalize_piece``), just precomputed once so the
        per-seam check becomes an O(1) index.  Returns an empty bytearray if the
        vocab size cannot be determined (callers then fall back to the dict).
        """
        tok = self.target_tokenizer
        try:
            vocab_size = int(getattr(tok, "vocab_size", 0) or 0)
            if vocab_size <= 0:
                vocab_size = max(tok.get_vocab().values()) + 1
        except Exception:
            return bytearray()
        if vocab_size <= 0:
            return bytearray()
        arr = bytearray(vocab_size)
        ws = (" ", "\n", "\t", "\r")
        for tid in range(vocab_size):
            try:
                right = canonicalize_piece(decode_one(tok, tid))
            except Exception:
                right = ""
            arr[tid] = 1 if ((not right) or right[0] in ws) else 0
        return arr

    def _is_safe_seam(self, left_target_id: int, right_target_id: int) -> bool:
        """A seam is safe when target BPE cannot merge across it.

        The decisive signal is a leading whitespace/newline marker on the right
        token (SentencePiece ``\u2581`` / GPT ``\u0120`` are normalized to a
        space by ``canonicalize_piece``); such a token always starts a fresh
        merge group in the target tokenizer, so the stitched join is already
        canonical there.
        """
        if self._seam_bitmap_enabled:
            arr = self._seam_safe_arr
            if arr is None:
                arr = self._build_seam_bitmap()
                self._seam_safe_arr = arr
            if arr and 0 <= right_target_id < len(arr):
                return bool(arr[right_target_id])
        cached = self._safe_seam_cache.get(right_target_id)
        if cached is not None:
            return cached
        right = canonicalize_piece(decode_one(self.target_tokenizer, right_target_id))
        safe = (not right) or right[0] in (" ", "\n", "\t", "\r")
        self._safe_seam_cache[right_target_id] = safe
        return safe

    def _heal_segments(
        self,
        draft_ids: Sequence[int],
        target_ids: List[int],
        segments: List[MappingSegment],
        protect: Optional[Set[int]] = None,
    ) -> Tuple[List[int], List[MappingSegment]]:
        """Re-tokenize only the spans between *unsafe* seams to canonicalize them.

        Text is preserved (each healed run is the same draft text, re-encoded as
        one string) and the run's total draft log-prob is conserved, so the SMC
        importance weight stays valid -- only the proxy token boundaries change.

        ``protect`` lists segment indices that must never be merged into a healed
        run (special/control tokens whose draft surface form would re-decode to
        junk text); seams touching them are treated as safe.
        """
        protect = protect or set()
        n = len(segments)
        if n <= 1:
            return target_ids, segments

        # risky[k] == True  <=>  the seam *before* segment k may carry a merge.
        risky = [False] * n
        any_risky = False
        for k in range(1, n):
            if k in protect or (k - 1) in protect:
                continue
            b = segments[k].target_start
            if b <= 0 or b >= len(target_ids):
                continue
            if not self._is_safe_seam(target_ids[b - 1], target_ids[b]):
                risky[k] = True
                any_risky = True
        if not any_risky:
            return target_ids, segments

        new_target_ids: List[int] = []
        new_segments: List[MappingSegment] = []
        i = 0
        while i < n:
            j = i
            while j + 1 < n and risky[j + 1]:
                j += 1
            if j == i:
                seg = segments[i]
                start = len(new_target_ids)
                new_target_ids.extend(target_ids[seg.target_start : seg.target_end])
                new_segments.append(
                    MappingSegment(
                        seg.draft_start,
                        seg.draft_end,
                        start,
                        len(new_target_ids),
                        seg.draft_logq,
                    )
                )
            else:
                group = segments[i : j + 1]
                d0 = group[0].draft_start
                d1 = group[-1].draft_end
                # Memoize the (decode draft -> encode target) round-trip: a healed
                # run is a pure function of its draft-id tuple, and GSM8K repeats
                # the same numbers/words constantly, so this turns most heals into
                # a dict hit instead of two Python tokenizer calls.
                heal_key = tuple(draft_ids[d0:d1])
                canon = self.cache.get(heal_key)
                if canon is None:
                    self._heal_reencodes += 1
                    text = self._decode_many(self.draft_tokenizer, draft_ids[d0:d1])
                    canon = encode_text(self.target_tokenizer, text)
                    self.cache.put(heal_key, canon)
                total_logq = sum(s.draft_logq for s in group)
                start = len(new_target_ids)
                new_target_ids.extend(canon)
                new_segments.append(
                    MappingSegment(d0, d1, start, len(new_target_ids), float(total_logq))
                )
                self._heal_groups += 1
                if self._online_ngram:
                    self._maybe_promote_ngram(heal_key, canon)
            i = j + 1

        return new_target_ids, new_segments

    def _maybe_promote_ngram(self, key: Tuple[int, ...], canon: List[int]) -> None:
        """G4: promote a hot healed run into the n-gram trie.

        Lossless: ``_longest_ngram`` will then return ``canon`` for this exact
        draft tuple as a single aligned segment, which ``_segments_from_alignment``
        scores with ``sum(logq[d0:d1])`` -- identical to the merged healed segment
        this run already produces, so the resulting MappingResult is unchanged.
        """
        ngram_map = self._ngram_map_int
        if ngram_map is None:
            return
        n = len(key)
        if n < 2 or n > self._online_ngram_max_len or key in ngram_map:
            return
        count = self._heal_run_counts.get(key, 0) + 1
        if count < self._online_ngram_min:
            self._heal_run_counts[key] = count
            return
        if (
            len(ngram_map) - self._artifact_ngram_base
            >= self._online_ngram_max_entries
        ):
            return
        ngram_map[key] = list(canon)
        if self._ngram_first is not None:
            self._ngram_first.add(key[0])
        if n > self._max_ngram_len_fast:
            self._max_ngram_len_fast = n
        self._online_ngram_promotions += 1
        self._heal_run_counts.pop(key, None)

    def _longest_ngram(self, draft_ids: Sequence[int], start: int) -> Tuple[int, List[int]]:
        if self._ngram_map_int is not None:
            # Lossless fast path: skip the whole probe when this id starts no
            # n-gram, else look up int-tuple keys (no per-probe string build).
            if draft_ids[start] not in self._ngram_first:
                return 0, []
            max_len = min(self._max_ngram_len_fast, len(draft_ids) - start)
            for length in range(max_len, 1, -1):
                mapped = self._ngram_map_int.get(
                    tuple(draft_ids[start : start + length])
                )
                if mapped is not None:
                    return length, list(mapped)
            return 0, []
        if self.artifact is None:
            return 0, []
        max_len = min(self.artifact.max_ngram_len(), len(draft_ids) - start)
        for length in range(max_len, 1, -1):
            mapped = self.artifact.get_ngram(draft_ids[start : start + length])
            if mapped is not None:
                return length, mapped
        return 0, []

    def _fallback_map_tuple(self, draft_ids: Tuple[int, ...]) -> Tuple[List[int], bool]:
        cached = self.cache.get(draft_ids)
        if cached is not None:
            return cached, True
        text = self._decode_many(self.draft_tokenizer, draft_ids)
        target_ids = encode_text(self.target_tokenizer, text)
        self.cache.put(draft_ids, target_ids)
        return target_ids, False

    def _segments_from_alignment(
        self, aligned: Iterable[AlignedSegment], logq: Sequence[float]
    ) -> List[MappingSegment]:
        out: List[MappingSegment] = []
        for seg in aligned:
            draft_logq = sum(logq[seg.draft_start : seg.draft_end])
            out.append(
                MappingSegment(
                    draft_start=seg.draft_start,
                    draft_end=seg.draft_end,
                    target_start=seg.target_start,
                    target_end=seg.target_end,
                    draft_logq=float(draft_logq),
                )
            )
        return out

    @staticmethod
    def _merge_zero_target_segments(
        segments: Sequence[MappingSegment],
    ) -> List[MappingSegment]:
        """Attach zero-target byte fragments to the next visible event.

        Byte-fallback tokenizers can represent one UTF-8 character with several
        draft tokens whose isolated target encodings are empty. Dropping those
        segments in ``_terminal_charge`` silently loses draft path probability.
        Merging them into the next visible segment conserves the sampled charge
        and gives the combined byte event a valid target location.
        """

        merged: List[MappingSegment] = []
        pending: List[MappingSegment] = []
        for segment in segments:
            if segment.target_end <= segment.target_start:
                pending.append(segment)
                continue
            if pending:
                merged.append(
                    MappingSegment(
                        draft_start=pending[0].draft_start,
                        draft_end=segment.draft_end,
                        target_start=segment.target_start,
                        target_end=segment.target_end,
                        draft_logq=float(
                            sum(item.draft_logq for item in pending)
                            + segment.draft_logq
                        ),
                    )
                )
                pending.clear()
            else:
                merged.append(segment)
        if pending:
            if merged:
                previous = merged[-1]
                merged[-1] = MappingSegment(
                    draft_start=previous.draft_start,
                    draft_end=pending[-1].draft_end,
                    target_start=previous.target_start,
                    target_end=previous.target_end,
                    draft_logq=float(
                        previous.draft_logq
                        + sum(item.draft_logq for item in pending)
                    ),
                )
            else:
                merged.extend(pending)
        return merged

    @staticmethod
    def _terminal_charge(target_len: int, segments: Iterable[MappingSegment]) -> List[float]:
        target_logq = [0.0] * target_len
        for seg in segments:
            if seg.target_end <= seg.target_start:
                continue
            # A mapped segment represents one sampled draft-path event. Its
            # complete path score must be visible as soon as any target token
            # from the segment is committed; charging at the final target
            # subtoken leaves an unterminated prefix with no proposal debit.
            target_logq[seg.target_start] += seg.draft_logq
        return target_logq

    @staticmethod
    def _decode_many(tokenizer: Any, ids: Sequence[int]) -> str:
        try:
            return tokenizer.decode(
                list(ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(list(ids))


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
