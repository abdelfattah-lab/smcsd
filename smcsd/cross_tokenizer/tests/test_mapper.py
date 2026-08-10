from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from smcsd.cross_tokenizer.artifacts import (
    MappingArtifact,
    MappingMetadata,
    MappingStats,
    load_artifact,
    save_artifact,
    tokenizer_fingerprint,
)
from smcsd.cross_tokenizer.build import _load_corpus_texts, build_artifact
from smcsd.cross_tokenizer.bucketing import bucket_proxy_sequences
from smcsd.cross_tokenizer.dtw import AlignedSegment
from smcsd.cross_tokenizer.lineage import plan_cross_draft_lineage
from smcsd.cross_tokenizer.mapper import BoundaryState, TokenMapper
from smcsd.cross_tokenizer.spans import shared_text_spans


class FakeTokenizer:
    def __init__(self, pieces, name="fake"):
        self.pieces = dict(pieces)
        self.name_or_path = name
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.all_special_ids = []
        self.special_tokens_map = {}
        self._text_to_id = {text: idx for idx, text in self.pieces.items()}

    def get_vocab(self):
        return {text: idx for idx, text in self.pieces.items()}

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[int(idx)] for idx in ids)

    def encode(self, text, **_kwargs):
        out = []
        i = 0
        # Longest-match tokenization.
        sorted_pieces = sorted(self._text_to_id, key=len, reverse=True)
        while i < len(text):
            for piece in sorted_pieces:
                if text.startswith(piece, i):
                    out.append(self._text_to_id[piece])
                    i += len(piece)
                    break
            else:
                raise ValueError(f"Cannot encode suffix {text[i:]!r}")
        return out


class TokenMapperTest(unittest.TestCase):
    def test_live_mapping_conserves_logq_for_many_to_one(self):
        draft = FakeTokenizer({1: "f", 2: "la", 3: "ke"}, name="draft")
        target = FakeTokenizer({10: "flake"}, name="target")
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            mode="live",
        )

        result = mapper.map_tokens([1, 2, 3], [-0.1, -0.2, -0.3])

        self.assertEqual(result.target_ids, [10])
        self.assertAlmostEqual(sum(result.target_logq), -0.6)
        self.assertEqual(len(result.segments), 1)

    def test_live_mapping_repairs_incomplete_utf8_alignment(self):
        draft = FakeTokenizer(
            {1: "π", 2: " ", 3: "≈", 4: "x"}, name="draft"
        )
        target = FakeTokenizer(
            {10: "π", 11: " ", 12: "≈", 13: "x"}, name="target"
        )
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            mode="live",
        )
        incomplete = [
            AlignedSegment(0, 1, 0, 1),
            AlignedSegment(3, 4, 3, 4),
        ]

        with patch(
            "smcsd.cross_tokenizer.mapper.align_token_texts",
            return_value=incomplete,
        ):
            result = mapper.map_tokens(
                [1, 2, 3, 4], [-0.1, -0.2, -0.3, -0.4]
            )

        self.assertEqual(result.target_ids, [10, 11, 12, 13])
        self.assertAlmostEqual(sum(result.target_logq), -1.0)
        self.assertEqual(
            (result.segments[0].draft_start, result.segments[0].draft_end),
            (0, 4),
        )
        self.assertEqual(mapper._alignment_repairs, 1)

    def test_hybrid_single_map_conserves_logq_for_one_to_many(self):
        draft = FakeTokenizer({1: "flake"}, name="draft")
        target = FakeTokenizer({10: "f", 11: "lake"}, name="target")
        artifact = build_artifact(
            draft_tokenizer=draft,
            target_tokenizer=target,
            draft_tokenizer_name="draft",
            target_tokenizer_name="target",
        )
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            artifact=artifact,
            mode="hybrid",
        )

        result = mapper.map_tokens([1], [-1.25])

        self.assertEqual(result.target_ids, [10, 11])
        # The complete sampled draft-path score is available on the first
        # target subtoken, so a decode stop after ``f`` still has a complete
        # proposal charge.
        self.assertEqual(result.target_logq, [-1.25, 0.0])
        self.assertAlmostEqual(sum(result.target_logq), -1.25)
        self.assertEqual(result.single_hits, 1)

    def test_hybrid_heals_unsafe_word_internal_single_map_seam(self):
        draft = FakeTokenizer({1: "1", 2: "6"}, name="draft")
        target = FakeTokenizer({10: "16", 11: "1", 12: "6"}, name="target")
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            artifact=build_artifact(
                draft_tokenizer=draft,
                target_tokenizer=target,
                draft_tokenizer_name="draft",
                target_tokenizer_name="target",
            ),
            mode="hybrid",
        )

        result = mapper.map_tokens([1, 2], [-0.2, -0.3])

        self.assertEqual(result.target_ids, [10])
        self.assertEqual(result.target_logq, [-0.5])
        self.assertEqual(mapper._heal_groups, 1)

    def test_hybrid_prefers_longest_static_ngram(self):
        draft = FakeTokenizer({1: "1", 2: "6"}, name="draft")
        target = FakeTokenizer({10: "16", 11: "1", 12: "6"}, name="target")
        artifact = build_artifact(
            draft_tokenizer=draft,
            target_tokenizer=target,
            draft_tokenizer_name="draft",
            target_tokenizer_name="target",
            corpus_texts=["16"],
            max_ngram_entries=1,
        )
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            artifact=artifact,
            mode="hybrid",
        )

        result = mapper.map_tokens([1, 2], [-0.2, -0.3])

        self.assertEqual(result.target_ids, [10])
        self.assertEqual(result.target_logq, [-0.5])
        self.assertEqual(result.trie_hits, 1)
        self.assertEqual(mapper._heal_groups, 0)

    def test_hybrid_unmapped_token_uses_live_fallback(self):
        draft = FakeTokenizer({1: "abcdef"}, name="draft")
        target = FakeTokenizer(
            {10: "a", 11: "b", 12: "c", 13: "d", 14: "e", 15: "f"},
            name="target",
        )
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            artifact=build_artifact(
                draft_tokenizer=draft,
                target_tokenizer=target,
                draft_tokenizer_name="draft",
                target_tokenizer_name="target",
                max_single_token_span=4,
            ),
            mode="hybrid",
        )

        result = mapper.map_tokens([1], [-0.75])

        self.assertEqual(result.target_ids, [10, 11, 12, 13, 14, 15])
        self.assertEqual(result.target_logq, [-0.75, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(result.dtw_fallbacks, 1)

    def test_default_flush_marks_overlength_mapping_for_discard(self):
        draft = FakeTokenizer({1: "abcdef"}, name="draft")
        target = FakeTokenizer(
            {10: "a", 11: "b", 12: "c", 13: "d", 14: "e", 15: "f"},
            name="target",
        )
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            artifact=build_artifact(
                draft_tokenizer=draft,
                target_tokenizer=target,
                draft_tokenizer_name="draft",
                target_tokenizer_name="target",
            ),
            mode="hybrid",
        )

        batch = mapper.map_batch(
            [[1]],
            [[-0.75]],
            max_proxy_len=4,
        )

        self.assertEqual(batch.raw_proxy_lens, [6])
        self.assertEqual(batch.proxy_lens, [0])
        self.assertEqual(batch.overlength_mask, [True])
        self.assertEqual(batch.empty_proxy_mask, [False])
        self.assertEqual(batch.proxy_ids, [[0, 0, 0, 0]])

    def test_default_flush_discards_all_empty_mapped_block(self):
        draft = FakeTokenizer({1: ""}, name="draft")
        target = FakeTokenizer({10: "x"}, name="target")
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            artifact=build_artifact(
                draft_tokenizer=draft,
                target_tokenizer=target,
                draft_tokenizer_name="draft",
                target_tokenizer_name="target",
            ),
            mode="hybrid",
        )

        batch = mapper.map_batch(
            [[1]],
            [[-0.75]],
            max_proxy_len=4,
        )

        self.assertEqual(batch.raw_proxy_lens, [0])
        self.assertEqual(batch.proxy_lens, [0])
        self.assertEqual(batch.overlength_mask, [True])
        self.assertEqual(batch.empty_proxy_mask, [True])
        self.assertEqual(mapper.stats.empty_proxy_rows, 1)

    def test_artifact_never_maps_special_surface_to_normal_token(self):
        draft = FakeTokenizer({1: "<|draft_control|>"}, name="draft")
        draft.all_special_ids = [1]
        target = FakeTokenizer({10: "<|draft_control|>"}, name="target")

        artifact = build_artifact(
            draft_tokenizer=draft,
            target_tokenizer=target,
            draft_tokenizer_name="draft",
            target_tokenizer_name="target",
        )

        self.assertEqual(artifact.special_token_map, {})
        self.assertNotIn("1", artifact.single_map)
        self.assertEqual(artifact.stats.blocked_specials, 1)

    def test_hybrid_uses_explicit_matching_special_token_map(self):
        draft = FakeTokenizer({1: "<|shared_control|>"}, name="draft")
        draft.all_special_ids = [1]
        target = FakeTokenizer({10: "<|shared_control|>"}, name="target")
        target.all_special_ids = [10]
        artifact = build_artifact(
            draft_tokenizer=draft,
            target_tokenizer=target,
            draft_tokenizer_name="draft",
            target_tokenizer_name="target",
        )
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            artifact=artifact,
            mode="hybrid",
        )

        result = mapper.map_tokens([1], [-0.4])

        self.assertEqual(artifact.special_token_map, {"1": 10})
        self.assertEqual(result.target_ids, [10])
        self.assertEqual(result.target_logq, [-0.4])

    def test_zero_target_fragment_does_not_unprotect_adjacent_special(self):
        draft = FakeTokenizer(
            {1: "x", 2: "", 3: "<|shared_control|>", 4: "y"},
            name="draft",
        )
        draft.all_special_ids = [3]
        target = FakeTokenizer(
            {10: "x", 20: "<|shared_control|>", 30: "y"},
            name="target",
        )
        target.all_special_ids = [20]
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            artifact=build_artifact(
                draft_tokenizer=draft,
                target_tokenizer=target,
                draft_tokenizer_name="draft",
                target_tokenizer_name="target",
            ),
            mode="hybrid",
        )

        result = mapper.map_tokens(
            [1, 2, 3, 4],
            [-0.1, -0.2, -0.3, -0.4],
        )

        self.assertEqual(result.target_ids, [10, 20, 30])
        self.assertAlmostEqual(sum(result.target_logq), -1.0)
        self.assertEqual(mapper._heal_groups, 0)

    def test_corpus_loader_preserves_significant_edge_whitespace(self):
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as corpus:
            corpus.write("  leading and trailing  \n\nindented\n")
            corpus.flush()
            texts = _load_corpus_texts(
                SimpleNamespace(
                    corpus_file=corpus.name,
                    gsm8k_num_questions=0,
                )
            )

        self.assertEqual(texts, ["  leading and trailing  ", "indented"])

    def test_shared_spans_mark_byte_identical_mappings(self):
        draft = FakeTokenizer({1: "flake"}, name="draft")
        target = FakeTokenizer({10: "f", 11: "lake"}, name="target")
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            artifact=build_artifact(
                draft_tokenizer=draft,
                target_tokenizer=target,
                draft_tokenizer_name="draft",
                target_tokenizer_name="target",
            ),
            mode="hybrid",
        )

        result = mapper.map_tokens([1], [-1.25])
        spans = shared_text_spans(
            draft_tokenizer=draft,
            target_tokenizer=target,
            draft_ids=[1],
            target_ids=result.target_ids,
            segments=result.segments,
        )

        self.assertEqual(len(spans), 1)
        self.assertTrue(spans[0].byte_identical)
        self.assertEqual(spans[0].text, "flake")

    def test_artifact_round_trip(self):
        draft = FakeTokenizer({1: "a"}, name="draft")
        target = FakeTokenizer({2: "a"}, name="target")
        artifact = MappingArtifact(
            metadata=MappingMetadata(
                draft_tokenizer="draft",
                target_tokenizer="target",
                draft_fingerprint=tokenizer_fingerprint(draft),
                target_fingerprint=tokenizer_fingerprint(target),
            ),
            stats=MappingStats(draft_vocab_size=1, target_vocab_size=1),
            single_map={"1": [2]},
        )

        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            save_artifact(artifact, f.name)
            loaded = load_artifact(f.name)

        self.assertEqual(loaded.get_single(1), [2])
        loaded.validate_for(draft, target)

    def test_hybrid_mapper_rejects_artifact_for_different_tokenizer(self):
        artifact_draft = FakeTokenizer({1: "a"}, name="artifact-draft")
        runtime_draft = FakeTokenizer({1: "a"}, name="runtime-draft")
        target = FakeTokenizer({2: "a"}, name="target")
        artifact = build_artifact(
            draft_tokenizer=artifact_draft,
            target_tokenizer=target,
            draft_tokenizer_name="artifact-draft",
            target_tokenizer_name="target",
        )

        with self.assertRaisesRegex(
            ValueError, "draft tokenizer fingerprint mismatch"
        ):
            TokenMapper(
                draft_tokenizer=runtime_draft,
                target_tokenizer=target,
                artifact=artifact,
                mode="hybrid",
            )

    def test_fingerprint_rejects_changed_vocab_with_same_name_and_size(self):
        artifact_draft = FakeTokenizer({1: "a"}, name="same-draft")
        runtime_draft = FakeTokenizer({1: "b"}, name="same-draft")
        target = FakeTokenizer({2: "a", 3: "b"}, name="target")
        artifact = build_artifact(
            draft_tokenizer=artifact_draft,
            target_tokenizer=target,
            draft_tokenizer_name="same-draft",
            target_tokenizer_name="target",
        )

        with self.assertRaisesRegex(
            ValueError, "draft tokenizer fingerprint mismatch"
        ):
            TokenMapper(
                draft_tokenizer=runtime_draft,
                target_tokenizer=target,
                artifact=artifact,
                mode="hybrid",
            )

    def test_proxy_bucketing(self):
        bucketed = bucket_proxy_sequences(
            [[1, 2, 3], [4]],
            [[-0.1, 0.0, -0.2], [-0.3]],
            pad_token_id=0,
            buckets=(2, 4),
        )

        self.assertEqual(bucketed.bucket_size, 4)
        self.assertEqual(bucketed.input_ids, [[1, 2, 3, 0], [4, 0, 0, 0]])
        self.assertEqual(
            bucketed.valid_mask,
            [[True, True, True, False], [True, False, False, False]],
        )

    def test_identity_vocab_fast_path_preserves_tokens(self):
        draft = FakeTokenizer({1: "a", 2: "b"}, name="draft")
        target = FakeTokenizer({1: "a", 2: "b"}, name="target")
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            mode="live",
        )

        result = mapper.map_batch([[1, 2]], [[-0.4, -0.7]], max_proxy_len=2)

        self.assertEqual(result.proxy_ids, [[1, 2]])
        self.assertEqual(result.proxy_lens, [2])
        self.assertEqual(result.proxy_logq, [[-0.4, -0.7]])
        self.assertEqual(result.row_stats, [{
            "trie_hits": 0,
            "single_hits": 2,
            "cache_hits": 0,
            "dtw_fallbacks": 0,
        }])
        self.assertEqual(mapper.stats.single_hits, 2)

    def test_streaming_boundary_carry_matches_one_block_mapping(self):
        draft = FakeTokenizer({1: "1", 2: "6", 3: " "}, name="draft")
        target = FakeTokenizer(
            {10: "16", 11: " ", 12: "1", 13: "6"}, name="target"
        )
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            artifact=build_artifact(
                draft_tokenizer=draft,
                target_tokenizer=target,
                draft_tokenizer_name="draft",
                target_tokenizer_name="target",
            ),
            mode="hybrid",
        )

        first = mapper.map_tokens([1], [-0.1], flush=False)
        self.assertEqual(first.target_ids, [])
        self.assertEqual(first.boundary_state.draft_suffix_ids, [1])

        second = mapper.map_tokens(
            [2, 3], [-0.2, -0.3], first.boundary_state, flush=False
        )
        self.assertEqual(second.target_ids, [10])
        self.assertEqual(second.boundary_state.draft_suffix_ids, [3])

        final = mapper.map_tokens(
            [], [], second.boundary_state, flush=True
        )
        one_block = mapper.map_tokens(
            [1, 2, 3], [-0.1, -0.2, -0.3], flush=True
        )
        streamed_ids = first.target_ids + second.target_ids + final.target_ids
        streamed_logq = (
            first.target_logq + second.target_logq + final.target_logq
        )
        self.assertEqual(streamed_ids, one_block.target_ids)
        self.assertAlmostEqual(sum(streamed_logq), sum(one_block.target_logq))
        self.assertEqual(
            draft.decode([1, 2, 3]), target.decode(streamed_ids)
        )
        self.assertTrue(final.boundary_state.is_empty)

    def test_eos_flushes_an_unsafe_carried_suffix(self):
        draft = FakeTokenizer({1: "x", 9: "<eos>"}, name="draft")
        target = FakeTokenizer({10: "x", 99: "<eos>"}, name="target")
        draft.eos_token_id = 9
        draft.all_special_ids = [9]
        target.eos_token_id = 99
        target.all_special_ids = [99]
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            mode="live",
        )

        first = mapper.map_tokens([1], [-0.1], flush=False)
        flushed = mapper.map_tokens(
            [9], [-0.2], first.boundary_state, flush=False
        )

        self.assertEqual(flushed.target_ids, [10, 99])
        self.assertAlmostEqual(sum(flushed.target_logq), -0.3)
        self.assertTrue(flushed.boundary_state.is_empty)

    def test_overlength_stream_commits_maximal_stable_prefix(self):
        draft = FakeTokenizer({1: "abc", 2: " def"}, name="draft")
        target = FakeTokenizer(
            {
                10: "a",
                11: "b",
                12: "c",
                13: " ",
                14: "d",
                15: "e",
                16: "f",
            },
            name="target",
        )
        mapper = TokenMapper(
            draft_tokenizer=draft,
            target_tokenizer=target,
            artifact=build_artifact(
                draft_tokenizer=draft,
                target_tokenizer=target,
                draft_tokenizer_name="draft",
                target_tokenizer_name="target",
            ),
            mode="hybrid",
        )

        batch = mapper.map_batch(
            [[1, 2]],
            [[-0.4, -0.7]],
            max_proxy_len=4,
            boundary_states=[BoundaryState()],
            flush=False,
        )

        self.assertEqual(batch.raw_proxy_lens, [7])
        self.assertEqual(batch.proxy_lens, [3])
        self.assertEqual(batch.proxy_ids[0][:3], [10, 11, 12])
        self.assertEqual(batch.overlength_mask, [False])
        self.assertEqual(batch.boundary_states[0].draft_suffix_ids, [2])
        remainder = mapper.map_tokens(
            [], [], batch.boundary_states[0], flush=True
        )
        self.assertEqual(
            batch.proxy_ids[0][:3] + remainder.target_ids,
            [10, 11, 12, 13, 14, 15, 16],
        )

    def test_lineage_plans_deferred_multi_token_bonus_suffix(self):
        plan = plan_cross_draft_lineage(
            current_verified_ids=[10],
            previous_last_draft_ids=[9],
            proposal_draft_ids=[[11, 12, 13]],
            bonus_draft_ids=[[21, 22]],
            overlength_mask=[False],
            draft_deferred=True,
        )

        self.assertEqual(plan.histories, [[10, 11, 12, 13, 21]])
        self.assertEqual(plan.draft_visible_lens, [5])
        self.assertEqual(plan.draft_verified_ids, [22])
        self.assertEqual(plan.prev_last_draft_ids, [21])
        self.assertEqual(plan.normal_deferred_suffix, [[13, 21]])
        self.assertEqual(plan.normal_legacy_suffix, [[]])
        self.assertEqual(plan.overlength_suffix, [[]])

    def test_lineage_plans_overlength_multi_token_bonus_suffix(self):
        plan = plan_cross_draft_lineage(
            current_verified_ids=[10],
            previous_last_draft_ids=[9],
            proposal_draft_ids=[[11, 12, 13]],
            bonus_draft_ids=[[21, 22, 23]],
            overlength_mask=[True],
            draft_deferred=True,
        )

        self.assertEqual(plan.histories, [[10, 21, 22]])
        self.assertEqual(plan.draft_visible_lens, [3])
        self.assertEqual(plan.draft_verified_ids, [23])
        self.assertEqual(plan.prev_last_draft_ids, [22])
        self.assertEqual(plan.normal_deferred_suffix, [[]])
        self.assertEqual(plan.overlength_suffix, [[21, 22]])

    def test_lineage_advances_draft_without_target_bonus_on_carry(self):
        plan = plan_cross_draft_lineage(
            current_verified_ids=[10],
            previous_last_draft_ids=[9],
            proposal_draft_ids=[[11, 12, 13]],
            bonus_draft_ids=[[]],
            overlength_mask=[False],
            draft_deferred=True,
            emit_target_bonus=[False],
        )

        self.assertEqual(plan.histories, [[10, 11, 12]])
        self.assertEqual(plan.draft_visible_lens, [3])
        self.assertEqual(plan.draft_verified_ids, [13])
        self.assertEqual(plan.prev_last_draft_ids, [12])
        self.assertEqual(plan.normal_deferred_suffix, [[]])
        self.assertEqual(plan.normal_legacy_suffix, [[]])
        self.assertEqual(plan.overlength_suffix, [[]])


if __name__ == "__main__":
    unittest.main()
