from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DraftLineagePlan:
    histories: list[list[int]]
    draft_visible_lens: list[int]
    draft_verified_ids: list[int]
    prev_last_draft_ids: list[int]
    normal_deferred_suffix: list[list[int]]
    normal_legacy_suffix: list[list[int]]
    overlength_suffix: list[list[int]]


def plan_cross_draft_lineage(
    *,
    current_verified_ids: Sequence[int],
    previous_last_draft_ids: Sequence[int],
    proposal_draft_ids: Sequence[Sequence[int]],
    bonus_draft_ids: Sequence[Sequence[int]],
    overlength_mask: Sequence[bool],
    draft_deferred: bool,
    emit_target_bonus: Sequence[bool] | None = None,
) -> DraftLineagePlan:
    """Plan draft-side prefix growth after cross-tokenizer target verification.

    `draft_seq_lens` excludes the next `draft_verified_id`, so each history row
    contains exactly the draft tokens that become committed prefix before that
    next seed.  Overlength rows discard the mapped proposal on the target side,
    so they must discard it on the draft side too.
    """
    bs = len(current_verified_ids)
    if emit_target_bonus is None:
        emit_target_bonus = [True] * bs
    if not (
        len(previous_last_draft_ids)
        == len(proposal_draft_ids)
        == len(bonus_draft_ids)
        == len(overlength_mask)
        == len(emit_target_bonus)
        == bs
    ):
        raise ValueError("all lineage inputs must have the same batch size")

    histories: list[list[int]] = []
    draft_visible_lens: list[int] = []
    draft_verified_ids: list[int] = []
    prev_last_draft_ids: list[int] = []
    normal_deferred_suffix: list[list[int]] = [[] for _ in range(bs)]
    normal_legacy_suffix: list[list[int]] = [[] for _ in range(bs)]
    overlength_suffix: list[list[int]] = [[] for _ in range(bs)]

    for row in range(bs):
        current = int(current_verified_ids[row])
        proposal = [int(token_id) for token_id in proposal_draft_ids[row]]
        bonus = [int(token_id) for token_id in bonus_draft_ids[row]]
        if not bool(emit_target_bonus[row]):
            if not proposal:
                histories.append([])
                draft_visible_lens.append(0)
                draft_verified_ids.append(current)
                prev_last_draft_ids.append(int(previous_last_draft_ids[row]))
                continue
            # A target carry round emits no bonus.  Advance the draft policy
            # through its full sampled block, leaving its final sampled token
            # as the next uncommitted verified seed (the same convention used
            # by the target side for its last committed proxy token).
            history = [current] + proposal[:-1]
            histories.append(history)
            draft_visible_lens.append(len(history))
            draft_verified_ids.append(proposal[-1])
            prev_last_draft_ids.append(history[-1])
            continue
        if not bonus:
            histories.append([])
            draft_visible_lens.append(0)
            draft_verified_ids.append(current)
            prev_last_draft_ids.append(int(previous_last_draft_ids[row]))
            continue

        bonus_prefix = bonus[:-1]
        draft_verified_ids.append(int(bonus[-1]))
        if bool(overlength_mask[row]):
            history = [current] + bonus_prefix
            if bonus_prefix:
                overlength_suffix[row] = bonus_prefix
                prev_last_draft_ids.append(int(bonus_prefix[-1]))
            else:
                prev_last_draft_ids.append(current)
        else:
            if not proposal:
                raise ValueError("non-overlength rows require proposal draft ids")
            history = [current] + proposal + bonus_prefix
            if bonus_prefix:
                prev_last_draft_ids.append(int(bonus_prefix[-1]))
                if draft_deferred:
                    normal_deferred_suffix[row] = [proposal[-1]] + bonus_prefix
                else:
                    normal_legacy_suffix[row] = bonus_prefix
            else:
                prev_last_draft_ids.append(int(proposal[-1]))

        histories.append(history)
        draft_visible_lens.append(len(history))

    return DraftLineagePlan(
        histories=histories,
        draft_visible_lens=draft_visible_lens,
        draft_verified_ids=draft_verified_ids,
        prev_last_draft_ids=prev_last_draft_ids,
        normal_deferred_suffix=normal_deferred_suffix,
        normal_legacy_suffix=normal_legacy_suffix,
        overlength_suffix=overlength_suffix,
    )
