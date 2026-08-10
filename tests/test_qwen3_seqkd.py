from smcsd.qwen3_seqkd import (
    SplitConfig,
    assign_split,
    expected_user_content,
    normalized_reasoning,
    question_id,
    split_question_ids,
    strict_terminal_answer,
    validation_error,
)
from scripts.qwen3_build_seqkd_dataset import build_rows


def test_strict_terminal_answer_requires_exact_final_marker():
    assert strict_terminal_answer("Work.\n#### 1,200.00") == "1200"
    assert strict_terminal_answer("Work.\n#### 12 \n") is None
    assert strict_terminal_answer("Work.\n#### 12\nMore work") is None
    assert strict_terminal_answer("Work.\n#### twelve") is None


def test_normalized_reasoning_ignores_case_unicode_and_whitespace():
    first = "Step\u00a0One:   Add values.\n\n#### 42"
    second = " step one: add VALUES.\n#### 42"
    assert normalized_reasoning(first) == normalized_reasoning(second)


def test_validation_requires_train_prompt_gold_and_exact_marker():
    question = "A value is 40. What is it plus 2?"
    identifier = question_id(question)
    train_questions = {identifier: "42"}
    valid = {
        "source": "gsm8k",
        "correct": True,
        "sft_messages": [
            {"role": "user", "content": expected_user_content(question)},
            {"role": "assistant", "content": "40 + 2 = 42.\n#### 42"},
        ],
    }
    assert validation_error(valid, train_questions) == (None, identifier, "42")

    wrong_gold = {
        **valid,
        "sft_messages": [
            valid["sft_messages"][0],
            {"role": "assistant", "content": "40 + 2 = 42.\n#### 41"},
        ],
    }
    assert validation_error(wrong_gold, train_questions)[0] == (
        "assistant_final_disagrees_with_gsm8k_gold"
    )

    test_only_question = {
        **valid,
        "sft_messages": [
            {"role": "user", "content": expected_user_content("A test-only question.")},
            {"role": "assistant", "content": "Answer.\n#### 42"},
        ],
    }
    assert validation_error(test_only_question, train_questions)[0] == (
        "question_not_in_openai_gsm8k_main_train"
    )


def test_question_level_splits_are_deterministic_and_disjoint():
    identifiers = [question_id(f"Question {index}") for index in range(1000)]
    config = SplitConfig()
    first = split_question_ids(identifiers, config)
    second = split_question_ids(reversed(identifiers), config)
    assert first == second
    assert sum(map(len, first.values())) == len(identifiers)
    assert len(set().union(*map(set, first.values()))) == len(identifiers)
    assert {
        assign_split(identifier, config) for identifier in identifiers
    } == {"train", "dev", "intrinsic", "end_to_end_development"}


def test_build_rows_deduplicates_normalized_reasoning_per_question():
    question = "A value is 40. What is it plus 2?"
    train_questions = {question_id(question): "42"}
    duplicate_a = {
        "source": "gsm8k",
        "correct": True,
        "sft_messages": [
            {"role": "user", "content": expected_user_content(question)},
            {"role": "assistant", "content": "Add values.\n#### 42"},
        ],
    }
    duplicate_b = {
        **duplicate_a,
        "sft_messages": [
            duplicate_a["sft_messages"][0],
            {"role": "assistant", "content": " add VALUES. \n#### 42"},
        ],
    }
    split_rows, rejections, _ = build_rows(
        [(1, duplicate_a), (2, duplicate_b)], train_questions, SplitConfig()
    )
    assert sum(len(rows) for rows in split_rows.values()) == 1
    assert rejections["duplicate_normalized_reasoning"] == 1
