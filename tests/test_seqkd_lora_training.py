from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_lora_train_distill.py"
SPEC = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
assert SPEC is not None and SPEC.loader is not None
train = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train
SPEC.loader.exec_module(train)


class FakeTokenizer:
    eos_token_id = 255
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        assert tokenize is False
        user = messages[0]["content"]
        prefix = f"<user>{user}</user><assistant>"
        if add_generation_prompt:
            return prefix
        return prefix + messages[-1]["content"] + "</assistant>"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


def test_messages_for_accepts_both_supported_schemas():
    sft = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert train.messages_for({"sft_messages": sft}) == sft
    assert train.messages_for(
        {
            "messages": [{"role": "user", "content": "question"}],
            "teacher_text": "answer",
        }
    ) == sft


def test_seqkd_dataset_masks_prompt_and_keeps_response_labels():
    tokenizer = FakeTokenizer()
    row = {
        "sft_messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    dataset = train.SeqKDDataset([row], tokenizer, max_length=128)
    assert len(dataset) == 1
    example = dataset[0]
    prompt = tokenizer.apply_chat_template(
        row["sft_messages"][:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
    assert example["labels"][:prompt_length] == [-100] * prompt_length
    assert example["labels"][prompt_length:] == example["input_ids"][prompt_length:]
    assert example["input_ids"][-1] == tokenizer.eos_token_id


def test_seqkd_collator_masks_padding():
    collator = train.SeqKDCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]
