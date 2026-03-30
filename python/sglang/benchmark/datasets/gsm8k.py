import os
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow
from sglang.utils import download_and_cache_file, read_jsonl

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def _get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def _get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += _get_one_example(lines, i, True) + "\n\n"
    return ret


@dataclass
class GSM8KDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    num_shots: int
    fixed_output_len: Optional[int]
    apply_chat_template: bool

    @classmethod
    def from_args(cls, args: Namespace) -> "GSM8KDataset":
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            num_shots=getattr(args, "gsm8k_num_shots", 5),
            fixed_output_len=getattr(args, "sharegpt_output_len", None),
            apply_chat_template=args.apply_chat_template,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        # Download if needed
        data_path = self.dataset_path
        if not data_path or not os.path.isfile(data_path):
            data_path = download_and_cache_file(GSM8K_URL)

        lines = list(read_jsonl(data_path))
        few_shot_examples = _get_few_shot_examples(lines, self.num_shots)

        filtered: List[DatasetRow] = []
        for i in range(self.num_shots, len(lines)):
            if len(filtered) >= self.num_requests:
                break

            raw = few_shot_examples + _get_one_example(lines, i, False)

            if self.apply_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if tokenizer.bos_token:
                    prompt = prompt.replace(tokenizer.bos_token, "")
            else:
                prompt = raw

            prompt_len = len(tokenizer.encode(prompt))
            output_len = self.fixed_output_len if self.fixed_output_len else 512

            filtered.append(
                DatasetRow(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                )
            )

        print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered])}")
        print(f"#Output tokens: {np.sum([x.output_len for x in filtered])}")
        return filtered
