import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("SGLANG_ENABLE_SPEC_V2", "1")
venv_bin = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

import sglang as sgl


MODEL_PATH = "/home/cc2869/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DRAFT_MODEL_PATH = MODEL_PATH
PROMPTS = [
    "The capital of France is",
    "Write one sentence about why overlap scheduling matters for inference systems.",
    "List two prime numbers and one composite number.",
    "In one short paragraph, explain speculative decoding.",
]
SAMPLING_PARAMS = {
    "temperature": 0,
    "max_new_tokens": 32,
    "ignore_eos": True,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Override the global attention backend for the SMC probe.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()
    engine_kwargs = dict(
        model_path=MODEL_PATH,
        speculative_algorithm="SMC",
        speculative_draft_model_path=DRAFT_MODEL_PATH,
        smc_n_particles=4,
        smc_gamma=3,
        page_size=1,
        cuda_graph_max_bs=4,
        mem_fraction_static=0.45,
        trust_remote_code=True,
        log_level="info",
    )
    if args.attention_backend is not None:
        engine_kwargs["attention_backend"] = args.attention_backend

    with sgl.Engine(**engine_kwargs) as engine:
        server_info = engine.get_server_info()
        print(
            "SERVER_INFO",
            json.dumps(
                {
                    "speculative_algorithm": server_info.get("speculative_algorithm"),
                    "disable_overlap_schedule": server_info.get(
                        "disable_overlap_schedule"
                    ),
                    "smc_n_particles": server_info.get("smc_n_particles"),
                    "smc_gamma": server_info.get("smc_gamma"),
                    "attention_backend": server_info.get("attention_backend"),
                    "avg_spec_accept_length": server_info.get("internal_states", [{}])[
                        0
                    ].get("avg_spec_accept_length"),
                },
                indent=2,
            ),
        )

        outputs = engine.generate(PROMPTS, SAMPLING_PARAMS)
        for i, (prompt, output) in enumerate(zip(PROMPTS, outputs, strict=True), start=1):
            print(f"PROMPT_{i}: {prompt}")
            print(f"OUTPUT_{i}: {output['text']}")
            print(
                "META_{}: {}".format(
                    i, json.dumps(output.get("meta_info"), indent=2, default=str)
                )
            )
            print("-" * 80)

    print(f"TOTAL_SECONDS {time.time() - start:.2f}")


if __name__ == "__main__":
    main()
