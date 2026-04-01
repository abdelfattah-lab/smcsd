"""Compare outputs across SSD, SGLang, vLLM at temp=0.

Uses /v1/completions with text prompt. Servers auto-add BOS, producing 49
tokens from the 48-token chat template. This matches SSD's internal behavior.
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(__file__))
from bench_paths import HF_CACHE_DIR, resolve_snapshot

TARGET_70B = resolve_snapshot(f"{HF_CACHE_DIR}/models--meta-llama--Llama-3.1-70B-Instruct")
PROMPT = "Explain the theory of general relativity in exactly three sentences."


def get_text_prompt():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TARGET_70B)
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return text, tokenizer


def query_server(url, text_prompt, max_tokens=128):
    import requests
    payload = {"model": TARGET_70B, "prompt": text_prompt,
               "temperature": 0, "max_tokens": max_tokens, "stream": False}
    resp = requests.post(f"{url}/v1/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["text"], data.get("usage", {})


def query_ssd(text_prompt, max_tokens=128):
    import ssd.paths  # noqa
    from ssd import LLM, SamplingParams

    llm = LLM(TARGET_70B, enforce_eager=False, num_gpus=4,
              kvcache_block_size=256, max_num_seqs=1, max_model_len=8192)
    sp = SamplingParams(temperature=0, max_new_tokens=max_tokens)
    outputs, _ = llm.generate([text_prompt], [sp], use_tqdm=False)
    return outputs[0]["text"], {"completion_tokens": len(outputs[0]["token_ids"])}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ssd", "sglang", "vllm"], required=True)
    p.add_argument("--url", type=str, default=None)
    p.add_argument("--max_tokens", type=int, default=128)
    args = p.parse_args()

    text_prompt, tokenizer = get_text_prompt()
    print(f"Prompt: {text_prompt[:100]}...")

    if args.backend == "ssd":
        text, usage = query_ssd(text_prompt, args.max_tokens)
    else:
        assert args.url, "Need --url for server backends"
        text, usage = query_server(args.url, text_prompt, args.max_tokens)

    print(f"=== {args.backend.upper()} ===")
    print(text)
    print(f"=== END ({len(text)} chars, usage={usage}) ===")
    with open(f"/tmp/{args.backend}_output.txt", "w") as f:
        f.write(text)
