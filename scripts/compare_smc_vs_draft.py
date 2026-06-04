"""Compare draft-only SMC vLLM vs normal vLLM on the same draft model.

This checkpoint compares:

- SMCVLLMEngine in draft-only mode (currently no target verify/resampling),
  using `n_particles=1`
- a normal vLLM `LLM` using the same draft model

Both engines are driven from the same tokenized prompts rather than raw text.

Each engine runs in its own subprocess for CUDA context isolation.

Usage:
  python scripts/compare_smc_vs_draft.py
  python scripts/compare_smc_vs_draft.py --temperature 0.8 --max-tokens 200
  python scripts/compare_smc_vs_draft.py --mode smc      # SMC only
  python scripts/compare_smc_vs_draft.py --mode draft    # draft model only
  python scripts/compare_smc_vs_draft.py --mode both     # default
  python scripts/compare_smc_vs_draft.py --particles 4 --gamma 4
"""

import argparse
import multiprocessing as mp
import traceback

from transformers import AutoTokenizer

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

PROMPTS = [
    "The capital of France is",
    "Write one sentence about why overlap scheduling matters for inference systems.",
    "List two prime numbers and one composite number.",
    "In one short paragraph, explain speculative decoding.",
    "What is 1+1?",
    "Explain the difference between a list and a tuple in Python.",
    "What are the main causes of the French Revolution?",
]

def _smc_worker(prompt_token_ids, args, queue):
    try:
        from smcsd.vllm_backend.engine import SMCVLLMEngine

        engine = SMCVLLMEngine(
            model_path=args.model,
            draft_model_path=args.draft_model,
            n_particles=1,
            gamma=args.gamma,
            tp_size=1,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem,
            enable_prefix_caching=False
        )
        sampling_params = {
            "draft_temperature": args.temperature,
            "target_temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        outputs = engine.generate(input_ids=prompt_token_ids, sampling_params=sampling_params)
        engine.shutdown()
        results = [outputs] if isinstance(outputs, dict) else outputs
        queue.put(("ok", [{"text": r["text"], "n_particles": len(r.get("particles", []))} for r in results]))
    except Exception as e:
        queue.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def _draft_worker(prompt_token_ids, args, queue):
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.draft_model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem,
            enforce_eager=True,
            async_scheduling=False,
            enable_prefix_caching=False
        )
        sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
        prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
        raw_outputs = llm.generate(prompts, sp)
        results = [{"text": out.outputs[0].text} for out in raw_outputs]
        queue.put(("ok", results))
    except Exception as e:
        queue.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def _run_in_subprocess(worker_fn, prompt_token_ids, args) -> list[dict]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=worker_fn, args=(prompt_token_ids, args, queue))
    proc.start()
    proc.join()
    status, payload = queue.get()
    if status == "err":
        raise RuntimeError(payload)
    return payload


def print_smc(prompts, results, args):
    print(f"\n{'=' * 64}")
    print(
        f"  SMC-vLLM  |  target={args.model.split('/')[-1]}"
        f"  draft={args.draft_model.split('/')[-1]}"
    )
    print(
        f"             |  N={args.particles}  γ={args.gamma}"
        f"  temp={args.temperature}  max_tokens={args.max_tokens}"
    )
    print(f"{'=' * 64}")
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\n  [{i}] {prompt!r}")
        print(f"  → {result['text'][:args.max_display_chars]}")
        if result.get("n_particles", 0) > 1:
            print(f"    (particles: {result['n_particles']})")


def print_draft(prompts, results, args):
    print(f"\n{'=' * 64}")
    print(f"  vLLM draft-only  |  model={args.draft_model.split('/')[-1]}")
    print(f"                   |  temp={args.temperature}  max_tokens={args.max_tokens}")
    print(f"{'=' * 64}")
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\n  [{i}] {prompt!r}")
        print(f"  → {result['text'][:args.max_display_chars]}")


def print_comparison(prompts, smc_results, draft_results, max_chars):
    print(f"\n{'=' * 64}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * 64}")
    for i, prompt in enumerate(prompts):
        smc_text = smc_results[i]["text"][:max_chars]
        draft_text = draft_results[i]["text"][:max_chars]
        match = "SAME" if smc_text.strip() == draft_text.strip() else "DIFF"
        print(f"\n[{i}] {prompt!r}  [{match}]")
        print(f"  SMC  : {smc_text}")
        print(f"  Draft: {draft_text}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="target model path")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL,
                        help="draft model path (SMC draft and standalone baseline)")
    parser.add_argument("--mode", choices=["smc", "draft", "both"], default="both")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--particles", "-N", type=int, default=1)
    parser.add_argument("--gamma", "-g", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.2)
    parser.add_argument("--max-display-chars", type=int, default=300)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.draft_model, trust_remote_code=True
    )
    prompt_token_ids = []
    for prompt in PROMPTS:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        rendered = dict(rendered)
        prompt_token_ids.append(rendered["input_ids"])

    smc_results = None
    draft_results = None

    if args.mode in ("smc", "both"):
        smc_results = _run_in_subprocess(_smc_worker, prompt_token_ids, args)
        print_smc(PROMPTS, smc_results, args)

    if args.mode in ("draft", "both"):
        draft_results = _run_in_subprocess(_draft_worker, prompt_token_ids, args)
        print_draft(PROMPTS, draft_results, args)

    if args.mode == "both" and smc_results and draft_results:
        print_comparison(PROMPTS, smc_results, draft_results, args.max_display_chars)


if __name__ == "__main__":
    main()
