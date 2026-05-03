"""Long-prompt TPS benchmark: fa3 eager vs flashinfer+cascade eager.

Builds a synthetic prompt of ~target_len tokens by repeating a passage,
asks the model to answer a question that requires reading the prompt,
and reports decode TPS.  Long prompts are where the Hydragen / cascade
inference wins live.

Usage:
  python scripts/bench_long_prompt.py
  python scripts/bench_long_prompt.py --target-len 8192 --particles 12
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from transformers import AutoTokenizer

from smcsd.engine import SMCEngine


PASSAGE = """\
The Roman Empire spanned three continents at its height, governing roughly 70 million
people across territories from Britain to North Africa to Mesopotamia. Its dominance
rested on three pillars: a remarkable military machine built around the legion, an
administrative apparatus that absorbed local elites into Roman civic life, and a
network of roads, aqueducts, and harbors that knit the provinces into a single
economic system. Trade in grain, olive oil, wine, and luxury goods flowed along these
arteries; Roman coinage circulated as far as India and the Baltic. Yet the same
infrastructure that bound the empire together also exposed it to long-running stresses:
plagues that travelled along the trade routes, fiscal pressure from a permanent army,
and a reliance on slave labour that hollowed out the small-holding peasantry.
"""


def build_prompt(tokenizer, target_tokens: int, question: str) -> str:
    """Build a chat-template prompt of approximately `target_tokens` tokens."""
    base_ids = tokenizer.encode(PASSAGE, add_special_tokens=False)
    needed = max(target_tokens - 256, 64)  # leave headroom for the question
    repeats = max(1, needed // len(base_ids))
    body = (PASSAGE + "\n") * repeats
    instruction = (
        f"You are given a long context passage.  Read it carefully and answer "
        f"the question that follows in 2-3 sentences.\n\n"
        f"=== CONTEXT ===\n{body}\n=== QUESTION ===\n{question}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_one(
    *, model: str, draft: str, prompt: str, cascade: bool,
    n_particles: int, gamma: int, max_new: int, mem_fraction: float, seed: int,
    disable_cuda_graph: bool = False,
) -> tuple[float, int, str]:
    backend = "flashinfer" if cascade else "fa3"
    mode_str = "eager" if disable_cuda_graph else "graphs"
    print(f"\n--- Booting SMCEngine cascade={cascade} attn={backend} {mode_str} ---")
    torch.manual_seed(seed)
    engine = SMCEngine(
        model_path=model,
        draft_model_path=draft,
        n_particles=n_particles,
        gamma=gamma,
        draft_temperature=0.7,
        target_temperature=0.7,
        resample_threshold=0.5,
        shared_prefix_attn=cascade,
        attention_backend=backend,
        mem_fraction_static=mem_fraction,
        max_running_requests=1,
        disable_cuda_graph=disable_cuda_graph,
    )
    sp = {"temperature": 0.7, "max_new_tokens": max_new}
    # Warmup forward (don't time)
    _ = engine.generate(prompt, sampling_params=sp)
    t0 = time.perf_counter()
    out = engine.generate(prompt, sampling_params=sp)
    dt = time.perf_counter() - t0
    completion = out["completion_tokens"]
    text = out["text"]
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return dt, completion, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--draft", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--target-len", type=int, default=4096,
                    help="approximate input prompt length in tokens")
    ap.add_argument("--question", default="What three pillars supported Roman dominance?")
    ap.add_argument("--particles", type=int, default=12)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable-cuda-graph", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt = build_prompt(tok, args.target_len, args.question)
    actual_len = len(tok.encode(prompt))
    print(f"prompt actual_len: {actual_len} tokens (target {args.target_len})")
    print(f"settings: N={args.particles} γ={args.gamma} max_new={args.max_new}")

    print("\n========== fa3 baseline (eager) ==========")
    t_fa3, n_fa3, _ = run_one(
        model=args.model, draft=args.draft, prompt=prompt,
        cascade=False,
        n_particles=args.particles, gamma=args.gamma,
        max_new=args.max_new, mem_fraction=args.mem_fraction, seed=args.seed,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    print(f"  fa3:  {t_fa3:.2f}s, {n_fa3} new tokens, {n_fa3/t_fa3:.1f} tok/s")

    print("\n========== flashinfer + cascade (eager) ==========")
    t_cas, n_cas, _ = run_one(
        model=args.model, draft=args.draft, prompt=prompt,
        cascade=True,
        n_particles=args.particles, gamma=args.gamma,
        max_new=args.max_new, mem_fraction=args.mem_fraction, seed=args.seed,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    print(f"  cas:  {t_cas:.2f}s, {n_cas} new tokens, {n_cas/t_cas:.1f} tok/s")

    print("\n=" * 1 + "=" * 50)
    print(f"  prompt_len={actual_len}  N={args.particles}  γ={args.gamma}")
    print(f"  fa3 baseline:        {n_fa3/t_fa3:>7.1f} tok/s ({t_fa3:.2f}s for {n_fa3} tok)")
    print(f"  flashinfer+cascade:  {n_cas/t_cas:>7.1f} tok/s ({t_cas:.2f}s for {n_cas} tok)")
    sp = (n_cas / t_cas) / (n_fa3 / t_fa3)
    marker = "🚀" if sp >= 1.10 else ("≈" if sp >= 0.95 else "  ")
    print(f"  cascade speedup:     {sp:>7.2f}x  {marker}")
    print("=" * 51)


if __name__ == "__main__":
    main()
