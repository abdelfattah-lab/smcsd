"""Open-loop serving benchmark: Poisson arrivals against an sglang-style
/generate endpoint (stock server with sampling n>1, or smcsd.http_server).

Each request is one benchmark question. Client-side selection: majority
vote over returned choices when n>1 (the same contract for both systems),
single answer otherwise. Reports achieved QPS, latency percentiles,
accuracy, and token counts.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter


async def one_request(session, url, prompt, sp, results, idx):
    t0 = time.perf_counter()
    try:
        async with session.post(
            url + "/generate",
            json={"text": prompt, "sampling_params": sp},
            timeout=600,
        ) as resp:
            data = await resp.json()
    except Exception as e:  # noqa: BLE001
        results[idx] = {"error": str(e), "latency": time.perf_counter() - t0}
        return
    lat = time.perf_counter() - t0
    if isinstance(data, list):
        texts = [d.get("text", "") for d in data]
        toks = sum(d.get("meta_info", {}).get("completion_tokens", 0) for d in data)
    else:
        texts = [data.get("text", "")]
        toks = data.get("meta_info", {}).get("completion_tokens", 0)
    results[idx] = {"texts": texts, "latency": lat, "tokens": toks}


async def run(args, prompts):
    import aiohttp

    rng = random.Random(args.seed)
    results = [None] * len(prompts)
    sp = {"max_new_tokens": args.max_new_tokens,
          "temperature": args.temperature}
    if args.n > 1:
        sp["n"] = args.n
    async with aiohttp.ClientSession() as session:
        tasks = []
        t_start = time.perf_counter()
        for i, p in enumerate(prompts):
            tasks.append(asyncio.create_task(
                one_request(session, args.url, p, sp, results, i)))
            if i < len(prompts) - 1:
                await asyncio.sleep(rng.expovariate(args.rate))
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t_start
    return results, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--benchmark", choices=["gsm8k", "math500"], required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="model path for prompt construction")
    ap.add_argument("--num-questions", type=int, default=100)
    ap.add_argument("--rate", type=float, default=1.0, help="arrivals/sec")
    ap.add_argument("--n", type=int, default=1,
                    help="choices per request (stock parallel sampling)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disable-thinking", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    if args.benchmark == "gsm8k":
        from accuracy_test_gsm8k import load_gsm8k as load, extract_answer
    else:
        from accuracy_test_math500 import (
            load_math500 as load, extract_boxed, normalize_answer)

        def extract_answer(t):
            return normalize_answer(extract_boxed(t))

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts, labels = load(tok, args.num_questions,
                           disable_thinking=args.disable_thinking)

    results, wall = asyncio.run(run(args, prompts))

    lats = [r["latency"] for r in results if r and "latency" in r]
    errs = sum(1 for r in results if r and "error" in r)
    acc = 0
    total_tokens = 0
    for r, gold in zip(results, labels):
        if not r or "texts" not in r:
            continue
        total_tokens += r.get("tokens", 0)
        votes = [extract_answer(t) for t in r["texts"]]
        votes = [v for v in votes if v is not None]
        pred = Counter(votes).most_common(1)[0][0] if votes else None
        acc += pred == gold
    lats.sort()

    def pct(p):
        return lats[min(len(lats) - 1, int(p * len(lats)))] if lats else 0

    print(f"requests={len(prompts)} errors={errs} rate={args.rate}/s n={args.n}")
    print(f"  achieved QPS:    {len(prompts) / wall:.2f}")
    print(f"  accuracy:        {acc}/{len(prompts)}")
    print(f"  latency p50/p95/p99: {pct(0.5):.1f} / {pct(0.95):.1f} / {pct(0.99):.1f} s")
    print(f"  mean latency:    {statistics.mean(lats):.1f}s" if lats else "")
    print(f"  completion tokens: {total_tokens}")
    print(f"  wall: {wall:.1f}s")


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
