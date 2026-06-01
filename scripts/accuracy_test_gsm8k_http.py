"""GSM8K accuracy benchmark for SMC over an HTTP server (online serving).

HTTP-server counterpart to ``accuracy_test_gsm8k.py`` (which runs the offline
``SMCEngine``). It uses the *same* preprocessing and scoring as the offline test
(zero-shot instruction + ``#### <number>`` format, ``extract_answer``), so the
online numbers are directly comparable to the offline ``smc_engine`` reference.

Why not sglang's ``few_shot_gsm8k``? That helper relies on stop strings
(``stop=["Question", ...]``) and SMC does not support stop strings
(``smcsd/common/utils.py``: "SMC speculative decoding does not yet support stop
strings"). The zero-shot ``#### <number>`` format needs no stop strings -- it
terminates on EOS -- so it works with SMC and matches the offline path.

The SMC HTTP server is the standard sglang ``http_server`` with the SMC scheduler
swapped in (see ``smcsd/http_server.py``). Requests go to the native ``/generate``
endpoint concurrently; acceptance length is read from ``/server_info`` afterwards
(the same source as sglang's GSM8KMixin).

Usage::

    source .venv/bin/activate

    # Launch the SMC server, run GSM8K, tear it down (self-contained):
    python scripts/accuracy_test_gsm8k_http.py -N 8 -g 8 --num-questions 200

    # Baseline (no speculative decoding) reference:
    python scripts/accuracy_test_gsm8k_http.py --mode baseline --num-questions 200

    # Benchmark an already-running server (launch it separately):
    python -m smcsd.http_server -N 8 -g 8 --port 30000 --trust-remote-code
    python scripts/accuracy_test_gsm8k_http.py --base-url http://127.0.0.1:30000

On shared machines you typically need ``CUDA_HOME`` set and a fresh
``FLASHINFER_WORKSPACE_BASE`` (see scripts/README.md).
"""

import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from datasets import load_dataset
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Shared preprocessing (mirrors accuracy_test_gsm8k.py for comparable scoring)
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    """Extract numeric answer from model output or gold answer."""
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_line = lines[-1] if lines else text.strip()
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last_line)
    return numbers[-1].replace(",", "") if numbers else None


def format_instruction(question: str) -> str:
    """Build the instruction prompt for a GSM8K question."""
    return (
        "Solve this math problem step by step.\n"
        "At the very end, output ONLY the final numeric answer "
        "on a new line in the exact format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n"
    )


def load_gsm8k(tokenizer, num_questions: int):
    """Load GSM8K and build chat-template prompts + gold labels."""
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    prompts, labels = [], []
    for sample in dataset.select(range(num_questions)):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": format_instruction(sample["question"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        labels.append(extract_answer(sample["answer"]))
    assert all(label is not None for label in labels), "Some gold labels could not be parsed"
    return prompts, labels


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _split_base_url(base_url):
    netloc = base_url.split("://", 1)[-1]
    host, port = netloc.rsplit(":", 1)
    return host, int(port)


def _wait_until_healthy(base_url, proc, timeout):
    """Poll /health until the server is up, failing fast if the process dies."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(base_url + "/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def launch_smc_server(args):
    """Launch the SMC HTTP server (smcsd.http_server) as a subprocess."""
    base_url = f"http://127.0.0.1:{args.port}"
    log_path = f"/tmp/smc_http_server_{args.port}.log"
    cmd = [
        sys.executable, "-m", "smcsd.http_server",
        "--model", args.model,
        "--draft-model", args.draft_model,
        "-N", str(args.particles),
        "-g", str(args.gamma),
        "--draft-temperature", str(args.temperature),
        "--target-temperature", str(args.temperature),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        # user-facing concurrent groups; expanded by (N+1) inside the server
        "--max-running-requests", str(args.parallel),
        "--attention-backend", args.attention_backend,
        "--trust-remote-code",
    ]
    if args.resample_threshold is not None:
        cmd += ["--resample-threshold", str(args.resample_threshold)]
    if args.mem_fraction_static is not None:
        cmd += ["--mem-fraction-static", str(args.mem_fraction_static)]
    if args.cuda_graph_max_bs is not None:
        cmd += ["--cuda-graph-max-bs", str(args.cuda_graph_max_bs)]

    print(f"Launching SMC HTTP server (log -> {log_path}):\n  {' '.join(cmd)}\n")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    if not _wait_until_healthy(base_url, proc, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        if proc.poll() is None:
            kill_process_tree(proc.pid)
        with open(log_path) as f:
            tail = "".join(f.readlines()[-40:])
        raise RuntimeError(
            f"SMC server did not become healthy. Last lines of {log_path}:\n{tail}"
        )
    return proc, base_url


def launch_baseline_server(args):
    """Launch a vanilla sglang server (no speculative decoding) via `sglang serve`."""
    base_url = f"http://127.0.0.1:{args.port}"
    launch = [
        "--trust-remote-code",
        "--attention-backend", args.attention_backend,
        "--max-running-requests", str(args.parallel),
    ]
    if args.mem_fraction_static is not None:
        launch += ["--mem-fraction-static", str(args.mem_fraction_static)]
    if args.cuda_graph_max_bs is not None:
        launch += ["--cuda-graph-max-bs", str(args.cuda_graph_max_bs)]
    proc = popen_launch_server(
        args.model, base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, other_args=launch
    )
    return proc, base_url


def get_accept_length(base_url):
    """Server-wide average speculative acceptance length (None if unavailable).

    Mirrors sglang's GSM8KMixin: read after the eval, since the key is absent
    until at least one speculative forward has happened.
    """
    try:
        info = requests.get(base_url + "/server_info", timeout=10).json()
    except requests.RequestException:
        return None
    states = info.get("internal_states") or []
    if states:
        return states[0].get("avg_spec_accept_length")
    return None


# ---------------------------------------------------------------------------
# Evaluation (concurrent native /generate, no stop strings -- SMC-compatible)
# ---------------------------------------------------------------------------


def _generate_one(base_url, prompt, args):
    sampling_params = {"max_new_tokens": args.max_new_tokens}
    if args.ignore_eos:
        sampling_params["ignore_eos"] = True
    resp = requests.post(
        base_url + "/generate",
        json={"text": prompt, "sampling_params": sampling_params},
        timeout=args.request_timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["text"], data["meta_info"].get("completion_tokens", 0)


def run_gsm8k(base_url, prompts, labels, args):
    requests.get(base_url + "/flush_cache")
    n = len(prompts)
    preds = [None] * n
    ntoks = [0] * n
    samples = {}

    tic = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {
            ex.submit(_generate_one, base_url, p, args): i
            for i, p in enumerate(prompts)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                text, nt = fut.result()
                preds[i] = extract_answer(text)
                ntoks[i] = nt
                if i < 3:
                    samples[i] = (nt, text)
            except Exception as e:  # count failures as invalid, keep going
                preds[i] = None
                if i < 3:
                    samples[i] = (0, f"<request failed: {e}>")
            done += 1
            elapsed = time.perf_counter() - tic
            correct = sum(
                pred == label
                for pred, label in zip(preds, labels)
                if pred is not None
            )
            print(
                f"\r[{done}/{n}] acc={correct}/{done} ({correct / done:.1%}) "
                f"tps={sum(ntoks) / elapsed:.0f} elapsed={elapsed:.0f}s",
                end="",
                flush=True,
            )
    latency = time.perf_counter() - tic
    print()
    for i in sorted(samples):
        nt, text = samples[i]
        print(f"\n--- Q{i} ({nt} tokens) ---\n{text[:400]}")

    correct = sum(pred == label for pred, label in zip(preds, labels))
    invalid = sum(p is None for p in preds)
    total_tokens = sum(ntoks)
    return {
        "accuracy": correct / n,
        "invalid": invalid / n,
        "correct": correct,
        "n": n,
        "output_throughput": total_tokens / latency if latency else 0.0,
        "latency": latency,
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=["smc_server", "baseline"], default="smc_server",
        help="smc_server = SMC HTTP server, baseline = vanilla sglang server",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="Benchmark an already-running server instead of launching one.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)

    smc = parser.add_argument_group("SMC parameters")
    smc.add_argument("--particles", "-N", type=int, default=8)
    smc.add_argument("--gamma", "-g", type=int, default=8)
    smc.add_argument(
        "--temperature", type=float, default=0.7,
        help="SMC draft+target sampling temperature (server-side).",
    )
    smc.add_argument("--resample-threshold", type=float, default=None)

    bench = parser.add_argument_group("benchmark")
    bench.add_argument("--num-questions", type=int, default=200)
    bench.add_argument("--max-new-tokens", type=int, default=512)
    bench.add_argument(
        "--parallel", type=int, default=16,
        help="Concurrent requests AND server concurrent-group capacity.",
    )
    bench.add_argument(
        "--ignore-eos", action="store_true",
        help="Pass ignore_eos for throughput comparisons (matches offline flag).",
    )
    bench.add_argument("--request-timeout", type=float, default=600.0)
    bench.add_argument("--accuracy-thres", type=float, default=None,
                       help="If set, exit non-zero when accuracy is below this.")
    bench.add_argument("--accept-length-thres", type=float, default=None,
                       help="If set, exit non-zero when avg accept length is below this.")

    srv = parser.add_argument_group("server overrides")
    srv.add_argument("--port", type=int, default=30000)
    srv.add_argument("--attention-backend", default="triton", choices=["triton", "fa3"])
    srv.add_argument("--mem-fraction-static", type=float, default=None)
    srv.add_argument("--cuda-graph-max-bs", type=int, default=None)
    args = parser.parse_args()

    label = {
        "smc_server": "SMC HTTP server",
        "baseline": "Baseline HTTP server (vanilla)",
    }[args.mode]
    print(f"Mode: {label} | Model: {args.model}")
    if args.mode == "smc_server":
        print(
            f"  draft={args.draft_model}, N={args.particles}, gamma={args.gamma}, "
            f"smc_temperature={args.temperature}"
        )
    print(
        f"  num_questions={args.num_questions}, max_new_tokens={args.max_new_tokens}, "
        f"parallel={args.parallel}\n"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, labels = load_gsm8k(tokenizer, args.num_questions)

    proc = None
    if args.base_url:
        base_url = args.base_url
        print(f"Benchmarking existing server at {base_url}\n")
    elif args.mode == "baseline":
        proc, base_url = launch_baseline_server(args)
    else:
        proc, base_url = launch_smc_server(args)

    try:
        metrics = run_gsm8k(base_url, prompts, labels, args)
        accept_len = get_accept_length(base_url)
    finally:
        if proc is not None:
            kill_process_tree(proc.pid)

    print(f"\n{'=' * 55}")
    print(f"  {label}")
    if args.mode == "smc_server":
        print(f"  N={args.particles}, gamma={args.gamma}, smc_temp={args.temperature}")
    print(f"{'=' * 55}")
    print(f"  Accuracy:          {metrics['correct']}/{metrics['n']} "
          f"({100 * metrics['accuracy']:.1f}%)")
    print(f"  Invalid:           {100 * metrics['invalid']:.1f}%")
    print(f"  Output throughput: {metrics['output_throughput']:.1f} tok/s")
    print(f"  Total tokens:      {metrics['total_tokens']}")
    print(f"  Wall time:         {metrics['latency']:.1f}s")
    if accept_len is not None:
        print(f"  Accept length:     {accept_len:.3f}")
    elif args.mode == "smc_server":
        print("  Accept length:     n/a (avg_spec_accept_length not reported)")
    print(f"{'=' * 55}")

    failed = False
    if args.accuracy_thres is not None and metrics["accuracy"] < args.accuracy_thres:
        print(f"FAIL: accuracy {metrics['accuracy']:.3f} < threshold {args.accuracy_thres}")
        failed = True
    if args.accept_length_thres is not None:
        if accept_len is None:
            print("FAIL: accept-length threshold set but no accept length reported")
            failed = True
        elif accept_len < args.accept_length_thres:
            print(f"FAIL: accept length {accept_len:.3f} < threshold {args.accept_length_thres}")
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
