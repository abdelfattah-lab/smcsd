"""SMC speculative decoding on a hybrid Qwen target (Qwen3-Next / Qwen3.5).

Smoke / iteration entry-point for adding hybrid-model support to smcsd.
First task: just *launch* SMCEngine with target = hybrid Qwen + draft = small
dense Qwen (matching tokenizer) on Modal H200s, generate a handful of tokens
on a couple of GSM8K questions, and surface the first failure.

The expectation on the first run is that SMCEngine init or the first decode
step will crash — that's the data point we use to decide what to patch.

Usage:
    # Tiny smoke test (default): 2 questions, 2 particles, 32 new tokens
    modal run modal_apps/run_smc_hybrid_qwen.py

    # Full GSM8K eval after wiring is correct:
    modal run modal_apps/run_smc_hybrid_qwen.py \
        --num-questions 200 --max-new-tokens 512 \
        --particles 8 --gamma 4

    # Use the Qwen3.5 family instead of Qwen3-Next:
    modal run modal_apps/run_smc_hybrid_qwen.py \
        --target-model Qwen/Qwen3.5-30B-A3B-Instruct
"""

from __future__ import annotations

import modal

APP_NAME = "smcsd-smc-hybrid-qwen"
SMCSD_DIR = "/root/smcsd"
HF_HOME = "/root/.cache/huggingface"
SMCSD_REPO = "https://github.com/abdelfattah-lab/smcsd.git"
# The image clones this branch from origin; local edits are then layered
# on top via add_local_dir / add_local_file so we can iterate without
# re-pushing every patch. Push when you want a clean cold-start image.
SMCSD_BRANCH = "qwen35-hybrid"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "ca-certificates", "curl", "protobuf-compiler")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
        'git config --global url."https://github.com/".insteadOf "git@github.com:"',
        f"git clone --branch {SMCSD_BRANCH} --recurse-submodules {SMCSD_REPO} {SMCSD_DIR}",
        f"cd {SMCSD_DIR} && pip install --upgrade pip wheel",
        f'export PATH=$HOME/.cargo/bin:$PATH && cd {SMCSD_DIR} && pip install -e 3rdparty/sglang/python',
        f"cd {SMCSD_DIR} && pip install -e .",
        "pip install datasets transformers numpy",
    )
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*"
    )
    # Ship the local working tree so runtime smcsd matches what we just edited
    # without needing a fresh git push every iteration.
    .add_local_dir(
        "smcsd",
        f"{SMCSD_DIR}/smcsd",
        copy=True,
    )
    .add_local_file(
        "scripts/accuracy_test_gsm8k.py",
        f"{SMCSD_DIR}/scripts/accuracy_test_gsm8k.py",
        copy=True,
    )
    # The cached `git clone --recurse-submodules` layer pinned the sglang
    # submodule at an older commit; overlay our patched files directly so
    # SMC-specific tweaks (skip draft-MTP rewrite, don't restrict draft
    # full_attention_layer_ids on independent drafts) take effect without
    # busting the whole image cache.
    .add_local_file(
        "3rdparty/sglang/python/sglang/srt/configs/model_config.py",
        f"{SMCSD_DIR}/3rdparty/sglang/python/sglang/srt/configs/model_config.py",
        copy=True,
    )
    .add_local_file(
        "3rdparty/sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py",
        f"{SMCSD_DIR}/3rdparty/sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py",
        copy=True,
    )
    .env({"HF_HOME": HF_HOME, "PYTHONUNBUFFERED": "1"})
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("smcsd-hf-cache", create_if_missing=True)


@app.function(
    gpu="H200:4",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hf-llama-token")],
    volumes={HF_HOME: hf_cache},
)
def run_smc_hybrid(
    target_model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct",
    draft_model: str = "Qwen/Qwen3-0.6B",
    num_questions: int = 3,
    max_new_tokens: int = 1024,
    particles: int = 2,
    gamma: int = 2,
    temperature: float = 0.7,
    tp_size: int = 4,
    mem_fraction_static: float = 0.35,
    cuda_graph_max_bs: int = 2,
    max_running_requests: int = 2,
    attention_backend: str = "fa3",
    resample_threshold: float = 0.5,
    context_length: int = 8192,
    disable_cuda_graph: bool = False,
    mode: str = "smc_engine",
) -> None:
    import os
    import shlex
    import subprocess
    import sys

    # Token plumbing — Modal secret may set any of these; SGLang/HF look for HF_TOKEN.
    for k in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HF_HUB_TOKEN",
    ):
        token = os.environ.get(k)
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            break

    print("Upgrading torch + sglang-kernel ...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch==2.11.0",
            "torchvision",
            "torchaudio==2.11.0",
            "sglang-kernel==0.4.2",
        ],
        check=False,
    )

    # Allow capping the model's declared max_position_embeddings down to the
    # rope-derived effective context length (Qwen3-Next claims 262144 but
    # only ~40960 is rope-supported in practice).
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    print("\n=== nvidia-smi ===", flush=True)
    subprocess.run(["nvidia-smi"], check=False)

    cmd = [
        sys.executable,
        "-u",
        "scripts/accuracy_test_gsm8k.py",
        "--mode", mode,
        "--model", target_model,
        "--draft-model", draft_model,
        "--num-questions", str(num_questions),
        "--max-new-tokens", str(max_new_tokens),
        "--particles", str(particles),
        "--gamma", str(gamma),
        "--temperature", str(temperature),
        "--mem-fraction-static", str(mem_fraction_static),
        "--cuda-graph-max-bs", str(cuda_graph_max_bs),
        "--max-running-requests", str(max_running_requests),
        "--attention-backend", attention_backend,
        "--resample-threshold", str(resample_threshold),
        "--tp-size", str(tp_size),
        "--context-length", str(context_length),
        "--seed", "0",
    ]
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")

    print("Running:", " ".join(shlex.quote(p) for p in cmd), flush=True)
    rc = subprocess.run(cmd, cwd=SMCSD_DIR)
    print(f"smc-hybrid rc={rc.returncode}", flush=True)
    if rc.returncode != 0:
        sys.exit(rc.returncode)


# Medium pair: Qwen3.6-27B target + Qwen3.5-9B draft on 2×H200. Both hybrid
# (Mamba+attention), both full causal LMs (unlike the small Qwen3.5 MTP variants),
# and both use the same extended Qwen3.5/3.6 vocab. Exercises the
# hybrid-target + hybrid-draft path that the 80B+0.6B-dense pair didn't.
@app.function(
    gpu="H200:2",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hf-llama-token")],
    volumes={HF_HOME: hf_cache},
)
def run_smc_hybrid_medium(
    target_model: str = "Qwen/Qwen3.6-27B",
    draft_model: str = "Qwen/Qwen3.5-9B",
    num_questions: int = 3,
    max_new_tokens: int = 1024,
    particles: int = 2,
    gamma: int = 2,
    temperature: float = 0.7,
    tp_size: int = 2,
    mem_fraction_static: float = 0.25,
    cuda_graph_max_bs: int = 2,
    max_running_requests: int = 2,
    attention_backend: str = "fa3",
    resample_threshold: float = 0.5,
    context_length: int = 8192,
    disable_cuda_graph: bool = True,
    mode: str = "smc_engine",
) -> None:
    import os, shlex, subprocess, sys
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
              "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
        token = os.environ.get(k)
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            break

    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    print("\n=== nvidia-smi ===", flush=True)
    subprocess.run(["nvidia-smi"], check=False)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade",
         "torch==2.11.0", "torchvision", "torchaudio==2.11.0",
         "sglang-kernel==0.4.2"], check=False,
    )

    cmd = [
        sys.executable, "-u", "scripts/accuracy_test_gsm8k.py",
        "--mode", mode,
        "--model", target_model,
        "--draft-model", draft_model,
        "--num-questions", str(num_questions),
        "--max-new-tokens", str(max_new_tokens),
        "--particles", str(particles),
        "--gamma", str(gamma),
        "--temperature", str(temperature),
        "--mem-fraction-static", str(mem_fraction_static),
        "--cuda-graph-max-bs", str(cuda_graph_max_bs),
        "--max-running-requests", str(max_running_requests),
        "--attention-backend", attention_backend,
        "--resample-threshold", str(resample_threshold),
        "--tp-size", str(tp_size),
        "--context-length", str(context_length),
        "--seed", "0",
    ]
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    print("Running:", " ".join(shlex.quote(p) for p in cmd), flush=True)
    rc = subprocess.run(cmd, cwd=SMCSD_DIR)
    print(f"smc-hybrid-medium rc={rc.returncode}", flush=True)
    if rc.returncode != 0:
        sys.exit(rc.returncode)


# Small-model (Qwen3.5-2B/9B class) variant on a single H200. Both models in this
# pair are hybrid GDN/Mamba — exercises hybrid-target + hybrid-draft, which is a
# different code path than the 80B + 0.6B-dense pair tested above.
@app.function(
    gpu="H200:1",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hf-llama-token")],
    volumes={HF_HOME: hf_cache},
)
def run_smc_hybrid_small(
    target_model: str = "Qwen/Qwen3.5-9B",
    draft_model: str = "Qwen/Qwen3.5-2B",
    num_questions: int = 3,
    max_new_tokens: int = 1024,
    particles: int = 2,
    gamma: int = 2,
    temperature: float = 0.7,
    tp_size: int = 1,
    mem_fraction_static: float = 0.55,
    cuda_graph_max_bs: int = 2,
    max_running_requests: int = 2,
    attention_backend: str = "fa3",
    resample_threshold: float = 0.5,
    context_length: int = 8192,
    disable_cuda_graph: bool = True,
    mode: str = "smc_engine",
) -> None:
    import os, shlex, subprocess, sys
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
              "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
        token = os.environ.get(k)
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            break

    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    print("\n=== nvidia-smi ===", flush=True)
    subprocess.run(["nvidia-smi"], check=False)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade",
         "torch==2.11.0", "torchvision", "torchaudio==2.11.0",
         "sglang-kernel==0.4.2"], check=False,
    )

    cmd = [
        sys.executable, "-u", "scripts/accuracy_test_gsm8k.py",
        "--mode", mode,
        "--model", target_model,
        "--draft-model", draft_model,
        "--num-questions", str(num_questions),
        "--max-new-tokens", str(max_new_tokens),
        "--particles", str(particles),
        "--gamma", str(gamma),
        "--temperature", str(temperature),
        "--mem-fraction-static", str(mem_fraction_static),
        "--cuda-graph-max-bs", str(cuda_graph_max_bs),
        "--max-running-requests", str(max_running_requests),
        "--attention-backend", attention_backend,
        "--resample-threshold", str(resample_threshold),
        "--tp-size", str(tp_size),
        "--context-length", str(context_length),
        "--seed", "0",
    ]
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    print("Running:", " ".join(shlex.quote(p) for p in cmd), flush=True)
    rc = subprocess.run(cmd, cwd=SMCSD_DIR)
    print(f"smc-hybrid-small rc={rc.returncode}", flush=True)
    if rc.returncode != 0:
        sys.exit(rc.returncode)


@app.local_entrypoint()
def main(
    target_model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct",
    draft_model: str = "Qwen/Qwen3-0.6B",
    num_questions: int = 3,
    max_new_tokens: int = 1024,
    particles: int = 2,
    gamma: int = 2,
    temperature: float = 0.7,
    tp_size: int = 4,
    mem_fraction_static: float = 0.35,
    cuda_graph_max_bs: int = 2,
    max_running_requests: int = 2,
    attention_backend: str = "fa3",
    resample_threshold: float = 0.5,
    context_length: int = 8192,
    disable_cuda_graph: bool = False,
    mode: str = "smc_engine",
):
    run_smc_hybrid.remote(
        target_model=target_model,
        draft_model=draft_model,
        num_questions=num_questions,
        max_new_tokens=max_new_tokens,
        particles=particles,
        gamma=gamma,
        temperature=temperature,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,
        cuda_graph_max_bs=cuda_graph_max_bs,
        max_running_requests=max_running_requests,
        attention_backend=attention_backend,
        resample_threshold=resample_threshold,
        context_length=context_length,
        disable_cuda_graph=disable_cuda_graph,
        mode=mode,
    )


@app.local_entrypoint()
def medium(
    target_model: str = "Qwen/Qwen3.6-27B",
    draft_model: str = "Qwen/Qwen3.5-9B",
    num_questions: int = 3,
    max_new_tokens: int = 1024,
    particles: int = 2,
    gamma: int = 2,
    temperature: float = 0.7,
    tp_size: int = 2,
    mem_fraction_static: float = 0.25,
    cuda_graph_max_bs: int = 2,
    max_running_requests: int = 2,
    attention_backend: str = "fa3",
    resample_threshold: float = 0.5,
    context_length: int = 8192,
    disable_cuda_graph: bool = True,
    mode: str = "smc_engine",
):
    """Hybrid-target + hybrid-draft smoke test (Qwen3.6-27B + Qwen3.5-9B)."""
    run_smc_hybrid_medium.remote(
        target_model=target_model, draft_model=draft_model,
        num_questions=num_questions, max_new_tokens=max_new_tokens,
        particles=particles, gamma=gamma, temperature=temperature,
        tp_size=tp_size, mem_fraction_static=mem_fraction_static,
        cuda_graph_max_bs=cuda_graph_max_bs,
        max_running_requests=max_running_requests,
        attention_backend=attention_backend,
        resample_threshold=resample_threshold,
        context_length=context_length,
        disable_cuda_graph=disable_cuda_graph,
        mode=mode,
    )


@app.local_entrypoint()
def small(
    target_model: str = "Qwen/Qwen3.5-9B",
    draft_model: str = "Qwen/Qwen3.5-2B",
    num_questions: int = 3,
    max_new_tokens: int = 1024,
    particles: int = 2,
    gamma: int = 2,
    temperature: float = 0.7,
    tp_size: int = 1,
    mem_fraction_static: float = 0.55,
    cuda_graph_max_bs: int = 2,
    max_running_requests: int = 2,
    attention_backend: str = "fa3",
    resample_threshold: float = 0.5,
    context_length: int = 8192,
    disable_cuda_graph: bool = True,
    mode: str = "smc_engine",
):
    """Smoke test the hybrid-target + hybrid-draft path (Qwen3.5-9B + Qwen3.5-2B)."""
    run_smc_hybrid_small.remote(
        target_model=target_model, draft_model=draft_model,
        num_questions=num_questions, max_new_tokens=max_new_tokens,
        particles=particles, gamma=gamma, temperature=temperature,
        tp_size=tp_size, mem_fraction_static=mem_fraction_static,
        cuda_graph_max_bs=cuda_graph_max_bs,
        max_running_requests=max_running_requests,
        attention_backend=attention_backend,
        resample_threshold=resample_threshold,
        context_length=context_length,
        disable_cuda_graph=disable_cuda_graph,
        mode=mode,
    )
