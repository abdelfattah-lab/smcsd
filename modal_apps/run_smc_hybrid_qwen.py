"""SMC speculative decoding smoke test for the Qwen3.5 hybrid-pair on Modal H200.

Runs scripts/accuracy_test_gsm8k.py with target = Qwen3.5-9B and draft =
Qwen3.5-2B (both hybrid GDN/Mamba, dense draft mode) on a single H200.
Mirrors the local invocation:

    python scripts/accuracy_test_gsm8k.py \
      --mode smc_engine \
      --model Qwen/Qwen3.5-9B --draft-model Qwen/Qwen3.5-2B \
      --smc-draft-mode dense --disable-thinking \
      --particles 12 --gamma 8 --temperature 0.7 --seed 123 \
      --attention-backend fa3 --smc-fast-resample \
      --mem-fraction-static 0.9 --max-running-requests 1 \
      --max-total-tokens 8192 --cuda-graph-max-bs 16 --num-questions 100

The image clones the hybrid-models branch and overlays the local working
tree on top so iteration doesn't require pushing each edit.
"""

from __future__ import annotations

import modal

APP_NAME = "smcsd-smc-hybrid-qwen"
SMCSD_DIR = "/root/smcsd"
HF_HOME = "/root/.cache/huggingface"
SMCSD_REPO = "https://github.com/abdelfattah-lab/smcsd.git"
# The image clones main from origin for image bootstrap (pip install -e .).
# The local working tree is overlaid via add_local_dir / add_local_file so
# runtime smcsd/ and scripts/ match whatever is on disk — no push needed to
# iterate. The local hybrid-models port adds eagle3/dflash/smc_draft_mode
# / hybrid Qwen multi-step backend on top of main's slot-major core.
SMCSD_BRANCH = "main"

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
        # Bump the sglang submodule to eagle3-smc-upstream-sync-2 tip
        # (3f7c85da1) — main pins an older commit; our hybrid features
        # depend on this one (configure_scheduler_process, smc_draft_mode
        # ServerArgs fields, etc.).
        f"cd {SMCSD_DIR}/3rdparty/sglang && git fetch origin eagle3-smc-upstream-sync-2 && git checkout 3f7c85da12542394dae12b33ac436a91e5f19173",
        f"cd {SMCSD_DIR} && pip install --upgrade pip wheel",
        f'export PATH=$HOME/.cargo/bin:$PATH && cd {SMCSD_DIR} && pip install -e 3rdparty/sglang/python',
        f"cd {SMCSD_DIR} && pip install -e .",
        "pip install datasets transformers numpy",
    )
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*"
    )
    .add_local_dir("smcsd", f"{SMCSD_DIR}/smcsd", copy=True)
    .add_local_file(
        "scripts/accuracy_test_gsm8k.py",
        f"{SMCSD_DIR}/scripts/accuracy_test_gsm8k.py",
        copy=True,
    )
    .env({"HF_HOME": HF_HOME, "PYTHONUNBUFFERED": "1"})
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("smcsd-hf-cache", create_if_missing=True)


def _run_eval(
    *,
    target_model: str,
    draft_model: str,
    num_questions: int,
    max_new_tokens: int,
    particles: int,
    gamma: int,
    temperature: float,
    seed: int,
    smc_draft_mode: str,
    disable_thinking: bool,
    smc_fast_resample: bool,
    attention_backend: str,
    mem_fraction_static: float,
    max_running_requests: int,
    max_total_tokens: int,
    cuda_graph_max_bs: int,
    mode: str,
    tp_size: int = 1,
    disable_cuda_graph: bool = False,
    batch_size: int = 1,
    enable_timing: bool = False,
    timing_every: int = 50,
    label: str = "smc-hybrid",
) -> None:
    import os
    import shlex
    import subprocess
    import sys

    for k in (
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN",
    ):
        token = os.environ.get(k)
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            break

    # Hybrid Qwen3.5/3.6 declares max_position_embeddings beyond what rope actually
    # supports — let SGLang cap context_length to the rope-derived effective length.
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    if enable_timing:
        os.environ["SMCSD_TIMING"] = "1"
        os.environ["SMCSD_TIMING_EVERY"] = str(timing_every)

    print("Upgrading torch + sglang-kernel ...", flush=True)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--upgrade",
            "torch==2.11.0", "torchvision", "torchaudio==2.11.0",
            "sglang-kernel==0.4.2",
        ],
        check=False,
    )

    print("\n=== nvidia-smi ===", flush=True)
    subprocess.run(["nvidia-smi"], check=False)

    cmd = [
        sys.executable, "-u", "scripts/accuracy_test_gsm8k.py",
        "--mode", mode,
        "--model", target_model,
        "--draft-model", draft_model,
        "--smc-draft-mode", smc_draft_mode,
        "--particles", str(particles),
        "--gamma", str(gamma),
        "--temperature", str(temperature),
        "--seed", str(seed),
        "--attention-backend", attention_backend,
        "--mem-fraction-static", str(mem_fraction_static),
        "--max-running-requests", str(max_running_requests),
        "--max-total-tokens", str(max_total_tokens),
        "--cuda-graph-max-bs", str(cuda_graph_max_bs),
        "--num-questions", str(num_questions),
        "--max-new-tokens", str(max_new_tokens),
        "--tp-size", str(tp_size),
        "--batch-size", str(batch_size),
    ]
    if disable_thinking:
        cmd.append("--disable-thinking")
    if smc_fast_resample:
        cmd.append("--smc-fast-resample")
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")

    print("Running:", " ".join(shlex.quote(p) for p in cmd), flush=True)
    rc = subprocess.run(cmd, cwd=SMCSD_DIR)
    print(f"{label} rc={rc.returncode}", flush=True)
    if rc.returncode != 0:
        sys.exit(rc.returncode)


@app.function(
    gpu="H200:1",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hf-llama-token")],
    volumes={HF_HOME: hf_cache},
)
def run_smc_hybrid(
    target_model: str = "Qwen/Qwen3.5-9B",
    draft_model: str = "Qwen/Qwen3.5-2B",
    num_questions: int = 100,
    max_new_tokens: int = 512,
    particles: int = 12,
    gamma: int = 8,
    temperature: float = 0.7,
    seed: int = 123,
    smc_draft_mode: str = "dense",
    disable_thinking: bool = True,
    smc_fast_resample: bool = True,
    attention_backend: str = "fa3",
    mem_fraction_static: float = 0.9,
    max_running_requests: int = 1,
    max_total_tokens: int = 8192,
    cuda_graph_max_bs: int = 16,
    mode: str = "smc_engine",
) -> None:
    _run_eval(
        target_model=target_model, draft_model=draft_model,
        num_questions=num_questions, max_new_tokens=max_new_tokens,
        particles=particles, gamma=gamma, temperature=temperature,
        seed=seed, smc_draft_mode=smc_draft_mode,
        disable_thinking=disable_thinking, smc_fast_resample=smc_fast_resample,
        attention_backend=attention_backend,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        max_total_tokens=max_total_tokens,
        cuda_graph_max_bs=cuda_graph_max_bs, mode=mode,
        tp_size=1, disable_cuda_graph=False, label="smc-hybrid",
    )


# 27B target + 2B draft on 2×H200. Both hybrid (Mamba+attention) with
# different state shapes — needs --disable-cuda-graph (per per-worker
# MambaPool constraint) and tp_size=2 to fit the 27B target.
@app.function(
    gpu="H200:2",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hf-llama-token")],
    volumes={HF_HOME: hf_cache},
)
def run_smc_hybrid_medium(
    target_model: str = "Qwen/Qwen3.6-27B",
    draft_model: str = "Qwen/Qwen3.5-2B",
    num_questions: int = 100,
    max_new_tokens: int = 512,
    particles: int = 12,
    gamma: int = 8,
    temperature: float = 0.7,
    seed: int = 123,
    smc_draft_mode: str = "dense",
    disable_thinking: bool = True,
    smc_fast_resample: bool = True,
    attention_backend: str = "fa3",
    mem_fraction_static: float = 0.4,
    max_running_requests: int = 1,
    max_total_tokens: int = 8192,
    cuda_graph_max_bs: int = 16,
    mode: str = "smc_engine",
    tp_size: int = 2,
    disable_cuda_graph: bool = True,
    batch_size: int = 1,
    enable_timing: bool = False,
    timing_every: int = 50,
) -> None:
    _run_eval(
        target_model=target_model, draft_model=draft_model,
        num_questions=num_questions, max_new_tokens=max_new_tokens,
        particles=particles, gamma=gamma, temperature=temperature,
        seed=seed, smc_draft_mode=smc_draft_mode,
        disable_thinking=disable_thinking, smc_fast_resample=smc_fast_resample,
        attention_backend=attention_backend,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        max_total_tokens=max_total_tokens,
        cuda_graph_max_bs=cuda_graph_max_bs, mode=mode,
        tp_size=tp_size, disable_cuda_graph=disable_cuda_graph,
        batch_size=batch_size,
        enable_timing=enable_timing, timing_every=timing_every,
        label="smc-hybrid-medium",
    )


@app.local_entrypoint()
def main(
    target_model: str = "Qwen/Qwen3.5-9B",
    draft_model: str = "Qwen/Qwen3.5-2B",
    num_questions: int = 100,
    max_new_tokens: int = 512,
    particles: int = 12,
    gamma: int = 8,
    temperature: float = 0.7,
    seed: int = 123,
    smc_draft_mode: str = "dense",
    disable_thinking: bool = True,
    smc_fast_resample: bool = True,
    attention_backend: str = "fa3",
    mem_fraction_static: float = 0.9,
    max_running_requests: int = 1,
    max_total_tokens: int = 8192,
    cuda_graph_max_bs: int = 16,
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
        seed=seed,
        smc_draft_mode=smc_draft_mode,
        disable_thinking=disable_thinking,
        smc_fast_resample=smc_fast_resample,
        attention_backend=attention_backend,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        max_total_tokens=max_total_tokens,
        cuda_graph_max_bs=cuda_graph_max_bs,
        mode=mode,
    )


@app.local_entrypoint()
def medium(
    target_model: str = "Qwen/Qwen3.6-27B",
    draft_model: str = "Qwen/Qwen3.5-2B",
    num_questions: int = 100,
    max_new_tokens: int = 512,
    particles: int = 12,
    gamma: int = 8,
    temperature: float = 0.7,
    seed: int = 123,
    smc_draft_mode: str = "dense",
    disable_thinking: bool = True,
    smc_fast_resample: bool = True,
    attention_backend: str = "fa3",
    mem_fraction_static: float = 0.4,
    max_running_requests: int = 1,
    max_total_tokens: int = 8192,
    cuda_graph_max_bs: int = 16,
    mode: str = "smc_engine",
    tp_size: int = 2,
    disable_cuda_graph: bool = True,
    batch_size: int = 1,
    enable_timing: bool = False,
    timing_every: int = 50,
):
    run_smc_hybrid_medium.remote(
        target_model=target_model,
        draft_model=draft_model,
        num_questions=num_questions,
        max_new_tokens=max_new_tokens,
        particles=particles,
        gamma=gamma,
        temperature=temperature,
        seed=seed,
        smc_draft_mode=smc_draft_mode,
        disable_thinking=disable_thinking,
        smc_fast_resample=smc_fast_resample,
        attention_backend=attention_backend,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        max_total_tokens=max_total_tokens,
        cuda_graph_max_bs=cuda_graph_max_bs,
        mode=mode,
        tp_size=tp_size,
        disable_cuda_graph=disable_cuda_graph,
        batch_size=batch_size,
        enable_timing=enable_timing,
        timing_every=timing_every,
    )
