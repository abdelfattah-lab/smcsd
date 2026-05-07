"""Modal H200 launcher for the standalone cross-tokenizer SMC script.

Runs scripts/cross_tokenizer_smc.py (HuggingFace Transformers, no sglang).
Defaults to Llama-3.1-8B-Instruct target + Qwen3-0.6B draft.

Usage:
    modal run scripts/run_cross_tokenizer_modal.py::run --num-questions 10
"""

from __future__ import annotations

import modal

APP_NAME = "smcsd-cross-tokenizer-smc"
SMCSD_DIR = "/root/smcsd"
HF_HOME = "/root/.cache/huggingface"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "ca-certificates", "curl")
    .run_commands(
        "pip install --upgrade pip wheel",
        "pip install torch==2.8.0 torchvision torchaudio==2.8.0",
        "pip install transformers==4.57.1 datasets numpy accelerate sentencepiece protobuf",
    )
    .add_local_file(
        "scripts/cross_tokenizer_smc.py",
        f"{SMCSD_DIR}/scripts/cross_tokenizer_smc.py",
        copy=True,
    )
    .env({"HF_HOME": HF_HOME, "PYTHONUNBUFFERED": "1"})
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("smcsd-hf-cache", create_if_missing=True)


@app.function(
    gpu="H200:1",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("hf-llama-token")],
    volumes={HF_HOME: hf_cache},
)
def run_xt_smc(
    target: str = "meta-llama/Llama-3.1-8B-Instruct",
    draft: str = "Qwen/Qwen3-0.6B",
    num_questions: int = 10,
    particles: int = 4,
    gamma: int = 4,
    temperature: float = 0.7,
    resample_threshold: float = 0.5,
    max_new_tokens: int = 512,
    seed: int = 0,
    use_chat_template: bool = False,
) -> None:
    import os, shlex, subprocess, sys

    for k in (
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN",
    ):
        token = os.environ.get(k)
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            break

    print("\n=== nvidia-smi ===", flush=True)
    subprocess.run(["nvidia-smi"], check=False)

    cmd = [
        sys.executable, "-u", "scripts/cross_tokenizer_smc.py",
        "--target", target,
        "--draft", draft,
        "--num-questions", str(num_questions),
        "--particles", str(particles),
        "--gamma", str(gamma),
        "--temperature", str(temperature),
        "--resample-threshold", str(resample_threshold),
        "--max-new-tokens", str(max_new_tokens),
        "--seed", str(seed),
    ]
    if use_chat_template:
        cmd.append("--use-chat-template")

    print("Running:", " ".join(shlex.quote(p) for p in cmd), flush=True)
    rc = subprocess.run(cmd, cwd=SMCSD_DIR)
    print(f"xt-smc rc={rc.returncode}", flush=True)
    if rc.returncode != 0:
        sys.exit(rc.returncode)


@app.local_entrypoint()
def run(
    target: str = "meta-llama/Llama-3.1-8B-Instruct",
    draft: str = "Qwen/Qwen3-0.6B",
    num_questions: int = 10,
    particles: int = 4,
    gamma: int = 4,
    temperature: float = 0.7,
    resample_threshold: float = 0.5,
    max_new_tokens: int = 512,
    seed: int = 0,
    use_chat_template: bool = False,
):
    run_xt_smc.remote(
        target=target,
        draft=draft,
        num_questions=num_questions,
        particles=particles,
        gamma=gamma,
        temperature=temperature,
        resample_threshold=resample_threshold,
        max_new_tokens=max_new_tokens,
        seed=seed,
        use_chat_template=use_chat_template,
    )
