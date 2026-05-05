"""Qwen3.6-27B baseline MMLU (no speculative decoding) on the upstream branch.

Same env as run_qwen_baseline.py but invokes scripts/accuracy_test_mmlu.py.

Usage:
    modal run modal_apps/run_qwen_baseline_mmlu.py
    modal run modal_apps/run_qwen_baseline_mmlu.py --num-questions 500 --subject high_school_mathematics
"""

from __future__ import annotations

import modal

APP_NAME = "smcsd-qwen-baseline-mmlu"
SMCSD_DIR = "/root/smcsd"
HF_HOME = "/root/.cache/huggingface"
SMCSD_REPO = "https://github.com/abdelfattah-lab/smcsd.git"
SMCSD_BRANCH = "upstream"

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
    .add_local_file(
        "scripts/accuracy_test_mmlu.py",
        "/tmp/smcsd_local_accuracy_test_mmlu.py",
        copy=True,
    )
    .env({"HF_HOME": HF_HOME, "PYTHONUNBUFFERED": "1"})
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("smcsd-hf-cache", create_if_missing=True)


@app.function(
    gpu="H200",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hf-llama-token")],
    volumes={HF_HOME: hf_cache},
)
def run_qwen_baseline_mmlu(
    target_model: str = "Qwen/Qwen3.6-27B",
    num_questions: int = 100,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    subject: str = "",
    mem_fraction_static: float = 0.92,
    max_running_requests: int = 16,
    cuda_graph_max_bs: int = 16,
    ignore_eos: bool = False,
) -> None:
    import os
    import shlex
    import shutil
    import subprocess
    import sys

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
        [sys.executable, "-m", "pip", "install", "--upgrade",
         "torch==2.11.0", "torchvision", "torchaudio==2.11.0",
         "sglang-kernel==0.4.2"],
        check=False,
    )

    # Drop the local-workspace MMLU script into the cloned repo.
    shutil.copyfile(
        "/tmp/smcsd_local_accuracy_test_mmlu.py",
        f"{SMCSD_DIR}/scripts/accuracy_test_mmlu.py",
    )
    print("Installed scripts/accuracy_test_mmlu.py from local workspace copy", flush=True)

    cmd = [
        sys.executable,
        "scripts/accuracy_test_mmlu.py",
        "--mode", "baseline",
        "--model", target_model,
        "--num-questions", str(num_questions),
        "--max-new-tokens", str(max_new_tokens),
        "--batch-size", str(batch_size),
        "--mem-fraction-static", str(mem_fraction_static),
        "--max-running-requests", str(max_running_requests),
        "--cuda-graph-max-bs", str(cuda_graph_max_bs),
        "--seed", "0",
    ]
    if subject:
        cmd += ["--subject", subject]
    if ignore_eos:
        cmd.append("--ignore-eos")
    print("Running:", " ".join(shlex.quote(p) for p in cmd), flush=True)
    rc = subprocess.run(cmd, cwd=SMCSD_DIR)
    print(f"qwen baseline mmlu rc={rc.returncode}", flush=True)
    if rc.returncode != 0:
        sys.exit(rc.returncode)


@app.local_entrypoint()
def main(
    target_model: str = "Qwen/Qwen3.6-27B",
    num_questions: int = 100,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    subject: str = "",
    mem_fraction_static: float = 0.92,
    max_running_requests: int = 16,
    cuda_graph_max_bs: int = 16,
    ignore_eos: bool = False,
):
    run_qwen_baseline_mmlu.remote(
        target_model=target_model,
        num_questions=num_questions,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        subject=subject,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        ignore_eos=ignore_eos,
    )
