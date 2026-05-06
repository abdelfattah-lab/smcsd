"""Qwen3.6-27B NVFP4A16 baseline GSM8K / MMLU on H200.

Loads the compressed-tensors NVFP4A16 checkpoint produced by
quantize_qwen_nvfp4.py from the `smcsd-quantized-models` Modal volume.

Notes:
  - Uses the smcsd-bundled sglang because mainline sglang doesn't ship the
    `Qwen3_5ForConditionalGeneration` architecture; the smcsd fork does.
  - NVFP4A16 = FP4 weights, BF16 activations — works on Hopper. Pure NVFP4
    (W4A4) raises NotImplementedError on H200 in the bundled sglang.

Usage:
    modal run modal_apps/run_qwen_baseline_nvfp4.py
    modal run modal_apps/run_qwen_baseline_nvfp4.py --benchmark mmlu --num-questions 100
"""

from __future__ import annotations

import modal

APP_NAME = "smcsd-qwen-baseline-nvfp4"
SMCSD_DIR = "/root/smcsd"
HF_HOME = "/root/.cache/huggingface"
QUANT_DIR = "/quantized"
SMCSD_REPO = "https://github.com/abdelfattah-lab/smcsd.git"
SMCSD_BRANCH = "upstream"

# Files we want to source from the original HF repo when llmcompressor's
# saved checkpoint is missing them. NEVER copy weight shards — those would
# clobber/sit alongside the quantized model.safetensors.
AUX_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
)

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
        "pip install datasets transformers numpy compressed-tensors",
    )
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*"
    )
    .add_local_file(
        "scripts/accuracy_test_gsm8k.py",
        "/tmp/smcsd_local_accuracy_test_gsm8k.py",
        copy=True,
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
quant_vol = modal.Volume.from_name("smcsd-quantized-models", create_if_missing=True)


@app.function(
    gpu="H200",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hf-llama-token")],
    volumes={HF_HOME: hf_cache, QUANT_DIR: quant_vol},
)
def run(
    quant_name: str = "Qwen3.6-27B-W4A16",
    src_model: str = "Qwen/Qwen3.6-27B",
    benchmark: str = "gsm8k",
    num_questions: int = 100,
    max_new_tokens: int = 2048,
    batch_size: int = 8,
    subject: str = "",
    mem_fraction_static: float = 0.92,
    max_running_requests: int = 16,
    cuda_graph_max_bs: int = 16,
    ignore_eos: bool = False,
) -> None:
    import json
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

    quant_path = os.path.join(QUANT_DIR, quant_name)
    print(f"[run] quantized dir: {quant_path}", flush=True)
    if not os.path.isdir(quant_path):
        print(f"[run] ERROR: {quant_path} not found in volume; "
              f"run quantize_qwen_nvfp4.py first", flush=True)
        sys.exit(2)

    # Cleanup: a previous version of this script accidentally copied BF16
    # shards from the HF snapshot dir into the quantized dir. Remove any
    # shards / sharded-index files; keep the single NVFP4A16 model.safetensors.
    purged = []
    for f in sorted(os.listdir(quant_path)):
        if f.startswith("model-") and f.endswith(".safetensors"):
            os.remove(os.path.join(quant_path, f))
            purged.append(f)
        elif f == "model.safetensors.index.json":
            os.remove(os.path.join(quant_path, f))
            purged.append(f)
    if purged:
        print(f"[run] purged stray BF16 shards: {len(purged)} files "
              f"(first: {purged[0]})", flush=True)
        quant_vol.commit()

    print(f"[run] contents now: {sorted(os.listdir(quant_path))}", flush=True)

    # Patch config: replace llmcompressor's stripped text-only config with the
    # full original (multimodal) one, then graft the quantization_config back in.
    from huggingface_hub import snapshot_download
    cfg_path = os.path.join(quant_path, "config.json")
    with open(cfg_path) as f:
        qcfg = json.load(f)
    needs_full_orig = (
        "Qwen3_5ForCausalLM" in qcfg.get("architectures", [])
        or "vision_start_token_id" not in qcfg
        or not all(
            os.path.exists(os.path.join(quant_path, f)) for f in AUX_FILES[:3]
        )
    )
    if needs_full_orig:
        # Pull config + each named aux file. Don't iterate the snapshot dir
        # blindly — that picks up cached weight shards too.
        snapshot_download(src_model, allow_patterns=["config.json", *AUX_FILES])
        # After snapshot_download, the files are in the standard HF cache layout.
        # Resolve the snapshot dir via huggingface_hub and read from there.
        from huggingface_hub import HfApi
        # Simpler: re-download to a path we control via local_dir.
        local_dir = "/tmp/qwen36_orig_meta"
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            src_model,
            allow_patterns=["config.json", *AUX_FILES],
            local_dir=local_dir,
        )
        with open(os.path.join(local_dir, "config.json")) as f:
            src_cfg = json.load(f)
        for k in ("quantization_config", "compression_config"):
            if k in qcfg:
                src_cfg[k] = qcfg[k]
        with open(cfg_path, "w") as f:
            json.dump(src_cfg, f, indent=2)
        for fname in AUX_FILES:
            src = os.path.join(local_dir, fname)
            dst = os.path.join(quant_path, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"[run]   copied {fname}", flush=True)
        print(f"[run] config.json arch={src_cfg.get('architectures')}", flush=True)
        quant_vol.commit()

    # Use the local-workspace eval scripts.
    shutil.copyfile(
        "/tmp/smcsd_local_accuracy_test_gsm8k.py",
        f"{SMCSD_DIR}/scripts/accuracy_test_gsm8k.py",
    )
    shutil.copyfile(
        "/tmp/smcsd_local_accuracy_test_mmlu.py",
        f"{SMCSD_DIR}/scripts/accuracy_test_mmlu.py",
    )

    if benchmark == "gsm8k":
        script = "scripts/accuracy_test_gsm8k.py"
    elif benchmark == "mmlu":
        script = "scripts/accuracy_test_mmlu.py"
    else:
        print(f"[run] unknown benchmark {benchmark!r}", flush=True)
        sys.exit(2)

    cmd = [
        sys.executable,
        script,
        "--mode", "baseline",
        "--model", quant_path,
        "--num-questions", str(num_questions),
        "--max-new-tokens", str(max_new_tokens),
        "--batch-size", str(batch_size),
        "--mem-fraction-static", str(mem_fraction_static),
        "--max-running-requests", str(max_running_requests),
        "--cuda-graph-max-bs", str(cuda_graph_max_bs),
        "--seed", "0",
    ]
    if benchmark == "mmlu" and subject:
        cmd += ["--subject", subject]
    if ignore_eos:
        cmd.append("--ignore-eos")

    print("Running:", " ".join(shlex.quote(p) for p in cmd), flush=True)
    rc = subprocess.run(cmd, cwd=SMCSD_DIR)
    print(f"qwen baseline nvfp4 ({benchmark}) rc={rc.returncode}", flush=True)
    if rc.returncode != 0:
        sys.exit(rc.returncode)


@app.local_entrypoint()
def main(
    quant_name: str = "Qwen3.6-27B-W4A16",
    src_model: str = "Qwen/Qwen3.6-27B",
    benchmark: str = "gsm8k",
    num_questions: int = 100,
    max_new_tokens: int = 2048,
    batch_size: int = 8,
    subject: str = "",
    mem_fraction_static: float = 0.92,
    max_running_requests: int = 16,
    cuda_graph_max_bs: int = 16,
    ignore_eos: bool = False,
):
    run.remote(
        quant_name=quant_name,
        src_model=src_model,
        benchmark=benchmark,
        num_questions=num_questions,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        subject=subject,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        ignore_eos=ignore_eos,
    )
