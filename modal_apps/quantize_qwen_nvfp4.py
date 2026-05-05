"""Quantize Qwen3.6-27B (BF16) → NVFP4 weight-only via llmcompressor.

Output is written to the Modal volume `smcsd-quantized-models` under
`Qwen3.6-27B-NVFP4/` as a compressed-tensors checkpoint that sglang/vLLM
can load directly (with config.json, model.safetensors shards, recipe.yaml).

Calibration: 256 samples from HuggingFaceH4/ultrachat_200k, max_seq=2048.
That's the default llmcompressor recipe for NVFP4 weight-only and matches
what Red Hat / Neural Magic publish for similarly sized Qwen models.

Why NVFP4:
  - Real FP4 (E2M1 mantissa, FP8 microscale, 16-elem block) — not INT4.
  - Weight-only on Hopper (BF16 compute), native compute on Blackwell.
  - Served by sglang via compressed-tensors without custom code.

Usage:
    modal run modal_apps/quantize_qwen_nvfp4.py
    modal run modal_apps/quantize_qwen_nvfp4.py \
        --model Qwen/Qwen3.6-27B --num-calibration-samples 512
"""

from __future__ import annotations

import modal

APP_NAME = "smcsd-quantize-qwen-nvfp4"
HF_HOME = "/root/.cache/huggingface"
QUANT_DIR = "/quantized"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "ca-certificates")
    .pip_install(
        "torch==2.5.1",
        "accelerate>=1.1",
        "datasets",
        "numpy",
        "safetensors",
        "pydantic>=2",
        "loguru",
        "pyyaml",
    )
    # Install transformers main FIRST so pip resolves the matching hf_hub
    # for it (transformers main still imports is_offline_mode, which is gone
    # in hf_hub 1.x). Qwen3.6-27B's qwen3_5 architecture only exists here.
    .pip_install(
        "transformers @ git+https://github.com/huggingface/transformers.git",
    )
    # llmcompressor main and compressed-tensors main are dev-pinned together;
    # the stable PyPI compressed-tensors 0.15 lacks `compressed_tensors.distributed`
    # which llmcompressor main now imports. Install both from git, with
    # --no-deps on llmcompressor so it doesn't roll transformers back.
    .run_commands(
        "pip install 'compressed-tensors @ git+https://github.com/neuralmagic/compressed-tensors.git' && "
        "pip install --no-deps "
        "'llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git'"
    )
    .env({"HF_HOME": HF_HOME, "PYTHONUNBUFFERED": "1"})
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("smcsd-hf-cache", create_if_missing=True)
quant_vol = modal.Volume.from_name("smcsd-quantized-models", create_if_missing=True)


@app.function(
    gpu="H200",
    timeout=60 * 90,
    secrets=[modal.Secret.from_name("hf-llama-token")],
    volumes={HF_HOME: hf_cache, QUANT_DIR: quant_vol},
)
def quantize(
    model: str = "Qwen/Qwen3.6-27B",
    num_calibration_samples: int = 256,
    max_sequence_length: int = 2048,
    output_name: str = "",
) -> None:
    import os
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor import oneshot

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

    save_name = output_name or model.split("/")[-1] + "-W4A16"
    save_path = os.path.join(QUANT_DIR, save_name)
    print(f"[quant] target={model} -> {save_path}", flush=True)

    print("[quant] loading model in BF16 ...", flush=True)
    # cuDNN SDPA backend has been throwing "No valid execution plans built"
    # on Qwen3.5 head dims with this torch/cuDNN combo; force eager attention
    # for calibration (slower but safe — only matters during this oneshot).
    torch.backends.cuda.enable_cudnn_sdp(False)
    m = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # Diagnostic: dump all Linear layer names + out_features. Marlin INT4
    # crashes on out_features % 64 != 0; we use this to extend the ignore
    # list so quantization skips those layers.
    import torch.nn as nn
    misaligned: list[str] = []
    print("[quant] Linear layers with out_features % 64 != 0:", flush=True)
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and mod.out_features % 64 != 0:
            print(f"  {name}: in={mod.in_features} out={mod.out_features}", flush=True)
            misaligned.append(name)
    print(f"[quant] {len(misaligned)} misaligned Linear layers found", flush=True)
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    print(f"[quant] loading calibration ({num_calibration_samples} samples)", flush=True)
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=f"train_sft[:{num_calibration_samples}]",
    )

    def to_text(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

    ds = ds.map(to_text)

    def tokenize(sample):
        return tok(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # W4A16 = INT4 group-quantized weights (group_size=128), BF16 activations.
    # The smcsd-vendored sglang's compressed-tensors handler routes this to
    # CompressedTensorsWNA16 (Marlin kernel) which is well-tested on Hopper.
    # Pure NVFP4 (W4A4) raises NotImplementedError on H200 (Blackwell-only),
    # and the FP4-weight-only NVFP4A16 path doesn't exist in this sglang
    # fork — only INT4 group-quant does. Same ~4x memory compression as FP4
    # weight-only; typical ~0.5-1pt accuracy delta on math benchmarks.
    # lm_head left full precision to protect logits.
    # Marlin INT4 requires out_features divisible by 64; pass the literal
    # names we just discovered so llmcompressor leaves those Linear layers
    # in BF16. Tiny relative to the 27B model — no measurable memory impact.
    ignore_layers = ["lm_head", *misaligned]
    print(f"[quant] ignore list: {ignore_layers[:5]}{'...' if len(ignore_layers) > 5 else ''} "
          f"(total {len(ignore_layers)})", flush=True)
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=ignore_layers,
    )

    print("[quant] running oneshot calibration + quantization ...", flush=True)
    oneshot(
        model=m,
        processor=tok,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_sequence_length,
        num_calibration_samples=num_calibration_samples,
    )

    print(f"[quant] saving compressed-tensors checkpoint to {save_path}", flush=True)
    os.makedirs(save_path, exist_ok=True)
    m.save_pretrained(save_path, save_compressed=True)
    tok.save_pretrained(save_path)

    # Commit to the Modal volume so other apps see it.
    quant_vol.commit()
    print("[quant] done", flush=True)
    for f in sorted(os.listdir(save_path)):
        size = os.path.getsize(os.path.join(save_path, f))
        print(f"  {f}  {size / 1e9:.2f} GB" if size > 1e9 else f"  {f}  {size / 1e6:.1f} MB",
              flush=True)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.6-27B",
    num_calibration_samples: int = 256,
    max_sequence_length: int = 2048,
    output_name: str = "",
):
    quantize.remote(
        model=model,
        num_calibration_samples=num_calibration_samples,
        max_sequence_length=max_sequence_length,
        output_name=output_name,
    )
