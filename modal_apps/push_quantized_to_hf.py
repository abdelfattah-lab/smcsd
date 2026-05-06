"""Push a quantized checkpoint from the smcsd-quantized-models Modal volume
to the Hugging Face Hub.

Defaults to private=True. Pass --private False if you want it public.

Usage:
    modal run modal_apps/push_quantized_to_hf.py --repo-id you/Qwen3.6-27B-W4A16
    modal run modal_apps/push_quantized_to_hf.py \
        --repo-id you/Qwen3.6-27B-W4A16 --private False
"""

from __future__ import annotations

import modal

APP_NAME = "smcsd-push-quant-to-hf"
QUANT_DIR = "/quantized"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.27", "hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)

app = modal.App(APP_NAME, image=image)
quant_vol = modal.Volume.from_name("smcsd-quantized-models", create_if_missing=False)


@app.function(
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hf-llama-token")],
    volumes={QUANT_DIR: quant_vol},
)
def push(
    repo_id: str,
    quant_name: str = "Qwen3.6-27B-W4A16",
    private: bool = True,
    commit_message: str = "Upload W4A16 INT4 quantized checkpoint",
) -> None:
    import os
    from huggingface_hub import HfApi, create_repo, upload_folder

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

    src = os.path.join(QUANT_DIR, quant_name)
    if not os.path.isdir(src):
        raise SystemExit(f"{src} not found in volume")
    print(f"[push] source: {src}", flush=True)
    print(f"[push] target: {repo_id} (private={private})", flush=True)
    for f in sorted(os.listdir(src)):
        size = os.path.getsize(os.path.join(src, f))
        unit = "GB" if size > 1e9 else "MB"
        scale = 1e9 if size > 1e9 else 1e6
        print(f"  {f}  {size / scale:.2f} {unit}", flush=True)

    api = HfApi()
    create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    info = upload_folder(
        repo_id=repo_id,
        folder_path=src,
        commit_message=commit_message,
        repo_type="model",
    )
    print(f"[push] done: {info}", flush=True)
    print(f"[push] view at https://huggingface.co/{repo_id}", flush=True)


@app.local_entrypoint()
def main(
    repo_id: str,
    quant_name: str = "Qwen3.6-27B-W4A16",
    private: bool = True,
    commit_message: str = "Upload W4A16 INT4 quantized checkpoint",
):
    push.remote(
        repo_id=repo_id,
        quant_name=quant_name,
        private=private,
        commit_message=commit_message,
    )
