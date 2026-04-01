# bench/

Benchmarking scripts for SSD, SGLang, and vLLM.



We benchmark SGLang/vLLM via their server APIs rather than offline/in-process for two reasons:
1. Server mode exposes `/metrics` endpoints that offline mode doesn't.
2. We found server mode more performant in practice.

There is a small confound from HTTP overhead and prefill time being included in the
wall-clock measurement, but since this is a decode-heavy workload (512 output tokens,
each forward pass takes much longer than an HTTP round-trip), we find the difference to empirically be negligible.

## Setup

Use separate envs for all three stacks (`ssd`, `sglang`, `vllm`) because required FlashInfer versions by each conflict. 

```bash
# SSD (from repo root)
# uses pyproject.toml + uv.lock
uv sync

# SGLang 0.5.9
conda create -n sglang python=3.11
conda activate sglang
pip install "sglang[all]==0.5.9"

# vLLM 0.16.0
conda create -n vllm016 python=3.11
conda activate vllm016
pip install "vllm==0.16.0"
```

SGLang/vLLM envs also need: `transformers`, `aiohttp`, `wandb` (optional).

## Model paths

SGLang/vLLM baselines read from `bench_paths.py`. Edit it or set env vars:
```bash
export SSD_HF_CACHE=/path/to/huggingface/hub
export BENCH_LLAMA_70B=/path/to/Llama-3.3-70B-Instruct
export BENCH_LLAMA_1B=/path/to/Llama-3.2-1B-Instruct
# etc.
```

SSD reads from `ssd/paths.py` (separate, env vars documented there).

## Datasets

4 datasets (humaneval, alpaca, gsm8k, ultrafeedback), 128 prompts each with `--all`.
Set `SSD_DATASET_DIR` or edit `ssd/paths.py`. Generate with `scripts/get_data_from_hf.py`.

## Usage

### SSD (native, no server)
`bench.py` and `bench_helpers.py` are SSD-only (import from `ssd.*`).
```bash
python -O bench.py --llama --size 70 --async --spec --k 7 --f 3 --b 1 \
    --temp 0 --numseqs 128 --output_len 512 --all --gpus 5
```

### SGLang
```bash
conda activate sglang
# Speculative decoding (default):
python bench/run_sglang_bench.py --llama
# Autoregressive baseline:
python bench/run_sglang_bench.py --llama --mode ar
# With wandb logging:
python bench/run_sglang_bench.py --llama --wandb --group mygroup --name myrun
```

### vLLM
```bash
conda activate vllm016
# Speculative decoding (default):
python bench/run_vllm_bench.py --llama
# Autoregressive baseline:
python bench/run_vllm_bench.py --llama --mode ar
```

## Troubleshooting

If you see Triton errors like `OSError: [Errno 116] Stale file handle`, set cache dirs to local disk (not network filesystems):

```bash
export TRITON_CACHE_DIR=/scratch/$USER/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/$USER/torchinductor_cache
```

If a run crashes or you Ctrl-C, stale worker processes can linger and hold resources.
Before rerunning, kill them and verify GPUs are free:

```bash
pkill -9 -f "bench.py|vllm.entrypoints|sglang.launch_server|VLLM::|sglang::"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```
