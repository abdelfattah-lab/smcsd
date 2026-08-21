#!/usr/bin/env bash
set -euo pipefail

SMCSD_ROOT="${SMCSD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SMCSD_IMAGE="${SMCSD_IMAGE:-lmsysorg/sglang:v0.5.16}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-9B}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3.5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$TARGET_MODEL}"
GPU_DEVICE="${GPU_DEVICE:-0}"
PORT="${PORT:-30000}"
PARTICLES="${PARTICLES:-4}"
GAMMA="${GAMMA:-4}"
RESAMPLE_THRESHOLD="${RESAMPLE_THRESHOLD:-0.5}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.4}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-auto}"
REASONING_PARSER="${REASONING_PARSER:-auto}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-false}"
DISABLE_FLASHINFER_AUTOTUNE="${DISABLE_FLASHINFER_AUTOTUNE:-false}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-true}"
RANDOM_SEED="${RANDOM_SEED:-0}"
ENABLE_METRICS="${ENABLE_METRICS:-true}"
SMC_GRAPH_STATS="${SMC_GRAPH_STATS:-0}"
SMC_GRAPH_STATS_INTERVAL="${SMC_GRAPH_STATS_INTERVAL:-100}"
SMC_SMC_STATS="${SMC_SMC_STATS:-0}"

extra=()
if [[ "${DISABLE_CUDA_GRAPH}" == "true" ]]; then
  extra+=(--disable-cuda-graph)
fi
if [[ "${DISABLE_FLASHINFER_AUTOTUNE}" == "true" ]]; then
  extra+=(--disable-flashinfer-autotune)
fi
if [[ "${SKIP_SERVER_WARMUP}" == "true" ]]; then
  extra+=(--skip-server-warmup)
fi
if [[ "${ENABLE_METRICS}" == "true" ]]; then
  extra+=(--enable-metrics)
fi

exec docker run --rm \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  --network=host \
  -v "${SMCSD_ROOT}:/workspace/smcsd" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -w /workspace/smcsd \
  -e PYTHONPATH=/workspace/smcsd:/workspace/smcsd/3rdparty/sglang/python \
  -e HF_HOME=/root/.cache/huggingface \
  -e FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-smcsd-pi \
  -e "SMC_GRAPH_STATS=${SMC_GRAPH_STATS}" \
  -e "SMC_GRAPH_STATS_INTERVAL=${SMC_GRAPH_STATS_INTERVAL}" \
  -e "SMC_SMC_STATS=${SMC_SMC_STATS}" \
  "${SMCSD_IMAGE}" \
  python -m smcsd.http_server \
    --model "${TARGET_MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --particles "${PARTICLES}" \
    --gamma "${GAMMA}" \
    --resample-threshold "${RESAMPLE_THRESHOLD}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --random-seed "${RANDOM_SEED}" \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --reasoning-parser "${REASONING_PARSER}" \
    --default-chat-template-kwargs "${DEFAULT_CHAT_TEMPLATE_KWARGS}" \
    --trust-remote-code \
    "${extra[@]}"
