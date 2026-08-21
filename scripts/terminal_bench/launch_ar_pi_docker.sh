#!/usr/bin/env bash
set -euo pipefail

SMCSD_ROOT="${SMCSD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SMCSD_IMAGE="${SMCSD_IMAGE:-lmsysorg/sglang:v0.5.16}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$TARGET_MODEL}"
GPU_DEVICE="${GPU_DEVICE:-0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-auto}"
REASONING_PARSER="${REASONING_PARSER:-auto}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-true}"
DISABLE_FLASHINFER_AUTOTUNE="${DISABLE_FLASHINFER_AUTOTUNE:-false}"
RANDOM_SEED="${RANDOM_SEED:-0}"
ENABLE_METRICS="${ENABLE_METRICS:-true}"

extra=()
if [[ "${SKIP_SERVER_WARMUP}" == "true" ]]; then
  extra+=(--skip-server-warmup)
fi
if [[ "${DISABLE_FLASHINFER_AUTOTUNE}" == "true" ]]; then
  extra+=(--disable-flashinfer-autotune)
fi
if [[ "${ENABLE_METRICS}" == "true" ]]; then
  extra+=(--enable-metrics)
fi

# This is a true target-only baseline: stock SGLang autoregressive decoding,
# with no draft model and no SM-CSD scheduler. Agent-facing settings match the
# SM-CSD launcher; target-only AR receives the remaining single-runner KV budget.
exec docker run --rm \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  --network=host \
  -v "${SMCSD_ROOT}:/workspace/smcsd" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -w /workspace/smcsd \
  -e PYTHONPATH=/workspace/smcsd/3rdparty/sglang/python \
  -e HF_HOME=/root/.cache/huggingface \
  -e FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-ar-pi \
  "${SMCSD_IMAGE}" \
  python -m sglang.launch_server \
    --model-path "${TARGET_MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tp-size "${TP_SIZE}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --random-seed "${RANDOM_SEED}" \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --reasoning-parser "${REASONING_PARSER}" \
    --default-chat-template-kwargs "${DEFAULT_CHAT_TEMPLATE_KWARGS}" \
    --trust-remote-code \
    "${extra[@]}"
