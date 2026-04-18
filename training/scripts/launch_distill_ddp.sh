#!/usr/bin/env bash
# DDP stage-1 distillation across GPUs via torchrun.
# Usage:
#   GPUS=0,1,2,3 bash training/scripts/launch_distill_ddp.sh training/checkpoints/stage1_run1
set -euo pipefail

GPUS="${GPUS:-0,1,2,3}"
OUT="${1:-training/checkpoints/stage1_run1}"
LOG_EVERY="${LOG_EVERY:-100}"
MAX_STEPS="${MAX_STEPS:-}"
FIND_UNUSED="${FIND_UNUSED:-0}"
DEBUG="${DEBUG:-0}"

cd "$(dirname "$0")/../.."

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N=${#GPU_LIST[@]}
MASTER_PORT="${MASTER_PORT:-29500}"

EXTRA=()
if [ "$FIND_UNUSED" = "1" ]; then EXTRA+=("--find-unused-parameters"); fi
if [ -n "$MAX_STEPS" ]; then EXTRA+=(--max-steps "$MAX_STEPS"); fi

ENV_PREFIX="CUDA_VISIBLE_DEVICES=$GPUS"
if [ "$DEBUG" = "1" ]; then
    ENV_PREFIX="$ENV_PREFIX TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=WARN"
fi

echo "launching torchrun: nproc=$N  GPUS=$GPUS  out=$OUT  find_unused=$FIND_UNUSED  debug=$DEBUG"
env CUDA_VISIBLE_DEVICES="$GPUS" \
    $( [ "$DEBUG" = "1" ] && echo "TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=WARN" ) \
    torchrun --nproc_per_node="$N" --standalone --master_port="$MASTER_PORT" \
        -m training.training.distill \
        --config training/configs/mvp.yaml \
        --output-dir "$OUT" \
        --log-every "$LOG_EVERY" \
        --seed 0 \
        "${EXTRA[@]}"
