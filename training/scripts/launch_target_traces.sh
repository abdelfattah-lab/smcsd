#!/usr/bin/env bash
# Launch data-parallel target-trace generation across GPUs.
# Each process is pinned to one GPU via CUDA_VISIBLE_DEVICES, handles its rank slice.
set -euo pipefail

OUT="${1:-training/data_cache/target_traces}"
GPUS="${GPUS:-1,2,3,4,5,6}"
BATCH="${BATCH:-64}"

cd "$(dirname "$0")/../.."  # smcsd/

IFS=',' read -ra GPU_LIST <<< "$GPUS"
W=${#GPU_LIST[@]}

mkdir -p "$OUT"
echo "launching W=$W processes across GPUs: ${GPU_LIST[*]} | out=$OUT | batch=$BATCH"

PIDS=()
for RANK in $(seq 0 $((W-1))); do
    GPU=${GPU_LIST[$RANK]}
    LOG="$OUT/rank${RANK}.log"
    CUDA_VISIBLE_DEVICES=$GPU python -m training.data.gen_target_traces \
        --target meta-llama/Llama-3.2-1B-Instruct \
        --output-dir "$OUT" \
        --rank "$RANK" --world-size "$W" \
        --batch-size "$BATCH" --sort-by-length \
        --device cuda:0 > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "  rank $RANK -> physical GPU $GPU (pid $!, log $LOG)"
done

for PID in "${PIDS[@]}"; do
    wait "$PID"
done
echo "all ranks done"
