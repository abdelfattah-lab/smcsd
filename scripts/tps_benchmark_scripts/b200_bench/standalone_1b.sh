#!/bin/bash
# STANDALONE spec decoding with Llama-3.2-1B drafter: tree (topk=10) threshold sweep + chain baseline.
# ShareGPT bs=1, temp 0.7, triton. 8B on GPU0, 70B TP=4 GPUs 0-3.
set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test
export HF_HUB_CACHE=/data/models/hf-hub
export TRITON_CACHE_DIR=/data/models/tmpcache/triton
export TORCHINDUCTOR_CACHE_DIR=/data/models/tmpcache/inductor
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

cd ~/smcsd
HARNESS=scripts/tps_benchmark_scripts/bench_offline_throughput.py
OUTDIR=/tmp/claude-1008/-home-yahya/f2a90585-1413-4217-b5d9-173e5a14947b/scratchpad/results
CSV="$OUTDIR/standalone_1b.csv"
echo "run,model,method,tps,accept_len" > "$CSV"

NUM_PROMPTS=50
TEMP=0.7

COMMON_DS=(--dataset-name sharegpt --num-prompts "$NUM_PROMPTS" --max-running-requests 1
           --mem-fraction-static 0.60 --attention-backend triton
           --extra-request-body "{\"temperature\": $TEMP}")

run_one () {
  local name="$1"; shift
  local model="$1"; shift
  local method="$1"; shift
  local log="$OUTDIR/${name}.log"
  echo "=== RUN $name ==="
  if timeout 3600 python "$HARNESS" "$@" "${COMMON_DS[@]}" > "$log" 2>&1; then
    local tps accept
    tps=$(grep "Output token throughput" "$log" | awk '{print $NF}' | tail -1)
    accept=$(grep -oE "accept len: [0-9.]+" "$log" | awk '{s+=$3; c++} END {if (c>0) printf "%.2f", s/c; else print "n/a"}')
    echo "$name,$model,$method,${tps:-ERROR},${accept:-n/a}" >> "$CSV"
    echo "=== DONE $name tps=$tps accept=$accept ==="
  else
    echo "$name,$model,$method,ERROR,n/a" >> "$CSV"
    echo "=== FAILED $name (see $log)"
    tail -5 "$log"
  fi
  sleep 10
}

# tree: topk=10 steps=6 draft=32 (same shape as the EAGLE3 runs)
sa_tree () {
  local name="$1" model_label="$2" model="$3" tp="$4" gpus="$5" ts="$6" ta="$7"
  CUDA_VISIBLE_DEVICES=$gpus run_one "$name" "$model_label" "sa1b_tree_s${ts}_a${ta}" \
    --backend engine \
    --model-path "$model" \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
    --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32 \
    --speculative-accept-threshold-single "$ts" --speculative-accept-threshold-acc "$ta" \
    --dtype bfloat16 \
    --tp-size "$tp"
}

# chain: repo baseline shape (topk=1 steps=4 draft=5), lossless
sa_chain () {
  local name="$1" model_label="$2" model="$3" tp="$4" gpus="$5"
  CUDA_VISIBLE_DEVICES=$gpus run_one "$name" "$model_label" "sa1b_chain_lossless" \
    --backend engine \
    --model-path "$model" \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --dtype bfloat16 \
    --tp-size "$tp"
}

M8=meta-llama/Llama-3.1-8B-Instruct
M70=meta-llama/Llama-3.3-70B-Instruct

sa_tree  sa1b_8b_tree_lossless 8B "$M8" 1 0 1.0 1.0
sa_tree  sa1b_8b_tree_lossy075 8B "$M8" 1 0 0.75 0.75
sa_tree  sa1b_8b_tree_lossy050 8B "$M8" 1 0 0.5 0.5
sa_chain sa1b_8b_chain 8B "$M8" 1 0

sa_tree  sa1b_70b_tree_lossless 70B "$M70" 4 0,1,2,3 1.0 1.0
sa_tree  sa1b_70b_tree_lossy075 70B "$M70" 4 0,1,2,3 0.75 0.75
sa_tree  sa1b_70b_tree_lossy050 70B "$M70" 4 0,1,2,3 0.5 0.5
sa_chain sa1b_70b_chain 70B "$M70" 4 0,1,2,3

echo "=== ALL RUNS COMPLETE ==="
cat "$CSV"
