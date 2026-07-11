#!/bin/bash
# Lossy tree SD (SGLang EAGLE3 + relaxed acceptance) vs SMC-SD (n=66, gamma=16)
# ShareGPT, bs=1 (--max-running-requests 1), B200s. 8B on GPU0; 70B TP=4 on GPUs 0-3.
# triton attention backend for both arms (fa3 sgl-kernel build lacks sm100 support here).
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
mkdir -p "$OUTDIR"
CSV="$OUTDIR/lossy_vs_smc.csv"
echo "run,model,method,tps,accept_len" > "$CSV"

NUM_PROMPTS=50
TEMP=0.7

COMMON_DS=(--dataset-name sharegpt --num-prompts "$NUM_PROMPTS" --max-running-requests 1
           --mem-fraction-static 0.60 --attention-backend triton
           --extra-request-body "{\"temperature\": $TEMP}")

# run_one <opt_flag ""|-O> <name> <model_label> <method_label> <args...>
run_one () {
  local optflag="$1"; shift
  local name="$1"; shift
  local model="$1"; shift
  local method="$1"; shift
  local log="$OUTDIR/${name}.log"
  echo "=== RUN $name ==="
  if timeout 3600 python $optflag "$HARNESS" "$@" "${COMMON_DS[@]}" > "$log" 2>&1; then
    local tps accept
    tps=$(grep "Output token throughput" "$log" | awk '{print $NF}' | tail -1)
    accept=$(grep -oE "accept len: [0-9.]+" "$log" | awk '{s+=$3; c++} END {if (c>0) printf "%.2f", s/c; else print "n/a"}')
    echo "$name,$model,$method,${tps:-ERROR},${accept:-n/a}" >> "$CSV"
    echo "=== DONE $name tps=$tps accept=$accept ==="
  else
    echo "$name,$model,$method,ERROR,n/a" >> "$CSV"
    echo "=== FAILED $name (see $log, last lines:) ==="
    tail -5 "$log"
  fi
  sleep 10
}

eagle3_8b () {
  local name="$1" ts="$2" ta="$3"
  CUDA_VISIBLE_DEVICES=0 run_one "" "$name" 8B "eagle3_s${ts}_a${ta}" \
    --backend engine \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32 \
    --speculative-accept-threshold-single "$ts" --speculative-accept-threshold-acc "$ta" \
    --dtype bfloat16 \
    --tp-size 1
}

# ---------------- 8B: GPU 0 ----------------
eagle3_8b eagle3_8b_lossless 1.0 1.0
eagle3_8b eagle3_8b_lossy075 0.75 0.75
eagle3_8b eagle3_8b_lossy050 0.5 0.5

CUDA_VISIBLE_DEVICES=0 run_one "-O" smc_8b_n66g16 8B smc_n66_g16 \
  --backend smc_engine \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
  --smc-n-particles 66 --smc-gamma 16 \
  --smc-draft-temperature "$TEMP" --smc-target-temperature "$TEMP" \
  --cuda-graph-max-bs 66 --disable-piecewise-cuda-graph \
  --tp-size 1

# ---------------- 70B: TP=4, GPUs 0-3 ----------------
eagle3_70b () {
  local name="$1" ts="$2" ta="$3"
  CUDA_VISIBLE_DEVICES=0,1,2,3 run_one "" "$name" 70B "eagle3_s${ts}_a${ta}" \
    --backend engine \
    --model-path meta-llama/Llama-3.3-70B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.3-Instruct-70B \
    --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32 \
    --speculative-accept-threshold-single "$ts" --speculative-accept-threshold-acc "$ta" \
    --dtype bfloat16 \
    --tp-size 4
}

eagle3_70b eagle3_70b_lossless 1.0 1.0
eagle3_70b eagle3_70b_lossy075 0.75 0.75
eagle3_70b eagle3_70b_lossy050 0.5 0.5

CUDA_VISIBLE_DEVICES=0,1,2,3 run_one "-O" smc_70b_n66g16 70B smc_n66_g16 \
  --backend smc_engine \
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
  --smc-n-particles 66 --smc-gamma 16 \
  --smc-draft-temperature "$TEMP" --smc-target-temperature "$TEMP" \
  --cuda-graph-max-bs 66 --disable-piecewise-cuda-graph \
  --tp-size 4

echo "=== ALL RUNS COMPLETE ==="
cat "$CSV"
