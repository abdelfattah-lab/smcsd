#!/bin/bash
# Peak target-model KV pool occupancy, single request (P=256, L=512), 8B, bs=1.
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
CSV="$OUTDIR/kv_overhead.csv"
echo "run,peak_kv_tokens" > "$CSV"

COMMON=(--model-path meta-llama/Llama-3.1-8B-Instruct --tp-size 1
        --dataset-name random --random-input-len 256 --random-output-len 512
        --random-range-ratio 1.0 --num-prompts 1 --max-running-requests 1
        --mem-fraction-static 0.60 --attention-backend triton --skip-warmup
        --extra-request-body '{"temperature": 0.7}')

run_kv () {
  local name="$1"; shift
  local optflag="$1"; shift
  local log="$OUTDIR/kv_${name}.log"
  echo "=== $name ==="
  timeout 900 python $optflag "$HARNESS" "$@" "${COMMON[@]}" > "$log" 2>&1
  local peak
  peak=$(grep -oE "#token: [0-9]+" "$log" | awk '{if ($2>m) m=$2} END {print m}')
  echo "$name,${peak:-ERR}" >> "$CSV"
  echo "$name peak=$peak"
}

run_kv chain "" \
  --backend engine \
  --speculative-algorithm STANDALONE \
  --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
  --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
  --dtype bfloat16

run_kv tree050 "" \
  --backend engine \
  --speculative-algorithm STANDALONE \
  --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
  --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32 \
  --speculative-accept-threshold-single 0.5 --speculative-accept-threshold-acc 0.5 \
  --dtype bfloat16

SMC_LOG_KV_USAGE=1 run_kv smc_n6g16 "-O" \
  --backend smc_engine \
  --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
  --smc-n-particles 6 --smc-gamma 16 \
  --smc-draft-temperature 0.7 --smc-target-temperature 0.7 \
  --cuda-graph-max-bs 6 --disable-piecewise-cuda-graph

echo DONE; cat "$CSV"
