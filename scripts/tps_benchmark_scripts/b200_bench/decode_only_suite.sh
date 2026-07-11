#!/bin/bash
# Full rerun of the Fig-6/Table-2 matrix with prefill calibration (decode-only tps).
# ShareGPT bs=1, 50 prompts, temp 0.7, triton backend. 8B GPU0; 70B TP=4 GPUs 0-3.
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
CSV="$OUTDIR/decode_only_suite.csv"
echo "run,model,method,e2e_tps,decode_tps,prefill_s,accept_len" > "$CSV"

NUM_PROMPTS=50
TEMP=0.7

COMMON_DS=(--dataset-name sharegpt --num-prompts "$NUM_PROMPTS" --max-running-requests 1
           --mem-fraction-static 0.60 --attention-backend triton --measure-prefill
           --extra-request-body "{\"temperature\": $TEMP}")

run_one () {
  local optflag="$1"; shift
  local name="$1"; shift
  local model="$1"; shift
  local method="$1"; shift
  local log="$OUTDIR/${name}_dec.log"
  echo "=== RUN $name ==="
  if timeout 3600 python $optflag "$HARNESS" "$@" "${COMMON_DS[@]}" > "$log" 2>&1; then
    local e2e dec pre accept
    e2e=$(grep "Output token throughput" "$log" | awk '{print $NF}' | tail -1)
    dec=$(grep "Decode-only output throughput" "$log" | awk '{print $NF}' | tail -1)
    pre=$(grep "Prefill calibration duration" "$log" | awk '{print $NF}' | tail -1)
    accept=$(grep -oE "accept len: [0-9.]+" "$log" | awk '{s+=$3; c++} END {if (c>0) printf "%.2f", s/c; else print "n/a"}')
    echo "$name,$model,$method,${e2e:-ERR},${dec:-ERR},${pre:-ERR},${accept:-n/a}" >> "$CSV"
    echo "=== DONE $name e2e=$e2e decode=$dec ==="
  else
    echo "$name,$model,$method,ERROR,ERROR,ERROR,n/a" >> "$CSV"
    echo "=== FAILED $name ==="; tail -5 "$log"
  fi
  sleep 10
}

sa_tree () { # name label model tp gpus ts ta
  CUDA_VISIBLE_DEVICES=$5 run_one "" "$1" "$2" "sa1b_tree_s${6}_a${7}" \
    --backend engine --model-path "$3" \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
    --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32 \
    --speculative-accept-threshold-single "$6" --speculative-accept-threshold-acc "$7" \
    --dtype bfloat16 --tp-size "$4"
}
sa_chain () { # name label model tp gpus
  CUDA_VISIBLE_DEVICES=$5 run_one "" "$1" "$2" "sa1b_chain_lossless" \
    --backend engine --model-path "$3" \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --dtype bfloat16 --tp-size "$4"
}
eagle3 () { # name label model head tp gpus ts ta
  CUDA_VISIBLE_DEVICES=$6 run_one "" "$1" "$2" "eagle3_s${7}_a${8}" \
    --backend engine --model-path "$3" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$4" \
    --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32 \
    --speculative-accept-threshold-single "$7" --speculative-accept-threshold-acc "$8" \
    --dtype bfloat16 --tp-size "$5"
}
smc () { # name label model tp gpus
  CUDA_VISIBLE_DEVICES=$5 run_one "-O" "$1" "$2" "smc_n6_g16" \
    --backend smc_engine --model-path "$3" \
    --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
    --smc-n-particles 6 --smc-gamma 16 \
    --smc-draft-temperature "$TEMP" --smc-target-temperature "$TEMP" \
    --cuda-graph-max-bs 6 --disable-piecewise-cuda-graph \
    --tp-size "$4"
}

M8=meta-llama/Llama-3.1-8B-Instruct
M70=meta-llama/Llama-3.3-70B-Instruct
H8=lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B
H70=lmsys/sglang-EAGLE3-LLaMA3.3-Instruct-70B

# ---- 8B ----
sa_chain d_sa1b_8b_chain 8B "$M8" 1 0
sa_tree  d_sa1b_8b_tree_lossless 8B "$M8" 1 0 1.0 1.0
sa_tree  d_sa1b_8b_tree_lossy075 8B "$M8" 1 0 0.75 0.75
sa_tree  d_sa1b_8b_tree_lossy050 8B "$M8" 1 0 0.5 0.5
eagle3   d_eagle3_8b_lossy050 8B "$M8" "$H8" 1 0 0.5 0.5
smc      d_smc_8b_n6g16 8B "$M8" 1 0

# ---- 70B ----
sa_chain d_sa1b_70b_chain 70B "$M70" 4 0,1,2,3
sa_tree  d_sa1b_70b_tree_lossless 70B "$M70" 4 0,1,2,3 1.0 1.0
sa_tree  d_sa1b_70b_tree_lossy075 70B "$M70" 4 0,1,2,3 0.75 0.75
sa_tree  d_sa1b_70b_tree_lossy050 70B "$M70" 4 0,1,2,3 0.5 0.5
eagle3   d_eagle3_70b_lossy050 70B "$M70" "$H70" 4 0,1,2,3 0.5 0.5
smc      d_smc_70b_n6g16 70B "$M70" 4 0,1,2,3

# ---- Cactus (vLLM) ----
VPY=/data/models/cactus-venv/bin/python
cd /data/models/cactus
for cfg in "rs 8B $M8 1 0" "cactus 8B $M8 1 0" "rs 70B $M70 4 0,1,2,3" "cactus 70B $M70 4 0,1,2,3"; do
  read -r method label model tp gpus <<< "$cfg"
  name="d_${method}_${label}"
  log="$OUTDIR/${name}_dec.log"
  echo "=== RUN $name ==="
  if CUDA_VISIBLE_DEVICES=$gpus timeout 3600 $VPY bench_sharegpt_bs1.py \
      --model "$model" --draft meta-llama/Llama-3.2-1B-Instruct \
      --method "$method" --delta 1.0 --num-spec-tokens 10 --tp "$tp" > "$log" 2>&1; then
    line=$(grep "^RESULT" "$log" | tail -1)
    e2e=$(echo "$line" | grep -oE "tps=[0-9.]+" | head -1 | cut -d= -f2)
    dec=$(echo "$line" | grep -oE "decode_tps=[0-9.]+" | cut -d= -f2)
    pre=$(echo "$line" | grep -oE "prefill_s=[0-9.]+" | cut -d= -f2)
    ar=$(echo "$line" | grep -oE "tracker_ar=[0-9.]+" | cut -d= -f2)
    echo "$name,$label,vllm_${method},${e2e:-ERR},${dec:-ERR},${pre:-ERR},${ar:-n/a}" >> "$CSV"
    echo "=== DONE $name e2e=$e2e decode=$dec ==="
  else
    echo "$name,$label,vllm_${method},ERROR,ERROR,ERROR,n/a" >> "$CSV"
    echo "=== FAILED $name ==="; tail -5 "$log"
  fi
  sleep 10
done

echo "=== ALL RUNS COMPLETE ==="
cat "$CSV"
