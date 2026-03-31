#!/bin/bash
set -euo pipefail

# --- Sweep parameters ---
NUM_PROMPTS_LIST=(4 8 16)
# (gamma, n) pairs
GAMMA_N_PAIRS=(
  "8 8"
  "10 8"
  "10 6"
  "12 8"
  "16 8"
  "8 6"
  "8 4"
  "10 6"
)

# --- Fixed parameters ---
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL="meta-llama/Llama-3.2-1B-Instruct"
DRAFT_TEMP=0.7
TARGET_TEMP=0.7
ATTENTION_BACKEND="triton"
MEM_FRACTION=0.60
INPUT_LEN=256
OUTPUT_LEN=512

# --- Output CSV ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${1:-results_smc_${TIMESTAMP}.csv}"
echo "method,gamma,n,tps,b" > "$OUTFILE"
echo "Writing results to $OUTFILE"

for b in "${NUM_PROMPTS_LIST[@]}"; do
  for pair in "${GAMMA_N_PAIRS[@]}"; do
    sleep 5  # brief pause between runs to let system stabilize
    read -r gamma n <<< "$pair"

    # max-running-requests = num_prompts * n_particles (each prompt fans out)
    max_rr=$((b * n))
    # cuda-graph-bs = max_running_requests (cover the full batch)
    cuda_bs=$((max_rr))  # add some slack to ensure we cover the full batch (in case of some stragglers) and get good graph reuse. We don't want this to be too large though to avoid OOMs.

    echo "=== b=$b  gamma=$gamma  n=$n  max_rr=$max_rr  cuda_bs=$cuda_bs ==="

    LOGFILE=$(mktemp /tmp/bench_smc_XXXXXX.log)
    if python -m sglang.bench_offline_throughput \
        --model-path "$MODEL" \
        --speculative-algorithm SMC \
        --speculative-draft-model-path "$DRAFT_MODEL" \
        --smc-n-particles "$n" \
        --smc-gamma "$gamma" \
        --smc-draft-temperature "$DRAFT_TEMP" \
        --smc-target-temperature "$TARGET_TEMP" \
        --smc-pingpong-overlap \
        --attention-backend "$ATTENTION_BACKEND" \
        --mem-fraction-static "$MEM_FRACTION" \
        --max-running-requests "$max_rr" \
        --cuda-graph-max-bs "$cuda_bs" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$b" \
        2>&1 | tee "$LOGFILE"; then

      tps=$(grep "Output token throughput" "$LOGFILE" | awk '{print $NF}')
      if [[ -z "$tps" ]]; then
        echo "ERROR: no throughput in output (b=$b gamma=$gamma n=$n)"
        echo "smc,$gamma,$n,ERROR,$b" >> "$OUTFILE"
      else
        echo "smc,$gamma,$n,$tps,$b" >> "$OUTFILE"
        echo "  -> tps=$tps"
      fi
    else
      echo "ERROR: benchmark failed (b=$b gamma=$gamma n=$n), see $LOGFILE"
      echo "smc,$gamma,$n,ERROR,$b" >> "$OUTFILE"
    fi
    rm -f "$LOGFILE"
  done
done
echo ""
echo "Done. Results in $OUTFILE"
