#!/bin/bash
set -euo pipefail

# SMC v2 throughput sweep — Llama 3.1-8B target + Llama 3.2-1B draft, triton.
# Uses bench_offline_throughput.py with --backend smc_engine, which routes
# through SMCEngine (the engine-level --speculative-algorithm SMC path was
# removed with v1 retirement).

# --- Sweep parameters ---
NUM_PROMPTS_LIST=(1 4 8 16)
GAMMA_N_PAIRS=(
  "8 8"
  "10 8"
  "10 6"
  "12 8"
  "16 8"
  "8 6"
  "8 4"
  "12 6"
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
TP=1
METHOD_LABEL="smc_triton"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS="${SCRIPT_DIR}/bench_offline_throughput.py"

# --- Output CSV ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${1:-smc_1b_8b_triton_${TIMESTAMP}.csv}"
echo "method,gamma,n,tps,b" > "$OUTFILE"
echo "Writing results to $OUTFILE"

for b in "${NUM_PROMPTS_LIST[@]}"; do
  for pair in "${GAMMA_N_PAIRS[@]}"; do
    sleep 5
    read -r gamma n <<< "$pair"

    # --max-running-requests = concurrent user groups (SMCEngine expands
    # to G*(N+1) internally for the req pool).  --cuda-graph-max-bs is
    # the peak decode batch = groups * particles.
    max_rr=$b
    cuda_bs=$((b * n))

    echo "=== b=$b  gamma=$gamma  n=$n  max_rr=$max_rr  cuda_bs=$cuda_bs ==="

    LOGFILE=$(mktemp /tmp/bench_smc_XXXXXX.log)
    if python -O "$HARNESS" \
        --backend smc_engine \
        --model-path "$MODEL" \
        --speculative-draft-model-path "$DRAFT_MODEL" \
        --smc-n-particles "$n" \
        --smc-gamma "$gamma" \
        --smc-draft-temperature "$DRAFT_TEMP" \
        --smc-target-temperature "$TARGET_TEMP" \
        --attention-backend "$ATTENTION_BACKEND" \
        --mem-fraction-static "$MEM_FRACTION" \
        --max-running-requests "$max_rr" \
        --cuda-graph-max-bs "$cuda_bs" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --extra-request-body "{\"temperature\": $TARGET_TEMP}" \
        --num-prompts "$b" \
        --tp-size "$TP" \
        2>&1 | tee "$LOGFILE"; then

      tps=$(grep "Output token throughput" "$LOGFILE" | awk '{print $NF}')
      if [[ -z "$tps" ]]; then
        echo "ERROR: no throughput in output (b=$b gamma=$gamma n=$n)"
        echo "${METHOD_LABEL},$gamma,$n,ERROR,$b" >> "$OUTFILE"
      else
        echo "${METHOD_LABEL},$gamma,$n,$tps,$b" >> "$OUTFILE"
        echo "  -> tps=$tps"
      fi
    else
      echo "ERROR: benchmark failed (b=$b gamma=$gamma n=$n), see $LOGFILE"
      echo "${METHOD_LABEL},$gamma,$n,ERROR,$b" >> "$OUTFILE"
    fi
    rm -f "$LOGFILE"
  done
done
echo ""
echo "Done. Results in $OUTFILE"
