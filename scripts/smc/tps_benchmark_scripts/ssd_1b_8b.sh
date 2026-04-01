#!/bin/bash
set -euo pipefail

# --- Sweep parameters ---
NUM_PROMPTS_LIST=(1 4 8 16)

# --- Fixed parameters ---
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL="meta-llama/Llama-3.2-1B-Instruct"
MEM_FRACTION=0.60
INPUT_LEN=256
OUTPUT_LEN=512
SSD_NUM_GPUS=2
SPECULATE_K=7
ASYNC_FAN_OUT=3
SSD_TEMP=0.7

# --- Output CSV ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${1:-ssd_1b_8b_${TIMESTAMP}.csv}"
echo "method,gamma,n,tps,b" > "$OUTFILE"
echo "Writing results to $OUTFILE"

for b in "${NUM_PROMPTS_LIST[@]}"; do
  max_rr=$b

  echo "=== b=$b  k=$SPECULATE_K  fan_out=$ASYNC_FAN_OUT  max_rr=$max_rr ==="

  LOGFILE=$(mktemp /tmp/bench_ssd_XXXXXX.log)
  if python -O -m sglang.bench_offline_throughput \
      --backend ssd \
      --model-path "$MODEL" \
      --ssd-draft-model "$DRAFT_MODEL" \
      --ssd-speculate \
      --ssd-draft-async \
      --ssd-speculate-k "$SPECULATE_K" \
      --ssd-async-fan-out "$ASYNC_FAN_OUT" \
      --ssd-num-gpus "$SSD_NUM_GPUS" \
      --mem-fraction-static "$MEM_FRACTION" \
      --ssd-max-num-seqs "$max_rr" \
      --dataset-name random \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --ssd-temperature "$SSD_TEMP" \
      --num-prompts "$b" \
      2>&1 | tee "$LOGFILE"; then

    tps=$(grep "Output token throughput" "$LOGFILE" | awk '{print $NF}')
    if [[ -z "$tps" ]]; then
      echo "ERROR: no throughput in output (b=$b)"
      echo "ssd,$SPECULATE_K,$ASYNC_FAN_OUT,ERROR,$b" >> "$OUTFILE"
    else
      echo "ssd,$SPECULATE_K,$ASYNC_FAN_OUT,$tps,$b" >> "$OUTFILE"
      echo "  -> tps=$tps"
    fi
  else
    echo "ERROR: benchmark failed (b=$b), see $LOGFILE"
    echo "ssd,$SPECULATE_K,$ASYNC_FAN_OUT,ERROR,$b" >> "$OUTFILE"
  fi
  rm -f "$LOGFILE"
done
echo ""
echo "Done. Results in $OUTFILE"
