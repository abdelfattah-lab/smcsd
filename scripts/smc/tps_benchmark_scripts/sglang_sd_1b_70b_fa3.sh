#!/bin/bash
set -euo pipefail

# --- Sweep parameters ---
NUM_PROMPTS_LIST=(1 4 8 16)

# --- Fixed parameters ---
MODEL="meta-llama/Llama-3.1-70B-Instruct"
DRAFT_MODEL="meta-llama/Llama-3.2-1B-Instruct"
ATTENTION_BACKEND="fa3"
MEM_FRACTION=0.60
INPUT_LEN=256
OUTPUT_LEN=512

# --- Output CSV ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${1:-sglang_1b_70b_fa3_v1_${TIMESTAMP}.csv}"
echo "method,gamma,n,tps,b" > "$OUTFILE"
echo "Writing results to $OUTFILE"

# ---- standalone speculative decoding ----
for b in "${NUM_PROMPTS_LIST[@]}"; do
  echo "=== standalone  b=$b ==="

  LOGFILE=$(mktemp /tmp/bench_standalone_XXXXXX.log)
  if python -m sglang.bench_offline_throughput \
      --model-path "$MODEL" \
      --speculative-algorithm STANDALONE \
      --speculative-draft-model-path "$DRAFT_MODEL" \
      --speculative-eagle-topk 1 \
      --speculative-num-steps 4 \
      --speculative-num-draft-tokens 5 \
      --mem-fraction-static "$MEM_FRACTION" \
      --attention-backend "$ATTENTION_BACKEND" \
      --dataset-name random \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --extra-request-body '{"temperature": 0.7}' \
      --num-prompts "$b" \
      --tp 4 \
      2>&1 | tee "$LOGFILE"; then

    tps=$(grep "Output token throughput" "$LOGFILE" | awk '{print $NF}')
    if [[ -z "$tps" ]]; then
      echo "ERROR: no throughput in output (standalone b=$b)"
      echo "standalone_fa3_v1,0,1,ERROR,$b" >> "$OUTFILE"
    else
      echo "standalone_fa3_v1,0,1,$tps,$b" >> "$OUTFILE"
      echo "  -> tps=$tps"
    fi
  else
    echo "ERROR: benchmark failed (standalone b=$b), see $LOGFILE"
    echo "standalone_fa3_v1,0,1,ERROR,$b" >> "$OUTFILE"
  fi
  rm -f "$LOGFILE"
done

echo ""
echo "Done. Results in $OUTFILE"
