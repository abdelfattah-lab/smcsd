#!/bin/bash
set -euo pipefail

# --- Sweep parameters ---
NUM_PROMPTS_LIST=(4 8 16)

# --- Fixed parameters ---
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL="meta-llama/Llama-3.2-1B-Instruct"
ATTENTION_BACKEND="triton"
MEM_FRACTION=0.60
INPUT_LEN=256
OUTPUT_LEN=512

# --- Output CSV ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${1:-results_baseline_${TIMESTAMP}.csv}"
echo "method,gamma,n,tps,b" > "$OUTFILE"
echo "Writing results to $OUTFILE"

# ---- sglang baseline (no speculation) ----
for b in "${NUM_PROMPTS_LIST[@]}"; do
    sleep 5  # brief pause between runs to let system stabilize
  echo "=== sglang  b=$b ==="

  LOGFILE=$(mktemp /tmp/bench_sglang_XXXXXX.log)
  if python -m sglang.bench_offline_throughput \
      --model-path "$MODEL" \
      --attention-backend "$ATTENTION_BACKEND" \
      --mem-fraction-static "$MEM_FRACTION" \
      --dataset-name random \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --num-prompts "$b" \
      2>&1 | tee "$LOGFILE"; then

    tps=$(grep "Output token throughput" "$LOGFILE" | awk '{print $NF}')
    if [[ -z "$tps" ]]; then
      echo "ERROR: no throughput in output (sglang b=$b)"
      echo "sglang,0,1,ERROR,$b" >> "$OUTFILE"
    else
      echo "sglang,0,1,$tps,$b" >> "$OUTFILE"
      echo "  -> tps=$tps"
    fi
  else
    echo "ERROR: benchmark failed (sglang b=$b), see $LOGFILE"
    echo "sglang,0,1,ERROR,$b" >> "$OUTFILE"
  fi
  rm -f "$LOGFILE"
done

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
      --attention-backend "$ATTENTION_BACKEND" \
      --mem-fraction-static "$MEM_FRACTION" \
      --dataset-name random \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --num-prompts "$b" \
      2>&1 | tee "$LOGFILE"; then

    tps=$(grep "Output token throughput" "$LOGFILE" | awk '{print $NF}')
    if [[ -z "$tps" ]]; then
      echo "ERROR: no throughput in output (standalone b=$b)"
      echo "standalone,0,1,ERROR,$b" >> "$OUTFILE"
    else
      echo "standalone,0,1,$tps,$b" >> "$OUTFILE"
      echo "  -> tps=$tps"
    fi
  else
    echo "ERROR: benchmark failed (standalone b=$b), see $LOGFILE"
    echo "standalone,0,1,ERROR,$b" >> "$OUTFILE"
  fi
  rm -f "$LOGFILE"
done

echo ""
echo "Done. Results in $OUTFILE"