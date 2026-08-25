#!/usr/bin/env bash
set -euo pipefail

# Reproducible five-method primary holdout. Completed stages are reused.
run_dir="work_dirs/semantic_olympiadbench_holdout_v1"
config="configs/semantic/semantic_smc_olympiadbench_holdout_v1.json"
generator="Qwen/Qwen3.5-9B"
verifier_a="Qwen/Qwen3.8-27B"
verifier_b="Qwen/Qwen3-32B"
cuda_toolkit="${CUDA_HOME:-/usr/local/cuda}"

generation_devices="${SMC_GENERATION_DEVICES:-0}"
verifier_a_devices="${SMC_VERIFIER_A_DEVICES:-1,2}"
verifier_b_devices="${SMC_VERIFIER_B_DEVICES:-3,4}"
smc_devices="${SMC_ONLINE_DEVICES:-5,6,7}"

mkdir -p "$run_dir" results/olympiadbench

if [[ ! -s "$run_dir/independent_n8_trajectories.jsonl" || ! -s "$run_dir/independent_n8_summary.json" ]]; then
  CUDA_HOME="$cuda_toolkit" CUDA_VISIBLE_DEVICES="$generation_devices" \
    .venv/bin/python scripts/accuracy_test_olympiadbench.py \
      --model "$generator" \
      --num-questions 100 \
      --start-index 50 \
      --selection-seed 0 \
      --n-samples 8 \
      --max-new-tokens 16384 \
      --temperature 0.7 \
      --seed 0 \
      --base-gpu-id 0 \
      --tp 1 \
      --attention-backend triton \
      --mem-fraction-static 0.75 \
      --dump-trajectories "$run_dir/independent_n8_trajectories.jsonl" \
      --summary-output "$run_dir/independent_n8_summary.json"
fi

if [[ ! -s "$run_dir/qwen38_terminal_scores.jsonl" || ! -s "$run_dir/qwen38_terminal_summary.json" ]]; then
  CUDA_HOME="$cuda_toolkit" CUDA_VISIBLE_DEVICES="$verifier_a_devices" \
    .venv/bin/python scripts/offline_pointwise_error_audit.py \
      --trajectories "$run_dir/independent_n8_trajectories.jsonl" \
      --scorer "$verifier_a" \
      --fractions 1.0 \
      --batch-size 512 \
      --base-gpu-id 0 \
      --dp 2 \
      --max-running-requests 128 \
      --max-mamba-cache-size 128 \
      --seed 0 \
      --bootstrap-samples 1000 \
      --save-scores "$run_dir/qwen38_terminal_scores.jsonl" \
      --summary-output "$run_dir/qwen38_terminal_summary.json"
fi

if [[ ! -s "$run_dir/qwen3_32b_terminal_scores.jsonl" || ! -s "$run_dir/qwen3_32b_terminal_summary.json" ]]; then
  CUDA_HOME="$cuda_toolkit" CUDA_VISIBLE_DEVICES="$verifier_b_devices" \
    .venv/bin/python scripts/offline_pointwise_error_audit.py \
      --trajectories "$run_dir/independent_n8_trajectories.jsonl" \
      --scorer "$verifier_b" \
      --fractions 1.0 \
      --batch-size 512 \
      --base-gpu-id 0 \
      --dp 2 \
      --max-running-requests 128 \
      --seed 1 \
      --bootstrap-samples 1000 \
      --save-scores "$run_dir/qwen3_32b_terminal_scores.jsonl" \
      --summary-output "$run_dir/qwen3_32b_terminal_summary.json"
fi

if [[ ! -s "$run_dir/semantic_smc_problems.jsonl" || ! -s "$run_dir/semantic_smc_summary.json" ]]; then
  CUDA_HOME="$cuda_toolkit" CUDA_VISIBLE_DEVICES="$smc_devices" \
    .venv/bin/python scripts/online_semantic_particlescale.py \
      --trajectories "$run_dir/independent_n8_trajectories.jsonl" \
      --method smc \
      --generator "$generator" \
      --scorer "$verifier_a" \
      --rubrics validity \
      --rubric-weights 1 \
      --initial-prefix-tokens 2048 \
      --checkpoint-interval 2048 \
      --total-budget 16384 \
      --beta 12 \
      --ess-threshold 0.75 \
      --temperature 0.7 \
      --seed 71 \
      --generator-base-gpu-id 0 \
      --generator-tp 1 \
      --generator-mem-fraction-static 0.7 \
      --verifier-base-gpu-id 1 \
      --verifier-dp 2 \
      --verifier-tp 1 \
      --verifier-mem-fraction-static 0.75 \
      --verifier-batch-size 512 \
      --max-running-requests 128 \
      --generator-cost-summary "$run_dir/independent_n8_summary.json" \
      --save-particles "$run_dir/semantic_smc_particles.jsonl" \
      --save-events "$run_dir/semantic_smc_events.jsonl" \
      --save-problems "$run_dir/semantic_smc_problems.jsonl" \
      --summary-output "$run_dir/semantic_smc_summary.json"
fi

.venv/bin/python scripts/analyze_olympiadbench_semantic_smc_holdout.py \
  --config "$config" \
  --trajectories "$run_dir/independent_n8_trajectories.jsonl" \
  --generation-summary "$run_dir/independent_n8_summary.json" \
  --verifier-a-scores "$run_dir/qwen38_terminal_scores.jsonl" \
  --verifier-a-summary "$run_dir/qwen38_terminal_summary.json" \
  --verifier-a-gpus 2 \
  --verifier-b-scores "$run_dir/qwen3_32b_terminal_scores.jsonl" \
  --verifier-b-summary "$run_dir/qwen3_32b_terminal_summary.json" \
  --verifier-b-gpus 2 \
  --smc-problems "$run_dir/semantic_smc_problems.jsonl" \
  --smc-summary "$run_dir/semantic_smc_summary.json" \
  --save-outcomes "$run_dir/method_outcomes.jsonl" \
  --output results/olympiadbench/semantic_smc_holdout_v1_summary.json
