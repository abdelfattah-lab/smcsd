#!/bin/bash
# E2: delayed-resampling accuracy sweep (400q GSM8K, N=8, thr=0.5)
cd ~/smcsd
mkdir -p paper/e2_results
PY=~/miniconda3/envs/test/bin/python
launch() {
  tag=$1; GPU=$2; SB=$3; D=$4; G=$5
  CUDA_VISIBLE_DEVICES=$GPU SMC_SELF_BONUS=$SB SMC_DELAY_RESAMPLE=$D \
  nohup $PY scripts/accuracy_test_gsm8k.py \
    --mode smc_engine --model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B-Instruct \
    --particles 8 --gamma $G --temperature 0.7 --attention-backend triton \
    --num-questions 400 > paper/e2_results/${tag}.log 2>&1 &
  echo "launched $tag gpu=$GPU"
}
launch d1_sb0_g8 2 0 1 8
launch d1_sb1_g8 3 1 1 8
launch d2_sb0_g8 4 0 2 8
launch d2_sb1_g8 5 1 2 8
launch d0_sb1_g4 6 1 0 4
launch d2_sb1_g4 7 1 2 4
