python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm SMC \
    --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
    --smc-n-particles 6 \
    --smc-gamma 8 \
    --smc-draft-temperature 0.7 \
    --smc-target-temperature 0.7 \
    --smc-pingpong-overlap \
    --attention-backend triton \
    --mem-fraction-static 0.60 \
    --max-running-requests 64 \
    --cuda-graph-bs 64 \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 512 \
    --num-prompts 16
    # --profile \
    # --profile-num-steps 5 \
    # --profile-decode-only