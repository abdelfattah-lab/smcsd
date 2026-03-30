
python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
    --speculative-eagle-topk 1 \
    --speculative-num-steps 4 \
    --speculative-num-draft-tokens 5 \
    --attention-backend triton \
    --mem-fraction-static 0.60 \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 512 \
    --num-prompts 16
