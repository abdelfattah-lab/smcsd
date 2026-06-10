from vllm.config import ProfilerConfig

from smcsd.vllm_backend.engine import SMCVLLMEngine


PROMPTS = [
    "The capital of France is",
    "Write one sentence about why overlap scheduling matters for inference systems.",
    "List two prime numbers and one composite number.",
    "In one short paragraph, explain speculative decoding.",
    "What is 1+1?",
    # "what is 5+5+2",
    # "Why is the sky blue?",
    # "Where is the Cornell Tech campus located?"
]

engine = SMCVLLMEngine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    draft_model_path="meta-llama/Llama-3.2-1B-Instruct",
    n_particles=12,
    gamma=8,
    max_num_seqs=60,
    max_num_batched_tokens=4096,
    tp_size=1,
    max_model_len=1024,
    gpu_memory_utilization=0.5,
    enable_prefix_caching=False,
    # profiler_config=ProfilerConfig(
    #     profiler="torch",
    #     torch_profiler_dir="/home/xq88/smcsd/tmp/smc_traces",
    # ),
)
print("see config")
print(engine._vllm_config.scheduler_config.max_num_seqs)
print(engine._engine.scheduler.max_concurrent_groups)

# engine._engine.profile(is_start=True, profile_prefix="smc_trace")
out = engine.generate(
    prompt=PROMPTS,
    sampling_params={
        "draft_temperature": 0.7,
        "target_temperature": 0.7,
    },
)
# engine._engine.profile(is_start=False)

results = [out] if isinstance(out, dict) else out

for prompt_idx, result in enumerate(results):
    print(f"\nprompt[{prompt_idx}]: {PROMPTS[prompt_idx]}")
    print(f"final output: {result['text']}")
    for particle_idx, particle_ids in enumerate(result["particles"]):
        text = engine.tokenizer.decode(particle_ids, skip_special_tokens=True)
        print(
            f"particle[{particle_idx}] ({len(particle_ids)} tokens): {text}"
        )

engine.shutdown()
