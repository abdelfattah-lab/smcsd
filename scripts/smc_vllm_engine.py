from smcsd.vllm_backend.engine import SMCVLLMEngine

engine = SMCVLLMEngine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    draft_model_path="meta-llama/Llama-3.2-1B-Instruct",
    n_particles=8,
    gamma=8,
    temperature=0.7,
    tp_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.6,
)

out = engine.generate(
    prompt="Write one sentence about speculative decoding.",
    sampling_params={
        "max_tokens": 32,
        "temperature": 0.7,
    },
)

print("text:", out["text"])
print("output_ids:", out["output_ids"])
print("num_particles:", len(out["particles"]))
print("particle_lengths:", [len(p) for p in out["particles"]])

engine.shutdown()

