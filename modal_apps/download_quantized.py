"""Download a quantized checkpoint from the smcsd-quantized-models Modal
volume to the local filesystem.

Usage:
    modal volume get smcsd-quantized-models /Qwen3.6-27B-W4A16 ./Qwen3.6-27B-W4A16

That CLI is the easiest path. This module is here only so you can `cat` it
to see the expected layout / size.
"""

# Layout under the volume:
#   /Qwen3.6-27B-W4A16/
#     config.json                       (multimodal arch + quantization_config)
#     model.safetensors                 (~17.65 GB, INT4 group-quant + scales)
#     recipe.yaml                       (llmcompressor recipe used)
#     tokenizer.json, tokenizer_config.json, vocab.json, merges.txt,
#     chat_template.jinja, generation_config.json, special_tokens_map.json,
#     preprocessor_config.json, video_preprocessor_config.json
