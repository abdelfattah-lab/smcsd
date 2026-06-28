from transformers import AutoConfig
c = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
d = c.to_dict()
for k in ['hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','intermediate_size','vocab_size','rope_theta','rope_scaling','max_position_embeddings','rms_norm_eps']:
    print(f"  {k} = {d.get(k)}")
