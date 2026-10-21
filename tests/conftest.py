import pytest

from canirun.hardware import GiB, Hardware, Memory

LLAMA3_8B = {
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "tie_word_embeddings": False,
}

QWEN3_30B_A3B = {
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "tie_word_embeddings": False,
    "sliding_window": None,
    "use_sliding_window": False,
}

GEMMA3_12B = {
    "model_type": "gemma3",
    "text_config": {
        "model_type": "gemma3_text",
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "sliding_window": 1024,
        "sliding_window_pattern": 6,
        "vocab_size": 262208,
    },
}

DEEPSEEK_V3 = {
    "model_type": "deepseek_v3",
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "vocab_size": 129280,
    "tie_word_embeddings": False,
}


def machine(vram_gb=None, gpu_bw=0.0, ram_gb=16.0, ram_free_gb=None, ram_bw=50.0, unified=False):
    ram_free = (ram_free_gb if ram_free_gb is not None else ram_gb) * GiB
    ram = Memory("RAM", ram_gb * GiB, ram_free, ram_bw, "cpu")
    gpu = None
    if vram_gb:
        gpu = Memory("GPU", vram_gb * GiB, vram_gb * GiB, gpu_bw, "metal" if unified else "cuda")
    return Hardware(cpu_name="test", ram=ram, gpu=gpu, unified=unified)


@pytest.fixture
def llama():
    from canirun.models import spec_from_config
    return spec_from_config(LLAMA3_8B, "llama3-8b", exact_params=8_030_261_248)
