import json

import pytest

from canirun import models
from canirun.models import ModelError, count_params, spec_from_config

from conftest import DEEPSEEK_V3, GEMMA3_12B, LLAMA3_8B, QWEN3_30B_A3B


def test_llama3_param_count_from_config():
    total, active = count_params(LLAMA3_8B)
    assert total == active
    assert total == pytest.approx(8.03e9, rel=0.005)


def test_llama3_kv_layout(llama):
    assert llama.kv_dim == 2 * 8 * 128
    assert (llama.full_attn_layers, llama.swa_layers) == (32, 0)
    assert llama.attention == "gqa"
    assert llama.param_source == "safetensors"


def test_moe_active_params():
    spec = spec_from_config(QWEN3_30B_A3B, "qwen3-30b-a3b")
    assert spec.is_moe
    assert spec.total_params == pytest.approx(30.5e9, rel=0.01)
    assert spec.active_params == pytest.approx(3.3e9, rel=0.05)


def test_exact_params_keep_moe_split():
    spec = spec_from_config(QWEN3_30B_A3B, "q", exact_params=30_532_122_624)
    assert spec.total_params == 30_532_122_624
    assert spec.active_params == pytest.approx(3.3e9, rel=0.05)


def test_disabled_sliding_window_is_full_attention():
    spec = spec_from_config({**QWEN3_30B_A3B, "sliding_window": 4096, "use_sliding_window": False}, "q")
    assert spec.swa_layers == 0


def test_gemma3_nested_config_and_sliding_pattern():
    spec = spec_from_config(GEMMA3_12B, "gemma3-12b")
    assert spec.n_layers == 48
    assert spec.full_attn_layers == 8
    assert spec.swa_layers == 40
    assert spec.sliding_window == 1024
    assert spec.kv_dim == 2 * 8 * 256  # explicit head_dim, not hidden / heads


def test_layer_types_win():
    cfg = {**LLAMA3_8B, "sliding_window": 128,
           "layer_types": ["sliding_attention", "full_attention"] * 16}
    spec = spec_from_config(cfg, "x")
    assert (spec.full_attn_layers, spec.swa_layers) == (16, 16)


def test_linear_attention_layers_have_no_kv():
    cfg = {**LLAMA3_8B, "layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8}
    assert spec_from_config(cfg, "x").full_attn_layers == 8


def test_deepseek_mla():
    spec = spec_from_config(DEEPSEEK_V3, "deepseek-v3")
    assert spec.attention == "mla"
    assert spec.kv_dim == 512 + 64
    assert spec.total_params == pytest.approx(671e9, rel=0.01)
    assert spec.active_params == pytest.approx(37e9, rel=0.05)


def test_missing_fields_raise():
    with pytest.raises(ModelError):
        spec_from_config({"model_type": "weird"}, "weird")


def test_resolve_alias_and_repo():
    assert models.resolve("Llama3-8B") == models.ALIASES["llama3-8b"]
    assert models.resolve("org/model") == "org/model"


def test_resolve_suggests_close_alias():
    with pytest.raises(ModelError, match="qwen3-8b"):
        models.resolve("qwen3-8")


def test_load_local_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(LLAMA3_8B))
    spec = models.load(str(tmp_path))
    assert spec.n_layers == 32
    assert spec.param_source == "config"
