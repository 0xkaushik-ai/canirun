import pytest

from canirun import estimate as est
from canirun.hardware import GiB, nvml_bandwidth_gbs
from canirun.models import spec_from_config
from canirun.quants import KV_BYTES, QUANTS_BY_NAME

from conftest import GEMMA3_12B, QWEN3_30B_A3B, machine

Q4 = QUANTS_BY_NAME["q4_k_m"]
Q8 = QUANTS_BY_NAME["q8_0"]
F16 = QUANTS_BY_NAME["f16"]


def test_kv_cache_llama3_is_128kb_per_token(llama):
    assert est.kv_cache_bytes(llama, 1) == 128 * 1024
    assert est.kv_cache_bytes(llama, 8192) == 1 * GiB


def test_kv_quant_halves_cache(llama):
    assert est.kv_cache_bytes(llama, 8192, KV_BYTES["q8_0"]) == pytest.approx(0.53 * GiB, rel=0.01)


def test_sliding_window_caps_kv():
    spec = spec_from_config(GEMMA3_12B, "g")
    short = est.kv_cache_bytes(spec, 1024)
    long = est.kv_cache_bytes(spec, 32768)
    # only the 8 global layers keep growing past the 1024-token window
    assert long - short == spec.kv_dim * 2 * 8 * (32768 - 1024)


def test_q4_weights_about_4_9_gb(llama):
    e = est.estimate(llama, machine(ram_gb=64), Q4, 8192)
    assert e.weights_bytes / 1e9 == pytest.approx(4.9, abs=0.1)


def test_fits_fully_on_big_gpu(llama):
    e = est.estimate(llama, machine(vram_gb=24, gpu_bw=1008), Q4, 8192)
    assert e.placement == "gpu"
    assert e.gpu_layers == 32
    # ceiling is 1008 GB/s / ~5 GB read; 70% efficiency puts it in the 120s-140s
    assert 110 < e.tok_s < 150


def test_split_on_small_gpu(llama):
    e = est.estimate(llama, machine(vram_gb=4, gpu_bw=192, ram_gb=16, ram_bw=46), Q4, 8192)
    assert e.placement == "split"
    assert 0 < e.gpu_layers < 32
    cpu_only = est.estimate(llama, machine(ram_gb=16, ram_bw=46), Q4, 8192)
    assert cpu_only.placement == "cpu"
    assert e.tok_s > cpu_only.tok_s


def test_more_gpu_layers_is_faster(llama):
    small = est.estimate(llama, machine(vram_gb=3, gpu_bw=192), Q4, 8192)
    big = est.estimate(llama, machine(vram_gb=5, gpu_bw=192), Q4, 8192)
    assert big.gpu_layers > small.gpu_layers
    assert big.tok_s > small.tok_s


def test_wont_fit(llama):
    e = est.estimate(llama, machine(ram_gb=8), F16, 8192)
    assert e.placement == "none"
    assert e.tok_s is None


def test_unified_memory_never_splits(llama):
    hw = machine(vram_gb=10, gpu_bw=400, ram_gb=16, unified=True)
    assert est.estimate(llama, hw, Q8, 8192).placement == "gpu"
    assert est.estimate(llama, hw, F16, 8192).placement == "none"


def test_moe_speed_uses_active_params():
    spec = spec_from_config(QWEN3_30B_A3B, "q")
    hw = machine(ram_gb=64, ram_bw=50)
    e = est.estimate(spec, hw, Q4, 4096)
    assert e.weights_bytes > 17 * GiB          # all experts must be in memory
    # but only ~3B are read per token: several times a dense 30B (~2 tok/s here)
    assert e.tok_s > 8


def test_close_apps_flag(llama):
    hw = machine(ram_gb=16, ram_free_gb=4)
    by_name = {e.quant.name: e for e in est.estimate_all(llama, hw, 8192)}
    assert by_name["Q4_K_M"].needs_free_ram
    assert by_name["Q4_K_M"].fits
    assert by_name["F16"].placement == "none"


def test_recommend_prefers_quality_at_reading_speed(llama):
    ests = est.estimate_all(llama, machine(vram_gb=24, gpu_bw=1008), 8192)
    assert est.recommend(ests).quant.name == "Q8_0"     # never F16


def test_recommend_full_gpu_over_offloaded_higher_quant():
    dense_32b = spec_from_config({
        "model_type": "qwen3", "hidden_size": 5120, "intermediate_size": 25600,
        "num_hidden_layers": 64, "num_attention_heads": 64, "num_key_value_heads": 8,
        "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": False,
    }, "qwen3-32b")
    ests = est.estimate_all(dense_32b, machine(vram_gb=24, gpu_bw=1008, ram_gb=32, ram_bw=60), 8192)
    best = est.recommend(ests)
    assert best.placement == "gpu"
    assert best.quant.name == "Q4_K_M"


def test_recommend_moe_full_gpu_over_split_q8():
    spec = spec_from_config(QWEN3_30B_A3B, "q")
    best = est.recommend(est.estimate_all(spec, machine(vram_gb=24, gpu_bw=1008, ram_gb=32, ram_bw=60), 8192))
    assert best.placement == "gpu"
    assert best.quant.name == "Q5_K_M"


def test_recommend_slow_machine_takes_q4(llama):
    ests = est.estimate_all(llama, machine(ram_gb=32, ram_bw=20), 8192)
    assert est.recommend(ests).quant.name == "Q4_K_M"


def test_recommend_none_when_nothing_fits(llama):
    assert est.recommend(est.estimate_all(llama, machine(ram_gb=3), 8192)) is None


def test_recommend_keeps_quality_when_apps_must_close(llama):
    hw = machine(ram_gb=16, ram_free_gb=5, ram_bw=50)
    best = est.recommend(est.estimate_all(llama, hw, 8192))
    assert best.quant.bits_per_weight >= 4.5
    assert best.needs_free_ram


def test_native_mxfp4_rows():
    from canirun.quants import quants_for
    names = [q.name for q in quants_for("mxfp4")]
    assert names == ["F16", "Q8_0", "MXFP4"]
    assert quants_for("") is not None and len(quants_for("")) == 7


def test_nvml_bandwidth_formula():
    assert nvml_bandwidth_gbs(128, 6001) == pytest.approx(192, rel=0.01)   # RTX 3050 Laptop
    assert nvml_bandwidth_gbs(384, 10501) == pytest.approx(1008, rel=0.01)  # RTX 4090
    assert nvml_bandwidth_gbs(5120, 1593) == pytest.approx(2039, rel=0.01)  # A100 80GB HBM2e
