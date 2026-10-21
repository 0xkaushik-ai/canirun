import json
import urllib.parse

import pytest

from canirun import bench, cli
from canirun.models import ModelError, spec_from_config

from conftest import LLAMA3_8B, machine


@pytest.fixture
def gguf_spec():
    spec = spec_from_config(LLAMA3_8B, "llama3-8b", exact_params=8_030_261_248)
    spec.file_quant, spec.file_bytes = "Q4_K_M", 4_920_000_000
    return spec


def _result(**kw):
    base = dict(backend="ollama", backend_version="ollama 0.15", model="m", source="ollama:m",
                quant="Q4_K_M", file_gb=4.6, n_layers=32, gpu_layers=32, predicted_tok_s=110.0,
                measured_tok_s=100.0, prompt_tok_s=900.0,
                hardware={"cpu": "c", "ram_gb": 32, "ram_bw_gbs": 50, "gpu": "RTX 4090",
                          "vram_gb": 24, "gpu_bw_gbs": 1008, "unified": False})
    return bench.BenchResult(**{**base, **kw})


def test_needs_a_real_file(llama):
    with pytest.raises(ModelError, match="gguf"):
        bench.run(llama, machine(vram_gb=24, gpu_bw=1008), "llama3-8b", "cpu")


def test_runs_ollama_at_predicted_layers(gguf_spec, monkeypatch):
    calls = {}

    def fake(model, all_on_gpu, layers):
        calls.update(model=model, all_on_gpu=all_on_gpu, layers=layers)
        return 100.0, 900.0, "ollama test"

    monkeypatch.setattr(bench, "_ollama_bench", fake)
    r = bench.run(gguf_spec, machine(vram_gb=24, gpu_bw=1008), "ollama:llama3", "cpu")
    assert calls == {"model": "llama3", "all_on_gpu": True, "layers": 32}
    assert r.measured_tok_s == 100.0
    assert r.predicted_tok_s > 0


def test_backs_off_when_out_of_memory(gguf_spec, monkeypatch):
    attempts = []

    def fake(model, all_on_gpu, layers):
        attempts.append(layers)
        if len(attempts) == 1:
            raise RuntimeError("CUDA error: out of memory")
        return 20.0, None, "ollama test"

    monkeypatch.setattr(bench, "_ollama_bench", fake)
    r = bench.run(gguf_spec, machine(vram_gb=4, gpu_bw=192), "ollama:llama3", "cpu")
    assert attempts[1] == attempts[0] - 2
    assert r.gpu_layers == attempts[1]
    assert "out of memory" in r.notes[0]


def test_other_failures_are_reported(gguf_spec, monkeypatch):
    def fake(*a):
        raise RuntimeError("model not found")

    monkeypatch.setattr(bench, "_ollama_bench", fake)
    with pytest.raises(ModelError, match="model not found"):
        bench.run(gguf_spec, machine(vram_gb=24, gpu_bw=1008), "ollama:llama3", "cpu")


def test_missing_llama_bench(gguf_spec, monkeypatch, tmp_path):
    monkeypatch.delenv("LLAMA_BENCH", raising=False)
    monkeypatch.setattr(bench.shutil, "which", lambda _: None)
    with pytest.raises(ModelError, match="llama-bench not found"):
        bench.run(gguf_spec, machine(vram_gb=24, gpu_bw=1008), str(tmp_path / "m.gguf"), "cpu")


def test_error_pct():
    assert _result().error_pct == pytest.approx(10.0)


def test_share_url_is_prefilled_issue():
    url = bench.share_url(_result())
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert url.startswith(bench.ISSUES_URL)
    assert q["labels"] == ["benchmark"]
    assert "RTX 4090" in q["title"][0]
    assert "+10%" in q["body"][0]
    assert len(url) < 8000  # GitHub's practical URL limit


def test_save(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "RESULTS_DIR", tmp_path)
    path = bench.save(_result(model="phi3:mini"))
    assert json.loads(path.read_text())["model"] == "phi3:mini"
    assert ":" not in path.name


def test_cli_rejects_what_if_with_benchmark():
    with pytest.raises(SystemExit):
        cli.parse_args(["ollama:x", "--benchmark", "--gpu", "4090"])
