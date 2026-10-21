import json

import pytest

from canirun import cli, presets

from conftest import LLAMA3_8B


@pytest.fixture
def config_path(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(LLAMA3_8B))
    return str(p)


def test_what_if_gpu_json(config_path, capsys):
    assert cli.main([config_path, "--gpu", "4090", "--ram", "32", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    by_quant = {e["quant"]: e for e in out["estimates"]}
    assert by_quant["Q4_K_M"]["placement"] == "gpu"
    assert out["recommended"] == "Q8_0"


def test_what_if_mac(config_path, capsys):
    assert cli.main([config_path, "--gpu", "m4 max", "--ram", "64", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hardware"]["unified"] is True
    assert out["recommended"] == "Q8_0"


def test_table_output(config_path, capsys):
    assert cli.main([config_path, "--ram", "16", "--ram-bw", "50", "--cpu-only"]) == 0
    out = capsys.readouterr().out
    assert "Q4_K_M" in out and "recommended" in out


def test_unknown_gpu(config_path, capsys):
    assert cli.main([config_path, "--gpu", "voodoo5"]) == 1
    assert "Unknown GPU" in capsys.readouterr().err


def test_list_commands(capsys):
    assert cli.main(["--list-models", "--list-gpus"]) == 0
    out = capsys.readouterr().out
    assert "llama3-8b" in out and "rtx 4090" in out


@pytest.mark.parametrize("name,key", [
    ("4090", "rtx 4090"),
    ("RTX 4070 Ti Super", "rtx 4070 ti super"),
    ("NVIDIA GeForce RTX 3090", "rtx 3090"),
])
def test_gpu_lookup(name, key):
    assert presets.find_gpu(name)[0] == key


def test_apple_lookup():
    assert presets.find_apple("Apple M3 Max")[0] == "m3 max"
    assert presets.find_apple("Apple M2")[0] == "m2"
