import struct

import pytest

from canirun import gguf
from canirun.models import ModelError


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    out = _str(key)
    if isinstance(value, str):
        return out + struct.pack("<I", 8) + _str(value)
    if isinstance(value, list):
        if value and isinstance(value[0], str):
            return out + struct.pack("<IIQ", 9, 8, len(value)) + b"".join(_str(v) for v in value)
        return out + struct.pack("<IIQ", 9, 4, len(value)) + b"".join(struct.pack("<I", v) for v in value)
    return out + struct.pack("<I", 4) + struct.pack("<I", value)


def write_gguf(path, meta: dict, tensors: list[tuple[str, list[int]]]):
    body = b"".join(_kv(k, v) for k, v in meta.items())
    for name, shape in tensors:
        body += _str(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        body += struct.pack("<IQ", 12, 0)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)) + body + b"\0" * 4096)


LLAMA_META = {
    "general.architecture": "llama",
    "general.file_type": 15,
    "llama.block_count": 2,
    "llama.embedding_length": 64,
    "llama.feed_forward_length": 128,
    "llama.attention.head_count": 8,
    "llama.attention.head_count_kv": 2,
    "llama.context_length": 4096,
    "tokenizer.ggml.tokens": [f"t{i}" for i in range(5000)],  # longer than the kept-array limit
}
TENSORS = [
    ("token_embd.weight", [64, 5000]),
    ("output.weight", [64, 5000]),
    *[(f"blk.{i}.ffn_up.weight", [64, 128]) for i in range(2)],
]


def test_reads_header_and_builds_spec(tmp_path):
    p = tmp_path / "tiny.gguf"
    write_gguf(p, LLAMA_META, TENSORS)
    meta, tensors = gguf.read_header(p)
    assert meta["llama.block_count"] == 2
    assert meta["tokenizer.ggml.tokens"] == {"_len": 5000}
    assert tensors[0] == ("token_embd.weight", [64, 5000])

    spec = gguf.load_file(p)
    assert spec.file_quant == "Q4_K_M"
    assert spec.vocab_size == 5000
    assert spec.total_params == 2 * 64 * 5000 + 2 * 64 * 128
    assert spec.kv_dim == 2 * 2 * 8  # 2 kv heads x head_dim 8 (64 / 8 heads)
    assert spec.tied_embeddings is False
    assert spec.file_bytes == p.stat().st_size


def test_tied_embeddings_when_no_output_tensor(tmp_path):
    p = tmp_path / "tied.gguf"
    write_gguf(p, LLAMA_META, [t for t in TENSORS if t[0] != "output.weight"])
    assert gguf.load_file(p).tied_embeddings is True


def test_hybrid_layers_without_kv(tmp_path):
    p = tmp_path / "hybrid.gguf"
    meta = {**LLAMA_META, "llama.block_count": 4, "llama.attention.head_count_kv": [0, 2, 0, 2]}
    write_gguf(p, meta, TENSORS)
    spec = gguf.load_file(p)
    assert spec.full_attn_layers == 2


def test_not_gguf(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"nope" * 10)
    with pytest.raises(ModelError, match="not a GGUF"):
        gguf.read_header(p)


def test_spec_from_ollama_style_metadata():
    # Ollama's /api/show returns the same keys, with long arrays nulled out
    meta = {**LLAMA_META, "tokenizer.ggml.tokens": None, "llama.vocab_size": 5000,
            "general.parameter_count": 1_000_000}
    spec = gguf.spec_from_gguf(meta, TENSORS, "ollama-model", file_bytes=600_000)
    assert spec.total_params == 1_000_000
    assert spec.file_quant == "Q4_K_M"
