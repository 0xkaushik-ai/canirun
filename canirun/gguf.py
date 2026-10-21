"""Read GGUF metadata (from a file or Ollama) into a ModelSpec.

Only the header is read: a few MB at most, never the weights.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

from .models import ModelError, ModelSpec, spec_from_config

# llama.cpp's llama_ftype enum (general.file_type)
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4",
}

# GGUF value types -> struct format (8 = string, 9 = array)
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
            10: "<Q", 11: "<q", 12: "<d"}
_MAX_KEPT_ARRAY = 4096  # per-layer arrays are kept; token lists only keep their length


class _Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ModelError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self) -> str:
        return self.f.read(self.unpack("<Q")).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in _SCALARS:
            return self.unpack(_SCALARS[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            item_type, count = self.unpack("<I"), self.unpack("<Q")
            if item_type in _SCALARS and count <= _MAX_KEPT_ARRAY:
                return [self.value(item_type) for _ in range(count)]
            if item_type in _SCALARS:
                self.f.seek(struct.calcsize(_SCALARS[item_type]) * count, 1)
            else:
                for _ in range(count):
                    self.value(item_type)
            return {"_len": count}
        raise ModelError(f"unknown GGUF value type {vtype}")


def read_header(path: str | Path) -> tuple[dict, list[tuple[str, list[int]]]]:
    """Return (metadata, [(tensor name, shape)])."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ModelError(f"{path} is not a GGUF file")
        r = _Reader(f)
        version = r.unpack("<I")
        if version < 2:
            raise ModelError(f"GGUF v{version} is too old; re-convert the model")
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.unpack("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            shape = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
            r.unpack("<I")  # ggml type
            r.unpack("<Q")  # data offset
            tensors.append((name, shape))
    return meta, tensors


def _per_layer(value, n_layers: int) -> list[int]:
    if isinstance(value, list):
        return list(value)
    return [value] * n_layers


def spec_from_gguf(meta: dict, tensors: list[tuple[str, list[int]]], name: str,
                   file_bytes: int) -> ModelSpec:
    arch = meta.get("general.architecture")
    if not arch:
        raise ModelError(f"{name}: GGUF has no general.architecture")

    def a(key, default=None):
        return meta.get(f"{arch}.{key}", default)

    n_layers = a("block_count")
    hidden = a("embedding_length")
    if not n_layers or not hidden:
        raise ModelError(f"{name}: unsupported GGUF architecture {arch!r}")

    heads = _per_layer(a("attention.head_count"), n_layers)
    kv_heads = _per_layer(a("attention.head_count_kv", a("attention.head_count")), n_layers)
    names = {t[0] for t in tensors}
    embd = next((s for n, s in tensors if n == "token_embd.weight"), None)
    tokens = meta.get("tokenizer.ggml.tokens")
    vocab = a("vocab_size") or (tokens or {}).get("_len") or (embd[1] if embd else None)
    if not vocab:
        raise ModelError(f"{name}: GGUF has no vocabulary size")

    ffn = a("feed_forward_length")
    cfg = {
        "model_type": arch,
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": max(heads),
        "num_key_value_heads": max(kv_heads),
        "head_dim": a("attention.key_length") or hidden // max(heads),
        "vocab_size": vocab,
        "intermediate_size": max(ffn) if isinstance(ffn, list) else ffn,
        "max_position_embeddings": a("context_length"),
        "tie_word_embeddings": bool(tensors) and "output.weight" not in names,
        "num_experts": a("expert_count"),
        "num_experts_per_tok": a("expert_used_count"),
        "moe_intermediate_size": a("expert_feed_forward_length"),
        "n_shared_experts": a("expert_shared_count"),
        "first_k_dense_replace": a("leading_dense_block_count"),
        "sliding_window": a("attention.sliding_window"),
    }
    if any(k == 0 for k in kv_heads):  # hybrid: layers without attention keep no KV cache
        cfg["layer_types"] = ["full_attention" if k else "linear_attention" for k in kv_heads]
    if a("attention.kv_lora_rank"):  # DeepSeek MLA
        rope = a("rope.dimension_count")
        cfg.update({
            "kv_lora_rank": a("attention.kv_lora_rank"),
            "q_lora_rank": a("attention.q_lora_rank"),
            "qk_rope_head_dim": rope,
            "qk_nope_head_dim": a("attention.key_length") - rope,
            "v_head_dim": a("attention.value_length"),
        })

    params = meta.get("general.parameter_count")
    if not params and tensors:
        params = sum(math.prod(shape) for _, shape in tensors)

    spec = spec_from_config({k: v for k, v in cfg.items() if v is not None}, name, params)
    spec.param_source = "gguf"
    spec.file_quant = FILE_TYPES.get(meta.get("general.file_type"), "")
    spec.file_bytes = file_bytes
    return spec


def load_file(path: str | Path) -> ModelSpec:
    path = Path(path)
    meta, tensors = read_header(path)
    return spec_from_gguf(meta, tensors, path.stem, path.stat().st_size)
