"""Turn a Hugging Face config.json into the numbers the estimator needs."""
from __future__ import annotations

import difflib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "canirun" / "models"
CACHE_VERSION = 2  # bump when ModelSpec or the parsing changes

# Short names -> ungated HF repos with the same architecture as the original.
ALIASES: dict[str, str] = {
    "llama3-8b": "unsloth/llama-3-8b-Instruct",
    "llama3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "llama3.1-70b": "unsloth/Meta-Llama-3.1-70B-Instruct",
    "llama3.2-1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama3.3-70b": "unsloth/Llama-3.3-70B-Instruct",
    "llama3-70b": "unsloth/Llama-3.3-70B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B",
    "qwen3-coder-30b-a3b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "qwen3-next-80b-a3b": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "gemma2-9b": "unsloth/gemma-2-9b-it",
    "gemma2-27b": "unsloth/gemma-2-27b-it",
    "gemma3-1b": "unsloth/gemma-3-1b-it",
    "gemma3-4b": "unsloth/gemma-3-4b-it",
    "gemma3-12b": "unsloth/gemma-3-12b-it",
    "gemma3-27b": "unsloth/gemma-3-27b-it",
    "mistral-7b": "unsloth/mistral-7b-instruct-v0.3",
    "mistral-nemo-12b": "unsloth/Mistral-Nemo-Instruct-2407",
    "mistral-small-24b": "unsloth/Mistral-Small-24B-Instruct-2501",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "phi4": "microsoft/phi-4",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1-distill-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}

# safetensors dtypes that pack several weights per element (MXFP4, GPTQ, AWQ),
# so their element count is not a parameter count.
PACKED_DTYPES = {"U8", "I8", "U32", "I32"}


class ModelError(Exception):
    pass


@dataclass
class ModelSpec:
    name: str
    total_params: int
    active_params: int          # == total_params for dense models
    n_layers: int
    hidden_size: int
    vocab_size: int
    tied_embeddings: bool
    kv_dim: int                 # KV cache elements per token per attention layer
    full_attn_layers: int
    swa_layers: int             # sliding-window layers only cache `sliding_window` tokens
    sliding_window: int
    max_context: int
    attention: str = "gqa"      # "gqa" | "mla"
    n_experts: int = 0
    experts_per_token: int = 0
    param_source: str = "config"
    native_quant: str = ""      # e.g. "mxfp4" for gpt-oss

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.hidden_size


def _get(cfg: dict, *keys, default=None):
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return default


def _flatten(cfg: dict) -> dict:
    """Multimodal configs (Gemma 3, Mistral 3, Llama 4) nest the LLM in text_config."""
    text = cfg.get("text_config")
    if isinstance(text, dict):
        merged = {k: v for k, v in cfg.items() if k != "text_config"}
        merged.update(text)
        merged.setdefault("model_type", cfg.get("model_type", ""))
        return merged
    return cfg


def _attention_layout(cfg: dict, n_layers: int) -> tuple[int, int, int]:
    """Return (full-attention layers, sliding-window layers, window size)."""
    window = cfg.get("sliding_window") or 0
    model_type = str(cfg.get("model_type", ""))

    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        full = sum(1 for t in layer_types if t == "full_attention")
        swa = sum(1 for t in layer_types if t == "sliding_attention")
        # anything else (linear_attention, mamba) keeps no growing KV cache
        return full, (swa if window else 0), window

    if cfg.get("full_attention_interval"):  # Qwen3-Next: 1 in N layers is attention
        return n_layers // cfg["full_attention_interval"], 0, 0

    if not window or cfg.get("use_sliding_window") is False:
        return n_layers, 0, 0

    if model_type.startswith("gemma2"):
        swa = n_layers // 2
    elif model_type.startswith("gemma3"):
        pattern = cfg.get("sliding_window_pattern") or 6
        swa = n_layers - n_layers // pattern
    else:
        swa = n_layers
    return n_layers - swa, swa, window


def _moe_layers(cfg: dict, n_layers: int) -> int:
    first_dense = cfg.get("first_k_dense_replace") or 0
    dense_only = set(cfg.get("mlp_only_layers") or [])
    step = cfg.get("decoder_sparse_step") or 1
    return sum(
        1 for i in range(n_layers)
        if i >= first_dense and i not in dense_only and (i + 1) % step == 0
    )


def count_params(cfg: dict) -> tuple[int, int]:
    """Estimate (total, active) parameters from config alone.

    Used when exact safetensors counts are unavailable; also the source of the
    active-parameter split for MoE models.
    """
    cfg = _flatten(cfg)
    h = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = _get(cfg, "num_key_value_heads", default=n_heads)
    hd = _get(cfg, "head_dim", default=h // n_heads)
    vocab = cfg["vocab_size"]
    inter = _get(cfg, "intermediate_size", default=4 * h)

    if cfg.get("kv_lora_rank"):  # DeepSeek MLA
        nope, rope = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
        v_hd = cfg.get("v_head_dim", nope)
        q_rank = cfg.get("q_lora_rank")
        q = h * q_rank + q_rank * n_heads * (nope + rope) if q_rank else h * n_heads * (nope + rope)
        kv = h * (cfg["kv_lora_rank"] + rope) + cfg["kv_lora_rank"] * n_heads * (nope + v_hd)
        attn = q + kv + n_heads * v_hd * h
    else:
        attn = h * hd * (n_heads + 2 * n_kv) + n_heads * hd * h

    dense_mlp = 3 * h * inter
    n_experts = _get(cfg, "num_local_experts", "num_experts", "n_routed_experts", default=0)
    top_k = _get(cfg, "num_experts_per_tok", "experts_per_token", default=0)

    embed = vocab * h * (1 if cfg.get("tie_word_embeddings", False) else 2)

    if not n_experts:
        total = embed + n_layers * (attn + dense_mlp)
        return total, total

    expert = 3 * h * _get(cfg, "moe_intermediate_size", default=inter)
    shared = (cfg.get("n_shared_experts") or 0) * expert
    if cfg.get("shared_expert_intermediate_size"):
        shared += 3 * h * cfg["shared_expert_intermediate_size"]
    n_moe = _moe_layers(cfg, n_layers)
    moe_layer = n_experts * expert + shared + h * n_experts
    total = embed + n_layers * attn + n_moe * moe_layer + (n_layers - n_moe) * dense_mlp
    active = total - n_moe * (n_experts - top_k) * expert
    return total, active


def spec_from_config(cfg: dict, name: str, exact_params: int | None = None) -> ModelSpec:
    raw = cfg
    cfg = _flatten(cfg)
    try:
        h = cfg["hidden_size"]
        n_layers = cfg["num_hidden_layers"]
        n_heads = cfg["num_attention_heads"]
        vocab = cfg["vocab_size"]
    except KeyError as e:
        raise ModelError(f"{name}: config.json has no {e.args[0]!r}; unsupported architecture") from None

    n_kv = _get(cfg, "num_key_value_heads", default=n_heads)
    hd = _get(cfg, "head_dim", default=h // n_heads)

    if cfg.get("kv_lora_rank"):
        attention = "mla"
        kv_dim = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]  # compressed latent, K and V share it
    else:
        attention = "gqa"
        kv_dim = 2 * n_kv * hd

    full, swa, window = _attention_layout(cfg, n_layers)
    est_total, est_active = count_params(raw)

    total, source = est_total, "config"
    if exact_params:
        total, source = exact_params, "safetensors"
    active = total - (est_total - est_active)

    n_experts = _get(cfg, "num_local_experts", "num_experts", "n_routed_experts", default=0)
    return ModelSpec(
        name=name,
        total_params=total,
        active_params=active,
        n_layers=n_layers,
        hidden_size=h,
        vocab_size=vocab,
        tied_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        kv_dim=kv_dim,
        full_attn_layers=full,
        swa_layers=swa,
        sliding_window=window,
        max_context=_get(cfg, "max_position_embeddings", default=0),
        attention=attention,
        n_experts=n_experts,
        experts_per_token=_get(cfg, "num_experts_per_tok", "experts_per_token", default=0),
        param_source=source,
        native_quant=str((cfg.get("quantization_config") or {}).get("quant_method", "")).lower(),
    )


def resolve(name: str) -> str:
    """Map a short alias to a repo id; pass repo ids and paths through."""
    key = name.lower().strip()
    if key in ALIASES:
        return ALIASES[key]
    if "/" in name or Path(name).exists():
        return name
    close = difflib.get_close_matches(key, ALIASES, n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ModelError(
        f"Unknown model {name!r}.{hint} Use a Hugging Face repo id (org/name) "
        "or see `canirun --list-models`."
    )


def _exact_params(repo: str) -> int | None:
    from huggingface_hub import get_safetensors_metadata

    try:
        meta = get_safetensors_metadata(repo)
    except Exception:
        return None
    counts = meta.parameter_count
    if not counts or PACKED_DTYPES & set(counts):
        return None
    return sum(counts.values())


def _cache_path(repo: str) -> Path:
    return MODEL_CACHE / (repo.replace("/", "--") + ".json")


def load(name: str, refresh: bool = False) -> ModelSpec:
    target = resolve(name)
    path = Path(target)
    if path.exists():
        cfg_path = path / "config.json" if path.is_dir() else path
        return spec_from_config(json.loads(cfg_path.read_text()), path.stem if path.is_file() else path.name)

    cached = _cache_path(target)
    if not refresh:
        try:
            data = json.loads(cached.read_text())
            if data.pop("_cache_version", None) == CACHE_VERSION:
                return ModelSpec(**data)
        except (OSError, ValueError, TypeError):
            pass

    spec = _fetch(target)
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({**asdict(spec), "_cache_version": CACHE_VERSION}))
    except OSError:
        pass
    return spec


def _fetch(target: str) -> ModelSpec:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

    try:
        cfg_file = hf_hub_download(target, "config.json")
    except GatedRepoError:
        raise ModelError(
            f"{target} is gated. Accept its license on huggingface.co and run "
            "`hf auth login`, or use an alias from `canirun --list-models`."
        ) from None
    except RepositoryNotFoundError:
        raise ModelError(f"No Hugging Face repo named {target!r}.") from None
    except EntryNotFoundError:
        raise ModelError(
            f"{target} has no config.json (GGUF-only repos aren't supported yet). "
            "Try the original model's repo."
        ) from None
    except Exception as e:  # network errors etc.
        raise ModelError(f"Couldn't fetch {target}: {e}") from None

    cfg = json.loads(Path(cfg_file).read_text())
    return spec_from_config(cfg, target.split("/")[-1], _exact_params(target))
