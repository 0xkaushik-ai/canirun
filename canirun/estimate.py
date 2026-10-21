"""Memory and speed estimates. Pure functions: no I/O, easy to test.

Memory = weights + KV cache + runtime overhead.
Speed  = decode is memory-bound: every token streams the active weights and
         the KV cache once, so tok/s ~= efficiency * bandwidth / bytes_read.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .hardware import Hardware
from .models import ModelSpec
from .quants import QUANTS, Quant

GiB = 1024**3

# Fraction of theoretical bandwidth llama.cpp reaches during decode.
# Starting points from public benchmarks; v0.2 calibrates them with --benchmark.
EFFICIENCY = {"cuda": 0.70, "metal": 0.70, "cpu": 0.75}
# MoE decode reaches less of it: routing plus many small expert matmuls.
# Uncalibrated guess from public Qwen3-30B-A3B / gpt-oss runs.
MOE_EFFICIENCY = 0.6

RUNTIME_OVERHEAD = 0.5 * GiB    # CUDA/Metal context, allocator slack
UBATCH = 512                    # llama.cpp default micro-batch for prompt processing
SPEED_CTX_FRACTION = 0.5        # speed is quoted at a half-full context
MIN_READABLE_TOK_S = 8.0        # roughly reading speed
FAST_ENOUGH_TOK_S = 30.0        # beyond this, prefer quality over speed
SPEED_TRADEOFF = 0.6            # accept down to 60% of the best Q4+ speed for quality
OS_RESERVE = 2.5 * GiB          # RAM the OS and a desktop need even with apps closed


@dataclass
class Estimate:
    quant: Quant
    weights_bytes: float
    kv_bytes: float
    overhead_bytes: float
    placement: str              # "gpu" | "split" | "cpu" | "none"
    gpu_layers: int
    n_layers: int
    tok_s: float | None
    gpu_bytes: float = 0.0
    cpu_bytes: float = 0.0
    needs_free_ram: bool = False    # only fits after closing other apps

    @property
    def total_bytes(self) -> float:
        return self.weights_bytes + self.kv_bytes + self.overhead_bytes

    @property
    def fits(self) -> bool:
        return self.placement != "none"


def kv_cache_bytes(spec: ModelSpec, ctx: int, kv_bytes_per_elem: float = 2.0) -> float:
    """2 (K and V) x layers x kv_heads x head_dim x bytes x tokens, generalised.

    kv_dim already includes the K/V factor (or the shared MLA latent), and
    sliding-window layers never hold more than `sliding_window` tokens.
    """
    swa_tokens = min(ctx, spec.sliding_window) if spec.sliding_window else ctx
    tokens = spec.full_attn_layers * ctx + spec.swa_layers * swa_tokens
    return spec.kv_dim * kv_bytes_per_elem * tokens


def overhead_bytes(spec: ModelSpec) -> float:
    # compute buffer is dominated by the f32 logits for one micro-batch
    compute = UBATCH * (spec.vocab_size + 4 * spec.hidden_size) * 4
    return RUNTIME_OVERHEAD + compute


def _split_weights(spec: ModelSpec, params: int, bpw: float) -> tuple[float, float]:
    """Bytes in the repeating transformer layers vs. the embedding/output head."""
    head = spec.embed_params * bpw
    n_embed_tensors = 1 if spec.tied_embeddings else 2
    layers = max(params - n_embed_tensors * spec.embed_params, 0) * bpw
    return layers, head * n_embed_tensors


def estimate(spec: ModelSpec, hw: Hardware, quant: Quant, ctx: int, kv_bytes_per_elem: float = 2.0) -> Estimate:
    bpw = quant.bytes_per_weight
    weights = spec.total_params * bpw
    kv = kv_cache_bytes(spec, ctx, kv_bytes_per_elem)
    overhead = overhead_bytes(spec)
    total = weights + kv + overhead

    # Bytes streamed per generated token. The input embedding is a single-row
    # lookup, but the output head reads its whole matrix.
    layer_read, _ = _split_weights(spec, spec.active_params, bpw)
    head_read = spec.embed_params * bpw
    kv_read = kv_cache_bytes(spec, max(int(ctx * SPEED_CTX_FRACTION), 1), kv_bytes_per_elem)

    def tok_s(gpu_read: float, cpu_read: float) -> float:
        t = 0.0
        if gpu_read:
            t += gpu_read / (hw.gpu.bandwidth_gbs * 1e9 * EFFICIENCY[hw.gpu.kind])
        if cpu_read:
            t += cpu_read / (hw.ram.bandwidth_gbs * 1e9 * EFFICIENCY["cpu"])
        if not t:
            return 0.0
        return (MOE_EFFICIENCY if spec.is_moe else 1.0) / t

    n = spec.n_layers
    base = dict(quant=quant, weights_bytes=weights, kv_bytes=kv, overhead_bytes=overhead, n_layers=n)
    gpu = hw.gpu if hw.gpu and hw.gpu.bandwidth_gbs > 0 else None

    # 1. Everything on the GPU.
    if gpu and total <= gpu.usable_bytes:
        return Estimate(**base, placement="gpu", gpu_layers=n,
                        tok_s=tok_s(layer_read + head_read + kv_read, 0), gpu_bytes=total)

    # 2. Split: as many layers as fit on the GPU, the rest (and the
    #    embedding/output head) in system RAM. Not possible with unified memory.
    if gpu and not hw.unified:
        layer_store, head_store = _split_weights(spec, spec.total_params, bpw)
        per_layer = (layer_store + kv) / n
        budget = gpu.usable_bytes - overhead
        g = max(0, min(n - 1, int(budget // per_layer))) if budget > 0 else 0
        cpu_need = (n - g) * per_layer + head_store
        if g > 0 and cpu_need <= hw.ram.usable_bytes:
            frac = g / n
            speed = tok_s(
                frac * (layer_read + kv_read),
                (1 - frac) * (layer_read + kv_read) + head_read,
            )
            return Estimate(**base, placement="split", gpu_layers=g, tok_s=speed,
                            gpu_bytes=g * per_layer + overhead, cpu_bytes=cpu_need)

    # 3. CPU only (on unified memory the GPU limit above is the whole story).
    if not hw.unified and total <= hw.ram.usable_bytes:
        return Estimate(**base, placement="cpu", gpu_layers=0,
                        tok_s=tok_s(0, layer_read + head_read + kv_read), cpu_bytes=total)

    return Estimate(**base, placement="none", gpu_layers=0, tok_s=None)


def estimate_all(spec: ModelSpec, hw: Hardware, ctx: int, kv_bytes_per_elem: float = 2.0,
                 quants: list[Quant] = QUANTS) -> list[Estimate]:
    results = [estimate(spec, hw, q, ctx, kv_bytes_per_elem) for q in quants]

    # "Free RAM" moves with whatever else is open; re-check against total RAM
    # minus what the OS itself needs, and flag those as "close other apps".
    roomy_bytes = hw.ram.total_bytes - OS_RESERVE
    if not hw.unified and roomy_bytes > hw.ram.usable_bytes:
        roomy = replace(hw, ram=replace(hw.ram, usable_bytes=roomy_bytes))
        for i, e in enumerate(results):
            if not e.fits:
                retry = estimate(spec, roomy, e.quant, ctx, kv_bytes_per_elem)
                if retry.fits:
                    retry.needs_free_ram = True
                    results[i] = retry
    return results


def recommend(estimates: list[Estimate], min_tok_s: float = MIN_READABLE_TOK_S) -> Estimate | None:
    """Best quality that is fast enough.

    Fully on the GPU beats any GPU/CPU split, when a Q4-or-better quant fits.

    "Fast enough" means reading speed, and not giving up more than
    SPEED_TRADEOFF of the speed the fastest Q4-or-better quant reaches (a
    fully-on-GPU Q4 beats a half-offloaded Q6) — unless it is already past
    FAST_ENOUGH_TOK_S, where more speed stops mattering.
    F16 is never recommended: Q8_0 is near-lossless at half the size.
    Q4_K_M is the quality floor: lossier quants are only suggested when
    nothing Q4 or better fits. If nothing is fast enough, take the fastest.
    """
    fitting = [e for e in estimates if e.fits and e.quant.name != "F16"]
    if not fitting:
        return None
    decent = [e for e in fitting if e.quant.bits_per_weight >= 4.5]
    on_gpu = [e for e in decent if e.placement == "gpu"]
    pool = on_gpu or decent or fitting
    best_speed = max(e.tok_s for e in pool)
    threshold = max(min_tok_s, min(SPEED_TRADEOFF * best_speed, FAST_ENOUGH_TOK_S))
    fast = [e for e in pool if e.tok_s >= threshold]
    if fast:
        return max(fast, key=lambda e: e.quant.bits_per_weight)
    return max(pool, key=lambda e: e.tok_s)
