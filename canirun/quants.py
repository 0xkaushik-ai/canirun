"""Quantization formats and their effective bits per weight.

Bits per weight are measured from real llama.cpp GGUF files (Llama-3-8B), so
they include the higher-precision tensors (embeddings, output) each mix keeps.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quant:
    name: str
    bits_per_weight: float
    quality: str

    @property
    def bytes_per_weight(self) -> float:
        return self.bits_per_weight / 8


QUANTS: list[Quant] = [
    Quant("F16", 16.0, "lossless"),
    Quant("Q8_0", 8.50, "near-lossless"),
    Quant("Q6_K", 6.57, "excellent"),
    Quant("Q5_K_M", 5.70, "very good"),
    Quant("Q4_K_M", 4.89, "good"),
    Quant("Q3_K_M", 4.00, "lossy"),
    Quant("Q2_K", 3.17, "very lossy"),
]

QUANTS_BY_NAME = {q.name.lower(): q for q in QUANTS}

# Models released pre-quantized. Their GGUFs keep the native format for the
# experts, so smaller K-quants barely shrink them; show the native file instead.
# (gpt-oss-20b's MXFP4 GGUF is 12.1 GB for 20.9B params.)
NATIVE_QUANTS = {
    "mxfp4": [QUANTS[0], QUANTS[1], Quant("MXFP4", 4.63, "native")],
}


# Labels for formats a GGUF file may use that aren't in the table above.
OTHER_QUALITY = {
    "bf16": "lossless", "f32": "lossless", "q5_k_s": "very good", "q5_0": "very good",
    "q4_k_s": "good", "q4_0": "good", "q4_1": "good", "iq4_xs": "good", "iq4_nl": "good",
    "mxfp4": "native", "q3_k_l": "lossy", "q3_k_s": "lossy", "iq3_m": "lossy", "iq3_s": "lossy",
    "iq3_xs": "lossy", "iq3_xxs": "lossy",
}


def quants_for(native_quant: str, file_quant: str = "", file_bpw: float = 0.0) -> list[Quant]:
    """Quants to show. A specific GGUF file replaces its row with its exact size."""
    quants = NATIVE_QUANTS.get(native_quant, QUANTS)
    if not (file_quant and file_bpw):
        return quants
    known = QUANTS_BY_NAME.get(file_quant.lower())
    quality = known.quality if known else OTHER_QUALITY.get(file_quant.lower(), "very lossy")
    mine = Quant(file_quant, file_bpw, quality)
    rest = [q for q in quants if q.name.lower() != file_quant.lower()]
    return sorted([*rest, mine], key=lambda q: -q.bits_per_weight)

# Bytes per element of the KV cache (llama.cpp --cache-type-k/v).
# q8_0 and q4_0 store one fp16 scale per block of 32 values.
KV_BYTES = {
    "f16": 2.0,
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
}
