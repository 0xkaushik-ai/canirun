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


def quants_for(native_quant: str) -> list[Quant]:
    return NATIVE_QUANTS.get(native_quant, QUANTS)

# Bytes per element of the KV cache (llama.cpp --cache-type-k/v).
# q8_0 and q4_0 store one fp16 scale per block of 32 values.
KV_BYTES = {
    "f16": 2.0,
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
}
