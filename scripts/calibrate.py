"""Fit canirun's speed constants to saved --benchmark results.

    python scripts/calibrate.py [results.json ...] [--gguf-dir DIR]

Time per token is linear in the unknowns:

    t = gpu_bytes/gpu_bw * (1/eff_gpu) + cpu_bytes/cpu_bw * (1/eff_cpu)
        + gpu_layers * overhead_gpu + cpu_layers * overhead_cpu

so a weighted least-squares fit (relative error) recovers all four.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from canirun import bench, estimate as est, models
from canirun.hardware import GiB, Hardware, Memory
from canirun.quants import Quant

UNKNOWNS = ["1/eff_gpu", "1/eff_cpu", "overhead_gpu_s", "overhead_cpu_s"]


def load_spec(result: dict, gguf_dir: Path | None):
    source = result.get("source") or result["model"]
    if result["backend"] == "ollama":
        return models.load(source if source.startswith("ollama:") else f"ollama:{source}")
    for candidate in (Path(source), (gguf_dir or Path(".")) / source, (gguf_dir or Path(".")) / f"{result['model']}.gguf"):
        if candidate.is_file():
            return models.load(str(candidate))
    raise FileNotFoundError(f"can't find the GGUF for {result['model']}; pass --gguf-dir")


def rows(results: list[dict], gguf_dir: Path | None):
    for r in results:
        spec = load_spec(r, gguf_dir)
        if spec.is_moe:
            continue  # MoE gets its own factor; fit dense models only
        quant = Quant(spec.file_quant, spec.file_bytes * 8 / spec.total_params, "")
        g, n = r["gpu_layers"], spec.n_layers
        gpu_b, cpu_b = est.bytes_per_token(spec, quant, g, bench.kv_tokens_for(r["backend"] == "ollama"))
        hw = r["hardware"]
        yield r, [gpu_b / (hw.get("gpu_bw_gbs", 1) * 1e9) if g else 0.0,
                  cpu_b / (hw["ram_bw_gbs"] * 1e9), g, n - g], 1 / r["measured_tok_s"]


def nnls_fit(A: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Least squares on relative error, dropping any unknown that goes negative."""
    W = A / t[:, None]
    active = list(range(A.shape[1]))
    while True:
        x = np.zeros(A.shape[1])
        x[active] = np.linalg.lstsq(W[:, active], np.ones(len(t)), rcond=None)[0]
        neg = [i for i in active if x[i] < 0]
        if not neg:
            return x
        active.remove(min(neg, key=lambda i: x[i]))


def _hardware(h: dict) -> Hardware:
    ram = Memory("RAM", h["ram_gb"] * GiB, h["ram_gb"] * GiB, h["ram_bw_gbs"], "cpu", "measured")
    gpu = Memory(h["gpu"], h["vram_gb"] * GiB, h["vram_gb"] * GiB, h["gpu_bw_gbs"], "cuda") if h.get("gpu") else None
    return Hardware(cpu_name=h["cpu"], ram=ram, gpu=gpu, unified=h.get("unified", False))


def check(data, gguf_dir) -> int:
    """Markdown table: every saved result vs. what the current constants predict."""
    print("| model | quant | GPU layers | backend | predicted | measured | error |")
    print("|---|---|---|---|---|---|---|")
    errors = []
    for r, _, _ in data:
        spec = load_spec(r, gguf_dir)
        quant = Quant(spec.file_quant, spec.file_bytes * 8 / spec.total_params, "")
        pred = est.decode_speed(spec, _hardware(r["hardware"]), quant, r["gpu_layers"],
                                bench.kv_tokens_for(r["backend"] == "ollama"))
        err = (pred / r["measured_tok_s"] - 1) * 100
        errors.append(abs(err))
        print(f"| {r['model']} | {r['quant']} | {r['gpu_layers']}/{r['n_layers']} | {r['backend']} "
              f"| {pred:.1f} | {r['measured_tok_s']:.1f} | {err:+.0f}% |")
    print(f"\nmedian |error| {np.median(errors):.0f}%, max {max(errors):.0f}% over {len(errors)} runs")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", type=Path)
    p.add_argument("--gguf-dir", type=Path)
    p.add_argument("--check", action="store_true", help="score the current constants instead of fitting")
    args = p.parse_args()
    files = args.files or sorted(bench.RESULTS_DIR.glob("*.json"))
    results = [json.loads(f.read_text()) for f in files]
    data = list(rows(results, args.gguf_dir))
    if len(data) < 4:
        print(f"need at least 4 dense-model results, have {len(data)}", file=sys.stderr)
        return 1

    if args.check:
        return check(data, args.gguf_dir)

    A = np.array([d[1] for d in data])
    t = np.array([d[2] for d in data])
    x = nnls_fit(A, t)
    print("fit:")
    print(f"  EFFICIENCY cuda = {1 / x[0]:.3f}" if x[0] else "  EFFICIENCY cuda = (no data)")
    print(f"  EFFICIENCY cpu  = {1 / x[1]:.3f}" if x[1] else "  EFFICIENCY cpu  = (no data)")
    print(f"  LAYER_OVERHEAD cuda = {x[2] * 1e6:.1f} us,  cpu = {x[3] * 1e6:.1f} us")
    print("\nper result (measured vs fitted):")
    for (r, row, tm), tf in zip(data, A @ x):
        print(f"  {r['model'][:22]:22} {r['gpu_layers']:>3}/{r['n_layers']:<3} "
              f"{r['measured_tok_s']:7.1f} vs {1 / tf:7.1f} tok/s  ({(tm / tf - 1) * 100:+.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
