"""--benchmark: run the model for real and compare with the prediction."""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__, estimate as est, ollama
from .hardware import GiB, Hardware
from .models import ModelError, ModelSpec
from .quants import Quant

ISSUES_URL = "https://github.com/0xkaushik-ai/canirun/issues/new"
RESULTS_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "canirun" / "bench"
N_GEN = 128
N_PROMPT = 512
REPEATS = 3
PROMPT = "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle."


@dataclass
class BenchResult:
    backend: str
    backend_version: str
    model: str
    source: str                 # what was passed on the command line
    quant: str
    file_gb: float
    n_layers: int
    gpu_layers: int
    predicted_tok_s: float
    measured_tok_s: float
    prompt_tok_s: float | None
    hardware: dict
    canirun_version: str = __version__
    notes: list[str] = field(default_factory=list)

    @property
    def error_pct(self) -> float:
        return (self.predicted_tok_s / self.measured_tok_s - 1) * 100


def kv_tokens_for(is_ollama: bool) -> int:
    """Average KV fill while decoding: llama-bench starts empty, Ollama after the prompt."""
    return N_GEN // 2 + (64 if is_ollama else 0)


def find_llama_bench(explicit: str | None) -> str | None:
    for candidate in (explicit, os.environ.get("LLAMA_BENCH"), shutil.which("llama-bench")):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _hardware_summary(hw: Hardware, cpu: str) -> dict:
    d = {"cpu": cpu, "ram_gb": round(hw.ram.total_bytes / GiB, 1), "ram_bw_gbs": round(hw.ram.bandwidth_gbs, 1)}
    if hw.gpu:
        d.update(gpu=hw.gpu.name, vram_gb=round(hw.gpu.total_bytes / GiB, 1),
                 gpu_bw_gbs=round(hw.gpu.bandwidth_gbs, 1), unified=hw.unified)
    return d


def _plan_layers(spec: ModelSpec, hw: Hardware, quant: Quant, ctx: int, ngl: int | None) -> int:
    if ngl is not None:
        return min(ngl, spec.n_layers)
    e = est.estimate(spec, hw, quant, ctx)
    if not e.fits:
        raise ModelError(f"canirun predicts {spec.name} won't fit in memory here, so there's nothing to benchmark.")
    return e.gpu_layers


def _llama_bench(binary: str, path: str, ngl_all: bool, layers: int) -> tuple[float, float | None, str]:
    cmd = [binary, "-m", path, "-ngl", str(99 if ngl_all else layers),
           "-p", str(N_PROMPT), "-n", str(N_GEN), "-r", str(REPEATS), "-o", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise RuntimeError("\n".join(tail))
    rows = json.loads(proc.stdout)
    tg = next(r for r in rows if r.get("n_gen", 0) > 0)
    pp = next((r for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0), None)
    version = f"llama.cpp {tg.get('build_commit', '?')}"
    return tg["avg_ts"], (pp["avg_ts"] if pp else None), version


def _ollama_bench(model: str, ngl_all: bool, layers: int) -> tuple[float, float | None, str]:
    # Ollama counts the output layer too, so "all" is n_layers + 1.
    num_gpu = 999 if ngl_all else layers
    ollama.unload(model)
    ollama.generate(model, "Hi", 1, 2048, num_gpu)  # load + warm up
    decode, prompt = [], []
    for _ in range(REPEATS):
        r = ollama.generate(model, PROMPT, N_GEN, 2048, num_gpu)
        if r.get("eval_duration"):
            decode.append(r["eval_count"] / (r["eval_duration"] / 1e9))
        if r.get("prompt_eval_duration") and r.get("prompt_eval_count", 0) > 8:
            prompt.append(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9))
    ollama.unload(model)
    if not decode:
        raise RuntimeError("Ollama returned no timing data")
    version = "ollama " + ollama._get("/api/version").get("version", "?")
    return statistics.median(decode), (statistics.median(prompt) if prompt else None), version


def run(spec: ModelSpec, hw: Hardware, target: str, cpu: str, ngl: int | None = None,
        llama_bench: str | None = None, progress=lambda msg: None) -> BenchResult:
    if not spec.file_quant or not spec.file_bytes:
        raise ModelError("--benchmark needs a real model file: a .gguf path or ollama:<model>.")
    quant = Quant(spec.file_quant, spec.file_bytes * 8 / spec.total_params, "")
    is_ollama = target.lower().startswith(ollama.PREFIX)

    if not is_ollama:
        binary = find_llama_bench(llama_bench)
        if not binary:
            raise ModelError("llama-bench not found. Put it on PATH, set LLAMA_BENCH, or pass --llama-bench PATH.")

    layers = _plan_layers(spec, hw, quant, N_PROMPT + N_GEN, ngl)
    notes = []
    while True:
        on_gpu_fully = layers >= spec.n_layers and hw.gpu is not None
        progress(f"Running {spec.name} with {layers}/{spec.n_layers} layers on the GPU…")
        try:
            if is_ollama:
                measured, prompt, version = _ollama_bench(target[len(ollama.PREFIX):], on_gpu_fully, layers)
            else:
                measured, prompt, version = _llama_bench(binary, target, on_gpu_fully, layers)
            break
        except (RuntimeError, ModelError) as e:
            # The predicted split didn't actually fit: note it and back off.
            if layers <= 0 or "memory" not in str(e).lower():
                raise ModelError(f"Benchmark failed: {e}") from None
            notes.append(f"out of memory at {layers} GPU layers; prediction was too optimistic")
            layers = max(layers - 2, 0)

    kv_tokens = kv_tokens_for(is_ollama)
    predicted = est.decode_speed(spec, hw, quant, layers, kv_tokens)
    return BenchResult(
        backend="ollama" if is_ollama else "llama-bench",
        backend_version=version,
        model=spec.name,
        source=target if is_ollama else Path(target).name,
        quant=spec.file_quant,
        file_gb=round(spec.file_bytes / GiB, 2),
        n_layers=spec.n_layers,
        gpu_layers=layers,
        predicted_tok_s=round(predicted, 1),
        measured_tok_s=round(measured, 1),
        prompt_tok_s=round(prompt, 1) if prompt else None,
        hardware=_hardware_summary(hw, cpu),
        notes=notes,
    )


def save(result: BenchResult) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in result.model)
    path = RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}-{result.quant}.json"
    path.write_text(json.dumps(asdict(result), indent=2))
    return path


def share_url(result: BenchResult) -> str:
    """Pre-filled GitHub issue so people can contribute calibration data in one click."""
    hw = result.hardware
    gpu = f"{hw['gpu']} ({hw['vram_gb']} GB, {hw['gpu_bw_gbs']} GB/s)" if hw.get("gpu") else "none"
    title = f"Benchmark: {result.model} {result.quant} on {hw.get('gpu') or hw['cpu']}"
    body = "\n".join([
        f"**GPU:** {gpu}",
        f"**CPU / RAM:** {hw['cpu']} · {hw['ram_gb']} GB @ {hw['ram_bw_gbs']} GB/s",
        f"**Model:** {result.model} {result.quant} ({result.file_gb} GB), "
        f"{result.gpu_layers}/{result.n_layers} layers on GPU",
        f"**Backend:** {result.backend_version} · canirun {result.canirun_version}",
        "",
        "| | predicted | measured | error |",
        "|---|---|---|---|",
        f"| decode | {result.predicted_tok_s} tok/s | {result.measured_tok_s} tok/s | {result.error_pct:+.0f}% |",
        "",
        "```json",
        json.dumps(asdict(result), separators=(",", ":")),
        "```",
    ])
    return f"{ISSUES_URL}?{urllib.parse.urlencode({'title': title, 'body': body, 'labels': 'benchmark'})}"
