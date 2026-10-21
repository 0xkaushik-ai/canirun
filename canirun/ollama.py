"""Minimal Ollama API client (stdlib only)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .gguf import spec_from_gguf
from .models import ModelError, ModelSpec

PREFIX = "ollama:"


def host() -> str:
    h = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    return h if h.startswith("http") else f"http://{h}"


def _post(path: str, body: dict, timeout: float = 30) -> dict:
    req = urllib.request.Request(
        host() + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except ValueError:
            pass
        raise ModelError(f"Ollama: {detail}") from None
    except (urllib.error.URLError, OSError) as e:
        raise ModelError(f"Can't reach Ollama at {host()} ({e}). Is `ollama serve` running?") from None


def _get(path: str) -> dict:
    try:
        with urllib.request.urlopen(host() + path, timeout=10) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError) as e:
        raise ModelError(f"Can't reach Ollama at {host()} ({e}). Is `ollama serve` running?") from None


def _full_name(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def load(model: str) -> ModelSpec:
    info = _post("/api/show", {"model": model})
    if info.get("details", {}).get("format") not in (None, "gguf"):
        raise ModelError(f"Ollama model {model} isn't GGUF")
    tensors = [(t["name"], t.get("shape", [])) for t in info.get("tensors") or []]
    sizes = {m["name"]: m["size"] for m in _get("/api/tags").get("models", [])}
    size = sizes.get(_full_name(model), 0)
    return spec_from_gguf(info.get("model_info", {}), tensors, model, size)


def generate(model: str, prompt: str, num_predict: int, num_ctx: int, num_gpu: int | None) -> dict:
    options = {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0, "seed": 1}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    return _post("/api/generate", {"model": model, "prompt": prompt, "stream": False,
                                   "options": options, "keep_alive": "2m"}, timeout=600)


def unload_all(timeout: float = 15) -> None:
    """Free the GPU before measuring it: unload every model Ollama has resident."""
    import time

    for m in _get("/api/ps").get("models", []):
        unload(m["name"])
    deadline = time.time() + timeout
    while _get("/api/ps").get("models") and time.time() < deadline:
        time.sleep(0.5)
    time.sleep(1.5)  # the runner process releases VRAM shortly after it leaves /api/ps


def unload(model: str) -> None:
    try:
        _post("/api/generate", {"model": model, "keep_alive": 0})
    except ModelError:
        pass
