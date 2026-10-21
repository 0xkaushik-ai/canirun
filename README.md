# canirun

**Can your machine run that LLM?** One command tells you whether it fits, which quant to use, and how fast it will be.

```
$ canirun llama3-8b

llama-3-8b-Instruct  8.0B params · ctx 8,192
GPU  RTX 3050 Laptop · 3.8 GB free @ 192 GB/s
RAM  15.3 GB · 11.3 GB free @ 46 GB/s

Quant     Needs  Runs on             Speed  Quality
F16     16.7 GB  ✗ won't fit             —  lossless
Q8_0     9.7 GB  ◐ 12/32 on GPU   ~7 tok/s  near-lossless
Q6_K     7.9 GB  ◐ 15/32 on GPU  ~10 tok/s  excellent      ← recommended
Q5_K_M   7.1 GB  ◐ 17/32 on GPU  ~12 tok/s  very good
Q4_K_M   6.3 GB  ◐ 19/32 on GPU  ~14 tok/s  good
Q3_K_M   5.5 GB  ◐ 22/32 on GPU  ~19 tok/s  lossy
Q2_K     4.7 GB  ◐ 27/32 on GPU  ~29 tok/s  very lossy

→ Q6_K runs with 15/32 layers on the GPU (llama.cpp -ngl 15) at ~10 tok/s.
```

It detects your GPU, VRAM, RAM and memory bandwidth, reads the model's architecture from Hugging Face, and works out where every layer goes, including **GPU + CPU splits** for models that don't fit in VRAM.

## Install

```bash
pipx install git+https://github.com/0xkaushik-ai/canirun   # PyPI release coming: pipx install canirun-llm
```

## Usage

```bash
canirun qwen3-30b-a3b                       # built-in alias
canirun mistralai/Devstral-Small-2505       # any Hugging Face repo
canirun llama3-8b --ctx 32768 --kv-quant q8_0
canirun ./my-model/config.json              # local config, works offline
canirun ./Qwen3-8B-Q4_K_M.gguf              # a GGUF file: exact size, header only
canirun ollama:llama3.2                     # a model you've pulled in Ollama

# What-if: hardware you don't have
canirun qwen3-32b --gpu 4090 --ram 32
canirun llama3.3-70b --gpu "m4 max" --ram 64
canirun gemma3-27b --vram 16 --gpu-bw 640

canirun --list-models     # aliases
canirun --list-gpus       # GPU and Mac presets
canirun llama3-8b --json  # for scripts
```

## Check it against reality: `--benchmark`

```
$ canirun ollama:phi3:mini --benchmark

phi3:mini Q4_0 · 2.03 GB · 32/32 layers on GPU · ollama 0.15.4
           predicted       measured    error
decode    70.0 tok/s     74.2 tok/s      -6%
prompt             —    2,286 tok/s
```

`--benchmark` runs the model at the GPU layer count canirun predicts and compares the result with the prediction. It works with any Ollama model (`ollama:<name>`), or with a `.gguf` file plus [llama-bench](https://github.com/ggml-org/llama.cpp) on your PATH (or `--llama-bench PATH`). Use `--ngl N` to test a specific split. If the predicted split runs out of memory, canirun steps down and reports it, because that means the fit prediction was wrong.

**Please share your result.** canirun prints a link that opens a pre-filled GitHub issue (or use `--share`). Every result from new hardware makes the estimates better for everyone.

## How it works

**Memory = weights + KV cache + overhead**

- **Weights** = parameters × bits per weight ÷ 8. An 8B model at Q4_K_M (4.89 bits) is 4.9 GB.
- **KV cache** = 2 × layers × KV heads × head dim × bytes × context. For Llama 3 8B at fp16 that's 128 KB per token, so 1 GB at 8K context. canirun also handles:
  - **GQA**: only the KV heads count, not the query heads.
  - **Sliding-window layers** (Gemma 2/3, gpt-oss) only cache the window.
  - **MLA** (DeepSeek V3/R1) caches a compressed 576-value latent per layer, which makes the cache about 30× smaller.
  - **Linear-attention layers** (Qwen3-Next) keep no cache.
- **Overhead**: about 0.5 GB of runtime plus llama.cpp's compute buffer.

**Speed: decode is memory-bound.** Each new token streams all the active weights and the KV cache through memory once:

```
tok/s ≈ efficiency × bandwidth ÷ bytes read per token
```

A 4090 (1008 GB/s) reading a 5 GB model has a ceiling of about 200 tok/s. llama.cpp reaches about 88% of the bandwidth, plus a fixed ~45 µs per layer per token, which lands at about 150 tok/s.

**GPU + CPU split**: when the model doesn't fit in VRAM, canirun puts as many layers on the GPU as fit and the rest in RAM. Time per token is `gpu_bytes/gpu_bw + cpu_bytes/cpu_bw`, so the slow RAM part dominates. That's why going from 17 to 19 layers on the GPU matters.

**MoE models** (Qwen3-30B-A3B, gpt-oss, DeepSeek) need memory for *all* experts, but each token reads only the *active* ones. A 30B MoE with 3B active needs 30B worth of RAM but runs close to 3B speed.

**Recommendation**:
1. A quant that fits fully on the GPU beats any GPU/CPU split.
2. Within that, it picks the best quality that still runs at reading speed (8 tok/s or more) without giving up more than 40% of the speed.
3. Q4_K_M is the quality floor, and F16 is never recommended because Q8_0 is near-lossless at half the size.

## Accuracy

The speed constants are fitted to real `--benchmark` runs with [`scripts/calibrate.py`](scripts/calibrate.py). Time per token is linear in the unknowns: 1/efficiency for the GPU and the CPU, plus a fixed cost per layer. So least squares recovers them from a handful of runs at different GPU/CPU splits.

| Fitted on | GPU efficiency | per-layer overhead | CPU efficiency* |
|---|---|---|---|
| RTX 3050 Laptop + i7-11800H, 13 runs (Ollama + llama-bench) | 0.88 | 45 µs | 0.95 |

\*CPU efficiency is relative to canirun's own RAM test, which reads about 89% of spec.

The median error is **6%** on those runs; see [docs/calibration.md](docs/calibration.md). As an independent check, the same constants predict **148 tok/s** for Llama 3 8B Q4_K_M on an RTX 4090. Public llama-bench results are around 150.

Known gaps:
- **Not measured yet:** Apple Silicon and MoE models. Their factors come from public numbers until someone shares a benchmark.
- **Models under 1B** split across the GPU and CPU run up to 22% faster than predicted.
- **Not supported yet:** AMD and Intel GPU detection (use `--vram` and `--gpu-bw`), multi-GPU tensor parallelism, smarter MoE expert offload (`-ot` / `--n-cpu-moe`), and prompt-speed prediction.

## Roadmap

- [x] v0.1: hardware detection, HF config parsing, GQA/MoE/MLA/SWA handling, GPU+CPU split, what-if presets
- [x] v0.2: `--benchmark` (llama.cpp or Ollama), GGUF and Ollama inputs, calibrated speed model
- [ ] v0.2.x: calibration from community results: Apple Silicon, MoE, more GPUs
- [ ] v0.3: a web version with a page for every model × GPU

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

MIT licensed.
