# canirun

**Can your machine run that LLM?** One command tells you whether it fits, which quant to use, and how fast it will be.

```
$ canirun llama3-8b

llama-3-8b-Instruct  8.0B params · ctx 8,192
GPU  RTX 3050 Laptop · 3.8 GB free @ 192 GB/s
RAM  15.3 GB · 7.2 GB free @ 42 GB/s

Quant     Needs  Runs on             Speed  Quality
F16     16.7 GB  ✗ won't fit             —  lossless
Q8_0     9.7 GB  ◐ 12/32 on GPU   ~5 tok/s  near-lossless
Q6_K     7.9 GB  ◐ 15/32 on GPU   ~7 tok/s  excellent
Q5_K_M   7.1 GB  ◐ 17/32 on GPU   ~9 tok/s  very good      ← recommended
Q4_K_M   6.3 GB  ◐ 19/32 on GPU  ~11 tok/s  good
Q3_K_M   5.5 GB  ◐ 22/32 on GPU  ~14 tok/s  lossy
Q2_K     4.7 GB  ◐ 27/32 on GPU  ~23 tok/s  very lossy

→ Q5_K_M runs with 17/32 layers on the GPU (llama.cpp -ngl 17) at ~9 tok/s.
```

It detects your GPU, VRAM, RAM and memory bandwidth, reads the model's architecture from Hugging Face, and works out where every layer goes, including **GPU + CPU splits** for models that don't fit in VRAM.

## Install

```bash
pipx install canirun-llm     # or: pip install canirun-llm
```

## Usage

```bash
canirun qwen3-30b-a3b                       # built-in alias
canirun mistralai/Devstral-Small-2505       # any Hugging Face repo
canirun llama3-8b --ctx 32768 --kv-quant q8_0
canirun ./my-model/config.json              # local config, works offline

# What-if: hardware you don't have
canirun qwen3-32b --gpu 4090 --ram 32
canirun llama3.3-70b --gpu "m4 max" --ram 64
canirun gemma3-27b --vram 16 --gpu-bw 640

canirun --list-models     # aliases
canirun --list-gpus       # GPU and Mac presets
canirun llama3-8b --json  # for scripts
```

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

A 4090 (1008 GB/s) reading a 5 GB model has a ceiling of about 200 tok/s, and llama.cpp reaches about 70% of that.

**GPU + CPU split**: when the model doesn't fit in VRAM, canirun puts as many layers on the GPU as fit and the rest in RAM. Time per token is `gpu_bytes/gpu_bw + cpu_bytes/cpu_bw`, so the slow RAM part dominates. That's why going from 17 to 19 layers on the GPU matters.

**MoE models** (Qwen3-30B-A3B, gpt-oss, DeepSeek) need memory for *all* experts, but each token reads only the *active* ones. A 30B MoE with 3B active needs 30B worth of RAM but runs close to 3B speed.

**Recommendation**:
1. A quant that fits fully on the GPU beats any GPU/CPU split.
2. Within that, it picks the best quality that still runs at reading speed (8 tok/s or more) without giving up more than 40% of the speed.
3. Q4_K_M is the quality floor, and F16 is never recommended because Q8_0 is near-lossless at half the size.

## Accuracy

These are estimates, not measurements. The efficiency factors come from public llama.cpp benchmarks and haven't been calibrated per machine yet; that's the next release. If your real numbers differ, please [open an issue](../../issues) with your hardware, the model, and the `llama-bench` output. That data is how the estimates improve.

Known gaps: AMD and Intel GPU detection (use `--vram` and `--gpu-bw`), multi-GPU tensor parallelism, smarter MoE expert offload (`-ot` / `--n-cpu-moe`), and prompt-processing speed.

## Roadmap

- [x] v0.1: hardware detection, HF config parsing, GQA/MoE/MLA/SWA handling, GPU+CPU split, what-if presets
- [ ] v0.2: `--benchmark` runs llama.cpp and compares predicted vs. real speed; community calibration data
- [ ] v0.3: a web version with a page for every model × GPU

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

MIT licensed.
