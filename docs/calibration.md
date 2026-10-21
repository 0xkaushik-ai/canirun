# Calibration data

Every `canirun --benchmark` run, compared with what the current constants predict.
Regenerate with:

```bash
python scripts/calibrate.py --check          # score the current constants
python scripts/calibrate.py                  # refit them
```

## RTX 3050 Laptop (4 GB, 192 GB/s) · i7-11800H · 16 GB DDR4-3200 (46 GB/s measured)

Ollama 0.15.4 and llama.cpp `d454a20` (CUDA 12.4). 128 generated tokens, median of 3 runs. These runs
were also used for the fit, so this shows how well the model fits, not how well it predicts other
machines. For that, see the 4090 check in the README.

| model | quant | GPU layers | backend | predicted | measured | error |
|---|---|---|---|---|---|---|
| llama3.2 | Q4_K_M | 28/28 | ollama | 75.2 | 74.8 | +1% |
| qwen3-0.6b-q4_k_m | Q4_K_M | 28/28 | llama-bench | 273.9 | 247.4 | +11% |
| qwen3-0.6b-q4_k_m | Q4_K_M | 14/28 | llama-bench | 126.4 | 162.3 | -22% |
| qwen3-0.6b-q4_k_m | Q4_K_M | 0/28 | llama-bench | 98.3 | 93.7 | +5% |
| llama3.2 | Q4_K_M | 28/28 | ollama | 75.2 | 74.8 | +1% |
| llama3.2 | Q4_K_M | 14/28 | ollama | 28.9 | 28.4 | +2% |
| llama3.2 | Q4_K_M | 0/28 | ollama | 19.5 | 19.8 | -1% |
| phi3:mini | Q4_0 | 32/32 | ollama | 70.0 | 74.2 | -6% |
| phi3:mini | Q4_0 | 16/32 | ollama | 28.6 | 26.6 | +7% |
| phi3:mini | Q4_0 | 0/32 | ollama | 18.3 | 15.0 | +22% |
| llama3.2 | Q4_K_M | 28/28 | ollama | 75.2 | 74.9 | +0% |
| phi3:mini | Q4_0 | 32/32 | ollama | 70.0 | 74.2 | -6% |
| qwen3-0.6b-q4_k_m | Q4_K_M | 20/28 | llama-bench | 152.8 | 195.3 | -22% |

median |error| 6%, max 22% over 13 runs

What the fit shows:

- **Fully on the GPU,** 3B models land within ±6%. A pure bandwidth model needs a per-layer cost
  (about 45 µs per layer per token: kernel launches and syncs) to explain why a 0.6B model tops out
  at 250 tok/s rather than 400.
- **Splits are dominated by the CPU side.** llama3.2 at 14/28 layers on the GPU runs at 28 tok/s,
  not halfway between 75 (all GPU) and 20 (all CPU). Half the layers on a 4.6× slower memory take
  most of the time.
- **Tiny models split badly predicted** (−22%): the output head is a quarter of a 0.6B model, and
  how llama.cpp handles it in a split matters more than the model assumes.

Have different hardware? Run `canirun <model> --benchmark --share`.
