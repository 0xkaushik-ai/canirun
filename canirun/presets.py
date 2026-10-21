"""Known hardware specs, used for what-if runs (--gpu) and as detection fallbacks."""
from __future__ import annotations

import re


# name -> (memory GB, bandwidth GB/s)
GPU_PRESETS: dict[str, tuple[float, float]] = {
    # NVIDIA desktop
    "rtx 3060": (12, 360),
    "rtx 3060 ti": (8, 448),
    "rtx 3070": (8, 448),
    "rtx 3080": (10, 760),
    "rtx 3090": (24, 936),
    "rtx 4060": (8, 272),
    "rtx 4060 ti": (16, 288),
    "rtx 4070": (12, 504),
    "rtx 4070 ti super": (16, 672),
    "rtx 4080": (16, 717),
    "rtx 4090": (24, 1008),
    "rtx 5060 ti": (16, 448),
    "rtx 5070": (12, 672),
    "rtx 5070 ti": (16, 896),
    "rtx 5080": (16, 960),
    "rtx 5090": (32, 1792),
    # NVIDIA datacenter
    "a100 40gb": (40, 1555),
    "a100 80gb": (80, 2039),
    "h100": (80, 3350),
    # AMD
    "rx 7900 xtx": (24, 960),
    "rx 7900 xt": (20, 800),
    "rx 9070 xt": (16, 640),
}

# Apple Silicon: chip -> (default memory GB, bandwidth GB/s).
# Memory varies per machine, so --ram overrides the default.
APPLE_CHIPS: dict[str, tuple[float, float]] = {
    "m1": (16, 68),
    "m1 pro": (16, 200),
    "m1 max": (32, 400),
    "m1 ultra": (64, 800),
    "m2": (16, 100),
    "m2 pro": (16, 200),
    "m2 max": (32, 400),
    "m2 ultra": (64, 800),
    "m3": (16, 100),
    "m3 pro": (18, 150),
    "m3 max": (36, 400),
    "m3 ultra": (96, 819),
    "m4": (16, 120),
    "m4 pro": (24, 273),
    "m4 max": (36, 546),
    "m5": (16, 153),
}


def _norm(name: str) -> str:
    name = name.lower().replace("nvidia", "").replace("geforce", "").replace("apple", "")
    name = re.sub(r"[-_]", " ", name)
    name = re.sub(r"\brtx(\d)", r"rtx \1", name)
    return re.sub(r"\s+", " ", name).strip()


def _lookup(table: dict, name: str):
    """Longest-key match, so 'rtx 4070 ti super' wins over 'rtx 4070'."""
    n = _norm(name)
    hits = [k for k in table if re.search(rf"(^|\s){re.escape(k)}($|\s)", n)]
    if not hits:
        # allow bare model numbers like "4090"
        hits = [k for k in table if k.split()[-1] == n or k.endswith(" " + n)]
    return max(hits, key=len) if hits else None


def find_gpu(name: str) -> tuple[str, float, float] | None:
    key = _lookup(GPU_PRESETS, name)
    return (key, *GPU_PRESETS[key]) if key else None


def find_apple(name: str) -> tuple[str, float, float] | None:
    key = _lookup(APPLE_CHIPS, name)
    return (key, *APPLE_CHIPS[key]) if key else None
