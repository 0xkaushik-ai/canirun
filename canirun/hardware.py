"""Detect RAM, GPU, and memory bandwidth on the local machine."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from . import presets

GiB = 1024**3
CACHE_FILE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "canirun" / "hardware.json"


@dataclass
class Memory:
    name: str
    total_bytes: float
    usable_bytes: float         # what a model can actually claim right now
    bandwidth_gbs: float
    kind: str                   # "cuda" | "metal" | "cpu"
    bandwidth_source: str = "measured"


@dataclass
class Hardware:
    cpu_name: str
    ram: Memory
    gpu: Memory | None = None
    unified: bool = False       # Apple Silicon: GPU and CPU share one pool
    notes: list[str] = field(default_factory=list)


def cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def nvml_bandwidth_gbs(bus_width_bits: int, max_mem_clock_mhz: int) -> float:
    """NVML reports the memory clock at half the data rate (GDDR and HBM alike)."""
    return bus_width_bits / 8 * max_mem_clock_mhz * 1e6 * 2 / 1e9


def detect_nvidia() -> Memory | None:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                bw = nvml_bandwidth_gbs(
                    pynvml.nvmlDeviceGetMemoryBusWidth(h),
                    pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM),
                )
                src = "spec"
            except Exception:
                preset = presets.find_gpu(name)
                bw, src = (preset[2], "spec") if preset else (0.0, "unknown")
            gpus.append(Memory(name, mem.total, mem.free, bw, "cuda", src))
    finally:
        pynvml.nvmlShutdown()

    if not gpus:
        return None
    if len(gpus) == 1:
        return gpus[0]
    # llama.cpp splits layers across GPUs: capacity adds up, and the slowest
    # card bounds bandwidth (exact when the cards match).
    names = {g.name for g in gpus}
    label = f"{len(gpus)}× {gpus[0].name}" if len(names) == 1 else " + ".join(g.name for g in gpus)
    return Memory(
        label,
        sum(g.total_bytes for g in gpus),
        sum(g.usable_bytes for g in gpus),
        min(g.bandwidth_gbs for g in gpus),
        "cuda",
        gpus[0].bandwidth_source,
    )


def apple_gpu_fraction(total_bytes: float) -> float:
    """macOS lets the GPU wire ~2/3 of RAM on small machines, ~3/4 on large ones."""
    return 0.67 if total_bytes <= 36 * GiB else 0.75


def detect_apple(chip: str, ram_total: float) -> Hardware | None:
    found = presets.find_apple(chip)
    if not found:
        return None
    key, _, bw = found
    # The binned Max chips have less bandwidth than the full ones.
    cores = psutil.cpu_count(logical=False) or 0
    if key == "m3 max" and cores == 14:
        bw = 300
    elif key == "m4 max" and cores == 14:
        bw = 410
    usable = ram_total * apple_gpu_fraction(ram_total)
    mem = Memory(chip, ram_total, usable, bw, "metal", "spec")
    return Hardware(
        cpu_name=chip,
        ram=Memory("Unified memory", ram_total, usable, bw, "cpu", "spec"),
        gpu=mem,
        unified=True,
        notes=[
            "macOS caps GPU memory at ~"
            f"{apple_gpu_fraction(ram_total):.0%} of RAM; raise it with "
            "`sudo sysctl iogpu.wired_limit_mb=<MB>`."
        ],
    )


def measure_ram_bandwidth(size_mb: int = 512, repeats: int = 6, threads: int | None = None) -> float:
    """Multi-threaded sequential read benchmark, in GB/s.

    Decode streams weights from memory, so read bandwidth is what matters.
    numpy releases the GIL inside reductions, so threads run in parallel.
    """
    import numpy as np

    threads = threads or psutil.cpu_count(logical=False) or 4
    n = size_mb * 2**20 // 8
    buf = np.ones(n, dtype=np.float64)
    bounds = [(i * n // threads, (i + 1) * n // threads) for i in range(threads)]

    def read(b):
        return float(buf[b[0]:b[1]].sum())

    best = float("inf")
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(read, bounds))  # warm up
        for _ in range(repeats):
            t = time.perf_counter()
            list(pool.map(read, bounds))
            best = min(best, time.perf_counter() - t)
    return buf.nbytes / best / 1e9


def ram_bandwidth(remeasure: bool = False) -> float:
    key = f"{cpu_name()}|{psutil.virtual_memory().total}"
    if not remeasure:
        try:
            cached = json.loads(CACHE_FILE.read_text())
            if cached.get("key") == key:
                return cached["ram_bandwidth_gbs"]
        except Exception:
            pass
    bw = measure_ram_bandwidth()
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps({"key": key, "ram_bandwidth_gbs": bw}))
    except OSError:
        pass
    return bw


def detect(remeasure: bool = False) -> Hardware:
    vm = psutil.virtual_memory()
    cpu = cpu_name()

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        hw = detect_apple(cpu, vm.total)
        if hw:
            return hw

    ram = Memory("RAM", vm.total, vm.available, ram_bandwidth(remeasure), "cpu")
    return Hardware(cpu_name=cpu, ram=ram, gpu=detect_nvidia())
