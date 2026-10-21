"""canirun <model> — will it fit, which quant, how fast."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import __version__, estimate as est, hardware, models, presets
from .hardware import GiB, Hardware, Memory
from .quants import KV_BYTES, quants_for

console = Console()
err_console = Console(stderr=True)


def _gb(b: float) -> str:
    return f"{b / GiB:.1f} GB"


def _params(n: int) -> str:
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="canirun",
        description="Can your machine run that LLM? Memory, best quant, and tokens/sec.",
    )
    p.add_argument("model", nargs="?", help="alias (llama3-8b), HF repo id (org/name), or path to config.json")
    p.add_argument("--ctx", type=int, default=8192, help="context length in tokens (default 8192)")
    p.add_argument("--kv-quant", choices=list(KV_BYTES), default="f16", help="KV cache type (llama.cpp -ctk/-ctv)")
    what_if = p.add_argument_group("what-if (override detected hardware)")
    what_if.add_argument("--gpu", help="pretend you have this GPU or Mac chip, e.g. 4090, 'm4 max'")
    what_if.add_argument("--vram", type=float, help="GPU memory in GB")
    what_if.add_argument("--gpu-bw", type=float, help="GPU memory bandwidth in GB/s")
    what_if.add_argument("--ram", type=float, help="system RAM in GB")
    what_if.add_argument("--ram-bw", type=float, help="system RAM bandwidth in GB/s")
    what_if.add_argument("--cpu-only", action="store_true", help="ignore the GPU")
    p.add_argument("--rebench", action="store_true", help="re-measure RAM bandwidth instead of using the cache")
    p.add_argument("--refresh", action="store_true", help="re-fetch the model config instead of using the cache")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--list-models", action="store_true", help="show built-in model aliases")
    p.add_argument("--list-gpus", action="store_true", help="show GPU / Mac presets for --gpu")
    p.add_argument("--version", action="version", version=f"canirun {__version__}")
    args = p.parse_args(argv)
    if not (args.model or args.list_models or args.list_gpus):
        p.error("which model? e.g. `canirun llama3-8b`")
    if args.ctx <= 0:
        p.error("--ctx must be positive")
    return args


def build_hardware(args: argparse.Namespace) -> Hardware:
    """Detect what's needed, then apply what-if overrides on top."""
    # A full hypothetical machine skips detection (and the bandwidth benchmark).
    pure_what_if = args.gpu or (args.ram and args.ram_bw)

    if args.gpu and (apple := presets.find_apple(args.gpu)):
        key, default_mem, bw = apple
        ram = (args.ram or default_mem) * GiB
        hw = hardware.detect_apple(key, ram)
        hw.cpu_name = f"Apple {key.upper()}"
        hw.notes = []
        return hw

    if pure_what_if:
        vm_total = (args.ram or 32) * GiB
        ram = Memory("RAM", vm_total, vm_total * 0.8, args.ram_bw or 60, "cpu", "assumed")
        hw = Hardware(cpu_name="hypothetical", ram=ram)
    else:
        with console.status("Measuring memory bandwidth…", spinner="dots") if not args.json else _null():
            hw = hardware.detect(remeasure=args.rebench)

    if args.ram:
        hw.ram.total_bytes = args.ram * GiB
        hw.ram.usable_bytes = args.ram * GiB * 0.8  # leave room for the OS
    if args.ram_bw:
        hw.ram.bandwidth_gbs, hw.ram.bandwidth_source = args.ram_bw, "given"

    if args.gpu:
        found = presets.find_gpu(args.gpu)
        if not found:
            raise models.ModelError(f"Unknown GPU {args.gpu!r}; see `canirun --list-gpus` or use --vram/--gpu-bw.")
        key, vram, bw = found
        hw.gpu = Memory(key.upper(), vram * GiB, vram * GiB - 0.3 * GiB, bw, "cuda", "spec")
    if args.vram or args.gpu_bw:
        if hw.gpu is None:
            if not (args.vram and args.gpu_bw):
                raise models.ModelError("No GPU detected: pass both --vram and --gpu-bw.")
            hw.gpu = Memory("GPU", 0, 0, 0, "cuda", "given")
        if args.vram:
            hw.gpu.total_bytes, hw.gpu.usable_bytes = args.vram * GiB, (args.vram - 0.3) * GiB
        if args.gpu_bw:
            hw.gpu.bandwidth_gbs = args.gpu_bw
    if args.cpu_only:
        hw.gpu, hw.unified = None, False
    return hw


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fit_cell(e: est.Estimate, hw: Hardware) -> Text:
    if e.needs_free_ram:
        return Text("◌ close apps", style="yellow")
    if e.placement == "gpu":
        return Text("✓ GPU" if not hw.unified else "✓ fits", style="green")
    if e.placement == "split":
        return Text(f"◐ {e.gpu_layers}/{e.n_layers} on GPU", style="yellow")
    if e.placement == "cpu":
        return Text("◐ CPU only", style="yellow")
    return Text("✗ won't fit", style="red")


def _short_gpu(name: str) -> str:
    return name.replace("NVIDIA ", "").replace("GeForce ", "").replace(" GPU", "")


def render(spec: models.ModelSpec, hw: Hardware, ests: list[est.Estimate], best, args) -> None:
    moe = f" · MoE {spec.experts_per_token}/{spec.n_experts} experts, {_params(spec.active_params)} active" if spec.is_moe else ""
    console.print()
    console.print(f"[bold]{spec.name}[/]  [dim]{_params(spec.total_params)} params{moe} · ctx {args.ctx:,}[/]")

    if hw.gpu and hw.unified:
        console.print(f"[dim]{hw.cpu_name} · {_gb(hw.ram.total_bytes)} unified, "
                      f"{_gb(hw.gpu.usable_bytes)} usable by GPU @ {hw.gpu.bandwidth_gbs:.0f} GB/s[/]")
    else:
        if hw.gpu:
            console.print(f"[dim]GPU  {_short_gpu(hw.gpu.name)} · {_gb(hw.gpu.usable_bytes)} free "
                          f"@ {hw.gpu.bandwidth_gbs:.0f} GB/s[/]")
        console.print(f"[dim]RAM  {_gb(hw.ram.total_bytes)} · {_gb(hw.ram.usable_bytes)} free "
                      f"@ {hw.ram.bandwidth_gbs:.0f} GB/s[/]")

    t = Table(box=None, pad_edge=False, padding=(0, 1), header_style="bold dim")
    t.add_column("Quant")
    t.add_column("Needs", justify="right")
    t.add_column("Runs on")
    t.add_column("Speed", justify="right")
    t.add_column("Quality", style="dim")
    t.add_column("")
    for e in ests:
        speed = f"~{e.tok_s:.0f} tok/s" if e.tok_s and e.tok_s >= 1 else (f"~{e.tok_s:.1f} tok/s" if e.tok_s else "—")
        mark = Text("← recommended", style="bold cyan") if e is best else Text("")
        style = "bold" if e is best else ("dim" if not e.fits else "")
        t.add_row(Text(e.quant.name, style=style), _gb(e.total_bytes), _fit_cell(e, hw), speed, e.quant.quality, mark)
    console.print()
    console.print(t)
    console.print()

    if best is None:
        console.print("[red]✗ Doesn't fit at any quant.[/] Try a smaller context (--ctx), --kv-quant q8_0, or a smaller model.")
    else:
        where = {"gpu": "fully on the GPU" if not hw.unified else "in memory",
                 "split": f"with {best.gpu_layers}/{best.n_layers} layers on the GPU (llama.cpp -ngl {best.gpu_layers})",
                 "cpu": "on the CPU"}[best.placement]
        console.print(f"[bold cyan]→ {best.quant.name}[/] runs {where} at [bold]~{best.tok_s:.0f} tok/s[/].")
        if best.needs_free_ram:
            console.print("[yellow]  Only after closing other apps: it needs more RAM than is free right now.[/]")

    kv = ests[0].kv_bytes
    if spec.max_context and args.ctx > spec.max_context:
        console.print(f"[yellow]! {spec.name} supports {spec.max_context:,} tokens; --ctx {args.ctx:,} is beyond that.[/]")
    console.print(f"[dim]KV cache {_gb(kv)} at {args.ctx:,} tokens ({args.kv_quant}). "
                  f"Speeds are llama.cpp decode estimates at a half-full context.[/]")
    for note in hw.notes:
        console.print(f"[dim]{note}[/]")
    if spec.native_quant == "mxfp4":
        console.print("[dim]Released in MXFP4; lower GGUF quants are barely smaller, so use the native file.[/]")
    if spec.param_source == "config":
        console.print("[dim]Parameter count estimated from config.json.[/]")


def _to_json(spec, hw, ests, best, args) -> str:
    return json.dumps({
        "model": asdict(spec),
        "context": args.ctx,
        "kv_quant": args.kv_quant,
        "hardware": asdict(hw),
        "estimates": [{
            "quant": e.quant.name,
            "total_gb": round(e.total_bytes / GiB, 2),
            "weights_gb": round(e.weights_bytes / GiB, 2),
            "kv_cache_gb": round(e.kv_bytes / GiB, 2),
            "placement": e.placement,
            "gpu_layers": e.gpu_layers,
            "tok_s": round(e.tok_s, 1) if e.tok_s else None,
        } for e in ests],
        "recommended": best.quant.name if best else None,
    }, indent=2)


def _list_models() -> None:
    t = Table("alias", "Hugging Face repo", box=None, header_style="bold dim")
    for alias, repo in models.ALIASES.items():
        t.add_row(alias, repo)
    console.print(t)


def _list_gpus() -> None:
    t = Table("--gpu", "memory", "bandwidth", box=None, header_style="bold dim")
    for name, (mem, bw) in {**presets.GPU_PRESETS, **presets.APPLE_CHIPS}.items():
        t.add_row(name, f"{mem:g} GB", f"{bw:g} GB/s")
    console.print(t)
    console.print("[dim]Mac chips use their base memory; set yours with --ram.[/]")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_models:
        _list_models()
    if args.list_gpus:
        _list_gpus()
    if not args.model:
        return 0

    try:
        with console.status(f"Fetching {args.model} config…", spinner="dots") if not args.json else _null():
            spec = models.load(args.model, refresh=args.refresh)
        hw = build_hardware(args)
    except models.ModelError as e:
        err_console.print(f"[red]error:[/] {e}")
        return 1

    ests = est.estimate_all(spec, hw, args.ctx, KV_BYTES[args.kv_quant], quants_for(spec.native_quant))
    best = est.recommend(ests)
    if args.json:
        print(_to_json(spec, hw, ests, best, args))
    else:
        render(spec, hw, ests, best, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
