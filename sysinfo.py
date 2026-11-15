"""Gather hardware / ComfyUI facts for the advisor and planner.

Every import of torch, psutil or ComfyUI modules is lazy and wrapped, so this works (with empty facts)
outside ComfyUI too.

Note on VRAM numbers: with the default cudaMallocAsync allocator, torch's own allocator statistics read 0
(ComfyUI's System Info panel shows "Torch VRAM Total 0 B" for the same reason). The driver-level
torch.cuda.mem_get_info() is the reliable source, so that is what is used here.
"""
from __future__ import annotations

import os
import platform

from . import core

GB = 1024 ** 3
H3_HINTS = ("h3", "minimax")


def _torch():
    try:
        import torch  # noqa: WPS433
        return torch
    except Exception:
        return None


def gather_facts() -> core.SystemFacts:
    facts = core.SystemFacts()
    facts.windows = platform.system() == "Windows"
    torch = _torch()

    if torch is not None:
        try:
            parts = torch.__version__.split("+")[0].split(".")[:2]
            facts.torch_version = (int(parts[0]), int(parts[1]))
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(dev)
                free, total = torch.cuda.mem_get_info(dev)
                facts.gpu = core.GpuCaps(
                    name=props.name, cc=(props.major, props.minor),
                    vram_total_gb=total / GB, vram_free_gb=free / GB, nvidia=True)
        except Exception:
            pass

    try:
        import psutil
        vm = psutil.virtual_memory()
        facts.ram_total_gb = vm.total / GB
        facts.ram_free_gb = vm.available / GB
    except Exception:
        pass

    try:
        from comfy.cli_args import args  # type: ignore
        facts.cache_none = bool(getattr(args, "cache_none", False))
        facts.reserve_vram_gb = getattr(args, "reserve_vram", None)
        facts.fp8_text_enc_flag = bool(getattr(args, "fp8_e4m3fn_text_enc", False))
        try:
            from comfy.cli_args import enables_dynamic_vram  # type: ignore
            facts.dynamic_vram = bool(enables_dynamic_vram())
        except Exception:
            facts.dynamic_vram = None
    except Exception:
        pass

    try:
        from comfyui_version import __version__  # type: ignore
        facts.comfy_version = str(__version__)
    except Exception:
        pass
    return facts


def comfy_version() -> str:
    try:
        from comfyui_version import __version__  # type: ignore
        return str(__version__)
    except Exception:
        return "unknown"


def planning_budget_gb(facts: core.SystemFacts, source: str = "stable", desktop_overhead_gb: float = 0.8,
                       override_gb: float = 0.0, reserve_gb: float = -1.0) -> float:
    """VRAM the sampler can plan on.

    stable: total VRAM - reserve - desktop overhead. Deterministic between runs, so planner outputs
            (and therefore prompt-cache keys) do not flicker when a browser tab grabs a little VRAM.
    live:   free VRAM right now - reserve.
    override_gb > 0 replaces everything.
    """
    if override_gb and override_gb > 0:
        return float(override_gb)
    reserve = core.reserve_for(facts) if reserve_gb < 0 else float(reserve_gb)
    if facts.gpu.vram_total_gb <= 0:
        return 0.0
    if source == "live" and facts.gpu.vram_free_gb > 0:
        return max(0.0, facts.gpu.vram_free_gb - reserve)
    return max(0.0, facts.gpu.vram_total_gb - reserve - desktop_overhead_gb)


def h3_model_files(folder_name: str) -> dict:
    """filename -> size in GB for H3-looking files in a ComfyUI model folder."""
    out = {}
    try:
        import folder_paths  # type: ignore
        names = folder_paths.get_filename_list(folder_name)
    except Exception:
        return out
    for n in names:
        low = n.lower()
        if not any(h in low for h in H3_HINTS):
            continue
        size = 0.0
        try:
            full = folder_paths.get_full_path(folder_name, n)
            if full and os.path.exists(full):
                size = os.path.getsize(full) / GB
        except Exception:
            pass
        out[n] = size
    return out


def lora_names() -> list:
    try:
        import folder_paths  # type: ignore
        return list(folder_paths.get_filename_list("loras"))
    except Exception:
        return []
