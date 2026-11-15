"""System-level nodes: advisor report and the stage-boundary memory barrier."""
from __future__ import annotations

import gc
import time

from comfy_api.latest import io, ui

from . import core, sysinfo

CATEGORY = "h3_smart"
GB = 1024 ** 3


class H3Smart_SystemAdvisor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3Smart_SystemAdvisor",
            display_name="H3 System Advisor",
            category=f"{CATEGORY}/system",
            description=("Reads your GPU, RAM, launch flags and installed H3 files, then tells you which quantisations "
                         "your GPU runs natively, whether the weights fit in VRAM+RAM, and what to change. Runs on its own."),
            inputs=[io.Combo.Input("task", options=list(core.MODES), default="fl2va")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return time.time_ns()          # memory figures change between queues, so never reuse the old report

    @classmethod
    def execute(cls, task="fl2va") -> io.NodeOutput:
        facts = sysinfo.gather_facts()
        dit = sysinfo.h3_model_files("diffusion_models")
        te = sysinfo.h3_model_files("text_encoders")
        report = core.build_advisor_report(facts, dit, te, task)
        print("[H3Smart] system advisor\n" + report)
        return io.NodeOutput(report, ui=ui.PreviewText(report))


def _free_gb():
    """(free VRAM GB, available RAM GB) - either may be 0.0 if unknown."""
    vram = ram = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.mem_get_info()[0] / GB
    except Exception:
        pass
    try:
        import psutil
        ram = psutil.virtual_memory().available / GB
    except Exception:
        pass
    return vram, ram


def _trim_process_memory() -> str:
    """Ask the OS to take back freed heap pages. Best effort, never raises."""
    try:
        import ctypes
        import platform
        if platform.system() == "Windows":
            k32 = ctypes.windll.kernel32
            k32.SetProcessWorkingSetSize(k32.GetCurrentProcess(), ctypes.c_size_t(-1), ctypes.c_size_t(-1))
            return "working set trimmed"
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        return "malloc_trim done"
    except Exception as e:                      # noqa: BLE001
        return f"trim unavailable ({type(e).__name__})"


class H3Smart_MemoryBarrier(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        t = io.MatchType.Template("T")
        return io.Schema(
            node_id="H3Smart_MemoryBarrier",
            display_name="H3 Memory Barrier",
            category=f"{CATEGORY}/system",
            description=("Pass-through that unloads models and frees VRAM/RAM at a stage boundary. Put one between the "
                         "conditioning and the sampler (drops the 32B text encoder before the DiT loads) and one between "
                         "the sampler and VAE decode (drops the DiT before the VAE loads)."),
            inputs=[
                io.MatchType.Input("value", template=t, tooltip="Any value; passed through unchanged."),
                io.Boolean.Input("unload_models", default=True, tooltip="Unload every loaded model from VRAM."),
                io.Boolean.Input("collect_garbage", default=True, tooltip="Run Python GC and empty the CUDA cache."),
                io.Boolean.Input("trim_process_memory", default=False, advanced=True,
                                 tooltip="Experimental: ask the OS to reclaim freed process memory (Windows working-set trim / glibc malloc_trim)."),
                io.String.Input("label", default="", advanced=True),
            ],
            outputs=[io.MatchType.Output(template=t, display_name="value")],
        )

    @classmethod
    def execute(cls, value, unload_models=True, collect_garbage=True, trim_process_memory=False, label="") -> io.NodeOutput:
        import comfy.model_management as mm

        v0, r0 = _free_gb()
        if unload_models:
            mm.unload_all_models()
        if collect_garbage:
            gc.collect()
            mm.soft_empty_cache(True)
        extra = ""
        if trim_process_memory:
            extra = "; " + _trim_process_memory()
        v1, r1 = _free_gb()
        msg = (f"barrier{' ' + label if label else ''}: VRAM free {v0:.1f} -> {v1:.1f} GB, "
               f"RAM available {r0:.1f} -> {r1:.1f} GB{extra}")
        print(f"[H3Smart] {msg}")
        return io.NodeOutput(value, ui=ui.PreviewText(msg))
