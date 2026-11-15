"""Pure-Python logic for H3 Smart.

Nothing in here imports torch or ComfyUI, so everything can be unit-tested anywhere.
Where a constant or formula mirrors ComfyUI's own code, the source is named so it can
be re-checked after a ComfyUI update:

  * comfy_extras/nodes_minimax_h3.py   frame grid (17k+5), canvas rules, token geometry
  * comfy_extras/nodes_resolution.py   ResolutionSelector (1 MP = 1024*1024 px)
  * comfy/ldm/minimax/model.py         DiT size (hidden 5376, 50 blocks, ffn 14336, patch 1x2x2)
  * comfy/model_management.py          supports_fp8_compute / supports_nvfp4_compute
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- geometry
FPS = 24
AUDIO_LATENT_FPS = 40
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344          # native area cap for the open weights
MIN_FRAMES = 5
SOFT_MIN_FRAMES = 107            # ~4.5 s, shortest clip the docs describe (4-15 s)
TRAINED_MIN_FRAMES = 124         # native tooltip: "trained range is ~124-362"
TRAINED_MAX_FRAMES = 362

# DiT geometry (defaults of MiniMaxH3Model in comfy/ldm/minimax/model.py)
DIT_HIDDEN = 5376
DIT_LAYERS = 50
DIT_FFN = 14336
DIT_PATCH = (1, 2, 2)

# Rough per-token activation footprint of one block in bf16, at the MLP peak:
#   residual + normed input + gate/up (2*ffn) + activated ffn  ~= 5376*2 + 28672 + 14336 elements
# 53,760 elements * 2 bytes ~= 107.5 KB. Rounded up a little for fp32 modulation tensors.
DEFAULT_KB_PER_TOKEN = 110.0

ASPECTS = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "21:9": (21, 9),
    "3:2": (3, 2),
    "2:3": (2, 3),
}


def align_up(n: int) -> int:
    """Smallest frame count >= n on the model's 17k+5 grid (what the native nodes do)."""
    n = max(MIN_FRAMES, int(n))
    return n + ((MIN_FRAMES - n) % 17)


def align_down(n: int) -> int:
    n = max(MIN_FRAMES, int(n))
    return max(MIN_FRAMES, n - ((n - MIN_FRAMES) % 17))


def frames_from_seconds(seconds: float, snap: str = "up") -> int:
    """Seconds -> frame count on the 17k+5 grid. snap: 'up' (native behaviour), 'nearest', 'down'."""
    target = max(MIN_FRAMES, round(float(seconds) * FPS))
    up, down = align_up(target), align_down(target)
    if snap == "up":
        return up
    if snap == "down":
        return down
    return down if (target - down) < (up - target) else up


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def audio_latent_len(frame_count: int) -> int:
    return round(frame_count / FPS * AUDIO_LATENT_FPS)


def temporal_shape(length: int):
    """(aligned frame count, video latent frames, audio latent length) as the native nodes compute them."""
    fc = align_up(max(MIN_FRAMES, length))
    return fc, video_latent_t(fc), audio_latent_len(fc)


def _round_to(value: float, multiple: int = CANVAS_MULTIPLE) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def solve_canvas(aspect_w: float, aspect_h: float, megapixels: float,
                 multiple: int = CANVAS_MULTIPLE, max_pixels: int = MAX_PIXELS):
    """Same maths as ComfyUI's ResolutionSelector (1 MP = 1024*1024 px), capped at the model's area limit."""
    total = min(megapixels * 1024 * 1024, max_pixels)
    scale = math.sqrt(total / (aspect_w * aspect_h))
    return _round_to(aspect_w * scale, multiple), _round_to(aspect_h * scale, multiple)


def aspect_from_size(width: float, height: float, max_denominator: int = 32):
    """Best simple ratio for an image size (used so I2V first frames are not stretched)."""
    if width <= 0 or height <= 0:
        raise ValueError("image has no size")
    ratio = width / height
    best = (1, 1)
    best_err = abs(ratio - 1.0)
    for d in range(1, max_denominator + 1):
        n = round(ratio * d)
        if n < 1:
            continue
        err = abs(ratio - n / d)
        if err < best_err - 1e-9:
            best, best_err = (n, d), err
    g = math.gcd(*best)
    return best[0] // g, best[1] // g


# --------------------------------------------------------------------------- token / memory estimates
def tokens_per_latent_frame(width: int, height: int) -> int:
    """VAE is /16 spatially, the DiT patches 2x2 => one token per 32x32 pixels per latent frame."""
    return (height // 32) * (width // 32)


@dataclass
class TokenBreakdown:
    video: int = 0
    audio: int = 0
    text: int = 0
    refs: int = 0

    @property
    def total(self) -> int:
        return self.video + self.audio + self.text + self.refs


def count_tokens(width: int, height: int, frames: int, text_tokens: int = 300,
                 ref_images: int = 0, ref_image_size: str = "match",
                 ref_video_seconds: float = 0.0) -> TokenBreakdown:
    fc, lat_t, audio_t = temporal_shape(frames)
    per_frame = tokens_per_latent_frame(width, height)
    t = TokenBreakdown(video=lat_t * per_frame, audio=2 * audio_t, text=int(text_tokens))
    if ref_images:
        if ref_image_size == "max":
            # worst case: a source image with a 2048 px short edge at the canvas aspect ratio
            short = 2048
            ar = max(width, height) / max(1, min(width, height))
            per_image = (short // 32) * (_round_to(short * ar) // 32)
        else:
            per_image = per_frame
        t.refs += ref_images * per_image
    if ref_video_seconds > 0:
        n = align_down(round(ref_video_seconds * FPS))
        t.refs += video_latent_t(n) * per_frame
    return t


@dataclass
class VramEstimate:
    activations_gb: float
    weights_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return self.activations_gb + self.weights_gb + self.overhead_gb


def estimate_vram(total_tokens: int, kb_per_token: float = DEFAULT_KB_PER_TOKEN,
                  weights_working_gb: float = 0.6, overhead_gb: float = 0.8,
                  calibration: float = 1.0) -> VramEstimate:
    """Heuristic peak VRAM during one DiT forward pass.

    weights_working_gb: the block(s) ComfyUI keeps resident while streaming the rest.
    overhead_gb: CUDA context, VAE/conditioning leftovers, fragmentation.
    This is an estimate, not a measurement: calibrate `calibration` against a real run.
    """
    act = total_tokens * kb_per_token * 1000.0 * calibration / (1024 ** 3)
    return VramEstimate(act, weights_working_gb, overhead_gb)


def status_for(est_gb: float, budget_gb: float) -> str:
    if budget_gb <= 0:
        return "unknown"
    if est_gb <= 0.85 * budget_gb:
        return "ok"
    if est_gb <= budget_gb:
        return "tight"
    return "over"


@dataclass
class Plan:
    width: int
    height: int
    frames: int
    megapixels: float
    tokens: TokenBreakdown
    estimate: VramEstimate
    budget_gb: float
    status: str
    notes: list = field(default_factory=list)
    adjustments: list = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return self.frames / FPS


def plan_generation(aspect: tuple, megapixels: float, seconds: float, budget_gb: float,
                    snap: str = "up", strategy: str = "none", text_tokens: int = 300,
                    ref_images: int = 0, ref_image_size: str = "match", ref_video_seconds: float = 0.0,
                    kb_per_token: float = DEFAULT_KB_PER_TOKEN, calibration: float = 1.0,
                    min_megapixels: float = 0.2) -> Plan:
    """Pick width/height/frames for a request and, optionally, shrink it until it fits `budget_gb`.

    strategy: 'none' | 'reduce_resolution' | 'reduce_duration'
    """
    mp = min(float(megapixels), MAX_PIXELS / (1024 * 1024))
    frames = frames_from_seconds(seconds, snap)
    adjustments = []

    def build(mp_, frames_):
        w, h = solve_canvas(aspect[0], aspect[1], mp_)
        tok = count_tokens(w, h, frames_, text_tokens, ref_images, ref_image_size, ref_video_seconds)
        est = estimate_vram(tok.total, kb_per_token, calibration=calibration)
        return w, h, tok, est

    w, h, tok, est = build(mp, frames)
    first = (w, h, frames)

    if strategy == "reduce_resolution":
        while est.total_gb > budget_gb and mp * 0.92 >= min_megapixels:
            mp *= 0.92
            w, h, tok, est = build(mp, frames)
        if (w, h) != first[:2]:
            adjustments.append(f"resolution {first[0]}x{first[1]} -> {w}x{h} to fit the VRAM budget")
    elif strategy == "reduce_duration":
        while est.total_gb > budget_gb and frames - 17 >= SOFT_MIN_FRAMES:
            frames -= 17
            w, h, tok, est = build(mp, frames)
        if frames != first[2]:
            adjustments.append(f"length {first[2]} -> {frames} frames to fit the VRAM budget")

    status = status_for(est.total_gb, budget_gb)
    notes = []
    if frames < TRAINED_MIN_FRAMES:
        notes.append(f"{frames} frames is below the ~{TRAINED_MIN_FRAMES}-frame range MiniMax lists as trained; expect weaker results")
    if frames > TRAINED_MAX_FRAMES:
        notes.append(f"{frames} frames is above the ~{TRAINED_MAX_FRAMES}-frame trained range (untested)")
    if status == "over" and strategy == "none":
        notes.append("estimate exceeds the VRAM budget: lower megapixels/duration or choose a fit strategy")
    if status == "over" and strategy != "none":
        notes.append("still over budget at the smallest allowed size; ComfyUI may still run it by offloading, slowly")
    return Plan(w, h, frames, w * h / (1024 * 1024), tok, est, budget_gb, status, notes, adjustments)


def format_plan(plan: Plan, extra_lines=()) -> str:
    t, e = plan.tokens, plan.estimate
    lines = [
        f"Canvas   : {plan.width} x {plan.height}  ({plan.megapixels:.2f} MP)",
        f"Length   : {plan.frames} frames = {plan.seconds:.2f} s @ {FPS} fps  (17k+5 grid)",
        f"Tokens   : {t.total:,} total  (video {t.video:,} | audio {t.audio:,} | text ~{t.text:,} | refs {t.refs:,})",
        f"Est. VRAM: ~{e.total_gb:.1f} GB peak  (activations {e.activations_gb:.1f} + weights in flight {e.weights_gb:.1f} + overhead {e.overhead_gb:.1f})",
        f"Budget   : {plan.budget_gb:.1f} GB  -> {plan.status.upper()}",
    ]
    for a in plan.adjustments:
        lines.append(f"Adjusted : {a}")
    for n in plan.notes:
        lines.append(f"Note     : {n}")
    lines.extend(extra_lines)
    lines.append("(heuristic estimate, not a measurement)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- sampler presets
MODES = ("fl2va", "ref2va")
SPEEDS = ("turbo", "balanced", "quality")

# Defaults follow ComfyUI's official MiniMax H3 templates: res_multistep + simple, BasicGuider (no CFG),
# 20 steps base, optional Lightning LoRA (8 steps FL2VA / 4 steps REF2VA).
_PRESETS = {
    ("fl2va", "turbo"): dict(steps=8, lora=True),
    ("fl2va", "balanced"): dict(steps=20, lora=False),
    ("fl2va", "quality"): dict(steps=25, lora=False),
    ("ref2va", "turbo"): dict(steps=4, lora=True),
    ("ref2va", "balanced"): dict(steps=20, lora=False),
    ("ref2va", "quality"): dict(steps=25, lora=False),
}
DEFAULT_SAMPLER = "res_multistep"
DEFAULT_SCHEDULER = "simple"


def sampler_preset(mode: str, speed: str, steps_override: int = 0) -> dict:
    if (mode, speed) not in _PRESETS:
        raise ValueError(f"unknown preset {mode!r}/{speed!r}")
    p = dict(_PRESETS[(mode, speed)])
    if steps_override and steps_override > 0:
        p["steps"] = int(steps_override)
    p.update(sampler=DEFAULT_SAMPLER, scheduler=DEFAULT_SCHEDULER, mode=mode, speed=speed)
    return p


def score_turbo_lora(filename: str, mode: str, steps: int) -> int:
    """Higher is a better match. FL2VA and REF2VA adapters must never be mixed, so a wrong-mode file is heavily penalised."""
    n = filename.lower()
    score = 0
    if "h3" not in n and "minimax" not in n:
        return -100
    if any(k in n for k in ("turbo", "lightning", "lightx2v")):
        score += 10
    else:
        return -50
    own, other = (("fl2v", "ref2v") if mode == "fl2va" else ("ref2v", "fl2v"))
    if own in n:
        score += 10
    if other in n:
        score -= 50
    if f"{steps}step" in n or f"{steps}-step" in n or f"{steps}_step" in n:
        score += 5
    if "comfyui" in n:
        score += 2
    if "bf16" in n:
        score += 1
    return score


def pick_turbo_lora(filenames, mode: str, steps: int):
    best, best_score = None, 19
    for f in sorted(filenames):
        s = score_turbo_lora(f, mode, steps)
        if s > best_score:
            best, best_score = f, s
    return best


# --------------------------------------------------------------------------- file classification / hardware rules
def classify_h3_file(name: str) -> dict:
    """Guess what an H3-related model file is from its name (these are community naming conventions)."""
    n = name.lower()
    info = {"name": name, "mode": None, "quant": "unknown", "pruned": "pruned" in n, "h3": ("h3" in n or "minimax" in n)}
    if "ref2v" in n or "r2v" in n:
        info["mode"] = "ref2va"
    elif "fl2v" in n or "t2v" in n or "i2v" in n:
        info["mode"] = "fl2va"
    if n.endswith(".gguf"):
        info["quant"] = "gguf"
    elif "nvfp4" in n:
        info["quant"] = "nvfp4"
    elif "mxfp8" in n:
        info["quant"] = "mxfp8"
    elif re.search(r"int4|w4a4|w4a8|nf4", n):
        info["quant"] = "int4"
    elif "int8" in n:
        info["quant"] = "int8"
    elif re.search(r"fp8|e4m3|e5m2", n):
        info["quant"] = "fp8"
    elif re.search(r"bf16|fp16|fp32", n):
        info["quant"] = "bf16"
    return info


@dataclass
class GpuCaps:
    name: str = "unknown"
    cc: tuple = (0, 0)
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    nvidia: bool = True


def quant_support(cc: tuple, windows: bool = False, torch_version: tuple = (2, 5)) -> dict:
    """Which quantised matmul paths the GPU runs natively. Mirrors comfy.model_management.supports_*_compute."""
    major, minor = cc
    fp8 = False
    if major >= 9:
        fp8 = True
    elif major == 8 and minor >= 9:
        fp8 = torch_version >= (2, 3) and not (windows and torch_version < (2, 4))
    return {"int8": True, "fp8": fp8, "nvfp4": major >= 10}


@dataclass
class SystemFacts:
    gpu: GpuCaps = field(default_factory=GpuCaps)
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    windows: bool = False
    torch_version: tuple = (2, 5)
    cache_none: bool = False
    reserve_vram_gb: float | None = None
    fp8_text_enc_flag: bool = False
    dynamic_vram: bool | None = None
    comfy_version: str = ""


def reserve_for(facts: SystemFacts) -> float:
    if facts.reserve_vram_gb is not None:
        return float(facts.reserve_vram_gb)
    return 0.6 if facts.windows else 0.4   # ComfyUI picks an OS-dependent default; this is only a planning figure


def vram_budget_gb(facts: SystemFacts) -> float:
    """VRAM the sampler can plan on: what is free right now minus the reserve."""
    free = facts.gpu.vram_free_gb or facts.gpu.vram_total_gb * 0.85
    return max(0.0, free - reserve_for(facts))


def _fmt_gb(x: float) -> str:
    return f"{x:.1f} GB"


def build_advisor_report(facts: SystemFacts, dit_files: dict, te_files: dict, mode: str = "fl2va") -> str:
    """Human-readable advice. dit_files / te_files map filename -> size in GB (0 if unknown)."""
    g = facts.gpu
    caps = quant_support(g.cc, facts.windows, facts.torch_version)
    out = []
    add = out.append

    add("== System ==")
    add(f"GPU     : {g.name}  (compute capability {g.cc[0]}.{g.cc[1]})")
    add(f"VRAM    : {_fmt_gb(g.vram_total_gb)} total, {_fmt_gb(g.vram_free_gb)} free, reserve {_fmt_gb(reserve_for(facts))}")
    add(f"RAM     : {_fmt_gb(facts.ram_total_gb)} total, {_fmt_gb(facts.ram_free_gb)} available")
    add("Quant   : INT8 matmul " + ("yes" if caps["int8"] else "no")
        + " | FP8 matmul " + ("yes" if caps["fp8"] else "no")
        + " | NVFP4 matmul " + ("yes" if caps["nvfp4"] else "no"))
    if facts.comfy_version:
        add(f"ComfyUI : {facts.comfy_version}")

    add("")
    add("== Findings ==")
    if not caps["nvfp4"]:
        add("[NOTE] This GPU has no NVFP4 matmul. NVFP4 files (the official H3 text encoder is one) still load but are")
        add("       dequantised on the fly, which is slower than on Blackwell. The H3 Cached Conditioning nodes avoid re-running it.")
    if not caps["fp8"]:
        add("[NOTE] This GPU has no FP8 matmul: FP8 diffusion models only save storage, not compute. Prefer INT8 ConvRot.")
    if caps["int8"]:
        add("[ OK ] INT8 ConvRot checkpoints run on the INT8 tensor-core path here.")
    if facts.cache_none:
        add("[NOTE] --cache-none is on: every queue re-runs every node, including the 32B text encoder.")
        add("       Use H3Smart cached-conditioning nodes so an unchanged prompt skips the encoder entirely.")
    if facts.fp8_text_enc_flag:
        add("[NOTE] --fp8_e4m3fn-text-enc is set. H3's encoder ships pre-quantised, so this flag is unlikely to matter much")
        add("       (not verified); the encoder's real cost is its size, see 'Weight residency' below.")
    if facts.dynamic_vram is False:
        add("[NOTE] Dynamic VRAM is off; large models will rely on estimate-based offloading.")

    budget = vram_budget_gb(facts)
    ram_usable = max(0.0, facts.ram_free_gb - 3.0)   # keep ~3 GB for the OS and the browser
    resident = budget + ram_usable
    add("")
    add("== Weight residency ==")
    add(f"VRAM budget {_fmt_gb(budget)} + usable RAM {_fmt_gb(ram_usable)} = ~{_fmt_gb(resident)} that can hold weights at once.")
    for label, files in (("Diffusion", dit_files), ("Text encoder", te_files)):
        for name, size in sorted(files.items()):
            if size <= 0:
                continue
            info = classify_h3_file(name)
            if size <= resident:
                verdict = "fits in VRAM+RAM"
            else:
                over = size - resident
                verdict = f"~{_fmt_gb(over)} must stream from disk each pass"
            tags = []
            if info["quant"] != "unknown":
                tags.append(info["quant"])
            if info["mode"]:
                tags.append(info["mode"])
            if info["pruned"]:
                tags.append("pruned")
            add(f"  {label:12s} {name}  [{_fmt_gb(size)}; {', '.join(tags) or 'unclassified'}]: {verdict}")

    add("")
    add("== Suggestions ==")
    best = recommend_diffusion(dit_files, caps, mode)
    if best:
        add(f"Best installed {mode} diffusion file for this GPU: {best}")
    else:
        add(f"No installed {mode} diffusion file found in models/diffusion_models.")
    biggest = max(dit_files.values(), default=0.0)
    if biggest and biggest > resident:
        add("The diffusion model does not fit in VRAM+RAM, so each sampling step re-reads part of it from disk.")
        add("  1. Use the Turbo LoRA (8 steps FL2VA / 4 steps REF2VA) - fewer steps means fewer passes over the weights.")
        add("  2. Try a smaller community quant (GGUF Q4/Q3 via ComfyUI-GGUF, or INT4 ConvRot builds ~10-12 GB);")
        add("     read that file's model card for loader requirements and expect some quality loss.")
        add("  3. Close other RAM-heavy programs; every GB of free RAM reduces disk streaming.")
    add(f"Plan clips with the Budget Planner (VRAM budget {_fmt_gb(budget)}).")
    return "\n".join(out)


def recommend_diffusion(dit_files: dict, caps: dict, mode: str):
    """Pick the installed diffusion file best suited to the GPU, matching `mode`."""
    best, best_score = None, None
    for name, size in dit_files.items():
        info = classify_h3_file(name)
        if info["mode"] not in (mode, None):
            continue
        s = 0
        q = info["quant"]
        if q == "int8":
            s += 30
        elif q == "int4":
            s += 22
        elif q == "gguf":
            s += 20
        elif q == "fp8":
            s += 25 if caps["fp8"] else 12
        elif q == "nvfp4":
            s += 40 if caps["nvfp4"] else 5
        elif q == "bf16":
            s += 10
        if info["pruned"]:
            s += 5
        if info["mode"] == mode:
            s += 8
        if best_score is None or s > best_score:
            best, best_score = name, s
    return best


# --------------------------------------------------------------------------- cache keys
CACHE_VERSION = 1


class Hasher:
    """Small helper so cache keys are built the same way everywhere."""

    def __init__(self):
        self._h = hashlib.blake2b(digest_size=20)

    def text(self, s) -> "Hasher":
        data = ("" if s is None else str(s)).encode("utf-8")
        self._h.update(len(data).to_bytes(8, "little"))
        self._h.update(data)
        return self

    def raw(self, b) -> "Hasher":
        self._h.update(b)
        return self

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def _canon(value) -> str:
    s = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    if len(s) > 4096:
        s = "H" + hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()
    return s


def graph_signature(prompt, ref, max_nodes: int = 500):
    """Digest of the node a link points at and everything upstream of it, from the API prompt graph.

    This lets a node identify *which* model files feed a lazy input (e.g. the text encoder) without
    executing the loaders. Literal inputs (file names, seeds, widget values) are hashed; node ids are not,
    so renumbering nodes does not change the signature. Returns None if the graph is unavailable.
    """
    if not isinstance(prompt, dict) or not (isinstance(ref, (list, tuple)) and len(ref) == 2):
        return None
    memo: dict = {}
    budget = [max_nodes]

    def digest(node_id, stack):
        key = str(node_id)
        if key in memo:
            return memo[key]
        node = prompt.get(key)
        if node is None or key in stack or budget[0] <= 0:
            return "missing"
        budget[0] -= 1
        parts = [str(node.get("class_type", "?"))]
        inputs = node.get("inputs", {}) or {}
        for name in sorted(inputs):
            v = inputs[name]
            if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], int) and str(v[0]) in prompt:
                parts.append(f"{name}=@{digest(v[0], stack | {key})}:{v[1]}")
            else:
                parts.append(f"{name}={_canon(v)}")
        d = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=16).hexdigest()
        memo[key] = d
        return d

    return f"{digest(ref[0], frozenset())}:{ref[1]}"


def build_cache_key(kind: str, comfy_version: str, parts: list) -> str:
    h = Hasher().text(CACHE_VERSION).text(kind).text(comfy_version)
    for p in parts:
        h.text(_canon(p) if not isinstance(p, str) else p)
    return h.hexdigest()
