"""Persistent-cache conditioning nodes for MiniMax H3.

These wrap ComfyUI's native MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo nodes. On a cache hit the
text encoder (and VAE) are never evaluated, because `clip`/`vae` are lazy inputs; on a miss the native node
runs unchanged and its result is stored. Because the native `execute` does the real work, behaviour and
bug-fixes follow whatever ComfyUI version is installed.
"""
from __future__ import annotations

import time

from comfy_api.latest import io, ui

from . import cond_cache, core, sysinfo

CATEGORY = "h3_smart"
GB = 1024 ** 3

CACHE_MODES = ["read_write", "refresh", "off"]
CACHE_MODE_TIP = ("read_write: reuse a matching entry, otherwise encode and store it. "
                  "refresh: always re-encode and overwrite. off: behave like the native node.")


# --------------------------------------------------------------------------- helpers
def _native(node_id: str):
    import nodes  # ComfyUI's nodes.py
    cls = nodes.NODE_CLASS_MAPPINGS.get(node_id)
    if cls is None:
        raise RuntimeError(
            f"ComfyUI's native node '{node_id}' was not found. H3 Smart wraps it, so please update ComfyUI "
            "to a version with MiniMax H3 support.")
    return cls


def _args(out):
    """NodeOutput -> tuple of results (also tolerates plain tuples from older nodes)."""
    return tuple(getattr(out, "args", out))


def _is_link(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], int)


def _graph(cls):
    h = getattr(cls, "hidden", None)
    prompt = getattr(h, "prompt", None) if h is not None else None
    uid = getattr(h, "unique_id", None) if h is not None else None
    return prompt, uid


def _own_inputs(cls) -> dict:
    prompt, uid = _graph(cls)
    if isinstance(prompt, dict) and uid is not None:
        return (prompt.get(str(uid)) or {}).get("inputs", {}) or {}
    return {}


def _linked(cls, name: str, default: bool) -> bool:
    """Is `name` wired to another node? Falls back to `default` when the graph is unavailable."""
    inputs = _own_inputs(cls)
    if not inputs:
        return default
    return _is_link(inputs.get(name))


def _signature(cls, name: str):
    prompt, _ = _graph(cls)
    ref = _own_inputs(cls).get(name)
    return core.graph_signature(prompt, ref) if _is_link(ref) else None


def _av_latent(shapes: dict):
    """Rebuild the (zero) audio-video latent from stored shapes; the sampler adds the noise."""
    import torch
    import comfy.model_management
    import comfy.nested_tensor
    dev = comfy.model_management.intermediate_device()
    video = torch.zeros(shapes["video"], device=dev)
    audio = torch.zeros(shapes["audio"], device=dev)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _latent_shapes(latent) -> dict:
    t = latent["samples"].tensors
    return {"video": list(t[0].shape), "audio": list(t[1].shape)}


def _ordered(d):
    """Autogrow dict -> [(name, value)] in numeric-suffix order, skipping unconnected slots."""
    if not d:
        return []
    def idx(n):
        try:
            return int(str(n).rsplit("_", 1)[-1])
        except ValueError:
            return 0
    return [(k, d[k]) for k in sorted(d, key=idx) if d[k] is not None]


def _identity_note(sigs: dict, tag: str) -> str:
    missing = [k for k, v in sigs.items() if v is None]
    if missing and not tag:
        return (" (could not read the loader graph for " + ", ".join(missing)
                + "; change 'cache_tag' whenever you switch encoder/VAE files)")
    return ""


class _CachedBase(io.ComfyNode):
    """Shared hit/miss logic. Subclasses supply `_key_parts` and `_run_native`."""

    KIND = ""

    @classmethod
    def _key(cls, kw: dict):
        """Cache key = native-node inputs (by content) + identity of the upstream loaders + versions."""
        sigs = cls._signatures()
        h = core.Hasher().text(core.CACHE_VERSION).text(cls.KIND).text(sysinfo.comfy_version())
        h.text(kw.get("cache_tag", ""))
        for name in sorted(sigs):
            h.text(f"{name}:{sigs[name]}")
        cls._hash_inputs(h, kw)
        return h.hexdigest(), sigs

    @classmethod
    def _lookup(cls, kw: dict):
        if kw.get("cache_mode", "read_write") != "read_write":
            return None, None, None
        key, sigs = cls._key(kw)
        return cond_cache.load(key), key, sigs

    @classmethod
    def _finish(cls, kw, key, sigs, cond, latent, t_start):
        status = "cache off"
        mode = kw.get("cache_mode", "read_write")
        if mode != "off":
            max_bytes = int(float(kw.get("max_cache_gb", 2.0)) * GB)
            ok = cond_cache.save(key, cond, _latent_shapes(latent),
                                 {"kind": cls.KIND, "comfy": sysinfo.comfy_version()}, max_bytes)
            status = ("MISS - encoded and stored in %.1fs" if ok else "MISS - encoded (not cacheable) in %.1fs") % (time.time() - t_start)
            status += _identity_note(sigs, kw.get("cache_tag", ""))
        return status

    @classmethod
    def _report(cls, status: str) -> None:
        print(f"[H3Smart] {cls.KIND} conditioning: {status}")


class H3Smart_CachedConditioning(_CachedBase):
    KIND = "fl2va"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3Smart_CachedConditioning",
            display_name="H3 Cached Conditioning (T2V / I2V / FL2V)",
            category=f"{CATEGORY}/conditioning",
            description=("Drop-in replacement for MiniMax H3 Image to Video that remembers its result on disk. "
                         "An unchanged prompt/frames/size skips the text encoder entirely - ideal with --cache-none "
                         "when you iterate on seed, steps or LoRA."),
            inputs=[
                io.Clip.Input("clip", lazy=True, tooltip="H3 text encoder. Not even loaded on a cache hit."),
                io.Vae.Input("vae", lazy=True, tooltip="H3 video VAE (used to encode keyframes on a cache miss)."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Frame count at 24 fps (124 = ~5 s). Snapped to the 17k+5 grid like the native node."),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Combo.Input("cache_mode", options=CACHE_MODES, default="read_write", tooltip=CACHE_MODE_TIP),
                io.String.Input("cache_tag", default="", advanced=True,
                                tooltip="Mixed into the cache key. Change it to force fresh entries."),
                io.Float.Input("max_cache_gb", default=2.0, min=0.1, max=100.0, step=0.1, advanced=True,
                               tooltip="Least-recently-used entries are deleted beyond this size."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(),
                     io.String.Output(display_name="cache_status")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def _signatures(cls):
        return {"clip": _signature(cls, "clip"), "vae": _signature(cls, "vae")}

    @classmethod
    def _hash_inputs(cls, h, kw):
        frames = core.temporal_shape(kw.get("length", 124))[0]
        h.text(kw.get("prompt", "")).text(kw.get("width")).text(kw.get("height")).text(frames)
        for name in ("first_frame", "last_frame"):
            cond_cache.digest_value(h.text(name), kw.get(name))

    @classmethod
    def check_lazy_status(cls, **kw):
        hit, _, _ = cls._lookup(kw)
        if hit is not None:
            return []
        need = ["clip"]
        if kw.get("vae") is None and _linked(cls, "vae", True):
            need.append("vae")
        return [n for n in need if kw.get(n) is None]

    @classmethod
    def execute(cls, clip=None, vae=None, prompt="", width=1344, height=768, length=124,
                first_frame=None, last_frame=None, cache_mode="read_write", cache_tag="",
                max_cache_gb=2.0) -> io.NodeOutput:
        kw = dict(prompt=prompt, width=width, height=height, length=length, first_frame=first_frame,
                  last_frame=last_frame, cache_mode=cache_mode, cache_tag=cache_tag, max_cache_gb=max_cache_gb)
        t0 = time.time()
        entry, key, sigs = cls._lookup(kw)
        if entry is not None:
            status = f"HIT - encoder skipped ({time.time() - t0:.2f}s)"
            cls._report(status)
            return io.NodeOutput(entry["cond"], _av_latent(entry["latent_shapes"]), status,
                                 ui=ui.PreviewText(status))
        if key is None and cache_mode != "off":
            key, sigs = cls._key(kw)
        if clip is None:
            raise RuntimeError("H3 Cached Conditioning needs a text encoder (clip) on a cache miss; connect the CLIP loader.")
        out = _args(_native("MiniMaxH3ImageToVideo").execute(
            clip=clip, vae=vae, prompt=prompt, width=width, height=height, length=length,
            first_frame=first_frame, last_frame=last_frame))
        cond, latent = out[0], out[1]
        status = cls._finish(kw, key, sigs, cond, latent, t0)
        cls._report(status)
        return io.NodeOutput(cond, latent, status, ui=ui.PreviewText(status))


class H3Smart_CachedReferenceConditioning(_CachedBase):
    KIND = "ref2va"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3Smart_CachedReferenceConditioning",
            display_name="H3 Cached Conditioning (Reference to Video)",
            category=f"{CATEGORY}/conditioning",
            description=("Drop-in replacement for MiniMax H3 Reference to Video with an on-disk cache. "
                         "Reference images/videos/audio are hashed by content, so editing a reference invalidates the entry."),
            inputs=[
                io.Clip.Input("clip", lazy=True, tooltip="H3 text encoder. Not even loaded on a cache hit."),
                io.Vae.Input("vae", optional=True, lazy=True, tooltip="Video VAE (encodes reference images/videos)."),
                io.Vae.Input("audio_vae", optional=True, lazy=True, tooltip="Audio VAE (encodes reference audio)."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Frame count at 24 fps (124 = ~5 s, trained range ~124-362)."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="'match' scales references to the generation's pixel area; 'max' uses a 2048 px short edge "
                                       "(better identity, but reference tokens ride through every step and can be much slower)."),
                io.Autogrow.Input("ref_images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15 s)"),
                    prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"),
                    prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                    prefix="ref_audio_", min=0, max=3)),
                io.Combo.Input("cache_mode", options=CACHE_MODES, default="read_write", tooltip=CACHE_MODE_TIP),
                io.String.Input("cache_tag", default="", advanced=True,
                                tooltip="Mixed into the cache key. Change it to force fresh entries."),
                io.Float.Input("max_cache_gb", default=2.0, min=0.1, max=100.0, step=0.1, advanced=True,
                               tooltip="Least-recently-used entries are deleted beyond this size."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(),
                     io.String.Output(display_name="cache_status")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def _signatures(cls):
        return {"clip": _signature(cls, "clip"), "vae": _signature(cls, "vae"),
                "audio_vae": _signature(cls, "audio_vae")}

    @classmethod
    def _hash_inputs(cls, h, kw):
        frames = core.temporal_shape(kw.get("length", 124))[0]
        h.text(kw.get("prompt", "")).text(kw.get("width")).text(kw.get("height")).text(frames)
        h.text(kw.get("ref_image_size", "match"))
        for group in ("ref_images", "ref_videos", "ref_video_audios", "ref_audios"):
            h.text(group)
            for name, value in _ordered(kw.get(group)):
                cond_cache.digest_value(h.text(name), value)

    @classmethod
    def check_lazy_status(cls, **kw):
        hit, _, _ = cls._lookup(kw)
        if hit is not None:
            return []
        need = ["clip"]
        for n in ("vae", "audio_vae"):
            if kw.get(n) is None and _linked(cls, n, False):
                need.append(n)
        return [n for n in need if kw.get(n) is None]

    @classmethod
    def execute(cls, clip=None, vae=None, audio_vae=None, prompt="", width=1344, height=768, length=124,
                ref_image_size="match", ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None, cache_mode="read_write", cache_tag="", max_cache_gb=2.0) -> io.NodeOutput:
        kw = dict(prompt=prompt, width=width, height=height, length=length, ref_image_size=ref_image_size,
                  ref_images=ref_images, ref_videos=ref_videos, ref_video_audios=ref_video_audios,
                  ref_audios=ref_audios, cache_mode=cache_mode, cache_tag=cache_tag, max_cache_gb=max_cache_gb)
        t0 = time.time()
        entry, key, sigs = cls._lookup(kw)
        if entry is not None:
            status = f"HIT - encoder skipped ({time.time() - t0:.2f}s)"
            cls._report(status)
            return io.NodeOutput(entry["cond"], _av_latent(entry["latent_shapes"]), status,
                                 ui=ui.PreviewText(status))
        if key is None and cache_mode != "off":
            key, sigs = cls._key(kw)
        if clip is None:
            raise RuntimeError("H3 Cached Conditioning needs a text encoder (clip) on a cache miss; connect the CLIP loader.")
        out = _args(_native("MiniMaxH3ReferenceToVideo").execute(
            clip=clip, prompt=prompt, width=width, height=height, length=length, ref_image_size=ref_image_size,
            vae=vae, audio_vae=audio_vae, ref_images=ref_images, ref_videos=ref_videos,
            ref_video_audios=ref_video_audios, ref_audios=ref_audios))
        cond, latent = out[0], out[1]
        status = cls._finish(kw, key, sigs, cond, latent, t0)
        cls._report(status)
        return io.NodeOutput(cond, latent, status, ui=ui.PreviewText(status))


class H3Smart_CondCacheManager(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3Smart_CondCacheManager",
            display_name="H3 Conditioning Cache Manager",
            category=f"{CATEGORY}/system",
            description="Shows how much conditioning is cached on disk, or clears it.",
            inputs=[io.Combo.Input("action", options=["show", "clear_all"], default="show",
                                   tooltip="clear_all deletes every cached conditioning entry.")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return time.time_ns()          # always run: the cache changes between queues

    @classmethod
    def execute(cls, action="show") -> io.NodeOutput:
        removed = cond_cache.clear() if action == "clear_all" else 0
        items = cond_cache.entries()
        total = sum(s for _, s, _ in items)
        lines = [f"Cache folder: {cond_cache.cache_dir()}"]
        if action == "clear_all":
            lines.append(f"Removed {removed} entries.")
        lines.append(f"{len(items)} entries, {total / (1024 ** 2):.1f} MB")
        text = "\n".join(lines)
        return io.NodeOutput(text, ui=ui.PreviewText(text))
