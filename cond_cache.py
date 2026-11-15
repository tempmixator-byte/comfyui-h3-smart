"""Persistent on-disk cache for H3 conditioning (text-encoder output + encoded keyframes/references).

Why: with `--cache-none` (or after a restart) ComfyUI re-runs the whole text encoder for every queued
prompt. For H3 that encoder is a 32B-class model (15.7 GB even in NVFP4), which on an 8 GB card is
mostly weight streaming. The conditioning it produces is a few MB, so keeping it on disk turns
"change the seed / steps / LoRA" iterations into a cache hit that never loads the encoder at all.

Safety rules:
  * entries are written atomically (tmp file + os.replace)
  * entries are loaded with torch.load(weights_only=True); anything else is treated as a miss
  * only plain containers + tensors + scalars are cached; anything unusual disables caching for that run
  * the cache is pruned oldest-first (by last use) to stay under a size limit
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from . import core

SUFFIX = ".h3cond"
_PRIMITIVES = (int, float, str, bool, type(None))


def cache_dir() -> Path:
    override = os.environ.get("H3_SMART_CACHE_DIR")
    if override:
        base = Path(override)
    else:
        try:
            import folder_paths  # type: ignore
            base = Path(folder_paths.get_user_directory()) / "h3_smart" / "cond_cache"
        except Exception:
            base = Path.home() / ".cache" / "h3_smart" / "cond_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _path(key: str) -> Path:
    return cache_dir() / f"{key}{SUFFIX}"


# ------------------------------------------------------------------ tree helpers
def is_cacheable(obj, _depth=0) -> bool:
    """True if obj consists only of lists/tuples/str-keyed dicts of tensors and plain scalars."""
    if _depth > 12:
        return False
    if isinstance(obj, _PRIMITIVES):
        return True
    if isinstance(obj, (list, tuple)):
        return all(is_cacheable(x, _depth + 1) for x in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and is_cacheable(v, _depth + 1) for k, v in obj.items())
    try:
        import torch
        return isinstance(obj, torch.Tensor) and not getattr(obj, "is_nested", False)
    except Exception:
        return False


def to_cpu(obj):
    """Detach every tensor and move it to CPU (cached conditioning never needs to live on the GPU)."""
    try:
        import torch
    except Exception:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu").contiguous()
    if isinstance(obj, list):
        return [to_cpu(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(x) for x in obj)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj


# ------------------------------------------------------------------ hashing of inputs
def digest_tensor(h: core.Hasher, t) -> None:
    """Hash a tensor's shape, dtype and exact contents without making a second copy."""
    import torch
    if t is None:
        h.text("none")
        return
    t = t.detach()
    h.text(tuple(t.shape)).text(str(t.dtype))
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.float()
    arr = t.to("cpu").contiguous().numpy()
    flat = arr.reshape(-1)
    step = 1 << 24          # hash in 16M-element slices so huge videos never need a giant temporary
    for i in range(0, flat.shape[0], step):
        h.raw(memoryview(flat[i:i + step]).cast("B"))


def digest_value(h: core.Hasher, v) -> None:
    """Hash an input that may be a tensor, an AUDIO dict, a list of those, or a plain scalar."""
    try:
        import torch
        is_tensor = isinstance(v, torch.Tensor)
    except Exception:
        is_tensor = False
    if v is None:
        h.text("none")
    elif is_tensor:
        digest_tensor(h, v)
    elif isinstance(v, dict):
        for k in sorted(v, key=str):
            h.text(k)
            digest_value(h, v[k])
    elif isinstance(v, (list, tuple)):
        h.text(len(v))
        for x in v:
            digest_value(h, x)
    else:
        h.text(repr(v))


# ------------------------------------------------------------------ store
def load(key: str):
    """Return the cached payload dict or None. Corrupt/untrusted entries are deleted and reported as a miss."""
    p = _path(key)
    if not p.exists():
        return None
    try:
        import torch
        payload = torch.load(p, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("v") != core.CACHE_VERSION or "cond" not in payload:
            raise ValueError("unexpected payload")
        os.utime(p, None)                      # mark as recently used (LRU)
        return payload
    except Exception as e:                      # noqa: BLE001
        print(f"[H3Smart] discarding unreadable cache entry {p.name}: {e}")
        try:
            p.unlink()
        except OSError:
            pass
        return None


def save(key: str, cond, latent_shapes: dict, meta: dict, max_bytes: int) -> bool:
    """Write an entry atomically. Returns False (and writes nothing) if the conditioning is not cacheable."""
    cpu_cond = to_cpu(cond)
    if not is_cacheable(cpu_cond):
        print("[H3Smart] conditioning contains non-serialisable objects; not cached")
        return False
    try:
        import torch
        p = _path(key)
        tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
        torch.save({"v": core.CACHE_VERSION, "cond": cpu_cond, "latent_shapes": latent_shapes,
                    "meta": dict(meta, saved_at=time.time())}, tmp)
        os.replace(tmp, p)
        prune(max_bytes)
        return True
    except Exception as e:                      # noqa: BLE001
        print(f"[H3Smart] could not write cache entry: {e}")
        return False


def entries():
    d = cache_dir()
    out = []
    for f in d.glob(f"*{SUFFIX}"):
        try:
            st = f.stat()
            out.append((f, st.st_size, st.st_mtime))
        except OSError:
            pass
    return out


def total_bytes() -> int:
    return sum(s for _, s, _ in entries())


def prune(max_bytes: int) -> int:
    """Delete least-recently-used entries until the cache fits. Returns the number removed."""
    items = sorted(entries(), key=lambda t: t[2])   # oldest first
    total = sum(s for _, s, _ in items)
    removed = 0
    while items and total > max_bytes:
        f, size, _ = items.pop(0)
        try:
            f.unlink()
            total -= size
            removed += 1
        except OSError:
            pass
    return removed


def clear() -> int:
    n = 0
    for f, _, _ in entries():
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n
