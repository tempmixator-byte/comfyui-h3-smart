"""End-to-end through ComfyUI's REAL executor (CPU): proves lazy inputs skip the loaders on a cache hit.

Loaders are counted fakes (the real ones need the 32 GB model files); the conditioning node, the native H3
node, the V3 machinery, lazy evaluation and --cache-none semantics are all real.
"""
import asyncio

import pytest

pytestmark = pytest.mark.usefixtures("require_real")


@pytest.fixture(scope="module")
def require_real(real):
    if not real:
        pytest.skip("needs ComfyUI source (COMFYUI_SRC)")


LOADS = {"clip": 0, "vae": 0}
SINK = []


class _FakeClip:
    def tokenize(self, prompt, images=None):
        return {"n": len(images or [])}

    def encode_from_tokens_scheduled(self, tokens):
        import torch
        return [[torch.randn(1, 8 + 4 * tokens["n"], 16), {"pooled_output": torch.randn(1, 16)}]]


class _FakeVae:
    def encode(self, img):
        import torch
        return torch.randn(1, 24, 1, img.shape[1] // 16, img.shape[2] // 16)


class T_FakeCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"salt": ("INT", {"default": 0})}}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self, salt):
        LOADS["clip"] += 1
        return (_FakeClip(),)


class T_FakeVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"salt": ("INT", {"default": 0})}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self, salt):
        LOADS["vae"] += 1
        return (_FakeVae(),)


class T_Sink:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive": ("CONDITIONING",), "latent": ("LATENT",), "status": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ()
    FUNCTION = "go"
    OUTPUT_NODE = True
    CATEGORY = "test"

    def go(self, positive, latent, status):
        SINK.append((positive, latent, status))
        return {}


class _Server:
    client_id = None
    last_node_id = None
    last_prompt_id = None

    def send_sync(self, *a, **k):
        pass

    def queue_updated(self, *a, **k):
        pass


@pytest.fixture()
def env(pack, monkeypatch):
    import nodes as comfy_nodes
    ext = asyncio.run(_maybe_await(pack.comfy_entrypoint()))
    for cls in asyncio.run(ext.get_node_list()):
        monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    for cls in (T_FakeCLIPLoader, T_FakeVAELoader, T_Sink):
        monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    LOADS.update(clip=0, vae=0)
    SINK.clear()
    return comfy_nodes


async def _maybe_await(x):
    return await x if asyncio.iscoroutine(x) else x


def _prompt(text="a cat", length=124):
    return {
        "1": {"class_type": "T_FakeCLIPLoader", "inputs": {"salt": 0}},
        "2": {"class_type": "T_FakeVAELoader", "inputs": {"salt": 0}},
        "3": {"class_type": "H3Smart_CachedConditioning",
              "inputs": {"clip": ["1", 0], "vae": ["2", 0], "prompt": text, "width": 288, "height": 160,
                         "length": length, "cache_mode": "read_write", "cache_tag": "", "max_cache_gb": 2.0}},
        "4": {"class_type": "T_Sink", "inputs": {"positive": ["3", 0], "latent": ["3", 1], "status": ["3", 2]}},
    }


def _run(prompt, cache_type, pid):
    import execution
    ex = execution.PromptExecutor(_Server(), cache_type=cache_type, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
    ex.execute(prompt, pid, {}, ["4"])
    assert ex.success, ex.status_messages
    return ex


@pytest.mark.parametrize("mode", ["NONE", "CLASSIC"])
def test_second_run_never_loads_the_text_encoder(env, mode):
    import execution
    ct = getattr(execution.CacheType, mode)

    _run(_prompt(), ct, "r1")
    assert LOADS == {"clip": 1, "vae": 1}
    assert SINK[-1][2].startswith("MISS")

    # --cache-none re-executes every node on each queue, but lazy inputs mean the encoder loader is skipped on a hit
    _run(_prompt(), ct, "r2")
    assert LOADS["clip"] == 1, f"text encoder loader ran again in {mode} mode"
    assert SINK[-1][2].startswith("HIT")
    assert SINK[-1][1]["samples"].is_nested

    # a different prompt must load and encode again
    _run(_prompt("a dog"), ct, "r3")
    assert LOADS["clip"] == 2 and SINK[-1][2].startswith("MISS")

    # ...and going back to the first prompt is a hit again with no loading
    _run(_prompt(), ct, "r4")
    assert LOADS["clip"] == 2 and SINK[-1][2].startswith("HIT")


def test_loader_swap_in_graph_invalidates(env):
    import execution
    ct = execution.CacheType.NONE
    _run(_prompt(), ct, "a")
    p = _prompt()
    p["1"]["inputs"]["salt"] = 7            # stands in for choosing a different encoder file
    _run(p, ct, "b")
    assert LOADS["clip"] == 2 and SINK[-1][2].startswith("MISS")
