"""Integration tests against the real comfy_api and the real native H3 nodes (CPU).

Skipped automatically when ComfyUI source is not available (set COMFYUI_SRC to run them).
"""
import asyncio

import pytest

pytestmark = pytest.mark.usefixtures("require_real")


@pytest.fixture(scope="module")
def require_real(real):
    if not real:
        pytest.skip("needs ComfyUI source (COMFYUI_SRC)")


@pytest.fixture(scope="module")
def io_():
    from comfy_api.latest import io
    return io


@pytest.fixture(scope="module")
def node_classes(pack):
    ext = pack.comfy_entrypoint
    e = asyncio.run(_ep(ext))
    return {c.__name__: c for c in asyncio.run(e.get_node_list())}


async def _ep(fn):
    r = fn()
    return await r if asyncio.iscoroutine(r) else r


# --------------------------------------------------------------------------- registration / schemas
def test_all_nodes_register_with_real_schema_validation(node_classes):
    assert len(node_classes) == 8
    for name, cls in node_classes.items():
        info = cls.GET_NODE_INFO_V1()                    # real V3 -> V1 conversion: validates every input/output
        assert info["name"] == name
        assert name.startswith("H3Smart_")
        assert info["category"].startswith("h3_smart")
        cls()                                            # VALIDATE_CLASS: needs define_schema + execute


def _input_types(cls):
    schema = cls.GET_SCHEMA()
    out = {}
    for i in schema.inputs:
        out[i.id] = type(i).__qualname__.split(".")[0]
    return out


def test_cached_nodes_accept_every_native_input(node_classes):
    """The wrapper must be a drop-in: same input names/types as the native node, plus cache options."""
    import nodes as comfy_nodes
    pairs = [("H3Smart_CachedConditioning", "MiniMaxH3ImageToVideo"),
             ("H3Smart_CachedReferenceConditioning", "MiniMaxH3ReferenceToVideo")]
    for mine, native in pairs:
        mine_t = _input_types(node_classes[mine])
        native_t = _input_types(comfy_nodes.NODE_CLASS_MAPPINGS[native])
        missing = {k: v for k, v in native_t.items() if mine_t.get(k) != v}
        assert not missing, f"{mine} differs from native {native}: {missing}"


def test_lazy_flags(node_classes):
    s = node_classes["H3Smart_CachedConditioning"].GET_SCHEMA()
    lazy = {i.id for i in s.inputs if getattr(i, "lazy", None)}
    assert lazy == {"clip", "vae"}


# --------------------------------------------------------------------------- cache behaviour
class _FakeNative:
    """Stands in for the native node (the real one needs the 32B encoder). Counts how often it runs."""
    calls = 0

    @classmethod
    def execute(cls, **kw):
        import torch
        import comfy.nested_tensor
        from comfy_api.latest import io
        cls.calls += 1
        cond = [[torch.randn(1, 12, 8), {"minimax_token_tags": torch.arange(12),
                                           "minimax_keyframes": [{"kind": "first", "latent": torch.randn(1, 24, 1, 4, 4)}],
                                           "note": "x"}]]
        video = torch.zeros(1, 24, 7, 4, 4)
        audio = torch.zeros(1, 32, 2, 40)
        return io.NodeOutput(cond, {"samples": comfy.nested_tensor.NestedTensor((video, audio))})


@pytest.fixture()
def fake_native(monkeypatch):
    import nodes as comfy_nodes
    _FakeNative.calls = 0
    monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, "MiniMaxH3ImageToVideo", _FakeNative)
    monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, "MiniMaxH3ReferenceToVideo", _FakeNative)
    return _FakeNative


def _with_graph(cls, io, graph, uid):
    """Clone the class with hidden inputs set, exactly like the executor does."""
    return cls.PREPARE_CLASS_CLONE({"hidden_inputs": {io.Hidden.prompt: graph, io.Hidden.unique_id: uid}})


GRAPH = {
    "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "te.safetensors", "type": "minimax"}},
    "2": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
    "3": {"class_type": "H3Smart_CachedConditioning", "inputs": {"clip": ["1", 0], "vae": ["2", 0], "prompt": "a cat"}},
}


def test_miss_then_hit_skips_encoder_and_rebuilds_latent(node_classes, io_, fake_native):
    import torch
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    clip, vae = object(), object()
    kw = dict(prompt="a cat", width=864, height=480, length=124)

    r1 = cls.execute(clip=clip, vae=vae, **kw)
    assert fake_native.calls == 1 and r1[2].startswith("MISS")

    # the executor asks which lazy inputs are needed: on a hit, none
    assert cls.check_lazy_status(clip=None, vae=None, **kw) == []
    r2 = cls.execute(clip=None, vae=None, **kw)
    assert fake_native.calls == 1                       # native node (encoder) not run again
    assert r2[2].startswith("HIT")
    # identical content
    assert torch.equal(r1[0][0][0], r2[0][0][0])
    assert torch.equal(r1[0][0][1]["minimax_token_tags"], r2[0][0][1]["minimax_token_tags"])
    assert r2[0][0][1]["note"] == "x"
    # latent rebuilt with the stored shapes, as a nested tensor
    lat = r2[1]["samples"]
    assert lat.is_nested
    assert list(lat.tensors[0].shape) == [1, 24, 7, 4, 4] and list(lat.tensors[1].shape) == [1, 32, 2, 40]


def test_check_lazy_status_requests_loaders_on_miss(node_classes, io_, fake_native):
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    need = cls.check_lazy_status(clip=None, vae=None, prompt="never seen", width=864, height=480, length=124)
    assert set(need) == {"clip", "vae"}
    # once they are provided there is nothing left to ask for
    assert cls.check_lazy_status(clip=object(), vae=object(), prompt="never seen", width=864, height=480, length=124) == []


@pytest.mark.parametrize("change", [
    dict(prompt="a dog"), dict(width=960), dict(height=544), dict(length=141), dict(cache_tag="v2"),
])
def test_any_input_change_misses(node_classes, io_, fake_native, change):
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    kw = dict(prompt="a cat", width=864, height=480, length=124)
    cls.execute(clip=object(), vae=object(), **kw)
    kw.update(change)
    r = cls.execute(clip=object(), vae=object(), **kw)
    assert r[2].startswith("MISS") and fake_native.calls == 2


def test_length_on_same_grid_shares_entry(node_classes, io_, fake_native):
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    base = dict(prompt="p", width=864, height=480)
    cls.execute(clip=object(), vae=object(), length=120, **base)      # snaps up to 124
    r = cls.execute(clip=None, vae=None, length=124, **base)
    assert r[2].startswith("HIT")


def test_image_content_is_part_of_the_key(node_classes, io_, fake_native):
    import torch
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    img = torch.rand(1, 64, 96, 3)
    base = dict(prompt="p", width=96, height=64, length=124)
    cls.execute(clip=object(), vae=object(), first_frame=img, **base)
    assert cls.execute(clip=None, vae=None, first_frame=img.clone(), **base)[2].startswith("HIT")
    edited = img.clone()
    edited[0, 0, 0, 0] += 0.01
    assert cls.check_lazy_status(clip=None, vae=None, first_frame=edited, **base) != []
    assert cls.check_lazy_status(clip=None, vae=None, **base) != []       # no first frame at all -> different key


def test_loader_change_invalidates_via_graph_signature(node_classes, io_, fake_native):
    base_cls = node_classes["H3Smart_CachedConditioning"]
    kw = dict(prompt="a cat", width=864, height=480, length=124)
    a = _with_graph(base_cls, io_, GRAPH, "3")
    a.execute(clip=object(), vae=object(), **kw)
    assert a.check_lazy_status(clip=None, vae=None, **kw) == []
    g2 = {k: dict(v, inputs=dict(v["inputs"])) for k, v in GRAPH.items()}
    g2["1"]["inputs"]["clip_name"] = "another_encoder_q4.safetensors"
    b = _with_graph(base_cls, io_, g2, "3")
    assert set(b.check_lazy_status(clip=None, vae=None, **kw)) == {"clip", "vae"}


def test_refresh_and_off_modes(node_classes, io_, fake_native):
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    kw = dict(prompt="a cat", width=864, height=480, length=124)
    cls.execute(clip=object(), vae=object(), **kw)
    assert cls.execute(clip=object(), vae=object(), cache_mode="refresh", **kw)[2].startswith("MISS")
    assert fake_native.calls == 2
    r = cls.execute(clip=object(), vae=object(), cache_mode="off", **kw)
    assert r[2] == "cache off" and fake_native.calls == 3
    assert set(cls.check_lazy_status(clip=None, vae=None, cache_mode="off", **kw)) == {"clip", "vae"}


def test_miss_without_clip_gives_clear_error(node_classes, io_, fake_native):
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    with pytest.raises(RuntimeError, match="text encoder"):
        cls.execute(clip=None, vae=None, prompt="x", width=864, height=480, length=124)


def test_corrupt_entry_is_discarded_not_crashing(node_classes, io_, fake_native, tmp_path):
    from h3_smart_under_test import cond_cache
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    kw = dict(prompt="a cat", width=864, height=480, length=124)
    cls.execute(clip=object(), vae=object(), **kw)
    for f, _, _ in cond_cache.entries():
        f.write_bytes(b"not a torch file")
    r = cls.execute(clip=object(), vae=object(), **kw)
    assert r[2].startswith("MISS") and fake_native.calls == 2


def test_unsafe_pickle_is_rejected(node_classes, io_, fake_native):
    """weights_only loading must refuse arbitrary objects planted in the cache folder."""
    import torch
    from h3_smart_under_test import cond_cache, core
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    kw = dict(prompt="a cat", width=864, height=480, length=124)
    cls.execute(clip=object(), vae=object(), **kw)

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    for f, _, _ in cond_cache.entries():
        torch.save({"v": core.CACHE_VERSION, "cond": Evil(), "latent_shapes": {}}, f)
    assert cond_cache.load(f.stem) is None


def test_lru_prune(node_classes, io_, fake_native):
    import os, time
    from h3_smart_under_test import cond_cache
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    for i in range(3):
        cls.execute(clip=object(), vae=object(), prompt=f"p{i}", width=864, height=480, length=124)
        time.sleep(0.02)
    files = sorted(cond_cache.entries(), key=lambda t: t[2])
    size = files[0][1]
    assert cond_cache.prune(int(size * 1.5)) == 2          # keeps only the newest
    assert len(cond_cache.entries()) == 1


def test_non_serialisable_conditioning_is_not_cached(node_classes, io_, monkeypatch):
    import nodes as comfy_nodes
    import torch, comfy.nested_tensor

    class Weird:
        @classmethod
        def execute(cls, **kw):
            lat = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 32, 2, 4)))}
            return io_.NodeOutput([[torch.zeros(1), {"obj": object()}]], lat)

    monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, "MiniMaxH3ImageToVideo", Weird)
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    r = cls.execute(clip=object(), vae=object(), prompt="x", width=64, height=64, length=124)
    assert "not cacheable" in r[2]


def test_reference_node_hashes_autogrow_content(node_classes, io_, fake_native):
    import torch
    cls = _with_graph(node_classes["H3Smart_CachedReferenceConditioning"], io_,
                      {"1": GRAPH["1"], "2": GRAPH["2"],
                       "3": {"class_type": "H3Smart_CachedReferenceConditioning",
                             "inputs": {"clip": ["1", 0], "vae": ["2", 0], "prompt": "p"}}}, "3")
    a, b = torch.rand(1, 32, 32, 3), torch.rand(1, 32, 32, 3)
    kw = dict(prompt="p", width=864, height=480, length=124)
    cls.execute(clip=object(), vae=object(), ref_images={"ref_image_1": a}, **kw)
    assert cls.execute(clip=None, vae=None, ref_images={"ref_image_1": a.clone()}, **kw)[2].startswith("HIT")
    assert cls.check_lazy_status(clip=None, vae=None, ref_images={"ref_image_1": b}, **kw) != []
    assert cls.check_lazy_status(clip=None, vae=None, ref_images={"ref_image_1": a, "ref_image_2": b}, **kw) != []
    # audio_vae is not wired in this graph, so it must never be requested (would crash the executor)
    assert "audio_vae" not in cls.check_lazy_status(clip=None, vae=None, ref_images={"ref_image_1": b}, **kw)


# --------------------------------------------------------------------------- other nodes
def test_planner_executes(node_classes):
    r = node_classes["H3Smart_BudgetPlanner"].execute(
        aspect_ratio="16:9", megapixels=0.98, duration_seconds=8.0, fit_strategy="reduce_resolution",
        vram_budget_override_gb=5.5)
    w, h, length, report, status = r.args
    assert w % 32 == 0 and h % 32 == 0 and length % 17 == 5
    assert (w, h) != (1344, 768)
    assert status in ("ok", "tight", "over")
    assert "Est. VRAM" in report


def test_planner_match_image(node_classes):
    import torch
    r = node_classes["H3Smart_BudgetPlanner"].execute(
        aspect_ratio="match_image", megapixels=0.4, duration_seconds=5.0, image=torch.zeros(1, 1920, 1080, 3),
        vram_budget_override_gb=6.0)
    w, h = r.args[0], r.args[1]
    assert h > w                                          # portrait image -> portrait canvas


def test_prompt_builder_executes_and_reports_errors(node_classes):
    cls = node_classes["H3Smart_ShotPromptBuilder"]
    r = cls.execute(mode="FL2VA", length_frames=192, style="Live-action, cinematic",
                    overall_soundscape="Rain falls.", non_diegetic_music="N/A",
                    shot1_text="a cyclist opens an umbrella.", shot1_camera="pull out",
                    shot1_amplitude="small", shot1_speed="slow", shot2_text="", shot3_text="")
    prompt, warns = r.args
    assert "8.00-second mark" in prompt and "pulls out with small amplitude at slow speed" in prompt
    with pytest.raises(ValueError, match="cut time"):
        cls.execute(mode="T2VA", length_frames=124, style="", overall_soundscape="x.", non_diegetic_music="N/A",
                    shot1_text="a", shot2_text="b", shot2_cut_seconds=0.0)


def test_memory_barrier_passes_value_through(node_classes):
    value = object()
    r = node_classes["H3Smart_MemoryBarrier"].execute(value=value, unload_models=True, collect_garbage=True,
                                                      trim_process_memory=True, label="t")
    assert r.args[0] is value


def test_advisor_runs_without_gpu(node_classes):
    r = node_classes["H3Smart_SystemAdvisor"].execute(task="fl2va")
    assert "== System ==" in r.args[0]


def test_cache_manager_show_and_clear(node_classes, io_, fake_native):
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    cls.execute(clip=object(), vae=object(), prompt="p", width=864, height=480, length=124)
    mgr = node_classes["H3Smart_CondCacheManager"]
    assert "1 entries" in mgr.execute(action="show").args[0]
    assert "Removed 1" in mgr.execute(action="clear_all").args[0]
    assert "0 entries" in mgr.execute(action="show").args[0]


def _fake_model():
    """Smallest object that satisfies H3Smart_SamplingSetup: provides a real flow model_sampling."""
    import comfy.model_sampling as ms

    class Cfg:
        sampling_settings = {"shift": 3.0, "multiplier": 1.0}

    class Sampling(ms.ModelSamplingDiscreteFlow, ms.CONST):
        pass

    sampling = Sampling(Cfg)

    class Model:
        def get_model_object(self, name):
            assert name == "model_sampling"
            return sampling

    return Model()


def test_sampling_setup_balanced_no_lora(node_classes):
    r = node_classes["H3Smart_SamplingSetup"].execute(model=_fake_model(), task="fl2va", speed="balanced")
    model, sampler, sigmas, steps, info = r.args
    assert steps == 20 and len(sigmas) == 21
    assert float(sigmas[-1]) == 0.0
    assert "LoRA: none" in info


def test_sampling_setup_turbo_without_lora_file_is_actionable(node_classes):
    with pytest.raises(RuntimeError, match="Turbo LoRA"):
        node_classes["H3Smart_SamplingSetup"].execute(model=_fake_model(), task="fl2va", speed="turbo")


# --------------------------------------------------------------------------- wrapper <-> REAL native node
class _FakeClip:
    """Behaves like a CLIP for the native node: tokenize + encode_from_tokens_scheduled."""
    def __init__(self):
        self.encodes = 0

    def tokenize(self, prompt, images=None):
        return {"prompt": prompt, "n_images": len(images or [])}

    def encode_from_tokens_scheduled(self, tokens):
        import torch
        self.encodes += 1
        n = 8 + 4 * tokens["n_images"]
        return [[torch.randn(1, n, 16), {"pooled_output": torch.randn(1, 16), "minimax_token_tags": torch.arange(n)}]]


class _FakeVae:
    def encode(self, img):
        import torch
        # real VAE: [1,H,W,3] pixels -> [1,24,1,H/16,W/16] latent
        return torch.randn(1, 24, 1, img.shape[1] // 16, img.shape[2] // 16)


def test_wrapper_with_real_native_node_roundtrip(node_classes, io_):
    import torch
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    clip, vae = _FakeClip(), _FakeVae()
    first, last = torch.rand(1, 100, 180, 3), torch.rand(1, 120, 120, 3)
    kw = dict(prompt="a cat", width=288, height=160, length=124, first_frame=first, last_frame=last)

    miss = cls.execute(clip=clip, vae=vae, **kw)
    assert miss[2].startswith("MISS") and clip.encodes == 1

    assert cls.check_lazy_status(clip=None, vae=None, **kw) == []
    hit = cls.execute(clip=None, vae=None, **kw)
    assert hit[2].startswith("HIT") and clip.encodes == 1            # encoder untouched on the second run

    kf_miss = miss[0][0][1]["minimax_keyframes"]
    kf_hit = hit[0][0][1]["minimax_keyframes"]
    assert [k["resolved_frame_index"] for k in kf_hit] == [0, 123]
    for a, b in zip(kf_miss, kf_hit):
        assert torch.equal(a["latent"], b["latent"])
    assert torch.equal(miss[0][0][0], hit[0][0][0])

    # the rebuilt latent has exactly the shape the real native node produced
    ms, hs = miss[1]["samples"], hit[1]["samples"]
    assert [list(t.shape) for t in ms.tensors] == [list(t.shape) for t in hs.tensors]
    assert hs.tensors[0].shape == (1, 24, 37, 160 // 16, 288 // 16)


def test_wrapper_output_matches_native_node_on_a_miss(node_classes, io_):
    """On a miss the wrapper must return exactly what the native node returns."""
    import nodes as comfy_nodes
    import torch
    native = comfy_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3ImageToVideo"]
    cls = _with_graph(node_classes["H3Smart_CachedConditioning"], io_, GRAPH, "3")
    torch.manual_seed(0)
    a = native.execute(clip=_FakeClip(), vae=_FakeVae(), prompt="p", width=96, height=64, length=22)
    torch.manual_seed(0)
    b = cls.execute(clip=_FakeClip(), vae=_FakeVae(), prompt="p", width=96, height=64, length=22)
    assert torch.equal(a[0][0][0], b[0][0][0])
    assert [list(t.shape) for t in a[1]["samples"].tensors] == [list(t.shape) for t in b[1]["samples"].tensors]


def test_planner_geometry_agrees_with_native_latent(node_classes):
    """core.temporal_shape / token maths must agree with what the native node really builds."""
    from h3_smart_under_test import core
    import nodes as comfy_nodes
    native = comfy_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3ImageToVideo"]
    for w, h, length in [(864, 480, 124), (1344, 768, 124), (608, 352, 192), (736, 416, 107)]:
        out = native.execute(clip=_FakeClip(), vae=_FakeVae(), prompt="p", width=w, height=h, length=length)
        video, audio = out[1]["samples"].tensors
        fc, lat_t, audio_t = core.temporal_shape(length)
        assert video.shape[2] == lat_t, (w, h, length)
        assert audio.shape[-1] == audio_t, (w, h, length)
        # one token per 32x32 px per latent frame (latent /16, patch 2x2)
        assert video.shape[3] // 2 * (video.shape[4] // 2) == core.tokens_per_latent_frame(w, h)


def test_native_signatures_still_accept_what_we_pass():
    import inspect
    import nodes as comfy_nodes
    fl = inspect.signature(comfy_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3ImageToVideo"].execute).parameters
    for k in ("clip", "vae", "prompt", "width", "height", "length", "first_frame", "last_frame"):
        assert k in fl
    r2 = inspect.signature(comfy_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3ReferenceToVideo"].execute).parameters
    for k in ("clip", "vae", "audio_vae", "prompt", "width", "height", "length", "ref_image_size",
              "ref_images", "ref_videos", "ref_video_audios", "ref_audios"):
        assert k in r2
