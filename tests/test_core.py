"""Pure-logic tests (no ComfyUI needed)."""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name):
    # load submodules of the (hyphen-named) pack with relative imports intact
    pkg = "h3_smart_pure"
    if pkg not in sys.modules:
        spec = importlib.util.spec_from_loader(pkg, loader=None, is_package=True)
        mod = importlib.util.module_from_spec(spec)
        mod.__path__ = [str(ROOT)]
        sys.modules[pkg] = mod
    return importlib.import_module(f"{pkg}.{name}")


import importlib  # noqa: E402

core = _load("core")
prompting = _load("prompting")


# ---------------------------------------------------------------- frame grid / geometry
@pytest.mark.parametrize("n,expected", [(5, 5), (6, 22), (22, 22), (23, 39), (120, 124), (124, 124), (125, 141)])
def test_align_up(n, expected):
    assert core.align_up(n) == expected
    assert core.align_up(n) % 17 == 5


def test_align_down_and_snap_modes():
    assert core.align_down(120) == 107
    assert core.frames_from_seconds(5, "up") == 124
    assert core.frames_from_seconds(5, "down") == 107
    assert core.frames_from_seconds(5, "nearest") == 124      # 120 is 4 above 107... closer to 124
    assert core.frames_from_seconds(4.6, "nearest") in (107, 124)


def test_latent_geometry_matches_native_formula():
    # native: latent_t = ((fc-5)//17)*5+2 ; 124 frames -> 37 latent frames
    assert core.video_latent_t(124) == 37
    assert core.video_latent_t(5) == 2
    fc, lt, at = core.temporal_shape(120)
    assert (fc, lt) == (124, 37)
    assert at == round(124 / 24 * 40)


# Golden values copied from ComfyUI's official H3 template "Size Settings Reference".
@pytest.mark.parametrize("mp,expected", [
    (0.2, (608, 352)), (0.3, (736, 416)), (0.4, (864, 480)), (0.5, (960, 544)), (0.6, (1056, 608)),
    (0.7, (1152, 640)), (0.8, (1216, 672)), (0.9, (1280, 736)), (0.98, (1344, 768)),
])
def test_canvas_matches_official_table(mp, expected):
    assert core.solve_canvas(16, 9, mp) == expected


def test_canvas_is_capped_at_model_area():
    w, h = core.solve_canvas(16, 9, 5.0)
    assert w * h <= core.MAX_PIXELS + 32 * 32 * 2
    assert w % 32 == 0 and h % 32 == 0


def test_aspect_from_size():
    assert core.aspect_from_size(1920, 1080) == (16, 9)
    assert core.aspect_from_size(1024, 1024) == (1, 1)
    assert core.aspect_from_size(1080, 1920) == (9, 16)


# ---------------------------------------------------------------- tokens / VRAM planning
def test_token_counts_example():
    t = core.count_tokens(1344, 768, 124)
    assert t.video == 37 * (24 * 42)
    assert t.audio == 2 * round(124 / 24 * 40)


def test_longer_or_bigger_costs_more():
    base = core.count_tokens(864, 480, 124).total
    assert core.count_tokens(864, 480, 192).total > base
    assert core.count_tokens(1344, 768, 124).total > base


def test_refs_add_tokens_and_max_costs_more_than_match():
    none = core.count_tokens(1344, 768, 124).total
    match = core.count_tokens(1344, 768, 124, ref_images=3, ref_image_size="match").total
    big = core.count_tokens(1344, 768, 124, ref_images=3, ref_image_size="max").total
    assert none < match < big


def test_plan_reduces_resolution_until_it_fits():
    budget = 5.5
    over = core.plan_generation((16, 9), 0.98, 8.0, budget, strategy="none")
    assert over.status == "over"
    fit = core.plan_generation((16, 9), 0.98, 8.0, budget, strategy="reduce_resolution")
    assert fit.estimate.total_gb <= budget
    assert fit.width * fit.height < 1344 * 768
    assert fit.frames == over.frames                       # duration untouched
    assert fit.adjustments


def test_plan_reduces_duration_keeps_grid():
    fit = core.plan_generation((16, 9), 0.98, 12.0, 5.5, strategy="reduce_duration")
    assert fit.frames % 17 == 5
    assert fit.frames >= core.SOFT_MIN_FRAMES
    assert (fit.width, fit.height) == (1344, 768)


def test_plan_status_ok_for_small_request():
    p = core.plan_generation((16, 9), 0.4, 5.0, 6.0)
    assert p.status == "ok"
    assert "Canvas" in core.format_plan(p)


def test_plan_warns_outside_trained_range():
    p = core.plan_generation((16, 9), 0.4, 4.0, 6.0, snap="down")
    assert any("below" in n for n in p.notes)


# ---------------------------------------------------------------- presets / LoRA picking
def test_presets_follow_official_templates():
    assert core.sampler_preset("fl2va", "turbo")["steps"] == 8
    assert core.sampler_preset("ref2va", "turbo")["steps"] == 4
    assert core.sampler_preset("fl2va", "balanced")["steps"] == 20
    assert core.sampler_preset("fl2va", "balanced")["lora"] is False
    assert core.sampler_preset("fl2va", "turbo", steps_override=6)["steps"] == 6
    assert core.sampler_preset("fl2va", "turbo")["sampler"] == "res_multistep"


LORAS = [
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    "some_style_lora.safetensors",
]


def test_lora_pick_never_mixes_tasks():
    assert core.pick_turbo_lora(LORAS, "fl2va", 8) == LORAS[0]
    assert core.pick_turbo_lora(LORAS, "ref2va", 4) == LORAS[1]
    assert core.pick_turbo_lora([LORAS[1]], "fl2va", 8) is None
    assert core.pick_turbo_lora([LORAS[0]], "ref2va", 4) is None
    assert core.pick_turbo_lora(["some_style_lora.safetensors"], "fl2va", 8) is None


# ---------------------------------------------------------------- hardware rules (RTX 3060 Ti = sm 8.6)
def test_quant_support_matches_comfyui_rules():
    assert core.quant_support((8, 6)) == {"int8": True, "fp8": False, "nvfp4": False}
    assert core.quant_support((8, 9), torch_version=(2, 5))["fp8"] is True
    assert core.quant_support((8, 9), windows=True, torch_version=(2, 3))["fp8"] is False
    assert core.quant_support((9, 0))["fp8"] is True
    assert core.quant_support((12, 0))["nvfp4"] is True


def test_classify_files():
    c = core.classify_h3_file("minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    assert (c["mode"], c["quant"], c["pruned"]) == ("fl2va", "int8", True)
    assert core.classify_h3_file("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")["quant"] == "nvfp4"
    assert core.classify_h3_file("minimax_h3_ref2va_pruned_fp8_scaled.safetensors")["mode"] == "ref2va"
    assert core.classify_h3_file("model-Q4_K_M.gguf")["quant"] == "gguf"


def _screenshot_rig():
    """The machine from the user's screenshot."""
    return core.SystemFacts(
        gpu=core.GpuCaps("NVIDIA GeForce RTX 3060 Ti", (8, 6), 8.0, 6.97),
        ram_total_gb=31.77, ram_free_gb=12.2, windows=True, torch_version=(2, 14),
        cache_none=True, reserve_vram_gb=1.0, fp8_text_enc_flag=True, dynamic_vram=True)


def test_advisor_report_for_screenshot_rig():
    facts = _screenshot_rig()
    dit = {"minimax_h3_fl2va_pruned_int8_convrot.safetensors": 20.97,
           "minimax_h3_fl2va_pruned_fp8_scaled.safetensors": 21.0,
           "minimax_h3_ref2va_pruned_int8_convrot.safetensors": 20.97}
    te = {"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 15.7}
    r = core.build_advisor_report(facts, dit, te, "fl2va")
    assert "RTX 3060 Ti" in r
    assert "no NVFP4" in r.replace("NVFP4 matmul no", "no NVFP4") or "NVFP4 matmul no" in r
    assert "--cache-none" in r
    assert "fp8_e4m3fn-text-enc" in r
    assert "stream from disk" in r                               # 21 GB DiT > ~15 GB resident capacity
    assert "Best installed fl2va diffusion file for this GPU: minimax_h3_fl2va_pruned_int8_convrot" in r


def test_vram_budget_for_screenshot_rig():
    facts = _screenshot_rig()
    assert core.vram_budget_gb(facts) == pytest.approx(5.97, abs=0.01)


# ---------------------------------------------------------------- graph signature / keys
def _graph(clip_name="te_a.safetensors"):
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
        "3": {"class_type": "H3Smart_CachedConditioning",
              "inputs": {"clip": ["1", 0], "vae": ["2", 0], "prompt": "a cat"}},
    }


def test_graph_signature_tracks_loader_files_not_node_ids():
    a = core.graph_signature(_graph(), ["1", 0])
    assert a == core.graph_signature(_graph(), ["1", 0])
    assert a != core.graph_signature(_graph("te_b.safetensors"), ["1", 0])
    renumbered = {"9": _graph()["1"], "3": _graph()["3"]}
    assert core.graph_signature(renumbered, ["9", 0]) == a


def test_graph_signature_follows_upstream_chain():
    g = {"1": {"class_type": "A", "inputs": {"x": 1}},
         "2": {"class_type": "B", "inputs": {"src": ["1", 0]}}}
    g2 = {"1": {"class_type": "A", "inputs": {"x": 2}},
          "2": {"class_type": "B", "inputs": {"src": ["1", 0]}}}
    assert core.graph_signature(g, ["2", 0]) != core.graph_signature(g2, ["2", 0])
    assert core.graph_signature(None, ["2", 0]) is None


def test_cache_key_changes_with_every_part():
    k = core.build_cache_key("fl2va", "0.43.0", ["a", "b"])
    assert k == core.build_cache_key("fl2va", "0.43.0", ["a", "b"])
    assert k != core.build_cache_key("fl2va", "0.45.0", ["a", "b"])
    assert k != core.build_cache_key("ref2va", "0.43.0", ["a", "b"])
    assert k != core.build_cache_key("fl2va", "0.43.0", ["a", "c"])


# ---------------------------------------------------------------- prompt builder
S = prompting.ShotSpec


def test_timestamps():
    assert prompting.format_timestamp(3.5) == "00:03.500"
    assert prompting.format_timestamp(65.25) == "01:05.250"


def test_camera_sentences_match_guide_examples():
    assert prompting.camera_sentence("push in", "small", "slow") == "The camera pushes in with small amplitude at slow speed."
    assert prompting.camera_sentence("pan right", "large", "fast") == "The camera pans right with large amplitude at fast speed."
    assert prompting.camera_sentence("static shot") == "The camera holds a static shot."
    assert prompting.camera_sentence("none") == ""


def test_i2va_prompt_structure():
    p, w = prompting.build_prompt("I2VA", [S("the woman from <Picture 1> lifts her gaze.", "truck right", "small", "slow")],
                                  "Live-action, cinematic", "Rain ticks on glass.", "Slow cello.", 124)
    lines = p.split("\n\n")
    assert lines[0] == "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    assert lines[1].startswith("integrated_multimodal_description: [Shot 1] Live-action, cinematic, the woman")
    assert "The camera trucks right with small amplitude at slow speed." in lines[1]
    assert lines[2] == "overall_soundscape: Rain ticks on glass."
    assert lines[3] == "non_diegetic_music: Slow cello."
    assert w == []


def test_t2va_has_no_instruction_and_multi_shot_cut_syntax():
    p, w = prompting.build_prompt("T2VA", [S("a baker opens shutters."), S("A close-up of steam.", cut_seconds=3.5)],
                                  "Live-action", "Shutters scrape.", "N/A", 124)
    assert p.startswith("integrated_multimodal_description: [Shot 1]")
    assert "[Shot 2] At 00:03.500, the camera cuts to a close-up of steam." in p
    assert "[Shot 1] Live-action, a baker" in p


def test_fl2va_alignment_line_uses_duration_and_last_shot():
    p, w = prompting.build_prompt("FL2VA", [S("she raises the umbrella.")], "Live-action", "Rain falls.", "N/A", 192)
    first = p.split("\n\n")[0]
    assert "Picture 2 (from Shot 1) aligns with the 8.00-second mark" in first
    assert first.startswith("How the reference pictures align with the target video \u2014 Picture 1 (from Shot 1)")


def test_l2va_alignment_line():
    p, _ = prompting.build_prompt("L2VA", [S("glass falls.")], "", "Crash.", "N/A", 144)
    assert p.startswith("How the reference pictures align with the target video \u2014 <Picture 1> (from [Shot 1]) aligns with the 6.00-second mark")


def test_cut_time_errors():
    with pytest.raises(prompting.PromptError):
        prompting.build_prompt("T2VA", [S("a"), S("b", cut_seconds=0.0)], "", "x.", "N/A", 124)
    with pytest.raises(prompting.PromptError):
        prompting.build_prompt("T2VA", [S("a"), S("b", cut_seconds=9.0)], "", "x.", "N/A", 124)   # clip is 5.17 s
    with pytest.raises(prompting.PromptError):
        prompting.build_prompt("T2VA", [S("a"), S("b", cut_seconds=3.0), S("c", cut_seconds=2.0)], "", "x.", "N/A", 192)
    with pytest.raises(prompting.PromptError):
        prompting.build_prompt("T2VA", [S("  ")], "", "x.", "N/A", 124)


def test_lint_catches_guide_violations():
    _, w = prompting.build_prompt("T2VA", [S("He says <d>hello there</d> loudly.")], "", "Wind. Rain. Birds. Cars. Dogs.", "A. B. C. D.", 124)
    joined = " ".join(w)
    assert "language tag" in joined
    assert "overall_soundscape has 5 sentences" in joined
    assert "non_diegetic_music has 4 sentences" in joined


def test_lint_voiceover_and_single_shot_fl2va():
    _, w = prompting.build_prompt(
        "FL2VA", [S("a"), S("b", cut_seconds=3.0)], "", "Wind.", "N/A", 124)
    assert any("single shot" in x for x in w)
    _, w2 = prompting.build_prompt(
        "T2VA", [S("The man (S1) says in an off-screen voiceover: <d>[English] hi</d>.")], "", "Wind.", "N/A", 124)
    assert any("lips remain" in x for x in w2)


def test_empty_soundscape_becomes_na_with_warning():
    p, w = prompting.build_prompt("T2VA", [S("a")], "", "", "", 124)
    assert "overall_soundscape: N/A" in p
    assert any("empty" in x for x in w)
