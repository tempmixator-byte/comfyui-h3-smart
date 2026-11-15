"""The shipped example workflows, checked against the real ComfyUI.

1. The UI JSON is internally consistent (links <-> slots, types, widget counts).
2. The same graph, converted to an API prompt, passes ComfyUI's own validate_prompt (required inputs, link
   types, value ranges, combo values), using empty placeholder model files.
"""
import asyncio
import importlib.util
import json
import pathlib

import pytest

pytestmark = pytest.mark.usefixtures("require_real")
ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = ["h3_smart_t2v.json", "h3_smart_i2v.json", "h3_smart_r2v.json"]


@pytest.fixture(scope="module")
def require_real(real):
    if not real:
        pytest.skip("needs ComfyUI source (COMFYUI_SRC)")


@pytest.fixture(scope="module")
def builder(pack):
    import nodes as comfy_nodes
    ext = pack.comfy_entrypoint()
    ext = asyncio.run(ext) if asyncio.iscoroutine(ext) else ext
    for cls in asyncio.run(ext.get_node_list()):
        comfy_nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
    spec = importlib.util.spec_from_file_location("build_workflows", ROOT / "tools" / "build_workflows.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    defs = mod.Defs(comfy_nodes)
    return mod, defs


@pytest.fixture(scope="module")
def graphs(builder):
    mod, defs = builder
    return {"h3_smart_t2v.json": mod.build_fl2va(defs, False), "h3_smart_i2v.json": mod.build_fl2va(defs, True),
            "h3_smart_r2v.json": mod.build_ref2va(defs)}


@pytest.fixture(scope="module")
def placeholder_models(tmp_path_factory):
    import folder_paths
    d = tmp_path_factory.mktemp("models")
    for folder, names in {
        "diffusion_models": ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
        "text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "vae": ["minimax_h3_video_vae_int8_convrot.safetensors", "minimax_h3_audio_vae_fp32.safetensors"],
        "loras": ["minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"],
    }.items():
        sub = d / folder
        sub.mkdir()
        for n in names:
            (sub / n).write_bytes(b"")
        folder_paths.add_model_folder_path(folder, str(sub))
    inp = pathlib.Path(folder_paths.get_input_directory())
    inp.mkdir(parents=True, exist_ok=True)
    (inp / "example.png").write_bytes(b"\x89PNG")
    folder_paths.filename_list_cache.clear()
    return d


@pytest.mark.parametrize("name", FILES)
def test_shipped_files_match_generator(name, graphs):
    on_disk = json.loads((ROOT / "example_workflows" / name).read_text())
    assert on_disk == graphs[name].to_ui(), "example_workflows is stale: run tools/build_workflows.py"


@pytest.mark.parametrize("name", FILES)
def test_ui_json_is_consistent(name):
    wf = json.loads((ROOT / "example_workflows" / name).read_text())
    nodes = {n["id"]: n for n in wf["nodes"]}
    assert wf["last_node_id"] == max(nodes) and wf["last_link_id"] == max(l[0] for l in wf["links"])
    for lid, src, so, dst, ds, typ in wf["links"]:
        s, d = nodes[src], nodes[dst]
        assert s["outputs"][so]["type"] == typ
        assert lid in s["outputs"][so]["links"]
        assert d["inputs"][ds]["link"] == lid
        in_t = d["inputs"][ds]["type"]
        assert in_t in (typ, "*"), f"link {lid}: {typ} -> {in_t} on {d['type']}.{d['inputs'][ds]['name']}"
    # every non-null link id referenced by a node exists exactly once
    ids = {l[0] for l in wf["links"]}
    for n in nodes.values():
        for i in n["inputs"]:
            if i["link"] is not None:
                assert i["link"] in ids
    assert len({n["id"] for n in wf["nodes"]}) == len(wf["nodes"])


@pytest.mark.parametrize("name", FILES)
def test_widget_values_count_matches_definitions(name, builder):
    _, defs = builder
    wf = json.loads((ROOT / "example_workflows" / name).read_text())
    for n in wf["nodes"]:
        if n["type"] == "MarkdownNote" or n["type"] in ("SaveVideo", "CreateVideo"):
            continue                                       # dynamic / version-dependent widget sets, see generator
        d = defs.get(n["type"])
        expected = sum(1 + (1 if e["control"] else 0) for e in d["inputs"] if e["widget"])
        assert len(n["widgets_values"]) == expected, n["type"]


@pytest.mark.parametrize("name", FILES)
def test_prompt_passes_real_validate_prompt(name, graphs, placeholder_models):
    import execution
    prompt = graphs[name].to_api()
    valid, err, good, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (err, node_errors)
    assert not node_errors
    assert any(prompt[o]["class_type"] == "SaveVideo" for o in good)


def test_cached_conditioning_is_wired_lazily_to_the_loaders(graphs):
    """The text-encoder loader must feed only the cache node, otherwise it would load on every run."""
    for name in ("h3_smart_t2v.json", "h3_smart_i2v.json", "h3_smart_r2v.json"):
        g = graphs[name]
        clip_loader = next(i for i, n in g.nodes.items() if n["type"] == "CLIPLoader")
        consumers = [g.nodes[l["dst"]]["type"] for l in g.links if l["src"] == clip_loader]
        assert len(consumers) == 1 and consumers[0].startswith("H3Smart_Cached")
