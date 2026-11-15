"""Generate the example workflows in ../example_workflows from real ComfyUI node definitions.

Dev tool, not loaded by ComfyUI. It boots ComfyUI (CPU) from COMFYUI_SRC, reads each node's real input
definitions (so widget order can never drift from what the frontend expects), builds one internal graph and
writes it as:
  * UI-format JSON (what you drag onto the canvas)            -> example_workflows/*.json
  * (optionally) an API prompt, used by tests to run ComfyUI's real validate_prompt on the same graph

    COMFYUI_SRC=/path/to/ComfyUI python tools/build_workflows.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "example_workflows"

UNET_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
UNET_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENC = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"}


def boot():
    """Boot ComfyUI on CPU and register the pack so definitions can be read."""
    src = os.environ.get("COMFYUI_SRC")
    if not src:
        sys.exit("set COMFYUI_SRC to a ComfyUI checkout (the version you target, e.g. v0.43.0)")
    sys.path.insert(0, src)
    sys.argv = ["build", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False))
    spec = importlib.util.spec_from_file_location("h3_smart_build", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3_smart_build"] = mod
    spec.loader.exec_module(mod)
    ext = mod.comfy_entrypoint()
    ext = asyncio.run(ext) if asyncio.iscoroutine(ext) else ext
    for cls in asyncio.run(ext.get_node_list()):
        nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
    return nodes


class Defs:
    """Ordered input / output definitions of a node type, from the live registry."""

    def __init__(self, nodes_mod):
        self.nodes = nodes_mod
        self._cache = {}

    def get(self, ntype):
        if ntype in self._cache:
            return self._cache[ntype]
        cls = self.nodes.NODE_CLASS_MAPPINGS[ntype]
        inputs, outputs = [], []
        if hasattr(cls, "GET_NODE_INFO_V1"):
            info = cls.GET_NODE_INFO_V1()
            order = info.get("input_order") or {}
            for section in ("required", "optional"):
                for name in order.get(section, list((info["input"].get(section) or {}).keys())):
                    spec = info["input"][section][name]
                    inputs.append(self._entry(name, spec[0], spec[1] if len(spec) > 1 else {}, section == "optional"))
            outputs = list(zip(info["output_name"], info["output"]))
        else:
            it = cls.INPUT_TYPES()
            for section in ("required", "optional"):
                for name, spec in (it.get(section) or {}).items():
                    inputs.append(self._entry(name, spec[0], spec[1] if len(spec) > 1 else {}, section == "optional"))
            names = getattr(cls, "RETURN_NAMES", None) or cls.RETURN_TYPES
            outputs = list(zip(names, cls.RETURN_TYPES))
        d = {"inputs": inputs, "outputs": outputs}
        self._cache[ntype] = d
        return d

    @staticmethod
    def _entry(name, typ, opts, optional):
        options = None
        if isinstance(typ, list):
            options, typ = typ, "COMBO"
        elif typ == "COMBO":
            options = opts.get("options", [])
        widget = (typ in WIDGET_TYPES) and not opts.get("forceInput")
        default = opts.get("default")
        if default is None and options:
            default = options[0]
        return {"name": name, "type": typ, "widget": widget, "optional": optional, "options": options,
                "default": default, "control": bool(opts.get("control_after_generate"))}


class Graph:
    def __init__(self, defs: Defs, title: str):
        self.defs, self.title = defs, title
        self.nodes, self.links, self.groups = {}, [], []
        self._nid, self._lid = 0, 0

    def add(self, ntype, title=None, pos=(0, 0), size=(320, 120), values=None, widgets=None, color=None,
            extra_slots=None, properties=None):
        self._nid += 1
        self.nodes[self._nid] = dict(id=self._nid, type=ntype, title=title, pos=list(pos), size=list(size),
                                     values=values or {}, widgets=widgets, color=color,
                                     extra_slots=extra_slots or [], properties=properties or {})
        return self._nid

    def note(self, text, pos, size=(420, 260), title="Note"):
        self._nid += 1
        self.nodes[self._nid] = dict(id=self._nid, type="MarkdownNote", title=title, pos=list(pos), size=list(size),
                                     values={}, widgets=[text], color="#432", extra_slots=[], properties={}, frontend_only=True)
        return self._nid

    def link(self, src, out, dst, inp):
        """out: output name or index; inp: input name (a widget input becomes a socket)."""
        d_src = self.defs.get(self.nodes[src]["type"])
        names = [o[0] for o in d_src["outputs"]]
        out_idx = out if isinstance(out, int) else names.index(out)
        typ = d_src["outputs"][out_idx][1]
        if typ.startswith("COMFY_MATCHTYPE"):
            typ = self._resolved(src) or typ
        self._lid += 1
        self.links.append(dict(id=self._lid, src=src, out=out_idx, dst=dst, inp=inp, type=typ))
        return self._lid

    def group(self, title, node_ids, pad=40, color="#3f789e"):
        xs = [self.nodes[n]["pos"][0] for n in node_ids]
        ys = [self.nodes[n]["pos"][1] for n in node_ids]
        xe = [self.nodes[n]["pos"][0] + self.nodes[n]["size"][0] for n in node_ids]
        ye = [self.nodes[n]["pos"][1] + self.nodes[n]["size"][1] for n in node_ids]
        self.groups.append({"id": len(self.groups) + 1, "title": title, "color": color, "flags": {},
                            "bounding": [min(xs) - pad, min(ys) - pad - 40, max(xe) - min(xs) + 2 * pad,
                                         max(ye) - min(ys) + 2 * pad + 40]})

    def _resolved(self, nid):
        """Concrete type flowing through a MatchType (pass-through) node: the type of what feeds its 'value'."""
        feed = [l for l in self.links if l["dst"] == nid and l["inp"] == "value"]
        return feed[0]["type"] if feed else None

    # ------------------------------------------------------------------ emit
    def _linked_inputs(self, nid):
        return {l["inp"]: l for l in self.links if l["dst"] == nid}

    def to_ui(self):
        ui_nodes = []
        for nid, n in self.nodes.items():
            if n.get("frontend_only"):
                ui_nodes.append({"id": nid, "type": n["type"], "pos": n["pos"], "size": n["size"], "flags": {},
                                 "order": 0, "mode": 0, "inputs": [], "outputs": [], "title": n["title"],
                                 "properties": {}, "widgets_values": n["widgets"], "color": n["color"], "bgcolor": "#653"})
                continue
            d = self.defs.get(n["type"])
            linked = self._linked_inputs(nid)
            sockets, widget_inputs, wvals = [], [], []
            for e in d["inputs"]:
                if e["type"] in ("COMFY_AUTOGROW_V3",):
                    continue
                if e["widget"]:
                    if e["name"] in linked:
                        widget_inputs.append({"name": e["name"], "type": e["type"] if e["type"] != "COMBO" else "COMBO",
                                              "widget": {"name": e["name"]}, "link": linked[e["name"]]["id"]})
                    v = n["values"].get(e["name"], e["default"])
                    wvals.append(v)
                    if e["control"]:
                        wvals.append("randomize")
                else:
                    is_match = e["type"].startswith("COMFY_MATCHTYPE")
                    s = {"name": e["name"], "type": (self._resolved(nid) or "*") if is_match else e["type"],
                         "link": linked[e["name"]]["id"] if e["name"] in linked else None}
                    if e["optional"]:
                        s["shape"] = 7
                    sockets.append(s)
            for slot in n["extra_slots"]:                       # Autogrow slots, e.g. ref_images.ref_image_0
                s = {"name": slot["name"], "label": slot["label"], "shape": 7, "type": slot["type"],
                     "link": linked[slot["name"]]["id"] if slot["name"] in linked else None}
                sockets.append(s)
            if n["widgets"] is not None:
                wvals = n["widgets"]
            outs = []
            for i, (oname, otype) in enumerate(d["outputs"]):
                ids = [l["id"] for l in self.links if l["src"] == nid and l["out"] == i]
                if otype.startswith("COMFY_MATCHTYPE"):
                    otype = self._resolved(nid) or "*"
                outs.append({"name": oname, "type": otype, "links": ids or None})
            props = {"Node name for S&R": n["type"]}
            if n["type"].startswith("H3Smart_"):
                props["cnr_id"] = "h3-smart"
            else:
                props["cnr_id"] = "comfy-core"
            props.update(n["properties"])
            node = {"id": nid, "type": n["type"], "pos": n["pos"], "size": n["size"], "flags": {}, "order": 0,
                    "mode": 0, "inputs": sockets + widget_inputs, "outputs": outs, "properties": props,
                    "widgets_values": wvals}
            if n["title"]:
                node["title"] = n["title"]
            if n["color"]:
                node["color"] = n["color"]
                node["bgcolor"] = "#353"
            ui_nodes.append(node)
        # slot numbers for links: index within the node's serialized inputs array
        slot_of = {}
        for node in ui_nodes:
            for idx, inp in enumerate(node["inputs"]):
                slot_of[(node["id"], inp["name"])] = idx
        links = [[l["id"], l["src"], l["out"], l["dst"], slot_of[(l["dst"], l["inp"])], l["type"]] for l in self.links]
        # a stable execution order is not required by the frontend; give a topological hint
        order = self._topo()
        for node in ui_nodes:
            node["order"] = order.get(node["id"], 0)
        return {"id": f"h3-smart-{self.title}", "revision": 0, "last_node_id": self._nid, "last_link_id": self._lid,
                "nodes": ui_nodes, "links": links, "groups": self.groups, "config": {},
                "extra": {"ds": {"scale": 0.6, "offset": [200, 100]}, "frontendVersion": "1.57.6"}, "version": 0.4}

    def _topo(self):
        deps = {nid: set() for nid in self.nodes}
        for l in self.links:
            deps[l["dst"]].add(l["src"])
        order, seen = {}, set()

        def visit(n):
            if n in seen:
                return
            seen.add(n)
            for d in sorted(deps[n]):
                visit(d)
            order[n] = len(order)
        for n in sorted(self.nodes):
            visit(n)
        return order

    def to_api(self):
        """API-format prompt of the same graph (frontend-only nodes dropped). Used for validate_prompt in tests."""
        prompt = {}
        for nid, n in self.nodes.items():
            if n.get("frontend_only"):
                continue
            d = self.defs.get(n["type"])
            inputs = {}
            for e in d["inputs"]:
                if e["type"] == "COMFY_AUTOGROW_V3":
                    continue
                if e["widget"]:
                    inputs[e["name"]] = n["values"].get(e["name"], e["default"])
                    if e["control"]:
                        pass
            for l in self._linked_inputs(nid).values():
                inputs[l["inp"]] = [str(l["src"]), l["out"]]
            prompt[str(nid)] = {"class_type": n["type"], "inputs": inputs}
        return prompt


# ============================================================================ workflows
NOTE_COMMON = """### H3 Smart: what the pieces do
**System Advisor** reads your GPU/RAM/flags and installed H3 files and prints advice (queue it once).
**Budget Planner** turns aspect + megapixels + seconds into width/height/frames and shrinks the request if the
estimated VRAM exceeds your budget.
**Cached Conditioning** is the native H3 conditioning node with an on-disk cache: an unchanged prompt/frames/size
skips the 15 GB text encoder entirely, even with `--cache-none`.
**Sampling Setup** picks the matching Turbo LoRA + steps + official sampler/scheduler in one node.
**Memory Barrier** (x2) drops the text encoder before the DiT loads, and the DiT before VAE decode.
"""


def _models_and_decode(g: Graph, unet_name, task, base_x=0):
    """Shared plumbing: loaders, sampler stage, decode and save. Returns ids needed for wiring."""
    ids = {}
    ids["unet"] = g.add("UNETLoader", pos=(base_x, 120), size=(420, 90), values={"unet_name": unet_name, "weight_dtype": "default"})
    ids["clip"] = g.add("CLIPLoader", pos=(base_x, 260), size=(420, 130),
                        values={"clip_name": TEXT_ENC, "type": "minimax", "device": "default"})
    ids["vae"] = g.add("VAELoader", title="Video VAE", pos=(base_x, 440), size=(420, 70), values={"vae_name": VIDEO_VAE})
    ids["avae"] = g.add("VAELoader", title="Audio VAE", pos=(base_x, 560), size=(420, 70), values={"vae_name": AUDIO_VAE})
    g.group("Models", [ids["unet"], ids["clip"], ids["vae"], ids["avae"]], color="#3f789e")
    return ids


def build_fl2va(defs: Defs, with_image: bool) -> Graph:
    g = Graph(defs, "i2v" if with_image else "t2v")
    ids = _models_and_decode(g, UNET_FL2VA, "fl2va")
    adv = g.add("H3Smart_SystemAdvisor", pos=(0, -260), size=(520, 330), values={"task": "fl2va"}, color="#432")

    img = None
    if with_image:
        img = g.add("LoadImage", title="First frame", pos=(0, 760), size=(320, 360), values={"image": "example.png"})

    planner = g.add("H3Smart_BudgetPlanner", pos=(560, 120), size=(430, 380),
                    values={"aspect_ratio": "match_image" if with_image else "16:9", "megapixels": 0.4,
                            "duration_seconds": 5.0, "fit_strategy": "reduce_resolution", "task": "fl2va"})
    mode = "I2VA" if with_image else "T2VA"
    builder_values = {
        "mode": mode, "style": "Live-action, cinematic",
        "shot1_text": ("the woman from <Picture 1> turns toward the window as rain streaks the glass, her breath fogging the pane."
                       if with_image else
                       "a baker in a flour-dusted apron lifts a tray of golden loaves from a stone oven in a small village bakery at dawn."),
        "shot1_camera": "push in", "shot1_amplitude": "small", "shot1_speed": "slow",
        "overall_soundscape": "Soft rain taps the window. A kettle murmurs in the next room. Her slow breathing is faintly audible.",
        "non_diegetic_music": "A sparse solo piano plays unhurried minor chords.",
    }
    builder = g.add("H3Smart_ShotPromptBuilder", pos=(1030, 120), size=(470, 640), values=builder_values)
    cond = g.add("H3Smart_CachedConditioning", pos=(1560, 120), size=(420, 330), values={"cache_mode": "read_write"}, color="#323")
    bar1 = g.add("H3Smart_MemoryBarrier", title="Memory Barrier (drop text encoder)", pos=(2040, 120), size=(360, 130),
                 values={"label": "after text encoder"})
    setup = g.add("H3Smart_SamplingSetup", pos=(2040, 320), size=(420, 280), values={"task": "fl2va", "speed": "turbo"})
    guider = g.add("BasicGuider", pos=(2520, 120), size=(240, 60))
    noise = g.add("RandomNoise", pos=(2520, 240), size=(260, 100), values={"noise_seed": 1234})
    sampler = g.add("SamplerCustomAdvanced", pos=(2820, 120), size=(260, 140))
    bar2 = g.add("H3Smart_MemoryBarrier", title="Memory Barrier (drop DiT)", pos=(2820, 320), size=(360, 130),
                 values={"label": "before VAE decode"})
    dec = g.add("VAEDecode", pos=(3240, 120), size=(230, 60))
    adec = g.add("VAEDecodeAudio", pos=(3240, 240), size=(230, 60))
    cv = g.add("CreateVideo", pos=(3520, 120), size=(270, 110), widgets=[24], values={"fps": 24.0})
    sv = g.add("SaveVideo", pos=(3520, 290), size=(420, 360), widgets=["video/H3Smart", "auto", "auto"],
               values={"filename_prefix": "video/H3Smart", "format": "auto", "codec": "auto"})

    g.link(ids["unet"], 0, setup, "model")
    g.link(ids["clip"], 0, cond, "clip")
    g.link(ids["vae"], 0, cond, "vae")
    if with_image:
        g.link(img, 0, cond, "first_frame")
        g.link(img, 0, planner, "image")
    g.link(planner, "width", cond, "width")
    g.link(planner, "height", cond, "height")
    g.link(planner, "length", cond, "length")
    g.link(planner, "length", builder, "length_frames")
    g.link(builder, "prompt", cond, "prompt")
    g.link(cond, "positive", bar1, "value")
    g.link(setup, "MODEL", guider, "model")
    g.link(bar1, 0, guider, "conditioning")
    g.link(noise, 0, sampler, "noise")
    g.link(guider, 0, sampler, "guider")
    g.link(setup, "SAMPLER", sampler, "sampler")
    g.link(setup, "SIGMAS", sampler, "sigmas")
    g.link(cond, "LATENT", sampler, "latent_image")
    g.link(sampler, "output", bar2, "value")
    g.link(bar2, 0, dec, "samples")
    g.link(ids["vae"], 0, dec, "vae")
    g.link(bar2, 0, adec, "samples")
    g.link(ids["avae"], 0, adec, "vae")
    g.link(dec, 0, cv, "images")
    g.link(adec, 0, cv, "audio")
    g.link(cv, 0, sv, "video")

    g.group("Plan + prompt", [planner, builder] + ([img] if img else []), color="#a1309b")
    g.group("Conditioning (cached on disk)", [cond, bar1], color="#8A8")
    g.group("Sampling", [setup, guider, noise, sampler, bar2], color="#b58b2a")
    g.group("Decode + save", [dec, adec, cv, sv], color="#3f789e")

    how = ("I2V: connect a first frame; the planner takes its aspect ratio so it is not stretched. For first+last frame, add a second LoadImage to `last_frame` and set the builder mode to FL2VA."
           if with_image else
           "T2V: no image needed. Describe the scene in the Shot Prompt Builder (up to 3 timed shots).")
    g.note(NOTE_COMMON + "\n" + how + "\n\n**Model files** are the official Comfy-Org/MiniMax-H3 names. Put the Turbo LoRA "
           "(`minimax_h3_fl2v_turbo_8step_...`) in `models/loras`; Sampling Setup finds it automatically. "
           "Set speed to `balanced` for 20 steps without the LoRA.", pos=(560, -300), size=(880, 330), title="Read me")
    return g


def build_ref2va(defs: Defs) -> Graph:
    g = Graph(defs, "r2v")
    ids = _models_and_decode(g, UNET_REF2VA, "ref2va")
    adv = g.add("H3Smart_SystemAdvisor", pos=(0, -260), size=(520, 330), values={"task": "ref2va"}, color="#432")
    img1 = g.add("LoadImage", title="Reference 1  (<Picture 1>)", pos=(0, 760), size=(320, 360), values={"image": "example.png"})
    img2 = g.add("LoadImage", title="Reference 2  (<Picture 2>)", pos=(360, 760), size=(320, 360), values={"image": "example.png"})
    planner = g.add("H3Smart_BudgetPlanner", pos=(560, 120), size=(430, 380),
                    values={"aspect_ratio": "16:9", "megapixels": 0.4, "duration_seconds": 5.0,
                            "fit_strategy": "reduce_resolution", "task": "ref2va", "ref_images": 2,
                            "ref_image_size": "match"})
    prompt = g.add("PrimitiveStringMultiline", title="Prompt (use <Picture 1>, <Picture 2>)", pos=(1030, 120), size=(470, 300),
                   values={"value": "<Picture 1> and <Picture 2> sit across from each other at a rain-streaked diner window at night, "
                                    "talking quietly while neon light flickers across their faces."})
    slots = [
        {"name": "ref_images.ref_image_0", "label": "ref_image_0", "type": "IMAGE"},
        {"name": "ref_images.ref_image_1", "label": "ref_image_1", "type": "IMAGE"},
        {"name": "ref_images.ref_image_2", "label": "ref_image_2", "type": "IMAGE"},
        {"name": "ref_videos.ref_video_0", "label": "ref_video_0", "type": "IMAGE"},
        {"name": "ref_video_audios.ref_video_audio_0", "label": "ref_video_audio_0", "type": "AUDIO"},
        {"name": "ref_audios.ref_audio_0", "label": "ref_audio_0", "type": "AUDIO"},
    ]
    cond = g.add("H3Smart_CachedReferenceConditioning", pos=(1560, 120), size=(440, 420),
                 values={"ref_image_size": "match", "cache_mode": "read_write"}, color="#323", extra_slots=slots)
    bar1 = g.add("H3Smart_MemoryBarrier", title="Memory Barrier (drop text encoder)", pos=(2060, 120), size=(360, 130),
                 values={"label": "after text encoder"})
    setup = g.add("H3Smart_SamplingSetup", pos=(2060, 320), size=(420, 280), values={"task": "ref2va", "speed": "turbo"})
    guider = g.add("BasicGuider", pos=(2540, 120), size=(240, 60))
    noise = g.add("RandomNoise", pos=(2540, 240), size=(260, 100), values={"noise_seed": 1234})
    sampler = g.add("SamplerCustomAdvanced", pos=(2840, 120), size=(260, 140))
    bar2 = g.add("H3Smart_MemoryBarrier", title="Memory Barrier (drop DiT)", pos=(2840, 320), size=(360, 130),
                 values={"label": "before VAE decode"})
    dec = g.add("VAEDecode", pos=(3260, 120), size=(230, 60))
    adec = g.add("VAEDecodeAudio", pos=(3260, 240), size=(230, 60))
    cv = g.add("CreateVideo", pos=(3540, 120), size=(270, 110), widgets=[24], values={"fps": 24.0})
    sv = g.add("SaveVideo", pos=(3540, 290), size=(420, 360), widgets=["video/H3Smart_ref", "auto", "auto"],
               values={"filename_prefix": "video/H3Smart_ref", "format": "auto", "codec": "auto"})

    g.link(ids["unet"], 0, setup, "model")
    g.link(ids["clip"], 0, cond, "clip")
    g.link(ids["vae"], 0, cond, "vae")
    g.link(ids["avae"], 0, cond, "audio_vae")
    g.link(img1, 0, cond, "ref_images.ref_image_0")
    g.link(img2, 0, cond, "ref_images.ref_image_1")
    g.link(planner, "width", cond, "width")
    g.link(planner, "height", cond, "height")
    g.link(planner, "length", cond, "length")
    g.link(prompt, 0, cond, "prompt")
    g.link(cond, "positive", bar1, "value")
    g.link(setup, "MODEL", guider, "model")
    g.link(bar1, 0, guider, "conditioning")
    g.link(noise, 0, sampler, "noise")
    g.link(guider, 0, sampler, "guider")
    g.link(setup, "SAMPLER", sampler, "sampler")
    g.link(setup, "SIGMAS", sampler, "sigmas")
    g.link(cond, "LATENT", sampler, "latent_image")
    g.link(sampler, "output", bar2, "value")
    g.link(bar2, 0, dec, "samples")
    g.link(ids["vae"], 0, dec, "vae")
    g.link(bar2, 0, adec, "samples")
    g.link(ids["avae"], 0, adec, "vae")
    g.link(dec, 0, cv, "images")
    g.link(adec, 0, cv, "audio")
    g.link(cv, 0, sv, "video")

    g.group("Plan + prompt", [planner, prompt, img1, img2], color="#a1309b")
    g.group("Conditioning (cached on disk)", [cond, bar1], color="#8A8")
    g.group("Sampling", [setup, guider, noise, sampler, bar2], color="#b58b2a")
    g.group("Decode + save", [dec, adec, cv, sv], color="#3f789e")
    g.note(NOTE_COMMON + "\n**REF2VA** needs the `ref2va` diffusion checkpoint and the 4-step REF2VA Turbo LoRA "
           "(`minimax_h3_ref2v_turbo_4step_...`). Never mix FL2VA and REF2VA files. Reference tokens ride through "
           "every step, so keep `ref_image_size` on `match` on 8 GB cards. Tell the planner how many references you use.",
           pos=(560, -300), size=(880, 330), title="Read me")
    return g


def main():
    nodes_mod = boot()
    defs = Defs(nodes_mod)
    OUT.mkdir(exist_ok=True)
    builds = {
        "h3_smart_t2v.json": build_fl2va(defs, with_image=False),
        "h3_smart_i2v.json": build_fl2va(defs, with_image=True),
        "h3_smart_r2v.json": build_ref2va(defs),
    }
    for name, g in builds.items():
        (OUT / name).write_text(json.dumps(g.to_ui(), indent=2), encoding="utf-8")
        print("wrote", OUT / name, f"({len(g.nodes)} nodes, {len(g.links)} links)")
    return builds


if __name__ == "__main__":
    main()
