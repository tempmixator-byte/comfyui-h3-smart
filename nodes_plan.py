"""Planning nodes: budget planner, one-node sampling setup (Turbo LoRA + sampler + sigmas), prompt builder."""
from __future__ import annotations

from comfy_api.latest import io, ui

from . import core, prompting, sysinfo

CATEGORY = "h3_smart"


# --------------------------------------------------------------------------- Budget planner
class H3Smart_BudgetPlanner(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3Smart_BudgetPlanner",
            display_name="H3 Budget Planner",
            category=f"{CATEGORY}/planning",
            description=("Turns 'aspect + megapixels + seconds' into width / height / frame count on H3's 17k+5 grid, "
                         "estimates peak VRAM from the real token count, and can shrink the request until it fits "
                         "your card. Outputs feed the conditioning node directly."),
            inputs=[
                io.Combo.Input("aspect_ratio", options=list(core.ASPECTS) + ["match_image"], default="16:9",
                               tooltip="match_image takes the ratio from the connected image, so first/last frames are not stretched."),
                io.Float.Input("megapixels", default=0.4, min=0.1, max=0.98, step=0.01,
                               tooltip="Target size (1 MP = 1024x1024 px, same as ComfyUI's Resolution Selector). 0.98 = official 768p (1344x768)."),
                io.Float.Input("duration_seconds", default=5.0, min=0.5, max=15.0, step=0.5),
                io.Combo.Input("fit_strategy", options=["reduce_resolution", "reduce_duration", "none"], default="reduce_resolution",
                               tooltip="What to shrink if the estimate exceeds the VRAM budget. 'none' only reports."),
                io.Combo.Input("task", options=list(core.MODES), default="fl2va",
                               tooltip="ref2va adds reference tokens (see the advanced reference inputs)."),
                io.Image.Input("image", optional=True, tooltip="Only used when aspect_ratio = match_image."),
                io.Combo.Input("length_snap", options=["up", "nearest", "down"], default="up", advanced=True,
                               tooltip="'up' matches the native nodes. 'down'/'nearest' avoid rounding long."),
                io.Combo.Input("budget_source", options=["stable", "live"], default="stable", advanced=True,
                               tooltip="stable: total VRAM - reserve - desktop overhead (same result every run, keeps cache keys steady). "
                                       "live: free VRAM right now - reserve."),
                io.Float.Input("vram_budget_override_gb", default=0.0, min=0.0, max=256.0, step=0.1, advanced=True,
                               tooltip="0 = auto. Otherwise the VRAM (GB) the sampler may use."),
                io.Float.Input("reserve_vram_gb", default=-1.0, min=-1.0, max=32.0, step=0.1, advanced=True,
                               tooltip="-1 = use ComfyUI's --reserve-vram (or an OS default)."),
                io.Float.Input("desktop_overhead_gb", default=0.8, min=0.0, max=8.0, step=0.1, advanced=True,
                               tooltip="VRAM used by the OS compositor/browser (stable mode only)."),
                io.Int.Input("ref_images", default=0, min=0, max=9, advanced=True, tooltip="REF2VA: number of reference images."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match", advanced=True,
                               tooltip="Must match the conditioning node. 'max' is estimated as a 2048 px short edge per image."),
                io.Float.Input("ref_video_seconds", default=0.0, min=0.0, max=45.0, step=0.5, advanced=True,
                               tooltip="REF2VA: total seconds of reference video."),
                io.Int.Input("text_tokens", default=300, min=0, max=8000, advanced=True,
                             tooltip="Rough prompt length in tokens (a 150-word prompt is ~250)."),
                io.Float.Input("activation_kb_per_token", default=core.DEFAULT_KB_PER_TOKEN, min=10.0, max=1000.0, step=1.0, advanced=True,
                               tooltip="Heuristic bytes of activations per token. Raise it if you hit OOM below the estimate, lower it if you have lots of headroom."),
            ],
            outputs=[io.Int.Output(display_name="width"), io.Int.Output(display_name="height"),
                     io.Int.Output(display_name="length"), io.String.Output(display_name="report"),
                     io.String.Output(display_name="status")],
        )

    @classmethod
    def execute(cls, aspect_ratio="16:9", megapixels=0.4, duration_seconds=5.0, fit_strategy="reduce_resolution",
                task="fl2va", image=None, length_snap="up", budget_source="stable", vram_budget_override_gb=0.0,
                reserve_vram_gb=-1.0, desktop_overhead_gb=0.8, ref_images=0, ref_image_size="match",
                ref_video_seconds=0.0, text_tokens=300,
                activation_kb_per_token=core.DEFAULT_KB_PER_TOKEN) -> io.NodeOutput:
        extra = []
        if aspect_ratio == "match_image":
            if image is not None:
                h_px, w_px = int(image.shape[1]), int(image.shape[2])
                aspect = core.aspect_from_size(w_px, h_px)
                extra.append(f"Aspect   : {aspect[0]}:{aspect[1]} taken from the {w_px}x{h_px} image")
            else:
                aspect = core.ASPECTS["16:9"]
                extra.append("Aspect   : match_image selected but no image connected; using 16:9")
        else:
            aspect = core.ASPECTS[aspect_ratio]

        facts = sysinfo.gather_facts()
        budget = sysinfo.planning_budget_gb(facts, budget_source, desktop_overhead_gb,
                                            vram_budget_override_gb, reserve_vram_gb)
        if budget <= 0:
            extra.append("Budget   : no CUDA GPU detected and no override given; the estimate cannot be judged")
        if task == "fl2va" and ref_images:
            extra.append("Note     : reference inputs are ignored for task=fl2va")
            ref_images, ref_video_seconds = 0, 0.0

        plan = core.plan_generation(aspect, megapixels, duration_seconds, budget, snap=length_snap,
                                    strategy=fit_strategy, text_tokens=text_tokens, ref_images=ref_images,
                                    ref_image_size=ref_image_size, ref_video_seconds=ref_video_seconds,
                                    kb_per_token=activation_kb_per_token)
        report = core.format_plan(plan, extra)
        return io.NodeOutput(plan.width, plan.height, plan.frames, report, plan.status, ui=ui.PreviewText(report))


# --------------------------------------------------------------------------- Sampling setup
class H3Smart_SamplingSetup(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        import comfy.samplers
        return io.Schema(
            node_id="H3Smart_SamplingSetup",
            display_name="H3 Sampling Setup (Turbo LoRA + sampler + sigmas)",
            category=f"{CATEGORY}/sampling",
            description=("Replaces LoRA loader + switches + KSamplerSelect + BasicScheduler with one node. "
                         "Picks the right Turbo LoRA for the task (FL2VA and REF2VA adapters are never mixed), "
                         "the matching step count, and ComfyUI's official H3 sampler/scheduler."),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("task", options=list(core.MODES), default="fl2va",
                               tooltip="Must match the diffusion model you loaded (fl2va or ref2va checkpoint)."),
                io.Combo.Input("speed", options=list(core.SPEEDS), default="turbo",
                               tooltip="turbo: Lightning LoRA (8 steps FL2VA / 4 steps REF2VA). balanced: 20 steps. quality: 25 steps."),
                io.Int.Input("steps_override", default=0, min=0, max=200, advanced=True, tooltip="0 = use the preset."),
                io.Combo.Input("lora_name", options=["auto"] + sysinfo.lora_names(), default="auto", advanced=True,
                               tooltip="auto picks the matching H3 Turbo LoRA from models/loras."),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05, advanced=True),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default=core.DEFAULT_SAMPLER, advanced=True),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES, default=core.DEFAULT_SCHEDULER, advanced=True),
            ],
            outputs=[io.Model.Output(), io.Sampler.Output(), io.Sigmas.Output(),
                     io.Int.Output(display_name="steps"), io.String.Output(display_name="info")],
        )

    @classmethod
    def execute(cls, model, task="fl2va", speed="turbo", steps_override=0, lora_name="auto", lora_strength=1.0,
                sampler_name=core.DEFAULT_SAMPLER, scheduler=core.DEFAULT_SCHEDULER) -> io.NodeOutput:
        import comfy.samplers
        import comfy.sd
        import comfy.utils
        import folder_paths

        preset = core.sampler_preset(task, speed, steps_override)
        steps = preset["steps"]
        lines = [f"task={task}  speed={speed}  steps={steps}  sampler={sampler_name}/{scheduler}"]

        if preset["lora"]:
            chosen = lora_name
            if chosen == "auto":
                chosen = core.pick_turbo_lora(folder_paths.get_filename_list("loras"), task, steps)
            if not chosen:
                raise RuntimeError(
                    f"No H3 Turbo LoRA for task '{task}' found in models/loras. Download the matching "
                    "'minimax_h3_..._turbo_...' file from the Comfy-Org/MiniMax-H3 repository, or set speed to balanced/quality.")
            path = folder_paths.get_full_path_or_raise("loras", chosen)
            lora, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
            model, _ = comfy.sd.load_lora_for_models(model, None, lora, lora_strength, 0, lora_metadata=meta)
            lines.append(f"LoRA: {chosen} @ {lora_strength:g}")
            if (task == "fl2va" and "ref2v" in chosen.lower()) or (task == "ref2va" and "fl2v" in chosen.lower()):
                lines.append("WARNING: this LoRA looks like it belongs to the other task; do not mix FL2VA and REF2VA adapters.")
        else:
            lines.append("LoRA: none")

        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()
        info = "\n".join(lines)
        return io.NodeOutput(model, sampler, sigmas, steps, info, ui=ui.PreviewText(info))


# --------------------------------------------------------------------------- Prompt builder
def _shot_inputs(n: int):
    items = [
        io.String.Input(f"shot{n}_text", multiline=True, default="",
                        tooltip=("Shot 1: subject, composition, action (style is added from 'style'). " if n == 1 else
                                 "Text after 'the camera cuts to ...'. Leave empty to skip this shot. ")
                        + "Dialogue format: (S1) says: <d>[English] words</d>"),
        io.Combo.Input(f"shot{n}_camera", options=list(prompting.CAMERA_MOVES), default="none"),
        io.Combo.Input(f"shot{n}_amplitude", options=list(prompting.AMPLITUDES), default="normal", advanced=True),
        io.Combo.Input(f"shot{n}_speed", options=list(prompting.SPEEDS), default="normal", advanced=True),
    ]
    if n > 1:
        items.insert(1, io.Float.Input(f"shot{n}_cut_seconds", default=3.0 * (n - 1), min=0.0, max=15.0, step=0.1,
                                       tooltip="When this shot begins (strictly increasing, inside the clip)."))
        items.append(io.Combo.Input(f"shot{n}_cut_phrase", options=list(prompting.CUT_PHRASES), default=prompting.CUT_PHRASES[0], advanced=True))
    return items


class H3Smart_ShotPromptBuilder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs = [
            io.Combo.Input("mode", options=list(prompting.MODES), default="I2VA",
                           tooltip="T2VA text only | I2VA first frame | FL2VA first+last frame | L2VA last frame."),
            io.Int.Input("length_frames", default=124, min=5, max=3600, step=1,
                         tooltip="Connect the Budget Planner's length. Used for the alignment line and cut-time checks."),
            io.String.Input("style", default="Live-action, cinematic",
                            tooltip="Opening style words for [Shot 1], e.g. 'Live-action, cinematic' / '2D-animated' / 'claymation'."),
        ]
        for n in (1, 2, 3):
            inputs += _shot_inputs(n)
        inputs += [
            io.String.Input("overall_soundscape", multiline=True, default="",
                            tooltip="1-4 sentences: ambience, physical sounds, non-verbal human sounds. No dialogue. 'N/A' only for true silence."),
            io.String.Input("non_diegetic_music", multiline=True, default="N/A",
                            tooltip="1-3 sentences: score only the audience hears (instruments, tempo, rhythm, dynamics). N/A if none."),
        ]
        return io.Schema(
            node_id="H3Smart_ShotPromptBuilder",
            display_name="H3 Shot Prompt Builder",
            category=f"{CATEGORY}/prompt",
            description=("Assembles MiniMax's official base-mode prompt (alignment line, integrated_multimodal_description "
                         "with timed shots and camera sentences, overall_soundscape, non_diegetic_music) and lints it."),
            inputs=inputs,
            outputs=[io.String.Output(display_name="prompt"), io.String.Output(display_name="warnings")],
        )

    @classmethod
    def execute(cls, mode="I2VA", length_frames=124, style="", overall_soundscape="", non_diegetic_music="N/A",
                **shots) -> io.NodeOutput:
        specs = []
        for n in (1, 2, 3):
            specs.append(prompting.ShotSpec(
                text=shots.get(f"shot{n}_text", ""),
                camera=shots.get(f"shot{n}_camera", "none"),
                amplitude=shots.get(f"shot{n}_amplitude", "normal"),
                speed=shots.get(f"shot{n}_speed", "normal"),
                cut_seconds=float(shots.get(f"shot{n}_cut_seconds", 0.0)),
                cut_phrase=shots.get(f"shot{n}_cut_phrase", prompting.CUT_PHRASES[0]),
            ))
        try:
            prompt, warns = prompting.build_prompt(mode, specs, style, overall_soundscape, non_diegetic_music, length_frames)
        except prompting.PromptError as e:
            raise ValueError(f"H3 Shot Prompt Builder: {e}") from None
        warn_text = "\n".join(f"- {w}" for w in warns) if warns else "No warnings."
        return io.NodeOutput(prompt, warn_text, ui=ui.PreviewText(prompt + "\n\n--- lint ---\n" + warn_text))
