# H3 Smart

Smart nodes for running **MiniMax H3** in ComfyUI on a small card. Tuned for an
**RTX 3060 Ti (8 GB VRAM) + 32 GB RAM, Windows, ComfyUI 0.43**, launched with
`--cache-none --fp8_e4m3fn-text-enc --reserve-vram 1.0`, but nothing is hard-coded to that machine: every
number is read from your hardware at run time.

ComfyUI's offloading already does the heavy lifting. This pack does not replace it; it removes work the
graph repeats for no reason and tells you what your hardware can actually run.

## Why these nodes (what the 3060 Ti setup implies)

| Observation about the setup | Consequence | What H3 Smart does |
|---|---|---|
| `--cache-none` re-runs every node on every queue | The 32B Qwen3-VL text encoder (15.7 GB) is loaded and run again even when only the seed changed | **Cached Conditioning** stores the encoder output on disk; an unchanged prompt never loads the encoder |
| Compute capability 8.6 | No NVFP4 matmul (needs sm 10+) and no FP8 matmul (needs sm 8.9+); INT8 *is* supported | **System Advisor** says which quants run natively; the INT8 ConvRot DiT is the right file |
| 21 GB DiT vs ~6 GB VRAM budget + ~9 GB usable RAM | About a quarter of the weights stream from disk every step | **Sampling Setup** defaults to the Turbo LoRA (8 / 4 steps); the Advisor quantifies the streaming |
| 5 s at 1344x768 is ~38k tokens | Activations alone can exceed what is free | **Budget Planner** estimates peak VRAM from the real token count and can shrink the request to fit |
| Text encoder and DiT can't both be resident | Models linger between stages | **Memory Barrier** drops them at the two stage boundaries |

## Install

Put the folder in `ComfyUI/custom_nodes/` (or install from the Comfy Registry once published) and restart.
No extra Python packages. Requires ComfyUI >= 0.43.0 (the native MiniMax H3 nodes). Example workflows appear
under **Templates -> H3 Smart**.

Model files are the official ones from `Comfy-Org/MiniMax-H3` (same names as ComfyUI's templates). Put the Turbo
LoRAs in `models/loras`.

## Nodes (category `h3_smart`)

**H3 Cached Conditioning (T2V / I2V / FL2V)** and **(Reference to Video)**
Drop-in replacements for the native `MiniMax H3 Image to Video` / `Reference to Video`, with the same inputs
plus cache options. `clip` and `vae` are *lazy*: on a cache hit the executor never runs the loaders, so neither
the 15.7 GB encoder nor its VRAM/RAM traffic is touched. On a miss the **native node runs unchanged** and its
result is stored, so behaviour follows whatever ComfyUI version you have.
The cache key covers the prompt, size, frame count (on the 17k+5 grid), the *content* of every image / video /
audio input, the ComfyUI version, and a signature of the loader nodes feeding `clip`/`vae` (read from the prompt
graph, so choosing a different encoder file invalidates the entry automatically). `cache_tag` forces a fresh key.
Output `cache_status` tells you HIT or MISS.

**H3 Budget Planner**
Aspect (or `match_image`) + megapixels + seconds in; width / height / frame count out, using ComfyUI's own
Resolution Selector maths and the model's 17k+5 frame grid. It counts tokens (video, audio, text, references),
estimates peak VRAM, and with `fit_strategy` shrinks resolution or duration until it fits your budget.
Default budget mode is `stable` (total VRAM - reserve - desktop overhead) so results don't flicker between
runs and break cache keys.

**H3 Sampling Setup**
One node replacing LoRA loader + steps switches + `KSamplerSelect` + `BasicScheduler`. `task` x `speed` selects
steps and the Turbo LoRA (FL2VA 8 steps, REF2VA 4 steps, or 20/25 steps without a LoRA), finds the right file
in `models/loras`, and refuses to mix FL2VA and REF2VA adapters. Sampler/scheduler follow ComfyUI's official
template (`res_multistep`, `simple`, `BasicGuider`).

**H3 Memory Barrier**
Pass-through for any type. Unloads models and frees VRAM/RAM, then reports before/after. Place one between
conditioning and the sampler and one between the sampler and VAE decode. Optional (experimental) process
memory trimming.

**H3 Shot Prompt Builder**
Builds MiniMax's official base-mode prompt (T2VA / I2VA / FL2VA / L2VA): the alignment line, up to three timed
shots with `MM:SS.mmm` cuts, camera sentences, `overall_soundscape`, `non_diegetic_music`, then lints it
(dialogue tags, sentence counts, cut times). The Reference-to-Video prompt format is not covered.

**H3 System Advisor** / **H3 Conditioning Cache Manager**
The advisor reads GPU, RAM, launch flags and installed H3 files and writes a report. The manager shows or
clears the cache (`ComfyUI/user/h3_smart/cond_cache`, LRU-pruned to `max_cache_gb`, default 2 GB).

## Suggested settings for 8 GB VRAM

Planner estimates with the default heuristic and a 6.2 GB budget (16:9):

| Size | 5 s | 8 s | 10 s | 15 s |
|---|---|---|---|---|
| 608x352 (0.2 MP) | 2.3 GB ok | 2.7 ok | 3.1 ok | 3.8 ok |
| 864x480 (0.4 MP) | 3.0 ok | 3.9 ok | 4.5 ok | 6.0 tight |
| 1056x608 (0.6 MP) | 3.8 ok | 5.2 ok | 6.1 tight | 8.4 over |
| 1344x768 (0.98 MP) | 5.3 tight | 7.4 over | 8.9 over | 12.6 over |

Start at 0.4 MP, 5 s, Turbo, and let the planner shrink anything bigger. These figures are **estimates**
(see below). If a run OOMs below the estimate, raise `activation_kb_per_token`; if you have headroom, lower it.

If sampling is slow, the usual cause is disk streaming of the 21 GB DiT (the Advisor shows how much). Free RAM,
keep the Turbo LoRA on, or try a smaller community quant (GGUF Q4/Q3 via ComfyUI-GGUF, or an INT4 ConvRot
build) and read that file's model card; expect some quality loss.

## What is and isn't verified

Verified in a CPU sandbox against the **real** ComfyUI source (v0.43.0 and v0.45.0): all nodes' V3 schemas;
drop-in input parity with the native nodes; the cache hit/miss round trip using the real native node; lazy
evaluation through ComfyUI's real `PromptExecutor` in both `--cache-none` and classic mode (loaders really
are skipped); frame/latent/token geometry against what the native node builds; sampler sigmas; and the example
workflows pass ComfyUI's real `validate_prompt`. 94 tests.

**Not** verified (no GPU or model weights were available):
- Real generation speed or memory. The VRAM figures are a heuristic derived from the DiT's dimensions
  (hidden 5376, FFN 14336, 50 blocks), not measured on a 3060 Ti.
- The example workflows have not been opened in the actual frontend. They are generated from the live node
  definitions and structurally validated, but a widget in a native node whose definition changed between
  versions could load with its default value.
- The Memory Barrier's benefit with ComfyUI's dynamic VRAM; `trim_process_memory` is experimental.
- The effect of `--fp8_e4m3fn-text-enc` on a pre-quantised encoder (the Advisor says "unlikely to matter", unverified).

Cache safety: entries are written atomically, read with `torch.load(weights_only=True)` (a planted pickle is
rejected), and conditioning containing unusual objects (e.g. attached hooks) is simply not cached.
The cache keys *inputs*, not outputs: if a node upstream of the prompt is non-deterministic without any input
changing, use `cache_tag` or `refresh`.

## Development

```
COMFYUI_SRC=/path/to/ComfyUI pytest tests        # full suite against real ComfyUI (CPU)
pytest tests                                      # logic tests only, no ComfyUI needed
COMFYUI_SRC=/path/to/ComfyUI python tools/build_workflows.py   # regenerate example_workflows
```
Regenerate workflows against the ComfyUI version you target; a test fails if the shipped JSON is stale.
`core.py` documents which ComfyUI source files each formula mirrors, so it is easy to re-check after an update.

MIT licensed.
