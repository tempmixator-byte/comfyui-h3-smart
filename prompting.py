"""Prompt assembly for MiniMax H3 base modes (T2VA / I2VA / FL2VA / L2VA).

Follows MiniMax's official "Video Prompt Writing Guide" (docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md in
MiniMaxAI/MiniMax-H3):

  * Part one: an alignment instruction for keyframe modes, as the first line, then one blank line.
  * Part two: exactly three fields, in order, separated by blank lines:
        integrated_multimodal_description / overall_soundscape / non_diegetic_music
  * [Shot 1] has no timestamp; later shots start "[Shot N] At MM:SS.mmm, the camera cuts to ...",
    with strictly increasing cut times inside the clip duration.
  * Camera motion is a natural sentence: motion type + amplitude + speed.

The reference-to-video (Ref2VA) guide is a different, longer format and is not covered here.
Pure Python, no ComfyUI imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FPS = 24
MODES = ("T2VA", "I2VA", "FL2VA", "L2VA")
AMPLITUDES = ("normal", "small", "large")
SPEEDS = ("normal", "slow", "fast")
CUT_PHRASES = (
    "the camera cuts to",
    "the shot cuts to",
    "the shot transitions to",
    "the shot changes to",
    "the shot switches to",
)

# guide table -> natural verb phrase; (verb, takes_amplitude_and_speed)
CAMERA_MOVES = {
    "none": None,
    "push in": ("pushes in", True),
    "pull out": ("pulls out", True),
    "zoom in": ("zooms in", True),
    "zoom out": ("zooms out", True),
    "pan left": ("pans left", True),
    "pan right": ("pans right", True),
    "truck left": ("trucks left", True),
    "truck right": ("trucks right", True),
    "tilt up": ("tilts up", True),
    "tilt down": ("tilts down", True),
    "pedestal up": ("pedestals up", True),
    "pedestal down": ("pedestals down", True),
    "arc shot": ("arcs around the subject", True),
    "tracking shot": ("tracks the moving subject", True),
    "static shot": ("holds a static shot", False),
    "shake slightly": ("shakes slightly", False),
    "shake strongly": ("shakes strongly", False),
    "pov": ("shifts to the subject's point of view", False),
    "roll clockwise": ("rolls clockwise", True),
    "roll counterclockwise": ("rolls counterclockwise", True),
}

MAX_PROMPT_CHARS = 7000   # reported limit in community write-ups of MiniMax's validation rules; treated as a soft warning


class PromptError(ValueError):
    """A problem that makes the prompt unusable (as opposed to a style warning)."""


@dataclass
class ShotSpec:
    text: str = ""
    camera: str = "none"
    amplitude: str = "normal"
    speed: str = "normal"
    cut_seconds: float = 0.0
    cut_phrase: str = CUT_PHRASES[0]


def format_timestamp(seconds: float) -> str:
    """MM:SS.mmm, e.g. 3.5 -> 00:03.500."""
    ms = int(round(seconds * 1000))
    m, rem = divmod(ms, 60000)
    return f"{m:02d}:{rem // 1000:02d}.{rem % 1000:03d}"


def camera_sentence(camera: str, amplitude: str = "normal", speed: str = "normal") -> str:
    key = (camera or "none").strip().lower()
    if key not in CAMERA_MOVES:
        raise PromptError(f"unknown camera move {camera!r}")
    entry = CAMERA_MOVES[key]
    if entry is None:
        return ""
    verb, graded = entry
    parts = [f"The camera {verb}"]
    if graded:
        if amplitude == "small":
            parts.append("with small amplitude")
        elif amplitude == "large":
            parts.append("with large amplitude")
        if speed == "slow":
            parts.append("at slow speed")
        elif speed == "fast":
            parts.append("at fast speed")
    return " ".join(parts) + "."


_LEAD_WORDS = {"a", "an", "the", "close", "wide", "medium", "extreme", "over", "low", "high", "aerial", "two", "three",
               "slow", "fast", "tight", "overhead", "side", "front", "back"}


def _lower_first(text: str) -> str:
    """'A close-up of...' -> 'a close-up of...' so it reads after 'the camera cuts to'.

    Only common leading words are lowered, so names and acronyms ('NASA', 'Maya') are left alone."""
    first = text.split(" ", 1)[0]
    if first and first[0].isupper() and first[1:] == first[1:].lower() and first.lower() in _LEAD_WORDS:
        return first[0].lower() + text[1:]
    return text


def _sentences(text: str) -> int:
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return len(parts)


def alignment_instruction(mode: str, last_shot_index: int, duration_s: float) -> str:
    """Exact first-line instruction from the guide (empty for T2VA)."""
    s = f"{duration_s:.2f}"
    if mode == "T2VA":
        return ""
    if mode == "I2VA":
        return "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    if mode == "FL2VA":
        return ("How the reference pictures align with the target video \u2014 Picture 1 (from Shot 1) aligns with the "
                f"0.00-second mark of the target video; Picture 2 (from Shot {last_shot_index}) aligns with the "
                f"{s}-second mark of the target video.")
    if mode == "L2VA":
        return (f"How the reference pictures align with the target video \u2014 <Picture 1> (from [Shot {last_shot_index}]) "
                f"aligns with the {s}-second mark of the target video.")
    raise PromptError(f"unknown mode {mode!r}")


def lint(description: str, soundscape: str, music: str, mode: str, shot_count: int, total_len: int) -> list:
    w = []
    opens, closes = description.count("<d>"), description.count("</d>")
    if opens != closes:
        w.append("unbalanced <d> ... </d> dialogue tags")
    for m in re.finditer(r"<d>(.*?)</d>", description, flags=re.S):
        if not re.match(r"\s*\[[^\]]+\]", m.group(1)):
            w.append("dialogue inside <d> should start with a language tag such as [English]")
            break
    if re.search(r"in an off-screen voiceover", description) and "lips remain" not in description:
        w.append("voiceover lines should be followed by a statement that the on-screen character's lips remain closed")
    if "<d>" in soundscape or "<d>" in music:
        w.append("dialogue belongs in the description, not in overall_soundscape / non_diegetic_music")
    if soundscape.strip().upper() != "N/A":
        n = _sentences(soundscape)
        if n == 0:
            w.append("overall_soundscape is empty (write 1-4 sentences, or N/A only for true silence)")
        elif n > 4:
            w.append(f"overall_soundscape has {n} sentences; the guide asks for 1-4")
    if music.strip().upper() != "N/A":
        n = _sentences(music)
        if n == 0:
            w.append("non_diegetic_music is empty (write 1-3 sentences, or N/A)")
        elif n > 3:
            w.append(f"non_diegetic_music has {n} sentences; the guide asks for 1-3")
    if mode == "FL2VA" and shot_count > 1:
        w.append("FL2VA generally works best as a single shot so the model can interpolate between the frames")
    if total_len > MAX_PROMPT_CHARS:
        w.append(f"prompt is {total_len} characters; community notes on MiniMax's validation cite a ~{MAX_PROMPT_CHARS}-character limit")
    return w


def build_prompt(mode: str, shots, style: str, soundscape: str, music: str, length_frames: int):
    """Return (prompt, warnings). Raises PromptError for unusable input.

    shots: iterable of ShotSpec; blank-text shots are skipped. Shot 1's cut_seconds is ignored.
    """
    if mode not in MODES:
        raise PromptError(f"unknown mode {mode!r}")
    duration = length_frames / FPS
    active = [s for s in shots if s.text and s.text.strip()]
    if not active:
        raise PromptError("write at least shot 1")
    warnings = []

    chunks = []
    last_cut = 0.0
    for i, s in enumerate(active, start=1):
        text = s.text.strip()
        cam = camera_sentence(s.camera, s.amplitude, s.speed)
        if i == 1:
            lead = (style or "").strip().rstrip(",")
            body = f"{lead}, {text}" if lead else text
            chunk = f"[Shot 1] {body}"
        else:
            if s.cut_seconds <= last_cut:
                raise PromptError(f"shot {i} cut time {format_timestamp(s.cut_seconds)} must be later than the previous cut ({format_timestamp(last_cut)})")
            if s.cut_seconds >= duration:
                raise PromptError(f"shot {i} cut time {format_timestamp(s.cut_seconds)} is outside the {duration:.2f}s clip")
            if s.cut_seconds - last_cut < 1.5:
                warnings.append(f"shot {i} starts only {s.cut_seconds - last_cut:.1f}s after the previous cut; very short shots render poorly")
            last_cut = s.cut_seconds
            phrase = s.cut_phrase if s.cut_phrase in CUT_PHRASES else CUT_PHRASES[0]
            chunk = f"[Shot {i}] At {format_timestamp(s.cut_seconds)}, {phrase} {_lower_first(text)}"
        if cam:
            chunk = chunk.rstrip() + ("" if chunk.rstrip().endswith((".", "!", "?", ">")) else ".") + " " + cam
        chunks.append(chunk)

    description = " ".join(chunks)
    sound = (soundscape or "").strip() or "N/A"
    mus = (music or "").strip() or "N/A"

    if not (soundscape or "").strip():
        warnings.append("overall_soundscape was empty, so N/A (silence) was written; describe ambient/physical sounds if you want audio")

    parts = []
    instruction = alignment_instruction(mode, len(active), duration)
    if instruction:
        parts.append(instruction)
    parts.append(f"integrated_multimodal_description: {description}")
    parts.append(f"overall_soundscape: {sound}")
    parts.append(f"non_diegetic_music: {mus}")
    prompt = "\n\n".join(parts)

    warnings.extend(lint(description, sound, mus, mode, len(active), len(prompt)))
    return prompt, warnings
