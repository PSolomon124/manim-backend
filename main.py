import os
import re
import uuid
import asyncio
import ast
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ============================================================
# CONFIGURATION
# ============================================================

APP_VERSION = "7.2.0"

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "rendered_videos"
MEDIA_DIR = BASE_DIR / "media"
VOICE_DIR = BASE_DIR / "voiceovers"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VOICE = os.getenv(
    "EDGE_TTS_VOICE",
    "en-NG-EzinneNeural",
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Tezla Animator Rendering Engine",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/videos",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="videos",
)


# ============================================================
# DATA MODELS
# ============================================================

class SolutionStep(BaseModel):
    step_number: int
    math_latex: str = ""
    explanation: str = ""
    visual_action: Optional[str] = None
    emphasis: Optional[List[str]] = None
    scene: Optional[str] = None
    equation_state: Optional[str] = None
    segment_index: Optional[int] = None
    diagram_cues: Optional[List[dict]] = None
    starting_state: Optional[List[str]] = None
    ending_state: Optional[List[str]] = None
    layout_mode: Optional[str] = None
    retain: Optional[List[str]] = None
    remove: Optional[List[str]] = None
    focus: Optional[List[str]] = None


class RenderRequest(BaseModel):
    prompt: str
    solution_steps: Optional[List[SolutionStep]] = None
    diagram_spec: Optional[dict] = None
    visual_plan: Optional[dict] = None
    theme: Optional[dict] = None
    version: Optional[int] = None
    plan_version: Optional[str] = None


# ============================================================
# AI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are an expert educational mathematics animation programmer.

Generate a complete executable Manim Python script.

RULES:

1. Import:
   from manim import *

2. Define exactly:
   class GeneratedScene(Scene):

3. Do not use:
   - manim_voiceover
   - VoiceoverScene
   - EdgeTTSService

4. The script must execute directly using Manim.

5. Use MathTex for mathematical expressions.

6. Never put $ symbols inside MathTex.

7. Use valid LaTeX with SINGLE backslashes.

8. Keep all mathematical content inside the visible frame.

9. Avoid overlapping objects.

10. Prefer:
    next_to()
    arrange()
    shift()
    move_to()
    to_edge()
    align_to()

11. Keep equations readable.

12. Use educational animations instead of excessive decorative
    animation.

13. Define exactly:
    class GeneratedScene(Scene):

14. Return only valid Python code.
"""


# ============================================================
# CODE CLEANING
# ============================================================

def clean_code_block(code: str) -> str:

    code = str(code).strip()

    code = re.sub(
        r"^```(?:python|py)?\s*",
        "",
        code,
        flags=re.IGNORECASE,
    )

    code = re.sub(
        r"\s*```$",
        "",
        code,
    )

    return code.strip()


# ============================================================
# LATEX CLEANING
# ============================================================

def clean_latex(text: str) -> str:
    """
    Normalize LaTeX coming from Lovable / JSON / AI.
    """

    text = str(text or "").strip()

    if not text:
        return ""

    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2].strip()

    elif text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # Collapse repeated backslashes.
    text = re.sub(
        r"\\{2,}",
        r"\\",
        text,
    )

    commands = [
        "boxed",
        "quad",
        "text",
        "frac",
        "sqrt",
        "left",
        "right",
        "times",
        "cdot",
        "pm",
        "leq",
        "geq",
        "neq",
        "approx",
        "infty",
        "sin",
        "cos",
        "tan",
        "log",
        "ln",
        "theta",
        "alpha",
        "beta",
        "gamma",
        "pi",
    ]

    for command in commands:

        text = re.sub(
            rf"(?<!\\)\b{command}\b",
            rf"\\{command}",
            text,
        )

    text = text.replace("$", "")

    # Lovable/JSON can occasionally turn simple geometry labels into
    # malformed LaTeX commands such as \\b or \\h.  In TeX those are not the
    # intended variables and can render strangely.  Restore them to ordinary
    # italic math variables.
    text = re.sub(r"\\b(?![A-Za-z])", "b", text)
    text = re.sub(r"\\h(?![A-Za-z])", "h", text)

    # Keep common area formula variables simple and predictable.
    text = re.sub(r"(?<![A-Za-z])([bh])(?![A-Za-z])", r"\1", text)

    return text.strip()


# ============================================================
# SAFE PYTHON STRING
# ============================================================

def python_literal(text: str) -> str:
    return repr(str(text))


# ============================================================
# SPOKEN TEXT CLEANING
# ============================================================

def clean_spoken_text(text: str) -> str:
    """
    Explanation is expected to already be natural spoken English.

    This function only removes formatting artifacts.
    It does NOT try to convert full LaTeX mathematics into speech.
    """

    text = str(text or "").strip()

    if not text:
        return ""

    text = text.replace("$$", "")
    text = text.replace("$", "")

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # Remove markdown emphasis.
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")

    # Small set of safe conversions.
    replacements = {
        r"\\times": " times ",
        r"\\cdot": " times ",
        r"\\pm": " plus or minus ",
        r"\\leq": " less than or equal to ",
        r"\\geq": " greater than or equal to ",
        r"\\neq": " not equal to ",
        r"\\approx": " approximately ",
        r"\\infty": " infinity ",
        r"\\pi": " pi ",
        r"\\theta": " theta ",
        r"\\alpha": " alpha ",
        r"\\beta": " beta ",
        r"\\gamma": " gamma ",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(
            pattern,
            replacement,
            text,
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def clean_title(text: str) -> str:

    text = str(text or "").strip()

    text = text.replace("\n", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    if not text:
        return "Mathematics Solution"

    # Avoid extremely long question as title.
    if len(text) > 72:
        text = text[:69].rstrip() + "..."

    return text


# ============================================================
# DISPLAY TEXT
# ============================================================

def make_display_explanation(
    explanation: str,
    max_chars: int = 48,
    max_lines: int = 4,
) -> str:
    """
    Produce short screen text.

    Full explanation remains in narration.
    """

    text = clean_spoken_text(explanation)

    if not text:
        return ""

    words = text.split()

    lines = []
    current = ""

    for word in words:

        candidate = (
            f"{current} {word}"
            if current
            else word
        )

        if len(candidate) > max_chars:

            if current:
                lines.append(current)

            current = word

            if len(lines) >= max_lines:
                break

        else:
            current = candidate

    if (
        current
        and len(lines) < max_lines
    ):
        lines.append(current)

    result = "\n".join(lines)

    if len(words) > len(result.split()):
        result = result.rstrip(".") + "..."

    return result


# ============================================================
# EDGE TTS
# ============================================================

def generate_edge_tts_sync(
    text: str,
    output_file: str,
    voice: str = DEFAULT_VOICE,
):
    """
    Generate one narration MP3.
    """

    import edge_tts

    async def generate():

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
        )

        await communicate.save(
            output_file
        )

    asyncio.run(generate())


# ============================================================
# AUDIO DURATION
# ============================================================

def get_audio_duration(
    audio_file: str,
) -> float:

    path = Path(audio_file)

    if not path.exists():
        return 0.0

    try:

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )

        duration = float(
            result.stdout.strip()
        )

        return max(
            duration,
            0.1,
        )

    except Exception as exc:

        print(
            f"Could not determine audio duration: {exc}"
        )

        return 0.0


# ============================================================
# PREPARE NARRATION
# ============================================================

def prepare_step_audio(
    steps: List[SolutionStep],
    job_id: str,
):
    """
    Generate ALL voice files before constructing Manim script.

    Narration therefore becomes the master timeline.
    """

    voice_dir = VOICE_DIR / job_id

    voice_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_data = []

    print("=" * 70)
    print("PREPARING NARRATION")
    print("=" * 70)

    for index, step in enumerate(
        steps,
        start=1,
    ):

        narration = clean_spoken_text(
            step.explanation
        )

        if not narration:
            audio_data.append({
                "path": None,
                "duration": 0.0,
                "exists": False,
                "narration": "",
            })
            continue

        audio_file = (
            voice_dir /
            f"step_{index}.mp3"
        )

        audio_exists = False
        duration = 0.0

        try:

            print(
                f"TTS STEP {index}: {narration}"
            )

            generate_edge_tts_sync(
                narration,
                str(audio_file),
                DEFAULT_VOICE,
            )

            audio_exists = (
                audio_file.exists()
                and
                audio_file.stat().st_size > 0
            )

            if audio_exists:

                duration = get_audio_duration(
                    str(audio_file)
                )

                print(
                    f"TTS STEP {index} OK: "
                    f"{duration:.2f}s "
                    f"({audio_file.stat().st_size} bytes)"
                )

            else:

                print(
                    f"TTS STEP {index}: "
                    "file was not created."
                )

        except Exception as exc:

            print(
                f"EDGE TTS FAILED FOR STEP "
                f"{index}: {exc}"
            )

        audio_data.append(
            {
                "index": index,
                "narration": narration,
                "path": audio_file,
                "exists": audio_exists,
                "duration": duration,
            }
        )

    return audio_data


# ============================================================
# MANIM DIAGRAM SETUP
# ============================================================

def add_diagram_setup(
    lines: list,
    diagram_spec: Optional[dict],
):
    """
    Create diagram OBJECTS but do not immediately reveal everything.

    Visual actions during narration reveal/highlight them.
    """

    if not diagram_spec:
        return

    kind = str(
        diagram_spec.get(
            "kind",
            "",
        )
    ).strip()

    if kind == "parallelogram_to_triangles":

        lines.extend([
            "",
            "        # ========================================",
            "        # PARALLELOGRAM TEACHING DIAGRAM",
            "        # ========================================",
            "",
            "        p1 = LEFT * 5.4 + DOWN * 1.25",
            "        p2 = LEFT * 1.8 + DOWN * 1.25",
            "        p3 = LEFT * 0.25 + UP * 1.45",
            "        p4 = LEFT * 3.85 + UP * 1.45",
            "",
            "        parallelogram = Polygon(",
            "            p1, p2, p3, p4,",
            "            color=TEZLA_CYAN,",
            "            fill_opacity=0.10,",
            "            stroke_width=4,",
            "        )",
            "",
            "        base_line = Line(",
            "            p1, p2,",
            "            color=TEZLA_GOLD,",
            "            stroke_width=6,",
            "        )",
            "",
            "        height_foot = np.array([p4[0], p1[1], 0])",
            "",
            "        height_line = DashedLine(",
            "            p4,",
            "            height_foot,",
            "            color=TEZLA_PINK,",
            "            stroke_width=4,",
            "            dash_length=0.10,",
            "        )",
            "",
            "        diagonal = Line(",
            "            p1, p3,",
            "            color=TEZLA_GOLD,",
            "            stroke_width=5,",
            "        )",
            "",
            "        triangle_one = Polygon(",
            "            p1, p2, p3,",
            "            color=TEZLA_PURPLE,",
            "            fill_opacity=0.25,",
            "            stroke_opacity=0,",
            "        )",
            "",
            "        triangle_two = Polygon(",
            "            p1, p3, p4,",
            "            color=TEZLA_GREEN,",
            "            fill_opacity=0.25,",
            "            stroke_opacity=0,",
            "        )",
            "",
            "        base_label = MathTex(",
            "            r'b',",
            "            font_size=38,",
            "        )",
            "",
            "        base_label.next_to(",
            "            base_line,",
            "            DOWN,",
            "            buff=0.25,",
            "        )",
            "",
            "        height_label = MathTex(",
            "            r'h',",
            "            font_size=38,",
            "        )",
            "",
            "        height_label.next_to(",
            "            height_line,",
            "            LEFT,",
            "            buff=0.18,",
            "        )",
            "",
            "        right_angle = RightAngle(",
            "            Line(height_foot, p4),",
            "            Line(height_foot, p2),",
            "            length=0.18,",
            "            quadrant=(1, 1),",
            "        )",
            "",
            "        diagram_created = False",
            "        base_height_visible = False",
            "        diagonal_visible = False",
            "        triangles_visible = False",
            "",
        ])

    elif kind == "triangle_angles":

        lines.extend([
            "",
            "        # ========================================",
            "        # TRIANGLE / EXTERIOR-ANGLE DIAGRAM",
            "        # ========================================",
            "",
            "        ga = LEFT * 5.1 + DOWN * 1.35",
            "        gb = LEFT * 1.45 + DOWN * 1.35",
            "        gc = LEFT * 3.45 + UP * 1.75",
            "        gext = LEFT * 0.15 + DOWN * 1.35",
            "",
            "        angle_triangle = Polygon(",
            "            ga, gb, gc,",
            "            color=TEZLA_CYAN,",
            "            fill_opacity=0.10,",
            "            stroke_width=4,",
            "        )",
            "        exterior_ray = Line(",
            "            gb, gext,",
            "            color=TEZLA_GOLD,",
            "            stroke_width=5,",
            "        )",
            "",
            "        angle_a = Angle(Line(ga, gb), Line(ga, gc), radius=0.38, color=TEZLA_PURPLE)",
            "        angle_b = Angle(Line(gb, gc), Line(gb, ga), radius=0.42, color=TEZLA_PINK)",
            "        angle_c = Angle(Line(gc, ga), Line(gc, gb), radius=0.38, color=TEZLA_GREEN)",
            "        exterior_arc = Angle(Line(gb, gext), Line(gb, gc), radius=0.62, color=TEZLA_GOLD)",
            "",
            "        angle_a_label = MathTex(r'A', font_size=28, color=TEZLA_PURPLE).next_to(angle_a, UP, buff=0.05)",
            "        angle_b_label = MathTex(r'B', font_size=28, color=TEZLA_PINK).next_to(angle_b, UP, buff=0.05)",
            "        angle_c_label = MathTex(r'C', font_size=28, color=TEZLA_GREEN).next_to(angle_c, DOWN, buff=0.05)",
            "        exterior_label = Text('exterior', font_size=22, color=TEZLA_GOLD).next_to(exterior_arc, UP, buff=0.10)",
            "",
            "        angle_triangle_created = False",
            "        interior_angles_visible = False",
            "        exterior_angle_visible = False",
            "",
        ])

    elif kind == "triangle":

        lines.extend([
            "",
            "        # ========================================",
            "        # TRIANGLE TEACHING DIAGRAM",
            "        # ========================================",
            "",
            "        ta = LEFT * 5.0 + DOWN * 1.4",
            "        tb = LEFT * 1.2 + DOWN * 1.4",
            "        tc = LEFT * 3.0 + UP * 1.8",
            "",
            "        triangle = Polygon(",
            "            ta, tb, tc,",
            "            color=TEZLA_CYAN,",
            "            fill_opacity=0.14,",
            "            stroke_width=4,",
            "        )",
            "",
            "        triangle_base = Line(",
            "            ta, tb,",
            "            color=TEZLA_GOLD,",
            "            stroke_width=6,",
            "        )",
            "",
            "        triangle_height_foot = np.array([tc[0], ta[1], 0])",
            "",
            "        triangle_height = DashedLine(",
            "            tc,",
            "            triangle_height_foot,",
            "            color=TEZLA_PINK,",
            "            stroke_width=4,",
            "        )",
            "",
            "        triangle_base_label = MathTex(",
            "            r'b',",
            "            font_size=38,",
            "        ).next_to(triangle_base, DOWN, buff=0.25)",
            "",
            "        triangle_height_label = MathTex(",
            "            r'h',",
            "            font_size=38,",
            "        ).next_to(triangle_height, RIGHT, buff=0.20)",
            "",
            "        triangle_created = False",
            "        triangle_dimensions_visible = False",
            "",
        ])

    elif kind == "circle":

        lines.extend([
            "",
            "        # ========================================",
            "        # CIRCLE TEACHING DIAGRAM",
            "        # ========================================",
            "",
            "        circle = Circle(",
            "            radius=1.8,",
            "            color=TEZLA_CYAN,",
            "            fill_opacity=0.12,",
            "            stroke_width=4,",
            "        )",
            "",
            "        circle.move_to(LEFT * 3.2 + DOWN * 0.1)",
            "",
            "        center_dot = Dot(circle.get_center())",
            "",
            "        radius_line = Line(",
            "            circle.get_center(),",
            "            circle.get_right(),",
            "            color=TEZLA_GOLD,",
            "            stroke_width=5,",
            "        )",
            "",
            "        radius_label = MathTex(",
            "            r'r',",
            "            font_size=38,",
            "        ).next_to(radius_line, UP, buff=0.15)",
            "",
            "        circle_created = False",
            "        radius_visible = False",
            "",
        ])

    elif kind == "number_line":

        lines.extend([
            "",
            "        number_line = NumberLine(",
            "            x_range=[-5, 5, 1],",
            "            length=6.2,",
            "            include_numbers=True,",
            "            font_size=26,",
            "        )",
            "",
            "        number_line.move_to(LEFT * 3.0 + DOWN * 0.2)",
            "",
            "        number_line_created = False",
            "",
        ])

    elif kind == "coordinate_plane":

        lines.extend([
            "",
            "        plane = NumberPlane(",
            "            x_range=[-4, 4, 1],",
            "            y_range=[-3, 3, 1],",
            "            x_length=6.0,",
            "            y_length=4.5,",
            "            background_line_style={",
            "                'stroke_opacity': 0.25,",
            "            },",
            "        )",
            "",
            "        plane.move_to(LEFT * 3.0 + DOWN * 0.25)",
            "",
            "        plane_created = False",
            "",
        ])


# ============================================================
# VISUAL ACTIONS
# ============================================================

def add_visual_action(
    lines: list,
    visual_action: Optional[str],
    diagram_spec: Optional[dict],
    animation_time: float,
):
    """
    Add deterministic educational diagram actions.

    Lovable may provide visual_action.

    If it does not, safe defaults are used.
    """

    if not diagram_spec:
        return

    kind = str(
        diagram_spec.get(
            "kind",
            "",
        )
    ).strip()

    action = str(
        visual_action or ""
    ).strip().lower()

    rt = max(
        min(animation_time, 1.6),
        0.35,
    )
    # Keep each principal diagram event within the narration-derived budget.
    sync_rt = max(min(animation_time * 0.72, 1.15), 0.28)

    if kind == "parallelogram_to_triangles":

        if action in {
            "",
            "show_parallelogram",
            "introduce_shape",
        }:

            lines.extend([
                "        if not diagram_created:",
                "            self.play(",
                "                Create(parallelogram),",
                f"                run_time={rt:.2f},",
                "            )",
                "            diagram_created = True",
            ])

        elif action in {
            "show_base_height",
            "highlight_base_height",
            "show_dimensions",
        }:

            lines.extend([
                "        if not diagram_created:",
                "            self.play(",
                "                Create(parallelogram),",
                f"                run_time={max(rt * 0.55, 0.35):.2f},",
                "            )",
                "            diagram_created = True",
                "",
                "        if not base_height_visible:",
                "            self.play(",
                "                Create(base_line),",
                "                Create(height_line),",
                "                Write(base_label),",
                "                Write(height_label),",
                "                FadeIn(right_angle),",
                f"                run_time={max(rt * 0.75, 0.45):.2f},",
                "            )",
                "            base_height_visible = True",
                "        else:",
                "            self.play(",
                "                Indicate(base_line),",
                "                Indicate(height_line),",
                f"                run_time={rt:.2f},",
                "            )",
            ])

        elif action in {
            "draw_diagonal",
            "divide_parallelogram",
            "split_into_triangles",
        }:

            lines.extend([
                "        if not diagram_created:",
                "            self.play(",
                "                Create(parallelogram),",
                f"                run_time={max(rt * 0.5, 0.35):.2f},",
                "            )",
                "            diagram_created = True",
                "",
                "        if not diagonal_visible:",
                "            self.play(",
                "                Create(diagonal),",
                f"                run_time={rt:.2f},",
                "            )",
                "            diagonal_visible = True",
                "",
                "        if not triangles_visible:",
                "            self.play(",
                "                FadeIn(triangle_one),",
                "                FadeIn(triangle_two),",
                f"                run_time={max(rt * 0.65, 0.40):.2f},",
                "            )",
                "            triangles_visible = True",
            ])

        elif action in {
            "highlight_triangle",
            "show_half",
            "highlight_half",
        }:

            lines.extend([
                "        if not diagram_created:",
                "            self.play(",
                "                Create(parallelogram),",
                f"                run_time={max(rt * 0.4, 0.35):.2f},",
                "            )",
                "            diagram_created = True",
                "",
                "        if not diagonal_visible:",
                "            self.play(",
                "                Create(diagonal),",
                f"                run_time={max(rt * 0.5, 0.35):.2f},",
                "            )",
                "            diagonal_visible = True",
                "",
                "        if not triangles_visible:",
                "            self.play(",
                "                FadeIn(triangle_one),",
                "                FadeIn(triangle_two),",
                f"                run_time={max(rt * 0.5, 0.35):.2f},",
                "            )",
                "            triangles_visible = True",
                "",
                "        self.play(",
                "            Indicate(triangle_one),",
                f"            run_time={rt:.2f},",
                "        )",
            ])

        elif action in {
            "highlight_base",
            "show_base",
        }:

            lines.extend([
                "        if not diagram_created:",
                "            self.play(",
                "                Create(parallelogram),",
                f"                run_time={max(rt * 0.5, 0.35):.2f},",
                "            )",
                "            diagram_created = True",
                "",
                "        if not base_height_visible:",
                "            self.play(",
                "                Create(base_line),",
                "                Write(base_label),",
                f"                run_time={rt:.2f},",
                "            )",
                "            base_height_visible = True",
                "        else:",
                "            self.play(",
                "                Indicate(base_line),",
                f"                run_time={rt:.2f},",
                "            )",
            ])

        elif action in {
            "highlight_height",
            "show_height",
        }:

            lines.extend([
                "        if not diagram_created:",
                "            self.play(",
                "                Create(parallelogram),",
                f"                run_time={max(rt * 0.5, 0.35):.2f},",
                "            )",
                "            diagram_created = True",
                "",
                "        self.play(",
                "            Create(height_line),",
                "            Write(height_label),",
                f"            run_time={rt:.2f},",
                "        )",
            ])

        else:

            lines.extend([
                "        if not diagram_created:",
                "            self.play(",
                "                Create(parallelogram),",
                f"                run_time={rt:.2f},",
                "            )",
                "            diagram_created = True",
            ])

    elif kind == "triangle_angles":

        if action in {"", "show_triangle", "introduce_shape"}:
            lines.extend([
                "        if not angle_triangle_created:",
                f"            self.play(Create(angle_triangle), run_time={sync_rt:.2f})",
                "            angle_triangle_created = True",
            ])

        elif action in {"show_interior_angles", "highlight_angle_sum"}:
            lines.extend([
                "        if not angle_triangle_created:",
                f"            self.play(Create(angle_triangle), run_time={max(sync_rt * 0.45, 0.25):.2f})",
                "            angle_triangle_created = True",
                "        if not interior_angles_visible:",
                "            self.play(",
                "                Create(angle_a), Create(angle_b), Create(angle_c),",
                "                FadeIn(angle_a_label), FadeIn(angle_b_label), FadeIn(angle_c_label),",
                f"                run_time={max(sync_rt * 0.55, 0.30):.2f},",
                "            )",
                "            interior_angles_visible = True",
                "        else:",
                f"            self.play(Indicate(VGroup(angle_a, angle_b, angle_c)), run_time={sync_rt:.2f})",
            ])

        elif action in {"show_exterior_angle", "highlight_exterior_angle"}:
            lines.extend([
                "        if not angle_triangle_created:",
                f"            self.play(Create(angle_triangle), run_time={max(sync_rt * 0.40, 0.25):.2f})",
                "            angle_triangle_created = True",
                "        if not exterior_angle_visible:",
                "            self.play(",
                "                Create(exterior_ray), Create(exterior_arc), FadeIn(exterior_label),",
                f"                run_time={max(sync_rt * 0.60, 0.30):.2f},",
                "            )",
                "            exterior_angle_visible = True",
                "        else:",
                f"            self.play(Indicate(exterior_arc, color=TEZLA_GOLD), run_time={sync_rt:.2f})",
            ])

        else:
            lines.extend([
                "        if not angle_triangle_created:",
                f"            self.play(Create(angle_triangle), run_time={sync_rt:.2f})",
                "            angle_triangle_created = True",
            ])

    elif kind == "triangle":

        if action in {
            "show_base_height",
            "show_dimensions",
            "highlight_base_height",
        }:

            lines.extend([
                "        if not triangle_created:",
                "            self.play(",
                "                Create(triangle),",
                f"                run_time={max(rt * 0.6, 0.35):.2f},",
                "            )",
                "            triangle_created = True",
                "",
                "        if not triangle_dimensions_visible:",
                "            self.play(",
                "                Create(triangle_base),",
                "                Create(triangle_height),",
                "                Write(triangle_base_label),",
                "                Write(triangle_height_label),",
                f"                run_time={rt:.2f},",
                "            )",
                "            triangle_dimensions_visible = True",
            ])

        else:

            lines.extend([
                "        if not triangle_created:",
                "            self.play(",
                "                Create(triangle),",
                f"                run_time={rt:.2f},",
                "            )",
                "            triangle_created = True",
            ])

    elif kind == "circle":

        if action in {
            "show_radius",
            "highlight_radius",
        }:

            lines.extend([
                "        if not circle_created:",
                "            self.play(",
                "                Create(circle),",
                f"                run_time={max(rt * 0.6, 0.35):.2f},",
                "            )",
                "            circle_created = True",
                "",
                "        if not radius_visible:",
                "            self.play(",
                "                FadeIn(center_dot),",
                "                Create(radius_line),",
                "                Write(radius_label),",
                f"                run_time={rt:.2f},",
                "            )",
                "            radius_visible = True",
            ])

        else:

            lines.extend([
                "        if not circle_created:",
                "            self.play(",
                "                Create(circle),",
                f"                run_time={rt:.2f},",
                "            )",
                "            circle_created = True",
            ])

    elif kind == "number_line":

        lines.extend([
            "        if not number_line_created:",
            "            self.play(",
            "                Create(number_line),",
            f"                run_time={rt:.2f},",
            "            )",
            "            number_line_created = True",
        ])

    elif kind == "coordinate_plane":

        lines.extend([
            "        if not plane_created:",
            "            self.play(",
            "                Create(plane),",
            f"                run_time={rt:.2f},",
            "            )",
            "            plane_created = True",
        ])



def infer_diagram_spec(
    prompt: str,
    steps: List[SolutionStep],
    diagram_spec: Optional[dict],
) -> Optional[dict]:
    """Infer a deterministic educational diagram when Lovable omitted one."""
    if diagram_spec and str(diagram_spec.get("kind", "")).strip():
        return diagram_spec

    corpus = " ".join(
        [prompt]
        + [step.explanation or "" for step in steps]
        + [step.visual_action or "" for step in steps]
    ).lower()

    if "parallelogram" in corpus:
        return {"kind": "parallelogram_to_triangles", "params": {}, "animation": {}}

    if any(word in corpus for word in (
        "exterior angle", "interior angle", "triangle angle",
        "angles of a triangle", "angle of a triangle",
    )):
        return {"kind": "triangle_angles", "params": {}, "animation": {}}

    if "triangle" in corpus:
        return {"kind": "triangle", "params": {}, "animation": {}}

    if any(word in corpus for word in ("circle", "radius", "diameter", "circumference")):
        return {"kind": "circle", "params": {}, "animation": {}}

    if any(word in corpus for word in (
        "coordinate plane", "coordinate", "x-axis", "y-axis",
        "graph of", "plot the graph", "plot point",
    )):
        return {"kind": "coordinate_plane", "params": {}, "animation": {}}

    if any(word in corpus for word in (
        "number line", "inequality", "interval",
    )):
        return {"kind": "number_line", "params": {}, "animation": {}}

    return None


def infer_visual_action(
    step: SolutionStep,
    diagram_spec: Optional[dict],
    step_index: int,
) -> str:
    """Map the canonical narration to a visual event without rewriting speech."""
    if step.visual_action:
        return step.visual_action

    if not diagram_spec:
        return ""

    kind = str(diagram_spec.get("kind", "")).strip()
    speech = (step.explanation or "").lower()

    if kind == "triangle_angles":
        if "exterior" in speech:
            return "show_exterior_angle"
        if any(x in speech for x in ("interior", "inside angle", "triangle angle")):
            return "show_interior_angles"
        if any(x in speech for x in ("sum", "180", "one hundred and eighty")):
            return "highlight_angle_sum"
        return "show_triangle"

    if kind == "parallelogram_to_triangles":
        if "diagonal" in speech or "two triangle" in speech:
            return "draw_diagonal"
        if "height" in speech and "base" in speech:
            return "show_base_height"
        if "height" in speech:
            return "show_height"
        if "base" in speech:
            return "show_base"
        return "show_parallelogram"

    if kind == "triangle":
        if "height" in speech or "base" in speech:
            return "show_base_height"
        return "show_triangle"

    if kind == "circle":
        if "radius" in speech:
            return "show_radius"
        return "show_circle"

    return ""


# ============================================================
# DIRECT TEACHING SCRIPT BUILDER
# ============================================================

def build_direct_manim_script(
    prompt: str,
    steps: List[SolutionStep],
    job_id: str,
    diagram_spec: Optional[dict] = None,
) -> str:
    # Diagram generation is narration-driven. Lovable may send a diagram
    # explicitly; otherwise infer one from the same canonical explanation.
    diagram_spec = infer_diagram_spec(prompt, steps, diagram_spec)
    """
    V5 Teaching Timeline Engine.

    Important principle:

    AUDIO STARTS BEFORE VISUAL STEP ANIMATIONS.

    Therefore the visual animation happens WHILE the teacher
    is speaking instead of before the narration.
    """

    audio_data = prepare_step_audio(
        steps,
        job_id,
    )

    title = clean_title(prompt)
    title = re.sub(r"\$\$.*?\$\$|\$.*?\$", "", title)
    title = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", "", title)
    title = title.replace("{", "").replace("}", "").replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip()
    if not title or any(ch in title for ch in ("=", "^", "\\")):
        title = "Interactive Mathematics"
    if len(title) > 42:
        title = title[:39].rstrip() + "..."

    lines = []

    lines.append(
        "from manim import *"
    )

    lines.append(
        "import numpy as np"
    )

    lines.append("")

    lines.append(
        "class GeneratedScene(Scene):"
    )

    lines.append(
        "    def construct(self):"
    )

    lines.append(
        '        self.camera.background_color = "#07111f"'
    )

    lines.append("")

    # ========================================================
    # HEADER
    # ========================================================

    lines.extend([
        "        # Tezla Animator visual identity",
        '        TEZLA_CYAN = "#22D3EE"',
        '        TEZLA_GOLD = "#FBBF24"',
        '        TEZLA_PINK = "#F472B6"',
        '        TEZLA_GREEN = "#34D399"',
        '        TEZLA_PURPLE = "#A78BFA"',
        '        TEZLA_PANEL = "#0F1B2D"',
        '        MUTED = "#94A3B8"',
        "",
        "        brand = Text(",
        "            'TEZLA ANIMATOR',",
        "            font_size=18,",
        "            weight=BOLD,",
        "            color=TEZLA_CYAN,",
        "        )",
        "        brand.to_corner(UL, buff=0.28)",
        "",
        "        title = Text(",
        f"            {python_literal(title)},",
        "            font_size=28,",
        "            weight=BOLD,",
        "        )",
        "        if title.width > 10.2:\n            title.scale_to_fit_width(10.2)",
        "        title.to_edge(UP, buff=0.28)",
        "",
        "        divider = Line(",
        "            LEFT * 6.4, RIGHT * 6.4,",
        "            color=TEZLA_CYAN,",
        "            stroke_opacity=0.32,",
        "        )",
        "        divider.next_to(title, DOWN, buff=0.16)",
        "",
        "        self.play(",
        "            FadeIn(brand), FadeIn(title), Create(divider),",
        "            run_time=0.55,",
        "        )",
        "",
    ])

    # ========================================================
    # DIAGRAM OBJECTS
    # ========================================================

    add_diagram_setup(
        lines,
        diagram_spec,
    )

    # ========================================================
    # RIGHT-SIDE TEACHING PANEL
    # ========================================================

    lines.extend([
        "",
        f"        equation_anchor = {'RIGHT * 3.15 + UP * 0.35' if diagram_spec else 'ORIGIN + DOWN * 0.05'}",
        "",
        f"        teaching_panel = RoundedRectangle(width={5.65 if diagram_spec else 11.7}, height=4.65, corner_radius=0.22, stroke_color=TEZLA_CYAN, stroke_opacity=0.22, fill_color=TEZLA_PANEL, fill_opacity=0.35)",
        f"        teaching_panel.move_to({'RIGHT * 3.15 + DOWN * 0.05' if diagram_spec else 'ORIGIN + DOWN * 0.05'})",
        "        self.play(FadeIn(teaching_panel), run_time=0.30)",
        "",
        "        previous_equation = None",
        "        previous_explanation = None",
        "        previous_step_label = None",
        "",
    ])

    # ========================================================
    # STEPS
    # ========================================================

    for i, step in enumerate(
        steps,
        start=1,
    ):

        audio = audio_data[i - 1]

        narration_duration = float(
            audio["duration"]
        )

        audio_exists = bool(
            audio["exists"]
        )

        latex = clean_latex(
            step.math_latex
        )

        display_explanation = (
            make_display_explanation(
                step.explanation
            )
        )

        visual_action = infer_visual_action(
            step,
            diagram_spec,
            i,
        )

        scene_type = str(
            step.scene or "equation"
        ).strip().lower()

        equation_state = str(
            step.equation_state
            or ("introduce" if i == 1 else "transform")
        ).strip().lower()

        emphasis = [
            str(item).strip()
            for item in (step.emphasis or [])
            if str(item).strip()
        ]

        # Reserve most of narration for actual teaching.
        #
        # Do not let entrance animation consume entire speech.
        if narration_duration > 0:

            visual_budget = min(
                1.6,
                max(
                    0.7,
                    narration_duration * 0.24,
                ),
            )

        else:

            visual_budget = 1.0

        lines.extend([
            "",
            "        # ========================================",
            f"        # STEP {i}",
            "        # ========================================",
            "",
            f"        step_label = Text(",
            f"            {python_literal(f'Step {i}')},",
            "            font_size=24,",
            "            weight=BOLD,",
            "        )",
            "",
            "        step_label.move_to(",
            f"            {'RIGHT * 3.15 + UP * 2.55' if diagram_spec else 'LEFT * 5.0 + UP * 2.55'}",
            "        )",
            "",
        ])

        # ====================================================
        # EQUATION
        # ====================================================

        if latex:

            lines.extend([
                "        tex = MathTex(",
                f"            {python_literal(latex)},",
                "            font_size=42,",
                "        )",
                "",
                f"        max_equation_width = {5.0 if diagram_spec else 10.8}",
                "        if tex.width > max_equation_width:",
                "            tex.scale_to_fit_width(max_equation_width)",
                "        if tex.height > 2.15:",
                "            tex.scale_to_fit_height(2.15)",
                "        tex.move_to(equation_anchor)",
                "",
            ])

        else:
            lines.extend([
                "        tex = previous_equation.copy() if 'previous_equation' in locals() else VGroup()",
                "",
            ])

        # ====================================================
        # NARRATION DISPLAY
        # ====================================================
        # Captions are rendered by the Lovable player near its controls.
        lines.extend([
            "        explanation = VGroup()",
            "",
        ])

        # ====================================================
        # START AUDIO FIRST
        # ====================================================

        if audio_exists:

            audio_path = (
                str(audio["path"])
                .replace("\\", "/")
            )

            lines.extend([
                "        # Start narration BEFORE animation.",
                "        self.add_sound(",
                f"            {python_literal(audio_path)}",
                "        )",
                "",
            ])

        # ====================================================
        # STEP LABEL TRANSITION
        # ====================================================

        if i == 1:

            lines.extend([
                "        self.play(",
                "            FadeIn(step_label),",
                f"            run_time={min(0.35, visual_budget):.2f},",
                "        )",
                "",
            ])

        else:

            lines.extend([
                "        self.play(",
                "            FadeOut(previous_step_label),",
                "            FadeIn(step_label),",
                f"            run_time={min(0.30, visual_budget):.2f},",
                "        )",
                "",
            ])

        # ====================================================
        # VISUAL DIAGRAM ACTION DURING SPEECH
        # ====================================================

        add_visual_action(
            lines,
            visual_action,
            diagram_spec,
            visual_budget,
        )

        lines.append("")

        # ====================================================
        # EQUATION TRANSITION DURING SPEECH
        # ====================================================

        equation_runtime = min(
            0.70,
            max(
                0.30,
                visual_budget * 0.42,
            ),
        )

        if i == 1 or equation_state == "introduce":

            lines.extend([
                "        self.play(",
                "            Write(tex),",
                f"            run_time={equation_runtime:.2f},",
                "        )",
                "",
            ])

        elif equation_state == "hold":

            # Keep the current equation on screen during prose-only
            # narration. If Lovable supplied a different equation,
            # transition to it safely.
            if latex == clean_latex(steps[i - 2].math_latex):

                lines.extend([
                    "        tex = previous_equation",
                    "        self.play(",
                    "            Indicate(previous_equation, color=TEZLA_CYAN),",
                    f"            run_time={equation_runtime:.2f},",
                    "        )",
                    "",
                ])

            else:

                lines.extend([
                    "        self.play(",
                    "            ReplacementTransform(previous_equation, tex),",
                    f"            run_time={equation_runtime:.2f},",
                    "        )",
                    "",
                ])

        elif equation_state in {"transform", "final"}:

            lines.extend([
                "        self.play(",
                "            ReplacementTransform(previous_equation, tex),",
                f"            run_time={equation_runtime:.2f},",
                "        )",
                "",
            ])

        else:

            lines.extend([
                "        self.play(",
                "            ReplacementTransform(previous_equation, tex),",
                f"            run_time={equation_runtime:.2f},",
                "        )",
                "",
            ])

        # State-aware equation color gives the learner a visual cue.
        if equation_state == "final":
            lines.extend([
                "        tex.set_color(TEZLA_GOLD)",
                "",
            ])
        elif equation_state == "introduce":
            lines.extend([
                "        tex.set_color(TEZLA_CYAN)",
                "",
            ])

        if emphasis:

            lines.extend([
                "        self.play(",
                "            Indicate(tex, color=TEZLA_GOLD),",
                f"            run_time={min(0.40, equation_runtime):.2f},",
                "        )",
                "",
            ])

        explanation_runtime = 0.0

        # ====================================================
        # ESTIMATE ELAPSED ANIMATION TIME
        # ====================================================

        label_time = (
            min(
                0.35 if i == 1 else 0.30,
                visual_budget,
            )
        )

        # Diagram actions can contain sequential plays. Do not subtract an
        # approximate diagram duration from the audio guard; this prevents
        # the next narration clip from starting before the current one ends.
        diagram_time = 0.0

        emphasis_time = (
            min(0.40, equation_runtime)
            if emphasis
            else 0.0
        )

        estimated_elapsed = (
            label_time
            + diagram_time
            + equation_runtime
            + emphasis_time
            + explanation_runtime
        )

        # ====================================================
        # WAIT ONLY FOR REMAINDER OF NARRATION
        # ====================================================

        if narration_duration > 0:

            remaining = (
                narration_duration
                - estimated_elapsed
            )

            # Small buffer only.
            if remaining > 0.15:

                lines.extend([
                    f"        self.wait({remaining + 0.18:.2f})",
                    "",
                ])

        else:

            # If TTS failed, leave enough time to read.
            lines.extend([
                "        self.wait(0.18)",
                "",
            ])

        lines.extend([
            "        previous_equation = tex",
            "        previous_explanation = explanation",
            "        previous_step_label = step_label",
            "",
        ])

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    if steps:

        final_latex = clean_latex(
            steps[-1].math_latex
        )

        last_state = str(
            steps[-1].equation_state or ""
        ).strip().lower()

        should_highlight_final = (
            last_state == "final"
            or not any(step.equation_state for step in steps)
        )

        if final_latex and should_highlight_final:

            lines.extend([
                "",
                "        # ========================================",
                "        # FINAL ANSWER HIGHLIGHT",
                "        # ========================================",
                "",
                "        final_box = SurroundingRectangle(",
                "            previous_equation,",
                "            color=TEZLA_GOLD,",
                "            buff=0.25,",
                "            corner_radius=0.08,",
                "        )",
                "",
                "        final_text = Text(",
                "            'KEY RESULT',",
                "            font_size=22,",
                "            weight=BOLD,",
                "            color=TEZLA_GOLD,",
                "        )",
                "",
                "        final_text.next_to(",
                "            final_box,",
                "            UP,",
                "            buff=0.20,",
                "        )",
                "",
                "        self.play(",
                "            Create(final_box),",
                "            FadeIn(final_text),",
                "            previous_equation.animate.scale(1.06),",
                "            run_time=0.60,",
                "        )",
                "",
                # Short viewing buffer, not 2+ seconds.
                "        self.wait(0.75)",
            ])

    return "\n".join(lines)


# ============================================================
# GEMINI
# ============================================================

def generate_with_gemini(
    prompt: str,
) -> str:

    from google import genai

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            SYSTEM_PROMPT,
            "\n\nUSER REQUEST:\n",
            prompt,
        ],
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return clean_code_block(
        response.text
    )


# ============================================================
# GROQ
# ============================================================

def generate_with_groq(
    prompt: str,
) -> str:

    from groq import Groq

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    client = Groq(
        api_key=api_key
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "Groq returned an empty response."
        )

    return clean_code_block(
        content
    )


# ============================================================
# OPENROUTER
# ============================================================

def generate_with_openrouter(
    prompt: str,
    model: str,
) -> str:

    from openai import OpenAI

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "OpenRouter returned an empty response."
        )

    return clean_code_block(
        content
    )


# ============================================================
# AI FALLBACK
# ============================================================

def generate_ai_manim_code(
    prompt: str,
):

    providers = []

    if os.getenv(
        "GEMINI_API_KEY"
    ):

        providers.append(
            (
                "gemini",
                lambda: generate_with_gemini(
                    prompt
                ),
            )
        )

    if os.getenv(
        "GROQ_API_KEY"
    ):

        providers.append(
            (
                "groq",
                lambda: generate_with_groq(
                    prompt
                ),
            )
        )

    if os.getenv(
        "OPENROUTER_API_KEY"
    ):

        providers.append(
            (
                "openrouter-free",
                lambda: generate_with_openrouter(
                    prompt,
                    "openrouter/free",
                ),
            )
        )

        providers.append(
            (
                "openrouter-cohere",
                lambda: generate_with_openrouter(
                    prompt,
                    "cohere/north-mini-code:free",
                ),
            )
        )

    if not providers:

        raise RuntimeError(
            "No AI provider API keys are configured."
        )

    errors = []

    for (
        provider_name,
        provider_function,
    ) in providers:

        try:

            print(
                f"Trying AI provider: "
                f"{provider_name}"
            )

            code = (
                provider_function()
            )

            if (
                code
                and
                "GeneratedScene" in code
            ):

                return (
                    code,
                    provider_name,
                )

            errors.append(
                f"{provider_name}: "
                "invalid generated code"
            )

        except Exception as exc:

            print(
                f"{provider_name} failed: "
                f"{exc}"
            )

            errors.append(
                f"{provider_name}: {exc}"
            )

    raise RuntimeError(
        "All AI providers failed:\n"
        + "\n".join(errors)
    )


# ============================================================
# VALIDATE MANIM CODE
# ============================================================

def validate_manim_code(
    code: str,
):

    if not code:

        raise ValueError(
            "Generated Manim code is empty."
        )

    forbidden = [
        "manim_voiceover",
        "VoiceoverScene",
        "EdgeTTSService",
    ]

    for item in forbidden:

        if item in code:

            raise ValueError(
                "Generated code contains "
                f"forbidden dependency: {item}"
            )

    if (
        "class GeneratedScene(Scene)"
        not in code
    ):

        raise ValueError(
            "Generated code does not contain "
            "GeneratedScene."
        )

    if (
        "from manim import *"
        not in code
    ):

        raise ValueError(
            "Generated code does not import Manim."
        )


# ============================================================
# RENDER MANIM
# ============================================================


def _merge_v7_visual_plan(
    steps: List[SolutionStep],
    visual_plan: Optional[dict],
) -> List[dict]:
    """Merge v7/v7.2 planner beats into canonical narration steps.

    v7.2 treats ordered actions as authoritative. diagram_cues remains an alias
    for backwards compatibility with older Lovable payloads.
    """
    beats = {}
    if isinstance(visual_plan, dict):
        for beat in visual_plan.get("beats", []) or []:
            if isinstance(beat, dict):
                try:
                    beats[int(beat.get("segment_index"))] = beat
                except Exception:
                    pass

    merged = []
    for i, step in enumerate(steps):
        seg = step.segment_index if step.segment_index is not None else i
        beat = beats.get(int(seg), {})
        actions = beat.get("actions")
        if not isinstance(actions, list):
            actions = beat.get("diagram_cues")
        if not isinstance(actions, list):
            actions = step.diagram_cues or []

        # Enforce deterministic action order even if the model returned strings.
        cleaned_actions = []
        for j, cue in enumerate(actions):
            if not isinstance(cue, dict):
                continue
            c = dict(cue)
            try:
                c["order"] = int(c.get("order", j))
            except Exception:
                c["order"] = j
            cleaned_actions.append(c)
        cleaned_actions.sort(key=lambda c: c.get("order", 0))

        equation_latex = beat.get("equation_latex")
        if equation_latex is None:
            equation_latex = step.math_latex or ""

        merged.append({
            "step_number": step.step_number,
            "segment_index": int(seg),
            "math_latex": clean_latex(str(equation_latex or "")),
            # Canonical slideshow narration remains the source of speech.
            "explanation": clean_spoken_text(step.explanation or ""),
            "narration": step.explanation or "",
            "scene": beat.get("scene") or step.scene or "equation",
            "equation_state": beat.get("equation_state") or step.equation_state or "hold",
            "actions": cleaned_actions,
            "diagram_cues": cleaned_actions,
            "starting_state": beat.get("starting_state") or step.starting_state or [],
            "ending_state": beat.get("ending_state") or step.ending_state or [],
            "layout_mode": beat.get("layout_mode") or step.layout_mode or "split",
            "retain": beat.get("retain") or step.retain or [],
            "remove": beat.get("remove") or step.remove or [],
            "focus": beat.get("focus") or step.focus or [],
            "emphasis": step.emphasis or [],
        })
    return merged


def build_direct_manim_script_v7(
    prompt: str,
    steps: List[SolutionStep],
    job_id: str,
    visual_plan: Optional[dict] = None,
    theme: Optional[dict] = None,
) -> str:
    """
    Tezla v7: canonical narration is the master clock.
    AI supplies declarative cues only; generated Python never executes AI code.
    """
    merged = _merge_v7_visual_plan(steps, visual_plan)
    audio = prepare_step_audio(steps, job_id)

    for i, item in enumerate(merged):
        a = audio[i] if i < len(audio) else {}
        item["audio_path"] = str(a.get("path")) if a.get("path") else None
        item["audio_duration"] = float(a.get("duration") or 0.0)

    title = clean_title(prompt)
    title = re.sub(r"\$\$.*?\$\$|\$.*?\$", "", title)
    title = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", "", title)
    title = re.sub(r"[{}_^=\\\\]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    if not title:
        title = "Interactive Lesson"
    title = title[:56]

    payload_literal = repr(merged)
    theme_literal = repr(theme or {})

    return f"""from manim import *
import numpy as np
import ast
import math

BEATS = {payload_literal}
THEME = {theme_literal}
TITLE_TEXT = {title!r}

class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = THEME.get("background", "#07111f")
        CYAN = THEME.get("primary", "#22D3EE")
        GOLD = THEME.get("accent", "#FBBF24")
        GREEN = "#34D399"
        PINK = "#F472B6"
        PURPLE = "#A78BFA"
        MUTED = "#94A3B8"

        brand = Text("TEZLA ANIMATOR", font_size=17, weight=BOLD, color=CYAN).to_corner(UL, buff=0.25)
        title = Text(TITLE_TEXT, font_size=25, weight=BOLD)
        if title.width > 9.6:
            title.scale_to_fit_width(9.6)
        title.to_edge(UP, buff=0.25)
        divider = Line(LEFT*6.4, RIGHT*6.4, color=CYAN, stroke_opacity=0.28).next_to(title, DOWN, buff=0.16)
        self.add(brand, title, divider)

        objects = {{}}
        current_eq = None
        eq_anchor = RIGHT*3.15 + UP*0.15
        diagram_center = LEFT*3.05 + DOWN*0.35

        def vec(v, default=(0,0,0)):
            try:
                a = list(v)
                if len(a) == 2: a.append(0)
                return np.array([float(a[0]), float(a[1]), float(a[2])])
            except Exception:
                return np.array(default, dtype=float)

        def bounded_point(v):
            p = vec(v)
            p[0] = max(-5.8, min(1.0, p[0]))
            p[1] = max(-3.0, min(2.5, p[1]))
            return p

        def safe_color(c, fallback=CYAN):
            if isinstance(c, str) and (c.startswith("#") or c.isalpha()):
                return c
            return fallback

        def safe_math(s, size=34, color=WHITE):
            try:
                m = MathTex(str(s), font_size=size, color=color)
            except Exception:
                m = Text(str(s).replace("\\\\", ""), font_size=max(18, int(size*0.65)), color=color)
            return m

        def safe_expr(expr):
            allowed_names = {{
                "x": 0.0, "pi": math.pi, "e": math.e,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
                "abs": abs,
            }}
            allowed_nodes = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Load,
                ast.Name, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
                ast.Pow, ast.USub, ast.UAdd, ast.Mod,
            )
            tree = ast.parse(str(expr), mode="eval")
            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    raise ValueError("unsafe expression")
                if isinstance(node, ast.Name) and node.id not in allowed_names:
                    raise ValueError("unknown name")
                if isinstance(node, ast.Call):
                    if not isinstance(node.func, ast.Name) or node.func.id not in allowed_names:
                        raise ValueError("unsafe call")
            code = compile(tree, "<graph>", "eval")
            return lambda x: float(eval(code, {{"__builtins__": {{}}}}, {{**allowed_names, "x": x}}))

        def make_label(text, params, color=WHITE):
            kind = str(params.get("content_type", "text")).lower()
            if kind == "math":
                return safe_math(text, int(params.get("font_size", 28)), color)
            return Text(str(text), font_size=int(params.get("font_size", 24)), color=color)

        def semantic_anchor(cue, obj):
            # Place annotations semantically relative to a stable object.
            p = cue.get("params") if isinstance(cue.get("params"), dict) else {{}}
            layout = cue.get("layout") if isinstance(cue.get("layout"), dict) else {{}}
            anchor_id = str(cue.get("anchor_to") or layout.get("anchor_to") or p.get("anchor_to") or "")
            relation = str(cue.get("relation") or p.get("relation") or "above").lower()
            anchor = objects.get(anchor_id)
            dirs = {{"above": UP, "below": DOWN, "left": LEFT, "right": RIGHT,
                    "upper_left": UL, "upper_right": UR, "lower_left": DL, "lower_right": DR}}
            if anchor is not None:
                obj.next_to(anchor, dirs.get(relation, UP), buff=float(p.get("buff", .16)))
                return obj
            region = str(layout.get("region") or p.get("region") or "center").lower()
            centers = {{"left": LEFT*3.0+DOWN*.25, "right": RIGHT*3.0+DOWN*.25,
                       "center": DOWN*.25, "top": UP*1.75, "bottom": DOWN*2.25}}
            obj.move_to(centers.get(region, diagram_center))
            off = layout.get("offset") or p.get("offset")
            if isinstance(off, (list, tuple)) and len(off) >= 2:
                obj.shift(RIGHT*float(off[0]) + UP*float(off[1]))
            return obj

        def make_object(cue):
            typ = str(cue.get("object_type", "")).lower()
            p = cue.get("params") if isinstance(cue.get("params"), dict) else {{}}
            # v7.2 allows these semantic fields at cue level.
            p = dict(p)
            if cue.get("content_type"): p["content_type"] = cue.get("content_type")
            if cue.get("content") is not None: p["text"] = cue.get("content")
            color = safe_color(p.get("color"), CYAN)
            label = cue.get("content") or cue.get("label") or p.get("text") or ""
            stroke = max(1.0, min(8.0, float(p.get("stroke_width", 3))))

            if typ in ("point", "marker", "circuit_node"):
                center = bounded_point(p.get("point") or p.get("position") or [-3,0,0])
                obj = Dot(center, radius=max(.04, min(.18, float(p.get("radius", .08)))), color=color)
                if label:
                    lab = make_label(label, p, WHITE).scale(0.75).next_to(obj, UP, buff=.10)
                    return VGroup(obj, lab)
                return obj

            if typ in ("line", "ray", "circuit_edge"):
                a = bounded_point(p.get("start") or [-4,0,0]); bb = bounded_point(p.get("end") or [-2,0,0])
                return Line(a, bb, color=color, stroke_width=stroke)

            if typ in ("arrow", "vector", "force_arrow", "dimension"):
                a = bounded_point(p.get("start") or [-4,0,0]); bb = bounded_point(p.get("end") or [-2,0,0])
                arr = Arrow(a, bb, buff=0, color=color, stroke_width=stroke)
                if label:
                    lab = make_label(label, p, color).scale(.75).next_to(arr, UP, buff=.08)
                    return VGroup(arr, lab)
                return arr

            if typ in ("polygon", "region"):
                pts = p.get("points") or []
                pts = [bounded_point(x) for x in pts if isinstance(x, (list, tuple))]
                if len(pts) >= 3:
                    poly = Polygon(*pts, color=color, stroke_width=stroke)
                    if typ == "region" or p.get("fill_opacity"):
                        poly.set_fill(color, opacity=max(0, min(.45, float(p.get("fill_opacity", .12)))))
                    return poly

            if typ == "circle":
                center = bounded_point(p.get("center") or [-3,0,0])
                radius = max(.15, min(2.2, float(p.get("radius", 1.0))))
                return Circle(radius=radius, color=color, stroke_width=stroke).move_to(center)

            if typ == "arc":
                center = bounded_point(p.get("center") or [-3,0,0])
                radius = max(.15, min(2.0, float(p.get("radius", .7))))
                sa = float(p.get("start_angle", 0)); ang = float(p.get("angle", math.pi/2))
                return Arc(radius=radius, start_angle=sa, angle=ang, color=color).move_arc_center_to(center)

            if typ == "angle":
                a = bounded_point(p.get("a") or [-4,0,0]); v = bounded_point(p.get("vertex") or [-3,0,0]); bb = bounded_point(p.get("b") or [-3,1,0])
                return Angle(Line(v, a), Line(v, bb), radius=max(.18, min(.8, float(p.get("radius", .4)))), color=color)

            if typ in ("axes", "grid"):
                xr = p.get("x_range") or [-4,4,1]; yr = p.get("y_range") or [-3,3,1]
                axes = Axes(x_range=xr, y_range=yr, x_length=5.2, y_length=3.8, tips=False, axis_config={{"color": MUTED}})
                axes.move_to(diagram_center)
                return axes

            if typ in ("graph", "function"):
                axes_id = str(p.get("axes_id") or "axes")
                axes = objects.get(axes_id)
                if axes is None or not isinstance(axes, Axes):
                    axes = Axes(x_range=[-4,4,1], y_range=[-3,3,1], x_length=5.2, y_length=3.8, tips=False).move_to(diagram_center)
                    objects[axes_id] = axes; self.add(axes)
                expr = p.get("expression") or p.get("function") or "x"
                try: fn = safe_expr(expr)
                except Exception: fn = lambda x: x
                xr = p.get("x_range") or [-4,4]
                return axes.plot(fn, x_range=[float(xr[0]), float(xr[1])], color=color)

            if typ == "number_line":
                xr = p.get("x_range") or [-5,5,1]
                return NumberLine(x_range=xr, length=5.4, include_numbers=True, color=MUTED).move_to(diagram_center)

            if typ in ("label", "highlight", "brace"):
                obj = make_label(label or "", p, color)
                return semantic_anchor(cue, obj)

            if typ == "table":
                rows = p.get("rows") or [[""]]
                rows = [[str(x) for x in row] for row in rows[:6] if isinstance(row, list)]
                if rows:
                    return Table(rows, include_outer_lines=True).scale(.45).move_to(diagram_center)
            return None

        def run_actions(cues, beat_duration):
            # Execute v7.2 storyboard actions in strict order, not as one cue pile.
            ordered = [c for c in cues[:24] if isinstance(c, dict)]
            ordered.sort(key=lambda c: int(c.get("order", 0) or 0))
            if not ordered: return
            per = max(.18, min(.75, (beat_duration * .52) / max(1, len(ordered)))) if beat_duration > 0 else .35
            for cue in ordered:
                action = str(cue.get("action", "show")).lower()
                oid = str(cue.get("object_id", "")).strip()[:80]
                if not oid: continue
                old = objects.get(oid)

                if action in ("hide", "remove"):
                    if old is not None:
                        self.play(FadeOut(old), run_time=per); objects.pop(oid, None)
                    continue
                if action == "highlight":
                    if old is not None: self.play(Indicate(old, color=GOLD, scale_factor=1.06), run_time=per)
                    continue
                if action == "dim":
                    if old is not None: self.play(old.animate.set_opacity(.28), run_time=per)
                    continue
                if action == "move":
                    if old is not None:
                        target = old.copy(); semantic_anchor(cue, target)
                        self.play(old.animate.move_to(target.get_center()), run_time=per)
                    continue

                new = make_object(cue)
                if new is None: continue
                if action == "transform" and old is not None:
                    self.play(ReplacementTransform(old, new), run_time=per); objects[oid] = new
                elif old is None:
                    objects[oid] = new
                    if action == "write" or isinstance(new, (Text, MathTex)):
                        self.play(Write(new), run_time=per)
                    elif action == "draw" or isinstance(new, (Line, Polygon, Circle, Arc, Axes, NumberLine)):
                        self.play(Create(new), run_time=per)
                    else:
                        self.play(FadeIn(new), run_time=per)
                elif action in ("create", "draw", "write", "show"):
                    # Existing stable ID means the storyboard is referring to retained state.
                    if old.get_opacity() < .95:
                        self.play(old.animate.set_opacity(1.0), run_time=per)
        for beat in BEATS:
            step_start = self.time
            audio_path = beat.get("audio_path")
            audio_duration = max(0.0, float(beat.get("audio_duration") or 0.0))
            if audio_path:
                self.add_sound(audio_path)

            # Explicit planner removals happen at the beginning of the beat.
            rm_anims = []
            for oid in beat.get("remove", []) or []:
                obj = objects.get(str(oid))
                if obj is not None:
                    rm_anims.append(FadeOut(obj))
                    objects.pop(str(oid), None)
            if rm_anims:
                self.play(*rm_anims, run_time=min(.45, max(.2, audio_duration*.12)))

            tex = str(beat.get("math_latex") or "").strip()
            state = str(beat.get("equation_state") or "hold").lower()
            if state == "remove" and current_eq is not None:
                self.play(FadeOut(current_eq), run_time=min(.55, max(.25, audio_duration*.12)))
                current_eq = None
            if tex and state not in ("hold", "remove"):
                new_eq = safe_math(tex, 38, WHITE)
                # Scale down only; never enlarge a tiny formula.
                if new_eq.width > 5.25: new_eq.scale_to_fit_width(5.25)
                if new_eq.height > 2.15: new_eq.scale_to_fit_height(2.15)
                new_eq.move_to(eq_anchor)
                if state == "final":
                    new_eq.set_color(GREEN)
                if current_eq is None:
                    self.play(Write(new_eq), run_time=min(.9, max(.3, audio_duration*.20)))
                else:
                    self.play(TransformMatchingTex(current_eq, new_eq), run_time=min(1.0, max(.3, audio_duration*.22)))
                current_eq = new_eq

            run_actions(beat.get("actions") or beat.get("diagram_cues") or [], audio_duration)

            for oid in beat.get("focus", []) or []:
                obj = objects.get(str(oid))
                if obj is not None:
                    self.play(Indicate(obj, color=GOLD, scale_factor=1.04), run_time=min(.55, max(.25, audio_duration*.10)))

            elapsed = self.time - step_start
            if audio_duration > elapsed:
                self.wait(audio_duration - elapsed)
            elif audio_duration <= 0:
                self.wait(.35)

        self.wait(.25)
"""


def render_manim_script(
    script_path: Path,
    job_id: str,
) -> Path:
    """Run Manim with a non-blocking selectors watchdog."""
    import selectors
    import time

    print(f"Starting Manim render: {job_id}", flush=True)
    job_media_dir = MEDIA_DIR / job_id
    job_media_dir.mkdir(parents=True, exist_ok=True)

    manim_cmd = [
        "manim", "-ql", "--disable_caching",
        "--media_dir", str(job_media_dir),
        str(script_path), "GeneratedScene",
    ]
    print("Running:", " ".join(manim_cmd), flush=True)

    process = subprocess.Popen(
        manim_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    selector = selectors.DefaultSelector()
    if process.stdout:
        selector.register(process.stdout, selectors.EVENT_READ)

    output_lines = []
    started_at = time.monotonic()
    last_output_at = started_at

    try:
        while process.poll() is None:
            now = time.monotonic()
            if now - started_at > 420:
                process.kill()
                raise RuntimeError("Manim exceeded the 420-second render timeout.")
            if now - last_output_at > 120:
                process.kill()
                tail = "\n".join(output_lines[-80:])
                raise RuntimeError("Manim produced no output for 120 seconds. Last output:\n" + tail)

            events = selector.select(timeout=0.5)
            for key, _ in events:
                line = key.fileobj.readline()
                if line:
                    clean = line.rstrip()
                    output_lines.append(clean)
                    print("[MANIM]", clean, flush=True)
                    last_output_at = time.monotonic()

        # Drain remaining buffered output.
        if process.stdout:
            for line in process.stdout:
                clean = line.rstrip()
                if clean:
                    output_lines.append(clean)
                    print("[MANIM]", clean, flush=True)
    finally:
        try:
            selector.close()
        except Exception:
            pass
        if process.stdout:
            process.stdout.close()

    if process.returncode != 0:
        raise RuntimeError("Manim rendering failed.\n\n" + ("\n".join(output_lines) or "No Manim output was captured."))

    video_files = list(job_media_dir.rglob("GeneratedScene.mp4")) or list(job_media_dir.rglob("*.mp4"))
    if not video_files:
        raise RuntimeError("Manim finished successfully but no MP4 file was found.")

    source_video = max(video_files, key=lambda p: p.stat().st_mtime)
    final_video = OUTPUT_DIR / f"{job_id}.mp4"
    shutil.copy2(source_video, final_video)
    print(f"Video created: {final_video}", flush=True)
    return final_video

# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Tezla Animator Rendering Engine",
        "version": APP_VERSION,
        "renderer": "Manim",
        "tts": "Edge TTS",
        "voice": DEFAULT_VOICE,
        "sympy": False,
        "teaching_timeline": True,
        "visual_actions": True,
        "audio_synchronization": True,
        "diagrams": True,
        "latex_normalization": True,
        "scene_metadata": True,
        "equation_state_transitions": True,
        "canonical_lovable_narration": True,
        "tezla_visual_theme": True,
        "adaptive_layout": True,
        "captions_in_player_only": True,
        "no_upscale_short_math": True,
        "narration_overlap_guard": True,
    }


# ============================================================
# ENGINE TEST
# ============================================================

@app.get("/engine-test")
def engine_test():

    results = {}

    # MANIM
    try:

        result = subprocess.run(
            [
                "manim",
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

        results["manim"] = {
            "available": (
                result.returncode == 0
            ),
            "version": (
                result.stdout.strip()
                or
                result.stderr.strip()
            ),
        }

    except Exception as exc:

        results["manim"] = {
            "available": False,
            "error": str(exc),
        }

    # EDGE TTS
    try:

        import edge_tts

        results["edge_tts"] = {
            "available": True,
            "version": getattr(
                edge_tts,
                "__version__",
                "installed",
            ),
            "voice": DEFAULT_VOICE,
        }

    except Exception as exc:

        results["edge_tts"] = {
            "available": False,
            "error": str(exc),
        }

    # FFPROBE
    try:

        result = subprocess.run(
            [
                "ffprobe",
                "-version",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

        results["ffprobe"] = {
            "available": (
                result.returncode == 0
            ),
            "version": (
                result.stdout.splitlines()[0]
                if result.stdout
                else "installed"
            ),
        }

    except Exception as exc:

        results["ffprobe"] = {
            "available": False,
            "error": str(exc),
        }

    return {
        "status": "ok",
        "version": APP_VERSION,
        "results": results,
    }


# ============================================================
# TTS TEST
# ============================================================

@app.get("/tts-test")
def tts_test():
    """
    Useful diagnostic endpoint.

    Confirms that Render can actually contact Edge TTS,
    generate an MP3, and determine its duration.
    """

    test_id = str(
        uuid.uuid4()
    )

    test_dir = (
        VOICE_DIR /
        "tests"
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_file = (
        test_dir /
        f"{test_id}.mp3"
    )

    test_text = (
        "Welcome to Tezla Animator. "
        "This is a test of the mathematics "
        "lesson narration system."
    )

    try:

        generate_edge_tts_sync(
            test_text,
            str(audio_file),
            DEFAULT_VOICE,
        )

        if not audio_file.exists():

            raise RuntimeError(
                "Edge TTS completed but no "
                "audio file was created."
            )

        duration = get_audio_duration(
            str(audio_file)
        )

        size = (
            audio_file.stat().st_size
        )

        return {
            "status": "success",
            "voice": DEFAULT_VOICE,
            "duration_seconds": duration,
            "size_bytes": size,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        try:

            if audio_file.exists():
                audio_file.unlink()

        except Exception:
            pass




# ============================================================
# V7.1 LOW-MEMORY CHUNKED RENDERING
# ============================================================

V7_CHUNK_BEATS = max(1, int(os.getenv("TEZLA_MANIM_CHUNK_BEATS", "3")))

def _v7_chunk_steps(steps: List[SolutionStep], chunk_size: int = V7_CHUNK_BEATS):
    return [steps[i:i + chunk_size] for i in range(0, len(steps), chunk_size)]

def _v7_plan_for_segments(visual_plan: Optional[dict], segment_ids: set) -> Optional[dict]:
    if not isinstance(visual_plan, dict):
        return visual_plan
    out = dict(visual_plan)
    out["beats"] = [
        b for b in (visual_plan.get("beats") or [])
        if isinstance(b, dict) and b.get("segment_index") in segment_ids
    ]
    return out

def _v7_reconstruct_prefix(all_steps: List[SolutionStep], chunk_start: int, visual_plan: Optional[dict]):
    """Rebuild the exact live v7.2 stage required at a chunk boundary."""
    merged = _merge_v7_visual_plan(all_steps, visual_plan)
    if chunk_start <= 0 or chunk_start >= len(merged):
        return []

    desired = {str(x) for x in (merged[chunk_start].get("starting_state") or []) if str(x)}
    live = {}
    for beat in merged[:chunk_start]:
        # Explicit beat removals.
        for oid in beat.get("remove", []) or []:
            live.pop(str(oid), None)
        for cue in beat.get("actions", []) or beat.get("diagram_cues", []) or []:
            if not isinstance(cue, dict): continue
            oid = str(cue.get("object_id") or "").strip()
            if not oid: continue
            action = str(cue.get("action") or "create").lower()
            if action in ("hide", "remove"):
                live.pop(oid, None)
            elif action in ("create", "draw", "write", "show", "transform"):
                c = dict(cue); c["action"] = "show"; c["order"] = -1000 + len(live)
                live[oid] = c
            # highlight/dim/move do not replace the object's semantic definition.

    # starting_state is authoritative in 7.2. Older v7 payloads have no state list.
    if desired:
        live = {oid: cue for oid, cue in live.items() if oid in desired}
    return list(live.values())

def build_direct_manim_script_v7_chunk(
    prompt: str, all_steps: List[SolutionStep], chunk_steps: List[SolutionStep],
    chunk_start: int, job_id: str, visual_plan: Optional[dict] = None,
    theme: Optional[dict] = None,
) -> str:
    segs = {int(st.segment_index if st.segment_index is not None else (chunk_start+i)) for i,st in enumerate(chunk_steps)}
    plan = _v7_plan_for_segments(visual_plan, segs)
    # The canonical builder generates this chunk's TTS exactly once.
    code = build_direct_manim_script_v7(prompt, chunk_steps, job_id, plan, theme)
    import re as _re
    m = _re.search(r"BEATS = (.*?)\nTHEME =", code, flags=_re.S)
    if not m:
        raise RuntimeError("Could not locate v7 beat payload in generated chunk script.")
    merged = ast.literal_eval(m.group(1))
    if merged:
        prefix = _v7_reconstruct_prefix(all_steps, chunk_start, visual_plan)
        current_actions = merged[0].get("actions") or merged[0].get("diagram_cues") or []
        existing = {str(c.get("object_id")) for c in current_actions if isinstance(c, dict)}
        restored = [c for c in prefix if str(c.get("object_id")) not in existing]
        merged[0]["actions"] = restored + current_actions
        merged[0]["diagram_cues"] = merged[0]["actions"]
        if merged[0].get("math_latex") and str(merged[0].get("equation_state") or "hold").lower() == "hold":
            merged[0]["equation_state"] = "show"
    return _re.sub(r"BEATS = .*?\nTHEME =", "BEATS = " + repr(merged) + "\nTHEME =", code, count=1, flags=_re.S)

def concat_video_chunks(chunk_paths: List[Path], job_id: str) -> Path:
    if not chunk_paths:
        raise RuntimeError("No rendered chunks to concatenate.")
    if len(chunk_paths) == 1:
        final_video = OUTPUT_DIR / f"{job_id}.mp4"
        shutil.copy2(chunk_paths[0], final_video)
        return final_video
    concat_file = BASE_DIR / f"{job_id}_concat.txt"
    def esc(path: Path):
        return str(path.resolve()).replace("'", "'\\''")
    concat_file.write_text("".join(f"file '{esc(p)}'\n" for p in chunk_paths), encoding="utf-8")
    final_video = OUTPUT_DIR / f"{job_id}.mp4"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(final_video)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try: concat_file.unlink(missing_ok=True)
    except Exception: pass
    if result.returncode != 0 or not final_video.exists():
        raise RuntimeError("FFmpeg chunk concatenation failed.\n" + result.stdout[-5000:])
    return final_video

def render_v7_chunked(req: RenderRequest, job_id: str) -> Path:
    steps = list(req.solution_steps or [])
    chunks = _v7_chunk_steps(steps)
    print(f"V7.2 LOW-MEMORY STORYBOARD: {len(steps)} beats -> {len(chunks)} chunks of <= {V7_CHUNK_BEATS}", flush=True)
    rendered = []
    start = 0
    try:
        for ci, chunk in enumerate(chunks, start=1):
            chunk_job = f"{job_id}_c{ci:02d}"
            script = BASE_DIR / f"{chunk_job}.py"
            print(f"Rendering chunk {ci}/{len(chunks)}: beats {start}..{start+len(chunk)-1}", flush=True)
            code = build_direct_manim_script_v7_chunk(req.prompt, steps, chunk, start, chunk_job, req.visual_plan, req.theme)
            validate_manim_code(code)
            script.write_text(code, encoding="utf-8")
            rendered.append(render_manim_script(script, chunk_job))
            try: script.unlink(missing_ok=True)
            except Exception: pass
            # Delete Manim media for the completed subprocess before starting another.
            try: shutil.rmtree(MEDIA_DIR / chunk_job, ignore_errors=True)
            except Exception: pass
            import gc
            gc.collect()
            start += len(chunk)
        return concat_video_chunks(rendered, job_id)
    finally:
        # Voice files are job/chunk-local and can be reclaimed after final assembly.
        for ci in range(1, len(chunks)+1):
            try: shutil.rmtree(VOICE_DIR / f"{job_id}_c{ci:02d}", ignore_errors=True)
            except Exception: pass


# ============================================================
# GENERATE VIDEO
# ============================================================

@app.post("/generate-video")
def generate_video(
    req: RenderRequest,
):

    job_id = str(
        uuid.uuid4()
    )

    print("=" * 70)
    print(
        f"NEW VIDEO JOB: {job_id}"
    )
    print("=" * 70)

    print("Prompt:")
    print(req.prompt)

    print(
        "Solution steps:",
        len(
            req.solution_steps or []
        ),
    )

    print(
        "Diagram:",
        req.diagram_spec,
    )

    if req.solution_steps:

        for step in req.solution_steps:

            print(
                f"STEP {step.step_number}"
            )

            print(
                "Math:",
                step.math_latex,
            )

            print(
                "Explanation:",
                step.explanation,
            )

            print(
                "Visual action:",
                step.visual_action,
            )

            print(
                "Scene / equation state:",
                step.scene,
                "/",
                step.equation_state,
            )

    script_path = (
        BASE_DIR /
        f"{job_id}.py"
    )

    provider_used = (
        "direct-solution-steps-v6"
    )

    try:

        # ====================================================
        # LOVABLE-PROVIDED TEACHING STEPS
        # ====================================================

        if req.solution_steps:

            print(
                "Using Lovable-provided "
                "teaching steps."
            )

            is_v7 = (req.version or 0) >= 7 or bool(req.visual_plan)
            if is_v7:
                provider_used = "direct-visual-storyboard-v7.2-chunked"
                code = None
            else:
                code = build_direct_manim_script(
                    prompt=req.prompt,
                    steps=req.solution_steps,
                    job_id=job_id,
                    diagram_spec=req.diagram_spec,
                )

        # ====================================================
        # AI FALLBACK
        # ====================================================

        else:

            print(
                "No solution steps received."
            )

            print(
                "Using AI Manim fallback."
            )

            (
                code,
                provider_used,
            ) = generate_ai_manim_code(
                req.prompt
            )

        # ====================================================
        # VALIDATE
        # ====================================================

        if req.solution_steps and ((req.version or 0) >= 7 or req.visual_plan):
            # Free-tier-safe path: each Manim subprocess renders only a few beats,
            # exits completely, then FFmpeg joins the chunks without re-encoding.
            final_video = render_v7_chunked(req, job_id)
        else:
            validate_manim_code(code)
            print("GENERATED MANIM SCRIPT:")
            print("-" * 70)
            print(code)
            print("-" * 70)
            script_path.write_text(code, encoding="utf-8")
            print(f"Script saved: {script_path}")
            final_video = render_manim_script(script_path, job_id)

        # ====================================================
        # RESPONSE
        # ====================================================

        video_url = (
            f"/videos/{final_video.name}"
        )

        print(
            f"VIDEO READY: {video_url}"
        )

        return {
            "status": "success",
            "job_id": job_id,
            "provider_used": provider_used,
            "video_url": video_url,
            "version": APP_VERSION,
        }

    except subprocess.TimeoutExpired:

        raise HTTPException(
            status_code=504,
            detail=(
                "Manim rendering timed out "
                "after 600 seconds."
            ),
        )

    except Exception as exc:

        print(
            f"JOB FAILED: {job_id}"
        )

        print(
            f"ERROR: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        try:

            if script_path.exists():
                script_path.unlink()

        except Exception as exc:

            print(
                "Could not delete temporary "
                f"script: {exc}"
            )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000",
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
