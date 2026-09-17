import os
import re
import uuid
import asyncio
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

APP_VERSION = "5.0.0"

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
    math_latex: str
    explanation: str

    # NEW V5 FIELDS
    visual_action: Optional[str] = None
    emphasis: Optional[List[str]] = None


class RenderRequest(BaseModel):
    prompt: str
    solution_steps: Optional[List[SolutionStep]] = None
    diagram_spec: Optional[dict] = None


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

            narration = (
                f"Now we continue with step {index}."
            )

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
            "            color=TEAL,",
            "            fill_opacity=0.10,",
            "            stroke_width=4,",
            "        )",
            "",
            "        base_line = Line(",
            "            p1, p2,",
            "            color=YELLOW,",
            "            stroke_width=6,",
            "        )",
            "",
            "        height_foot = np.array([p4[0], p1[1], 0])",
            "",
            "        height_line = DashedLine(",
            "            p4,",
            "            height_foot,",
            "            color=RED,",
            "            stroke_width=4,",
            "            dash_length=0.10,",
            "        )",
            "",
            "        diagonal = Line(",
            "            p1, p3,",
            "            color=YELLOW,",
            "            stroke_width=5,",
            "        )",
            "",
            "        triangle_one = Polygon(",
            "            p1, p2, p3,",
            "            color=BLUE,",
            "            fill_opacity=0.25,",
            "            stroke_opacity=0,",
            "        )",
            "",
            "        triangle_two = Polygon(",
            "            p1, p3, p4,",
            "            color=GREEN,",
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
            "            color=TEAL,",
            "            fill_opacity=0.14,",
            "            stroke_width=4,",
            "        )",
            "",
            "        triangle_base = Line(",
            "            ta, tb,",
            "            color=YELLOW,",
            "            stroke_width=6,",
            "        )",
            "",
            "        triangle_height_foot = np.array([tc[0], ta[1], 0])",
            "",
            "        triangle_height = DashedLine(",
            "            tc,",
            "            triangle_height_foot,",
            "            color=RED,",
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
            "            color=TEAL,",
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
            "            color=YELLOW,",
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


# ============================================================
# DIRECT TEACHING SCRIPT BUILDER
# ============================================================

def build_direct_manim_script(
    prompt: str,
    steps: List[SolutionStep],
    job_id: str,
    diagram_spec: Optional[dict] = None,
) -> str:
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

    title = clean_title(
        prompt
    )

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
        '        self.camera.background_color = "#0b1220"'
    )

    lines.append("")

    # ========================================================
    # HEADER
    # ========================================================

    lines.extend([
        "        title = Text(",
        f"            {python_literal(title)},",
        "            font_size=30,",
        "            weight=BOLD,",
        "        )",
        "",
        "        title.scale_to_fit_width(12.0)",
        "        title.to_edge(UP, buff=0.25)",
        "",
        "        divider = Line(",
        "            LEFT * 6.4,",
        "            RIGHT * 6.4,",
        "            stroke_opacity=0.25,",
        "        )",
        "",
        "        divider.next_to(",
        "            title,",
        "            DOWN,",
        "            buff=0.18,",
        "        )",
        "",
        "        self.play(",
        "            FadeIn(title),",
        "            Create(divider),",
        "            run_time=0.65,",
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
        "        equation_anchor = RIGHT * 3.15 + UP * 1.15",
        "        explanation_anchor = RIGHT * 3.15 + DOWN * 1.20",
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

        visual_action = (
            step.visual_action
            or ""
        )

        # Reserve most of narration for actual teaching.
        #
        # Do not let entrance animation consume entire speech.
        if narration_duration > 0:

            visual_budget = min(
                1.6,
                max(
                    0.7,
                    narration_duration * 0.30,
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
            "            RIGHT * 3.15 + UP * 2.55",
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
                "            font_size=48,",
                "        )",
                "",
                "        tex.scale_to_fit_width(5.4)",
                "        tex.move_to(equation_anchor)",
                "",
            ])

        else:

            lines.extend([
                "        tex = Text(",
                "            'Continue',",
                "            font_size=32,",
                "        )",
                "",
                "        tex.move_to(equation_anchor)",
                "",
            ])

        # ====================================================
        # DISPLAY EXPLANATION
        # ====================================================

        if display_explanation:

            lines.extend([
                "        explanation = Text(",
                f"            {python_literal(display_explanation)},",
                "            font_size=19,",
                "            line_spacing=0.85,",
                "        )",
                "",
                "        explanation.scale_to_fit_width(5.4)",
                "        explanation.move_to(explanation_anchor)",
                "",
            ])

        else:

            lines.extend([
                "        explanation = Text(",
                "            '',",
                "            font_size=19,",
                "        )",
                "",
                "        explanation.move_to(explanation_anchor)",
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
            0.75,
            max(
                0.35,
                visual_budget * 0.55,
            ),
        )

        if i == 1:

            lines.extend([
                "        self.play(",
                "            Write(tex),",
                f"            run_time={equation_runtime:.2f},",
                "        )",
                "",
            ])

        else:

            lines.extend([
                "        self.play(",
                "            FadeOut(previous_equation),",
                "            Write(tex),",
                f"            run_time={equation_runtime:.2f},",
                "        )",
                "",
            ])

        # ====================================================
        # EXPLANATION TRANSITION DURING SPEECH
        # ====================================================

        explanation_runtime = min(
            0.45,
            max(
                0.25,
                visual_budget * 0.35,
            ),
        )

        if i == 1:

            lines.extend([
                "        self.play(",
                "            FadeIn(explanation),",
                f"            run_time={explanation_runtime:.2f},",
                "        )",
                "",
            ])

        else:

            lines.extend([
                "        self.play(",
                "            FadeOut(previous_explanation),",
                "            FadeIn(explanation),",
                f"            run_time={explanation_runtime:.2f},",
                "        )",
                "",
            ])

        # ====================================================
        # ESTIMATE ELAPSED ANIMATION TIME
        # ====================================================

        label_time = (
            min(
                0.35 if i == 1 else 0.30,
                visual_budget,
            )
        )

        # add_visual_action may use roughly visual_budget
        diagram_time = (
            visual_budget
            if diagram_spec
            else 0.0
        )

        estimated_elapsed = (
            label_time
            + diagram_time
            + equation_runtime
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
                    f"        self.wait({remaining:.2f})",
                    "",
                ])

        else:

            # If TTS failed, leave enough time to read.
            lines.extend([
                "        self.wait(2.2)",
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

        if final_latex:

            lines.extend([
                "",
                "        # ========================================",
                "        # FINAL ANSWER HIGHLIGHT",
                "        # ========================================",
                "",
                "        final_box = SurroundingRectangle(",
                "            previous_equation,",
                "            color=YELLOW,",
                "            buff=0.25,",
                "            corner_radius=0.08,",
                "        )",
                "",
                "        final_text = Text(",
                "            'Final Answer',",
                "            font_size=25,",
                "            weight=BOLD,",
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
                "            run_time=0.65,",
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

def render_manim_script(
    script_path: Path,
    job_id: str,
) -> Path:

    print(
        f"Starting Manim render: {job_id}"
    )

    job_media_dir = (
        MEDIA_DIR /
        job_id
    )

    job_media_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manim_cmd = [
        "manim",
        "-ql",
        "--disable_caching",
        "--media_dir",
        str(job_media_dir),
        str(script_path),
        "GeneratedScene",
    ]

    print(
        "Running:",
        " ".join(manim_cmd),
    )

    process = subprocess.run(
        manim_cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    print("MANIM STDOUT:")
    print(process.stdout)

    print("MANIM STDERR:")
    print(process.stderr)

    if process.returncode != 0:

        raise RuntimeError(
            "Manim rendering failed.\n\n"
            + process.stdout
            + "\n\n"
            + process.stderr
        )

    video_files = list(
        job_media_dir.rglob(
            "GeneratedScene.mp4"
        )
    )

    if not video_files:

        video_files = list(
            job_media_dir.rglob(
                "*.mp4"
            )
        )

    if not video_files:

        raise RuntimeError(
            "Manim finished successfully but "
            "no MP4 file was found."
        )

    source_video = max(
        video_files,
        key=lambda p: p.stat().st_mtime,
    )

    final_video = (
        OUTPUT_DIR /
        f"{job_id}.mp4"
    )

    shutil.copy2(
        source_video,
        final_video,
    )

    print(
        f"Video created: {final_video}"
    )

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

    script_path = (
        BASE_DIR /
        f"{job_id}.py"
    )

    provider_used = (
        "direct-solution-steps-v5"
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

            code = (
                build_direct_manim_script(
                    prompt=req.prompt,
                    steps=req.solution_steps,
                    job_id=job_id,
                    diagram_spec=req.diagram_spec,
                )
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

        validate_manim_code(
            code
        )

        # ====================================================
        # LOG SCRIPT
        # ====================================================

        print(
            "GENERATED MANIM SCRIPT:"
        )

        print("-" * 70)
        print(code)
        print("-" * 70)

        # ====================================================
        # WRITE SCRIPT
        # ====================================================

        script_path.write_text(
            code,
            encoding="utf-8",
        )

        print(
            f"Script saved: {script_path}"
        )

        # ====================================================
        # RENDER
        # ====================================================

        final_video = (
            render_manim_script(
                script_path,
                job_id,
            )
        )

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
