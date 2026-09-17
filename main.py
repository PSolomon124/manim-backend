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

APP_VERSION = "3.2.0"

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "rendered_videos"
MEDIA_DIR = BASE_DIR / "media"
VOICE_DIR = BASE_DIR / "voiceovers"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VOICE = "en-NG-EzinneNeural"


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


class RenderRequest(BaseModel):
    prompt: str
    solution_steps: Optional[List[SolutionStep]] = None

    # Example:
    #
    # {
    #     "kind": "parallelogram_to_triangles",
    #     "params": {}
    # }
    #
    diagram_spec: Optional[dict] = None


# ============================================================
# AI PROMPT
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

4. Use MathTex for mathematical expressions.

5. Never put $ symbols inside MathTex.

6. Use valid LaTeX with SINGLE backslashes.

7. Use proper spatial positioning:
   - to_edge()
   - next_to()
   - arrange()
   - align_to()
   - shift()
   - move_to()

8. Avoid overlapping equations, text and diagrams.

9. Keep diagrams and equations in separate visual zones.

10. Keep text readable.

11. Do not put LaTeX commands in spoken narration.

12. Return only executable Python code.
"""


# ============================================================
# TEXT CLEANING
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


def clean_spoken_text(text: str) -> str:
    text = str(text).strip()

    text = text.replace("$$", "")
    text = text.replace("$", "")

    latex_patterns = [
        r"\\boxed",
        r"\\quad",
        r"\\text",
        r"\\frac",
        r"\\sqrt",
        r"\\left",
        r"\\right",
        r"\\begin",
        r"\\end",
        r"\\times",
        r"\\cdot",
        r"\\pm",
        r"\\leq",
        r"\\geq",
        r"\\neq",
        r"\\approx",
        r"\\infty",
    ]

    for pattern in latex_patterns:
        text = re.sub(pattern, "", text)

    text = text.replace("\\", "")
    text = text.replace("{", "")
    text = text.replace("}", "")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_title(text: str) -> str:
    text = clean_spoken_text(text)

    if not text:
        return "Mathematics Solution"

    return text[:100]


def clean_latex(text: str) -> str:
    text = str(text).strip()

    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2]

    elif text.startswith("$") and text.endswith("$"):
        text = text[1:-1]

    # Keep LaTeX backslashes intact.
    text = re.sub(r"\\+", r"\\", text)

    latex_commands = [
        "boxed",
        "quad",
        "text",
        "frac",
        "sqrt",
        "left",
        "right",
        "begin",
        "end",
    ]

    for command in latex_commands:
        text = re.sub(
            rf"(?<!\\)\b{command}\b",
            rf"\\{command}",
            text,
        )

    return text.strip()


def escape_python_string(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
    )


# ============================================================
# EDGE TTS
# ============================================================

def generate_edge_tts_sync(
    text: str,
    output_file: str,
    voice: str = DEFAULT_VOICE,
):
    import edge_tts

    async def generate():
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
        )

        await communicate.save(output_file)

    asyncio.run(generate())


# ============================================================
# AUDIO DURATION
# ============================================================

def get_audio_duration(audio_file: str) -> float:

    if not os.path.exists(audio_file):
        return 4.0

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
                audio_file,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return max(
            float(result.stdout.strip()),
            1.0,
        )

    except Exception as exc:

        print(
            f"ffprobe duration error: {exc}"
        )

        return 4.0


# ============================================================
# DIAGRAM GENERATORS
# ============================================================

def add_diagram_to_script(
    script_lines: list,
    diagram_spec: Optional[dict],
):

    if not diagram_spec:
        return

    kind = diagram_spec.get("kind")

    # ========================================================
    # PARALLELOGRAM → TRIANGLES
    # ========================================================

    if kind == "parallelogram_to_triangles":

        script_lines.extend(
            [
                "        # ----------------------------",
                "        # PARALLELOGRAM DIAGRAM",
                "        # ----------------------------",
                "",
                "        p1 = LEFT * 4 + DOWN * 1.2",
                "        p2 = LEFT * 1 + DOWN * 1.2",
                "        p3 = RIGHT * 1.5 + UP * 1.0",
                "        p4 = LEFT * 1.5 + UP * 1.0",
                "",
                "        parallelogram = Polygon(",
                "            p1, p2, p3, p4,",
                "            color=TEAL_B,",
                "            stroke_width=3,",
                "        )",
                "",
                "        diagonal = Line(",
                "            p1, p3,",
                "            color=YELLOW,",
                "            stroke_width=2,",
                "        )",
                "",
                "        triangle = Polygon(",
                "            p1, p2, p3,",
                "            color=TEAL_C,",
                "            fill_color=TEAL_E,",
                "            fill_opacity=0.55,",
                "            stroke_width=3,",
                "        )",
                "",
                "        base_line = Line(",
                "            p1, p2,",
                "            color=GRAY_A,",
                "            stroke_width=2,",
                "        )",
                "",
                "        base_brace = Brace(",
                "            base_line,",
                "            DOWN,",
                "            buff=0.1,",
                "            color=GRAY_A,",
                "        )",
                "",
                '        base_label = MathTex("b", font_size=28)',
                "        base_label.next_to(",
                "            base_brace,",
                "            DOWN,",
                "            buff=0.08,",
                "        )",
                "",
                "        h_start = np.array([",
                "            p3[0],",
                "            p1[1],",
                "            0,",
                "        ])",
                "",
                "        height_line = DashedLine(",
                "            p3,",
                "            h_start,",
                "            color=GRAY_A,",
                "            stroke_width=2,",
                "        )",
                "",
                '        height_label = MathTex("h", font_size=28)',
                "        height_label.next_to(",
                "            height_line,",
                "            RIGHT,",
                "            buff=0.1,",
                "        )",
                "",
                "        self.play(",
                "            Create(parallelogram),",
                "            Create(base_brace),",
                "            Write(base_label),",
                "            Create(height_line),",
                "            Write(height_label),",
                "            run_time=1.2,",
                "        )",
                "",
                "        self.wait(0.5)",
                "",
                "        self.play(",
                "            Create(diagonal),",
                "            FadeIn(triangle),",
                "            run_time=1.0,",
                "        )",
                "",
                "        self.wait(0.8)",
                "",
            ]
        )

    # ========================================================
    # TRIANGLE
    # ========================================================

    elif kind == "triangle":

        script_lines.extend(
            [
                "        triangle = Polygon(",
                "            LEFT * 3 + DOWN * 1.5,",
                "            RIGHT * 3 + DOWN * 1.5,",
                "            UP * 2,",
                "            color=TEAL_B,",
                "            fill_color=TEAL_E,",
                "            fill_opacity=0.5,",
                "        )",
                "",
                "        self.play(",
                "            Create(triangle),",
                "            run_time=1.0,",
                "        )",
                "",
            ]
        )

    # ========================================================
    # CIRCLE
    # ========================================================

    elif kind == "circle":

        script_lines.extend(
            [
                "        circle = Circle(",
                "            radius=2,",
                "            color=TEAL_B,",
                "            fill_color=TEAL_E,",
                "            fill_opacity=0.3,",
                "        )",
                "",
                "        self.play(",
                "            Create(circle),",
                "            run_time=1.0,",
                "        )",
                "",
            ]
        )

    # ========================================================
    # NUMBER LINE
    # ========================================================

    elif kind == "number_line":

        script_lines.extend(
            [
                "        number_line = NumberLine(",
                "            x_range=[-5, 5, 1],",
                "            length=9,",
                "            include_numbers=True,",
                "        )",
                "",
                "        number_line.to_edge(DOWN, buff=1.5)",
                "",
                "        self.play(",
                "            Create(number_line),",
                "            run_time=1.0,",
                "        )",
                "",
            ]
        )

    # ========================================================
    # COORDINATE PLANE
    # ========================================================

    elif kind == "coordinate_plane":

        script_lines.extend(
            [
                "        plane = NumberPlane(",
                "            x_range=[-6, 6, 1],",
                "            y_range=[-4, 4, 1],",
                "            background_line_style={",
                '                "stroke_opacity": 0.3,',
                "            },",
                "        )",
                "",
                "        self.play(",
                "            Create(plane),",
                "            run_time=1.0,",
                "        )",
                "",
            ]
        )

    else:

        print(
            f"Unknown diagram type: {kind}"
        )


# ============================================================
# BUILD DIRECT MANIM SCRIPT
# ============================================================

def build_direct_manim_script(
    prompt: str,
    steps: List[SolutionStep],
    job_id: str,
    diagram_spec: Optional[dict] = None,
) -> str:

    voice_dir = VOICE_DIR / job_id

    voice_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    title = clean_title(prompt)

    safe_title = escape_python_string(title)

    script_lines = []

    # ========================================================
    # IMPORTS
    # ========================================================

    script_lines.extend(
        [
            "from manim import *",
            "import edge_tts",
            "import asyncio",
            "",
            "",
        ]
    )

    # ========================================================
    # TTS FUNCTION
    # ========================================================

    script_lines.extend(
        [
            "def generate_voice(text, output_file):",
            "    async def _generate():",
            f"        communicate = edge_tts.Communicate(",
            f"            text=text,",
            f"            voice={DEFAULT_VOICE!r},",
            "        )",
            "        await communicate.save(output_file)",
            "",
            "    asyncio.run(_generate())",
            "",
            "",
        ]
    )

    # ========================================================
    # SCENE
    # ========================================================

    script_lines.extend(
        [
            "class GeneratedScene(Scene):",
            "    def construct(self):",
            '        self.camera.background_color = "#0b1220"',
            "",
        ]
    )

    # ========================================================
    # HEADER
    # ========================================================

    script_lines.extend(
        [
            f'        title = Text("{safe_title}", font_size=32)',
            "        title.to_edge(UP, buff=0.35)",
            "",
            "        step_banner = Text(",
            '            "",',
            "            font_size=20,",
            "            color=YELLOW,",
            "        )",
            "",
            "        step_banner.next_to(",
            "            title,",
            "            DOWN,",
            "            buff=0.15,",
            "        )",
            "",
            "        self.play(",
            "            Write(title),",
            "            run_time=0.8,",
            "        )",
            "",
        ]
    )

    # ========================================================
    # DIAGRAM
    # ========================================================

    add_diagram_to_script(
        script_lines,
        diagram_spec,
    )

    # ========================================================
    # EQUATION AREA
    # ========================================================

    script_lines.extend(
        [
            "        equation_area = VGroup()",
            "        equation_area.move_to(",
            "            RIGHT * 2.7 + DOWN * 0.2",
            "        )",
            "",
        ]
    )

    # ========================================================
    # SOLUTION STEPS
    # ========================================================

    script_lines.append(
        "        prev = None"
    )

    script_lines.append("")

    for index, step in enumerate(
        steps,
        start=1,
    ):

        explanation = clean_spoken_text(
            step.explanation
        )

        if not explanation:
            explanation = f"Step {index}"

        safe_explanation = escape_python_string(
            explanation
        )

        cleaned_latex = clean_latex(
            step.math_latex
        )

        safe_latex = escape_python_string(
            cleaned_latex
        )

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        audio_file = (
            voice_dir /
            f"step_{index}.mp3"
        )

        try:

            generate_edge_tts_sync(
                explanation,
                str(audio_file),
                DEFAULT_VOICE,
            )

        except Exception as exc:

            print(
                f"TTS failed for step {index}: {exc}"
            )

        duration = get_audio_duration(
            str(audio_file)
        )

        # ----------------------------------------------------
        # STEP BANNER
        # ----------------------------------------------------

        banner_text = (
            f"Step {index}: {explanation}"
        )

        safe_banner = escape_python_string(
            banner_text
        )

        script_lines.extend(
            [
                "",
                "        # ----------------------------",
                f"        # STEP {index}",
                "        # ----------------------------",
                "",
                f'        new_banner = Text("{safe_banner}", font_size=20, color=YELLOW)',
                "",
                "        new_banner.next_to(",
                "            title,",
                "            DOWN,",
                "            buff=0.15,",
                "        )",
                "",
                "        self.play(",
                "            FadeOut(step_banner),",
                "            FadeIn(new_banner),",
                "            run_time=0.4,",
                "        )",
                "",
                "        step_banner = new_banner",
                "",
            ]
        )

        # ----------------------------------------------------
        # EQUATION
        # ----------------------------------------------------

        if cleaned_latex:

            script_lines.extend(
                [
                    f'        tex = MathTex(r"{safe_latex}", font_size=54)',
                    "",
                    "        # Keep equation inside safe area.",
                    "        tex.scale_to_fit_width(",
                    "            min(",
                    "                tex.width,",
                    "                config.frame_width * 0.43,",
                    "            )",
                    "        )",
                    "",
                    "        tex.move_to(",
                    "            RIGHT * 2.7 + UP * 0.2",
                    "        )",
                    "",
                ]
            )

        else:

            script_lines.extend(
                [
                    '        tex = Text("No equation", font_size=36)',
                    "        tex.move_to(",
                    "            RIGHT * 2.7 + UP * 0.2",
                    "        )",
                    "",
                ]
            )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        script_lines.extend(
            [
                f'        explanation = Text("{safe_explanation}", font_size=22)',
                "",
                "        explanation.scale_to_fit_width(",
                "            min(",
                "                explanation.width,",
                "                config.frame_width * 0.43,",
                "            )",
                "        )",
                "",
                "        explanation.next_to(",
                "            tex,",
                "            DOWN,",
                "            buff=0.35,",
                "        )",
                "",
            ]
        )

        # ----------------------------------------------------
        # EQUATION TRANSITION
        # ----------------------------------------------------

        script_lines.extend(
            [
                "        if prev is None:",
                "            self.play(",
                "                Write(tex),",
                "                run_time=1.0,",
                "            )",
                "        else:",
                "            self.play(",
                "                TransformMatchingTex(",
                "                    prev,",
                "                    tex,",
                "                ),",
                "                run_time=1.0,",
                "            )",
                "",
            ]
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        script_lines.extend(
            [
                "        self.play(",
                "            FadeIn(explanation),",
                "            run_time=0.5,",
                "        )",
                "",
            ]
        )

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        if audio_file.exists():

            safe_audio = (
                str(audio_file)
                .replace("\\", "/")
                .replace('"', '\\"')
            )

            script_lines.append(
                f'        self.add_sound("{safe_audio}")'
            )

            script_lines.append("")

        # ----------------------------------------------------
        # WAIT FOR NARRATION
        # ----------------------------------------------------

        script_lines.append(
            f"        self.wait({max(duration, 1.0):.2f})"
        )

        script_lines.append("")

        # ----------------------------------------------------
        # REMOVE EXPLANATION ONLY
        # ----------------------------------------------------

        script_lines.extend(
            [
                "        self.play(",
                "            FadeOut(explanation),",
                "            run_time=0.4,",
                "        )",
                "",
                "        prev = tex",
                "",
            ]
        )

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    script_lines.extend(
        [
            "        if prev is not None:",
            "",
            "            self.play(",
            "                prev.animate.set_color(YELLOW),",
            "                run_time=0.6,",
            "            )",
            "",
            '            final_label = Text("Final Answer", font_size=28, color=YELLOW)',
            "            final_label.next_to(",
            "                prev,",
            "                UP,",
            "                buff=0.3,",
            "            )",
            "",
            "            self.play(",
            "                FadeIn(final_label),",
            "            )",
            "",
            "            self.wait(1.5)",
        ]
    )

    return "\n".join(script_lines)


# ============================================================
# GEMINI
# ============================================================

def generate_with_gemini(prompt: str) -> str:

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

def generate_with_groq(prompt: str) -> str:

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

    content = response.choices[0].message.content

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

    content = response.choices[0].message.content

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

def generate_ai_manim_code(prompt: str):

    providers = []

    if os.getenv("GEMINI_API_KEY"):

        providers.append(
            (
                "gemini",
                lambda: generate_with_gemini(
                    prompt
                ),
            )
        )

    if os.getenv("GROQ_API_KEY"):

        providers.append(
            (
                "groq",
                lambda: generate_with_groq(
                    prompt
                ),
            )
        )

    if os.getenv("OPENROUTER_API_KEY"):

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

    for provider_name, provider_function in providers:

        try:

            print(
                f"Trying provider: {provider_name}"
            )

            code = provider_function()

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
                f"{provider_name}: invalid code"
            )

        except Exception as exc:

            print(
                f"{provider_name} failed: {exc}"
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

def validate_manim_code(code: str):

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
                f"Forbidden dependency: {item}"
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
# RENDER
# ============================================================

def render_manim_script(
    script_path: Path,
    job_id: str,
) -> Path:

    print(
        f"Starting render: {job_id}"
    )

    command = [
        "manim",
        "-ql",
        "--disable_caching",
        "--media_dir",
        str(MEDIA_DIR),
        str(script_path),
        "GeneratedScene",
    ]

    print(
        "Running:",
        " ".join(command),
    )

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600,
    )

    print(
        "MANIM STDOUT:"
    )

    print(
        process.stdout
    )

    print(
        "MANIM STDERR:"
    )

    print(
        process.stderr
    )

    if process.returncode != 0:

        raise RuntimeError(
            "Manim rendering failed.\n\n"
            + process.stdout
            + "\n\n"
            + process.stderr
        )

    # --------------------------------------------------------
    # FIND OUTPUT VIDEO
    # --------------------------------------------------------

    video_files = list(
        MEDIA_DIR.rglob(
            "GeneratedScene.mp4"
        )
    )

    if not video_files:

        video_files = list(
            MEDIA_DIR.rglob(
                "*.mp4"
            )
        )

    if not video_files:

        raise RuntimeError(
            "No MP4 was produced by Manim."
        )

    source_video = video_files[-1]

    final_video = (
        OUTPUT_DIR /
        f"{job_id}.mp4"
    )

    shutil.copy2(
        source_video,
        final_video,
    )

    print(
        f"Final video: {final_video}"
    )

    return final_video


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": (
            "Tezla Animator "
            "Rendering Engine"
        ),
        "version": APP_VERSION,
        "renderer": "Manim",
        "tts": "Edge TTS",
        "sympy": False,
        "diagram_support": True,
        "layout_engine": "spatial-zones",
    }


# ============================================================
# ENGINE TEST
# ============================================================

@app.get("/engine-test")
def engine_test():

    results = {}

    # --------------------------------------------------------
    # MANIM
    # --------------------------------------------------------

    try:

        result = subprocess.run(
            ["manim", "--version"],
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
                or result.stderr.strip()
            ),
        }

    except Exception as exc:

        results["manim"] = {
            "available": False,
            "error": str(exc),
        }

    # --------------------------------------------------------
    # EDGE TTS
    # --------------------------------------------------------

    try:

        import edge_tts

        results["edge_tts"] = {
            "available": True,
            "version": getattr(
                edge_tts,
                "__version__",
                "installed",
            ),
        }

    except Exception as exc:

        results["edge_tts"] = {
            "available": False,
            "error": str(exc),
        }

    # --------------------------------------------------------
    # FFPROBE
    # --------------------------------------------------------

    try:

        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=20,
        )

        results["ffprobe"] = {
            "available": (
                result.returncode == 0
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

    print(
        "Prompt:",
        req.prompt,
    )

    print(
        "Number of solution steps:",
        len(
            req.solution_steps or []
        ),
    )

    print(
        "Diagram specification:",
        req.diagram_spec,
    )

    script_path = (
        BASE_DIR /
        f"{job_id}.py"
    )

    provider_used = (
        "direct-solution-steps"
    )

    try:

        # ====================================================
        # DIRECT LOVABLE SOLUTION
        # ====================================================

        if req.solution_steps:

            print(
                "Using solution_steps "
                "from Lovable."
            )

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
                "No solution_steps supplied."
            )

            print(
                "Using AI Manim fallback."
            )

            code, provider_used = (
                generate_ai_manim_code(
                    req.prompt
                )
            )

        # ====================================================
        # VALIDATE
        # ====================================================

        validate_manim_code(
            code
        )

        # ====================================================
        # WRITE
        # ====================================================

        script_path.write_text(
            code,
            encoding="utf-8",
        )

        print(
            f"Generated script: "
            f"{script_path}"
        )

        # ====================================================
        # RENDER
        # ====================================================

        final_video = render_manim_script(
            script_path,
            job_id,
        )

        video_url = (
            f"/videos/{final_video.name}"
        )

        print(
            f"SUCCESS: {video_url}"
        )

        return {
            "status": "success",
            "job_id": job_id,
            "provider_used": provider_used,
            "video_url": video_url,
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
                f"Could not delete temp script: {exc}"
            )


# ============================================================
# LOCAL SERVER
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
