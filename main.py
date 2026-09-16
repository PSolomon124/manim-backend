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

APP_VERSION = "3.1.0"

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
    # {
    #   "kind": "parallelogram_to_triangles",
    #   "params": {}
    # }
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

2. Define exactly one scene:
   class GeneratedScene(Scene):

3. Do not use:
   - manim_voiceover
   - VoiceoverScene
   - EdgeTTSService

4. The script must be executable directly by Manim.

5. Use MathTex for mathematical expressions.

6. Do NOT put $ symbols inside MathTex.

7. Use valid LaTeX with SINGLE backslashes.

8. Common commands:
   \boxed{}
   \quad
   \text{}
   \frac{}{}
   \sqrt{}
   \left
   \right

9. Do not put LaTeX inside spoken narration.

10. Keep animations educational and readable.

11. Define exactly:
       class GeneratedScene(Scene):

12. Return only valid Python code.
"""


# ============================================================
# TEXT / LATEX CLEANING
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

    return text[:120]


def clean_latex(text: str) -> str:
    text = str(text).strip()

    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2]

    elif text.startswith("$") and text.endswith("$"):
        text = text[1:-1]

    # Convert repeated backslashes to a single backslash.
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


def escape_latex(text: str) -> str:
    # Do NOT escape LaTeX backslashes.
    return text.replace('"', '\\"')


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

        duration = float(result.stdout.strip())

        return max(duration, 1.0)

    except Exception as exc:
        print(
            f"Could not determine audio duration: {exc}"
        )

        return 4.0


# ============================================================
# DIAGRAM GENERATOR
# ============================================================

def add_diagram_to_script(
    script_lines: list,
    diagram_spec: Optional[dict],
):
    """
    Adds diagram code based on diagram_spec.
    """

    if not diagram_spec:
        return

    kind = diagram_spec.get("kind")

    # --------------------------------------------------------
    # PARALLELOGRAM TO TRIANGLES
    # --------------------------------------------------------

    if kind == "parallelogram_to_triangles":

        script_lines.append(
            "        # ========================================"
        )

        script_lines.append(
            "        # PARALLELOGRAM DIAGRAM"
        )

        script_lines.append(
            "        # ========================================"
        )

        script_lines.append(
            "        p = Polygon("
            "[-3, -1, 0], "
            "[1, -1, 0], "
            "[3, 1, 0], "
            "[-1, 1, 0], "
            "color=TEAL"
            ")"
        )

        script_lines.append(
            "        self.play(Create(p))"
        )

        script_lines.append(
            "        diagonal = Line("
            "[-3, -1, 0], "
            "[3, 1, 0], "
            "color=YELLOW"
            ")"
        )

        script_lines.append(
            "        self.play(Create(diagonal))"
        )

        script_lines.append(
            "        triangle = Polygon("
            "[-3, -1, 0], "
            "[1, -1, 0], "
            "[3, 1, 0], "
            "fill_opacity=0.35, "
            "color=TEAL"
            ")"
        )

        script_lines.append(
            "        self.play(FadeIn(triangle))"
        )

        script_lines.append(
            "        self.wait(1)"
        )

        script_lines.append(
            "        self.play("
            "FadeOut(p), "
            "FadeOut(diagonal), "
            "triangle.animate.move_to(ORIGIN)"
            ")"
        )

        script_lines.append("")

    # --------------------------------------------------------
    # TRIANGLE
    # --------------------------------------------------------

    elif kind == "triangle":

        script_lines.append(
            "        triangle = Polygon("
            "[-3, -2, 0], "
            "[3, -2, 0], "
            "[0, 2, 0], "
            "color=TEAL, "
            "fill_opacity=0.35"
            ")"
        )

        script_lines.append(
            "        self.play(Create(triangle))"
        )

        script_lines.append(
            "        self.wait(1)"
        )

    # --------------------------------------------------------
    # CIRCLE
    # --------------------------------------------------------

    elif kind == "circle":

        script_lines.append(
            "        circle = Circle("
            "radius=2, "
            "color=TEAL, "
            "fill_opacity=0.25"
            ")"
        )

        script_lines.append(
            "        self.play(Create(circle))"
        )

        script_lines.append(
            "        self.wait(1)"
        )

    # --------------------------------------------------------
    # NUMBER LINE
    # --------------------------------------------------------

    elif kind == "number_line":

        script_lines.append(
            "        number_line = NumberLine("
            "x_range=[-5, 5, 1], "
            "length=10, "
            "include_numbers=True"
            ")"
        )

        script_lines.append(
            "        self.play(Create(number_line))"
        )

        script_lines.append(
            "        self.wait(1)"
        )

    # --------------------------------------------------------
    # COORDINATE PLANE
    # --------------------------------------------------------

    elif kind == "coordinate_plane":

        script_lines.append(
            "        plane = NumberPlane("
            "x_range=[-6, 6, 1], "
            "y_range=[-4, 4, 1], "
            "background_line_style={"
            '"stroke_opacity": 0.35'
            "}"
            ")"
        )

        script_lines.append(
            "        self.play(Create(plane))"
        )

        script_lines.append(
            "        self.wait(1)"
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

    safe_title = title.replace(
        '"',
        '\\"',
    )

    script_lines = []

    # --------------------------------------------------------
    # IMPORTS
    # --------------------------------------------------------

    script_lines.append(
        "from manim import *"
    )

    script_lines.append(
        "import edge_tts"
    )

    script_lines.append(
        "import asyncio"
    )

    script_lines.append("")

    # --------------------------------------------------------
    # TTS FUNCTION
    # --------------------------------------------------------

    script_lines.append(
        "def generate_voice(text, output_file):"
    )

    script_lines.append(
        "    async def _generate():"
    )

    script_lines.append(
        f"        communicate = edge_tts.Communicate("
        f"text=text, "
        f"voice={DEFAULT_VOICE!r}"
        f")"
    )

    script_lines.append(
        "        await communicate.save(output_file)"
    )

    script_lines.append("")

    script_lines.append(
        "    asyncio.run(_generate())"
    )

    script_lines.append("")

    # --------------------------------------------------------
    # SCENE
    # --------------------------------------------------------

    script_lines.append(
        "class GeneratedScene(Scene):"
    )

    script_lines.append(
        "    def construct(self):"
    )

    script_lines.append(
        '        self.camera.background_color = "#0b1220"'
    )

    script_lines.append("")

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    script_lines.append(
        f'        title = Text("{safe_title}", font_size=36)'
    )

    script_lines.append(
        "        self.play(Write(title))"
    )

    script_lines.append(
        "        self.wait(0.8)"
    )

    script_lines.append(
        "        self.play(FadeOut(title))"
    )

    script_lines.append("")

    # --------------------------------------------------------
    # OPTIONAL DIAGRAM
    # --------------------------------------------------------

    add_diagram_to_script(
        script_lines,
        diagram_spec,
    )

    # --------------------------------------------------------
    # SOLUTION STEPS
    # --------------------------------------------------------

    script_lines.append(
        "        prev = None"
    )

    script_lines.append("")

    for index, step in enumerate(
        steps,
        start=1,
    ):

        # ----------------------------------------------------
        # CLEAN EXPLANATION
        # ----------------------------------------------------

        explanation = clean_spoken_text(
            step.explanation
        )

        if not explanation:
            explanation = f"Step {index}"

        safe_explanation = explanation.replace(
            '"',
            '\\"',
        )

        # ----------------------------------------------------
        # CLEAN LATEX
        # ----------------------------------------------------

        cleaned_latex = clean_latex(
            step.math_latex
        )

        latex = escape_latex(
            cleaned_latex
        )

        # ----------------------------------------------------
        # CREATE VOICEOVER
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
                f"Edge TTS failed for step "
                f"{index}: {exc}"
            )

        duration = get_audio_duration(
            str(audio_file)
        )

        # ----------------------------------------------------
        # STEP LABEL
        # ----------------------------------------------------

        script_lines.append(
            f'        step_label = Text('
            f'"Step {index}", '
            f'font_size=28'
            f')'
        )

        script_lines.append(
            "        step_label.to_edge(UP)"
        )

        # ----------------------------------------------------
        # EQUATION
        # ----------------------------------------------------

        if latex:

            script_lines.append(
                f'        tex = MathTex('
                f'r"{latex}", '
                f'font_size=72'
                f')'
            )

            script_lines.append(
                "        tex.scale_to_fit_width("
                "min("
                "config.frame_width - 2, "
                "tex.width * 3"
                ")"
                ")"
            )

        else:

            script_lines.append(
                '        tex = Text('
                '"No equation provided", '
                'font_size=40'
                ')'
            )

        script_lines.append(
            "        tex.move_to(ORIGIN)"
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        script_lines.append(
            f'        explanation = Text('
            f'"{safe_explanation}", '
            f'font_size=24'
            f')'
        )

        script_lines.append(
            "        explanation.to_edge(DOWN)"
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        script_lines.append(
            "        self.play("
            "Write(step_label)"
            ")"
        )

        # ----------------------------------------------------
        # TRANSITION BETWEEN EQUATIONS
        # ----------------------------------------------------

        script_lines.append(
            "        if prev is None:"
        )

        script_lines.append(
            "            self.play("
            "Write(tex), "
            "run_time=1.2"
            ")"
        )

        script_lines.append(
            "        else:"
        )

        script_lines.append(
            "            self.play("
            "TransformMatchingTex("
            "prev, "
            "tex"
            "), "
            "run_time=1.2"
            ")"
        )

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        if audio_file.exists():

            safe_audio_path = str(
                audio_file
            ).replace(
                "\\",
                "/",
            )

            script_lines.append(
                f'        self.add_sound('
                f'"{safe_audio_path}"'
                f')'
            )

        # ----------------------------------------------------
        # EXPLANATION ON SCREEN
        # ----------------------------------------------------

        script_lines.append(
            "        self.play("
            "Write(explanation)"
            ")"
        )

        script_lines.append(
            f"        self.wait("
            f"{max(duration, 1.0):.2f}"
            f")"
        )

        # ----------------------------------------------------
        # CLEAN SCREEN
        # ----------------------------------------------------

        script_lines.append(
            "        self.play("
            "FadeOut(step_label), "
            "FadeOut(explanation)"
            ")"
        )

        # ----------------------------------------------------
        # SAVE PREVIOUS EQUATION
        # ----------------------------------------------------

        script_lines.append(
            "        prev = tex"
        )

        script_lines.append("")

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    script_lines.append(
        "        if prev is not None:"
    )

    script_lines.append(
        "            self.play("
        "prev.animate.set_color(YELLOW), "
        "run_time=0.6"
        ")"
    )

    script_lines.append(
        "            self.wait(1.2)"
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
                f"Trying AI provider: "
                f"{provider_name}"
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
                f"{provider_name}: "
                f"invalid generated code"
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
                f"Generated code contains "
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

    manim_cmd = [
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
        " ".join(manim_cmd),
    )

    process = subprocess.run(
        manim_cmd,
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
    # FIND MP4
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
            "Manim finished but no MP4 "
            "was found."
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
        f"Video created: {final_video}"
    )

    return final_video


# ============================================================
# ROOT / HEALTH
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
        "diagrams": True,
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
        "Solution steps:",
        len(
            req.solution_steps or []
        ),
    )

    print(
        "Diagram:",
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
        # LOVABLE SOLUTION STEPS
        # ====================================================

        if req.solution_steps:

            print(
                "Using Lovable-provided "
                "solution steps."
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
                "No solution steps received."
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

        final_video = render_manim_script(
            script_path,
            job_id,
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
