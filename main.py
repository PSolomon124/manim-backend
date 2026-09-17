
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

APP_VERSION = "4.0.0"

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

8. Examples:
   \boxed{}
   \quad
   \text{}
   \frac{}{}
   \sqrt{}
   \left
   \right

9. Do not put LaTeX inside spoken narration.

10. Keep mathematical content inside the safe screen area.

11. Avoid overlapping objects.

12. Use next_to(), arrange(), shift(), move_to(),
    to_edge(), align_to().

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
# SPOKEN TEXT
# ============================================================

def clean_spoken_text(text: str) -> str:
    text = str(text).strip()

    text = text.replace("$$", "")
    text = text.replace("$", "")

    replacements = {
        r"\\boxed": "",
        r"\\quad": " ",
        r"\\text": "",
        r"\\frac": " divided by ",
        r"\\sqrt": " square root of ",
        r"\\left": "",
        r"\\right": "",
        r"\\begin": "",
        r"\\end": "",
        r"\\times": " times ",
        r"\\cdot": " times ",
        r"\\pm": " plus or minus ",
        r"\\leq": " less than or equal to ",
        r"\\geq": " greater than or equal to ",
        r"\\neq": " not equal to ",
        r"\\approx": " approximately ",
        r"\\infty": " infinity ",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

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


# ============================================================
# LATEX CLEANING
# ============================================================

def clean_latex(text: str) -> str:
    """
    Normalize LaTeX coming from Lovable/JSON/AI.

    Critical fix:
        \\\\text  -> \\text

    The renderer ultimately passes the result to MathTex
    as a normal Python string created with repr().
    """

    text = str(text).strip()

    if not text:
        return ""

    # Remove surrounding math delimiters.
    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2].strip()

    elif text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()

    # Normalize CR/LF.
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # --------------------------------------------------------
    # CRITICAL BACKSLASH NORMALIZATION
    # --------------------------------------------------------
    #
    # Incoming JSON / AI content can contain:
    #
    #   \\text
    #
    # when the actual LaTeX required by MathTex is:
    #
    #   \text
    #
    # Collapse repeated backslashes.
    #
    text = re.sub(r"\\{2,}", r"\\", text)

    # --------------------------------------------------------
    # Repair common commands that may arrive without slash.
    # --------------------------------------------------------

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
    ]

    for command in commands:
        text = re.sub(
            rf"(?<!\\)\b{command}\b",
            rf"\\{command}",
            text,
        )

    # Remove accidental dollar signs.
    text = text.replace("$", "")

    return text.strip()


# ============================================================
# SAFE PYTHON STRING
# ============================================================

def python_literal(text: str) -> str:
    """
    Safely convert text to a Python string literal.

    repr() is deliberately used instead of manually escaping
    LaTeX backslashes.
    """

    return repr(str(text))


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

    if not Path(audio_file).exists():
        return 3.0

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
            timeout=20,
        )

        duration = float(
            result.stdout.strip()
        )

        return max(duration, 1.0)

    except Exception as exc:

        print(
            f"Could not determine audio duration: {exc}"
        )

        return 3.0


# ============================================================
# TEXT WRAPPING
# ============================================================

def wrap_text_for_manim(
    text: str,
    max_chars: int = 65,
) -> str:

    words = text.split()

    if not words:
        return ""

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

        else:
            current = candidate

    if current:
        lines.append(current)

    return "\n".join(lines[:4])


# ============================================================
# DIAGRAM CODE
# ============================================================

def add_diagram_to_script(
    script_lines: list,
    diagram_spec: Optional[dict],
):
    """
    Creates a controlled diagram occupying the LEFT side
    of the screen.

    The mathematical solution occupies the RIGHT side.
    """

    if not diagram_spec:
        return

    kind = diagram_spec.get("kind")

    # ========================================================
    # PARALLELOGRAM
    # ========================================================

    if kind == "parallelogram_to_triangles":

        script_lines.extend([
            "",
            "        # ----------------------------------------",
            "        # LEFT-SIDE PARALLELOGRAM DIAGRAM",
            "        # ----------------------------------------",
            "",
            "        p1 = LEFT * 5 + DOWN * 1",
            "        p2 = LEFT * 1.5 + DOWN * 1",
            "        p3 = RIGHT * 0.3 + UP * 1.3",
            "        p4 = LEFT * 3.2 + UP * 1.3",
            "",
            "        parallelogram = Polygon(",
            "            p1, p2, p3, p4,",
            "            color=TEAL,",
            "            fill_opacity=0.15,",
            "            stroke_width=4,",
            "        )",
            "",
            "        base_line = Line(",
            "            p1, p2,",
            "            color=WHITE,",
            "            stroke_width=4,",
            "        )",
            "",
            "        diagonal = Line(",
            "            p1, p3,",
            "            color=YELLOW,",
            "            stroke_width=4,",
            "        )",
            "",
            "        height_line = DashedLine(",
            "            p4,",
            "            [p4[0], p1[1], 0],",
            "            color=RED,",
            "            stroke_width=3,",
            "            dash_length=0.08,",
            "        )",
            "",
            "        base_label = MathTex(",
            "            r\"b\",",
            "            font_size=38,",
            "        )",
            "        base_label.next_to(",
            "            base_line,",
            "            DOWN * 0.55,",
            "        )",
            "",
            "        height_label = MathTex(",
            "            r\"h\",",
            "            font_size=38,",
            "        )",
            "        height_label.next_to(",
            "            height_line,",
            "            RIGHT * 0.25,",
            "        )",
            "",
            "        diagram_group = VGroup(",
            "            parallelogram,",
            "            base_line,",
            "            diagonal,",
            "            height_line,",
            "            base_label,",
            "            height_label,",
            "        )",
            "",
            "        diagram_group.shift(LEFT * 0.4)",
            "",
            "        self.play(",
            "            Create(parallelogram),",
            "            Create(base_line),",
            "            Create(diagonal),",
            "            Create(height_line),",
            "            Write(base_label),",
            "            Write(height_label),",
            "            run_time=1.5,",
            "        )",
            "",
        ])

    # ========================================================
    # TRIANGLE
    # ========================================================

    elif kind == "triangle":

        script_lines.extend([
            "",
            "        # ----------------------------------------",
            "        # LEFT-SIDE TRIANGLE",
            "        # ----------------------------------------",
            "",
            "        triangle = Polygon(",
            "            LEFT * 4.5 + DOWN * 1.2,",
            "            LEFT * 0.8 + DOWN * 1.2,",
            "            LEFT * 2.6 + UP * 2.0,",
            "            color=TEAL,",
            "            fill_opacity=0.18,",
            "            stroke_width=4,",
            "        )",
            "",
            "        self.play(",
            "            Create(triangle),",
            "            run_time=1.2,",
            "        )",
            "",
        ])

    # ========================================================
    # CIRCLE
    # ========================================================

    elif kind == "circle":

        script_lines.extend([
            "",
            "        # ----------------------------------------",
            "        # LEFT-SIDE CIRCLE",
            "        # ----------------------------------------",
            "",
            "        circle = Circle(",
            "            radius=2.0,",
            "            color=TEAL,",
            "            fill_opacity=0.18,",
            "        )",
            "",
            "        circle.shift(LEFT * 2.5)",
            "",
            "        self.play(",
            "            Create(circle),",
            "            run_time=1.2,",
            "        )",
            "",
        ])

    # ========================================================
    # NUMBER LINE
    # ========================================================

    elif kind == "number_line":

        script_lines.extend([
            "",
            "        # ----------------------------------------",
            "        # NUMBER LINE",
            "        # ----------------------------------------",
            "",
            "        number_line = NumberLine(",
            "            x_range=[-5, 5, 1],",
            "            length=7,",
            "            include_numbers=True,",
            "        )",
            "",
            "        number_line.shift(LEFT * 2.2)",
            "",
            "        self.play(",
            "            Create(number_line),",
            "            run_time=1.2,",
            "        )",
            "",
        ])

    # ========================================================
    # COORDINATE PLANE
    # ========================================================

    elif kind == "coordinate_plane":

        script_lines.extend([
            "",
            "        # ----------------------------------------",
            "        # COORDINATE PLANE",
            "        # ----------------------------------------",
            "",
            "        plane = NumberPlane(",
            "            x_range=[-4, 4, 1],",
            "            y_range=[-3, 3, 1],",
            "            background_line_style={",
            '                "stroke_opacity": 0.25',
            "            },",
            "        )",
            "",
            "        plane.scale(0.75)",
            "        plane.shift(LEFT * 2.3)",
            "",
            "        self.play(",
            "            Create(plane),",
            "            run_time=1.2,",
            "        )",
            "",
        ])

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

    script_lines = []

    # ========================================================
    # IMPORT
    # ========================================================

    script_lines.append(
        "from manim import *"
    )

    script_lines.append("")

    # ========================================================
    # SCENE
    # ========================================================

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

    # ========================================================
    # TITLE
    # ========================================================

    script_lines.append(
        f"        title = Text("
        f"{python_literal(title)}, "
        f"font_size=34"
        f")"
    )

    script_lines.append(
        "        title.to_edge(UP, buff=0.35)"
    )

    script_lines.append(
        "        self.play(Write(title), run_time=0.8)"
    )

    script_lines.append(
        "        self.wait(0.4)"
    )

    # ========================================================
    # CONTENT DIVIDER
    # ========================================================

    script_lines.append(
        "        divider = Line("
        "LEFT * 6.5, "
        "RIGHT * 6.5, "
        "stroke_opacity=0.35"
        ")"
    )

    script_lines.append(
        "        divider.next_to(title, DOWN, buff=0.25)"
    )

    script_lines.append(
        "        self.play(Create(divider), run_time=0.3)"
    )

    script_lines.append("")

    # ========================================================
    # DIAGRAM
    # ========================================================

    add_diagram_to_script(
        script_lines,
        diagram_spec,
    )

    # ========================================================
    # SOLUTION AREA
    # ========================================================

    script_lines.append(
        "        previous_equation = None"
    )

    script_lines.append("")

    for index, step in enumerate(
        steps,
        start=1,
    ):

        # ====================================================
        # EXPLANATION
        # ====================================================

        explanation = clean_spoken_text(
            step.explanation
        )

        if not explanation:
            explanation = f"Now we move to step {index}."

        explanation = wrap_text_for_manim(
            explanation,
            max_chars=58,
        )

        # ====================================================
        # LATEX
        # ====================================================

        cleaned_latex = clean_latex(
            step.math_latex
        )

        # ====================================================
        # AUDIO
        # ====================================================

        audio_file = (
            voice_dir /
            f"step_{index}.mp3"
        )

        audio_exists = False

        try:

            generate_edge_tts_sync(
                explanation,
                str(audio_file),
                DEFAULT_VOICE,
            )

            audio_exists = audio_file.exists()

        except Exception as exc:

            print(
                f"Edge TTS failed for step "
                f"{index}: {exc}"
            )

        duration = (
            get_audio_duration(
                str(audio_file)
            )
            if audio_exists
            else 3.0
        )

        # ====================================================
        # STEP LABEL
        # ====================================================

        script_lines.append(
            f"        step_label = Text("
            f"{python_literal(f'Step {index}')}, "
            f"font_size=27"
            f")"
        )

        script_lines.append(
            "        step_label.to_edge("
            "UP, "
            "buff=0.9"
            ")"
        )

        # ====================================================
        # EQUATION
        # ====================================================

        if cleaned_latex:

            # CRITICAL:
            # repr() preserves a SINGLE LaTeX backslash.
            latex_literal = python_literal(
                cleaned_latex
            )

            script_lines.append(
                f"        tex = MathTex("
                f"{latex_literal}, "
                f"font_size=52"
                f")"
            )

            script_lines.append(
                "        tex.scale_to_fit_width(5.6)"
            )

            script_lines.append(
                "        tex.move_to("
                "RIGHT * 2.5 + UP * 0.8"
                ")"
            )

        else:

            script_lines.append(
                "        tex = Text("
                f"{python_literal('No equation provided')}, "
                f"font_size=30"
                f")"
            )

            script_lines.append(
                "        tex.move_to("
                "RIGHT * 2.5 + UP * 0.8"
                ")"
            )

        # ====================================================
        # EXPLANATION OBJECT
        # ====================================================

        script_lines.append(
            f"        explanation = Text("
            f"{python_literal(explanation)}, "
            f"font_size=21, "
            f"line_spacing=0.9"
            f")"
        )

        script_lines.append(
            "        explanation.set_width(5.6)"
        )

        script_lines.append(
            "        explanation.move_to("
            "RIGHT * 2.5 + DOWN * 1.45"
            ")"
        )

        # ====================================================
        # STEP LABEL
        # ====================================================

        script_lines.append(
            "        self.play("
            "FadeIn(step_label), "
            "run_time=0.35"
            ")"
        )

        # ====================================================
        # EQUATION TRANSITION
        # ====================================================

        if index == 1:

            script_lines.append(
                "        self.play("
                "Write(tex), "
                "run_time=1.0"
                ")"
            )

        else:

            # Avoid TransformMatchingTex because equations
            # can have completely different structures.
            script_lines.append(
                "        self.play("
                "FadeOut(previous_equation), "
                "Write(tex), "
                "run_time=0.8"
                ")"
            )

        # ====================================================
        # EXPLANATION
        # ====================================================

        script_lines.append(
            "        self.play("
            "FadeIn(explanation), "
            "run_time=0.5"
            ")"
        )

        # ====================================================
        # AUDIO
        # ====================================================

        if audio_exists:

            audio_path = (
                str(audio_file)
                .replace("\\", "/")
            )

            script_lines.append(
                f"        self.add_sound("
                f"{python_literal(audio_path)}"
                f")"
            )

        # ====================================================
        # WAIT FOR NARRATION
        # ====================================================

        script_lines.append(
            f"        self.wait("
            f"{max(duration, 1.5):.2f}"
            f")"
        )

        # ====================================================
        # CLEAN STEP
        # ====================================================

        script_lines.append(
            "        self.play("
            "FadeOut(step_label), "
            "FadeOut(explanation), "
            "run_time=0.4"
            ")"
        )

        script_lines.append(
            "        previous_equation = tex"
        )

        script_lines.append("")

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    if steps:

        final_latex = clean_latex(
            steps[-1].math_latex
        )

        if final_latex:

            script_lines.append(
                "        final_box = SurroundingRectangle("
                "previous_equation, "
                "color=YELLOW, "
                "buff=0.25"
                ")"
            )

            script_lines.append(
                "        final_text = Text("
                f"{python_literal('Final Answer')}, "
                "font_size=28"
                ")"
            )

            script_lines.append(
                "        final_text.next_to("
                "final_box, "
                "UP, "
                "buff=0.25"
                ")"
            )

            script_lines.append(
                "        self.play("
                "Create(final_box), "
                "FadeIn(final_text), "
                "run_time=0.8"
                ")"
            )

            script_lines.append(
                "        self.wait(2)"
            )

        else:

            script_lines.append(
                "        self.wait(1)"
            )

    else:

        script_lines.append(
            "        self.wait(1)"
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

    content = (
        response.choices[0]
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
        response.choices[0] 
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
                f"{provider_name}: invalid generated code" 
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
 
    if "class GeneratedScene(Scene)" not in code: 
 
        raise ValueError( 
            "Generated code does not contain " 
            "GeneratedScene." 
        ) 
 
    if "from manim import *" not in code: 
 
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
 
    # ======================================================== 
    # FIND ONLY THIS JOB'S VIDEO 
    # ======================================================== 
 
    video_files = list( 
        job_media_dir.rglob( 
            "GeneratedScene.mp4" 
        ) 
    ) 
 
    if not video_files: 
 
        video_files = list( 
            job_media_dir.rglob("*.mp4") 
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
# ROOT / HEALTH 
# ============================================================ 
 
@app.get("/") 
def root(): 
 
    return { 
        "status": "online", 
        "service": "Tezla Animator Rendering Engine", 
        "version": APP_VERSION, 
        "renderer": "Manim", 
        "tts": "Edge TTS", 
        "sympy": False, 
        "diagrams": True, 
        "latex_normalization": True, 
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
            "available": result.returncode == 0, 
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
            "available": result.returncode == 0, 
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
 
    job_id = str(uuid.uuid4()) 
 
    print("=" * 70) 
    print(f"NEW VIDEO JOB: {job_id}") 
    print("=" * 70) 
 
    print("Prompt:") 
    print(req.prompt) 
 
    print( 
        "Solution steps:", 
        len(req.solution_steps or []), 
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
        # LOG GENERATED SCRIPT 
        # ======================================================== 
 
        print("GENERATED MANIM SCRIPT:") 
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
