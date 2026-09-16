import os
import re
import subprocess
import uuid
import ast
from pathlib import Path
from typing import List, Optional

import sympy as sp

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google import genai
from groq import Groq
from openai import OpenAI


# ============================================================
# APP
# ============================================================

APP_VERSION = "3.1.0"

app = FastAPI(
    title="Tezla Animator - Direct & AI Engine",
    version=APP_VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "rendered_videos"
MEDIA_DIR = BASE_DIR / "media"
VOICE_DIR = BASE_DIR / "voiceovers"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)


app.mount(
    "/videos",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="videos",
)


# ============================================================
# MODELS
# ============================================================

class SolutionStep(BaseModel):
    step_number: int
    math_latex: str
    explanation: str


class RenderRequest(BaseModel):
    prompt: str

    # Existing Lovable payload
    solution_steps: Optional[List[SolutionStep]] = None

    # Optional deterministic mathematics input
    # Example:
    # "x**3 - 3*x"
    math_expression: Optional[str] = None

    # Optional operation:
    # derivative
    # integral
    # simplify
    # factor
    # solve
    operation: Optional[str] = None


# ============================================================
# AI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are a Manim Python script generator for educational videos.

Your output MUST contain ONLY valid Python code.

Do NOT use Markdown.

Do NOT use ```python.

Do NOT explain the code.

The generated script MUST:

1. Import:

from manim import *

2. Define exactly one scene:

class GeneratedScene(Scene):

3. Use Manim's normal Scene class.

4. Do NOT import manim_voiceover.

5. Do NOT use VoiceoverScene.

6. Do NOT use EdgeTTSService.

7. Do NOT generate audio.

8. Mathematical expressions must use valid LaTeX.

9. When using LaTeX commands, use SINGLE backslashes.

CORRECT:

MathTex(r"\boxed{x = 0 \quad \text{or} \quad x = -4}")

INCORRECT:

MathTex(r"\\boxed{x = 0 \\quad \\text{or} \\quad x = -4")

10. Do NOT put dollar signs inside MathTex.

CORRECT:

MathTex(r"x^2 + 4x = 0")

INCORRECT:

MathTex(r"$x^2 + 4x = 0$")

11. Valid LaTeX commands include:

\boxed{}
\quad
\text{}
\frac{}{}
\sqrt{}
\left
\right

12. Do not write malformed commands such as:

boxed{}
quad
text{}
frac{}{}
sqrt{}

13. Keep mathematical notation mathematically correct.

14. Make the animation educational and visually clear.

15. Do not place LaTeX commands inside spoken narration.

16. Do not use external voiceover packages.

17. Do not create audio files.

18. Keep the animation self-contained.

19. The script must be executable directly by Manim.

20. Return Python code only.
"""


# ============================================================
# CLEAN AI CODE
# ============================================================

def clean_code_block(code_text: str) -> str:
    """
    Removes accidental Markdown code fences from AI output.
    """

    if not code_text:
        return ""

    code = code_text.strip()

    code = re.sub(
        r"^\s*```(?:python|py)?\s*",
        "",
        code,
        flags=re.IGNORECASE,
    )

    code = re.sub(
        r"\s*```\s*$",
        "",
        code,
        flags=re.IGNORECASE,
    )

    return code.strip()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_spoken_text(text: str) -> str:
    """
    Converts mathematical/LaTeX-heavy text into safe
    plain text.

    This is only used for text that may be displayed
    visually or returned to the frontend.
    """

    if not text:
        return ""

    text = str(text).strip()

    text = text.replace("$$", "")
    text = text.replace("$", "")

    text = re.sub(
        r"\\(?:boxed|quad|text|frac|sqrt|left|right)\b",
        "",
        text,
    )

    text = text.replace("\\", "")
    text = text.replace("{", "")
    text = text.replace("}", "")

    text = text.replace('"', "'")
    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_title(text: str) -> str:
    """
    Cleans a prompt before putting it inside Manim Text().
    """

    if not text:
        return "Mathematics Solution"

    text = str(text).strip()

    text = text.replace("$$", "")
    text = text.replace("$", "")

    text = re.sub(
        r"\\(?:boxed|quad|text|frac|sqrt|left|right)\b",
        "",
        text,
    )

    text = text.replace("\\", "")
    text = text.replace("{", "")
    text = text.replace("}", "")

    text = text.replace('"', "'")
    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    return text[:100].strip()


# ============================================================
# LATEX CLEANING
# ============================================================

def clean_latex(text: str) -> str:
    """
    Normalizes AI/Lovable LaTeX.

    Important:
    MathTex receives SINGLE backslashes.

    Example:

        \\boxed{x=0}

    becomes:

        \boxed{x=0}
    """

    if not text:
        return r"\text{No expression}"

    text = str(text).strip()

    # Remove surrounding dollar delimiters.
    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2]

    elif text.startswith("$") and text.endswith("$"):
        text = text[1:-1]

    # Collapse repeated backslashes.
    #
    # Four backslashes -> one
    # Three backslashes -> one
    # Two backslashes -> one
    #
    text = re.sub(r"\\{2,}", r"\\", text)

    # Repair common commands where the backslash was lost.
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


def escape_latex_for_python(text: str) -> str:
    """
    Escapes only double quotes.

    IMPORTANT:
    Backslashes MUST NOT be doubled here because the generated
    Manim code uses raw Python strings.
    """

    return text.replace('"', '\\"')


# ============================================================
# SYMPY MATH ENGINE
# ============================================================

def safe_sympify(expression: str):
    """
    Converts a basic mathematical expression into SymPy.

    This uses a controlled namespace rather than allowing
    unrestricted names.
    """

    if not expression:
        raise ValueError("Mathematical expression is empty.")

    expression = str(expression).strip()

    if len(expression) > 500:
        raise ValueError(
            "Mathematical expression is too long."
        )

    allowed = {
        "x": sp.Symbol("x"),
        "y": sp.Symbol("y"),
        "z": sp.Symbol("z"),

        "pi": sp.pi,
        "E": sp.E,

        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,

        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,

        "sqrt": sp.sqrt,
        "log": sp.log,
        "exp": sp.exp,

        "Abs": sp.Abs,
    }

    try:
        return sp.sympify(
            expression,
            locals=allowed,
            evaluate=True,
        )
    except Exception as e:
        raise ValueError(
            f"Could not parse mathematical expression: {e}"
        )


def calculate_with_sympy(
    expression: str,
    operation: str = "derivative",
) -> dict:
    """
    Performs deterministic mathematical calculations.

    Supported operations:

    derivative
    integral
    simplify
    factor
    solve
    """

    expr = safe_sympify(expression)

    x = sp.Symbol("x")

    operation = (
        operation or "derivative"
    ).strip().lower()

    result = {
        "original": str(expr),
        "original_latex": sp.latex(expr),
        "operation": operation,
    }

    if operation in {
        "derivative",
        "differentiate",
        "diff",
    }:

        result_expr = sp.diff(expr, x)

        result["result"] = str(result_expr)
        result["result_latex"] = sp.latex(
            result_expr
        )

        critical_points = sp.solve(
            sp.Eq(result_expr, 0),
            x,
        )

        result["critical_points"] = [
            str(point)
            for point in critical_points
        ]

        result["critical_points_latex"] = [
            sp.latex(point)
            for point in critical_points
        ]

    elif operation in {
        "integral",
        "integrate",
    }:

        result_expr = sp.integrate(
            expr,
            x,
        )

        result["result"] = str(result_expr)
        result["result_latex"] = sp.latex(
            result_expr
        )

    elif operation == "simplify":

        result_expr = sp.simplify(expr)

        result["result"] = str(result_expr)
        result["result_latex"] = sp.latex(
            result_expr
        )

    elif operation == "factor":

        result_expr = sp.factor(expr)

        result["result"] = str(result_expr)
        result["result_latex"] = sp.latex(
            result_expr
        )

    elif operation == "solve":

        solutions = sp.solve(
            expr,
            x,
        )

        result["solutions"] = [
            str(solution)
            for solution in solutions
        ]

        result["solutions_latex"] = [
            sp.latex(solution)
            for solution in solutions
        ]

    else:

        raise ValueError(
            f"Unsupported SymPy operation: {operation}"
        )

    return result


def build_sympy_solution_steps(
    expression: str,
    operation: str = "derivative",
) -> List[SolutionStep]:
    """
    Creates deterministic mathematical steps from SymPy.

    This is useful when Lovable does not already send
    solution_steps.

    Narration can still be generated by Lovable.
    """

    data = calculate_with_sympy(
        expression,
        operation,
    )

    steps = []

    steps.append(
        SolutionStep(
            step_number=1,
            math_latex=(
                rf"f(x) = {data['original_latex']}"
            ),
            explanation=(
                f"Consider the function "
                f"{data['original']}."
            ),
        )
    )

    if operation in {
        "derivative",
        "differentiate",
        "diff",
    }:

        steps.append(
            SolutionStep(
                step_number=2,
                math_latex=(
                    rf"f'(x) = "
                    rf"{data['result_latex']}"
                ),
                explanation=(
                    "Taking the derivative gives "
                    f"{data['result']}."
                ),
            )
        )

        if data.get("critical_points"):

            points_latex = (
                r"\quad".join(
                    data["critical_points_latex"]
                )
            )

            points_text = ", ".join(
                data["critical_points"]
            )

            steps.append(
                SolutionStep(
                    step_number=3,
                    math_latex=(
                        rf"f'(x)=0"
                        rf"\quad\Rightarrow\quad"
                        rf"x = {points_latex}"
                    ),
                    explanation=(
                        "Setting the derivative equal "
                        f"to zero gives the critical "
                        f"points {points_text}."
                    ),
                )
            )

    else:

        steps.append(
            SolutionStep(
                step_number=2,
                math_latex=(
                    data.get(
                        "result_latex",
                        r"\text{Result}",
                    )
                ),
                explanation=(
                    f"The {operation} of the "
                    "expression gives "
                    f"{data.get('result', '')}."
                ),
            )
        )

    return steps


# ============================================================
# MANIM SCRIPT BUILDER
# ============================================================

def build_direct_manim_script(
    prompt: str,
    steps: List[SolutionStep],
    job_id: str,
) -> str:

    clean_prompt = clean_title(prompt)

    safe_title = (
        clean_prompt
        .replace("\\", "")
        .replace('"', '\\"')
    )

    script_parts = []

    script_parts.append(
        "from manim import *"
    )

    script_parts.append("")

    script_parts.append(
        "class GeneratedScene(Scene):"
    )

    script_parts.append(
        "    def construct(self):"
    )

    script_parts.append(
        f'        title = Text("{safe_title}", font_size=32)'
    )

    script_parts.append(
        "        title.to_edge(UP)"
    )

    script_parts.append(
        "        self.play(Write(title), run_time=1.2)"
    )

    script_parts.append(
        "        self.wait(0.4)"
    )

    script_parts.append(
        "        current_mobject = None"
    )

    script_parts.append("")

    for index, step in enumerate(steps):

        cleaned_latex = clean_latex(
            step.math_latex
        )

        latex = escape_latex_for_python(
            cleaned_latex
        )

        # Keep the explanation available as a comment.
        # Lovable remains responsible for narration.
        explanation = clean_spoken_text(
            step.explanation
        )

        safe_explanation = (
            explanation
            .replace('"', "'")
            .replace("\n", " ")
        )

        step_number = step.step_number

        script_parts.append(
            f"        # Step {step_number}"
        )

        script_parts.append(
            f"        # Narration from Lovable: "
            f"{safe_explanation}"
        )

        script_parts.append(
            f'        next_mobject = MathTex('
            f'r"{latex}", '
            f'font_size=44'
            f')'
        )

        script_parts.append(
            "        next_mobject.move_to(ORIGIN)"
        )

        if index == 0:

            script_parts.append(
                "        self.play("
                "Write(next_mobject), "
                "run_time=1.5"
                ")"
            )

        else:

            script_parts.append(
                "        self.play("
                "Transform("
                "current_mobject, "
                "next_mobject"
                "), "
                "run_time=1.5"
                ")"
            )

        script_parts.append(
            "        current_mobject = next_mobject"
        )

        script_parts.append(
            "        self.wait(1.2)"
        )

        script_parts.append("")

    script_parts.append(
        "        self.wait(1.5)"
    )

    return "\n".join(script_parts)


# ============================================================
# AI FALLBACK
# ============================================================

def solve_with_ai_fallback(
    prompt: str,
) -> str:

    full_prompt = f"""
{SYSTEM_PROMPT}

Generate an educational Manim scene for:

{prompt}
"""

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    gemini_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if gemini_key:

        try:

            print("Trying Gemini...")

            client = genai.Client(
                api_key=gemini_key
            )

            response = (
                client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt,
                )
            )

            if response.text:

                return clean_code_block(
                    response.text
                )

        except Exception as e:

            print(
                f"Gemini failed: {e}"
            )

    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    groq_key = os.getenv(
        "GROQ_API_KEY"
    )

    if groq_key:

        try:

            print("Trying Groq...")

            client = Groq(
                api_key=groq_key
            )

            response = (
                client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": full_prompt,
                        },
                    ],
                    temperature=0.2,
                )
            )

            content = (
                response
                .choices[0]
                .message
                .content
            )

            if content:

                return clean_code_block(
                    content
                )

        except Exception as e:

            print(
                f"Groq failed: {e}"
            )

    # --------------------------------------------------------
    # OPENROUTER
    # --------------------------------------------------------

    openrouter_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if openrouter_key:

        try:

            print(
                "Trying OpenRouter..."
            )

            client = OpenAI(
                base_url=(
                    "https://openrouter.ai/api/v1"
                ),
                api_key=openrouter_key,
            )

            models = [
                "openrouter/free",
                "cohere/north-mini-code:free",
            ]

            for model_id in models:

                try:

                    print(
                        f"Trying {model_id}"
                    )

                    response = (
                        client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {
                                    "role": "system",
                                    "content": SYSTEM_PROMPT,
                                },
                                {
                                    "role": "user",
                                    "content": full_prompt,
                                },
                            ],
                            temperature=0.2,
                        )
                    )

                    content = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                    if content:

                        return clean_code_block(
                            content
                        )

                except Exception as e:

                    print(
                        f"{model_id} failed: {e}"
                    )

        except Exception as e:

            print(
                f"OpenRouter failed: {e}"
            )

    raise HTTPException(
        status_code=500,
        detail="All AI providers failed.",
    )


# ============================================================
# VALIDATE GENERATED PYTHON
# ============================================================

def validate_python_code(
    code: str,
) -> None:

    if not code.strip():

        raise HTTPException(
            status_code=500,
            detail="Generated script is empty.",
        )

    if "class GeneratedScene" not in code:

        raise HTTPException(
            status_code=500,
            detail=(
                "Generated script does not contain "
                "GeneratedScene."
            ),
        )

    if "manim_voiceover" in code:

        raise HTTPException(
            status_code=500,
            detail=(
                "Generated script attempted to use "
                "manim_voiceover. The current engine "
                "does not use that package."
            ),
        )

    if "VoiceoverScene" in code:

        raise HTTPException(
            status_code=500,
            detail=(
                "Generated script attempted to use "
                "VoiceoverScene."
            ),
        )

    try:

        ast.parse(code)

    except SyntaxError as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Generated Manim Python contains "
                f"a syntax error: {e}"
            ),
        )


# ============================================================
# MANIM RENDER
# ============================================================

def render_manim_script(
    script_path: Path,
    job_id: str,
) -> Path:

    job_media_dir = (
        MEDIA_DIR / job_id
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
        "========================================"
    )

    print(
        f"Starting Manim render: {job_id}"
    )

    print(
        " ".join(
            str(item)
            for item in manim_cmd
        )
    )

    print(
        "========================================"
    )

    try:

        result = subprocess.run(
            manim_cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

    except subprocess.TimeoutExpired:

        raise HTTPException(
            status_code=504,
            detail=(
                "Manim rendering timed out "
                "after 10 minutes."
            ),
        )

    # Always print logs.
    print(
        "MANIM STDOUT:"
    )

    print(
        result.stdout[-12000:]
    )

    print(
        "MANIM STDERR:"
    )

    print(
        result.stderr[-12000:]
    )

    if result.returncode != 0:

        error_msg = (
            result.stderr
            or result.stdout
            or "Unknown Manim error."
        )

        print(
            "========================================"
        )

        print(
            f"MANIM RENDERING FAILED FOR JOB {job_id}"
        )

        print(
            error_msg
        )

        print(
            "========================================"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Manim Rendering Error:\n"
                + error_msg[-12000:]
            ),
        )

    # --------------------------------------------------------
    # FIND VIDEO
    # --------------------------------------------------------

    possible_outputs = list(
        job_media_dir.rglob(
            "GeneratedScene.mp4"
        )
    )

    if not possible_outputs:

        possible_outputs = list(
            job_media_dir.rglob(
                "*.mp4"
            )
        )

    if not possible_outputs:

        raise HTTPException(
            status_code=500,
            detail=(
                "Manim completed without producing "
                "an MP4 video."
            ),
        )

    source_video = possible_outputs[-1]

    final_video = (
        OUTPUT_DIR
        / f"{job_id}.mp4"
    )

    if final_video.exists():

        final_video.unlink()

    source_video.replace(
        final_video
    )

    return final_video


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():

    return {
        "status": "online",
        "service": "Tezla Animator Engine",
        "version": APP_VERSION,
        "math_engine": "SymPy",
        "animation_engine": "Manim",
        "narration": "Lovable",
    }


# ============================================================
# ENGINE TEST
# ============================================================

@app.get("/engine-test")
def engine_test():

    import sys

    result = {
        "status": "ok",
        "python": sys.version,
        "version": APP_VERSION,
    }

    # --------------------------------------------------------
    # Manim
    # --------------------------------------------------------

    try:

        manim_result = subprocess.run(
            [
                "manim",
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        result["manim"] = (
            manim_result.stdout.strip()
            or manim_result.stderr.strip()
        )

    except Exception as e:

        result["manim_error"] = str(e)

    # --------------------------------------------------------
    # SymPy
    # --------------------------------------------------------

    try:

        result["sympy"] = sp.__version__

    except Exception as e:

        result["sympy_error"] = str(e)

    # --------------------------------------------------------
    # FFprobe
    # --------------------------------------------------------

    try:

        ffprobe = subprocess.run(
            [
                "ffprobe",
                "-version",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        result["ffprobe"] = (
            ffprobe.stdout.splitlines()[0]
            if ffprobe.stdout
            else "available"
        )

    except Exception as e:

        result["ffprobe_error"] = str(e)

    return result


# ============================================================
# OPTIONAL SYMPY TEST ENDPOINT
# ============================================================

@app.post("/math-test")
def math_test(req: RenderRequest):

    if not req.math_expression:

        raise HTTPException(
            status_code=400,
            detail=(
                "math_expression is required."
            ),
        )

    try:

        data = calculate_with_sympy(
            req.math_expression,
            req.operation or "derivative",
        )

        return {
            "status": "success",
            "math": data,
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ============================================================
# GENERATE VIDEO
# ============================================================

@app.post("/generate-video")
def generate_math_video(
    req: RenderRequest,
):

    job_id = str(
        uuid.uuid4()
    )[:8]

    script_filename = (
        f"temp_{job_id}.py"
    )

    script_path = (
        BASE_DIR
        / script_filename
    )

    try:

        # ====================================================
        # PATH 1:
        # LOVABLE ALREADY PROVIDED SOLUTION STEPS
        # ====================================================

        if (
            req.solution_steps
            and len(req.solution_steps) > 0
        ):

            print(
                f"[{job_id}] "
                "Using solution steps supplied by Lovable."
            )

            generated_code = (
                build_direct_manim_script(
                    req.prompt,
                    req.solution_steps,
                    job_id,
                )
            )

            used_provider = (
                "Lovable Steps + SymPy-safe "
                "LaTeX + Manim"
            )

        # ====================================================
        # PATH 2:
        # SYMPY CALCULATES THE MATHEMATICS
        # ====================================================

        elif req.math_expression:

            print(
                f"[{job_id}] "
                "Using deterministic SymPy math engine."
            )

            try:

                steps = (
                    build_sympy_solution_steps(
                        req.math_expression,
                        req.operation
                        or "derivative",
                    )
                )

            except Exception as e:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SymPy could not solve the "
                        f"expression: {e}"
                    ),
                )

            generated_code = (
                build_direct_manim_script(
                    req.prompt,
                    steps,
                    job_id,
                )
            )

            used_provider = (
                "SymPy + Manim"
            )

        # ====================================================
        # PATH 3:
        # AI FALLBACK
        # ====================================================

        else:

            print(
                f"[{job_id}] "
                "Using AI fallback."
            )

            generated_code = (
                solve_with_ai_fallback(
                    req.prompt
                )
            )

            used_provider = (
                "AI Fallback Chain"
            )

        # ====================================================
        # CLEAN
        # ====================================================

        generated_code = (
            clean_code_block(
                generated_code
            )
        )

        # ====================================================
        # VALIDATE
        # ====================================================

        validate_python_code(
            generated_code
        )

        # ====================================================
        # SAVE
        # ====================================================

        script_path.write_text(
            generated_code,
            encoding="utf-8",
        )

        print(
            f"[{job_id}] "
            f"Script saved to {script_path}"
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
        # SUCCESS
        # ====================================================

        return {
            "status": "success",
            "job_id": job_id,
            "provider_used": used_provider,
            "video_url": (
                f"/videos/{final_video.name}"
            ),
            "math_engine": (
                "SymPy"
                if req.math_expression
                else "Lovable/AI"
            ),
            "narration_engine": "Lovable",
        }

    except HTTPException:

        raise

    except Exception as e:

        print(
            f"[{job_id}] "
            f"Unexpected error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        # ----------------------------------------------------
        # Remove temporary Python script
        # ----------------------------------------------------

        try:

            if script_path.exists():

                script_path.unlink()

        except Exception as e:

            print(
                "Could not remove temporary "
                f"script: {e}"
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
        "main:app",
        host="0.0.0.0",
        port=port,
    )
