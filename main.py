import os
import re
import json
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
# APP CONFIGURATION
# ============================================================

APP_VERSION = "3.0.2"

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "rendered_videos"
MEDIA_DIR = BASE_DIR / "media"
VOICE_DIR = BASE_DIR / "voiceovers"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Tezla Animator Rendering Engine",
    version=APP_VERSION,
)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# STATIC VIDEO FILES
# ============================================================

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
    solution_steps: Optional[List[SolutionStep]] = None


# ============================================================
# SYSTEM PROMPT FOR AI MANIM FALLBACK
# ============================================================

SYSTEM_PROMPT = r"""
You are an expert educational mathematics animation programmer.

Generate a complete executable Manim Python script.

IMPORTANT RULES:

1. Import:
   from manim import *

2. Define exactly one scene:
   class GeneratedScene(Scene):

3. Do NOT use:
   - manim_voiceover
   - VoiceoverScene
   - EdgeTTSService

4. The script must be directly executable by Manim.

5. Use MathTex for mathematical expressions.

6. Do NOT put $ symbols inside MathTex.

7. Use valid LaTeX with SINGLE backslashes.

8. Common valid commands include:
   \boxed{}
   \quad
   \text{}
   \frac{}{}
   \sqrt{}
   \left
   \right

9. Do not put LaTeX commands inside spoken narration.

10. Keep animations educational and easy to follow.

11. Use:
       class GeneratedScene(Scene):

12. Do not create additional Scene classes.

13. Return ONLY valid Python code.
"""


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def clean_code_block(code: str) -> str:
    """
    Remove Markdown code fences if an AI provider returns them.
    """

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
    """
    Convert explanation text into safe spoken narration.
    """

    text = str(text).strip()

    # Remove LaTeX delimiters
    text = text.replace("$$", "")
    text = text.replace("$", "")

    # Remove common LaTeX commands
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

    # Remove remaining LaTeX backslashes
    text = text.replace("\\", "")

    # Remove braces
    text = text.replace("{", "")
    text = text.replace("}", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_title(text: str) -> str:
    """
    Make a safe plain-text title for Manim Text().
    """

    text = clean_spoken_text(text)

    if not text:
        text = "Mathematics Solution"

    return text[:120]


def clean_latex(text: str) -> str:
    """
    Clean LaTeX without destroying valid commands.
    """

    text = str(text).strip()

    # Remove surrounding display math delimiters
    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2]

    elif text.startswith("$") and text.endswith("$"):
        text = text[1:-1]

    # Collapse multiple consecutive backslashes into one
    text = re.sub(r"\\+", r"\\", text)

    # Add missing backslashes to common commands
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
    """
    Escape only characters that would break the generated
    Python string.

    Do NOT double LaTeX backslashes.
    """

    return text.replace('"', '\\"')


# ============================================================
# EDGE TTS
# ============================================================

DEFAULT_VOICE = "en-NG-EzinneNeural"


def generate_edge_tts_sync(
    text: str,
    output_file: str,
    voice: str = DEFAULT_VOICE,
):
    """
    Generate MP3 narration using Edge TTS.
    """

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
    """
    Get MP3 duration using ffprobe.
    """

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

        return max(float(result.stdout.strip()), 1.0)

    except Exception as exc:
        print(f"Could not determine audio duration: {exc}")
        return 4.0


# ============================================================
# DIRECT MANIM SCRIPT GENERATION
# ============================================================

def build_direct_manim_script(
    prompt: str,
    steps: List[SolutionStep],
    job_id: str,
) -> str:

    voice_dir = VOICE_DIR / job_id
    voice_dir.mkdir(parents=True, exist_ok=True)

    title = clean_title(prompt)

    script_lines = []

    script_lines.append("from manim import *")
    script_lines.append("import edge_tts")
    script_lines.append("import asyncio")
    script_lines.append("from pathlib import Path")
    script_lines.append("")
    script_lines.append("")
    script_lines.append("def generate_voice(text, output_file):")
    script_lines.append("    async def _generate():")
    script_lines.append(
        f"        communicate = edge_tts.Communicate("
        f"text=text, voice={DEFAULT_VOICE!r})"
    )
    script_lines.append("        await communicate.save(output_file)")
    script_lines.append("")
    script_lines.append("    asyncio.run(_generate())")
    script_lines.append("")
    script_lines.append("")
    script_lines.append("class GeneratedScene(Scene):")
    script_lines.append("    def construct(self):")
    script_lines.append("")

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    safe_title = title.replace('"', '\\"')

    script_lines.append(
        f'        title = Text("{safe_title}", font_size=36)'
    )

    script_lines.append("        self.play(Write(title))")
    script_lines.append("        self.wait(1)")
    script_lines.append("        self.play(FadeOut(title))")
    script_lines.append("")

    # --------------------------------------------------------
    # STEPS
    # --------------------------------------------------------

    for index, step in enumerate(steps, start=1):

        explanation = clean_spoken_text(step.explanation)

        if not explanation:
            explanation = f"Step {index}"

        safe_explanation = explanation.replace('"', '\\"')

        cleaned_latex = clean_latex(step.math_latex)
        latex = escape_latex(cleaned_latex)

        audio_file = voice_dir / f"step_{index}.mp3"

        try:
            generate_edge_tts_sync(
                explanation,
                str(audio_file),
                DEFAULT_VOICE,
            )
        except Exception as exc:
            print(
                f"Edge TTS failed for step {index}: {exc}"
            )

        duration = get_audio_duration(str(audio_file))

        script_lines.append(
            f'        step_label = Text('
            f'"Step {index}", font_size=30'
            f')'
        )

        script_lines.append(
            "        step_label.to_edge(UP)"
        )

        script_lines.append(
            f'        explanation = Text('
            f'"{safe_explanation}", '
            f'font_size=26'
            f')'
        )

        script_lines.append(
            "        explanation.to_edge(DOWN)"
        )

        # ----------------------------------------------------
        # MATHEMATICAL EXPRESSION
        # ----------------------------------------------------

        if latex:

            script_lines.append(
                f'        equation = MathTex('
                f'r"{latex}", '
                f'font_size=44'
                f')'
            )

        else:

            script_lines.append(
                '        equation = Text('
                '"No equation provided", '
                'font_size=36'
                ')'
            )

        script_lines.append(
            "        equation.move_to(ORIGIN)"
        )

        script_lines.append(
            "        self.play(Write(step_label))"
        )

        script_lines.append(
            "        self.play(Write(equation))"
        )

        script_lines.append(
            "        self.play(Write(explanation))"
        )

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        if audio_file.exists():

            safe_audio_path = str(audio_file).replace("\\", "/")

            script_lines.append(
                f'        self.add_sound('
                f'"{safe_audio_path}", '
                f'time_offset=0'
                f')'
            )

        script_lines.append(
            f"        self.wait({max(duration, 1.0):.2f})"
        )

        script_lines.append(
            "        self.play("
            "FadeOut(step_label), "
            "FadeOut(equation), "
            "FadeOut(explanation)"
            ")"
        )

        script_lines.append("")

    script_lines.append(
        '        final_text = Text("Final Answer", font_size=42)'
    )

    script_lines.append(
        "        self.play(Write(final_text))"
    )

    script_lines.append(
        "        self.wait(2)"
    )

    return "\n".join(script_lines)


# ============================================================
# AI PROVIDERS
# ============================================================

def generate_with_gemini(prompt: str) -> str:

    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            SYSTEM_PROMPT,
            "\n\nUSER REQUEST:\n",
            prompt,
        ],
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    return clean_code_block(response.text)


def generate_with_groq(prompt: str) -> str:

    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    client = Groq(api_key=api_key)

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
        raise RuntimeError("Groq returned an empty response")

    return clean_code_block(content)


def generate_with_openrouter(
    prompt: str,
    model: str,
) -> str:

    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured"
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
            "OpenRouter returned an empty response"
        )

    return clean_code_block(content)


# ============================================================
# AI FALLBACK CHAIN
# ============================================================

def generate_ai_manim_code(prompt: str):

    providers = []

    if os.getenv("GEMINI_API_KEY"):
        providers.append(
            (
                "gemini",
                lambda: generate_with_gemini(prompt),
            )
        )

    if os.getenv("GROQ_API_KEY"):
        providers.append(
            (
                "groq",
                lambda: generate_with_groq(prompt),
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
                f"Trying AI provider: {provider_name}"
            )

            code = provider_function()

            if code and "GeneratedScene" in code:
                return code, provider_name

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
# CODE VALIDATION
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
                f"Generated code contains forbidden "
                f"dependency: {item}"
            )

    if "class GeneratedScene(Scene)" not in code:

        raise ValueError(
            "Generated code does not contain "
            "class GeneratedScene(Scene)."
        )

    if "from manim import *" not in code:

        raise ValueError(
            "Generated code does not import Manim."
        )


# ============================================================
# MANIM RENDER
# ============================================================

def render_manim_script(
    script_path: Path,
    job_id: str,
) -> Path:

    print(
        f"Starting Manim render for job {job_id}"
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
        "Running command:",
        " ".join(manim_cmd),
    )

    process = subprocess.run(
        manim_cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    print("Manim STDOUT:")
    print(process.stdout)

    print("Manim STDERR:")
    print(process.stderr)

    if process.returncode != 0:

        raise RuntimeError(
            "Manim rendering failed.\n\n"
            + process.stdout
            + "\n\n"
            + process.stderr
        )

    # --------------------------------------------------------
    # FIND GENERATED VIDEO
    # --------------------------------------------------------

    video_files = list(
        MEDIA_DIR.rglob("GeneratedScene.mp4")
    )

    if not video_files:

        # Some Manim versions may create a different path.
        video_files = list(
            MEDIA_DIR.rglob("*.mp4")
        )

    if not video_files:

        raise RuntimeError(
            "Manim completed but no MP4 file was found."
        )

    source_video = video_files[-1]

    final_video = OUTPUT_DIR / f"{job_id}.mp4"

    shutil.copy2(
        source_video,
        final_video,
    )

    print(
        f"Final video created: {final_video}"
    )

    return final_video


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Tezla Animator Rendering Engine",
        "version": APP_VERSION,
        "tts": "Edge TTS",
        "renderer": "Manim",
        "sympy": False,
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
            "version": result.stdout.strip()
            or result.stderr.strip(),
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
            "version": result.stdout.splitlines()[0]
            if result.stdout
            else "installed",
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
def generate_video(req: RenderRequest):

    job_id = str(uuid.uuid4())

    print("=" * 70)
    print(f"NEW VIDEO JOB: {job_id}")
    print("=" * 70)

    print("Prompt:")
    print(req.prompt)

    print(
        f"Solution steps received: "
        f"{len(req.solution_steps or [])}"
    )

    script_path = BASE_DIR / f"{job_id}.py"

    provider_used = "direct-solution-steps"

    try:

        # ====================================================
        # PATH 1:
        # LOVABLE ALREADY PROVIDED SOLUTION STEPS
        # ====================================================

        if req.solution_steps:

            print(
                "Using Lovable-provided solution steps."
            )

            code = build_direct_manim_script(
                req.prompt,
                req.solution_steps,
                job_id,
            )

        # ====================================================
        # PATH 2:
        # AI FALLBACK
        # ====================================================

        else:

            print(
                "No solution steps received."
            )

            print(
                "Falling back to AI Manim generation."
            )

            code, provider_used = (
                generate_ai_manim_code(
                    req.prompt
                )
            )

        # ====================================================
        # VALIDATE
        # ====================================================

        validate_manim_code(code)

        # ====================================================
        # WRITE SCRIPT
        # ====================================================

        script_path.write_text(
            code,
            encoding="utf-8",
        )

        print(
            f"Manim script written to: "
            f"{script_path}"
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

        video_url = f"/videos/{final_video.name}"

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

        # ----------------------------------------------------
        # REMOVE TEMP SCRIPT
        # ----------------------------------------------------

        try:

            if script_path.exists():
                script_path.unlink()

        except Exception as exc:

            print(
                f"Could not delete temp script: {exc}"
            )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv("PORT", "8000")
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
