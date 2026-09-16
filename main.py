````python
import os
import re
import subprocess
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ============================================================
# AI PROVIDERS
# ============================================================

from google import genai
from groq import Groq
from openai import OpenAI


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Tezla Animator - Direct & AI Engine",
    version="3.0.1"
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
    name="videos"
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

7. Mathematical expressions must use valid LaTeX.

8. When using LaTeX commands, use SINGLE backslashes.

CORRECT:

MathTex(r"\boxed{x = 0 \quad \text{or} \quad x = -4}")

INCORRECT:

MathTex(r"\\boxed{x = 0 \\quad \\text{or} \\quad x = -4}")

9. Do NOT put dollar signs inside MathTex.

CORRECT:

MathTex(r"x^2 + 4x = 0")

INCORRECT:

MathTex(r"$x^2 + 4x = 0$")

10. Valid LaTeX commands include:

\boxed{}
\quad
\text{}
\frac{}{}
\sqrt{}
\left
\right

11. Do not write malformed commands such as:

boxed{}
quad
text{}
frac{}{}

12. Keep mathematical notation mathematically correct.

13. Make the animation educational and visually clear.

14. Do not place LaTeX commands inside spoken narration.

15. The script must be executable directly by Manim.

16. Do not use external voiceover packages.
"""


# ============================================================
# CLEAN AI CODE
# ============================================================

def clean_code_block(code_text: str) -> str:

    code = code_text.strip()

    code = re.sub(
        r"^```(?:python)?\s*",
        "",
        code,
        flags=re.IGNORECASE
    )

    code = re.sub(
        r"\s*```$",
        "",
        code
    )

    return code.strip()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_spoken_text(text: str) -> str:
    """
    Converts mathematical/LaTeX-heavy text into safe
    plain spoken narration for Edge TTS.
    """

    text = text.strip()

    # Remove LaTeX delimiters
    text = text.replace("$$", "")
    text = text.replace("$", "")

    # Remove LaTeX commands while keeping their content
    text = re.sub(
        r"\\(?:boxed|quad|text|frac|sqrt|left|right)\b",
        "",
        text
    )

    # Remove remaining backslashes
    text = text.replace("\\", "")

    # Remove braces
    text = text.replace("{", "")
    text = text.replace("}", "")

    # Protect against quote problems
    text = text.replace('"', "'")

    # Flatten newlines
    text = text.replace("\n", " ")

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_title(text: str) -> str:
    """
    Converts a mathematical prompt into a clean visual title.

    The title is rendered using Text(), not MathTex(),
    so LaTeX commands must not remain in it.
    """

    text = text.strip()

    # Remove LaTeX delimiters
    text = text.replace("$$", "")
    text = text.replace("$", "")

    # Remove common LaTeX commands
    text = re.sub(
        r"\\(?:boxed|quad|text|frac|sqrt|left|right)\b",
        "",
        text
    )

    # Remove remaining backslashes
    text = text.replace("\\", "")

    # Remove braces
    text = text.replace("{", "")
    text = text.replace("}", "")

    # Protect generated Python string
    text = text.replace('"', "'")

    # Flatten newlines
    text = text.replace("\n", " ")

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text[:80].strip()


# ============================================================
# LATEX CLEANING
# ============================================================

def clean_latex(text: str) -> str:
    """
    Cleans and normalizes LaTeX before passing it to MathTex.

    Important:
    MathTex receives the final LaTeX command with SINGLE
    backslashes.

    Examples:

        \\boxed{x = 0}
    becomes:
        \boxed{x = 0}

    Dollar delimiters are also removed because MathTex
    does not require them.
    """

    text = str(text).strip()

    # --------------------------------------------------------
    # Remove surrounding dollar delimiters
    # --------------------------------------------------------

    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2]

    elif text.startswith("$") and text.endswith("$"):
        text = text[1:-1]

    # --------------------------------------------------------
    # Normalize accidental double/multiple backslashes
    #
    # Example:
    #
    # \\boxed
    #
    # becomes:
    #
    # \boxed
    #
    # This handles JSON/API escaped LaTeX.
    # --------------------------------------------------------

    text = re.sub(
        r"\\\\+",
        r"\\",
        text
    )

    # --------------------------------------------------------
    # Fix common AI mistakes where the backslash disappeared
    #
    # boxed{x} -> \boxed{x}
    # quad -> \quad
    # text{...} -> \text{...}
    # frac{...}{...} -> \frac{...}{...}
    # sqrt{...} -> \sqrt{...}
    # --------------------------------------------------------

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
            text
        )

    return text.strip()


def escape_latex(text: str) -> str:
    """
    Escape LaTeX only for insertion into the generated
    Python source code.

    IMPORTANT:
    DO NOT double LaTeX backslashes here.

    The generated source uses:

        r"..."

    which is a Python raw string.

    Therefore:

        \boxed
        \quad
        \text

    must remain single-backslash LaTeX commands.
    """

    return text.replace('"', '\\"')


# ============================================================
# DIRECT EDGE TTS
# ============================================================

def generate_edge_tts_sync(
    text: str,
    output_file: str,
    voice: str = "en-NG-EzinneNeural"
):

    import edge_tts

    async def generate():

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice
        )

        await communicate.save(
            output_file
        )

    asyncio.run(generate())


# ============================================================
# GET AUDIO DURATION
# ============================================================

def get_audio_duration(
    audio_file: str
) -> float:

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
                audio_file
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:

            return max(
                0.5,
                float(result.stdout.strip())
            )

    except Exception as e:

        print(
            f"Could not determine audio duration: {e}"
        )

    return 2.5


# ============================================================
# DIRECT MANIM SCRIPT
# ============================================================

def build_direct_manim_script(
    prompt: str,
    steps: List[SolutionStep],
    job_id: str
) -> str:

    clean_prompt = clean_title(prompt)

    job_voice_dir = (
        VOICE_DIR / job_id
    )

    job_voice_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Escape the title for Python source generation.
    safe_title = clean_prompt.replace("\\", "\\\\")
    safe_title = safe_title.replace('"', '\\"')

    script = f'''
from manim import *
import asyncio
import subprocess
from pathlib import Path
import edge_tts


VOICE_DIR = Path(r"{job_voice_dir}")


def generate_voice(
    text,
    filename,
    voice="en-NG-EzinneNeural"
):

    output_file = VOICE_DIR / filename

    async def _generate():

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice
        )

        await communicate.save(
            str(output_file)
        )

    asyncio.run(_generate())

    return output_file


def audio_duration(audio_file):

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
                str(audio_file)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:

            return max(
                0.5,
                float(result.stdout.strip())
            )

    except Exception:
        pass

    return 2.5


class GeneratedScene(Scene):

    def construct(self):

        # ====================================================
        # TITLE
        # ====================================================

        title = Text(
            "{safe_title}",
            font_size=32
        )

        title.to_edge(UP)

        self.play(
            Write(title),
            run_time=1.5
        )

        self.wait(0.5)

        current_mobject = None

'''

    # ========================================================
    # STEPS
    # ========================================================

    for step in steps:

        # ----------------------------------------------------
        # Spoken explanation
        # ----------------------------------------------------

        explanation = clean_spoken_text(
            step.explanation
        )

        safe_explanation = (
            explanation
            .replace("\\", "")
            .replace('"', '\\"')
        )

        # ----------------------------------------------------
        # Mathematical expression
        # ----------------------------------------------------

        cleaned_latex = clean_latex(
            step.math_latex
        )

        latex = escape_latex(
            cleaned_latex
        )

        # ----------------------------------------------------
        # Voice file
        # ----------------------------------------------------

        audio_filename = (
            f"step_{step.step_number}.mp3"
        )

        script += f'''
        # ====================================================
        # STEP {step.step_number}
        # ====================================================

        narration = r"{safe_explanation}"

        audio_file = generate_voice(
            narration,
            "{audio_filename}",
            voice="en-NG-EzinneNeural"
        )

        duration = audio_duration(
            audio_file
        )

        next_mobject = MathTex(
            r"{latex}",
            font_size=44
        )

        next_mobject.move_to(ORIGIN)

        # Add narration to the video
        self.add_sound(
            str(audio_file)
        )

        if current_mobject is None:

            self.play(
                Write(next_mobject),
                run_time=duration
            )

        else:

            self.play(
                Transform(
                    current_mobject,
                    next_mobject
                ),
                run_time=duration
            )

        current_mobject = next_mobject

        self.wait(0.2)

'''

    script += '''
        self.wait(1.5)
'''

    return script


# ============================================================
# AI FALLBACK
# ============================================================

def solve_with_ai_fallback(
    prompt: str
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

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt
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

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=0.2
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

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key
            )

            models = [
                "openrouter/free",
                "cohere/north-mini-code:free"
            ]

            for model_id in models:

                try:

                    print(
                        f"Trying {model_id}"
                    )

                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT
                            },
                            {
                                "role": "user",
                                "content": full_prompt
                            }
                        ],
                        temperature=0.2
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
        detail=(
            "All AI providers failed."
        )
    )


# ============================================================
# MANIM RENDER
# ============================================================

def render_manim_script(
    script_path: Path,
    job_id: str
) -> Path:

    manim_cmd = [
        "manim",
        "-ql",
        "--disable_caching",
        "--media_dir",
        str(MEDIA_DIR),
        str(script_path),
        "GeneratedScene"
    ]

    print(
        "Starting Manim render:"
    )

    print(
        " ".join(
            str(x)
            for x in manim_cmd
        )
    )

    try:

        result = subprocess.run(
            manim_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600
        )

        print(result.stdout)

    except subprocess.TimeoutExpired:

        raise HTTPException(
            status_code=504,
            detail=(
                "Manim rendering timed out."
            )
        )

    except subprocess.CalledProcessError as err:

        error_msg = (
            err.stderr
            or err.stdout
            or str(err)
        )

        print(
            "========================================"
        )

        print(
            f"MANIM RENDERING FAILED FOR JOB {job_id}"
        )

        print(error_msg)

        print(
            "========================================"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Manim Rendering Error:\n"
                + error_msg[-12000:]
            )
        )

    # ========================================================
    # FIND VIDEO
    # ========================================================

    possible_outputs = list(
        MEDIA_DIR.rglob(
            "GeneratedScene.mp4"
        )
    )

    if not possible_outputs:

        raise HTTPException(
            status_code=500,
            detail=(
                "Manim completed but "
                "GeneratedScene.mp4 was not found."
            )
        )

    source_video = possible_outputs[-1]

    final_video = (
        OUTPUT_DIR /
        f"{job_id}.mp4"
    )

    if final_video.exists():
        final_video.unlink()

    source_video.rename(
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
        "version": "3.0.1",
        "voice_engine": "Edge TTS",
        "voice": "en-NG-EzinneNeural"
    }


# ============================================================
# ENGINE TEST
# ============================================================

@app.get("/engine-test")
def engine_test():

    import sys

    result = {
        "status": "ok",
        "python": sys.version
    }

    # --------------------------------------------------------
    # Manim
    # --------------------------------------------------------

    try:

        manim_result = subprocess.run(
            ["manim", "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )

        result["manim"] = (
            manim_result.stdout.strip()
            or manim_result.stderr.strip()
        )

    except Exception as e:

        result["manim_error"] = str(e)

    # --------------------------------------------------------
    # Edge TTS
    # --------------------------------------------------------

    try:

        import edge_tts

        result["edge_tts"] = (
            edge_tts.__version__
        )

    except Exception as e:

        result["edge_tts_error"] = str(e)

    # --------------------------------------------------------
    # FFprobe
    # --------------------------------------------------------

    try:

        ffprobe = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=30
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
# GENERATE VIDEO
# ============================================================

@app.post("/generate-video")
def generate_math_video(
    req: RenderRequest
):

    job_id = str(
        uuid.uuid4()
    )[:8]

    script_filename = (
        f"temp_{job_id}.py"
    )

    script_path = (
        BASE_DIR /
        script_filename
    )

    try:

        # ====================================================
        # DIRECT SOLUTION STEPS
        # ====================================================

        if (
            req.solution_steps
            and len(req.solution_steps) > 0
        ):

            print(
                f"[{job_id}] "
                "Rendering directly from solution steps."
            )

            generated_code = (
                build_direct_manim_script(
                    req.prompt,
                    req.solution_steps,
                    job_id
                )
            )

            used_provider = (
                "Direct Script + Edge TTS"
            )

        # ====================================================
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
        # VALIDATE
        # ====================================================

        if not generated_code.strip():

            raise HTTPException(
                status_code=500,
                detail="Generated script is empty."
            )

        if "class GeneratedScene" not in generated_code:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Generated script does not contain "
                    "GeneratedScene."
                )
            )

        # Prevent AI from reintroducing
        # the old broken voiceover package.
        if "manim_voiceover" in generated_code:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Generated script attempted to use "
                    "manim_voiceover. The current engine "
                    "uses direct Edge TTS."
                )
            )

        # ====================================================
        # SAVE
        # ====================================================

        script_path.write_text(
            generated_code,
            encoding="utf-8"
        )

        print(
            f"[{job_id}] "
            f"Script saved to {script_path}"
        )

        # ====================================================
        # RENDER
        # ====================================================

        final_video = render_manim_script(
            script_path,
            job_id
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
            )
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
            detail=str(e)
        )

    finally:

        # Remove temporary Python file
        try:

            if script_path.exists():
                script_path.unlink()

        except Exception as e:

            print(
                f"Could not remove temp script: {e}"
            )
````
