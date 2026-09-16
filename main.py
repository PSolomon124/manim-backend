import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# AI PROVIDERS
from google import genai
from groq import Groq
from openai import OpenAI


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="Tezla Animator - Direct & AI Engine",
    version="2.1.0"
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/videos",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="videos"
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


# ============================================================
# AI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are a Manim Python script generator for educational mathematics
and science videos.

Your output MUST contain ONLY valid Python code.

DO NOT use Markdown.
DO NOT use ```python.
DO NOT explain the code outside the Python script.

The generated code MUST:

1. Import:

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.edge_tts import EdgeTTSService

2. Define exactly one scene:

class GeneratedScene(VoiceoverScene):

3. Inside construct():

self.set_speech_service(
    EdgeTTSService(voice="en-NG-EzinneNeural")
)

4. Use voiceover blocks for narration.

Example:

with self.voiceover(text="Explanation") as tracker:
    self.play(
        Write(equation),
        run_time=max(1.5, tracker.duration)
    )

5. Use raw strings for LaTeX:

MathTex(r"x^2 + 2x + 1")

6. Avoid unsupported Manim APIs.

7. Make the animation educational, readable and visually clean.

8. Keep mathematical notation correct.

9. Do not put LaTeX commands directly into spoken narration.
"""


# ============================================================
# CLEAN AI OUTPUT
# ============================================================

def clean_code_block(code_text: str) -> str:
    """
    Removes Markdown code fences if an AI model accidentally
    returns them.
    """

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
    Converts mathematical explanation into speech-friendly text.
    """

    text = text.replace("\\", "")
    text = text.replace("$", "")
    text = text.replace("{", "")
    text = text.replace("}", "")

    text = text.replace('"', "'")
    text = text.replace("\n", " ")

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_title(text: str) -> str:
    """
    Makes the user's prompt safe for Manim Text().
    """

    text = text.replace("\\", "")
    text = text.replace('"', "'")
    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    return text[:80]


def escape_latex(latex: str) -> str:
    """
    Escapes quotes while preserving LaTeX backslashes.
    """

    return latex.replace("\\", "\\\\").replace('"', '\\"')


# ============================================================
# DIRECT MANIM SCRIPT GENERATOR
# ============================================================

def build_direct_manim_script(
    prompt: str,
    steps: List[SolutionStep]
) -> str:

    clean_prompt = clean_title(prompt)

    script = f'''from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.edge_tts import EdgeTTSService


class GeneratedScene(VoiceoverScene):

    def construct(self):

        self.set_speech_service(
            EdgeTTSService(
                voice="en-NG-EzinneNeural"
            )
        )

        # ====================================================
        # TITLE
        # ====================================================

        title = Text(
            r"{clean_prompt}",
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

        explanation = clean_spoken_text(
            step.explanation
        )

        latex = escape_latex(
            step.math_latex
        )

        script += f'''
        # ====================================================
        # STEP {step.step_number}
        # ====================================================

        next_mobject = MathTex(
            r"{latex}",
            font_size=44
        )

        next_mobject.move_to(ORIGIN)

        with self.voiceover(
            text=r"{explanation}"
        ) as tracker:

            if current_mobject is None:

                self.play(
                    Write(next_mobject),
                    run_time=max(
                        1.5,
                        tracker.duration
                    )
                )

            else:

                self.play(
                    Transform(
                        current_mobject,
                        next_mobject
                    ),
                    run_time=max(
                        1.5,
                        tracker.duration
                    )
                )

        current_mobject = next_mobject

        self.wait(0.4)

'''

    script += '''
        self.wait(1.5)
'''

    return script


# ============================================================
# AI FALLBACK
# ============================================================

def solve_with_ai_fallback(prompt: str) -> str:

    full_prompt = f"""
{SYSTEM_PROMPT}

Generate a narrated educational Manim scene for:

{prompt}
"""

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    gemini_key = os.getenv("GEMINI_API_KEY")

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

                print("Gemini generated the script.")

                return clean_code_block(
                    response.text
                )

        except Exception as e:

            print(
                f"Gemini fallback failed: {e}"
            )

    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    groq_key = os.getenv("GROQ_API_KEY")

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

            content = response.choices[0].message.content

            if content:

                print("Groq generated the script.")

                return clean_code_block(
                    content
                )

        except Exception as e:

            print(
                f"Groq fallback failed: {e}"
            )

    # --------------------------------------------------------
    # OPENROUTER
    # --------------------------------------------------------

    openrouter_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if openrouter_key:

        try:

            print("Trying OpenRouter...")

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
                        f"Trying OpenRouter model: {model_id}"
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

                        print(
                            f"OpenRouter succeeded with {model_id}"
                        )

                        return clean_code_block(
                            content
                        )

                except Exception as e:

                    print(
                        f"OpenRouter model "
                        f"{model_id} failed: {e}"
                    )

        except Exception as e:

            print(
                f"OpenRouter initialization failed: {e}"
            )

    raise HTTPException(
        status_code=500,
        detail=(
            "No solution steps were provided and "
            "all AI providers failed."
        )
    )


# ============================================================
# MANIM RENDER FUNCTION
# ============================================================

def render_manim_script(
    script_path: Path,
    job_id: str
) -> Path:

    manim_cmd = [
        "manim",
        "-ql",
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
                "Manim rendering timed out "
                "after 10 minutes."
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
                + error_msg[-10000:]
            )
        )

    # --------------------------------------------------------
    # Find GeneratedScene.mp4
    # --------------------------------------------------------

    possible_outputs = list(
        MEDIA_DIR.rglob(
            "GeneratedScene.mp4"
        )
    )

    if not possible_outputs:

        raise HTTPException(
            status_code=500,
            detail=(
                "Manim finished but "
                "GeneratedScene.mp4 was not found."
            )
        )

    source_video = possible_outputs[-1]

    final_video = (
        OUTPUT_DIR /
        f"{job_id}.mp4"
    )

    # Remove old output if it exists
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
        "version": "2.1.0",
        "manim": "0.21.0",
        "voiceover": "0.3.7",
        "tts": "Edge TTS"
    }


# ============================================================
# ENGINE TEST
# ============================================================

@app.get("/engine-test")
def engine_test():

    try:

        manim_result = subprocess.run(
            [
                "manim",
                "--version"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        return {
            "status": "ok",
            "manim_version": manim_result.stdout.strip(),
            "python_version": (
                subprocess
                .run(
                    [
                        "python",
                        "--version"
                    ],
                    capture_output=True,
                    text=True
                )
                .stdout.strip()
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# VIDEO GENERATION ENDPOINT
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
        # GENERATE SCRIPT
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
                    req.solution_steps
                )
            )

            used_provider = (
                "Direct Script (No AI)"
            )

        else:

            print(
                f"[{job_id}] "
                "No solution steps. "
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
        # BASIC SCRIPT VALIDATION
        # ====================================================

        if not generated_code.strip():

            raise HTTPException(
                status_code=500,
                detail="Generated Python script is empty."
            )

        # Ensure GeneratedScene exists
        if "class GeneratedScene" not in generated_code:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Generated script does not contain "
                    "GeneratedScene."
                )
            )

        # ====================================================
        # SAVE SCRIPT
        # ====================================================

        script_path.write_text(
            generated_code,
            encoding="utf-8"
        )

        print(
            f"[{job_id}] Script saved to "
            f"{script_path}"
        )

        # ====================================================
        # RENDER
        # ====================================================

        final_video = render_manim_script(
            script_path,
            job_id
        )

        # ====================================================
        # RESPONSE
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
            f"[{job_id}] Unexpected error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # ====================================================
        # CLEAN TEMP PYTHON SCRIPT
        # ====================================================

        try:

            if script_path.exists():

                script_path.unlink()

        except Exception as cleanup_error:

            print(
                "Could not remove temporary "
                f"script: {cleanup_error}"
            )
