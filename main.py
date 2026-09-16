import os
import subprocess
import uuid
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Provider SDKs
from google import genai
from groq import Groq
from openai import OpenAI

app = FastAPI(title="Tezla Animator - Math Animation Engine")

# Enable CORS for frontend clients like Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static video storage directory setup
OS_OUTPUT_DIR = "rendered_videos"
os.makedirs(OS_OUTPUT_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=OS_OUTPUT_DIR), name="videos")

# Pydantic Schemas for solution steps
class SolutionStep(BaseModel):
    step_number: int
    math_latex: str        # e.g., "2x + 5 = 13"
    explanation: str       # e.g., "Subtract 5 from both sides"

class RenderRequest(BaseModel):
    prompt: str
    solution_steps: Optional[List[SolutionStep]] = None  # Optional pre-solved steps

SYSTEM_PROMPT = """
You are a Manim Python script generator for educational videos.
Your output MUST contain ONLY valid Python code using the Manim Community library.
Do NOT include markdown formatting or backticks (e.g., no ```python).

Rules:
1. Import statements MUST include:
   from manim import *
   from manim_voiceover import VoiceoverScene
   from manim_voiceover.services.edge import EdgeService

2. Define a single scene class named `GeneratedScene(VoiceoverScene)`.
3. In `construct(self)`:
   - Initialize voice: `self.set_speech_service(EdgeService(voice="en-NG-EzinneNeural"))`
   - Wrap visual animations in voiceover blocks:
     with self.voiceover(text="Explanation text here...") as tracker:
         self.play(Write(eq), run_time=tracker.duration)

4. Use raw strings `r"..."` for all MathTex expressions to prevent unescaped backslash crashes.
5. If explicit step-by-step resolution data is provided, follow its order EXACTLY.
"""

def clean_code_block(code_text: str) -> str:
    """Strips markdown python block wrappers if returned by AI models."""
    cleaned = re.sub(r"^```(?:python)?", "", code_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    return cleaned.strip()

def generate_code_with_fallback(req: RenderRequest) -> tuple[str, str]:
    """Generates code via fallback chain: Gemini -> Groq -> OpenRouter Free Models"""
    
    # Format explicit solution steps into prompt if present
    formatted_solution = ""
    if req.solution_steps and len(req.solution_steps) > 0:
        formatted_solution = "\nEXACT STEP-BY-STEP SOLUTION TO ANIMATE:\n"
        for step in req.solution_steps:
            formatted_solution += f"Step {step.step_number}: {step.explanation} | LaTeX: {step.math_latex}\n"

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Topic / Prompt: {req.prompt}\n"
        f"{formatted_solution}\n"
        f"INSTRUCTION: Create a Manim VoiceoverScene animating the mathematical steps provided above."
    )

    # --- TIER 1: GEMINI FREE TIER ---
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt
            )
            if response.text:
                print("Generated code via Gemini Free Tier")
                return clean_code_block(response.text), "Gemini"
        except Exception as e:
            print(f"Gemini Free Tier failed/exhausted ({str(e)}). Switching to Groq...")

    # --- TIER 2: GROQ FREE TIER ---
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.2
            )
            content = response.choices[0].message.content
            if content:
                print("Generated code via Groq Free Tier")
                return clean_code_block(content), "Groq"
        except Exception as e:
            print(f"Groq Free Tier failed/exhausted ({str(e)}). Switching to OpenRouter...")

    # --- TIER 3: OPENROUTER FREE ROUTER & MODELS ---
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        # OpenRouter's auto-router plus fallback free models
        free_models = [
            "openrouter/free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "cohere/north-mini-code:free"
        ]
        
        client = OpenAI(
            base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)",
            api_key=openrouter_key,
        )

        for model_id in free_models:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.2
                )
                content = response.choices[0].message.content
                if content:
                    print(f"Generated code via OpenRouter model: {model_id}")
                    return clean_code_block(content), f"OpenRouter ({model_id})"
            except Exception as e:
                print(f"OpenRouter model {model_id} failed: {str(e)}. Trying next model...")

    raise HTTPException(
        status_code=500,
        detail="All free API providers (Gemini, Groq, OpenRouter) are temporarily exhausted."
    )

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Tezla Animator - Manim Engine",
        "version": "1.1.0"
    }

@app.post("/generate-video")
def generate_math_video(req: RenderRequest):
    job_id = str(uuid.uuid4())[:8]
    script_filename = f"temp_{job_id}.py"

    # Step 1: Request Manim Python Code
    generated_code, used_provider = generate_code_with_fallback(req)

    # Step 2: Write script to file
    with open(script_filename, "w", encoding="utf-8") as f:
        f.write(generated_code)

    # Step 3: Render Video via Manim CLI
    video_output_path = os.path.join(OS_OUTPUT_DIR, f"{job_id}.mp4")

    manim_cmd = [
        "manim",
        "-ql",
        "--media_dir", "./media",
        script_filename,
        "GeneratedScene"
    ]

    try:
        subprocess.run(manim_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        expected_manim_output = os.path.join("media", "videos", f"temp_{job_id}", "480p15", "GeneratedScene.mp4")

        if os.path.exists(expected_manim_output):
            os.rename(expected_manim_output, video_output_path)
        else:
            raise FileNotFoundError(f"Render output missing at {expected_manim_output}")

    except subprocess.CalledProcessError as err:
        error_msg = err.stderr.decode("utf-8") if err.stderr else str(err)
        raise HTTPException(
            status_code=400,
            detail=f"Manim Rendering Error: {error_msg}"
        )
    finally:
        if os.path.exists(script_filename):
            os.remove(script_filename)

    # Step 4: Return Video Path URL and Provider Metadata
    return {
        "status": "success",
        "job_id": job_id,
        "provider_used": used_provider,
        "video_url": f"/videos/{job_id}.mp4"
    }
