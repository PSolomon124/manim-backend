import os
import subprocess
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Tezla Animator - Manim Engine")

# Setup directories for static video serving
OS_OUTPUT_DIR = "rendered_videos"
os.makedirs(OS_OUTPUT_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=OS_OUTPUT_DIR), name="videos")

# OpenAI API Initialization
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RenderRequest(BaseModel):
    prompt: str  # e.g., "Solve 2x + 5 = 13 step-by-step with voice narration"

# Strictly enforce valid imports, VoiceoverScene inheritance, and escaping
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

4. Use raw strings `r"..."` for all MathTex expressions to avoid unescaped backslash crashes.
"""

@app.post("/generate-video")
def generate_math_video(req: RenderRequest):
    job_id = str(uuid.uuid4())[:8]
    script_filename = f"temp_{job_id}.py"

    # Step 1: Request Manim Python Code from OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate a narrated Manim scene for: {req.prompt}"}
            ],
            temperature=0.2
        )
        generated_code = response.choices[0].message.content.strip()

        # Clean markdown code block artifacts
        if generated_code.startswith("```"):
            generated_code = generated_code.split("\n", 1)[1].rsplit("```", 1)[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Code Generation Failed: {str(e)}")

    # Step 2: Write script to file
    with open(script_filename, "w", encoding="utf-8") as f:
        f.write(generated_code)

    # Step 3: Execute Manim via CLI
    video_output_path = os.path.join(OS_OUTPUT_DIR, f"{job_id}.mp4")

    manim_cmd = [
        "manim",
        "-ql",  # Low quality (480p) for fast server rendering
        "--media_dir", "./media",
        script_filename,
        "GeneratedScene"
    ]

    try:
        subprocess.run(manim_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Fix: Manim outputs the file with the Class Name ("GeneratedScene.mp4")
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
        # Cleanup temporary Python file
        if os.path.exists(script_filename):
            os.remove(script_filename)

    # Step 4: Return Video Path URL
    return {
        "status": "success",
        "job_id": job_id,
        "video_url": f"/videos/{job_id}.mp4"
    }
