import os
import subprocess
import uuid
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# AI Provider SDKs (Fallback only)
from google import genai
from groq import Groq
from openai import OpenAI

app = FastAPI(title="Tezla Animator - Direct & AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OS_OUTPUT_DIR = "rendered_videos"
os.makedirs(OS_OUTPUT_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=OS_OUTPUT_DIR), name="videos")

class SolutionStep(BaseModel):
    step_number: int
    math_latex: str
    explanation: str

class RenderRequest(BaseModel):
    prompt: str
    solution_steps: Optional[List[SolutionStep]] = None

SYSTEM_PROMPT = """
You are a Manim Python script generator for educational videos.
Your output MUST contain ONLY valid Python code using the Manim Community library.
Do NOT include markdown formatting or backticks (e.g., no ```python).

Rules:
1. Import statements MUST include:
   from manim import *
   from manim_voiceover import VoiceoverScene
   from manim_voiceover.services.edge_tts import EdgeTTSService

2. Define a single scene class named `GeneratedScene(VoiceoverScene)`.
3. In `construct(self)`:
   - Initialize voice: `self.set_speech_service(EdgeTTSService(voice="en-NG-EzinneNeural"))`
   - Wrap visual animations in voiceover blocks:
     with self.voiceover(text=r"Explanation text here...") as tracker:
         self.play(Write(eq), run_time=tracker.duration)

4. Use raw strings `r"..."` for all MathTex expressions to prevent unescaped backslash crashes.
"""

def clean_code_block(code_text: str) -> str:
    cleaned = re.sub(r"^```(?:python)?", "", code_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    return cleaned.strip()

def build_direct_manim_script(prompt: str, steps: List[SolutionStep]) -> str:
    """Generates pure Python Manim code directly from steps without calling any AI model."""
    
    # Strip backslashes and quotes from title string to prevent syntax errors
    clean_prompt = prompt.replace('\\', '').replace('"', "'").replace('\n', ' ')[:40]

    script = f'''from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.edge_tts import EdgeTTSService

class GeneratedScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(EdgeTTSService(voice="en-NG-EzinneNeural"))
        
        # Display Title / Problem Prompt
        title = Text(r"{clean_prompt}", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        current_mobject = None
'''

    for step in steps:
        # Clean LaTeX commands out of spoken explanation text so EdgeTTS speaks naturally
        clean_explanation = re.sub(r'\\[a-zA-Z]+|\$|\{|\}', '', step.explanation).replace('"', "'").replace('\n', ' ')
        escaped_latex = step.math_latex.replace('"', '\\"')
        
        script += f'''
        # Step {step.step_number}
        next_mobject = MathTex(r"{escaped_latex}", font_size=44)
        
        with self.voiceover(text=r"{clean_explanation}") as tracker:
            if current_mobject is None:
                self.play(Write(next_mobject), run_time=max(1.5, tracker.duration))
            else:
                self.play(Transform(current_mobject, next_mobject), run_time=max(1.5, tracker.duration))
        
        if current_mobject is None:
            current_mobject = next_mobject
            
        self.wait(0.5)
'''

    script += '''
        self.wait(1)
'''
    return script

def solve_with_ai_fallback(prompt: str) -> str:
    full_prompt = f"{SYSTEM_PROMPT}\n\nGenerate a narrated Manim scene for: {prompt}"

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt)
            if response.text:
                return clean_code_block(response.text)
        except Exception as e:
            print(f"Gemini fallback failed: {str(e)}")

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": full_prompt}],
                temperature=0.2
            )
            if response.choices[0].message.content:
                return clean_code_block(response.choices[0].message.content)
        except Exception as e:
            print(f"Groq fallback failed: {str(e)}")

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        client = OpenAI(base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)", api_key=openrouter_key)
        for model_id in ["openrouter/free", "cohere/north-mini-code:free"]:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": full_prompt}],
                    temperature=0.2
                )
                if response.choices[0].message.content:
                    return clean_code_block(response.choices[0].message.content)
            except Exception as e:
                print(f"OpenRouter model {model_id} failed: {str(e)}")

    raise HTTPException(status_code=500, detail="No solution steps provided and AI providers failed.")

@app.get("/")
def health_check():
    return {"status": "online", "service": "Tezla Animator Engine", "version": "2.0.4"}

@app.post("/generate-video")
def generate_math_video(req: RenderRequest):
    job_id = str(uuid.uuid4())[:8]
    script_filename = f"temp_{job_id}.py"

    if req.solution_steps and len(req.solution_steps) > 0:
        print(f"Rendering job {job_id} directly from provided solution steps...")
        generated_code = build_direct_manim_script(req.prompt, req.solution_steps)
        used_provider = "Direct Script (No AI)"
    else:
        print(f"No solution steps provided for job {job_id}. Falling back to AI...")
        generated_code = solve_with_ai_fallback(req.prompt)
        used_provider = "AI Fallback Chain"

    with open(script_filename, "w", encoding="utf-8") as f:
        f.write(generated_code)

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
        print(f"MANIM RENDERING FAILED FOR JOB {job_id}:\n{error_msg}")
        raise HTTPException(status_code=400, detail=f"Manim Rendering Error: {error_msg}")
    finally:
        if os.path.exists(script_filename):
            os.remove(script_filename)

    return {
        "status": "success",
        "job_id": job_id,
        "provider_used": used_provider,
        "video_url": f"/videos/{job_id}.mp4"
    }
