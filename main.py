from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import uuid
import os

app = FastAPI()

# Allow Lovable frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MathRequest(BaseModel):
    prompt: str

@app.post("/api/generate-math-video")
async def generate_video(req: MathRequest):
    job_id = str(uuid.uuid4())[:8]
    
    # 1. Ask your LLM (Gemini/OpenAI) to write the Manim script based on req.prompt
    # 2. Save script as scene_{job_id}.py
    # 3. Compile: subprocess.run(f"manim -pql scene_{job_id}.py MathScene -o {job_id}.mp4")
    # 4. Upload generated .mp4 to Supabase Storage or S3
    
    real_video_url = f"https://your-storage-bucket.com/renders/{job_id}.mp4"
    
    return {
        "video_url": real_video_url,
        "latex_steps": ["Step 1: Parse equation", "Step 2: Plot parabola"]
    }
