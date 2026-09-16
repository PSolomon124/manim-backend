import os
import subprocess
import tempfile
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import edge_tts

app = FastAPI(title="Enhanced Manim Diagram & Video Renderer")

class AnimationRequest(BaseModel):
    title: str = "Mathematical Diagram"
    steps: list[str] = [
        "Initialize node parameters and connections",
        "Process forward propagation through layers",
        "Calculate loss and optimize target weights"
    ]
    voice: str = "en-US-ChristopherNeural"

def generate_tts_sync(text: str, output_path: str, voice: str):
    """Generates audio file from text using edge_tts synchronously."""
    async def _main():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
    asyncio.run(_main())

def build_diagram_manim_script(title: str, steps: list[str], temp_dir: str) -> str:
    """Creates a Manim script with geometric diagrams, animated graph nodes, and 1080p high-quality styling."""
    script_path = os.path.join(temp_dir, "render_scene.py")
    
    # Escape single quotes in text input
    clean_title = title.replace("'", "\\'")
    clean_steps = [s.replace("'", "\\'") for s in steps]

    code = f"""
import os
import asyncio
import subprocess
from manim import *
import edge_tts

def get_audio_duration(file_path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return float(result.stdout.strip())

def make_audio(text, filename, voice="{request_voice_placeholder}"):
    async def _gen():
        c = edge_tts.Communicate(text, voice)
        await c.save(filename)
    asyncio.run(_gen())

class DiagramScene(Scene):
    def construct(self):
        # High Quality Background Setup
        self.camera.background_color = "#0f172a"  # Dark slate background
        
        # Header Title
        title_mobject = Text("{clean_title}", font_size=40, color=BLUE_B, weight=BOLD)
        title_mobject.to_edge(UP, buff=0.5)
        
        title_audio = os.path.join("{temp_dir}", "title.mp3")
        make_audio("{clean_title}", title_audio)
        title_duration = get_audio_duration(title_audio)
        
        self.add_sound(title_audio)
        self.play(Write(title_mobject), run_time=min(1.5, title_duration))
        if title_duration > 1.5:
            self.wait(title_duration - 1.5)
            
        # Create Diagram Elements (Connected Network Nodes)
        node_a = Circle(radius=0.6, color=TEAL, fill_opacity=0.3).shift(LEFT * 4 + DOWN * 0.5)
        node_b = Square(side_length=1.2, color=PURPLE, fill_opacity=0.3).shift(DOWN * 0.5)
        node_c = Triangle(color=ORANGE, fill_opacity=0.3).scale(0.8).shift(RIGHT * 4 + DOWN * 0.5)
        
        label_a = Text("Input", font_size=20).move_to(node_a.get_center())
        label_b = Text("Process", font_size=20).move_to(node_b.get_center())
        label_c = Text("Output", font_size=20).move_to(node_c.get_center())
        
        arrow_1 = Arrow(node_a.get_right(), node_b.get_left(), buff=0.2, color=GRAY_A)
        arrow_2 = Arrow(node_b.get_right(), node_c.get_left(), buff=0.2, color=GRAY_A)
        
        diagram_group = VGroup(node_a, label_a, arrow_1, node_b, label_b, arrow_2, node_c, label_c)
        
        # Display Step Content dynamically with diagram highlights
        steps_data = {clean_steps}
        nodes = [(node_a, label_a), (node_b, label_b), (node_c, label_c)]
        
        self.play(Create(diagram_group), run_time=1.5)
        
        for idx, text in enumerate(steps_data):
            audio_file = os.path.join("{temp_dir}", f"step_{{idx}}.mp3")
            make_audio(text, audio_file)
            duration = get_audio_duration(audio_file)
            
            # Step Card Box
            card = Rectangle(height=1.2, width=11, color=BLUE_E, fill_color="#1e293b", fill_opacity=0.9)
            card.next_to(title_mobject, DOWN, buff=0.4)
            
            step_text = Text(f"Step {{idx+1}}: {{text}}", font_size=22, word_wrap=60, color=WHITE)
            step_text.move_to(card.get_center())
            
            step_group = VGroup(card, step_text)
            
            # Target Highlight Active Diagram Node
            active_node, active_label = nodes[idx % len(nodes)]
            
            self.add_sound(audio_file)
            self.play(
                FadeIn(step_group, shift=UP * 0.3),
                active_node.animate.set_fill(opacity=0.8).scale(1.15),
                active_label.animate.set_color(YELLOW),
                run_time=0.8
            )
            
            remaining_time = max(0.1, duration - 0.8)
            self.wait(remaining_time)
            
            self.play(
                FadeOut(step_group),
                active_node.animate.set_fill(opacity=0.3).scale(1/1.15),
                active_label.animate.set_color(WHITE),
                run_time=0.4
            )
            
        self.play(FadeOut(diagram_group), FadeOut(title_mobject), run_time=1.0)
""".replace("{request_voice_placeholder}", voice)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
        
    return script_path

@app.post("/generate-video")
def generate_video(request: AnimationRequest):
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = build_diagram_manim_script(request.title, request.steps, temp_dir)
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # High-Quality Rendering Configuration flags:
        # -qh: High quality (1080p, 60 fps)
        # --fps 60: Smooth animation frame rate
        cmd = [
            "manim",
            "-qh",
            "--fps", "60",
            "--media_dir", output_dir,
            script_path,
            "DiagramScene"
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Manim execution error: {result.stderr}")
            
        # Locate generated MP4 file
        rendered_files = list(Path(output_dir).rglob("DiagramScene.mp4"))
        if not rendered_files:
            raise HTTPException(status_code=500, detail="Output video file not found.")
            
        final_video_path = str(rendered_files[0])
        
        # Copy to a persistent path outside temp dir to stream back
        export_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(final_video_path, "rb") as src, open(export_path, "wb") as dst:
            dst.write(src.read())
            
        return FileResponse(export_path, media_type="video/mp4", filename="diagram_animation.mp4")
