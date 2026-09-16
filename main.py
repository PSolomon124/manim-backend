def build_direct_manim_script(prompt: str, steps: List[SolutionStep]) -> str:
    """Generates pure Python Manim code directly from steps without calling any AI model."""
    
    # Clean special LaTeX characters out of titles
    clean_prompt = prompt.replace('\\', '').replace('"', "'").replace('\n', ' ')[:40]

    script = f'''from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.edge import EdgeService

class GeneratedScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(EdgeService(voice="en-NG-EzinneNeural"))
        
        # Display Title / Problem Prompt
        title = Text(r"{clean_prompt}", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        current_mobject = None
'''

    for step in steps:
        # Sanitize speech text by removing LaTeX markup so EdgeTTS speaks plain text
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
