# storyteller_colab_final.py

# Install dependencies in Colab
!pip install -q diffusers transformers accelerate safetensors imageio imageio-ffmpeg

from huggingface_hub import notebook_login
from diffusers import DiffusionPipeline
import torch
import imageio
import numpy as np
from IPython.display import HTML, display
from base64 import b64encode
import re
import os

# --------------------
# 1. Hugging Face Login
# --------------------
notebook_login()

# --------------------
# 2. Story Functions
# --------------------
def get_story_from_user():
    print("Please enter your story. Press Enter twice to finish:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

def segment_story(story):
    sentence_pattern = r'[.!?]\s+'
    scenes = re.split(sentence_pattern, story)
    cleaned_scenes = []
    for scene in scenes:
        scene = scene.strip()
        if scene and len(scene) > 5:
            if not scene.endswith(('.', '!', '?')):
                scene += '.'
            cleaned_scenes.append(scene)
    return cleaned_scenes

def generate_video_prompts(scenes):
    prompts = []
    for i, scene in enumerate(scenes, 1):
        prompt = f"Scene {i}: {scene}"
        enhanced_prompt = f"{prompt}, cinematic lighting, high quality, detailed, smooth camera movement"
        prompts.append({
            'scene_number': i,
            'original_text': scene,
            'video_prompt': enhanced_prompt
        })
    return prompts

# --------------------
# 3. Video Generation Functions
# --------------------
def load_t2v_model(model_id="damo-vilab/text-to-video-ms-1.7b", device="cuda"):
    print("Loading text-to-video model...")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    print("Model loaded.")
    return pipe

def generate_video(pipe, prompt, output_path, num_steps=25, fps=8):
    print(f"Generating video for: {prompt}")
    video_frames = pipe(prompt, num_inference_steps=num_steps).frames
    video_frames_np = video_frames[0]
    if video_frames_np.dtype != np.uint8:
        video_frames_np = (video_frames_np * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(output_path, list(video_frames_np), fps=fps)
    print(f"Video saved to {output_path}")

# --------------------
# 4. Display All Videos at End
# --------------------
def display_all_videos(scene_count):
    for i in range(1, scene_count + 1):
        file_path = f"/content/scene_{i}.mp4"
        if os.path.exists(file_path):
            mp4 = open(file_path, 'rb').read()
            data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
            display(HTML(f"""<h3>Scene {i}</h3><video width=512 controls><source src="{data_url}" type="video/mp4"></video>"""))

# --------------------
# 5. Main Flow
# --------------------
if __name__ == "__main__":
    # Step 1: Story input
    story = get_story_from_user()
    print("\nYour story is:\n", story)

    # Step 2: Segment into scenes
    scenes = segment_story(story)
    print(f"\nStory segmented into {len(scenes)} scenes.")

    # Step 3: Generate prompts
    prompts = generate_video_prompts(scenes)
    for p in prompts:
        print(f"\nScene {p['scene_number']} Prompt: {p['video_prompt']}")

    # Step 4: Load model once
    model = load_t2v_model()

    # Step 5: Generate videos for each scene
    for p in prompts:
        output_file = f"/content/scene_{p['scene_number']}.mp4"
        generate_video(model, p['video_prompt'], output_file, num_steps=25, fps=8)

    # Step 6: Display all videos together
    print("\nAll scene videos generated. Displaying below...")
    display_all_videos(len(prompts))
