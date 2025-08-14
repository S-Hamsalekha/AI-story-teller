# storyteller.py

!pip install -q diffusers transformers accelerate safetensors imageio imageio-ffmpeg

import os
import re
import shutil
import torch
import imageio
import numpy as np
from huggingface_hub import notebook_login
from diffusers import DiffusionPipeline
from IPython.display import HTML, display
from base64 import b64encode
from PIL import Image

# --------------------
# Compile-time config
# --------------------
ENABLE_T2I_ANIMATION = False  # Set False to disable animation in T2I mode

# --------------------
# Hugging Face Login
# --------------------
notebook_login()

# --------------------
# Utility: Clear HF cache
# --------------------
def clear_hf_cache():
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        print("Clearing Hugging Face cache to free space...")
        shutil.rmtree(cache_dir)
        print("Cache cleared.")
    else:
        print("No cache to clear.")

# --------------------
# Story functions
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

def generate_prompts(scenes):
    prompts = []
    for i, scene in enumerate(scenes, 1):
        prompt = f"Scene {i}: {scene}"
        enhanced_prompt = f"{prompt}, cinematic lighting, high quality, detailed"
        prompts.append({
            'scene_number': i,
            'original_text': scene,
            'enhanced_prompt': enhanced_prompt
        })
    return prompts

# --------------------
# Model loading
# --------------------
def load_video_model(model_id="damo-vilab/text-to-video-ms-1.7b", device="cuda"):
    clear_hf_cache()
    print(f"Loading video model: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    print("Video model loaded.")
    return pipe

def choose_image_model():
    print("\nChoose an image model:")
    print("1. FLUX.1-schnell (~8GB, fast Flux)")
    print("2. Stable Diffusion XL (~6GB)")
    print("3. Stable Diffusion v1.5 (~4GB, fastest)")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "1":
        return "black-forest-labs/FLUX.1-schnell"
    elif choice == "2":
        return "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        return "runwayml/stable-diffusion-v1-5"

def load_image_model(model_id, device="cuda"):
    clear_hf_cache()
    print(f"Loading image model: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device)
    print("Image model loaded.")
    return pipe

def load_animation_model(model_id="cerspense/zeroscope_v2_576w", device="cuda"):
    clear_hf_cache()
    print(f"Loading lightweight animation model: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device)
    print("Animation model loaded.")
    return pipe

# --------------------
# Generation functions
# --------------------
def generate_video(pipe, prompt, output_path, num_steps=25, fps=8):
    print(f"Generating video for: {prompt}")
    video_frames = pipe(prompt, num_inference_steps=num_steps).frames
    video_frames_np = video_frames[0]
    if video_frames_np.dtype != np.uint8:
        video_frames_np = (video_frames_np * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(output_path, list(video_frames_np), fps=fps)
    print(f"Video saved to {output_path}")

def generate_image(pipe, prompt, output_path):
    print(f"Generating image for: {prompt}")
    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"Image saved to {output_path}")

def animate_image_with_model(pipe, image_path, output_path, num_frames=14, fps=8):
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    print(f"Animating {image_path} → {output_path}")
    frames = pipe(image, num_frames=num_frames).frames[0]
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(output_path, list(frames), fps=fps)
    print(f"Video saved to {output_path}")

# --------------------
# Display function
# --------------------
def display_all_files(scene_count, mode="video"):
    for i in range(1, scene_count + 1):
        file_path = f"/content/scene_{i}.{'mp4' if mode=='video' else 'png'}"
        if os.path.exists(file_path):
            if mode == "video":
                with open(file_path, 'rb') as f:
                    mp4 = f.read()
                data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
                display(HTML(f"""<h3>Scene {i}</h3>
                                 <video width=512 controls>
                                     <source src="{data_url}" type="video/mp4">
                                 </video>"""))
            else:
                with open(file_path, 'rb') as f:
                    img_bytes = f.read()
                img_b64 = b64encode(img_bytes).decode()
                display(HTML(f"<h3>Scene {i}</h3><img src='data:image/png;base64,{img_b64}' width='512'>"))

# --------------------
# Public model loading functions
# --------------------
def load_t2v_model():
    return load_video_model()

def load_t2i_models():
    model_id = choose_image_model()
    t2i_model = load_image_model(model_id)
    anim_model = None
    if ENABLE_T2I_ANIMATION:
        anim_model = load_animation_model()
    return t2i_model, anim_model

# --------------------
# Story runner (reuse loaded models)
# --------------------
def run_story_with_model(mode, model, anim_model=None):
    story = get_story_from_user()
    print("\nYour story is:\n", story)

    scenes = segment_story(story)
    print(f"\nStory segmented into {len(scenes)} scenes.")

    prompts = generate_prompts(scenes)
    for p in prompts:
        print(f"Scene {p['scene_number']} Prompt: {p['enhanced_prompt']}")

    if mode == "T2V":
        for p in prompts:
            output_file = f"/content/scene_{p['scene_number']}.mp4"
            generate_video(model, p['enhanced_prompt'], output_file)
        display_all_files(len(prompts), mode="video")

    elif mode == "T2I":
        for p in prompts:
            img_path = f"/content/scene_{p['scene_number']}.png"
            generate_image(model, p['enhanced_prompt'], img_path)
            if ENABLE_T2I_ANIMATION and anim_model:
                vid_path = f"/content/scene_{p['scene_number']}.mp4"
                animate_image_with_model(anim_model, img_path, vid_path)

        if ENABLE_T2I_ANIMATION and anim_model:
            display_all_files(len(prompts), mode="video")
        else:
            display_all_files(len(prompts), mode="image")


# Step 1 — Choose mode and load models once
mode = "T2V"  # or "T2V"

if mode == "T2V":
    t2v_model = load_t2v_model()
elif mode == "T2I":
    t2i_model, anim_model = load_t2i_models()


# Step 2 — Generate multiple stories without reloading models
run_story_with_model("T2V", t2v_model)  # Story 1
#run_story_with_model("T2I", t2i_model, anim_model)  # Story 2                