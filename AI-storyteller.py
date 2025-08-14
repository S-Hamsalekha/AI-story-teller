!pip install -q diffusers transformers accelerate safetensors imageio imageio-ffmpeg groq

import os
import torch
import imageio
import numpy as np
from base64 import b64encode
from groq import Groq
from diffusers import DiffusionPipeline
from IPython.display import HTML, display
from PIL import Image
from huggingface_hub import notebook_login
import json

# --------------------
# Compile-time config
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENABLE_T2I_ANIMATION = False

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
        import shutil
        shutil.rmtree(cache_dir)

# --------------------
# LLM-based scene segmentation + prompt generation
# --------------------
def generate_scenes_with_prompts_llm(story: str):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = input("Enter your GROQ API key: ").strip()
        os.environ["GROQ_API_KEY"] = api_key

    client = Groq(api_key=api_key)

    prompt_template = f"""
    You are a creative assistant for a text-to-image storytelling system. Your task is to segment a story into a series of distinct visual scenes.

    For each scene, you must provide the following details in a JSON object:
    1.  `scene_number`: (integer) The sequential number of the scene.
    2.  `description`: (string) A 1-2 sentence summary of the key action.
    3.  `prompt`: (string) A detailed, cinematic text prompt for an AI image generator. This prompt should include style, mood, lighting, and camera angles.

    Your response MUST be a single, valid JSON array of these objects. Do not include any conversational text, explanations, or any other content outside of the JSON array. The array should contain between 3 to 6 scenes.

    Story:
    \"\"\"{story}\"\"\"
    """

    print("Calling Groq LLaMA-3 8B for segmentation + prompt generation...")
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt_template}],
        temperature=0.7,
        max_tokens=1200
    )

    content = response.choices[0].message.content
    content = content.strip().lstrip('`json').rstrip('`').strip()

    try:
        scenes = json.loads(content)
        if not isinstance(scenes, list):
            raise ValueError("LLM did not return a JSON list.")
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON. Error: {e}\n\nReceived content:\n{content}") from e

    return scenes

# --------------------
# Model loading
# --------------------
def load_video_model(model_id="damo-vilab/text-to-video-ms-1.7b"):
    clear_hf_cache()
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(DEVICE)
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

def load_image_model(model_id):
    clear_hf_cache()
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(DEVICE)
    return pipe

def load_animation_model(model_id="cerspense/zeroscope_v2_576w"):
    clear_hf_cache()
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(DEVICE)
    return pipe

# --------------------
# Generation functions
# --------------------
def generate_video(pipe, prompt, output_path, num_steps=25, fps=8):
    video_frames = pipe(prompt, num_inference_steps=num_steps).frames
    video_frames_np = video_frames[0]
    if video_frames_np.dtype != np.uint8:
        video_frames_np = (video_frames_np * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(output_path, list(video_frames_np), fps=fps)

def generate_image(pipe, prompt, output_path):
    image = pipe(prompt).images[0]
    image.save(output_path)

def animate_image_with_model(pipe, image_path, output_path, num_frames=14, fps=8):
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    print(f"Animating {image_path} → {output_path}")
    frames = pipe(image, num_frames=num_frames).frames[0]
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(output_path, list(frames), fps=fps)

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
# Story runner (LLM-based)
# --------------------
def run_story(mode, model, anim_model=None):
    story = input("Please enter your story. Press Enter when done:\n")
    try:
        scenes = generate_scenes_with_prompts_llm(story)
        for scene in scenes:
            print(f"Scene {scene['scene_number']} Prompt: {scene['prompt']}")

        if mode == "T2V":
            for scene in scenes:
                output_file = f"/content/scene_{scene['scene_number']}.mp4"
                generate_video(model, scene['prompt'], output_file)
            display_all_files(len(scenes), mode="video")

        elif mode == "T2I":
            for scene in scenes:
                img_path = f"/content/scene_{scene['scene_number']}.png"
                generate_image(model, scene['prompt'], img_path)
                if ENABLE_T2I_ANIMATION and anim_model:
                    vid_path = f"/content/scene_{scene['scene_number']}.mp4"
                    animate_image_with_model(anim_model, img_path, vid_path)

            if ENABLE_T2I_ANIMATION and anim_model:
                display_all_files(len(scenes), mode="video")
            else:
                display_all_files(len(scenes), mode="image")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --------------------
# Sample calls (separate cells)
# --------------------
# Cell 1: Load models
t2i_model = load_image_model(choose_image_model())
anim_model = load_animation_model() if ENABLE_T2I_ANIMATION else None
# OR for video mode:
# t2v_model = load_video_model()


# Cell 2: Run story
run_story("T2I", t2i_model, anim_model)
# run_story("T2V", t2v_model)