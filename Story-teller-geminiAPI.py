from kaggle_secrets import UserSecretsClient
import os
!pip install elevenlabs
from elevenlabs import ElevenLabs, save, VoiceSettings


# Narration language: "en" for English, "kn" for Kannada
NARRATION_LANG = "kn"

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GEMINI_API_KEY")  # 👈 fetch your saved secret
os.environ["GEMINI_API_KEY"] = api_key               # 👈 now set as env variable
os.environ["ELEVENLABS_API_KEY"] = user_secrets.get_secret("ELEVENLABS_API_KEY")  # 👈 new

print("✅ API keys loaded")
assert os.getenv("GEMINI_API_KEY"), "Gemini API key missing!"
assert os.getenv("ELEVENLABS_API_KEY"), "ElevenLabs API key missing!"

from google import genai
client = genai.Client()  # reads GEMINI_API_KEY automatically

# quick ping (text) — ensures auth works
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with the word: PONG"
)
print(resp.text)

# ElevenLabs setup
from elevenlabs import ElevenLabs, save
el_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


def translate_to_kannada(text: str) -> str:
    """Translate narration into Kannada using Gemini, returning only plain text."""
    prompt = f"""
Translate the following narration into Kannada.

IMPORTANT:
- Output ONLY the translated Kannada text.
- Do NOT include explanations, extra words, or formatting.

Narration:
{text}
"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # Safely extract text
    if resp and hasattr(resp, "text") and resp.text:
        return resp.text.strip()

    # Fallback: try candidates
    if resp and getattr(resp, "candidates", None):
        for cand in resp.candidates:
            if cand and getattr(cand, "content", None):
                parts = getattr(cand.content, "parts", [])
                for part in parts:
                    if getattr(part, "text", None):
                        return part.text.strip()

    print("⚠️ Gemini translation failed, using original text.")
    return text


def get_voice_id(voice_name: str) -> str:
    """Look up a voice_id by its name (case-insensitive)."""
    voices = el_client.voices.get_all()
    for v in voices.voices:
        if v.name.lower() == voice_name.lower():
            return v.voice_id
    raise ValueError(f"Voice '{voice_name}' not found in your ElevenLabs account.")

def tts_elevenlabs(text: str, out_path: str, voice_name: str = "Monika Sogam", model_id: str = "eleven_multilingual_v2"):
    """Generate narration audio using ElevenLabs TTS by voice name."""
    voice_id = get_voice_id(voice_name)

    audio_generator = el_client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        )
    )
    audio_bytes = b"".join(audio_generator)
    with open(out_path, "wb") as f:
        f.write(audio_bytes)
    return out_path

# ===============================
# Step 2 + Step 3 : Story → Scenes → Images (clean JSON output)
# ===============================

import json, re
from google.genai import types
from PIL import Image
from io import BytesIO

def safe_json_parse(raw_text):
    """Extract valid JSON from Gemini output (handles code fences or extra text)."""
    # remove markdown code fences like ```json ... ```
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    # extract the first JSON-looking structure (array or object)
    m = re.search(r"(\[.*\]|\{.*\})", cleaned, flags=re.DOTALL)
    if m:
        cleaned = m.group(1)
    return json.loads(cleaned)

# 1) Take story input
story = input("Please enter your story:\n")

# 2) Ask Gemini to segment story into scenes (strict JSON only)
segmentation_prompt = f"""
You are a storytelling assistant. 

TASK:
- Break the following story into 3–5 distinct visual scenes.
- Output MUST be a single valid JSON array ONLY.
- Do NOT include explanations, natural language text, or code fences.
- Each object in the array must have:
  - "scene_number" (integer)
  - "description" (1–2 sentences, visual summary of what’s happening)
  - "prompt" (cinematic artistic illustration prompt for image generation)
  - "narration" (1–2 sentences, natural spoken narration text for TTS)

Story:
{story}
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=segmentation_prompt
)

print("\nGemini Output (raw):\n")
print(response.text)

# 3) Safe JSON parse
try:
    scenes = safe_json_parse(response.text)
except Exception as e:
    print("⚠️ Could not parse Gemini output as JSON.")
    print("Error:", e)
    scenes = []

# 4) Generate an image for each scene
image_files = []
for scene in scenes:
    print(f"🎨 Generating Scene {scene['scene_number']}: {scene['description']}")
    
    img_response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=scene['prompt'],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        ),
    )
    
    img = None
    if img_response and img_response.candidates:
        content = img_response.candidates[0].content
        if content and getattr(content, "parts", None):
            for part in content.parts:
                if getattr(part, "inline_data", None):
                    try:
                        img = Image.open(BytesIO(part.inline_data.data))
                    except Exception as e:
                        print(f"⚠️ Failed to decode image for scene {scene['scene_number']}: {e}")
                    break

    if img:
        filename = f"scene_{scene['scene_number']}.png"
        img.save(filename)
        image_files.append(filename)
        display(img)
        print(f"✅ Saved {filename}\n")
    else:
        print(f"⚠️ No image returned for scene {scene['scene_number']}.\n")
# ===============================
# Step 5 : Narration + Zoom + Final Stitch
# ===============================
!pip -q install gTTS moviepy

from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

def tts_gtts(text: str, out_path: str, lang: str = "en"):
    """Generate narration audio with gTTS."""
    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)
    return out_path

def animate_image_panzoom(image_path, duration=5, zoom=0.05):
    """
    Simple Ken Burns effect: slow zoom-in over `duration`.
    Resolution unification happens at final concatenate (compose mode).
    """
    clip = ImageClip(image_path)
    animated = clip.resize(lambda t: 1 + zoom * (t / duration)).set_duration(duration)
    animated = animated.set_position(("center", "center"))
    return animated

scene_clips = []
audio_handles = []   # keep refs to close later

for scene in scenes:
    n = scene["scene_number"]
    img_path = f"scene_{n}.png"
    if not os.path.exists(img_path):
        print(f"⚠️ Missing image for scene {n}: {img_path}. Skipping.")
        continue

    narration_text = scene.get("narration", scene.get("description", f"Scene {n}"))

    # Translate if Kannada requested
    if NARRATION_LANG == "kn":
        print(f"🌐 Translating narration for scene {n} to Kannada …")
        narration_text = translate_to_kannada(narration_text)
        print(f"Narration text :{narration_text}")
        voice_name = "Jessica"   # fallback (later replace with your Kannada voice)
    else:
        voice_name = "Rachel"   # English

    audio_path = f"scene_{n}.mp3"
    print(f"🔊 ElevenLabs TTS for scene {n} using voice '{voice_name}' …")
    tts_elevenlabs(narration_text, audio_path, voice_name=voice_name)  # ✅ use voice_name

    a = AudioFileClip(audio_path)
    dur = a.duration

    v = animate_image_panzoom(img_path, duration=dur, zoom=0.05)
    v = v.set_audio(a).set_duration(dur)
    scene_clips.append(v)
    audio_handles.append(a)



# Final stitch: compose enforces consistent frame size
if scene_clips:
    print("🔗 Stitching scenes …")
    final = concatenate_videoclips(scene_clips, method="compose")
    final.write_videofile("final_story.mp4", codec="libx264", audio_codec="aac", fps=24)

    print("🎬 Story video saved as final_story.mp4")
    from IPython.display import Video, display, FileLink
    display(Video("final_story.mp4", embed=True))
    display(FileLink("final_story.mp4"))

    # cleanup
    for a in audio_handles: a.close()
    for v in scene_clips: v.close()
    final.close()
else:
    print("⚠️ No scene clips were produced.")