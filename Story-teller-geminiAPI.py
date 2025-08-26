from kaggle_secrets import UserSecretsClient
import os
!pip install elevenlabs
from elevenlabs import ElevenLabs, save, VoiceSettings


# Narration language: "en" for English, "kn" for Kannada
NARRATION_LANG = "en"

# Choose which TTS engine to use: "gtts" or "11labs"
TTS_API = "gtts"   # options: "gtts", "11labs"
DEFAULT_SCENE_DURATION = 5  # fallback when no audio

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


def tts_gtts(text: str, out_path: str, lang: str = "en"):
    """Generate narration audio with gTTS (Indian English accent for en, Kannada supported)."""
    if lang == "en":
        tts = gTTS(text=text, lang="en", tld="co.in")  # 🇮🇳 Indian English accent
    elif lang == "kn":
        tts = gTTS(text=text, lang="kn")               # Kannada
    else:
        tts = gTTS(text=text, lang=lang)

    tts.save(out_path)
    return out_path

def tts_elevenlabs(text: str, out_path: str, voice_id: str = "ogSj7jM4rppgY9TgZMqW", model_id: str = "eleven_multilingual_v2"):
    """Generate narration audio using ElevenLabs TTS by voice id."""


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
    
def generate_tts(text: str, out_path: str, lang: str = "en"):
    """Dispatch TTS to GTTS or ElevenLabs based on TTS_API flag."""
    if TTS_API == "gtts":
        return tts_gtts(text, out_path, lang)
    elif TTS_API == "11labs":
        return tts_elevenlabs(text, out_path, voice_id="ogSj7jM4rppgY9TgZMqW")
    else:
        raise ValueError(f"Unsupported TTS_API: {TTS_API}")
# ===============================
# Step 2 + Step 3 : Story → Scenes → Images (clean JSON output)
# ===============================

import json, re
from google.genai import types
from PIL import Image
from io import BytesIO

def safe_json_parse(raw_text):
    """Extract valid JSON from Gemini output (handles code fences, junk tokens)."""
    # Remove markdown code fences
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    
    # Remove obvious junk like 'ophe' or trailing commas
    cleaned = re.sub(r"[^\{\}\[\],:\"'\w\s\.\-\n]", "", cleaned)  # strip invalid chars
    cleaned = re.sub(r",\s*}", "}", cleaned)  # fix trailing commas before }
    cleaned = re.sub(r",\s*]", "]", cleaned)  # fix trailing commas before ]
    
    # Extract first JSON-looking structure
    m = re.search(r"(\[.*\]|\{.*\})", cleaned, flags=re.DOTALL)
    if m:
        cleaned = m.group(1)
    
    try:
        return json.loads(cleaned)
    except Exception as e:
        print("⚠️ Still invalid JSON after cleaning. Error:", e)
        print("---- Cleaned text ----\n", cleaned[:500])
        return []


# 1) Take story input
default_story = """
In the ancient Gurukul of Sage Sandipani, two boys, Krishna and Sudama, forged a bond that transcended time and status. Krishna, the prince of Dwarka, was known for his divine charm and playful wit. Sudama, a humble Brahmin, was his closest friend, known for his poverty and unwavering devotion.
Years passed. Krishna became the king of Dwarka, a prosperous kingdom of gold and jewels. Sudama, meanwhile, remained in his small, thatched hut, struggling to feed his family. His wife, distraught by their poverty, urged him to visit his old friend, believing that Krishna would help.
Reluctantly, Sudama agreed. He couldn't go empty-handed, so his wife tied a small bundle of roasted rice (poha) in a worn-out cloth, a gift from a friend. With a heart full of doubt and a mind replaying memories of their shared childhood, Sudama embarked on his long journey to Dwarka.
Upon arriving at the palace gates, the guards, seeing his tattered clothes, scoffed at him. But when the news of his arrival reached Krishna, the king's face lit up with joy. He ran out, his royal robes and crown forgotten, to embrace his old friend. Krishna washed Sudama's feet, a gesture of profound respect, and seated him on a golden throne.
Krishna, with a playful smile, asked, "What have you brought for me, my friend?" Sudama, embarrassed by his simple gift, hid the small bundle behind his back. But Krishna, with his divine sight, knew what it was. He snatched the bundle and, with great delight, ate the roasted rice. He savored each grain, his love for Sudama making the humble offering taste sweeter than any royal feast.
After their heartfelt reunion, Sudama prepared to leave, his pride preventing him from asking for help. He returned home, his heart heavy with the thought that he had not received any gifts. But as he approached his village, he was stunned. His old hut was replaced by a magnificent palace, his family adorned in silks and jewels.
Tears of gratitude welled in his eyes. He realized that Krishna, in his divine generosity, had granted him wealth without ever being asked. The story of Krishna and Sudama became a timeless tale, a beautiful testament to the power of true friendship, unwavering faith, and the boundless compassion of the Divine."""    

user_input = input("Please enter your story (press Enter to use default):\n")
story = user_input if user_input else default_story

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

import random
from moviepy.editor import ImageClip, AudioFileClip, vfx

from moviepy.editor import ImageClip, vfx

def animate_scene(image_path, audio_path, duration, scene_number=None):
    base_clip = ImageClip(image_path).set_duration(duration)
    W, H = base_clip.size

    # Define pan/zoom ranges
    pan_fraction = 0.15   # pan up to 15% of width/height
    zoom_strength = 0.2   # zoom up to ±20%

    # Scale factor to ensure no black borders (1 + pan_fraction)
    scale_factor = 1 + pan_fraction

    zoomed = base_clip.resize(scale_factor)

    # ---- Define pan functions ----
    def pan_lr(get_frame, t):
        frame = get_frame(t)
        shift = int((pan_fraction * W) * (t / duration))
        return frame[:, shift:shift + W]

    def pan_rl(get_frame, t):
        frame = get_frame(t)
        shift = int((pan_fraction * W) * (1 - t / duration))
        return frame[:, shift:shift + W]

    def pan_tb(get_frame, t):
        frame = get_frame(t)
        shift = int((pan_fraction * H) * (t / duration))
        return frame[shift:shift + H, :]

    def pan_bt(get_frame, t):
        frame = get_frame(t)
        shift = int((pan_fraction * H) * (1 - t / duration))
        return frame[shift:shift + H, :]

    # ---- Define effects ----
    effects = [
        ("zoom_in", zoomed.resize(lambda t: 1 + zoom_strength * (t / duration))),
        ("zoom_out", zoomed.resize(lambda t: 1 + zoom_strength * (1 - t / duration))),
        ("pan_left_to_right", zoomed.fl(pan_lr, apply_to=['mask'])),
        ("pan_right_to_left", zoomed.fl(pan_rl, apply_to=['mask'])),
        ("pan_top_to_bottom", zoomed.fl(pan_tb, apply_to=['mask'])),
        ("pan_bottom_to_top", zoomed.fl(pan_bt, apply_to=['mask'])),
        ("static", base_clip)
    ]

    # Randomly pick effect
    effect_name, animated = random.choice(effects)
    if scene_number is not None:
        print(f"🎥 Scene {scene_number}: Using {effect_name} effect (scale={scale_factor})")

    # Fade in/out (video only)
    animated = animated.fx(vfx.fadein, 1).fx(vfx.fadeout, 1)

    # Add narration audio if available
    if audio_path:
        audio = AudioFileClip(audio_path).set_duration(duration)
        animated = animated.set_audio(audio)

    return animated






scene_clips = []
audio_handles = []   # keep refs to close later

scene_clips = []
audio_handles = []   # keep refs to close later

for scene in scenes:
    n = scene["scene_number"]
    img_path = f"scene_{n}.png"
    if not os.path.exists(img_path):
        print(f"⚠️ Missing image for scene {n}: {img_path}. Skipping.")
        continue

    narration_text = scene.get("narration", scene.get("description", f"Scene {n}"))
    audio_path = f"scene_{n}.mp3"

    # Translate narration if Kannada requested
    if NARRATION_LANG == "kn":
        print(f"🌐 Translating narration for scene {n} to Kannada …")
        narration_text = translate_to_kannada(narration_text)
        print(f"Narration text : {narration_text}")

    audio_path = f"scene_{n}.mp3"
    try:
        print(f"🔊 Generating TTS with {TTS_API} for scene {n} …")
        generate_tts(narration_text, audio_path, lang=NARRATION_LANG)
        a = AudioFileClip(audio_path)
        dur = a.duration
        v = animate_scene(img_path, audio_path, dur, scene_number=n)
        v = v.set_audio(a).set_duration(dur)
        audio_handles.append(a)
    except Exception as e:
        print(f"⚠️ TTS failed for scene {n}, using default duration {DEFAULT_SCENE_DURATION}s. Error: {e}")
        dur = DEFAULT_SCENE_DURATION
        v = animate_scene(img_path, None, dur, scene_number=n)

    scene_clips.append(v)

    


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