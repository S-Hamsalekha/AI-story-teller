from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GEMINI_API_KEY")  # 👈 fetch your saved secret
os.environ["GEMINI_API_KEY"] = api_key               # 👈 now set as env variable

print("✅ GEMINI_API_KEY loaded")

assert os.getenv("GEMINI_API_KEY"), "Token missing!"


from google import genai

client = genai.Client()  # reads GEMINI_API_KEY automatically
# quick ping (text) — ensures auth works
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with the word: PONG"
)
print(resp.text)



# ===============================
# Step 2 + Step 3 : Story → Scenes → Images (clean JSON output)
# ===============================

import json
from google.genai import types
from PIL import Image
from io import BytesIO

# 1) Take story input
story = input("Please enter your story:\n")

# 2) Ask Gemini to segment story into scenes (strict JSON only)
segmentation_prompt = f"""
You are a storytelling assistant. 

TASK:
- Break the following story into 3–5 distinct visual scenes.
- Output MUST be a single valid JSON array ONLY.
- Do NOT include any explanations, natural language text, or code fences.
- Each object in the array must have:
  - "scene_number" (integer)
  - "description" (string, 1–2 sentences, what’s happening)
  - "prompt" (string, a cinematic, artistic illustration prompt)

Story:
{story}
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=segmentation_prompt
)

print("\nGemini Output (raw):\n")
print(response.text)

# 3) Parse JSON directly
try:
    scenes = json.loads(response.text.strip())
except Exception as e:
    print("⚠️ Could not parse Gemini output as JSON.")
    print("Error:", e)
    scenes = []

# 4) Generate an image for each scene
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
    for part in img_response.candidates[0].content.parts:
        if getattr(part, "inline_data", None):
            img = Image.open(BytesIO(part.inline_data.data))
            break

    if img:
        filename = f"scene_{scene['scene_number']}.png"
        img.save(filename)
        display(img)
        print(f"✅ Saved {filename}\n")
    else:
        print("⚠️ No image returned for this scene.\n")