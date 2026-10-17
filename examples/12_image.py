"""
12_image.py — Make an image, then edit it on every provider whose key is set: one tool, nine APIs.

Keys: any of OPENAI_API_KEY, GOOGLE_AI_API_KEY, XAI_API_KEY, DASHSCOPE_API_KEY, RECRAFT_API_TOKEN,
BFL_API_KEY, KREA_API_KEY, KLING_API_KEY,
BYTEPLUS_API_KEY. Images go to AICHAIN_EXAMPLES_DIR (the temp directory by default).
"""

import os
import tempfile
from pathlib import Path

from yait_aichain.tools import image

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"
OUT.mkdir(parents=True, exist_ok=True)

PROVIDERS = [("OPENAI_API_KEY", "openai", "gpt-image-1-mini"),
             ("GOOGLE_AI_API_KEY", "google", "gemini-3.1-flash-image"),
             ("XAI_API_KEY", "xai", "grok-imagine-image"),
             ("DASHSCOPE_API_KEY", "qwen", "qwen-image-edit"),
             ("RECRAFT_API_TOKEN", "recraft", "recraftv3"),
             ("BFL_API_KEY", "bfl", "flux-kontext-pro"),
             ("KREA_API_KEY", "krea", "krea-3"),
             ("KLING_API_KEY", "kling", "kling-v3"),
             ("BYTEPLUS_API_KEY", "byteplus", "dola-seedream-5-0-flash-260915")]
ready = [p for p in PROVIDERS if os.getenv(p[0])]
if not ready:
    raise SystemExit("set one of: " + ", ".join(p[0] for p in PROVIDERS))

_, service, model = ready[0]
made = image(service, model=model, output_dir=str(OUT)).run(
    "A plain teal cardboard box on a white background, product photo", options={"size": "1024x1024"})
box = made["images"][0]["path"]
print(f"made by {model}: {box}\n")

for _, service, model in ready:
    try:
        edited = image(service, model=model, output_dir=str(OUT)).run(
            "Place this box on a marble kitchen counter, soft morning light", images=[box],
            options={"aspect_ratio": "1:1"})
    except Exception as exc:                  # one provider failing is not the example failing
        print(f"✗ {model:24} {type(exc).__name__}: {str(exc)[:90]}")
        continue
    notes = "; ".join(edited["notices"])
    print(f"✓ {model:24} {edited['images'][0]['path']}" + (f"  — {notes}" if notes else ""))
