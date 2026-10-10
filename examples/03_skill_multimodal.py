"""
03_skill_multimodal.py — Text → Image → Text, three providers.

  1. text  → text   Claude  (claude-haiku-5-5)   a model
  2. text  → image  Grok    (grok-imagine-image)  a tool — an image model cannot be an agent's mind
  3. image → text   Qwen    (qwen3-vl-plus)       a model that sees

Required env vars:
    ANTHROPIC_API_KEY
    XAI_API_KEY
    DASHSCOPE_API_KEY
"""

import base64
import os
import pathlib

from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.tools import image

HERE = pathlib.Path(__file__).parent

# ── 1. Text → Text (Claude) ───────────────────────────────────────────────────
text_skill = Skill(
    model = Model("claude-haiku-5-5", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": ["Describe a sunset in one sentence."]}]},
)

description = text_skill.run()
print(f"[text → text · Claude]\n{description}\n")

# ── 2. Text → Image (Grok) ────────────────────────────────────────────────────
drawn   = image("xai", model="grok-imagine-image").run(description, options={"aspect_ratio": "1:1"})
picture = drawn["images"][0]
img_path = HERE / "output_sunset.png"
img_path.write_bytes(base64.b64decode(picture["base64"]))
print(f"[text → image · Grok]\nsaved → {img_path}\n")

# ── 3. Image → Text (Qwen Vision) ────────────────────────────────────────────
vision_skill = Skill(
    model = Model("qwen3-vl-plus", api_key=os.getenv("DASHSCOPE_API_KEY")),
    input = {
        "messages": [{
            "role": "user",
            "parts": [
                {"type": "image", "source": {"kind": "base64",
                                              "data": picture["base64"],
                                              "mime": picture["mime_type"]}},
                {"type": "text",  "text": "What do you see in this image?"},
            ],
        }]
    },
)

analysis = vision_skill.run()
print(f"[image → text · Qwen]\n{analysis}")
