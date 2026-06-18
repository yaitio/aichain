"""
19_image_edit.py — Image → image (editing) across four providers, one Skill.

Editing an image is just an input image part + an image output: the same Skill
as generation, with the provider chosen by the model name. Here one base image
is restyled by every provider whose API key is present.

  OpenAI  gpt-image-1.5            (multipart /v1/images/edits)
  Google  gemini-3.1-flash-image   (conversational edit)
  xAI     grok-imagine-image       (JSON /v1/images/edits)
  Qwen    qwen-image-edit          (synchronous multimodal-generation)

Set whichever you have; the rest are skipped:
    OPENAI_API_KEY  GOOGLE_AI_API_KEY  XAI_API_KEY  DASHSCOPE_API_KEY
"""

import base64
import os
import pathlib
import struct
import zlib

from yait_aichain.models import Model
from yait_aichain.skills import Skill

HERE = pathlib.Path(__file__).parent


def _make_base_png(path: pathlib.Path, w: int = 1024, h: int = 1024) -> None:
    """Write a simple 'product' (a teal box on white) — no image libraries needed."""
    white, teal = (245, 245, 245), (30, 140, 150)
    rows = bytearray()
    for y in range(h):
        rows += b"\x00"                                   # PNG filter byte per row
        for x in range(w):
            in_box = (w * 0.35 < x < w * 0.65) and (h * 0.30 < y < h * 0.78)
            rows += bytes(teal if in_box else white)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + chunk(b"IEND", b"")
    )


base = HERE / "product_base.png"
_make_base_png(base)
print(f"base image → {base}\n")

instruction = "Place this product on a marble kitchen counter, soft morning light, photorealistic"
output = {"modalities": ["image"], "format": {"type": "image", "size": "1024x1024"}}

providers = [
    ("OPENAI_API_KEY",     "gpt-image-1.5"),
    ("GOOGLE_AI_API_KEY",  "gemini-3.1-flash-image"),
    ("XAI_API_KEY",        "grok-imagine-image"),
    ("DASHSCOPE_API_KEY",  "qwen-image-edit"),
    ("RECRAFT_API_TOKEN",  "recraftv3"),
    ("BFL_API_KEY",        "flux-kontext-pro"),
]

for env_var, model_name in providers:
    if not os.getenv(env_var):
        print(f"– {model_name:24} skipped ({env_var} not set)")
        continue
    edit = Skill(
        model  = Model(model_name),
        input  = {"messages": [{"role": "user", "parts": [
            # A local file is read and base64-encoded automatically.
            {"type": "image", "source": {"kind": "file", "path": str(base)}},
            instruction,
        ]}]},
        output = output,
    )
    result  = edit.run()
    out_path = HERE / f"edited_{model_name}.png"
    out_path.write_bytes(base64.b64decode(result["base64"]))
    print(f"✓ {model_name:24} → {out_path.name}  ({result['mime_type']})")
