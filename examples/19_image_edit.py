"""
19_image_edit.py — Image → image (editing) across providers, one tool.

Editing is the image tool given images: image(service, model=…).run(prompt,
images=[…]). One base image is restyled by every provider whose API key is set;
a provider that cannot do it says why, and the rest go on.

  OpenAI   gpt-image-1.5
  Google   gemini-3.1-flash-image
  xAI      grok-imagine-image
  Qwen     qwen-image-edit
  Recraft  recraftv3
  BFL      flux-kontext-pro

Required env vars:
    any one of OPENAI_API_KEY  GOOGLE_AI_API_KEY  XAI_API_KEY  DASHSCOPE_API_KEY
               RECRAFT_API_TOKEN  BFL_API_KEY
"""

import base64
import os
import pathlib
import struct
import zlib

from yait_aichain.tools import image

HERE = pathlib.Path(__file__).parent
EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/svg+xml": "svg"}


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

providers = [
    ("OPENAI_API_KEY",     "openai",  "gpt-image-1.5"),
    ("GOOGLE_AI_API_KEY",  "google",  "gemini-3.1-flash-image"),
    ("XAI_API_KEY",        "xai",     "grok-imagine-image"),
    ("DASHSCOPE_API_KEY",  "qwen",    "qwen-image-edit"),
    ("RECRAFT_API_TOKEN",  "recraft", "recraftv3"),
    ("BFL_API_KEY",        "bfl",     "flux-kontext-pro"),
]

for env_var, service, model in providers:
    if not os.getenv(env_var):
        print(f"– {model:24} skipped ({env_var} not set)")
        continue
    try:
        edited = image(service, model=model).run(instruction, images=[str(base)],
                                                 options={"size": "1024x1024"})
    except Exception as exc:                  # one provider failing is not the example failing
        print(f"✗ {model:24} {type(exc).__name__}: {str(exc)[:100]}")
        continue
    picture  = edited["images"][0]
    out_path = HERE / f"edited_{model}.{EXT.get(picture['mime_type'], 'png')}"
    out_path.write_bytes(base64.b64decode(picture["base64"]))
    notes = f"  — {'; '.join(edited['notices'])}" if edited["notices"] else ""
    print(f"✓ {model:24} → {out_path.name}  ({picture['mime_type']}){notes}")
