"""
06_media_input.py — An image, a PDF and a video to one model: the type follows the file.

Keys: GOOGLE_AI_API_KEY.
"""

import os
import struct
import tempfile
import zlib
from pathlib import Path

from yait_aichain import Model, Skill
from yait_aichain.tools import media_part

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"
OUT.mkdir(parents=True, exist_ok=True)


def png(path, w=64, h=64, rgb=(220, 30, 30)):            # a red square, drawn by hand
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    rows = b"".join(b"\0" + bytes(rgb) * w for _ in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def pdf(path, text="Invoice 1042. Total due: 42 EUR."):   # one page, written by hand
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    body, offsets = b"%PDF-1.4\n", []
    for n, obj in enumerate(objs, 1):
        offsets.append(len(body))
        body += b"%d 0 obj\n" % n + obj + b"\nendobj\n"
    xref = len(body)
    body += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    path.write_bytes(body + b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref)


png(OUT / "square.png")
pdf(OUT / "invoice.pdf")
inputs = {"image": (OUT / "square.png", "What colour is this shape?"),
          "pdf": (OUT / "invoice.pdf", "What is the total due?"),
          "video": (Path(__file__).parent / "assets" / "clip.mp4", "Describe this video in one sentence.")}

# Key: GOOGLE_AI_API_KEY from the environment; or Model("gemini-3.5-flash-lite", api_key="...")
model = Model("gemini-3.5-flash-lite")
for kind, (path, question) in inputs.items():
    skill = Skill(model, input={"messages": [{"role": "user", "parts": [question, media_part(str(path))]}]})
    print(f"[{kind}]\n{skill.run()}\n")
