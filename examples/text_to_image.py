"""
examples/text_to_image.py
=========================

Text-to-Image — Universal image generation across all supported models

Sends one identical prompt to every model listed in ``registry`` under the
``"text-to-image"`` task, saves each result to ``examples/output/`` and prints
a per-model status line.  Use this to verify that your API keys are wired
correctly and that the image-generation path works for every image-capable
provider.

Models are drawn automatically from the canonical registry; no hard-coded
model names appear in the script.  Results are grouped by provider.

Supported providers
-------------------
* OpenAI   — ``gpt-image-1`` family
* Google   — ``gemini-*-image*`` preview models
* xAI      — ``grok-imagine-image`` / ``grok-imagine-image-pro``

Usage
-----
Export the API keys for the providers you want to test (any subset works):

    export OPENAI_API_KEY="sk-..."
    export GOOGLE_AI_API_KEY="AIza..."
    export XAI_API_KEY="xai-..."

Then run:

    python examples/text_to_image.py

Models whose API key is not set are automatically skipped with a clear notice.
Images are written to ``examples/output/<provider>_<model>.<ext>`` where the
extension is derived from the image's MIME type.
"""

import base64
import os
import sys
import textwrap
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Model, registry
from skills import Skill


# ---------------------------------------------------------------------------
# Universal prompt — the same input is sent to every model
# ---------------------------------------------------------------------------

_PROMPT = (
    "A minimalist flat-design logo of a mountain at sunrise, "
    "soft pastel colours, clean geometric shapes, no text."
)

_INPUT = {
    "messages": [
        {
            "role": "user",
            "parts": [{"type": "text", "text": _PROMPT}],
        },
    ]
}

_OUTPUT = {
    "modalities": ["image"],
    "format":     {"type": "image", "size": "1024x1024"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COL    = 72
_OUTDIR = os.path.join(os.path.dirname(__file__), "output")

_MIME_TO_EXT = {
    "image/png":  "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif":  "gif",
}

def _header(text: str, char: str = "═") -> str:
    return char * _COL + f"\n  {text}\n" + char * _COL

def _wrap(text: str, indent: int = 6) -> str:
    prefix = " " * indent
    return textwrap.fill(text.strip(), width=_COL - indent,
                         initial_indent=prefix, subsequent_indent=prefix)

def _save(provider: str, model_name: str, result: dict) -> str:
    """Persist the base64 image and return the output path."""
    os.makedirs(_OUTDIR, exist_ok=True)
    ext  = _MIME_TO_EXT.get(result.get("mime_type", ""), "png")
    path = os.path.join(_OUTDIR, f"{provider}_{model_name}.{ext}")
    with open(path, "wb") as f:
        f.write(base64.b64decode(result["base64"]))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    providers = registry.providers(task="text-to-image")

    print()
    print(_header("Text-to-Image  ·  universal image-generation check"))
    print(f"\n  Prompt: {_PROMPT}\n")
    print(f"  Output directory: {_OUTDIR}\n")

    totals = {"ok": 0, "skip": 0, "fail": 0}

    for provider in providers:
        model_names = registry.models(provider=provider, task="text-to-image")

        print()
        print(f"  ── {provider.upper()}  ({len(model_names)} model(s)) " + "─" * max(0, _COL - 14 - len(provider)))

        for name in model_names:
            print(f"\n  ▸ {name}")

            # ── construct model ──────────────────────────────────────────
            try:
                model = Model(name)
            except ValueError as exc:
                print(f"    ⚠  skipped — {exc}")
                totals["skip"] += 1
                continue

            # ── build and run skill ──────────────────────────────────────
            skill = Skill(model=model, input=_INPUT, output=_OUTPUT)

            try:
                t0      = time.perf_counter()
                result  = skill.run()
                elapsed = time.perf_counter() - t0

                path = _save(provider, name, result)
                print(f"    ✓  {elapsed:.1f}s  →  {os.path.relpath(path)}")
                if result.get("revised_prompt"):
                    print(_wrap(f"revised: {result['revised_prompt']}"))
                totals["ok"] += 1

            except Exception as exc:
                print(f"    ✗  {type(exc).__name__}: {exc}")
                totals["fail"] += 1

    # ── summary ─────────────────────────────────────────────────────────────
    print()
    print("─" * _COL)
    ok   = totals["ok"]
    skip = totals["skip"]
    fail = totals["fail"]
    print(
        f"  {ok} passed  ·  {skip} skipped  ·  {fail} failed"
        f"  (of {ok + skip + fail} registered text-to-image models)"
    )
    print("─" * _COL)
    print()
