"""
examples/logo_creation.py
=========================

Chain — Engineer a logo prompt, then render it.
Models — claude-sonnet-4-6 (text) + gpt-image-1 (image)

A two-step Chain that shows how easily a text model and an image model compose:

  Step 1 — Prompt engineer (Claude)
      Takes a short brief (``{brand}``, ``{industry}``, ``{mood}``) and produces
      a single paragraph of richly-described visual direction suitable for a
      text-to-image model.  Output is stored as ``prompt_text``.

  Step 2 — Renderer (gpt-image-1)
      Reads ``{prompt_text}`` from the accumulated variables and returns a
      structured image result — ``base64``, ``mime_type``, ``revised_prompt``.
      The base64 payload is decoded and written to ``examples/output/``.

Variable flow
-------------
  run(variables={"brand": ..., "industry": ..., "mood": ...})
      │
      ▼
  prompt_engineer  ─►  accumulated["prompt_text"] = "<visual direction>"
      │
      ▼
  renderer         ─►  {"base64": ..., "mime_type": ..., "revised_prompt": ...}

Because Chains are provider-agnostic, you can swap either step to any other
model without touching the other step — e.g. use ``gemini-2.5-pro`` to engineer
the prompt and ``grok-imagine-image-pro`` to render it.

Usage
-----
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENAI_API_KEY="sk-..."

    python examples/logo_creation.py
"""

import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Model
from skills import Skill
from chain  import Chain


# ---------------------------------------------------------------------------
# Step 1 — Prompt engineer
# ---------------------------------------------------------------------------

prompt_engineer = Skill(
    model=Model("claude-sonnet-4-6"),
    input={
        "messages": [
            {
                "role": "system",
                "parts": [
                    {
                        "type": "text",
                        "text": (
                            "You are a senior brand designer.  Given a short "
                            "brief, produce ONE tightly-written paragraph "
                            "(max 90 words) describing the visual direction "
                            "for a minimalist logo.  Be concrete about shapes, "
                            "palette, composition and style.  Do NOT include "
                            "text or wordmarks in the logo.  Return only the "
                            "description — no preamble."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": (
                            "Brand    : {brand}\n"
                            "Industry : {industry}\n"
                            "Mood     : {mood}"
                        ),
                    }
                ],
            },
        ]
    },
    output={"modalities": ["text"], "format": {"type": "text"}},
    name="prompt_engineer",
    description="Turns a brief brand description into a visual-direction prompt.",
)

# ---------------------------------------------------------------------------
# Step 2 — Image renderer
# ---------------------------------------------------------------------------

renderer = Skill(
    model=Model("gpt-image-1"),
    input={
        "messages": [
            {
                "role": "user",
                "parts": [{"type": "text", "text": "{prompt_text}"}],
            }
        ]
    },
    output={"modalities": ["image"], "format": {"type": "image", "size": "1024x1024"}},
    name="renderer",
    description="Renders the engineered prompt as a logo image.",
)


# ---------------------------------------------------------------------------
# Chain — first step writes to custom key ``prompt_text``
# ---------------------------------------------------------------------------

pipeline = Chain(
    steps=[
        (prompt_engineer, "prompt_text"),   # (runner, output_key)
        renderer,                            # default key "result"
    ],
    name="logo_creation",
    description="Engineers a logo prompt, then renders it.",
)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

_OUTDIR = os.path.join(os.path.dirname(__file__), "output")

_MIME_TO_EXT = {
    "image/png":  "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif":  "gif",
}


BRIEFS = [
    {
        "brand":    "Lumen Labs",
        "industry": "AI research studio",
        "mood":     "scientific, calm, optimistic",
    },
    {
        "brand":    "Kintsugi Coffee",
        "industry": "specialty coffee roaster",
        "mood":     "warm, artisan, grounded",
    },
]


if __name__ == "__main__":
    os.makedirs(_OUTDIR, exist_ok=True)

    print(f"Chain : {pipeline!r}\n")
    print("=" * 70)

    for brief in BRIEFS:
        label = brief["brand"]
        print(f"\n── {label} " + "─" * (66 - len(label)))
        print(f"  brief: {brief}")

        result = pipeline.run(variables=brief)

        # ── show the engineered prompt ────────────────────────────────────
        prompt_text = pipeline.history[0]["output"]
        print(f"\n  ▸ engineered prompt:")
        for line in prompt_text.splitlines():
            print(f"    {line}")

        # ── save the rendered image ───────────────────────────────────────
        ext   = _MIME_TO_EXT.get(result.get("mime_type", ""), "png")
        slug  = label.lower().replace(" ", "_")
        path  = os.path.join(_OUTDIR, f"logo_{slug}.{ext}")
        with open(path, "wb") as f:
            f.write(base64.b64decode(result["base64"]))
        print(f"\n  ✓ saved: {os.path.relpath(path)}")
        if result.get("revised_prompt"):
            print(f"    revised by renderer: {result['revised_prompt'][:120]}...")

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("  • Text + image providers compose through the same Chain primitive.")
    print("  • Step 1's output (prompt_text) flowed into step 2 with zero glue.")
    print("  • Swap Model('gpt-image-1') → Model('grok-imagine-image-pro') to")
    print("    change providers without changing any other code.")
    print()
