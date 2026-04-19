"""
examples/qwen_skills.py
========================

Qwen / DashScope provider — runnable skill examples
=====================================================

Demonstrates three scenarios using Alibaba's DashScope API:

  1. **Text-to-text** — chat completion with qwen-max / qwen-turbo.
  2. **Image understanding** — describe an image with qwen-vl-max (vision).
  3. **Reasoning** — deep thinking with QwQ-32B (always-on CoT).

Each example is self-contained.  All three require only:

    export DASHSCOPE_API_KEY="sk-..."

Optional — select your nearest region (default: ap = international endpoint):

    export DASHSCOPE_REGION="ap"   # ap | us | cn | hk

Run the full suite::

    python examples/qwen_skills.py

Run one scenario::

    python examples/qwen_skills.py --scenario text
    python examples/qwen_skills.py --scenario vision
    python examples/qwen_skills.py --scenario reasoning
"""

from __future__ import annotations

import os
import sys

# ── Resolve library root ──────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _LIB_DIR)

try:
    import dotenv
    dotenv.load_dotenv(os.path.join(_LIB_DIR, ".env"))
except ImportError:
    pass

from skill import Skill
from models import Model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


def _check_key() -> bool:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("  ⚠  DASHSCOPE_API_KEY is not set — skipping Qwen examples.")
        print("     Export your key with:  export DASHSCOPE_API_KEY='sk-...'")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text-to-text
# ─────────────────────────────────────────────────────────────────────────────

def demo_text() -> None:
    """Chat completion with qwen-max and qwen-turbo."""
    _section("1. Text-to-text  (qwen-max, qwen-turbo)")

    if not _check_key():
        return

    # ── qwen-max ─────────────────────────────────────────────────────────────
    print("  Model: qwen-max")
    skill = Skill(
        model          = Model("qwen-max"),
        system_prompt  = "You are a concise technical assistant.",
        output_format  = "text",
    )
    result = skill.run(
        input = "Explain what an embedding vector is in two sentences."
    )
    print(f"  → {result}\n")

    # ── qwen-turbo (budget-friendly) ─────────────────────────────────────────
    print("  Model: qwen-turbo")
    skill_t = Skill(
        model          = Model("qwen-turbo"),
        system_prompt  = "You are a helpful assistant.",
        output_format  = "text",
    )
    result_t = skill_t.run(
        input = "Give me three creative names for a pet hamster."
    )
    print(f"  → {result_t}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vision / image understanding
# ─────────────────────────────────────────────────────────────────────────────

def demo_vision() -> None:
    """Describe a public image with qwen-vl-max."""
    _section("2. Image understanding  (qwen-vl-max)")

    if not _check_key():
        return

    # A reliable public image (Wikipedia commons — Eiffel Tower)
    IMAGE_URL = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "a/a7/Camponotus_flavomarginatus_ant.jpg/"
        "400px-Camponotus_flavomarginatus_ant.jpg"
    )

    print("  Model: qwen-vl-max")
    print(f"  Image: {IMAGE_URL}\n")

    skill = Skill(
        model         = Model("qwen-vl-max"),
        system_prompt = "You are a precise image analyst.",
        output_format = "text",
    )

    # Pass the image as a URL — the library wraps it in the correct content format
    result = skill.run(
        input   = "Describe what you see in this image in two or three sentences.",
        images  = [IMAGE_URL],
    )
    print(f"  → {result}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reasoning  (QwQ-32B)
# ─────────────────────────────────────────────────────────────────────────────

def demo_reasoning() -> None:
    """Solve a multi-step problem with QwQ-32B (always-on chain-of-thought)."""
    _section("3. Reasoning  (QwQ-32B)")

    if not _check_key():
        return

    print("  Model: QwQ-32B  (always-on thinking)")

    skill = Skill(
        model          = Model("QwQ-32B"),
        system_prompt  = "You are a careful mathematical reasoner.",
        output_format  = "text",
    )

    problem = (
        "A farmer has 17 sheep.  All but 9 die.  "
        "How many sheep does the farmer have left?  "
        "Show your reasoning step by step."
    )

    print(f"  Problem: {problem}\n")
    result = skill.run(input=problem)
    print(f"  → {result}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIOS = {
    "text":      demo_text,
    "vision":    demo_vision,
    "reasoning": demo_reasoning,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen / DashScope skill examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/qwen_skills.py               # run all\n"
            "  python examples/qwen_skills.py --scenario text\n"
            "  python examples/qwen_skills.py --scenario vision\n"
            "  python examples/qwen_skills.py --scenario reasoning\n"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=list(_SCENARIOS),
        help="Which scenario to run (default: all).",
    )
    args = parser.parse_args()

    print("\n══ Qwen / DashScope examples ══════════════════════════════════════\n")
    region = os.environ.get("DASHSCOPE_REGION", "ap")
    print(f"  Region: {region}")
    print()

    if args.scenario:
        _SCENARIOS[args.scenario]()
    else:
        for fn in _SCENARIOS.values():
            fn()

    print("\n══ Done ════════════════════════════════════════════════════════════\n")
