"""
examples/text_to_text.py
========================

Text-to-Text — Universal query across all supported models

Sends one identical prompt to every model listed in ``registry`` under the
``"text-to-text"`` task, running them sequentially and printing each response.
Use this to verify that your API keys are configured correctly and that the
library integrates cleanly with the current model versions.

Models are drawn automatically from the canonical registry; no hard-coded
model names appear in the script.  Results are grouped by provider.

Usage
-----
Export the API keys for the providers you want to test (any subset works):

    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GOOGLE_AI_API_KEY="AIza..."
    export XAI_API_KEY="xai-..."
    export PERPLEXITY_API_KEY="pplx-..."

Then run:

    python examples/text_to_text.py

Models whose API key is not set are automatically skipped with a clear notice.
"""

import os
import sys
import time
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Model, registry
from skills import Skill


# ---------------------------------------------------------------------------
# Reasoning models — excluded here, covered in text_to_text_reasoning.py
# ---------------------------------------------------------------------------
#
# These models either always reason internally (o-series, grok-4 series) or
# are specifically designed to be used with the ``reasoning`` parameter
# (grok-3-mini, sonar-reasoning-pro).  They are intentionally skipped in this
# example to keep the comparison apples-to-apples.

_REASONING_MODELS: frozenset[str] = frozenset({
    # OpenAI o-series  (always-on reasoning)
    "o1", "o3", "o4-mini",
    # xAI grok-4 series  (always-on reasoning)
    "grok-4-0709", "grok-4-fast-reasoning", "grok-4-1-fast-reasoning",
    # xAI grok-3-mini  (reasoning_effort parameter)
    "grok-3-mini", "grok-3-mini-fast",
    # Perplexity  (built-in chain-of-thought)
    "sonar-reasoning-pro",
})


# ---------------------------------------------------------------------------
# Universal prompt — the same input is sent to every model
# ---------------------------------------------------------------------------

_PROMPT = (
    "Name three core principles of clean code and explain each one "
    "in a single sentence."
)

_INPUT = {
    "messages": [
        {
            "role": "system",
            "parts": [
                {
                    "type": "text",
                    "text": (
                        "You are a helpful assistant. "
                        "Answer concisely and directly."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "parts": [{"type": "text", "text": _PROMPT}],
        },
    ]
}

_OUTPUT = {
    "modalities": ["text"],
    "format":     {"type": "text"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COL = 72   # terminal column width

def _header(text: str, char: str = "═") -> str:
    return char * _COL + f"\n  {text}\n" + char * _COL

def _wrap(text: str, indent: int = 6) -> str:
    prefix = " " * indent
    return textwrap.fill(text.strip(), width=_COL - indent,
                         initial_indent=prefix, subsequent_indent=prefix)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    providers = registry.providers(task="text-to-text")

    print()
    print(_header("Text-to-Text  ·  universal model check"))
    print(f"\n  Prompt: {_PROMPT}\n")

    totals = {"ok": 0, "skip": 0, "fail": 0}

    for provider in providers:
        model_names = registry.models(provider=provider, task="text-to-text")

        print()
        print(f"  ── {provider.upper()}  ({len(model_names)} model(s)) " + "─" * max(0, _COL - 14 - len(provider)))

        for name in model_names:
            if name in _REASONING_MODELS:
                print(f"\n  ▸ {name}  (→ text_to_text_reasoning.py)")
                continue

            print(f"\n  ▸ {name}")

            # ── construct model ──────────────────────────────────────────
            try:
                model = Model(name)
            except ValueError as exc:
                # API key not configured for this provider
                print(f"    ⚠  skipped — {exc}")
                totals["skip"] += 1
                continue

            # ── build and run skill ──────────────────────────────────────
            skill = Skill(model=model, input=_INPUT, output=_OUTPUT)

            try:
                t0      = time.perf_counter()
                answer  = skill.run()
                elapsed = time.perf_counter() - t0

                print(f"    ✓  {elapsed:.1f}s")
                for line in answer.strip().splitlines():
                    print(_wrap(line))
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
        f"  (of {ok + skip + fail} registered text-to-text models)"
    )
    print("─" * _COL)
    print()
