"""
examples/text_to_text_reasoning.py
====================================

Text-to-Text Reasoning — Universal query across all reasoning-capable models

Sends one identical prompt to every model that supports extended thinking or
chain-of-thought reasoning, running them sequentially and printing each
response.  Use this to verify that reasoning is wired correctly across
providers and to compare the quality of their deliberative answers.

The library translates the universal ``reasoning`` parameter to each
provider's native API automatically:

  Provider       Param accepted          Native field
  ─────────────────────────────────────────────────────────────────────────
  Anthropic      reasoning="low|med|hi"  thinking.budget_tokens (4K/10K/20K)
                                         temperature forced to 1.0 by the model
  OpenAI         reasoning="low|med|hi"  reasoning_effort ("low"/"medium"/"high")
                                         o-series ignores temperature
  Google AI      reasoning="low|med|hi"  generationConfig
                                           .thinkingConfig.thinkingBudget
                                           (2K/8K/24K)
  xAI (mini)     reasoning="low|med|hi"  reasoning_effort ("low"/"high")
                                         xAI maps medium → high
  xAI (grok-4)   n/a — always reasons    no parameter accepted
  Perplexity     n/a — built-in CoT      sonar-reasoning-pro reasons internally

Models whose API key is not set are automatically skipped.

Usage
-----
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GOOGLE_AI_API_KEY="AIza..."
    export XAI_API_KEY="xai-..."
    export PERPLEXITY_API_KEY="pplx-..."

    python examples/text_to_text_reasoning.py
"""

import os
import sys
import time
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Model
from skills import Skill


# ---------------------------------------------------------------------------
# Reasoning-capable models
#
# Each entry defines:
#   name     — model identifier (must match the registry / factory)
#   provider — used only for grouping the output display
#   options  — passed to Model(); includes reasoning= where applicable
#   note     — one-line description of how reasoning is implemented
# ---------------------------------------------------------------------------

_ENTRIES: list[dict] = [

    # ── OpenAI o-series ───────────────────────────────────────────────────
    # reasoning= maps to reasoning_effort in the request body.
    # o-series models ignore temperature; max_tokens caps the full output
    # (reasoning tokens + visible tokens combined).
    {
        "name":     "o3",
        "provider": "openai",
        "options":  {"max_tokens": 32768, "reasoning": "high"},
        "note":     "reasoning_effort=high",
    },
    {
        "name":     "o4-mini",
        "provider": "openai",
        "options":  {"max_tokens": 32768, "reasoning": "medium"},
        "note":     "reasoning_effort=medium",
    },
    {
        "name":     "o1",
        "provider": "openai",
        "options":  {"max_tokens": 32768, "reasoning": "medium"},
        "note":     "reasoning_effort=medium",
    },

    # ── Anthropic extended thinking ───────────────────────────────────────
    # reasoning= maps to thinking.budget_tokens (4 000 / 10 000 / 20 000).
    # AnthropicModel forces temperature=1.0 automatically when thinking is on.
    {
        "name":     "claude-opus-4-6",
        "provider": "anthropic",
        "options":  {"max_tokens": 32000, "reasoning": "high"},
        "note":     "thinking budget_tokens=20 000",
    },
    {
        "name":     "claude-sonnet-4-6",
        "provider": "anthropic",
        "options":  {"max_tokens": 16000, "reasoning": "medium"},
        "note":     "thinking budget_tokens=10 000",
    },

    # ── Google AI thinking budget ─────────────────────────────────────────
    # reasoning= maps to generationConfig.thinkingConfig.thinkingBudget
    # (2 048 / 8 192 / 24 576 tokens).
    {
        "name":     "gemini-2.5-pro",
        "provider": "google",
        "options":  {"max_tokens": 16384, "reasoning": "medium"},
        "note":     "thinkingBudget=8 192",
    },
    {
        "name":     "gemini-3.1-pro-preview",
        "provider": "google",
        "options":  {"max_tokens": 16384, "reasoning": "medium"},
        "note":     "thinkingBudget=8 192",
    },

    # ── xAI grok-3-mini — reasoning_effort ───────────────────────────────
    # reasoning= maps to reasoning_effort.  xAI only supports "low" and
    # "high"; the library maps "medium" → "high" automatically.
    {
        "name":     "grok-3-mini",
        "provider": "xai",
        "options":  {"max_tokens": 16384, "reasoning": "high"},
        "note":     "reasoning_effort=high",
    },
    {
        "name":     "grok-3-mini-fast",
        "provider": "xai",
        "options":  {"max_tokens": 16384, "reasoning": "high"},
        "note":     "reasoning_effort=high",
    },

    # ── xAI grok-4 series — always-on reasoning ───────────────────────────
    # grok-4 models reason internally on every request.  The reasoning
    # parameter is not accepted and must be omitted.
    {
        "name":     "grok-4-0709",
        "provider": "xai",
        "options":  {"max_tokens": 32768},
        "note":     "always reasons — no parameter",
    },
    {
        "name":     "grok-4-fast-reasoning",
        "provider": "xai",
        "options":  {"max_tokens": 32768},
        "note":     "always reasons — no parameter",
    },
    {
        "name":     "grok-4-1-fast-reasoning",
        "provider": "xai",
        "options":  {"max_tokens": 32768},
        "note":     "always reasons — no parameter",
    },

    # ── Perplexity — built-in chain-of-thought ────────────────────────────
    # sonar-reasoning-pro reasons internally; no parameter is accepted.
    {
        "name":     "sonar-reasoning-pro",
        "provider": "perplexity",
        "options":  {},
        "note":     "built-in CoT — no parameter",
    },
]


# ---------------------------------------------------------------------------
# Prompt — chosen to require deliberate reasoning, not pattern recall
# ---------------------------------------------------------------------------

_PROMPT = (
    "How many times do the hour hand and the minute hand of a clock "
    "overlap in a 12-hour period?  Provide the exact count and show "
    "your mathematical reasoning step by step."
)

_INPUT = {
    "messages": [
        {
            "role": "system",
            "parts": [
                {
                    "type": "text",
                    "text": (
                        "You are a precise analytical assistant. "
                        "Work through problems step by step before "
                        "stating your final answer."
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

_COL = 72

def _header(text: str, char: str = "═") -> str:
    return char * _COL + f"\n  {text}\n" + char * _COL

def _wrap(text: str, indent: int = 6) -> str:
    prefix = " " * indent
    return textwrap.fill(
        text.strip(), width=_COL - indent,
        initial_indent=prefix, subsequent_indent=prefix,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print()
    print(_header("Text-to-Text Reasoning  ·  reasoning model check"))
    print(f"\n  Prompt: {_PROMPT}\n")

    totals        = {"ok": 0, "skip": 0, "fail": 0}
    last_provider = None

    for entry in _ENTRIES:
        name     = entry["name"]
        provider = entry["provider"]
        options  = entry["options"]
        note     = entry["note"]

        # ── provider heading ─────────────────────────────────────────────
        if provider != last_provider:
            last_provider = provider
            print()
            print(
                f"  ── {provider.upper()} "
                + "─" * max(0, _COL - 6 - len(provider))
            )

        print(f"\n  ▸ {name}  [{note}]")

        # ── construct model ──────────────────────────────────────────────
        try:
            model = Model(name, options=options if options else None)
        except ValueError as exc:
            print(f"    ⚠  skipped — {exc}")
            totals["skip"] += 1
            continue

        # ── build and run skill ──────────────────────────────────────────
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

    # ── summary ──────────────────────────────────────────────────────────
    print()
    print("─" * _COL)
    ok   = totals["ok"]
    skip = totals["skip"]
    fail = totals["fail"]
    print(
        f"  {ok} passed  ·  {skip} skipped  ·  {fail} failed"
        f"  (of {ok + skip + fail} reasoning models)"
    )
    print("─" * _COL)
    print()
