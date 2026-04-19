"""
examples/json_schema_cross_provider.py
======================================

JSON Schema output — structured extraction across all providers

Sends the same review text to every registered text-to-text model and asks
each one to extract a structured product spec.  The ``json_schema`` output
format guarantees a validated Python ``dict`` back from every call —
regardless of provider.

Extracted schema
----------------
{
  "product_name": str,
  "price_usd":    float | null,
  "rating":       float,          # 1.0 – 5.0
  "pros":         [str, ...],     # 2–4 items
  "cons":         [str, ...],     # 1–3 items
  "verdict":      str             # one sentence
}

What this demonstrates
----------------------
One schema definition works across OpenAI, Anthropic, Google, xAI,
Perplexity, Kimi, and DeepSeek.  Without aichain each provider requires
different structured-output wiring:
  • OpenAI     — response_format={"type": "json_schema", "json_schema": {...}}
  • Anthropic  — tool_use trick (no native json_schema endpoint)
  • Google     — generationConfig.responseSchema
  • xAI        — similar to OpenAI
  • ...

Here: one ``output`` dict, every provider.

Usage
-----
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GOOGLE_AI_API_KEY="AIza..."
    export XAI_API_KEY="xai-..."
    export PERPLEXITY_API_KEY="pplx-..."
    export MOONSHOT_API_KEY="..."
    export DEEPSEEK_API_KEY="..."

    python examples/json_schema_cross_provider.py
"""

import json
import os
import sys
import time
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Model, registry
from skills import Skill


# ---------------------------------------------------------------------------
# Models excluded from this example
# (reasoning-only models can be less reliable for strict JSON extraction)
# ---------------------------------------------------------------------------

_SKIP: frozenset[str] = frozenset({
    "o1", "o3", "o4-mini",
    "grok-4-0709", "grok-4-fast-reasoning", "grok-4-1-fast-reasoning",
    "grok-3-mini", "grok-3-mini-fast",
    "sonar-reasoning-pro", "sonar-deep-research",
    "kimi-k2-thinking", "kimi-k2-thinking-turbo",
    "deepseek-reasoner",
})


# ---------------------------------------------------------------------------
# Input — one product review
# ---------------------------------------------------------------------------

_REVIEW = (
    "I've been using the SoundCore Nova X headphones for three weeks now.  "
    "Paid $129 on Amazon.  The noise cancelling is genuinely impressive — "
    "cuts out my open-plan office completely.  Battery life is solid at "
    "around 28 hours.  Build quality feels premium for the price.  On the "
    "downside, the companion app is buggy and crashes on iOS 17, and the ear "
    "cups get warm after an hour.  Overall I'd rate them 4 out of 5 — a "
    "great buy if you can live without the app."
)

_INPUT = {
    "messages": [
        {
            "role": "system",
            "parts": [{"type": "text", "text": (
                "You are a product data extractor. "
                "Extract the requested fields from the review exactly as written. "
                "Do not infer or hallucinate information not present in the text."
            )}],
        },
        {
            "role": "user",
            "parts": [{"type": "text", "text": "Review:\n\n{review}"}],
        },
    ]
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "product_name": {"type": "string",  "description": "Product name as mentioned."},
        "price_usd":    {"type": ["number", "null"], "description": "Price in USD, or null if not stated."},
        "rating":       {"type": "number",  "description": "Numeric rating on a 1–5 scale."},
        "pros":         {"type": "array",   "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
        "cons":         {"type": "array",   "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "verdict":      {"type": "string",  "description": "One-sentence overall verdict."},
    },
    "required": ["product_name", "price_usd", "rating", "pros", "cons", "verdict"],
    "additionalProperties": False,
}

_OUTPUT = {
    "modalities": ["text"],
    "format": {
        "type":   "json_schema",
        "name":   "product_spec",
        "schema": _SCHEMA,
        "strict": True,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COL = 72

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
    print(_header("JSON Schema extraction  ·  cross-provider"))
    print(f"\n  Review (excerpt): {_REVIEW[:80]}…\n")

    totals = {"ok": 0, "skip": 0, "fail": 0}

    for provider in providers:
        model_names = registry.models(provider=provider, task="text-to-text")

        print()
        print(f"  ── {provider.upper()}  ({len(model_names)} model(s)) "
              + "─" * max(0, _COL - 14 - len(provider)))

        for name in model_names:
            if name in _SKIP:
                print(f"\n  ▸ {name}  (→ skipped — reasoning model)")
                continue

            print(f"\n  ▸ {name}")

            try:
                model = Model(name)
            except ValueError as exc:
                print(f"    ⚠  skipped — {exc}")
                totals["skip"] += 1
                continue

            skill = Skill(model=model, input=_INPUT, output=_OUTPUT)

            try:
                t0     = time.perf_counter()
                result = skill.run(variables={"review": _REVIEW})
                elapsed = time.perf_counter() - t0

                print(f"    ✓  {elapsed:.1f}s  →  {type(result).__name__}")
                # Print a compact one-liner summary of the extracted fields
                print(f"       product : {result.get('product_name')}")
                print(f"       price   : ${result.get('price_usd')}  "
                      f"rating: {result.get('rating')}/5")
                print(f"       pros    : {len(result.get('pros', []))} items  "
                      f"cons: {len(result.get('cons', []))} items")
                print(_wrap(f"verdict: {result.get('verdict', '')}"))
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
