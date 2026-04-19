"""
examples/template_variables.py
===============================

Skill as a reusable function — template variables at run time

Builds ONE Skill once and calls it many times with different ``variables=``
dicts.  The prompt template stays on the Skill object; only the variable
values change per call.

This mirrors how you'd use a function: define the logic once, parameterise
the inputs.  The Skill handles provider serialisation, retries, and output
parsing on every call without any extra code.

What this demonstrates
----------------------
  • Default variables on the Skill (used when nothing is passed at run time).
  • Per-call variable override via ``skill.run(variables={...})``.
  • The same Skill object running against a batch of inputs.
  • ``max_retries=3`` — transient HTTP errors (429/503) are retried
    automatically; the calling code never sees them.

Compare with raw API usage, where you would reconstruct the full
``messages`` list, serialise it to the provider format, parse the
response, and implement retry logic — for every call and every provider.

Usage
-----
    export OPENAI_API_KEY="sk-..."      # or any single provider key

    python examples/template_variables.py
"""

import os
import sys
import time
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Model
from skills import Skill


# ---------------------------------------------------------------------------
# Build the Skill ONCE
# ---------------------------------------------------------------------------
#
# The prompt has two placeholders: {concept} and {audience}.
# Default values are set on the Skill; any call-time dict overrides them.

explainer = Skill(
    model = Model("gpt-4o"),
    input = {
        "messages": [
            {
                "role": "system",
                "parts": [{"type": "text", "text": (
                    "You are a clear, concise technical writer. "
                    "Explain concepts so that {audience} can understand them. "
                    "Use one short paragraph — no bullet lists, no headers."
                )}],
            },
            {
                "role": "user",
                "parts": [{"type": "text", "text": "Explain: {concept}"}],
            },
        ]
    },
    output = {
        "modalities": ["text"],
        "format":     {"type": "text"},
    },
    # ── Default variables — used when run() receives no overrides ──────
    variables = {
        "concept":  "recursion",
        "audience": "a first-year computer science student",
    },
    name        = "concept_explainer",
    description = "Explains any concept to a specified audience.",
    max_retries = 3,   # HTTP 429/503/504 retried automatically
)


# ---------------------------------------------------------------------------
# Batch inputs
# ---------------------------------------------------------------------------
#
# Each dict overrides the Skill's default variables for that single call.
# Keys not present in the override fall back to the Skill's defaults.

BATCH: list[dict] = [
    # ── same audience, different concepts ──────────────────────────────
    {"concept": "TCP/IP handshake",       "audience": "a first-year CS student"},
    {"concept": "gradient descent",       "audience": "a first-year CS student"},
    {"concept": "transformer attention",  "audience": "a first-year CS student"},
    # ── same concept, different audiences ──────────────────────────────
    {"concept": "compound interest",      "audience": "a twelve-year-old"},
    {"concept": "compound interest",      "audience": "a professional investor"},
    {"concept": "compound interest",      "audience": "a retired librarian"},
    # ── uses Skill-level defaults (no overrides) ───────────────────────
    {},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COL = 72

def _header(text: str, char: str = "═") -> str:
    return char * _COL + f"\n  {text}\n" + char * _COL

def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text.strip(), width=_COL - indent,
                         initial_indent=prefix, subsequent_indent=prefix)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print()
    print(_header("Template variables  ·  one Skill, many inputs"))
    print(f"\n  Skill : {explainer!r}")
    print(f"  Model : {explainer.model!r}")
    print(f"  Batch : {len(BATCH)} calls\n")
    print("─" * _COL)

    totals = {"ok": 0, "fail": 0}

    for i, overrides in enumerate(BATCH, start=1):

        # Merge defaults with overrides for display
        effective = {**explainer.variables, **overrides}
        concept   = effective.get("concept",  "?")
        audience  = effective.get("audience", "?")

        print(f"\n  [{i}/{len(BATCH)}]  concept={concept!r}  audience={audience!r}")

        try:
            t0      = time.perf_counter()
            result  = explainer.run(variables=overrides)   # ← only pass the overrides
            elapsed = time.perf_counter() - t0

            print(f"    ✓  {elapsed:.1f}s")
            print(_wrap(result))
            totals["ok"] += 1

        except Exception as exc:
            print(f"    ✗  {type(exc).__name__}: {exc}")
            totals["fail"] += 1

    # ── summary ─────────────────────────────────────────────────────────────
    print()
    print("─" * _COL)
    print(f"  {totals['ok']} succeeded  ·  {totals['fail']} failed  "
          f"(of {len(BATCH)} calls to the same Skill object)")
    print()
    print("  Key takeaways:")
    print("    • The Skill object was built once and called multiple times.")
    print("    • Empty dict {} → Skill-level defaults used unchanged.")
    print("    • max_retries=3 means any transient HTTP error is retried")
    print("      automatically — zero extra code in the calling loop.")
    print("─" * _COL)
    print()
