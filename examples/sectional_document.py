"""
examples/sectional_document.py
================================

Sectional document generation — no output-token limits

Generates a structured 4-section business brief where every section is
an independent model call.  Sections never compete for the same output
token budget; the document can be as long as needed.

How it works
------------
1. Sections are serialised into a ``section_queue`` JSON variable.
2. A ``SectionContextTool`` step advances the pointer before each section,
   exposing ``{current_section_title}``, ``{current_section_plan}``, and a
   ``{recent_summaries}`` rolling-context block to the write Skill.
3. After each section is written, a summarise Skill compresses it to 3–5
   bullets — keeping the next section's context window small.
4. ``assemble_document()`` collects all ``{sid}_content`` variables from the
   Chain's accumulated dict and joins them in position order.

Compare with a single large model call
---------------------------------------
  • One call: output truncates at the model's max_output_tokens ceiling.
    A 10-section report often hits this ceiling and loses its last sections.
  • Sectional: each section is a fresh call.  100 sections = 100 calls.
    The total document size is bounded only by your time and API budget.
  • The rolling context window (last 2 section summaries) keeps later
    sections coherent without passing the entire document back each time.

Usage
-----
    export ANTHROPIC_API_KEY="sk-ant-..."

    python examples/sectional_document.py

Swap ``Model("claude-sonnet-4-6")`` for any text-to-text model — nothing
else changes.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models  import Model
from skills  import Skill
from chain   import Chain
from tools   import SectionContextTool

# Import the _sectional helpers directly (they live in products/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "products"))
from _sectional import make_section_queue, assemble_document


# ---------------------------------------------------------------------------
# Document definition
# ---------------------------------------------------------------------------

TOPIC = "Electric vehicle adoption in Southeast Asia"

SECTIONS = [
    {
        "id":       "market_overview",
        "title":    "Market Overview",
        "plan":     (
            "Current EV market size in SE Asia, key growth drivers, "
            "and 3-year forecast.  Lead with the most striking number."
        ),
        "position": 1,
        "runner":   "skill",
    },
    {
        "id":       "competitive_landscape",
        "title":    "Competitive Landscape",
        "plan":     (
            "Top 5 OEMs by market share (BYD, VinFast, etc.), their "
            "positioning, price points, and primary target markets."
        ),
        "position": 2,
        "runner":   "skill",
    },
    {
        "id":       "barriers",
        "title":    "Key Barriers to Adoption",
        "plan":     (
            "Infrastructure gaps (charging), consumer price sensitivity, "
            "grid reliability, and regulatory fragmentation across markets."
        ),
        "position": 3,
        "runner":   "skill",
    },
    {
        "id":       "strategic_implications",
        "title":    "Strategic Implications",
        "plan":     (
            "3–5 prioritised actions for a new-market entrant, derived "
            "from the analysis above.  Tie each recommendation back to a "
            "specific finding from the earlier sections."
        ),
        "position": 4,
        "runner":   "skill",
    },
]


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

_MODEL = Model("claude-sonnet-4-6")

# Write skill — receives the current section's metadata + rolling context
write_skill = Skill(
    model = _MODEL,
    input = {
        "messages": [
            {
                "role": "system",
                "parts": [{"type": "text", "text": (
                    "You are a senior strategy analyst writing a business brief "
                    "on {topic}.  Write the section below in tight, opinionated "
                    "prose.  Be specific — cite estimated figures where you can. "
                    "Do not pad with generic statements.  No meta-commentary.\n\n"
                    "{recent_summaries}"
                )}],
            },
            {
                "role": "user",
                "parts": [{"type": "text", "text": (
                    "## {current_section_title}\n\n"
                    "Section plan: {current_section_plan}\n\n"
                    "Write this section now."
                )}],
            },
        ]
    },
    output = {"modalities": ["text"], "format": {"type": "text"}},
)

# Summarise skill — compresses each section for the rolling context window
summarise_skill = Skill(
    model = Model("claude-haiku-4-5-20251001"),
    input = {
        "messages": [
            {
                "role": "user",
                "parts": [{"type": "text", "text": (
                    "Compress the following section into 3–5 bullet points "
                    "capturing the key facts and figures only.  "
                    "No preamble — bullets only.\n\n"
                    "{current_section_content}"
                )}],
            }
        ]
    },
    output = {"modalities": ["text"], "format": {"type": "text"}},
)


# ---------------------------------------------------------------------------
# Build Chain steps dynamically
# ---------------------------------------------------------------------------
#
# For each section:
#   1. SectionContextTool  — advances queue, exposes current_section_* vars
#   2. write_skill         — generates section content
#   3. summarise_skill     — compresses to bullets for rolling context

section_ctx = SectionContextTool()

steps: list = []
for section in SECTIONS:
    sid = section["id"]

    steps.append((section_ctx, f"{sid}_ctx", {}))

    steps.append((write_skill, f"{sid}_content"))

    steps.append((
        summarise_skill,
        "current_section_actual_summary",
        {"current_section_content": f"{sid}_content"},
    ))


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

pipeline = Chain(
    steps     = steps,
    variables = {
        "topic":                         TOPIC,
        "section_queue":                 make_section_queue(SECTIONS),
        "recent_summaries":              "",
        "current_section_actual_summary": "",
        "prev_section_summary":          "",
        "two_sections_ago_summary":      "",
    },
    name = "sectional_brief",
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print(f"\n  Topic   : {TOPIC}")
    print(f"  Sections: {len(SECTIONS)}")
    print(f"  Steps   : {len(pipeline._steps)} total "
          f"(3 per section: context → write → summarise)")
    print(f"  Models  : write={_MODEL!r}  summarise=claude-haiku-4-5")
    print()
    print("  ── Generating ───────────────────────────────────────────────────\n")

    pipeline.run()

    document = assemble_document(pipeline.accumulated, SECTIONS)

    print("\n  ── Document ─────────────────────────────────────────────────────\n")
    print(document)

    # ── Section stats ────────────────────────────────────────────────────────
    print("\n  ── Section lengths ──────────────────────────────────────────────")
    total_chars = 0
    for section in SECTIONS:
        content = pipeline.accumulated.get(f"{section['id']}_content", "")
        total_chars += len(content)
        print(f"    {section['title']:30s}  {len(content):5d} chars")
    print(f"    {'TOTAL':30s}  {total_chars:5d} chars")
    print()
    print("  Key takeaway: each section was an independent model call.  The")
    print("  document is not bounded by any single output-token ceiling.")
    print()
