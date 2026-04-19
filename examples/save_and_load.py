"""
examples/save_and_load.py
==========================

Demonstrates how to persist a Skill to a YAML file and reload it later.

The example uses a "keyword extractor" skill:
  – Model  : gpt-4o
  – Input  : any text passage
  – Output : JSON object  {"keywords": ["word1", "word2", ...]}

Workflow
--------
1. Build the skill with a strict JSON-schema output.
2. Save it to ``examples/skills/keyword_extractor.yaml``.
3. Inspect the YAML on disk.
4. Load the skill back from the file (attaching a fresh model).
5. Run both the original and the reloaded skill; confirm they produce
   the same output shape.

Usage
-----
Set your OpenAI API key first:

    export OPENAI_API_KEY="sk-..."

Then run:

    python examples/save_and_load.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Model   # used only to build the initial skill
from skills import Skill

# ---------------------------------------------------------------------------
# 1. Build the skill
# ---------------------------------------------------------------------------

model = Model("gpt-4o")

# Strict schema: always returns {"keywords": ["...", "...", ...]}
KEYWORDS_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {
            "type":        "array",
            "items":       {"type": "string"},
            "minItems":    5,
            "maxItems":    5,
            "description": "Exactly 5 keywords extracted from the passage.",
        }
    },
    "required":             ["keywords"],
    "additionalProperties": False,
}

keyword_skill = Skill(
    model=model,
    input={
        "messages": [
            {
                "role": "system",
                "parts": [
                    {
                        "type": "text",
                        "text": (
                            "You are a keyword extraction engine.\n"
                            "Given a passage of text, return exactly 5 keywords "
                            "that best represent its main topics."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "parts": [{"type": "text", "text": "{passage}"}],
            },
        ]
    },
    output={
        "modalities": ["text"],
        "format": {
            "type":   "json_schema",
            "name":   "keywords_result",
            "schema": KEYWORDS_SCHEMA,
            "strict": True,
        },
    },
    name="keyword_extractor",
    description="Extracts 5 keywords from a text passage.",
    variables={
        "passage": "Artificial intelligence is transforming every industry.",
    },
)

# ---------------------------------------------------------------------------
# 2. Save to YAML
# ---------------------------------------------------------------------------

yaml_path = os.path.join(os.path.dirname(__file__), "skills", "keyword_extractor.yaml")
keyword_skill.save(yaml_path)
print(f"Saved  → {yaml_path}\n")

# ---------------------------------------------------------------------------
# 3. Show what the YAML looks like on disk
# ---------------------------------------------------------------------------

with open(yaml_path, "r") as fh:
    print("── YAML content ──────────────────────────────────────")
    print(fh.read())
    print("──────────────────────────────────────────────────────\n")

# ---------------------------------------------------------------------------
# 4. Load it back — the model is reconstructed from the YAML automatically
# ---------------------------------------------------------------------------

loaded_skill = Skill.load(yaml_path)

print(f"Loaded ← {loaded_skill!r}\n")

# ---------------------------------------------------------------------------
# 5. Run both and compare
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passage = (
        "The James Webb Space Telescope is revealing unprecedented details "
        "about the formation of galaxies and the atmospheres of exoplanets."
    )

    print(f"Passage: {passage}\n")

    print("Running original skill …")
    result_original = keyword_skill.run(variables={"passage": passage})
    keywords_original = result_original["keywords"]
    print(f"  Keywords (original) : {keywords_original}\n")

    print("Running loaded skill …")
    result_loaded   = loaded_skill.run(variables={"passage": passage})
    keywords_loaded = result_loaded["keywords"]
    print(f"  Keywords (loaded)   : {keywords_loaded}\n")

    assert isinstance(keywords_original, list), "Expected a list from original skill"
    assert isinstance(keywords_loaded,   list), "Expected a list from loaded skill"
    assert len(keywords_original) == 5,         "Expected exactly 5 keywords (original)"
    assert len(keywords_loaded)   == 5,         "Expected exactly 5 keywords (loaded)"
    print("✓ Both skills returned exactly 5 keywords.")
