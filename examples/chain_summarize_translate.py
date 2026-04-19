"""
examples/chain_summarize_translate.py
======================================

Chain — Summarise then Translate
Model  — gpt-4o for both steps

This example demonstrates a two-step Chain:

  Step 1 — Summariser
    Takes a long article text ({article}) and compresses it into a concise
    3-sentence summary.  The output is stored as "result" (the default
    output key) so the next step can reference it as {result}.

  Step 2 — Translator
    Receives the accumulated variables — including {result} from step 1
    and the original {language} variable — and translates the summary into
    the requested language.

Variable flow
-------------

  run(variables={"article": "...", "language": "Spanish"})
       │
       ▼
  ┌─────────────┐   output (str)
  │  summariser  │ ──────────────────► accumulated["result"] = "<summary>"
  └─────────────┘
       │  accumulated = {"article": ..., "language": "Spanish",
       │                 "result": "<summary>"}
       ▼
  ┌─────────────┐   output (str)
  │  translator  │ ──────────────────► final return value
  └─────────────┘

After the run, chain.history contains one record per step, each with the
variables that were passed in and the output that came out — a complete
audit trail of the pipeline.

Usage
-----
Set your OpenAI API key first:

    export OPENAI_API_KEY="sk-..."

Then run:

    python examples/chain_summarize_translate.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Model
from skills import Skill
from chain  import Chain

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = Model("gpt-4o")

# ---------------------------------------------------------------------------
# Step 1 — Summariser
# ---------------------------------------------------------------------------
# Output is a plain string → stored as accumulated["result"] by default.

summariser = Skill(
    model=model,
    input={
        "messages": [
            {
                "role": "system",
                "parts": [
                    {
                        "type": "text",
                        "text": (
                            "You are a precise summarisation engine.  "
                            "Condense the provided article into exactly "
                            "3 sentences.  Preserve key facts and figures.  "
                            "Do not add commentary or opinions."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": "Article:\n\n{article}",
                    }
                ],
            },
        ]
    },
    output={
        "modalities": ["text"],
        "format":     {"type": "text"},
    },
    name="summariser",
    description="Condenses any article into a 3-sentence summary.",
)

# ---------------------------------------------------------------------------
# Step 2 — Translator
# ---------------------------------------------------------------------------
# Receives {result} (summary from step 1) and {language} (from initial vars).

translator = Skill(
    model=model,
    input={
        "messages": [
            {
                "role": "system",
                "parts": [
                    {
                        "type": "text",
                        "text": (
                            "You are a professional translator.  "
                            "Translate the provided text into {language}.  "
                            "Preserve the original meaning exactly — "
                            "do not add or remove information."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": "Translate this text:\n\n{result}",
                    }
                ],
            },
        ]
    },
    output={
        "modalities": ["text"],
        "format":     {"type": "text"},
    },
    name="translator",
    description="Translates text into the requested language.",
)

# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------
# Bare Skill entries use "result" as the output key, so summariser's output
# is automatically available as {result} when translator runs.

pipeline = Chain(
    steps=[summariser, translator],
    name="summarise_and_translate",
    description="Summarises an article, then translates the summary.",
)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

ARTICLE = """
Artificial intelligence is reshaping the global economy at an unprecedented
pace.  According to a 2024 McKinsey report, generative AI alone could add
between $2.6 trillion and $4.4 trillion annually to the global economy across
63 analysed use cases.  The technology is automating knowledge work in sectors
ranging from software engineering and legal services to healthcare diagnostics
and financial analysis.

Adoption is not uniform, however.  While large enterprises in developed
economies are deploying AI tools rapidly, small and medium-sized businesses —
which account for over 90 % of all companies and roughly 70 % of employment in
OECD countries — lag significantly behind due to high implementation costs and
a shortage of technical talent.

Governments are responding with a mix of investment and regulation.  The
European Union's AI Act, which came into force in August 2024, imposes strict
requirements on high-risk AI applications in areas such as critical
infrastructure, education, and employment.  Meanwhile, the United States has
taken a lighter-touch approach, relying primarily on voluntary commitments from
major AI developers while the regulatory framework evolves.

Labour markets face genuine disruption.  The World Economic Forum's Future of
Jobs Report 2025 estimates that 85 million jobs may be displaced by AI and
automation by 2030, while 97 million new roles may emerge — but the skills
required for those new roles differ sharply from those being automated away,
placing the burden of re-skilling squarely on workers and educational
institutions.
"""

LANGUAGES = ["Spanish", "French", "German"]

if __name__ == "__main__":
    print(f"Chain  : {pipeline!r}")
    print(f"Model  : {model!r}\n")
    print("=" * 70)

    for language in LANGUAGES:
        print(f"\n── Target language: {language} " + "─" * (49 - len(language)))

        translated_summary = pipeline.run(
            variables={
                "article":  ARTICLE,
                "language": language,
            }
        )

        # ── show history ──────────────────────────────────────────────
        for record in pipeline.history:
            step_label = f"Step {record['step']} · {record['name']}"
            print(f"\n{step_label}")
            print(f"  Variables passed in : {sorted(record['input'].keys())}")
            print(f"  Output key          : {record['output_key']!r}")

            # Truncate long outputs for readability in the demo
            out_preview = str(record["output"])
            if len(out_preview) > 200:
                out_preview = out_preview[:197] + "..."
            print(f"  Output preview      : {out_preview}")

        print(f"\nFinal output ({language}):")
        for line in translated_summary.splitlines():
            print(f"  {line}")
        print()

    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Each step received the accumulated variables from all prior steps.")
    print("  • summariser's output was stored as 'result' (the default key).")
    print("  • translator read {result} and {language} without extra wiring.")
    print("  • chain.history gives a full audit trail of every step.")
