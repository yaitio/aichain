"""
09_chain_tool_skill.py — Tool + Skill in one chain.

Step 1  fetch("local") reads a page and keeps it under "page".
Step 2  Claude summarises it in three bullet points: {page} in the prompt is
        the page's text, since a Record substitutes as what a model reads.

Required env vars:
    ANTHROPIC_API_KEY

Required packages:
    pip install "yait-aichain[fetch]"
"""

import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain
from yait_aichain.tools import fetch

summarise = Skill(
    model = Model("claude-haiku-5-5", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": [
        "Summarise the following page in exactly 3 bullet points:\n\n{page}"
    ]}]},
)

chain  = Chain(steps=[(fetch("local"), "page", {"url": "source"}), summarise])
result = chain.run(variables={"source": "https://example.com"})

print(result.output)
