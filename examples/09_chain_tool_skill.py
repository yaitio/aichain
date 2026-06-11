"""
09_chain_tool_skill.py — Tool + Skill in one chain.

Step 1  convertToMD fetches a URL and converts it to Markdown.
Step 2  Claude summarises the page in three bullet points.

Required env vars:
    ANTHROPIC_API_KEY

Required packages:
    pip install markitdown
"""

import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain
from yait_aichain.tools import convertToMD

fetch = convertToMD()

summarise = Skill(
    model = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": [
        "Summarise the following page in exactly 3 bullet points:\n\n{result}"
    ]}]},
)

chain  = Chain(steps=[(fetch, "result", {"input": "source"}), summarise])
result = chain.run(variables={"source": "https://example.com"})

print(result)
