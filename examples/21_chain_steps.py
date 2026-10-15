"""
21_chain_steps.py — Connect steps whose names differ, and let the chain say what it needs.

Keys: ANTHROPIC_API_KEY; the page is read with fetch("local"), which needs none.
"""

import json

from yait_aichain import Chain, Model, Skill
from yait_aichain.chain import Step
from yait_aichain.tools import fetch

# Key: ANTHROPIC_API_KEY from the environment; or Model("claude-haiku-5-5", api_key="...")
summary = Skill(Model("claude-haiku-5-5"), name="summary",
                prompt="Summarise this page in two sentences:\n\n{text}")

chain = Chain([
    Step(fetch("local"), out="page", inputs={"url": "source"}),     # the tool's `url` reads `source`
    Step(summary, inputs={"text": "page.pages.0.text"}),           # a path into the step's result
], inputs={"source": "The URL of the page to summarise"})

print(json.dumps(chain.input_schema(), indent=2))
result = chain.run(variables={"source": "https://example.com"})
print(f"summary: {result['summary']}")
