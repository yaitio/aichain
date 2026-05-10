"""
08_chain.py — Two skills in sequence, two different providers.

Step 1  GPT writes a short paragraph on a topic.
Step 2  Claude reviews it and suggests one improvement.

The output of step 1 is automatically available as {result} in step 2.

Required env vars:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
"""

import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain

writer = Skill(
    model = Model("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    input = {"messages": [{"role": "user", "parts": [
        "Write a short paragraph (3 sentences) about {topic}."
    ]}]},
    name  = "writer",
)

reviewer = Skill(
    model = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": [
        "Review this paragraph and suggest one concrete improvement:\n\n{result}"
    ]}]},
    name  = "reviewer",
)

chain  = Chain(steps=[writer, reviewer])
result = chain.run(variables={"topic": "the future of AI agents"})

print("[writer]")
print(chain.history[0]["output"])
print("\n[reviewer]")
print(result)
