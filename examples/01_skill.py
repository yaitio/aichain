"""
01_skill.py — Run a prompt against a model.
"""

import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill

skill = Skill(
    model = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {
        "messages": [{
            "role": "user",
            "parts": ["What is {topic} in one sentence?"],
        }]
    },
)

result = skill.run(variables={"topic": "machine learning"})
print(result)
