"""
02_skill_models.py — Same prompt, three providers.

Swap Model() to change the provider. Everything else stays the same.

Required env vars:
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GOOGLE_AI_API_KEY
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models import Model
from skills import Skill

PROMPT = {
    "messages": [{
        "role": "user",
        "parts": ["What is machine learning in one sentence?"],
    }]
}

models = [
    Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    Model("gpt-4o-mini",       api_key=os.getenv("OPENAI_API_KEY")),
    Model("gemini-2.5-flash",  api_key=os.getenv("GOOGLE_AI_API_KEY")),
]

for model in models:
    skill  = Skill(model=model, input=PROMPT)
    result = skill.run()
    print(f"[{model.name}]\n{result}\n")
