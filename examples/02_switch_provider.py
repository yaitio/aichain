"""
02_switch_provider.py — The provider changes with one word, and nothing else does.

Keys: any of ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_AI_API_KEY; a model whose key is unset is skipped.
"""

import os

from yait_aichain import Model, Skill

MODELS = {"claude-haiku-5-5": "ANTHROPIC_API_KEY", "gpt-5.4-mini": "OPENAI_API_KEY",
          "gemini-3.5-flash-lite": "GOOGLE_AI_API_KEY"}

for name, key in MODELS.items():
    if not os.getenv(key):
        print(f"skipped {name} ({key} not set)")
        continue
    # Key: read from the environment variable above; or Model(name, api_key="...")
    skill = Skill(Model(name), prompt="In one sentence: why is the sky blue?")
    print(f"[{name}]")
    print(skill.run())
