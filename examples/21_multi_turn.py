"""
21_multi_turn.py — Directed multi-turn reasoning in ONE Skill.

Some tasks are a guided sequence of refinements: ask, get a result, refine it,
refine again, finish. That's a "directed" reasoning chain — *you* know the steps
(unlike an Agent, which decides them itself).

You express it inside a single Skill: the message template carries several
`user` turns separated by "generate here" markers — an `assistant` turn with no
`parts`. The model runs each turn in sequence with the full prior context, so the
last step sees everything the earlier steps produced. No Chain, no manual
threading of outputs.

  10 quotes  →  drop the 5 most clichéd + write 5 fresh  →  translate the result

Required env vars:
    ANTHROPIC_API_KEY     (or swap the model)
"""

import os

from yait_aichain.models import Model
from yait_aichain.skills import Skill

skill = Skill(
    model = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [
        {"role": "system",    "parts": ["You are a concise editor. Reply with just the list."]},
        {"role": "user",      "parts": ["Write 10 great quotes about {topic}."]},
        {"role": "assistant"},                       # ← generate here, keep in context
        {"role": "user",      "parts": ["Drop the 5 most clichéd and write 5 fresh ones."]},
        {"role": "assistant"},                       # ← generate here, keep in context
        {"role": "user",      "parts": ["Translate the final list into {language}."]},
    ]},
)

result = skill.run(variables={"topic": "perseverance", "language": "French"})

print("=== final ===")
print(result)

# Every turn the model produced, in order (history[-1] == the returned result).
print("\n=== each turn (skill.history) ===")
for i, turn in enumerate(skill.history, 1):
    print(f"\n--- turn {i} ---\n{turn}")

print(f"\ntotal tokens across the {len(skill.history)} turns: "
      f"{skill.last_usage.total_tokens:,}")
