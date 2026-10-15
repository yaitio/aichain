"""
05_multi_turn.py — A directed sequence of refinements in one Skill, without a Chain.

Keys: ANTHROPIC_API_KEY. An assistant turn with no parts means "the model answers here".
"""

from yait_aichain import Model, Skill

# Key: ANTHROPIC_API_KEY from the environment; or Model("claude-haiku-5-5", api_key="...")
skill = Skill(Model("claude-haiku-5-5"), input={"messages": [
    {"role": "system", "parts": ["You are a concise editor. Reply with just the list."]},
    {"role": "user", "parts": ["Write 10 short quotes about {topic}."]},
    {"role": "assistant"},
    {"role": "user", "parts": ["Drop the 5 most clichéd and write 5 fresh ones."]},
    {"role": "assistant"},
    {"role": "user", "parts": ["Translate the final list into {language}."]},
]})

skill.run(variables={"topic": "perseverance", "language": "French"})
for n, turn in enumerate(skill.history, 1):
    print(f"--- turn {n} ---\n{turn}\n")
print(f"total tokens: {skill.last_usage.total_tokens:,}")
