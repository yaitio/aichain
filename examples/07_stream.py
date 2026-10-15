"""
07_stream.py — The answer printed as it is written: the same Skill, streamed.

Keys: ANTHROPIC_API_KEY.
"""

from yait_aichain import Model, Skill

# Key: ANTHROPIC_API_KEY from the environment; or Model("claude-haiku-5-5", api_key="...")
skill = Skill(Model("claude-haiku-5-5"), prompt="Tell a four-sentence story about {hero}.")

pieces = 0
for piece in skill.stream(variables={"hero": "a lighthouse keeper"}):
    print(piece, end="", flush=True)
    pieces += 1
print(f"\n\npieces: {pieces}")
