"""
04_fallback_models.py — Survive a provider outage with one parameter: a list of models.

Keys: ANTHROPIC_API_KEY. The first model is a server that is not running.
"""

from yait_aichain import Model, Skill

down = Model("private/not-running", client_options={"url": "http://127.0.0.1:9"})
# Key: ANTHROPIC_API_KEY from the environment; or Model("claude-haiku-5-5", api_key="...")
skill = Skill([down, Model("claude-haiku-5-5")], prompt="Name one prime number above 100.")

print(skill.run())
swap = next(a for a in skill.last_adaptations if a.option == "model")
print(f"answered by: {swap.sent}")
print(f"why: {swap.why[:120]}")
