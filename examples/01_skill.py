"""
01_skill.py — One prompt, one model, one answer.

Keys: ANTHROPIC_API_KEY.
"""

from yait_aichain import Model, Skill

# Key: ANTHROPIC_API_KEY from the environment; or Model("claude-haiku-5-5", api_key="...")
skill = Skill(Model("claude-haiku-5-5"), prompt="What is {topic}, in one sentence?")
print(skill.run(variables={"topic": "machine learning"}))
