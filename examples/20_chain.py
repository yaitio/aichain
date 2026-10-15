"""
20_chain.py — Steps in order, each step's result under its own key: GPT writes, Claude reviews.

Keys: OPENAI_API_KEY, ANTHROPIC_API_KEY.
"""

from yait_aichain import Chain, Model, Skill

# Keys: OPENAI_API_KEY and ANTHROPIC_API_KEY from the environment; or Model(name, api_key="...")
writer = Skill(Model("gpt-5.4-mini"), name="writer",
               prompt="Write a three-sentence paragraph about {topic}.")
reviewer = Skill(Model("claude-haiku-5-5"), name="reviewer",
                 prompt="Suggest one concrete improvement to this paragraph:\n\n{writer}")

result = Chain([writer, reviewer]).run(variables={"topic": "why cities plant trees"})
print(f"[writer]\n{result['writer']}\n")
print(f"[reviewer]\n{result['reviewer']}")
print(f"\ncost: ${result.cost:.5f}")
