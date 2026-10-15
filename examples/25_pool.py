"""
25_pool.py — A list processed in parallel with one parameter, no threads in your code.

Keys: ANTHROPIC_API_KEY.
"""

from yait_aichain import Model, Pool, Skill

TOPICS = ["photosynthesis", "inflation", "plate tectonics", "vaccines", "compound interest"]

# Key: ANTHROPIC_API_KEY from the environment; or Model("claude-haiku-5-5", api_key="...")
skill = Skill(Model("claude-haiku-5-5"), prompt="Explain {topic} in one sentence.")
results = Pool(skill, items=[{"topic": t} for t in TOPICS], max_flows=5).run()

for topic, answer in zip(TOPICS, results):       # outputs come back in item order
    print(f"[{topic}]\n{answer}\n")
print(f"success: {results.success}  errors: {sum(e is not None for e in results.errors)}  "
      f"cost: ${results.cost:.5f}")
