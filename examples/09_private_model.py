"""
09_private_model.py — A model on your own server: the data stays with you, the code is the same.

Keys: none; PRIVATE_BASE_URL and PRIVATE_TEST_MODEL name your server and model. Start one with
`vllm serve Qwen/Qwen3-0.6B`, `ollama run qwen3:0.6b` or LM Studio's server.
"""

import os

from yait_aichain import Model, Skill

BASE_URL = os.getenv("PRIVATE_BASE_URL", "http://localhost:8000")
NAME = os.getenv("PRIVATE_TEST_MODEL", "Qwen/Qwen3-0.6B")

CONTRACT = ("The supplier delivers 500 units by 1 March. Late delivery costs 2% of the order a week. "
            "Either party may end the contract with 30 days' notice.")

model = Model(f"private/{NAME}", client_options={"url": BASE_URL})
skill = Skill(model, prompt="Summarise this contract in two sentences:\n\n{contract}")
print(skill.run(variables={"contract": CONTRACT}))
print(f"tokens: {skill.last_usage.total_tokens}")
print(f"cost: {skill.last_usage.cost}")          # None: a model you host has no price per token
