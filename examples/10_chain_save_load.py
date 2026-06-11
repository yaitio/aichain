"""
10_chain_save_load.py — Save a chain to YAML, reload and run it.

API keys are never written to disk — resolved from env vars at load time.

Required env vars:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY

Required packages:
    pip install pyyaml
"""

import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain

YAML_PATH = os.path.join(os.path.dirname(__file__), "writer_reviewer.yaml")

# ── Define and save ───────────────────────────────────────────────────────────
writer = Skill(
    model = Model("gpt-4o-mini",    api_key=os.getenv("OPENAI_API_KEY")),
    input = {"messages": [{"role": "user", "parts": ["Write one sentence about {topic}."]}]},
    name  = "writer",
)

reviewer = Skill(
    model = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": ["Improve this sentence: {result}"]}]},
    name  = "reviewer",
)

chain = Chain(steps=[writer, reviewer], name="writer_reviewer")
chain.save(YAML_PATH)
print(f"saved → {YAML_PATH}\n")

# ── Reload and run ────────────────────────────────────────────────────────────
loaded = Chain.load(YAML_PATH)
print(f"loaded: {loaded}\n")

result = loaded.run(variables={"topic": "quantum computing"})
print(result)
