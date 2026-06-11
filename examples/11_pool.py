"""
11_pool.py — Run the same skill in parallel for multiple inputs.

5 topics are summarised simultaneously instead of one by one.

Required env vars:
    ANTHROPIC_API_KEY
"""

import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.pool import Pool, DONE, FAILED

skill = Skill(
    model = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": [
        "Explain {topic} in exactly one sentence."
    ]}]},
)

items = [
    {"topic": "neural networks"},
    {"topic": "quantum computing"},
    {"topic": "blockchain"},
    {"topic": "vector databases"},
    {"topic": "reinforcement learning"},
]

pool    = Pool(skill, items=items, max_flows=5)
results = pool.run()

# ── Results ───────────────────────────────────────────────────────────────────
for i, (item, result) in enumerate(zip(items, results)):
    print(f"[{item['topic']}]\n{result}\n")

# ── Status summary ────────────────────────────────────────────────────────────
s = pool.status
print(f"done={s[DONE]}  failed={s[FAILED]}")

# ── History (duration per task) ───────────────────────────────────────────────
print()
for r in pool.history:
    print(f"  [{r['index']}] {r['variables']['topic']:<25} {r['duration']}s")
