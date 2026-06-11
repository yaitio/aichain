"""
12_pool_chain.py — Chain as a Pool runner.

For each URL: fetch page → summarise. All URLs processed in parallel.

Each item runs its own full Chain independently.
Pool fires all Chains at the same time.

Required env vars:
    ANTHROPIC_API_KEY

Required packages:
    pip install markitdown
"""

import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain
from yait_aichain.pool import Pool, DONE, FAILED
from yait_aichain.tools import convertToMD

fetch = convertToMD()

summarise = Skill(
    model = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": [
        "Summarise in one sentence:\n\n{result}"
    ]}]},
)

per_item = Chain(steps=[(fetch, "result", {"input": "source"}), summarise])

items = [
    {"source": "https://fr.lipsum.com"},
    {"source": "https://de.lipsum.com"},
    {"source": "https://es.lipsum.com"},
]

pool    = Pool(per_item, items=items, max_flows=3)
results = pool.run()

# ── Results ───────────────────────────────────────────────────────────────────
for item, result in zip(items, results):
    print(f"[{item['source']}]\n{result}\n")

# ── Status ────────────────────────────────────────────────────────────────────
s = pool.status
print(f"done={s[DONE]}  failed={s[FAILED]}")
