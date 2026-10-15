"""
34_budget.py — One ceiling in money across a run: the next step is refused before it begins.

Keys: none. Runs offline: a priced scripted model makes each call cost exactly $0.004.
"""

from yait_aichain import BudgetExceeded, Chain, Skill
from yait_aichain.testing import ScriptedModel

# A real application passes Model("..."), priced from the registry; this one
# reports 1,000 tokens each way at $1 and $3 per million: $0.004 a call.
model = ScriptedModel(["A short paragraph."], usage={"input": 1000, "output": 1000},
                      price={"input": 1.0, "output": 3.0})
chain = Chain([Skill(model, prompt=f"Write part {n} of the story.", name=f"step_{n}")
               for n in range(1, 5)], max_cost=0.01)

# A budget bounds beginning, not exceeding: step 3 starts at $0.008, under the
# limit, and ends at $0.012. Step 4 would start over it, so it never runs.
try:
    chain.run()
except BudgetExceeded as stop:
    print(f"refused: {stop.what}")
    print(f"spent: ${stop.spent:.3f} of ${stop.limit:.3f}")
print(f"calls made: {len(model.requests)}")
