"""
28_agent_stop.py — "Done" on the model's word and "done" on a check are different ends.

Keys: ANTHROPIC_API_KEY. A safe holds a code from 1 to 1000; the run ends when the check sees it open.
"""

import random

from yait_aichain import Agent, Model
from yait_aichain.agent import check, nudge, stalled, step_count
from yait_aichain.tools import tool

SECRET = random.randint(1, 1000)
safe = {"open": False, "probes": 0}


@tool(risk="read")
def probe(guess: int) -> str:
    """Try a code on the safe; it answers higher, lower or open.

    Parameters
    ----------
    guess : int
        The code to try, 1 to 1000.
    """
    safe["probes"] += 1
    if guess == SECRET:
        safe["open"] = True
        return "open"
    return "higher" if guess < SECRET else "lower"


# Key: ANTHROPIC_API_KEY from the environment; or Model("claude-haiku-5-5", api_key="...")
agent = Agent(Model("claude-haiku-5-5"), tools=[probe], stop_when=[
    check(lambda state: safe["open"], name="safe_open"),                     # a fact, checked
    nudge(stalled(3), "You are repeating yourself: halve the range you have left."),
    step_count(15),                                                          # the ceiling
])
result = agent.run("Open the safe. Each probe answers higher, lower or open.")

print(f"secret: {SECRET}  probes: {safe['probes']}")
print(f"stopped_by: {result.stopped_by}  success: {result.success}")
