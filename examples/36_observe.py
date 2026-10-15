"""
36_observe.py — What happened inside a run, without print() in your code or a tracing SDK.

Keys: none. Runs offline: one Tracer watches a Chain, a Pool and an Agent.
"""

import logging
import sys

from yait_aichain import Agent, Chain, Pool, Skill, Tracer
from yait_aichain.testing import ScriptedModel, call
from yait_aichain.tools import tool

# The library logs through `yait_aichain.*` and attaches no handler: you choose where.
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="log │ %(message)s")


@tool(risk="read")
def word_count(text: str) -> int:
    """Count the words in a text.

    Parameters
    ----------
    text : str
        The text to count.
    """
    return len(text.split())


# A real application passes Model("..."); the scripts play the models here.
tracer = Tracer()
writer = Skill(ScriptedModel(["Rain on the river."]), prompt="A line about {topic}.",
               name="line", hooks=[tracer])
chain = Chain([writer, (word_count, {"text": "line"})], name="poem", hooks=[tracer])
chain.run(variables={"topic": "rain"})
Pool(writer, items=[{"topic": "sun"}, {"topic": "snow"}], name="lines", hooks=[tracer]).run()
agent = Agent(ScriptedModel([call("word_count", text="one two three"), "Three words."]),
              tools=[word_count], name="counter", hooks=[tracer])
result = agent.run("How many words in 'one two three'?")

print("\ntimeline:")
for e in tracer.events:
    print(f"{e.type:<18} {e.name or '':<10} {e.run_id or ''}")
print("\njournal:")
for entry in result.journal:
    print(f"  {entry['action']} → {entry['outcome']}")
