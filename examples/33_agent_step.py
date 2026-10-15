"""
33_agent_step.py — Outside code drives the loop, one step per invocation.

Keys: none. Runs offline. The state between steps is a list of plain messages,
so it can sit in a queue or a database between two serverless calls.
"""

import json

from yait_aichain import Agent
from yait_aichain.models import dangling_calls, tool_result_turn
from yait_aichain.testing import ScriptedModel, call

WEATHER = {"Paris": "14°C, rain", "Rome": "22°C, sun"}

# A real application passes Model("..."); the script plays the model here.
model = ScriptedModel([call("weather", city="Paris"), call("weather", city="Rome"),
                       "Rome is warmer: 22°C and sunny, against 14°C and rain in Paris."])
agent = Agent(model)          # no tools of its own: the world answers the calls

messages = [{"role": "user", "parts": ["Which is warmer today, Paris or Rome?"]}]
for n in range(1, 10):
    reply = agent.step(messages)
    print(f"step {n}: {reply if isinstance(reply, str) else [c.name for c in reply.calls]}")
    if isinstance(reply, str):
        print(f"answer: {reply}")
        break
    messages.append(reply.as_turn())
    for c in reply.calls:                        # the world: your code, an API, a person
        messages.append(tool_result_turn(c.id, WEATHER[c.arguments["city"]]))
    messages = json.loads(json.dumps(messages))  # plain data between invocations

# A call nobody answered is caught before it reaches a provider.
unanswered = messages[:2]                    # the first call, without its result
print(f"dangling: {dangling_calls(unanswered)}")
