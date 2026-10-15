"""
10_what_was_sent.py — The library never changes a request in silence.

Keys: none. Runs offline: recorded() keeps the real Model and replaces only its
transport, so the adaptations and the price are the library's own answers.
"""

import warnings

from yait_aichain import Model, Skill
from yait_aichain.testing import recorded

warnings.simplefilter("ignore")      # each adaptation is also a warning; printed below instead

# recorded() sends nothing, so a placeholder key will do here.
model = Model("gpt-5.5", options={"temperature": 0.2, "top_k": 5}, api_key="sk-placeholder")
skill = Skill(model, prompt="Name one planet.")

with recorded(model, reply="Mars") as rec:
    answer = skill.run()

print(f"asked for: {model.effective_options['_asked']}")
for a in skill.last_adaptations:
    print(f"{a.kind} {a.option}: {a.why}")
sent = rec.requests[0]
print(f"sent to: {sent.path}")
print(f"body keys: {sorted(sent.body)}")
print(f"auth header: {sent.headers.get('Authorization')}")
print(f"answer: {answer}")
print(f"cost: ${skill.last_usage.cost:.6f}")
