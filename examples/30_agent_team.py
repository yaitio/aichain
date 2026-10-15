"""
30_agent_team.py — A lead hands work to its team: an agent that searches, a chain that reads.

Keys: ANTHROPIC_API_KEY and any one of BRAVE_SEARCH_API_KEY, EXA_API_KEY, TAVILY_API_KEY,
PARALLEL_API_KEY, PERPLEXITY_API_KEY. Members are held to the lead's policy and budget.
"""

import os

from yait_aichain import Agent, Chain, Model, Skill, Tracer
from yait_aichain.chain import Step
from yait_aichain.tools import fetch, search

SERVICES = {"BRAVE_SEARCH_API_KEY": "brave", "EXA_API_KEY": "exa", "TAVILY_API_KEY": "tavily",
            "PARALLEL_API_KEY": "parallel", "PERPLEXITY_API_KEY": "perplexity"}
service = next((name for key, name in SERVICES.items() if os.getenv(key)), None)
if service is None:
    raise SystemExit("set one of: " + ", ".join(SERVICES))

# Key: ANTHROPIC_API_KEY from the environment; or Model("claude-haiku-5-5", api_key="...")
model = Model("claude-haiku-5-5")
researcher = Agent(model, tools=[search(service)], name="researcher",
                   description="Searches the web and answers with sources")
reader = Chain([Step(fetch("local"), out="page", inputs={"url": "url"}),
                Step(Skill(model, name="summary", prompt="Summarise in three bullets:\n\n{text}"),
                     inputs={"text": "page.pages.0.text"})],
               name="reader", description="Reads one web page and summarises it",
               inputs={"url": "The page's address"})

tracer = Tracer()
lead = Agent(model, team=[researcher, reader], max_cost=0.20, hooks=[tracer])
result = lead.run("What is the latest Python release, and what does python.org's front page say?")

for e in tracer.events:
    if e.type == "tool_call.started" and e.name == "delegate":
        print(f"delegate → {e.payload['arguments'].get('worker')}: {e.payload['arguments'].get('task')}")
print(f"\n{result.output}\n\ncost: ${result.cost:.4f} (members included)")
