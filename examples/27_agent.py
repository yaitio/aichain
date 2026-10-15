"""
27_agent.py — An agent picks its own steps between search and fetch; the result says how it ended.

Keys: ANTHROPIC_API_KEY and any one of BRAVE_SEARCH_API_KEY, EXA_API_KEY, TAVILY_API_KEY,
PARALLEL_API_KEY, PERPLEXITY_API_KEY, FIRECRAWL_API_KEY.
"""

import os

from yait_aichain import Agent, Model, Tracer
from yait_aichain.agent import step_count
from yait_aichain.tools import fetch, search

SERVICES = {"BRAVE_SEARCH_API_KEY": "brave", "EXA_API_KEY": "exa", "TAVILY_API_KEY": "tavily",
            "PARALLEL_API_KEY": "parallel", "PERPLEXITY_API_KEY": "perplexity",
            "FIRECRAWL_API_KEY": "firecrawl"}
service = next((name for key, name in SERVICES.items() if os.getenv(key)), None)
if service is None:
    raise SystemExit("set one of: " + ", ".join(SERVICES))

tracer = Tracer()
# Key: ANTHROPIC_API_KEY from the environment; or Model("claude-haiku-5-5", api_key="...")
agent = Agent(Model("claude-haiku-5-5"), tools=[search(service), fetch("local")],
              stop_when=[step_count(8)], max_cost=0.10, hooks=[tracer])
result = agent.run("Find the official Qdrant documentation and list its main sections.")

print(result.output)
print(f"\nstopped_by: {result.stopped_by}  steps: {result.steps_taken}  cost: ${result.cost:.4f}")
print("tools:", ", ".join(e.name for e in tracer.events if e.type == "tool_call.started"))
