"""
14_agent_tools.py — Agent with multiple tools, picks autonomously.

The agent decides which tool to use for each step:
  - search("perplexity")  for web research
  - fetch("local")        to read a specific page

Required env vars:
    ANTHROPIC_API_KEY
    PERPLEXITY_API_KEY

Required packages:
    pip install "yait-aichain[fetch]"
"""

import os
from yait_aichain.models import Model
from yait_aichain.agent  import Agent, step_count
from yait_aichain.tools import fetch, search

agent = Agent(
    Model("claude-haiku-4-5-20251001", api_key=os.getenv("ANTHROPIC_API_KEY")),
    tools     = [
        search("perplexity", api_key=os.getenv("PERPLEXITY_API_KEY")),
        fetch("local"),
    ],
    stop_when = [step_count(8)],
)

result = agent.run(
    "Find the official Qdrant documentation homepage URL, "
    "then fetch that page and tell me what the main sections are."
)

print(result.output)
print(f"\nsteps={result.steps_taken}  tokens={result.tokens_used:,}")
