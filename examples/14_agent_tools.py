"""
14_agent_tools.py — Agent with multiple tools, picks autonomously.

The agent decides which tool to use for each step:
  - searchPerplexity  for web research
  - convertToMD       to fetch and read a specific page

Required env vars:
    ANTHROPIC_API_KEY
    PERPLEXITY_API_KEY

Required packages:
    pip install markitdown
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models import Model
from agent  import Agent
from tools  import searchPerplexity, convertToMD

agent = Agent(
    orchestrator = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    tools        = [
        searchPerplexity(api_key=os.getenv("PERPLEXITY_API_KEY")),
        convertToMD(),
    ],
    max_steps    = 8,
    mode         = "agile",
)

result = agent.run(
    "Find the official Qdrant documentation homepage URL, "
    "then fetch that page and tell me what the main sections are."
)

print(result.output)
print(f"\nsteps={result.steps_taken}  tokens={result.tokens_used:,}")
