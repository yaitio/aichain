"""
13_agent.py — Basic agent with one tool.

The agent plans, searches, reflects, and stops autonomously.
Watch the steps it takes to complete the task.

Required env vars:
    ANTHROPIC_API_KEY
    PERPLEXITY_API_KEY
"""

import os
from yait_aichain.models import Model
from yait_aichain.agent  import Agent
from yait_aichain.tools import searchPerplexity

agent = Agent(
    orchestrator = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    tools        = [searchPerplexity(api_key=os.getenv("PERPLEXITY_API_KEY"))],
    max_steps    = 5,
)

result = agent.run("What are the top 3 vector databases in 2025? Give a one-line description of each.")

print(result.output)
print(f"\nsteps={result.steps_taken}  tokens={result.tokens_used:,}")
