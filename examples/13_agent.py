"""
13_agent.py — Basic agent with one tool.

A conversation that can act: the agent is asked what to do, the world answers,
and the loop goes round until it replies without asking for anything more.

``stop_when`` holds everything else that can end a run. Reaching a ceiling is
not success, so ``stopped_by`` tells you which it was.

Required env vars:
    ANTHROPIC_API_KEY
    PERPLEXITY_API_KEY
"""

import os
from yait_aichain.models import Model
from yait_aichain.agent  import Agent, step_count
from yait_aichain.tools import searchPerplexity

agent = Agent(
    Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    tools     = [searchPerplexity(api_key=os.getenv("PERPLEXITY_API_KEY"))],
    stop_when = [step_count(5)],
)

result = agent.run("What are the top 3 vector databases in 2025? Give a one-line description of each.")

print(result.output)
print(f"\nstopped_by={result.stopped_by}  steps={result.steps_taken}  "
      f"tokens={result.tokens_used:,}  ${result.cost or 0:.4f}")
