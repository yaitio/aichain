"""
15_agent_orchestrator.py — Orchestrator agent spawns sub-agents.

The orchestrator receives a high-level task and delegates each
sub-topic to a child agent via the built-in spawn_agent tool.

  Orchestrator
      ├── spawns SubAgent → researches "AI agents"
      ├── spawns SubAgent → researches "vector databases"
      └── spawns SubAgent → researches "LLM inference"
              ↓
      collects results, compiles final summary

Required env vars:
    ANTHROPIC_API_KEY
    PERPLEXITY_API_KEY
"""

import os
from yait_aichain.models import Model
from yait_aichain.agent  import Agent, step_count
from yait_aichain.tools import searchPerplexity

# ``team="auto"`` lets the agent describe and spawn workers as it needs them.
# Delegation here is first of all about *context*: each sub-agent researches a
# topic in its own conversation and hands back a conclusion, so the coordinator
# gains one turn per topic instead of a whole search history.
orchestrator = Agent(
    Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    tools     = [searchPerplexity(api_key=os.getenv("PERPLEXITY_API_KEY"))],
    team      = "auto",
    stop_when = [step_count(15)],
    instructions = (
        "You are a research coordinator. "
        "Given a multi-topic research task, delegate one worker per topic and "
        "collect their conclusions into a final structured summary."
    ),
)

result = orchestrator.run(
    "Research these three topics and give me a 2-sentence summary of each: "
    "1) AI agents in 2025, "
    "2) vector databases market, "
    "3) LLM inference optimisation."
)

print(result.output)
print(f"\nsteps={result.steps_taken}  tokens={result.tokens_used:,}")
