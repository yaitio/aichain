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

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models import Model
from agent  import Agent
from tools  import searchPerplexity

orchestrator = Agent(
    orchestrator = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    tools        = [searchPerplexity(api_key=os.getenv("PERPLEXITY_API_KEY"))],
    allow_spawn  = True,
    max_steps    = 15,
    mode         = "agile",
    persona = (
        "You are a research coordinator. "
        "When given a multi-topic research task, spawn one sub-agent per topic "
        "via spawn_agent(task=..., tools=['perplexity_search']). "
        "Collect all sub-agent results and compile a final structured summary."
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
