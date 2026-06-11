"""
agent
=====

Public API for the aichain 2.0 agent layer.

An ``Agent`` is an autonomous execution engine that plans and executes
complex tasks by combining :class:`~tools.Tool` instances (for external
actions) and dynamically-constructed LLM skills (for reasoning and
generation).  It requires only a task description at run time — it figures
out the steps itself.

Two execution modes
-------------------
``"waterfall"``
    The orchestrator creates a fixed plan upfront.  Steps run in order.
    Reflection after each step can trigger a retry or a hard stop on fatal
    failure, but the plan itself never changes.

``"agile"``
    Same structure, but the reflection phase can also replan — revising
    remaining steps or jumping back to an earlier one.  Best for open-ended
    or exploratory tasks where the path isn't fully knowable in advance.

Quick start
-----------
::

    from models import Model
    from tools  import BraveSearchTool, MarkItDownTool
    from agent  import Agent

    agent = Agent(
        orchestrator = Model("claude-opus-4-6"),
        tools        = [BraveSearchTool(), MarkItDownTool()],
        mode         = "waterfall",
        max_steps    = 6,
        max_tokens   = 40_000,
        name         = "research_agent",
    )

    result = agent.run(
        task = "Find the latest breakthroughs in fusion energy and write "
               "a 3-paragraph summary in English.",
    )

    if result:
        print(result.output)
        print(f"\\nUsed {result.tokens_used:,} tokens across {result.steps_taken} steps.")
    else:
        print("Agent failed:", result.error)

Agent with a custom persona
----------------------------
::

    agent = Agent(
        orchestrator = Model("gpt-4o"),
        persona      = (
            "You are a senior financial analyst specialising in tech equities. "
            "Always cite data sources and flag information older than 30 days."
        ),
        tools        = [BraveSearchTool()],
        mode         = "waterfall",
    )

Separate orchestrator and executor models
-----------------------------------------
The orchestrator handles all planning and reflection.  Executor models
handle the actual LLM skill steps.  Using a cheaper/faster model for
execution saves tokens::

    agent = Agent(
        orchestrator = Model("claude-opus-4-6"),       # strong reasoner
        executors    = [Model("gpt-4o"), Model("gemini-2.0-flash")],
        tools        = [BraveSearchTool()],
        mode         = "agile",
        max_tokens   = 80_000,
    )

Initial variables
-----------------
Pass seed data into the agent's memory before the first step::

    result = agent.run(
        task      = "Translate the latest AI news into the target language.",
        variables = {"language": "Ukrainian", "topic": "artificial intelligence"},
    )

Persistent memory across runs
------------------------------
Use :class:`FileBackend` to checkpoint state between separate ``run()``
calls::

    from agent import AgentMemory, FileBackend

    memory = AgentMemory(backend=FileBackend("~/.my_agent_state.json"))
    agent  = Agent(..., memory=memory)

    result = agent.run(task="Step 1 of multi-run workflow…")
    memory.flush()     # persist the final state to disk

    # On the next invocation, load the saved state:
    memory2 = AgentMemory(backend=FileBackend("~/.my_agent_state.json"))
    agent2  = Agent(..., memory=memory2)
    result2 = agent2.run(task="Step 2, with access to previous results…")

Inspecting results
------------------
::

    result = agent.run("Summarise the top 3 results for 'quantum computing 2025'")

    # Full step trace
    for rec in result.history:
        print(f"Step {rec['step']+1} [{rec['action_type']}] {rec['step_goal']}")
        print(f"  Assessment : {rec['reflection']['assessment']}")
        print(f"  Stored as  : {rec['stored_as']}")
        print(f"  Tokens     : {rec['tokens']:,}")

    # Final memory state
    print(result.memory)
"""

from ._agent  import Agent
from ._memory import AgentMemory, MemoryBackend, InMemoryBackend, FileBackend
from ._result import AgentResult

__all__ = [
    "Agent",
    "AgentMemory",
    "AgentResult",
    # Memory backends
    "MemoryBackend",
    "InMemoryBackend",
    "FileBackend",
]
