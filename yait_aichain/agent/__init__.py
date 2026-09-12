"""
agent
=====

``Agent`` — a conversation that can act.

The model is asked what to do, the world answers, the answer is appended, and
the loop goes round. It ends when the model replies without asking for an
action; everything else that can end it is a stop condition you pass in.

Quick start
-----------
::

    from yait_aichain.models import Model
    from yait_aichain.tools  import BraveSearchTool, MarkItDownTool
    from yait_aichain.agent  import Agent, step_count, cost_budget

    agent = Agent(
        Model("claude-opus-5"),
        tools     = [BraveSearchTool(), MarkItDownTool()],
        stop_when = [step_count(20), cost_budget(0.50)],
    )

    result = agent.run("Find the latest breakthroughs in fusion energy "
                       "and write a three-paragraph summary.")

    print(result.output)
    print(result.stopped_by)          # "answered" — or the ceiling that fired
    print(f"{result.tokens_used:,} tokens · ${result.cost:.4f}")

Two parameters
--------------
``mode`` — is the sequence frozen?

* ``"agile"`` (default): the next step is decided from what just happened.
* ``"waterfall"``: the model writes a plan first and holds to it. The freeze
  is what makes the plan cacheable and the bill predictable.

``team`` — who does the work?

* ``None`` (default): nobody else; the ``agent`` action does not exist.
* ``[agents]``: delegate, but only to these named workers.
* ``"auto"``: describe and spawn workers as the task needs them.

Six combinations, each a distinct requirement::

    Agent(model, tools)                                  # the loop
    Agent(model, tools, mode="waterfall")                # plan, then hold to it
    Agent(model, tools, team=[researcher, analyst])      # route across a cast
    Agent(model, tools, mode="waterfall", team=[...])    # plan across a cast
    Agent(model, tools, team="auto")                     # spawn as needed
    Agent(model, tools, mode="waterfall", team="auto")   # design a team, run it

Stopping
--------
The ordinary exit needs no condition — the model answers. ``stop_when`` holds
everything else, and the two kinds must not be confused: a ``check`` that
passes is a **success** with harness-verified evidence, a ceiling that is
reached is a **failure**. Whichever fired is named in ``result.stopped_by``, so
"finished" and "gave out" never look alike from the outside.

::

    stop_when = [
        step_count(30),                                  # ceiling
        token_budget(150_000),                           # ceiling
        cost_budget(0.50),                               # ceiling
        check(lambda s: s["steps"] > 2, name="enough"),  # success, verified
    ]

What it is made of
------------------
``Skill`` is the conversation, ``Tool`` is what gets called, ``Pool`` is one
action repeated over a list, and a delegated sub-task is another ``Agent``. The
agent composes the library rather than reimplementing it.
"""

from ._agent   import (Agent, step_count, token_budget, cost_budget, check,
                       nudge, stalled, repeating)
from ._journal import Journal, JournalEntry, evidence
from ._memory  import AgentMemory, MemoryBackend, InMemoryBackend, FileBackend
from ._result  import AgentResult
from ._swarm   import (Acceptance, Beacon, beacon, ATTENTION, COORDINATION,
                       acceptance_from)

__all__ = [
    "Agent",
    "AgentResult",
    # Swarm coordination
    "Acceptance",
    "acceptance_from",
    "Beacon",
    "beacon",
    "ATTENTION",
    "COORDINATION",
    # Stop conditions
    "step_count",
    "token_budget",
    "cost_budget",
    "check",
    "nudge",
    "stalled",
    "repeating",
    # Journal
    "Journal",
    "JournalEntry",
    "evidence",
    # Memory — no longer wired into the loop by default; kept for eviction,
    # which is where it earns its place once a conversation outgrows the window.
    "AgentMemory",
    "MemoryBackend",
    "InMemoryBackend",
    "FileBackend",
]
