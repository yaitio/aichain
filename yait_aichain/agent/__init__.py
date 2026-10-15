"""
agent — ``Agent``, a conversation that can act.

The model is asked what to do, the world answers, the answer is appended, and
the loop goes round until the model replies without asking for an action or a
stop condition fires. ``mode`` says whether the sequence is planned up front,
``team`` who does the work. See docs/agents/overview.md.
"""

from ._agent   import (Agent, step_count, token_budget, cost_budget, check,
                       nudge, stalled, repeating, verified)
from ._journal import Journal, JournalEntry, evidence
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
    "verified",
    # Journal
    "Journal",
    "JournalEntry",
    "evidence",
    # Memory — a standalone key-value store; the agent loop does not use it.
]


def __getattr__(name):
    # A name this package had raises with what to write instead.
    from .._hints import AGENT_NAMES
    if name in AGENT_NAMES:
        raise ImportError(f"cannot import name {name!r} from 'yait_aichain.agent' — "
                          f"{AGENT_NAMES[name]}", name=name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
