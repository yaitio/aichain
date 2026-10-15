"""
32_agent_pause.py — An agent waits for an outside event, and another process resumes it.

Keys: none. Runs offline: a scripted model asks for a refund the gate holds.
"""

import os
import tempfile
from pathlib import Path

from yait_aichain import Agent
from yait_aichain.state import FileStore
from yait_aichain.testing import ScriptedModel, call
from yait_aichain.tools import gate, tool

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"


@tool(risk="financial")
def issue_refund(amount: float) -> str:
    """Refund the customer.

    Parameters
    ----------
    amount : float
        Dollars to refund.
    """
    print(f"refunded ${amount}")
    return f"refunded {amount}"


def build(replies: list) -> Agent:
    """The same agent in both processes; only the store is shared."""
        # A real application passes Model("..."); the script plays the model here.
    return Agent(ScriptedModel(replies),
                 tools=[gate(issue_refund, reason="A manager approves refunds.")],
                 store=FileStore(str(OUT / "agent-runs")))


parked = build([call("gate_issue_refund", amount=42)]).run("Refund order #123.")
print(f"paused: {parked.awaiting['reason']} (run {parked.run_id})")

# Later, in another process — a webhook, a queue consumer:
result = build(["Done: the refund went through."]).resume(parked.run_id,
                                                          signal={"approved": True})
print(f"answer: {result.output}")
print(f"stopped_by: {result.stopped_by}")
