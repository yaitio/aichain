"""
24_chain_webhook.py — A run parks until an outside event resumes it.

Keys: none. Runs offline; the halves would be two serverless invocations sharing only a store.
"""

import os
import tempfile
from pathlib import Path

from yait_aichain import Chain
from yait_aichain.state import FileStore
from yait_aichain.tools import gate, tool

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"


@tool(risk="read")
def lookup_order(order_id: str) -> float:
    """The amount paid for an order."""             # no model reads it here, so one line will do
    return 42.0


@tool(risk="financial")
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order.

    Parameters
    ----------
    order_id : str
        The order to refund.
    amount : float
        Dollars to refund.
    """
    print(f"Refunded ${amount} for order {order_id}.")
    return "refund issued"


def build() -> Chain:                  # the same in both invocations: shared code, not an object
    return Chain([(lookup_order, "amount"),
                  (gate(issue_refund, reason="A manager approves the refund."), "refund")],
                 store=FileStore(str(OUT / "runs")))


def start(order_id: str) -> str:                          # invocation 1: the request
    parked = build().run(variables={"order_id": order_id})
    print(f"Paused — {parked.awaiting['reason']} (run {parked.run_id})")
    return parked.run_id


def webhook(run_id: str, approved: bool) -> None:          # invocation 2: the approval
    result = build().resume(run_id, signal={"approved": approved})
    print(f"Resumed with approved={approved} → {result.output!r}")


webhook(start("A-123"), approved=True)
webhook(start("A-124"), approved=False)
