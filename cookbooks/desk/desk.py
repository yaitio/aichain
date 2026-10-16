"""The support agent. run.py, handlers.py and simulate.py all build it here, the same way."""

import os
import sqlite3
from pathlib import Path

from yait_aichain import Agent, Model
from yait_aichain.state import FileStore
from yait_aichain.tools import ApprovalDecision, gate, tool

DATA = Path(__file__).parent / "data"
MODEL = os.getenv("DESK_MODEL", "claude-sonnet-5-5")
COLUMNS = ("id", "customer", "item", "amount", "status", "delivered_on", "refunded")
GUIDE = """Look the order up with lookup_order before anything else.
Refund up to $50 with refund. Above $50 use refund_with_approval: it asks a manager,
who may answer later; what it returns is after the manager decided. When a refund is
refused, tell the customer the reason given. Promise nothing you cannot do yourself."""


def _db():
    return sqlite3.connect(DATA / "shop.db", isolation_level=None, timeout=10)


@tool(risk="read")
def lookup_order(order_id: str) -> dict:
    """Look an order up: its customer, item, amount, status and delivery date.

    Parameters
    ----------
    order_id : str
        The order's id, such as A-1007.
    """
    row = _db().execute(f"SELECT {', '.join(COLUMNS)} FROM orders WHERE id = ?", (order_id,)).fetchone()
    return dict(zip(COLUMNS, row)) if row else {"error": f"no such order: {order_id}"}


@tool(risk="financial")
def refund(order_id: str, amount: float, reason: str) -> dict:
    """Refund an order, once, for at most what was paid.

    Parameters
    ----------
    order_id : str
        The order's id.
    amount : float
        Dollars to refund.
    reason : str
        Why, in the customer's words.
    """
    db = _db()
    db.execute("BEGIN IMMEDIATE")      # check and write as one: a retried call refunds once
    row = db.execute("SELECT amount, refunded FROM orders WHERE id = ?", (order_id,)).fetchone()
    why = ("no such order" if row is None else "already refunded" if row[1]
           else f"the order was ${row[0]:.2f}" if amount > row[0] else None)
    if why:
        db.execute("ROLLBACK")
        return {"refunded": False, "order_id": order_id, "why": why}
    db.execute("UPDATE orders SET refunded = 1, status = 'refunded' WHERE id = ?", (order_id,))
    db.execute("COMMIT")
    return {"refunded": True, "order_id": order_id, "amount": amount}


refund_with_approval = gate(refund, name="refund_with_approval", reason="Refunds over $50 need a manager",
                            resume_with={"approved": "bool", "note": "str"})


def up_to_50(request):
    """Support refunds up to $50; above that, only through the gate, which asks a manager."""
    if request.tool == "refund_with_approval" or float(request.arguments.get("amount", 0)) <= 50:
        return True
    return ApprovalDecision(False, "over $50: use refund_with_approval")


def tenant_key(context) -> str:
    """The tenant's own key when it has one, else the shared one. The run document
    stores the tenant's name, never this value."""
    tenant = (getattr(context, "tenant", None) or "").upper()
    return os.environ.get(f"ANTHROPIC_API_KEY_{tenant}") or os.environ["ANTHROPIC_API_KEY"]


def build_agent(hooks=None) -> Agent:
    return Agent(Model(MODEL, api_key=tenant_key),
                 tools=[lookup_order, refund, refund_with_approval],
                 instructions=(DATA / "policy.md").read_text() + "\n" + GUIDE,
                 policy={"rules": {"financial": "approve"}, "approve": up_to_50},
                 store=FileStore(str(DATA / "runs")), max_cost=0.10, name="desk", hooks=hooks)
