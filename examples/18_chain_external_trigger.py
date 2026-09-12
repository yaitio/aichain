"""
18_chain_external_trigger.py — Pause a run until an EXTERNAL trigger resumes it.

A pause is not only for humans: a webhook, a cron tick, or a queue message can
resume a run just as well. Here a chain looks an order up and issues a refund,
but the refund is wrapped in a Gate — the run parks itself and the approval
arrives LATER, from a completely separate piece of code.

The two halves below would be two separate serverless invocations (two AWS
Lambda calls, say) that share NOTHING but a state store: here a FileStore
directory, in production a store on shared infrastructure. The trigger needs
only the run_id and the store — no reference to the original chain object.

    Invocation 1  chain.run(...)                     → SuspendedResult, parked
    ...later...
    Invocation 2  approval_webhook(run_id, approved) → chain.resume(...)

Why a Chain and not an Agent: a chain's position is its state, so it can be
parked and picked up by another process. An agent's state is its conversation,
so it has no suspend — an agent that needs a human uses ``approve=`` instead
(see 20_observability.py).

Runs offline — no model, no key:
    python examples/18_chain_external_trigger.py

Required env vars:
    (none)
"""

import os
import tempfile

from yait_aichain.chain import Chain
from yait_aichain.state import FileStore, SuspendedResult
from yait_aichain.tools import Tool, Gate

# A shared, persistent store. In production it lives on shared infrastructure,
# so a different process can pick the run up.
STORE_DIR = os.environ.get("AICHAIN_RUNS",
                           os.path.join(tempfile.gettempdir(), "aichain_runs"))


class LookupOrder(Tool):
    name        = "lookup_order"
    description = "Look an order up."
    parameters  = {"type": "object",
                   "properties": {"order_id": {"type": "string"}},
                   "required": ["order_id"]}

    def run(self, order_id, **kwargs):
        return {"order_id": order_id, "amount": 42.0}


class IssueRefund(Tool):
    name        = "issue_refund"
    description = "Issue a refund to the customer."
    parameters  = {"type": "object",
                   "properties": {"order_id": {"type": "string"},
                                  "amount":   {"type": "number"}},
                   "required": ["order_id", "amount"]}

    def run(self, order_id, amount, **kwargs):
        print(f"💳  Refunded ${amount} for order {order_id}.")
        return "refund issued"


def build_chain() -> Chain:
    """Built identically in BOTH invocations — the code is shared, the object is not."""
    return Chain(
        steps=[
            (LookupOrder(), "order"),
            (Gate(IssueRefund(),
                  reason="A manager must approve the refund.",
                  hint={"on": "webhook"}),          # recorded for the scheduler, never acted on
             "refund"),
        ],
        store=FileStore(STORE_DIR),
    )


# ── Invocation 1 — start the run; it parks at the gate ───────────────────────
def start(order_id: str) -> str:
    result = build_chain().run(variables={"order_id": order_id})
    assert isinstance(result, SuspendedResult)
    print(f"⏸  Paused — {result.awaiting['reason']}")
    print(f"   run_id = {result.run_id}  (parked in {STORE_DIR})")
    print("   no refund yet — note there is no 💳 line above")
    return result.run_id


# ── Invocation 2 — the external trigger resumes the run ──────────────────────
def approval_webhook(run_id: str, approved: bool):
    # A fresh chain, as a different process would build it — it shares ONLY the store.
    output = build_chain().resume(run_id, signal={"approved": approved})
    print(f"▶  Resumed by the trigger (approved={approved}) → {output!r}")
    return output


if __name__ == "__main__":
    run_id = start("A-123")
    # ...time passes; a webhook delivers the manager's decision...
    approval_webhook(run_id, approved=True)

    # A refusal resumes the same way; the wrapped tool is skipped, not run.
    approval_webhook(start("A-124"), approved=False)
