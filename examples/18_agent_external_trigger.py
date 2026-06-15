"""
18_agent_external_trigger.py — Suspend an agent until an EXTERNAL trigger.

A pause is not only for humans: a webhook, a cron tick, or a queue message can
all resume a run. Here an agent must issue a refund, but the refund action is
wrapped in a Gate — the agent pauses for an approval decision that arrives
LATER, from a completely separate piece of code.

The two halves below would be two separate serverless invocations (e.g. two
AWS Lambda calls) that share NOTHING but a state store (here a FileStore
directory; in production an S3/DynamoDB-backed store). The trigger needs only
the run_id and the store — no reference to the original agent object.

    Invocation 1  agent.run(...)   → SuspendedResult (parked in the store)
    ...later...
    Invocation 2  approval_webhook(run_id, approved)  → agent.resume(...)

Required env vars:
    OPENAI_API_KEY
"""

import os
import tempfile
from yait_aichain.agent import Agent
from yait_aichain.models import Model
from yait_aichain.tools  import Tool, Gate
from yait_aichain.state  import FileStore, SuspendedResult

# A shared, persistent store. In production: S3Store / DynamoStore on shared
# infrastructure, so a different process/Lambda can pick the run up.
STORE_DIR = os.path.join(tempfile.gettempdir(), "aichain_runs")


class IssueRefund(Tool):
    name        = "issue_refund"
    description = "Issue a refund to the customer."
    parameters  = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        print("💳  Refund issued.")
        return "refund issued"


def build_agent() -> Agent:
    """Build the agent — done identically in BOTH invocations (code is shared)."""
    return Agent(
        orchestrator = Model("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
        tools        = [Gate(IssueRefund(),
                             reason="A manager must approve the refund.",
                             resume_with={"approved": "bool"})],
        store        = FileStore(STORE_DIR),
        max_steps    = 2,
    )


# ── Invocation 1 — start the run; it pauses at the approval gate ─────────────
def start() -> str:
    agent  = build_agent()
    result = agent.run("Issue a refund for order #123.")
    assert isinstance(result, SuspendedResult)
    print(f"⏸  Agent paused — {result.awaiting['reason']}")
    print(f"   run_id = {result.run_id}  (saved in {STORE_DIR})")
    return result.run_id


# ── Invocation 2 — the external trigger (webhook / cron) resumes the run ─────
def approval_webhook(run_id: str, approved: bool):
    # A fresh agent in a different process — it shares ONLY the store.
    agent  = build_agent()
    result = agent.resume(run_id, signal={"approved": approved})
    print(f"\n▶  Resumed by external trigger (approved={approved})")
    print(f"   success={result.success}  output={result.output!r}")


if __name__ == "__main__":
    run_id = start()
    # ...time passes; a separate webhook delivers the manager's decision...
    approval_webhook(run_id, approved=True)
