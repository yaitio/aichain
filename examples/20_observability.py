"""
20_observability.py — The step boundary: hooks, events, and an approval gate.

Runs with NO API key and NO network — a tiny scripted stand-in plays the model
so the output is identical every time and you can check it. The point is the
harness around the model, not the model:

  1. A Tracer hook records a structured event at every boundary.
  2. A PermissionPolicy marks a FINANCIAL tool as needing approval, and
     ``approve=`` is who answers. The run does not pause: the approver is an
     ordinary callable, so the wait lives wherever your application already
     waits — a console prompt, a queue, a UI.
  3. A refusal carries its reason back to the model as a tool result.
  4. The library's logging goes to a handler the application controls.

Run it:
    python examples/20_observability.py

Required env vars:
    (none)
"""

import json
import logging
import sys

from yait_aichain import Tracer
from yait_aichain.agent import Agent
from yait_aichain.tools import (Tool, PermissionPolicy, ApprovalDecision,
                                FINANCIAL)
# The stand-in model below speaks in the library's own reply types. A real
# Model produces these itself; nothing outside a test double imports them.
from yait_aichain.models._calls import ToolCall, ToolCallRequest


# ── A real tool, tagged with its risk class ─────────────────────────────────

class IssueRefund(Tool):
    name        = "issue_refund"
    description = "Issue a refund to the customer."
    risk        = FINANCIAL                       # ← the PermissionPolicy reads this
    parameters  = {"type": "object",
                   "properties": {"amount": {"type": "number"}},
                   "required": ["amount"]}

    def run(self, amount, options=None):
        print(f"    💳  IssueRefund.run() executed — refunded ${amount}")
        return f"refunded {amount}"


# ── A scripted stand-in for the model (so this runs offline) ────────────────
# A real Agent takes Model("gpt-4o-mini") here. This one goes through the same
# to_request → client.send → from_response seam and replays a script.

class ScriptedModel:
    name = "scripted"

    def __init__(self, replies):
        self.replies = replies
        self.client  = self._Client(self)

    class _Client:
        def __init__(self, outer):
            self.outer, self.i = outer, 0

        def _auth_headers(self):
            return {}

        def send(self, path, body, headers):
            i = min(self.i, len(self.outer.replies) - 1)
            self.i += 1
            return json.dumps({"i": i, "usage": {"input_tokens": 6,
                                                 "output_tokens": 6}})

    def to_request(self, messages, output, tools=None):
        return ("/scripted", {})

    def from_response(self, response, output):
        return self.replies[response["i"]]


def script(amount):
    """First turn: call the refund tool. Second turn: answer in prose."""
    return ScriptedModel([
        ToolCallRequest(calls=(ToolCall(id="call-1", name="issue_refund",
                                        arguments={"amount": amount}),)),
        "Done — I have reported the outcome of the refund.",
    ])


# ── Who answers when the policy says "approve" ──────────────────────────────
# Handed the tool, its risk class and the arguments it would run with:
# approving a name is approving nothing, the amount is the decision.

def manager(request):
    amount = request.arguments["amount"]
    print(f"    🙋 approver asked: {request.tool}({amount}) [{request.risk}]")
    if amount <= 50:
        return True
    return ApprovalDecision(False, "refunds over $50 need a ticket number")


logging.basicConfig(level=logging.INFO, format="    log │ %(message)s",
                    stream=sys.stdout)              # stdout, so it reads in order
policy = PermissionPolicy({"financial": "approve"})


def run(amount):
    tracer = Tracer()
    agent  = Agent(script(amount),
                   tools       = [IssueRefund()],
                   hooks       = [tracer],          # observability
                   permissions = policy,            # governance
                   approve     = manager)           # who answers "approve"
    result = agent.run(f"Refund ${amount} for order #123.")

    print("\n    event timeline:")
    for e in tracer.events:
        detail = e.payload.get("reason") or e.payload.get("granted", "")
        print(f"      {e.type:<22} {e.name or '':<14} {detail}")
    print(f"\n    success={result.success}  stopped_by={result.stopped_by}")
    return result


print("\n=== $42 — within the approver's limit, the tool runs ===")
run(42)

print("\n=== $500 — refused, and the model is told why ===")
run(500)
print("    (no 💳 line in this run: the refund never executed)")
