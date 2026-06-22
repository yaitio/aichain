"""
20_observability.py — the step boundary (1.4.4): hooks, events, permissions.

Runs with NO API key and NO network — a tiny scripted stand-in plays the
orchestrator so the output is identical every time and you can verify it. The
point of this example is the *harness* behavior around the model, not the model:

  1. A Tracer hook records a structured event at every boundary.
  2. A PermissionPolicy gates a FINANCIAL tool: the run PAUSES for approval
     (reusing suspend/resume), then you resume it with the decision.
  3. The library's logging is routed to a handler the app controls.

Run it:
    python examples/20_observability.py
"""

import json
import logging
import sys

from yait_aichain.agent  import Agent
from yait_aichain.tools  import Tool, PermissionPolicy, FINANCIAL
from yait_aichain        import Tracer
from yait_aichain.state  import SuspendedResult


# ── A real tool, tagged with its risk class ─────────────────────────────────────

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


# ── A scripted stand-in for the orchestrator LLM (so this runs offline) ─────────
# A real Agent uses Model("gpt-4o-mini", ...) here; we feed canned plan/action/
# reflection JSON through the same to_request → send → from_response seam.

class ScriptedModel:
    name = "scripted-orchestrator"

    def __init__(self, replies):
        self._replies, self._i = replies, 0

    def to_request(self, messages, output):
        return ("/noop", {})

    class _Client:
        def __init__(self, outer): self._outer = outer
        def _auth_headers(self): return {}
        def send(self, path, body, headers):
            o = self._outer
            reply = o._replies[min(o._i, len(o._replies) - 1)]
            o._i += 1
            return json.dumps({"text": reply,
                               "usage": {"input_tokens": 6, "output_tokens": 6}})

    @property
    def client(self):
        if not hasattr(self, "_c"): self._c = self._Client(self)
        return self._c

    def from_response(self, response, output):
        return response["text"]


# The orchestrator's three replies for a one-step plan that calls issue_refund:
SCRIPT = [
    json.dumps({"steps": [{"id": 1, "type": "tool",
                           "tool_name": "issue_refund", "goal": "Refund order #123"}]}),
    json.dumps({"type": "tool", "tool_name": "issue_refund", "kwargs": {"amount": 42}}),
    json.dumps({"decision": "final_answer", "assessment": "done",
                "final_answer": "Refund processed."}),
]


# ── 1. Route the library's logs to a handler we control ─────────────────────────

logging.basicConfig(level=logging.INFO, format="    log │ %(message)s",
                    stream=sys.stdout)        # → stdout so it reads in order

# ── 2. Build the agent with a Tracer (events) and a PermissionPolicy (gate) ─────

tracer = Tracer()
agent  = Agent(
    orchestrator = ScriptedModel(SCRIPT),
    tools        = [IssueRefund()],
    hooks        = [tracer],                                  # observability
    permissions  = PermissionPolicy({"financial": "approve"}),  # governance
    max_steps    = 1,
)

print("\n=== run() — the financial tool pauses for approval ===")
result = agent.run("Issue a refund of $42 for order #123.")

print("\n=== event timeline (from the Tracer hook) ===")
for e in tracer.events:
    print(f"    {e}")

assert isinstance(result, SuspendedResult), "expected the run to pause for approval"
print(f"\n⏸  Paused: {result.awaiting['reason']}")
print("    (no refund executed yet — note there is no 💳 line above)")

# ── 3. A decision arrives later; resume the SAME run with it ─────────────────────

print("\n=== resume(approved=True) — now the tool actually runs ===")
final = agent.resume(result.run_id, signal={"approved": True})
print(f"\n✓  success={final.success}  output={final.output!r}")

print("\nTry it: change 'approve' to 'deny' above — the refund is refused and the")
print("run still completes with a denial result (every tool call returns a result).")
