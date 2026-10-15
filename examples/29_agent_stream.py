"""
29_agent_stream.py — One ordered stream of events: what an interface is built on.

Keys: none. Runs offline: a scripted model asks for a refund, then answers.
"""

from yait_aichain import Agent
from yait_aichain.testing import ScriptedModel, call
from yait_aichain.tools import tool


@tool(risk="financial")
def issue_refund(amount: float) -> str:
    """Refund the customer.

    Parameters
    ----------
    amount : float
        Dollars to refund.
    """
    return f"refunded {amount}"


# A real application passes Model("..."); the script plays the model here.
model = ScriptedModel([call("issue_refund", amount=42), "Refunded $42 to your card."])
agent = Agent(model, tools=[issue_refund],
              policy={"rules": {"financial": "approve"}, "approve": lambda request: True})

for event in agent.stream("Refund order #123."):
    detail = event.payload.get("text") or event.payload.get("granted", "")
    print(f"{event.type:<20} {event.name or '':<14} {detail}")
