"""
35_policy.py — Who allows a risky call, and how a call is changed before it runs.

Keys: none. Runs offline: a scripted model asks for two refunds.
"""

import dataclasses

from yait_aichain import Agent, Interceptor, Tracer
from yait_aichain.testing import ScriptedModel, call
from yait_aichain.tools import ApprovalDecision, tool


@tool(risk="financial")
def issue_refund(amount: float, card: str) -> str:
    """Refund the customer to a card.

    Parameters
    ----------
    amount : float
        Dollars to refund.
    card : str
        The card number.
    """
    print(f"${amount:g} to card {card}: executed")
    return f"refunded {amount}"


class MaskCard(Interceptor):
    """Runs before the policy: nothing downstream sees the full number."""

    def before(self, call, run):
        masked = "****" + call.arguments["card"][-4:]
        return dataclasses.replace(call, arguments={**call.arguments, "card": masked})


def manager(request):
    if request.arguments["amount"] <= 50:
        return True
    return ApprovalDecision(False, "over $50 needs a ticket")


for amount in (42, 500):
    # A real application passes Model("..."); the script plays the model here.
    model = ScriptedModel([call("issue_refund", amount=amount, card="4111111111114242"), "Done."])
    tracer = Tracer()
    agent = Agent(model, tools=[issue_refund], hooks=[tracer], policy={
        "rules": {"financial": "approve"}, "approve": manager, "interceptors": [MaskCard()]})
    agent.run(f"Refund ${amount} to the card on file.")
    decided = next(e for e in tracer.events if e.type == "approval.decided").payload
    if not decided["granted"]:
        print(f"${amount}: refused: {decided['reason']}")
