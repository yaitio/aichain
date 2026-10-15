"""
23_chain_human.py — A person in the loop: the chain parks, the draft is not re-made.

Keys: none. Runs offline: a scripted model writes the draft.
"""

import os
import tempfile
from pathlib import Path

from yait_aichain import Chain, Skill
from yait_aichain.state import FileStore
from yait_aichain.testing import ScriptedModel
from yait_aichain.tools import tool, wait

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"


@tool(risk="external")
def send_reply(reply: str) -> str:
    """Send the approved reply to the customer.

    Parameters
    ----------
    reply : str
        The text to send.
    """
    print(f"sent: {reply}")
    return "sent"


# A real application passes Model("..."); the script plays the model here.
model = ScriptedModel(["We are sorry your order was late. A refund is on its way."])
chain = Chain([
    Skill(model, prompt="Draft a short reply to this complaint: {complaint}", name="draft"),
    wait(reason="A person approves or edits the draft.", resume_with={"reply": "str"}),
    send_reply,
], store=FileStore(str(OUT / "runs")))   # survives the process, so resume() can be elsewhere

parked = chain.run(variables={"complaint": "My order arrived two weeks late."})
print(f"draft: {parked.document['variables']['draft']}")
print(f"paused: {parked.awaiting['reason']}")

# A real application gets this from a UI or a queue, often in another process.
edited = parked.document["variables"]["draft"] + " Thank you for your patience."
chain.resume(parked.run_id, signal={"reply": edited})
print(f"draft calls: {len(model.requests)}")
