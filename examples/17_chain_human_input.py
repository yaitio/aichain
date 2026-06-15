"""
17_chain_human_input.py — Human-in-the-loop: pause a chain for manual input.

Step 1  GPT drafts a reply to a customer complaint.
Step 2  Wait — the chain PAUSES and asks a human to approve or edit the draft.
Step 3  A tool "sends" the human-approved reply.

`chain.run()` returns a SuspendedResult instead of a final answer: the run is
parked in the chain's store. A human reads the draft, types the final text,
and `chain.resume(run_id, signal=...)` continues from exactly where it paused —
step 1 is NOT re-run (its output is already in the saved variables).

Required env vars:
    OPENAI_API_KEY
"""

import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain
from yait_aichain.tools  import Tool, Wait
from yait_aichain.state  import SuspendedResult


# A tiny tool that "sends" the approved reply (here it just prints it).
class SendReply(Tool):
    name        = "send_reply"
    description = "Send the approved reply to the customer."
    parameters  = {
        "type": "object",
        "properties": {"reply": {"type": "string"}},
        "required": ["reply"],
    }

    def run(self, reply, **kwargs):
        print(f"\n📤  Sent to customer:\n{reply}")
        return "sent"


draft = Skill(
    model = Model("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    input = {"messages": [{"role": "user", "parts": [
        "Draft a short, polite support reply to this complaint:\n\n{complaint}"
    ]}]},
    name  = "draft",
)

chain = Chain(steps=[
    (draft, "draft"),
    Wait(reason="A human must approve or edit the draft before sending.",
         resume_with={"reply": "str"}),
    (SendReply(), "confirmation"),
])

# ── Run until it pauses for the human ───────────────────────────────────────
result = chain.run(variables={
    "complaint": "My order arrived two weeks late and the box was damaged.",
})

if isinstance(result, SuspendedResult):
    print("⏸  Chain paused —", result.awaiting["reason"])
    print("\n--- GPT draft ---")
    print(result.document["variables"]["draft"])

    # A human reviews and provides the final wording.
    edited = input("\n✏️  Edit/approve the reply (press Enter to keep the draft): ")
    final  = edited.strip() or result.document["variables"]["draft"]

    # ── Resume from the pause with the human's decision ─────────────────────
    chain.resume(result.run_id, signal={"reply": final})
    # step 1 (the draft) is not re-run — it is already in the saved variables.
else:
    print(result)
