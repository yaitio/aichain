"""
The step() seam's messages and state cross a JSON round trip between steps.

The first adaptation put a dataclass in the state and `json.dumps` failed;
with `default=str` the dedupe never matched and the list grew every step; a
saved turn's calls could not be rebuilt.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent                                  # noqa: E402
from yait_aichain.models import Model                                 # noqa: E402
from yait_aichain.models._calls import ToolCallRequest, tool_result_turn  # noqa: E402
from test_loop import Echo, _calls, _model                            # noqa: E402


class TestRoundTrip(unittest.TestCase):

    def test_three_steps_through_json(self):
        # gpt-5.5 declines temperature: an adaptation every step.
        m = Model("gpt-5.5", api_key="k", options={"temperature": 0.3})
        m.client = _model(_calls(("c1", "echo", {"value": "x"})), name="gpt-5.5").client
        agent = Agent(m, tools=[Echo()], verbose=0)
        messages, state = agent.opening("go"), agent.new_state()
        for _ in range(3):
            reply = agent.step(messages, state)
            messages.append(reply.as_turn())
            call = ToolCallRequest.from_turn(messages[-1]).calls[0]
            result, error = agent.execute(call, state)
            messages.append(tool_result_turn(call.id, result if error is None else error))
            messages = json.loads(json.dumps(messages))
            state = json.loads(json.dumps(state))
        self.assertTrue(state["adaptations"])                  # there was something to carry
        self.assertEqual(len(state["adaptations"]), len({json.dumps(a, sort_keys=True)
                                                         for a in state["adaptations"]}))

    def test_from_turn_inverts_as_turn(self):
        req = ToolCallRequest.from_turn({"role": "assistant", "parts": [{"type": "text", "text": "hm"}],
                                         "tool_calls": [{"id": "1", "name": "t", "arguments": {"a": 1}}]})
        self.assertEqual((req.text, req.calls[0].name, req.calls[0].arguments), ("hm", "t", {"a": 1}))
        self.assertEqual(ToolCallRequest.from_turn(req.as_turn()), req)


if __name__ == "__main__":
    unittest.main()
