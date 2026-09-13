"""
Driving the agent loop yourself needs nothing from a private module.

`Agent.step()` raises when a tool call has no result, and its message tells the
caller to append `tool_result_turn(call.id, result)`. That function lived only
in `yait_aichain.models._calls`: the library's own error message pointed at an
import path a user is told not to rely on. `llms.txt` teaches the external
loop, so what it teaches has to be importable from the public surface.
"""

import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")


class TestPublicDriverSurface(unittest.TestCase):

    def test_the_names_import_from_the_public_package(self):
        from yait_aichain.models import (ToolCall, ToolCallRequest,
                                         dangling_calls, tool_result_turn)
        call = ToolCall(id="c1", name="probe", arguments={"guess": 3})
        reply = ToolCallRequest(calls=(call,))
        history = [reply.as_turn()]
        self.assertEqual(dangling_calls(history), ["c1"])
        history.append(tool_result_turn(call.id, "lower"))
        self.assertEqual(dangling_calls(history), [])

    def test_the_error_message_names_a_public_function(self):
        import yait_aichain.models as models
        from yait_aichain import Agent, Model
        from yait_aichain.models import ToolCall, ToolCallRequest
        agent = Agent(Model("gpt-4o", api_key="k"))
        reply = ToolCallRequest(calls=(ToolCall(id="c9", name="t", arguments={}),))
        with self.assertRaises(ValueError) as ctx:
            agent.step([{"role": "user", "parts": ["hi"]}, reply.as_turn()])
        self.assertIn("tool_result_turn", str(ctx.exception))
        self.assertIn("tool_result_turn", models.__all__)


if __name__ == "__main__":
    unittest.main()
