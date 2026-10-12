"""
An Anthropic agent with extended thinking survives its second step.

With thinking on, the assistant turn before a tool result must open with its
thinking blocks, unchanged. The turn the library kept held only the calls, so
`Agent(Model("claude-…", options={"reasoning": "high"}), tools=[…])` answered
HTTP 400 on its second request. Offline: parse → as_turn → build_request.
"""

import unittest

from yait_aichain import Model
from yait_aichain.models._calls import tool_result_turn

THINKING = {"type": "thinking", "thinking": "I should look it up.", "signature": "sig=="}
REDACTED = {"type": "redacted_thinking", "data": "opaque=="}
REPLY = {"content": [THINKING, REDACTED,
                     {"type": "text", "text": "Looking."},
                     {"type": "tool_use", "id": "tu_1", "name": "lookup", "input": {"q": "x"}}],
         "usage": {"input_tokens": 1, "output_tokens": 1}}
OUT = {"modalities": ["text"], "format": {"type": "text"}}


class TestThinkingComesBack(unittest.TestCase):

    def test_parse_then_as_turn_then_build_keeps_the_blocks_first_and_verbatim(self):
        m = Model("claude-sonnet-4-5", api_key="k", options={"reasoning": "high"})
        request = m.from_response(REPLY, OUT)
        history = [{"role": "user", "parts": [{"type": "text", "text": "find x"}]},
                   request.as_turn(), tool_result_turn("tu_1", "found")]
        _, body = m.to_request(history, OUT)
        assistant = next(msg for msg in body["messages"] if msg["role"] == "assistant")
        self.assertEqual(assistant["content"][:2], [THINKING, REDACTED])
        self.assertEqual([b["type"] for b in assistant["content"][2:]], ["text", "tool_use"])

    def test_without_thinking_nothing_is_added(self):
        m = Model("claude-sonnet-4-5", api_key="k")
        request = m.from_response({"content": REPLY["content"][2:]}, OUT)
        self.assertEqual(request.opaque, {})
        self.assertNotIn("opaque", request.as_turn())


if __name__ == "__main__":
    unittest.main()
