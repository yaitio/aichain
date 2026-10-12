"""
The results of parallel calls go out the way each wire requires.

Gemini wants every functionResponse of one turn in one content, as many as
there were calls; it was sent one content per result, and any parallel turn
on Gemini answered 400. Chat Completions and the Responses API want a turn's
tool messages back to back; a tool that returned a picture put a user message
between them, and a parallel turn with an image answered 400 there. Two calls,
one with an image, on each family.
"""

import os
import unittest
import warnings

from yait_aichain import Model
from yait_aichain.models._calls import tool_result_turn

IMAGE = {"type": "image", "source": {"kind": "base64", "mime": "image/png", "data": "aGk="}}
CONVERSATION = [
    {"role": "user", "parts": [{"type": "text", "text": "look at both"}]},
    {"role": "assistant", "parts": [],
     "tool_calls": [{"id": "a", "name": "render", "arguments": {}},
                    {"id": "b", "name": "count", "arguments": {}}]},
    tool_result_turn("a", [IMAGE, "rendered"]),
    tool_result_turn("b", "3"),
]
OUT = {"modalities": ["text"], "format": {"type": "text"}}


def _body(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Model(name, api_key="k").to_request(CONVERSATION, OUT)[1]


class TestParallelResults(unittest.TestCase):

    def test_gemini_gets_one_content_with_every_response(self):
        contents = _body("gemini-2.5-flash")["contents"]
        responses = [c for c in contents if any("functionResponse" in p for p in c["parts"])]
        self.assertEqual(len(responses), 1)
        self.assertEqual(len(responses[0]["parts"]), 2)
        after = contents[contents.index(responses[0]) + 1:]
        self.assertTrue(any("inlineData" in p or "inline_data" in p for c in after for p in c["parts"]))

    def test_chat_completions_keeps_the_tool_messages_together(self):
        roles = [m["role"] for m in _body("deepseek-chat")["messages"]]
        first = roles.index("tool")
        self.assertEqual(roles[first:first + 3], ["tool", "tool", "user"])

    def test_the_responses_api_keeps_the_outputs_together(self):
        items = _body("gpt-5.4-mini")["input"]
        kinds = [i.get("type") or i.get("role") for i in items]
        first = kinds.index("function_call_output")
        self.assertEqual(kinds[first:first + 3], ["function_call_output", "function_call_output", "user"])


if __name__ == "__main__":
    unittest.main()
