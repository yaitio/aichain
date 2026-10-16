"""
What a provider attaches to a tool-calling turn comes back with it.

Gemini 3 signs each function call (`thoughtSignature` on the part) and
answers 400 when the call returns without it; a call rebuilt from its name
and arguments has none. MiMo asks for a turn's `reasoning_content` to be kept
with its tool calls in a multi-turn loop. Offline: parse → as_turn → build.
"""

import unittest
import warnings

from yait_aichain import Model
from yait_aichain.models._calls import tool_result_turn

OUT = {"modalities": ["text"], "format": {"type": "text"}}
ASK = {"role": "user", "parts": [{"type": "text", "text": "weather?"}]}


def _round_trip(name, response):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Model(name, api_key="k")
        request = m.from_response(response, OUT)
        call_id = request.calls[0].id
        return m.to_request([ASK, request.as_turn(), tool_result_turn(call_id, "sunny")], OUT)[1]


class TestGeminiSignatures(unittest.TestCase):

    def test_the_signed_parts_go_back_as_they_came(self):
        part = {"functionCall": {"name": "weather", "args": {"city": "Oslo"}},
                "thoughtSignature": "c2lnbmF0dXJl"}
        body = _round_trip("gemini-3-flash-preview", {"candidates": [{"content": {"parts": [part]}}]})
        model_turn = next(c for c in body["contents"] if c["role"] == "model")
        self.assertEqual(model_turn["parts"], [part])

    def test_an_unsigned_call_is_rebuilt_as_before(self):
        part = {"functionCall": {"name": "weather", "args": {"city": "Oslo"}}}
        body = _round_trip("gemini-3.1-flash-lite", {"candidates": [{"content": {"parts": [part]}}]})
        model_turn = next(c for c in body["contents"] if c["role"] == "model")
        self.assertEqual(model_turn["parts"], [part])


class TestReasoningContent(unittest.TestCase):

    REPLY = {"choices": [{"message": {
        "content": None, "reasoning_content": "I should look it up.",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "weather", "arguments": '{"city": "Oslo"}'}}]}}]}

    def _assistant(self, name):
        body = _round_trip(name, self.REPLY)
        return next(m for m in body["messages"] if m["role"] == "assistant")

    def test_kept_where_the_provider_asks_for_it(self):
        self.assertEqual(self._assistant("mimo-v2.6-flash")["reasoning_content"], "I should look it up.")

    def test_not_sent_where_it_does_not(self):
        self.assertNotIn("reasoning_content", self._assistant("deepseek-chat"))


if __name__ == "__main__":
    unittest.main()
