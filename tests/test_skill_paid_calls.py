"""
A call that was paid for is recorded, whether or not its answer parsed.

A response that failed to parse after it was billed lost its usage with the
exception and emitted no ``llm_call.ended``; a streamed Anthropic
``json_schema`` answer reached ``last_result`` as the library's own synthetic
tool call instead of the value.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain.clients._errors import InvalidStructuredOutputError
from yait_aichain.models import Model
from yait_aichain.models._calls import ToolCall, ToolCallRequest
from yait_aichain.skills import Skill


class TestPaidAndUnparsed(unittest.TestCase):

    def test_usage_and_ended_survive_a_parse_failure(self):
        model = Model("gpt-4o", api_key="k")
        model.client.send = MagicMock(return_value=json.dumps({
            "choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}).encode())
        model.client._auth_headers = MagicMock(return_value={})
        ended = []
        skill = Skill(model, prompt="hi", output={"format": {"type": "json"}},
                      hooks=[lambda ev: ended.append(ev) if ev.type == "llm_call.ended" else None])
        with self.assertRaises(InvalidStructuredOutputError):
            skill.run()
        self.assertEqual(skill.last_usage.total_tokens, 15)
        self.assertEqual(len(ended), 1)
        self.assertIsNotNone(ended[0].error)


class TestStructuredCallUnwrapped(unittest.TestCase):

    def test_the_synthetic_call_is_its_arguments(self):
        fmt = {"type": "json_schema", "name": "answer",
               "schema": {"type": "object", "properties": {"n": {"type": "integer"}},
                          "required": ["n"]}}
        req = ToolCallRequest(calls=(ToolCall(id="1", name="answer", arguments={"n": 1}),))
        self.assertEqual(Skill._structured_call(req, fmt), {"n": 1})
        bad = ToolCallRequest(calls=(ToolCall(id="1", name="answer", arguments={"n": "x"}),))
        with self.assertRaises(InvalidStructuredOutputError):
            Skill._structured_call(bad, fmt)


if __name__ == "__main__":
    unittest.main()
