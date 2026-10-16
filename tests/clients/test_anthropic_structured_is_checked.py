"""
Claude's structured answers are checked like everyone else's.

With `json_schema`, a reply with no tool call — cut off, or refused — came
back as `{}`, and a partial object went out unchecked, where every other
family raises `TruncatedResponseError` or `InvalidStructuredOutputError`.
"""

import unittest

from yait_aichain import InvalidStructuredOutputError, Model, TruncatedResponseError

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
OUT = {"modalities": ["text"], "format": {"type": "json_schema", "schema": SCHEMA}}


def _parse(response):
    return Model("claude-sonnet-4-5", api_key="k").from_response(response, OUT)


class TestStructured(unittest.TestCase):

    def test_cut_off_before_the_call_is_truncation(self):
        with self.assertRaises(TruncatedResponseError):
            _parse({"stop_reason": "max_tokens", "content": [], "usage": {"output_tokens": 64}})

    def test_no_call_otherwise_is_invalid(self):
        with self.assertRaises(InvalidStructuredOutputError):
            _parse({"stop_reason": "end_turn", "content": [{"type": "text", "text": "I won't."}]})

    def test_an_answer_off_the_schema_is_invalid(self):
        with self.assertRaises(InvalidStructuredOutputError) as caught:
            _parse({"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "id": "t", "name": "structured_output", "input": {"nom": "x"}}]})
        self.assertTrue(caught.exception.problems)

    def test_a_good_answer_comes_back(self):
        self.assertEqual(_parse({"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t", "name": "structured_output", "input": {"name": "Ann"}}]}), {"name": "Ann"})


if __name__ == "__main__":
    unittest.main()
