"""
A structured reply that fails does so for one of two reasons, and the fixes
are opposite: raise `max_tokens`, or change the schema. Both arrive as
HTTP 200, and before these classes existed both surfaced as one bare
JSONDecodeError — three identical failures at the same column are a ceiling,
not a fluctuation, but only if something says so.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yait_aichain import TruncatedResponseError, InvalidStructuredOutputError
from yait_aichain.clients._families.google import GoogleClient

SCHEMA = {"type": "object",
          "properties": {"slides": {"type": "array",
                                    "items": {"type": "string"}}},
          "required": ["slides"],
          "additionalProperties": False}
OUT = {"format": {"type": "json_schema", "schema": SCHEMA}}


def _google(text, finish="STOP", tokens=None):
    resp = {"candidates": [{"finishReason": finish,
                            "content": {"parts": [{"text": text}]}}]}
    if tokens is not None:
        resp["usageMetadata"] = {"candidatesTokenCount": tokens}
    return GoogleClient.__new__(GoogleClient).parse_response(resp, OUT)


class TestTruncation(unittest.TestCase):

    def test_a_ceiling_is_named_and_so_is_the_number(self):
        with self.assertRaises(TruncatedResponseError) as ctx:
            _google('{"slides": ["a", "b', finish="MAX_TOKENS", tokens=8192)
        self.assertEqual(ctx.exception.output_tokens, 8192)
        self.assertEqual(ctx.exception.finish_reason, "MAX_TOKENS")
        # The number is the ceiling that was hit, so it is what max_tokens
        # has to be raised above — saying only "increase max_tokens" leaves
        # the reader to find it.
        self.assertIn("8192", ctx.exception.message)

    def test_still_catchable_as_valueerror(self):
        """2.1.0 raised ValueError here; existing handlers must keep working."""
        with self.assertRaises(ValueError):
            _google('{"slides": ["a', finish="MAX_TOKENS", tokens=100)


class TestSchemaViolation(unittest.TestCase):

    def test_a_field_nobody_declared_is_caught(self):
        """Google strips `additionalProperties`, inverting the instruction.

        The model then answers with fields the schema forbade, and the only
        way this was ever noticed was diffing two providers by hand."""
        with self.assertRaises(InvalidStructuredOutputError) as ctx:
            _google('{"slides": ["a"], "notes": "added by itself"}')
        self.assertTrue(any("notes" in p for p in ctx.exception.problems))

    def test_a_missing_required_field_is_caught(self):
        with self.assertRaises(InvalidStructuredOutputError):
            _google('{"other": 1}')

    def test_garbage_is_not_reported_as_truncation(self):
        with self.assertRaises(InvalidStructuredOutputError) as ctx:
            _google("sorry, I cannot")
        self.assertNotIsInstance(ctx.exception, TruncatedResponseError)

    def test_a_valid_answer_passes(self):
        self.assertEqual(_google('{"slides": ["a"]}'), {"slides": ["a"]})


if __name__ == "__main__":
    unittest.main()
