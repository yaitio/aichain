"""
Streamed tool calls come out in the order they began, each its own.

Slots were sorted as strings, so a turn of twelve calls ran slot 10 before
slot 2; Google renumbers its parts in every event, so two whole calls from two
events both landed in slot "0:0" — the first lost, the second run with `{}`.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.models import Model
from yait_aichain.models._base import _ToolCallAssembly


class TestOrder(unittest.TestCase):

    def test_twelve_calls_keep_their_order(self):
        a = _ToolCallAssembly()
        for i in range(12):
            a.add([{"slot": i, "id": f"c{i}", "name": f"t{i}", "arguments": "{}"}])
        self.assertEqual([c.name for c in a.request().calls], [f"t{i}" for i in range(12)])

    def test_two_google_calls_in_two_events_are_two_calls(self):
        client = Model("gemini-3.1-flash-lite", api_key="k").client
        a = _ToolCallAssembly()
        for name, args in (("first", {"q": 1}), ("second", {"q": 2})):
            event = {"candidates": [{"index": 0, "content": {"parts": [
                {"functionCall": {"name": name, "args": args}}]}}]}
            a.add(client.stream_tool_fragments(event))
        calls = a.request().calls
        self.assertEqual([(c.name, c.arguments) for c in calls],
                         [("first", {"q": 1}), ("second", {"q": 2})])


if __name__ == "__main__":
    unittest.main()
