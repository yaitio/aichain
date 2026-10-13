"""
An answer in several text blocks is the whole answer.

Claude's first block, Gemini's first part and the Responses API's last
output_text were returned alone — the "assigned instead of accumulated"
defect fixed earlier for instructions, in the answer itself.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.models import Model

OUT = {"format": {"type": "text"}}


class TestJoined(unittest.TestCase):

    def test_claude(self):
        r = {"content": [{"type": "text", "text": "One, "}, {"type": "text", "text": "two."}],
             "stop_reason": "end_turn"}
        self.assertEqual(Model("claude-sonnet-4-5", api_key="k").from_response(r, OUT), "One, two.")

    def test_gemini_without_thoughts(self):
        r = {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True},
                                                   {"text": "One, "}, {"text": "two."}]},
                             "finishReason": "STOP"}]}
        self.assertEqual(Model("gemini-2.5-flash", api_key="k").from_response(r, OUT), "One, two.")

    def test_responses_api(self):
        r = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "One, "}, {"type": "output_text", "text": "two."}]}]}
        self.assertEqual(Model("gpt-5.5", api_key="k").from_response(r, OUT), "One, two.")


if __name__ == "__main__":
    unittest.main()
