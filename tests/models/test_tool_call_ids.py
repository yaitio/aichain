"""
Every tool call has its own id, and every result finds its call.

harmony carried `""` and Google the function's name, so two calls to one
function shared an id; `run()` answered under `call.id or call.name` while a
driver used `call.id`; `dangling_calls` skipped empty ids and compared against
a set, so a repeated id answered once hid a lost result; and an id from Kimi
(`functions.x:0`) sent to Anthropic was a 400.
"""

import unittest
import warnings

from yait_aichain import Model
from yait_aichain.models import ToolCall, ToolCallRequest, dangling_calls, tool_result_turn

OUT = {"modalities": ["text"], "format": {"type": "text"}}


class TestIds(unittest.TestCase):

    def test_missing_and_repeated_ids_are_made_unique(self):
        req = ToolCallRequest(calls=(ToolCall("", "a"), ToolCall("x", "b"), ToolCall("x", "c")))
        ids = [c.id for c in req.calls]
        self.assertEqual(len(set(ids)), 3)
        self.assertTrue(all(ids))

    def test_gemini_two_calls_to_one_function(self):
        m = Model("gemini-2.5-flash", api_key="k")
        reply = m.from_response({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "look", "args": {"q": 1}}},
            {"functionCall": {"name": "look", "args": {"q": 2}}}]}}]}, OUT)
        self.assertEqual(len({c.id for c in reply.calls}), 2)
        history = [{"role": "user", "parts": [{"type": "text", "text": "go"}]}, reply.as_turn(),
                   tool_result_turn(reply.calls[0].id, "1"), tool_result_turn(reply.calls[1].id, "2")]
        _, body = m.to_request(history, OUT)
        names = [p["functionResponse"]["name"] for c in body["contents"] for p in c["parts"]
                 if "functionResponse" in p]
        self.assertEqual(names, ["look", "look"])


class TestDangling(unittest.TestCase):

    def test_one_id_asked_twice_and_answered_once_is_dangling(self):
        history = [{"role": "assistant", "tool_calls": [{"id": "a", "name": "x"}]},
                   tool_result_turn("a", "ok"),
                   {"role": "assistant", "tool_calls": [{"id": "a", "name": "x"}]}]
        self.assertEqual(dangling_calls(history), ["a"])

    def test_an_empty_id_counts(self):
        self.assertEqual(dangling_calls([{"role": "assistant", "tool_calls": [{"id": "", "name": "x"}]}]), [""])


class TestAnotherProvidersIds(unittest.TestCase):

    def test_anthropic_receives_ids_it_accepts_on_both_sides(self):
        history = [{"role": "user", "parts": [{"type": "text", "text": "go"}]},
                   {"role": "assistant", "parts": [],
                    "tool_calls": [{"id": "functions.look:0", "name": "look", "arguments": {}}]},
                   tool_result_turn("functions.look:0", "ok")]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, body = Model("claude-sonnet-4-5", api_key="k").to_request(history, OUT)
        use = next(b for m in body["messages"] for b in m["content"]
                   if isinstance(b, dict) and b.get("type") == "tool_use")
        res = next(b for m in body["messages"] for b in m["content"]
                   if isinstance(b, dict) and b.get("type") == "tool_result")
        self.assertRegex(use["id"], r"^[A-Za-z0-9_-]+$")
        self.assertEqual(use["id"], res["tool_use_id"])


if __name__ == "__main__":
    unittest.main()
