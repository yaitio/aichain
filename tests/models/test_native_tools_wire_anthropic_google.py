"""
Native tool wire — Anthropic and Google shapes.

Anthropic: flat tools with ``input_schema``; calls as ``tool_use`` content
blocks; results as ``tool_result`` blocks inside USER messages (strict role
alternation — consecutive results merge into one user message). Structured
output is a forced tool call on this provider, so tools + json_schema is a
stated conflict, not a merge.

Google: ``functionDeclarations``; calls as ``functionCall`` parts on a model
turn; results as ``functionResponse`` parts keyed by NAME — this wire has no
call ids, so ``ToolCall.id`` degrades to the function name and the tool turn's
``call_id`` must carry the name.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.models import Model
from yait_aichain.models._calls import ToolCall, ToolCallRequest

SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web.",
        "parameters": {"type": "object",
                       "properties": {"q": {"type": "string"}},
                       "required": ["q"]},
    },
}

CONVO = [
    {"role": "user", "parts": [{"type": "text", "text": "find it"}]},
    {"role": "assistant",
     "parts": [{"type": "text", "text": "checking"}],
     "tool_calls": [{"id": "t1", "name": "search", "arguments": {"q": "it"}}]},
    {"role": "tool", "call_id": "t1",
     "parts": [{"type": "text", "text": "found: it"}]},
    {"role": "tool", "call_id": "t2",
     "parts": [{"type": "text", "text": "also: that"}]},
]

OUT = {"modalities": ["text"], "format": {"type": "text"}}


class TestAnthropicWire(unittest.TestCase):

    def _body(self, tools=None, messages=CONVO, out=OUT):
        m = Model("claude-opus-5", api_key="k")
        path, body = m.to_request(messages, out, tools=tools)
        self.assertEqual(path, "/v1/messages")
        return body

    def test_tools_flatten_to_input_schema(self):
        body = self._body(tools=[SEARCH])
        self.assertEqual(body["tools"], [{
            "name": "search", "description": "Search the web.",
            "input_schema": SEARCH["function"]["parameters"]}])

    def test_call_turn_carries_text_and_tool_use_blocks(self):
        msg = self._body(tools=[SEARCH])["messages"][1]
        self.assertEqual(msg["role"], "assistant")
        kinds = [b["type"] for b in msg["content"]]
        self.assertEqual(kinds, ["text", "tool_use"])
        use = msg["content"][1]
        self.assertEqual((use["id"], use["name"], use["input"]),
                         ("t1", "search", {"q": "it"}))

    def test_consecutive_results_merge_into_one_user_message(self):
        # Strict role alternation: two results in a row must not produce two
        # user messages back to back.
        msgs = self._body(tools=[SEARCH])["messages"]
        self.assertEqual(len(msgs), 3)
        results = msgs[2]["content"]
        self.assertEqual([b["type"] for b in results],
                         ["tool_result", "tool_result"])
        self.assertEqual([b["tool_use_id"] for b in results], ["t1", "t2"])

    def test_tools_plus_json_schema_is_a_stated_conflict(self):
        with self.assertRaises(ValueError) as ctx:
            self._body(tools=[SEARCH],
                       out={"modalities": ["text"],
                            "format": {"type": "json_schema", "name": "r",
                                       "schema": {"type": "object"}}})
        self.assertIn("forced tool call", str(ctx.exception))

    def test_response_tool_use_parses_to_typed_calls(self):
        m = Model("claude-opus-5", api_key="k")
        reply = m.from_response({
            "content": [
                {"type": "text", "text": "let me look"},
                {"type": "tool_use", "id": "a1", "name": "search",
                 "input": {"q": "cats"}},
            ],
            "stop_reason": "tool_use",
        }, OUT)
        self.assertIsInstance(reply, ToolCallRequest)
        self.assertEqual(reply.calls[0],
                         ToolCall(id="a1", name="search", arguments={"q": "cats"}))
        self.assertEqual(reply.text, "let me look")

    def test_plain_text_response_is_still_a_string(self):
        m = Model("claude-opus-5", api_key="k")
        got = m.from_response({"content": [{"type": "text", "text": "hi"}]}, OUT)
        self.assertEqual(got, "hi")


class TestGoogleWire(unittest.TestCase):

    def _body(self, tools=None):
        m = Model("gemini-3.1-pro-preview", api_key="k")
        path, body = m.to_request(CONVO, OUT, tools=tools)
        self.assertIn(":generateContent", path)
        return body

    def test_tools_become_function_declarations(self):
        body = self._body(tools=[SEARCH])
        decls = body["tools"][0]["functionDeclarations"]
        self.assertEqual(decls[0]["name"], "search")
        self.assertEqual(decls[0]["parameters"]["properties"]["q"]["type"],
                         "string")

    def test_call_turn_becomes_function_call_parts_on_a_model_turn(self):
        contents = self._body(tools=[SEARCH])["contents"]
        model_turn = contents[1]
        self.assertEqual(model_turn["role"], "model")
        fc = [p for p in model_turn["parts"] if "functionCall" in p][0]
        self.assertEqual(fc["functionCall"],
                         {"name": "search", "args": {"q": "it"}})

    def test_result_turn_becomes_function_response_keyed_by_name(self):
        contents = self._body(tools=[SEARCH])["contents"]
        frs = [p for c in contents for p in c["parts"] if "functionResponse" in p]
        self.assertEqual(len(frs), 2)
        self.assertEqual(frs[0]["functionResponse"]["name"], "t1")
        self.assertEqual(frs[0]["functionResponse"]["response"],
                         {"result": "found: it"})

    def test_response_function_call_parses_with_name_as_id(self):
        m = Model("gemini-3.1-pro-preview", api_key="k")
        reply = m.from_response({
            "candidates": [{"content": {"parts": [
                {"functionCall": {"name": "search", "args": {"q": "dogs"}}},
            ]}}],
        }, OUT)
        self.assertIsInstance(reply, ToolCallRequest)
        self.assertEqual(reply.calls[0],
                         ToolCall(id="search", name="search",
                                  arguments={"q": "dogs"}))

    def test_plain_text_response_is_still_a_string(self):
        m = Model("gemini-3.1-pro-preview", api_key="k")
        got = m.from_response({
            "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
        }, OUT)
        self.assertEqual(got, "hi")


if __name__ == "__main__":
    unittest.main()
