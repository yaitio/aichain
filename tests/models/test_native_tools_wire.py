"""
Native tool calling — the wire format, byte for byte, no network.

Two shapes live in one family and they are not interchangeable:

  chat completions   tools nested under "function"; calls in message.tool_calls;
                     results as role:"tool" messages keyed by tool_call_id
  Responses API      flat tools; calls as output items of type "function_call";
                     results as input items of type "function_call_output"

gpt-5.x routes through the second, everything else through the first. A tool
conversation serialized in the wrong shape is a provider 400 at best and a
model that silently never calls anything at worst — which is why these are
golden tests and not smoke tests.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.models import Model
from yait_aichain.models._calls import ToolCall, ToolCallRequest, tool_result_turn

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
    {"role": "system", "parts": [{"type": "text", "text": "be brief"}]},
    {"role": "user", "parts": [{"type": "text", "text": "find it"}]},
    {"role": "assistant",
     "tool_calls": [{"id": "c1", "name": "search", "arguments": {"q": "it"}}]},
    {"role": "tool", "call_id": "c1",
     "parts": [{"type": "text", "text": "found: it"}]},
]

OUT = {"modalities": ["text"], "format": {"type": "text"}}


class TestChatCompletionsWire(unittest.TestCase):
    """gpt-4o path — /v1/chat/completions."""

    def _body(self, tools=None):
        m = Model("gpt-4o", api_key="k")
        path, body = m.to_request(CONVO, OUT, tools=tools)
        self.assertIn("chat/completions", path)
        return body

    def test_tools_ride_in_the_body_verbatim(self):
        body = self._body(tools=[SEARCH])
        self.assertEqual(body["tools"], [SEARCH])

    def test_no_tools_no_field(self):
        self.assertNotIn("tools", self._body())

    def test_assistant_call_turn_serializes_with_json_string_arguments(self):
        msg = [m for m in self._body(tools=[SEARCH])["messages"]
               if m.get("tool_calls")][0]
        call = msg["tool_calls"][0]
        self.assertEqual(call["id"], "c1")
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "search")
        # arguments MUST be a JSON string on this wire, not a dict
        self.assertIsInstance(call["function"]["arguments"], str)
        self.assertEqual(json.loads(call["function"]["arguments"]), {"q": "it"})

    def test_tool_result_turn_serializes_with_tool_call_id(self):
        msg = [m for m in self._body(tools=[SEARCH])["messages"]
               if m.get("role") == "tool"][0]
        self.assertEqual(msg["tool_call_id"], "c1")
        self.assertEqual(msg["content"], "found: it")

    def test_response_with_tool_calls_parses_to_typed_calls(self):
        m = Model("gpt-4o", api_key="k")
        reply = m.from_response({
            "choices": [{"message": {
                "content": None,
                "tool_calls": [{"id": "x9", "type": "function",
                                "function": {"name": "search",
                                             "arguments": '{"q": "cats"}'}}],
            }}],
        }, OUT)
        self.assertIsInstance(reply, ToolCallRequest)
        self.assertEqual(reply.calls[0],
                         ToolCall(id="x9", name="search", arguments={"q": "cats"}))

    def test_calls_outrank_accompanying_text(self):
        # A reply that both says something and calls a tool is a call; the
        # text rides along. Collapsing to text would silently drop the action.
        m = Model("gpt-4o", api_key="k")
        reply = m.from_response({
            "choices": [{"message": {
                "content": "let me check",
                "tool_calls": [{"id": "a", "type": "function",
                                "function": {"name": "search",
                                             "arguments": "{}"}}],
            }}],
        }, OUT)
        self.assertIsInstance(reply, ToolCallRequest)
        self.assertEqual(reply.text, "let me check")

    def test_plain_text_response_is_still_a_string(self):
        m = Model("gpt-4o", api_key="k")
        self.assertEqual(
            m.from_response({"choices": [{"message": {"content": "hi"}}]}, OUT),
            "hi")

    def test_unparseable_arguments_become_an_empty_dict_not_an_exception(self):
        # The executor's schema check is the right place to complain — it can
        # name the tool. A parse crash here would kill the whole turn.
        m = Model("gpt-4o", api_key="k")
        reply = m.from_response({
            "choices": [{"message": {
                "tool_calls": [{"id": "b", "type": "function",
                                "function": {"name": "search",
                                             "arguments": "{broken"}}],
            }}],
        }, OUT)
        self.assertEqual(reply.calls[0].arguments, {})


class TestResponsesApiWire(unittest.TestCase):
    """gpt-5.x path — /v1/responses. Different shape on both sides."""

    def _body(self, tools=None):
        m = Model("gpt-5.6-luna", api_key="k")
        path, body = m.to_request(CONVO, OUT, tools=tools)
        self.assertEqual(path, "/v1/responses")
        return body

    def test_tools_are_flattened(self):
        body = self._body(tools=[SEARCH])
        self.assertEqual(body["tools"], [{
            "type": "function", "name": "search",
            "description": "Search the web.",
            "parameters": SEARCH["function"]["parameters"],
        }])

    def test_call_turn_becomes_a_function_call_input_item(self):
        items = [i for i in self._body(tools=[SEARCH])["input"]
                 if i.get("type") == "function_call"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["call_id"], "c1")
        self.assertEqual(items[0]["name"], "search")
        self.assertEqual(json.loads(items[0]["arguments"]), {"q": "it"})

    def test_result_turn_becomes_a_function_call_output_item(self):
        items = [i for i in self._body(tools=[SEARCH])["input"]
                 if i.get("type") == "function_call_output"]
        self.assertEqual(items, [{"type": "function_call_output",
                                  "call_id": "c1", "output": "found: it"}])

    def test_response_function_call_items_parse_to_typed_calls(self):
        m = Model("gpt-5.6-luna", api_key="k")
        reply = m.from_response({
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "function_call", "call_id": "r1",
                 "name": "search", "arguments": '{"q": "dogs"}'},
            ],
        }, OUT)
        self.assertIsInstance(reply, ToolCallRequest)
        self.assertEqual(reply.calls[0],
                         ToolCall(id="r1", name="search", arguments={"q": "dogs"}))

    def test_parallel_calls_all_survive(self):
        # Providers request several calls in one turn; executing only the
        # first would silently narrow the channel.
        m = Model("gpt-5.6-luna", api_key="k")
        reply = m.from_response({
            "output": [
                {"type": "function_call", "call_id": "p1", "name": "a",
                 "arguments": "{}"},
                {"type": "function_call", "call_id": "p2", "name": "b",
                 "arguments": "{}"},
            ],
        }, OUT)
        self.assertEqual([c.name for c in reply.calls], ["a", "b"])

    def test_plain_text_response_is_still_a_string(self):
        m = Model("gpt-5.6-luna", api_key="k")
        got = m.from_response({
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi"}]}],
        }, OUT)
        self.assertEqual(got, "hi")


class TestGates(unittest.TestCase):

    def test_a_family_without_tools_refuses_loudly(self):
        # Silently dropping declared tools produces a model that never calls
        # anything, with no error anywhere — the worst of today's failures.
        # Every text family now implements the wire, so the unsupporting
        # client is simulated: the gate must hold for any future family too.
        m = Model("gpt-4o", api_key="k")

        class _NoToolsClient:
            def build_request(self, *a, **k):        # pragma: no cover
                raise AssertionError("gate must fire before the build")

        m.client = _NoToolsClient()
        with self.assertRaises(ValueError) as ctx:
            m.to_request(CONVO, OUT, tools=[SEARCH])
        self.assertIn("does not support native tool calling", str(ctx.exception))

    def test_an_image_model_refuses_tools(self):
        m = Model("gpt-image-2", api_key="k")
        with self.assertRaises(ValueError):
            m.to_request([{"role": "user",
                           "parts": [{"type": "text", "text": "a cat"}]}],
                         {"modalities": ["image"], "format": {"type": "image"}},
                         tools=[SEARCH])


class TestTurnHelpers(unittest.TestCase):

    def test_as_turn_round_trips_through_the_chat_wire(self):
        req  = ToolCallRequest(calls=(ToolCall("c7", "search", {"q": "x"}),))
        msgs = [{"role": "user", "parts": [{"type": "text", "text": "go"}]},
                req.as_turn(), tool_result_turn("c7", {"hits": 3})]
        m = Model("gpt-4o", api_key="k")
        _, body = m.to_request(msgs, OUT, tools=[SEARCH])
        roles = [x.get("role") for x in body["messages"]]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        self.assertEqual(json.loads(body["messages"][2]["content"]), {"hits": 3})


class TestGrammar(unittest.TestCase):
    """The conversation schema itself — enforced at Skill construction."""

    def _validate(self, messages):
        from yait_aichain.skills._adapters import validate_input
        validate_input({"messages": messages})

    def test_the_tool_conversation_is_valid(self):
        self._validate(CONVO)

    def test_a_tool_turn_needs_a_call_id(self):
        with self.assertRaises(ValueError):
            self._validate([
                {"role": "user", "parts": ["go"]},
                {"role": "assistant",
                 "tool_calls": [{"id": "c", "name": "t", "arguments": {}}]},
                {"role": "tool", "parts": [{"type": "text", "text": "r"}]},
            ])

    def test_a_stray_tool_turn_is_rejected(self):
        # A result that answers no call is indistinguishable from conversation.
        with self.assertRaises(ValueError):
            self._validate([
                {"role": "user", "parts": [{"type": "text", "text": "go"}]},
                {"role": "tool", "call_id": "c",
                 "parts": [{"type": "text", "text": "r"}]},
            ])

    def test_assistant_after_tool_results_is_legal(self):
        self._validate(CONVO + [
            {"role": "assistant", "parts": [{"type": "text", "text": "done"}]},
        ])

    def test_an_assistant_call_turn_is_not_a_generate_marker(self):
        from yait_aichain.skills._adapters import is_generate_marker
        self.assertFalse(is_generate_marker(
            {"role": "assistant",
             "tool_calls": [{"id": "c", "name": "t", "arguments": {}}]}))

    def test_tool_calls_on_a_user_turn_are_rejected(self):
        with self.assertRaises(ValueError):
            self._validate([
                {"role": "user", "parts": [{"type": "text", "text": "go"}],
                 "tool_calls": [{"id": "c", "name": "t", "arguments": {}}]},
            ])


if __name__ == "__main__":
    unittest.main()
