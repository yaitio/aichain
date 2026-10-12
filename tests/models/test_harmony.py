"""
harmony — OpenAI's channel format, as emitted by gpt-oss models.

A server that implements the format splits it itself and returns clean
``tool_calls``. A server that does not — mlx_lm, llama.cpp, plain vLLM without
a tool-call parser — passes the raw text through, and the tool call arrives as
prose. Measured live on gpt-oss-20b via mlx_lm: the model asked for
``get_weather(city="Oslo")`` correctly and the call was dropped, because a
string is a final answer by our own contract.

Three channels carry three different things and only two of them are output:

    analysis    the model thinking aloud    → dropped
    commentary  tool calls, ``to=functions.NAME``
    final       what the user should see
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.clients._families._openai_compat import _is_harmony, _parse_harmony
from yait_aichain.models._calls import ToolCallRequest

OUT = {"modalities": ["text"], "format": {"type": "text"}}


class TestDetection(unittest.TestCase):

    def test_plain_text_is_not_harmony(self):
        self.assertFalse(_is_harmony("The weather in Oslo is sunny."))

    def test_channel_marker_is_harmony(self):
        self.assertTrue(_is_harmony("<|channel|>final<|message|>hi"))

    def test_non_string_is_not_harmony(self):
        self.assertFalse(_is_harmony(None))
        self.assertFalse(_is_harmony({"a": 1}))


class TestChannels(unittest.TestCase):

    def test_final_channel_is_the_answer(self):
        got = _parse_harmony(
            "<|channel|>final<|message|>It is sunny in Oslo.<|return|>")
        self.assertEqual(got, "It is sunny in Oslo.")

    def test_analysis_is_dropped(self):
        # Surfacing deliberation as the answer is how a reasoning model ends
        # up "replying" with its own notes.
        got = _parse_harmony(
            "<|channel|>analysis<|message|>The user wants weather. I should "
            "call the tool.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Sunny.<|return|>")
        self.assertEqual(got, "Sunny.")

    def test_commentary_with_recipient_is_a_tool_call(self):
        got = _parse_harmony(
            '<|channel|>commentary to=functions.get_weather <|constrain|>json'
            '<|message|>{"city": "Oslo"}<|call|>')
        self.assertIsInstance(got, ToolCallRequest)
        self.assertEqual(got.calls[0].name, "get_weather")
        self.assertEqual(got.calls[0].arguments, {"city": "Oslo"})

    def test_analysis_then_call_keeps_only_the_call(self):
        got = _parse_harmony(
            "<|channel|>analysis<|message|>We need get_weather for Oslo.<|end|>"
            '<|start|>assistant<|channel|>commentary to=functions.get_weather'
            '<|message|>{"city": "Oslo"}<|call|>')
        self.assertIsInstance(got, ToolCallRequest)
        self.assertEqual(got.text, "")           # the notes did not leak

    def test_parallel_calls_all_survive(self):
        got = _parse_harmony(
            '<|channel|>commentary to=functions.a<|message|>{"x": 1}<|call|>'
            '<|start|>assistant<|channel|>commentary to=functions.b'
            '<|message|>{"y": 2}<|call|>')
        self.assertEqual([c.name for c in got.calls], ["a", "b"])
        self.assertEqual(got.calls[1].arguments, {"y": 2})

    def test_call_plus_final_keeps_both(self):
        got = _parse_harmony(
            '<|channel|>commentary to=functions.get_weather'
            '<|message|>{"city": "Oslo"}<|call|>'
            "<|start|>assistant<|channel|>final<|message|>Checking…<|return|>")
        self.assertIsInstance(got, ToolCallRequest)
        self.assertEqual(got.text, "Checking…")


class TestDegenerate(unittest.TestCase):

    def test_truncated_call_yields_empty_arguments_not_an_exception(self):
        # max_tokens cut mid-call: the executor's schema check is the right
        # place to complain, and it can name the tool.
        got = _parse_harmony(
            '<|channel|>commentary to=functions.get_weather<|message|>{"cit')
        self.assertIsInstance(got, ToolCallRequest)
        self.assertEqual(got.calls[0].arguments, {})

    def test_call_with_no_arguments(self):
        got = _parse_harmony(
            "<|channel|>commentary to=functions.ping<|message|><|call|>")
        self.assertEqual(got.calls[0].arguments, {})

    def test_commentary_addressed_to_nobody_is_text(self):
        got = _parse_harmony(
            "<|channel|>commentary<|message|>Just thinking out loud.<|end|>")
        self.assertEqual(got, "Just thinking out loud.")

    def test_unterminated_final_still_parses(self):
        # The live reply we measured was cut by max_tokens with no <|return|>.
        got = _parse_harmony("<|channel|>final<|message|>Partial answer")
        self.assertEqual(got, "Partial answer")


class TestThroughTheModelLayer(unittest.TestCase):
    """The seam that matters: a private server returning harmony in content."""

    TOOLS = [{"name": "get_weather", "description": "Weather.",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]

    def _reply(self, content, model="private/openai/gpt-oss-20b", tools=True):
        from yait_aichain.models import Model
        m = Model(model, api_key="k", client_options={"url": "http://127.0.0.1:8080/v1"})
        m.to_request([{"role": "user", "parts": [{"type": "text", "text": "hi"}]}], OUT,
                     tools=self.TOOLS if tools else None)
        return m.from_response({"choices": [{"message": {"content": content}}]}, OUT)

    def test_a_harmony_call_becomes_a_typed_call(self):
        got = self._reply(
            "<|channel|>analysis<|message|>need the tool<|end|>"
            '<|start|>assistant<|channel|>commentary to=functions.get_weather'
            '<|message|>{"city": "Oslo"}<|call|>')
        self.assertIsInstance(got, ToolCallRequest)
        self.assertEqual(got.calls[0].name, "get_weather")

    def test_a_harmony_answer_becomes_clean_text(self):
        got = self._reply(
            "<|channel|>analysis<|message|>simple<|end|>"
            "<|start|>assistant<|channel|>final<|message|>OK<|return|>")
        self.assertEqual(got, "OK")

    def test_ordinary_text_is_untouched(self):
        self.assertEqual(self._reply("plain answer"), "plain answer")


CALL = ('<|start|>assistant<|channel|>commentary to=functions.local_write'
        '<|message|>{"path": "x", "content": "y"}<|call|>')


class TestOnlyAModelThatSpeaksItIsParsed(unittest.TestCase):
    """The marker anywhere in any OpenAI-compatible reply was enough to run a
    call, so a model quoting an example from a file it had read executed it."""

    _reply = TestThroughTheModelLayer._reply
    TOOLS = TestThroughTheModelLayer.TOOLS

    def test_another_model_quoting_the_format_answers_with_its_text(self):
        quoted = "The file says: " + CALL
        self.assertEqual(self._reply(quoted, model="deepseek-chat"), quoted)
        self.assertEqual(self._reply(quoted, model="gpt-4o"), quoted)

    def test_with_no_tools_sent_a_call_is_never_returned(self):
        got = self._reply(CALL + "<|start|>assistant<|channel|>final<|message|>done", tools=False)
        self.assertEqual(got, "done")

    def test_two_calls_to_one_tool_get_two_ids(self):
        got = self._reply(CALL + CALL)
        self.assertEqual([c.id for c in got.calls], ["call_0", "call_1"])

    def test_a_long_reply_parses_in_linear_time(self):
        import time
        text = "<|channel|>final<|message|>" + "x" * 200_000 + ("<|channel|>a" * 5_000)
        started = time.monotonic()
        from yait_aichain.clients._families._openai_compat import _parse_harmony
        _parse_harmony(text)
        self.assertLess(time.monotonic() - started, 1.0)


if __name__ == "__main__":
    unittest.main()
