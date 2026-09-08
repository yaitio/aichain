"""
The Responses API names its content differently, and refuses the other name.

Vision never worked on any model routed through the Responses API: the chat
encoder was reused there, so `text`/`image_url` went out where `input_text`/
`input_image` were required. Live, before the fix:

    Invalid value: 'text'. Supported values are: 'input_text', 'input_image',
    'input_audio', 'output_text', 'refusal', 'input_file'

It stayed hidden because a message with one text part was collapsed to a bare
string — the commonest shape of all. Every single-part run passed and every
multi-part one failed, which is why the collapse is gone as well as the
encoder fixed: a defect that only shows on the rarer path will hide there
again.
"""

import base64
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yait_aichain import Model

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
IMG = {"type": "image",
       "source": {"kind": "base64", "mime": "image/png", "data": PNG}}

RESPONSES_MODEL = "gpt-5.5"      # routes through /v1/responses
CHAT_MODEL      = "gpt-4o"       # routes through /v1/chat/completions


def _body(model, messages):
    _, body = Model(model, api_key="k").to_request(
        messages, {"format": {"type": "text"}})
    return body


class TestResponsesNames(unittest.TestCase):

    def test_text_and_image_use_the_input_spelling(self):
        body = _body(RESPONSES_MODEL, [{"role": "user", "parts": [
            {"type": "text", "text": "colour?"}, IMG]}])
        types = [c["type"] for c in body["input"][0]["content"]]
        self.assertEqual(types, ["input_text", "input_image"])

    def test_the_image_url_is_a_string_not_an_object(self):
        """The object form is the chat spelling; here it must be bare."""
        body = _body(RESPONSES_MODEL, [{"role": "user", "parts": [IMG]}])
        url = body["input"][0]["content"][0]["image_url"]
        self.assertIsInstance(url, str)
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_an_assistant_turn_says_output_text(self):
        body = _body(RESPONSES_MODEL, [
            {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "hello"}]}])
        self.assertEqual(body["input"][-1]["content"][0]["type"], "output_text")

    def test_a_lone_text_part_is_still_a_list(self):
        """No collapsing to a string: that is what hid this for so long."""
        body = _body(RESPONSES_MODEL, [{"role": "user", "parts": [
            {"type": "text", "text": "hi"}]}])
        content = body["input"][0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "input_text")

    def test_the_system_prompt_still_becomes_instructions(self):
        body = _body(RESPONSES_MODEL, [
            {"role": "system", "parts": [{"type": "text", "text": "be brief"}]},
            {"role": "user", "parts": [{"type": "text", "text": "hi"}]}])
        self.assertEqual(body["instructions"], "be brief")

    def test_an_image_from_a_tool_reaches_it_too(self):
        """The 2.2.0 feature that made this visible."""
        convo = [
            {"role": "user", "parts": [{"type": "text", "text": "shoot"}]},
            {"role": "assistant", "parts": [],
             "tool_calls": [{"id": "t1", "name": "shot", "arguments": {}}]},
            {"role": "tool", "call_id": "t1", "parts": [IMG]}]
        wire = json.dumps(_body(RESPONSES_MODEL, convo)["input"])
        self.assertIn("input_image", wire)
        self.assertNotIn('"type": "image_url"', wire)


class TestChatIsUnchanged(unittest.TestCase):
    """The other wire format keeps its own names."""

    def test_chat_still_uses_text_and_image_url(self):
        body = _body(CHAT_MODEL, [{"role": "user", "parts": [
            {"type": "text", "text": "colour?"}, IMG]}])
        types = [c["type"] for c in body["messages"][0]["content"]]
        self.assertEqual(types, ["text", "image_url"])

    def test_chat_keeps_the_image_url_object(self):
        body = _body(CHAT_MODEL, [{"role": "user", "parts": [IMG]}])
        self.assertIsInstance(body["messages"][0]["content"][0]["image_url"],
                              dict)


if __name__ == "__main__":
    unittest.main()
