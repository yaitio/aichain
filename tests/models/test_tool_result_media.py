"""
A tool that renders something must be able to show it.

Before this, `tool_result_turn` serialised anything non-string to JSON, so an
agent could produce an image and then only read numbers about it — a vision
loop had to be lifted out of the agent into a separate Skill. Providers
disagree about whether a tool result may carry media, so the split is checked
per family here rather than assumed.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yait_aichain import Model
from yait_aichain.models._calls import tool_result_turn, split_media_result

IMG = {"type": "image",
       "source": {"kind": "base64", "mime": "image/png", "data": "AA"}}

CONVO = [
    {"role": "user", "parts": [{"type": "text", "text": "render it"}]},
    {"role": "assistant", "parts": [],
     "tool_calls": [{"id": "t1", "name": "render", "arguments": {}}]},
    {"role": "tool", "call_id": "t1",
     "parts": [{"type": "text", "text": "done"}, IMG]},
]
OUT = {"format": {"type": "text"}}


class TestUniversalTurn(unittest.TestCase):

    def test_text_is_unchanged(self):
        turn = tool_result_turn("c", "hello")
        self.assertEqual(turn["parts"], [{"type": "text", "text": "hello"}])

    def test_a_dict_is_still_serialised(self):
        """Data is data — only media parts are carried through."""
        turn = tool_result_turn("c", {"rows": 3})
        self.assertEqual(turn["parts"][0]["type"], "text")
        self.assertEqual(json.loads(turn["parts"][0]["text"]), {"rows": 3})

    def test_a_media_part_survives(self):
        self.assertEqual(tool_result_turn("c", IMG)["parts"], [IMG])

    def test_a_mixed_list_keeps_order(self):
        turn = tool_result_turn("c", ["slide 1", IMG])
        self.assertEqual([p["type"] for p in turn["parts"]], ["text", "image"])


class TestSplit(unittest.TestCase):

    def test_a_text_only_result_is_returned_unchanged(self):
        msg = {"role": "tool", "call_id": "t1",
               "parts": [{"type": "text", "text": "done"}]}
        turn, follow_up = split_media_result(msg)
        self.assertIs(turn, msg)
        self.assertIsNone(follow_up)

    def test_media_moves_out_and_is_announced(self):
        turn, follow_up = split_media_result(CONVO[2])
        text = turn["parts"][0]["text"]
        self.assertIn("done", text)
        self.assertIn("image", text)      # the model is told what follows
        self.assertEqual(follow_up["role"], "user")
        self.assertEqual([p["type"] for p in follow_up["parts"]],
                         ["text", "image"])

    def test_the_follow_up_is_never_a_bare_image(self):
        """An image with no text arrives with no stated relation to anything.

        The caption names the call it came from, so the model is told what it
        is looking at. Whether that also improves how reliably the image is
        attended to is unmeasured: on gpt-4o-mini it moved 2 of 6 to 3 of 6,
        which on six samples is noise."""
        _, follow_up = split_media_result(CONVO[2])
        self.assertEqual(follow_up["parts"][0]["type"], "text")
        self.assertIn("t1", follow_up["parts"][0]["text"])


class TestTheWholePath(unittest.TestCase):
    """The agent's own sequence, not just the function at the end of it.

    The first version of this feature passed every test above and shipped an
    agent that never carried an image: `result_message` json-dumped the result
    one layer higher, so a base64 blob went out as prose and the model
    answered about a picture it had never seen. Tests that called
    `tool_result_turn` directly could not see it — they tested the function
    and not the path.

    Measured on the real path afterwards, sighted vs blind, three trials each:
    gpt-4o-mini 3/6 → 6/6, gemini-2.5-flash 0/6 → 5/6, blind 0/6 throughout.
    """

    def _as_the_agent_does(self, result):
        from yait_aichain.agent import _prompts as prompts
        rendered = prompts.result_message(result, None)
        return tool_result_turn("t1", rendered)

    def test_an_image_survives_the_agents_rendering_step(self):
        turn = self._as_the_agent_does([IMG, "rendered 1 slide"])
        self.assertIn("image", [p["type"] for p in turn["parts"]])

    def test_it_reaches_the_wire_from_there(self):
        turn = self._as_the_agent_does([IMG, "rendered 1 slide"])
        convo = CONVO[:2] + [turn]
        for name in ("claude-opus-5", "gpt-4o", "gemini-3.1-pro-preview"):
            with self.subTest(model=name):
                _, body = Model(name, api_key="k").to_request(convo, OUT)
                wire = body.get("messages") or body.get("contents")
                self.assertIn("AA", json.dumps(wire))

    def test_data_is_still_rendered_as_text(self):
        turn = self._as_the_agent_does({"rows": 3})
        self.assertEqual(turn["parts"][0]["type"], "text")

    def test_the_journal_never_receives_a_part(self):
        """An image cannot go in a written record; it is named instead."""
        from yait_aichain.agent import _prompts as prompts
        text = prompts.observation_text([IMG, "rendered 1 slide"], None)
        self.assertIsInstance(text, str)
        self.assertIn("image", text)
        self.assertNotIn("AA", text)


class TestOnTheWire(unittest.TestCase):
    """One situation, four wire formats, and the image reaches all of them."""

    def _wire(self, model_name: str) -> str:
        _, body = Model(model_name, api_key="k").to_request(CONVO, OUT)
        wire = body.get("messages") or body.get("contents") or body.get("input")
        return json.dumps(wire, ensure_ascii=False)

    def test_anthropic_carries_it_inside_the_tool_result(self):
        """The one family that accepts media in the result itself."""
        _, body = Model("claude-opus-5", api_key="k").to_request(CONVO, OUT)
        result = body["messages"][-1]["content"][0]
        self.assertEqual(result["type"], "tool_result")
        self.assertEqual([b["type"] for b in result["content"]],
                         ["text", "image"])

    def test_google_sends_it_as_the_next_message(self):
        _, body = Model("gemini-3.1-pro-preview", api_key="k").to_request(
            CONVO, OUT)
        self.assertIn("functionResponse", body["contents"][-2]["parts"][0])
        last = body["contents"][-1]["parts"]
        self.assertIn("text", last[0])
        self.assertIn("inlineData", last[1])

    def test_openai_chat_sends_it_as_the_next_message(self):
        _, body = Model("gpt-4o", api_key="k").to_request(CONVO, OUT)
        self.assertEqual(body["messages"][-2]["role"], "tool")
        self.assertEqual(body["messages"][-1]["role"], "user")
        content = body["messages"][-1]["content"]
        self.assertEqual([c["type"] for c in content], ["text", "image_url"])

    def test_every_family_receives_the_image(self):
        for name in ("claude-opus-5", "gpt-4o", "gpt-5.5",
                     "gemini-3.1-pro-preview"):
            with self.subTest(model=name):
                wire = self._wire(name)
                self.assertTrue(
                    "AA" in wire,
                    f"{name} dropped the image the tool returned")


if __name__ == "__main__":
    unittest.main()
