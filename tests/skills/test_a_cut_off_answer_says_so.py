"""
Prose cut at max_tokens is returned, and said to be cut.

An empty answer cut at the ceiling already raised. One with text did not:
`Skill.run` handed back half a sentence as the answer, and an agent ended on
it with `success=True, stopped_by="answered"` — nothing anywhere said the
model had been stopped mid-word. Found reconciling the τ²-bench defects,
2026-10-14.

The text is kept, because half an answer can be worth reading; what changes
is that it is never mistaken for a whole one. Pinned here, for every family's
own word for the ceiling, buffered and streamed:

  * `Skill.last_truncated` is True, and a warning says so once per model;
  * an agent stops with `stopped_by="truncated"`, `success=False`, and the
    partial text as its output;
  * a whole answer changes nothing.
"""

import json
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.agent import Agent
from yait_aichain.clients._families._structured import hit_ceiling
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.skills import _skill

HALF = "The answer is the first half of"

#: One reply cut at the ceiling, in each family's shape.
CUT = {
    "gpt-4o": {"choices": [{"message": {"content": HALF}, "finish_reason": "length"}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 16, "total_tokens": 21}},
    "claude-opus-5-5": {"content": [{"type": "text", "text": HALF}], "stop_reason": "max_tokens",
                        "usage": {"input_tokens": 5, "output_tokens": 16}},
    "gemini-3.8-flash": {"candidates": [{"content": {"parts": [{"text": HALF}]}, "finishReason": "MAX_TOKENS"}],
                         "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 16}},
    "gpt-5.5": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": HALF}]}],
                "usage": {"input_tokens": 5, "output_tokens": 16, "total_tokens": 21}},
}

WHOLE = {"choices": [{"message": {"content": "All of it."}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}


def _model(name, reply, stream=None):
    m = Model(name, api_key="k")
    m.client._post = MagicMock(return_value=json.dumps(reply).encode())
    if stream is not None:
        m.client._post_sse = MagicMock(side_effect=lambda *a, **k: iter(stream))
    m.client._auth_headers = MagicMock(return_value={})
    return m


def _quiet(fn):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in seen if "max_tokens" in str(w.message)]


class TestTheCeilingIsRead(unittest.TestCase):

    def test_every_family_word_for_it(self):
        for name, reply in CUT.items():
            with self.subTest(model=name):
                self.assertTrue(hit_ceiling(reply))
        self.assertFalse(hit_ceiling(WHOLE))

    def test_stream_events(self):
        self.assertTrue(hit_ceiling({"choices": [{"delta": {}, "finish_reason": "length"}]}))
        self.assertTrue(hit_ceiling({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}))
        self.assertTrue(hit_ceiling({"type": "response.incomplete",
                                     "response": {"status": "incomplete",
                                                  "incomplete_details": {"reason": "max_output_tokens"}}}))
        self.assertFalse(hit_ceiling({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}))


class TestASkillSaysSo(unittest.TestCase):

    def setUp(self):
        _skill._CUT_SAID.clear()

    def test_each_family_marks_the_answer_cut(self):
        for name, reply in CUT.items():
            with self.subTest(model=name):
                skill = Skill(_model(name, reply), prompt="hi")
                text, said = _quiet(skill.run)
                self.assertEqual(text, HALF)          # kept, not thrown away
                self.assertTrue(skill.last_truncated)
                self.assertEqual(len(said), 1)

    def test_once_per_model(self):
        skill = Skill(_model("gpt-4o", CUT["gpt-4o"]), prompt="hi")
        _quiet(skill.run)
        self.assertEqual(_quiet(skill.run)[1], [])
        self.assertTrue(skill.last_truncated)

    def test_a_whole_answer_is_not_marked(self):
        skill = Skill(_model("gpt-4o", WHOLE), prompt="hi")
        text, said = _quiet(skill.run)
        self.assertEqual((text, skill.last_truncated, said), ("All of it.", False, []))

    def test_a_streamed_answer_is_marked_too(self):
        events = [{"choices": [{"delta": {"content": HALF}}]},
                  {"choices": [{"delta": {}, "finish_reason": "length"}]},
                  {"choices": [], "usage": CUT["gpt-4o"]["usage"]}]
        skill = Skill(_model("gpt-4o", WHOLE, stream=events), prompt="hi")
        pieces, _ = _quiet(lambda: list(skill.stream()))
        self.assertEqual("".join(pieces), HALF)
        self.assertTrue(skill.last_truncated)

    def test_the_flag_resets_on_the_next_call(self):
        model = _model("gpt-4o", CUT["gpt-4o"])
        skill = Skill(model, prompt="hi")
        _quiet(skill.run)
        model.client._post = MagicMock(return_value=json.dumps(WHOLE).encode())
        _quiet(skill.run)
        self.assertFalse(skill.last_truncated)


class TestAnAgentDoesNotCallItAnAnswer(unittest.TestCase):

    def test_run_stops_as_truncated_with_the_text_kept(self):
        result, _ = _quiet(lambda: Agent(model=_model("gpt-4o", CUT["gpt-4o"])).run("hi"))
        self.assertFalse(result.success)
        self.assertEqual(result.stopped_by, "truncated")
        self.assertEqual(result.output, HALF)
        self.assertIn("max_tokens", result.error)

    def test_a_whole_answer_still_answers(self):
        result, _ = _quiet(lambda: Agent(model=_model("gpt-4o", WHOLE)).run("hi"))
        self.assertEqual((result.success, result.stopped_by), (True, "answered"))

    def test_step_leaves_the_fact_in_the_state(self):
        agent = Agent(model=_model("gpt-4o", CUT["gpt-4o"]))
        state = agent.new_state() if hasattr(agent, "new_state") else None
        if state is None:
            self.skipTest("no public state constructor")
        messages = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
        reply, _ = _quiet(lambda: agent.step(messages, state))
        self.assertEqual(reply, HALF)
        self.assertTrue(state["truncated"])


if __name__ == "__main__":
    unittest.main()
