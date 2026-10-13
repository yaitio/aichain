"""
DeepSeek's reasoning swap is the chat/reasoner pair's, and is priced as such.

``reasoning="low"`` on deepseek-v4-pro sent deepseek-chat — another,
retiring generation — and the cost was read at v4-pro's rate; a schema was
dropped when the system prompt merely said "JSON"; a fenced answer failed.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.models import Model
from yait_aichain.models._adaptation import DECLINED, SWAPPED
from yait_aichain.skills import Skill

MSG = [{"role": "system", "parts": [{"type": "text", "text": "Answer in JSON."}]},
       {"role": "user", "parts": [{"type": "text", "text": "hi"}]}]


class TestDeepSeek(unittest.TestCase):

    def test_v4_pro_is_not_swapped_to_a_legacy_model(self):
        m = Model("deepseek-v4-pro", api_key="k", options={"reasoning": "low"})
        _, body, made = m.build(MSG, {"format": {"type": "text"}})
        self.assertEqual(body["model"], "deepseek-v4-pro")
        self.assertEqual([a.kind for a in made if a.option == "reasoning"], [DECLINED])

    def test_the_pair_still_swaps(self):
        m = Model("deepseek-chat", api_key="k", options={"reasoning": "high"})
        _, body, made = m.build(MSG, {"format": {"type": "text"}})
        self.assertEqual(body["model"], "deepseek-reasoner")
        self.assertIn(SWAPPED, [a.kind for a in made])

    def test_a_schema_goes_in_even_when_the_prompt_says_json(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
        m = Model("deepseek-chat", api_key="k")
        _, body, _ = m.build(MSG, {"format": {"type": "json_schema", "schema": schema}})
        self.assertIn('"integer"', body["messages"][0]["content"])

    def test_a_swapped_call_is_priced_as_the_model_that_answered(self):
        m = Model("deepseek-chat", api_key="k", options={"reasoning": "high"})
        m.client.send = MagicMock(return_value=json.dumps({
            "model": "deepseek-reasoner",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0,
                      "total_tokens": 1_000_000}}).encode())
        m.client._auth_headers = MagicMock(return_value={})
        skill = Skill(m, prompt="hi")
        skill.run()
        plain = Model("deepseek-chat", api_key="k")
        plain.client.send = m.client.send
        plain.client._auth_headers = m.client._auth_headers
        cheap = Skill(plain, prompt="hi")
        cheap.run()
        self.assertNotEqual(skill.last_usage.cost, cheap.last_usage.cost)

    def test_a_fenced_json_answer_parses(self):
        m = Model("deepseek-chat", api_key="k")
        out = m.from_response({"choices": [{"message": {"content": "```json\n{\"n\": 1}\n```"},
                                            "finish_reason": "stop"}]},
                              {"format": {"type": "json"}})
        self.assertEqual(out, {"n": 1})


if __name__ == "__main__":
    unittest.main()
