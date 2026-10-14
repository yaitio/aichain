"""
How data moves through a Chain today, pinned before it is changed on purpose.

PLAN-agent-chain.md, phase 6, changes three of these: a dict output gets its
own key instead of spreading flat, a step with no key is named after its
runner instead of writing "result", and a wrong `input_map` name fails when
the chain is built instead of when it runs. Until then this is the behaviour
callers depend on, and the phases before 6 must not move it by accident. When
phase 6 lands, each test marked *changes in phase 6* is rewritten to the new
behaviour in the same commit — never deleted.
"""

import json
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Chain, Model, Skill
from yait_aichain.tools._base import Tool


def fetch(url: str):
    return {"title": "T", "body": "B " + url}


def count(text: str) -> int:
    return len(text.split())


class Upper(Tool):
    name = "upper"
    description = "Upper-case a string."
    parameters = {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}
    risk = "read"

    def run(self, s, options=None):
        return s.upper()


def _skill():
    m = Model("gpt-4o", api_key="k")
    m.client._post = MagicMock(return_value=json.dumps(
        {"choices": [{"message": {"content": "said"}}], "usage": {}}).encode())
    m.client._auth_headers = MagicMock(return_value={})
    return Skill(m, prompt="Say {topic}"), m


class TestDataFlowToday(unittest.TestCase):

    def test_a_dict_output_spreads_flat_and_its_key_is_not_stored(self):
        """Changes in phase 6: the dict will sit under "page"."""
        result = Chain([(fetch, "page"), (count, "words", {"text": "body"})]).run(
            variables={"url": "x"})
        self.assertEqual(result["body"], "B x")
        self.assertEqual(result["words"], 2)
        with self.assertRaises(KeyError):
            result["page"]

    def test_a_step_with_no_key_writes_result_and_the_next_overwrites_it(self):
        """Changes in phase 6: each output will be named after its runner."""
        def first(url: str):
            return "first"

        def second(url: str):
            return "second"

        result = Chain([first, second]).run(variables={"url": "x"})
        self.assertEqual(result["result"], "second")

    def test_a_wrong_input_map_name_is_caught_at_run_not_at_build(self):
        """Changes in phase 6: it will fail when the chain is built."""
        chain = Chain([(count, "words", {"text": "bdy"})])
        with self.assertRaises(ValueError) as caught:
            chain.run(variables={"url": "x"})
        self.assertIn("'bdy'", str(caught.exception))
        self.assertIn("['url']", str(caught.exception))


class TestInputMapByRunnerKind(unittest.TestCase):
    """Stays as it is: an input map renames a variable for each kind of runner."""

    def test_a_function(self):
        result = Chain([(count, "words", {"text": "url"})]).run(variables={"url": "a b c"})
        self.assertEqual(result["words"], 3)

    def test_a_tool(self):
        result = Chain([(Upper(), "loud", {"s": "url"})]).run(variables={"url": "x"})
        self.assertEqual(result["loud"], "X")

    def test_a_skill_fills_its_template_from_the_mapped_name(self):
        skill, model = _skill()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = Chain([(skill, "said", {"topic": "url"})]).run(variables={"url": "x"})
        self.assertEqual(result["said"], "said")
        body = model.client._post.call_args[0][1]
        self.assertIn("Say x", json.dumps(body))


if __name__ == "__main__":
    unittest.main()
