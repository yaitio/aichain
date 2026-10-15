"""
How data moves through a Chain, pinned across the change phase 6 made on purpose.

Pinned before phase 6 (PLAN-agent-chain.md) and rewritten by it, in 4.2.7: a
dict output now has its own key — and, for one version, is still spread flat
too, with a warning; a step with no key is named after its runner — and still
writes "result", which warns when read; a name no step writes may be a run
variable, so it still fails when the run reaches it, while a name only a later
step writes fails when the chain is built.
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

    def test_a_dict_output_sits_under_its_key_and_is_still_spread_with_a_warning(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = Chain([(fetch, "page"), (count, "words", {"text": "body"})]).run(
                variables={"url": "x"})
        self.assertEqual(result["page"], {"title": "T", "body": "B x"})
        self.assertEqual(result["page.body"], "B x")
        self.assertEqual(result["body"], "B x")               # the old spread, for one version
        self.assertEqual(result["words"], 2)
        self.assertTrue(any("spread" in str(w.message) for w in seen))

    def test_a_path_reads_into_it(self):
        result = Chain([(fetch, "page"), (count, "words", {"text": "page.body"})]).run(
            variables={"url": "x"})
        self.assertEqual(result["words"], 2)

    def test_a_step_with_no_key_is_named_after_its_runner(self):
        def first(url: str):
            return "first"

        def second(url: str):
            return "second"

        result = Chain([first, second]).run(variables={"url": "x"})
        self.assertEqual((result["first"], result["second"]), ("first", "second"))
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            self.assertEqual(result["result"], "second")      # the old key, for one version
        self.assertTrue(any("result['second']" in str(w.message) for w in seen))

    def test_a_name_no_step_writes_is_caught_when_the_run_reaches_it(self):
        chain = Chain([(count, "words", {"text": "bdy"})])     # may be a run variable
        with self.assertRaises(ValueError) as caught:
            chain.run(variables={"url": "x"})
        self.assertIn("'bdy'", str(caught.exception))
        self.assertIn("['url']", str(caught.exception))


class TestCaughtWhenBuilt(unittest.TestCase):

    def test_reading_what_only_a_later_step_writes(self):
        with self.assertRaises(ValueError) as caught:
            Chain([(count, "words", {"text": "page.body"}), (fetch, "page")])
        self.assertIn("only step 1 writes", str(caught.exception))

    def test_a_wrong_field_is_named_with_what_is_there(self):
        chain = Chain([(fetch, "page"), (count, "words", {"text": "page.bdy"})])
        with self.assertRaises(ValueError) as caught:
            chain.run(variables={"url": "x"})
        self.assertIn("did you mean 'page.body'", str(caught.exception))

    def test_two_steps_of_one_runner_with_no_keys(self):
        with self.assertRaises(ValueError):
            Chain([count, count])


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
