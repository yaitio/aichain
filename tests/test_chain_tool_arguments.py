"""
How a chain fills a tool's arguments (3.4.0).

A mapping value is a variable's name, a template, one placeholder alone (the
value itself), or a dict resolved key by key with literals kept. Before 3.4.0
a template was looked up as a variable's name, found nothing and was dropped
in silence — the tool then failed on the missing argument — and a dict raised
``TypeError: unhashable type: 'dict'``. The reranker's docstring showed exactly
such a mapping as a working example.

And what a saved chain keeps of a tool: search, fetch and decision tools wrote
no settings until 3.4.0, so a renamed search came back under its class name and
a decision tool did not load at all (``missing … 'questions'``).
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Chain
from yait_aichain.tools import Tool, decide, fetch, search


class Echo(Tool):
    name = "echo"; description = "Echo."; risk = "read"
    parameters = {"type": "object", "properties": {"input": {}, "options": {"type": "object"}},
                  "required": ["input"]}

    def run(self, input, options=None):
        return {"got": [input, options]}


V = {"docs": ["a", "b"], "q": "fox", "topic": "a fox"}


def got(mapping):
    return Chain(steps=[(Echo(), "o", mapping)]).run(variables=V)["got"]


class TestFillingArguments(unittest.TestCase):

    def test_a_name_is_the_variables_value(self):
        self.assertEqual(got({"input": "docs"}), [["a", "b"], None])

    def test_an_unmapped_parameter_takes_its_namesake(self):
        res = Chain(steps=[(Echo(), "o")]).run(variables={"input": "x"})
        self.assertEqual(res["got"], ["x", None])

    def test_a_template_is_substituted(self):
        self.assertEqual(got({"input": "Draw {topic} in watercolor"}), ["Draw a fox in watercolor", None])

    def test_one_placeholder_alone_is_the_value_itself(self):
        self.assertEqual(got({"input": "{docs}"}), [["a", "b"], None])     # a list stays a list

    def test_a_dict_is_resolved_and_its_literals_kept(self):
        self.assertEqual(got({"input": "docs", "options": {"query": "{q}", "top_n": 5, "lang": "en"}}),
                         [["a", "b"], {"query": "fox", "top_n": 5, "lang": "en"}])


class TestSaidNotDropped(unittest.TestCase):

    def test_a_template_naming_a_missing_variable(self):
        with self.assertRaisesRegex(ValueError, r"names \['colour'\].*it has \['docs', 'q', 'topic'\]"):
            got({"input": "Draw {colour} {topic}"})

    def test_a_name_that_is_not_a_variable(self):
        with self.assertRaisesRegex(ValueError, "mapped to the variable 'nope'"):
            got({"input": "nope"})

    def test_a_parameter_the_tool_does_not_take(self):
        with self.assertRaisesRegex(ValueError, r"maps \['inptu'\] into 'echo'"):
            got({"inptu": "docs"})


class TestWhatASavedChainKeeps(unittest.TestCase):

    def setUp(self):
        os.environ.setdefault("EXA_API_KEY", "k")
        os.environ.setdefault("OPENAI_API_KEY", "k")

    def _round_trip(self, tool):
        path = os.path.join(tempfile.mkdtemp(), "c.yaml")
        Chain(steps=[(tool, "o")]).save(path)
        return Chain.load(path)._steps[0][0], open(path).read()

    def test_a_renamed_search_comes_back_renamed_and_without_its_key(self):
        back, text = self._round_trip(search("exa", api_key="secret-key", name="web"))
        self.assertEqual(back.name, "web")
        self.assertNotIn("secret-key", text)

    def test_a_decision_comes_back_with_its_questions(self):
        q = {"urgent": {"type": "predicate", "instructions": "Is this urgent?"}}
        back, text = self._round_trip(decide("openai", questions=q, name="triage", api_key="secret-key"))
        self.assertEqual((back.name, back.questions), ("triage", q))
        self.assertNotIn("secret-key", text)

    def test_a_local_fetch_keeps_its_settings_but_not_its_headers(self):
        import warnings
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            back, text = self._round_trip(fetch("local", respect_robots=False, timeout=(5, 9),
                                                headers={"Cookie": "session=abc"}))
        self.assertEqual((back.respect_robots, back._timeout), (False, (5, 9)))
        self.assertNotIn("session=abc", text)
        self.assertTrue(any("headers are not saved" in str(w.message) for w in seen))


if __name__ == "__main__":
    unittest.main()
