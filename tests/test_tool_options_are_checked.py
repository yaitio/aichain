"""
A tool's schema is authority over its options, in both directions.

`PLAN.md`: *typos in a Tool `options` are ignored; the schemas exist —
validate.* They did exist, and nothing read them. `searchPerplexity` declares
seven option keys and `options={"recencyy": "day"}` passed validation, reached
the tool and was dropped on the floor: the search ran unfiltered and answered
confidently. A typo cost a run and left no trace — the same shape as every
other defect this repository spent a week removing, one layer down from the
model options that were fixed in 2.3.0.

Both directions matter and only one was obvious. A key the schema does not
declare is a mistake to report; a key the tool *reads* and the schema does not
declare is worse, because a model cannot discover it at all. Measuring found
exactly one of the second kind, and it is pinned below.
"""

import inspect
import os
import pkgutil
import re
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yait_aichain.tools as tools_pkg                              # noqa: E402
from yait_aichain.tools._base import Tool                           # noqa: E402
from yait_aichain.tools.search.perplexity import searchPerplexity   # noqa: E402
from yait_aichain.tools.convert.to_speech import (                  # noqa: E402
    convertToSpeech, ttsOpenAI, ttsQwen)


def _tool_classes():
    found = {}
    for info in pkgutil.walk_packages(tools_pkg.__path__,
                                      prefix="yait_aichain.tools."):
        try:
            module = __import__(info.name, fromlist=["_"])
        except Exception:
            continue                     # optional dependency; not our subject
        for obj in vars(module).values():
            if (inspect.isclass(obj) and issubclass(obj, Tool)
                    and obj is not Tool):
                found[obj.__name__] = obj
    return found


def _declared_options(cls):
    props = (getattr(cls, "parameters", {}) or {}).get("properties", {}) or {}
    return set((props.get("options") or {}).get("properties") or {})


class TestATypoIsReported(unittest.TestCase):

    def setUp(self):
        self.tool = searchPerplexity(api_key="k")

    def test_the_model_is_told_what_it_may_use(self):
        """`check_args` feeds an Agent's attempt budget, so the message has to
        be actionable: the wrong name and the right ones."""
        message = self.tool.check_args(
            {"input": "x", "options": {"recencyy": "day"}})
        self.assertIsNotNone(message)
        self.assertIn("options.recencyy", message)
        self.assertIn("options.recency", message)

    def test_a_correct_call_says_nothing(self):
        self.assertIsNone(self.tool.check_args(
            {"input": "x", "options": {"recency": "day"}}))

    def test_a_missing_argument_is_still_reported_first(self):
        """It is the more basic failure, and reporting both at once would
        make the model fix the wrong one."""
        message = self.tool.check_args({"options": {"recencyy": "day"}})
        self.assertIn("missing required", message)

    def test_a_person_is_warned_rather_than_refused(self):
        """The asymmetry with the model path is deliberate: a subclass may
        legitimately read an option its schema does not advertise, and
        refusing a person's call over the library's reading of their own
        schema would be the library knowing better."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.tool._validate("x", {"recencyy": "day"})
        self.assertTrue(any("recencyy" in str(w.message) for w in caught))

    def test_a_free_form_options_dict_is_left_alone(self):
        """A nested object with no `properties` of its own is a free-form dict
        by declaration. A schema is authority only over what it describes."""

        class Loose(Tool):
            name = "loose"
            parameters = {"type": "object",
                          "properties": {"input": {"type": "string"},
                                         "options": {"type": "object"}}}

            def run(self, input=None, options=None):
                return input

        self.assertEqual(Loose().unknown_keys(
            {"input": "x", "options": {"anything": 1}}), [])


class TestTheSchemaMatchesWhatTheToolReads(unittest.TestCase):
    """The direction nobody was looking. An option a tool honours and its
    schema hides cannot be found by a model at all, and now also produces a
    warning for the person who uses it — so the two have to agree."""

    _READS = re.compile(r'\bo(?:pts|ptions)?\.get\(\s*["\']([\w_]+)["\']')

    def test_no_tool_reads_an_option_it_does_not_declare(self):
        offenders = []
        for name, cls in sorted(_tool_classes().items()):
            declared = _declared_options(cls)
            if not declared:
                continue
            try:
                source = inspect.getsource(cls)
            except (OSError, TypeError):
                continue
            undeclared = sorted(set(self._READS.findall(source)) - declared)
            if undeclared:
                offenders.append(f"{name}: {undeclared}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_qwen_declares_its_region_and_its_siblings_do_not(self):
        """The one that was found. `region` is DashScope's; putting it on the
        shared TTS schema would advertise it on OpenAI, Google and xAI, where
        nothing reads it — the opposite defect, and the one 2.3.0 was about."""
        self.assertIn("region", _declared_options(ttsQwen))
        self.assertNotIn("region", _declared_options(ttsOpenAI))
        self.assertNotIn("region", _declared_options(convertToSpeech))

    def test_extending_a_shared_schema_does_not_mutate_it(self):
        """Schemas are class attributes shared by inheritance: extending one
        in place would add the key to every sibling."""
        shared = convertToSpeech.parameters["properties"]["options"]
        self.assertIsNot(shared,
                         ttsQwen.parameters["properties"]["options"])


class TestTheCheckReachesTheWholeToolSurface(unittest.TestCase):

    def test_most_tools_describe_their_options(self):
        """If they did not, this validation would be decoration. 28 of 32 on
        the day it was written — the number is here so that a drop is
        visible rather than assumed."""
        classes = _tool_classes()
        described = [c for c in classes.values() if _declared_options(c)]
        self.assertGreaterEqual(len(described), 28,
                                f"only {len(described)} of {len(classes)} "
                                "tools describe their options")


if __name__ == "__main__":
    unittest.main()
