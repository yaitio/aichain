"""
The vocabulary is a contract, and the declaration says who honours it.

Stage 0 made every changed request report itself, but the report could not
tell two very different things apart: `top_k` on OpenAI — a real option this
provider has no control for — and a mistyped `temperatur`, which is not an
option at all. Both came back "this provider has no such control", and they
call for opposite reactions.

The declaration in the provider data supplies what a request diff never can:
what was *meant* to be available. Comparing it against what actually arrives
is how a promise the library does not keep becomes visible.
"""

import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yait_aichain import Model
from yait_aichain.models import registry
from yait_aichain.models import _adaptation
from yait_aichain.models._options import (UNIVERSAL_OPTIONS, UNIVERSAL_FORMAT,
                                          accepted_by, is_universal)

MSGS = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]


def _adapt(name, options=None, fmt=None, messages=None):
    _adaptation.reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Model(name, api_key="k", options=options or None)
        m.to_request(messages or MSGS,
                     {"format": {"type": fmt and "image" or "text",
                                 **(fmt or {})}})
    return {a.option: a for a in m.last_adaptations}


class TestTheVocabularyIsDeclared(unittest.TestCase):

    def test_every_declared_option_is_one_the_library_knows(self):
        """A provider cannot claim a control the vocabulary has no word for —
        that is how a provider's own name leaks back in as a universal one."""
        known = set(UNIVERSAL_OPTIONS) | set(UNIVERSAL_FORMAT)
        for provider in registry.PROVIDERS:
            declared = accepted_by(provider)
            if declared is None:
                continue
            with self.subTest(provider=provider):
                self.assertEqual(declared - known, set())

    def test_a_provider_that_declares_nothing_is_not_a_provider_that_takes_nothing(self):
        """Absence of a claim is not a claim: data written before the
        declaration existed must keep behaving as it did."""
        self.assertIsNone(accepted_by("no-such-provider"))


class TestTheReportTellsThemApart(unittest.TestCase):

    def test_an_unknown_name_is_refused_where_it_was_written(self):
        """Not at the wire, and not — as before — nowhere at all. A misspelt
        option used to be accepted, ignored, and cost a whole run to notice."""
        with self.assertRaises(ValueError) as ctx:
            Model("gpt-4o", api_key="k", options={"temperatur": 0.5})
        self.assertIn("temperatur", str(ctx.exception))

    def test_the_refusal_lists_what_does_exist(self):
        """A typo is fixed by seeing the right spelling, not by being told no."""
        with self.assertRaises(ValueError) as ctx:
            Model("gpt-4o", api_key="k", options={"temperatur": 0.5})
        self.assertIn("temperature", str(ctx.exception))

    def test_it_points_at_the_other_vocabulary(self):
        """`quality` is a real option — of the other kind. Saying only "not
        known" would send a reader looking for a misspelling that is not
        there."""
        with self.assertRaises(ValueError) as ctx:
            Model("gpt-4o", api_key="k", options={"quality": "high"})
        self.assertIn("output=", str(ctx.exception))

    def test_a_real_option_the_provider_lacks_is_declined_with_a_way_round(self):
        made = _adapt("gpt-4o", {"top_k": 37})
        self.assertEqual(made["top_k"].kind, "declined")
        self.assertIn("top_p", made["top_k"].why)     # what to use instead

    def test_the_message_names_the_provider(self):
        made = _adapt("gpt-4o", {"top_k": 37})
        self.assertIn("openai", made["top_k"].why)

    def test_format_keys_are_vocabulary_too(self):
        """They were briefly reported as unknown names: a second vocabulary,
        asked per call rather than per model, and it has to be declared as
        well or every image option reads as a typo."""
        made = _adapt("gpt-image-2.5-flare", fmt={"seed": 42})
        self.assertEqual(made["seed"].kind, "declined")
        self.assertIn("openai", made["seed"].why)

    def test_a_shape_asked_for_either_way_is_honoured_either_way(self):
        """Providers split down the middle on this — some take pixels and
        ignore a ratio, some refuse pixels outright — and a caller writing one
        request for both used to get half of it silently dropped."""
        by_ratio = _adapt("gpt-image-2.5-flare", fmt={"aspect_ratio": "16:9"})
        self.assertEqual(by_ratio["aspect_ratio"].kind, "adapted")
        self.assertIn("size=", str(by_ratio["aspect_ratio"].sent))

        by_pixels = _adapt("gemini-2.5-flash-image", fmt={"size": "1024x1536"})
        self.assertEqual(by_pixels["size"].kind, "adapted")
        self.assertIn("aspectRatio", str(by_pixels["size"].sent))


class TestFlagsCannotBeTracedByValue(unittest.TestCase):
    """Every `True` in a body looks alike."""

    def test_a_boolean_needs_the_site_to_speak(self):
        """`cache_control=True` was matching an unrelated `enable_thinking=
        True` and reading as delivered."""
        made = _adapt("QwQ-32B", {"cache_control": True})
        self.assertEqual(made["cache_control"].kind, "declined")

    def test_and_is_believed_when_it_does(self):
        """Anthropic marks the end of the stable prefix, so it reports an
        adaptation rather than a decline — and only when there is a prefix to
        mark: a lone user message has none, and saying so is right."""
        with_prefix = [{"role": "system",
                        "parts": [{"type": "text", "text": "be brief"}]}, *MSGS]
        made = _adapt("claude-sonnet-4-6", {"cache_control": True},
                      messages=with_prefix)
        self.assertEqual(made["cache_control"].kind, "adapted")

        alone = _adapt("claude-sonnet-4-6", {"cache_control": True})
        self.assertEqual(alone["cache_control"].kind, "declined")


class TestAskingBeforeCalling(unittest.TestCase):

    def test_a_caller_can_check_without_making_a_request(self):
        self.assertNotIn("top_k", registry.accepts("openai"))
        self.assertIn("top_k", registry.accepts("anthropic"))

    def test_the_vocabulary_is_reachable(self):
        self.assertIn("reasoning", registry.OPTIONS)
        self.assertTrue(is_universal("aspect_ratio"))


if __name__ == "__main__":
    unittest.main()
