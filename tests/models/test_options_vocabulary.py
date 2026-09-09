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


class TestRenamedNames(unittest.TestCase):
    """Two names were replaced; the old ones still work.

    Which two, and why only two, came out of measuring rather than opinion:
    `aspect_ratio` goes on the wire under that name at four providers, `size`
    at three, `output_format` at two — the field's shared lexicon, not one
    vendor's. `output_compression` and `input_fidelity` were OpenAI's alone,
    the first carrying a redundant "output" inside a dict already called
    `output["format"]`, the second ambiguous about what the input is fidelity
    to.
    """

    def _fmt(self, name, **fmt):
        from yait_aichain.models import _adaptation
        _adaptation.reset_warnings()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Model(name, api_key="k")
            _, body = m.to_request(
                [{"role": "user", "parts": [{"type": "text", "text": "a cat"}]}],
                {"format": {"type": "image", **fmt}})
        return m, body

    def test_the_old_name_still_reaches_the_wire(self):
        """Removing it would break working code."""
        _, body = self._fmt("gpt-image-2.5-flare", output_format="webp",
                            output_compression=50)
        self.assertEqual(body["output_compression"], 50)

    def test_and_the_caller_is_told_it_has_a_new_name(self):
        """Accepting it in silence would leave somebody writing a name that
        is no longer in the documentation."""
        m, _ = self._fmt("gpt-image-2.5-flare", output_format="webp",
                         output_compression=50)
        note, = [a for a in m.last_adaptations if a.option == "output_compression"]
        self.assertEqual((note.kind, note.sent), ("translated", "compression"))

    def test_the_new_name_says_nothing(self):
        m, body = self._fmt("gpt-image-2.5-flare", output_format="webp",
                            compression=50)
        self.assertEqual(body["output_compression"], 50)
        self.assertEqual(m.last_adaptations, [])

    def test_a_name_shared_across_providers_was_left_alone(self):
        """`output_format` is BFL's word as much as OpenAI's — renaming a
        shared term would have made the vocabulary less neutral, not more."""
        from yait_aichain.models._options import ALIASES, UNIVERSAL_FORMAT
        self.assertNotIn("output_format", ALIASES)
        self.assertIn("output_format", UNIVERSAL_FORMAT)


class TestWrongEverywhereVersusWrongHere(unittest.TestCase):
    """Where to be strict, decided by one question: is this request wrong at
    every provider, or only at this one?

    `top_k` on Perplexity is wrong only there — Anthropic and Google honour
    it — and raising would force a caller to branch per provider, which is
    the promise the library exists to keep. `background` on a *text* result
    is wrong at every provider that will ever exist, because text has no
    background, so it stops here and nothing is sent.
    """

    def test_a_key_that_cannot_mean_anything_here_raises(self):
        for name in ("gpt-4o", "claude-sonnet-4-6", "gpt-image-2.5-flare"):
            with self.subTest(model=name):
                with self.assertRaises(ValueError) as ctx:
                    Model(name, api_key="k").to_request(
                        MSGS, {"format": {"type": "text",
                                          "background": "transparent"}})
                self.assertIn("no meaning for a 'text' result",
                              str(ctx.exception))

    def test_and_says_that_no_provider_would_have_taken_it(self):
        """Otherwise the reader's next move is to try another provider."""
        with self.assertRaises(ValueError) as ctx:
            Model("gpt-4o", api_key="k").to_request(
                MSGS, {"format": {"type": "text", "background": "transparent"}})
        self.assertIn("not just this one", str(ctx.exception))

    def test_a_gap_at_one_provider_is_still_only_a_notice(self):
        """The request does what was asked, minus a refinement — and code
        written once for several providers keeps running."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Model("sonar", api_key="k", options={"top_k": 37})
            _, body = m.to_request(MSGS, {"format": {"type": "text"}})
        self.assertNotIn("top_k", body)
        self.assertEqual([a.kind for a in m.last_adaptations], ["declined"])

    def test_a_key_right_for_the_output_but_wrong_for_the_model(self):
        """Asking a text model for an image is not nonsense in the same way:
        the key fits the request, the model does not."""
        from yait_aichain.models import _adaptation
        _adaptation.reset_warnings()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Model("gpt-4o", api_key="k")
            m.to_request(MSGS, {"format": {"type": "image",
                                           "background": "transparent"}})
        why, = [a.why for a in m.last_adaptations if a.option == "background"]
        self.assertIn("does not produce images", why)
