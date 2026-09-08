"""
The library may adapt a request to a provider. It may not do so in silence.

Every check here is about the *report*, not the adaptation: dropping `top_k`
where the wire has no such field is right, and forcing a reasoner's
temperature is right. What was wrong was that none of it was said, so a
caller comparing two models believed both arms ran with the settings they
were given.
"""

import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yait_aichain import Model
from yait_aichain.models import _adaptation

MSGS = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
TEXT = {"format": {"type": "text"}}


def _build(name, **options):
    _adaptation.reset_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = Model(name, api_key="k", options=options or None)
        _, body = m.to_request(MSGS, TEXT)
    return m.last_adaptations, body, caught


def _for(made, option):
    return [a for a in made if a.option == option]


class TestNothingVanishes(unittest.TestCase):

    def test_an_option_the_wire_has_no_field_for_is_declined(self):
        made, _, _ = _build("gpt-4o", top_k=37)
        self.assertEqual([a.kind for a in _for(made, "top_k")], ["declined"])

    def test_an_option_that_travels_says_nothing(self):
        """Anthropic takes top_k, so there is nothing to report."""
        made, body, _ = _build("claude-sonnet-4-6", top_k=37)
        self.assertEqual(_for(made, "top_k"), [])
        self.assertEqual(body["top_k"], 37)

    def test_a_default_nobody_chose_is_not_reported(self):
        """Only what the caller set. A provider ignoring its own default is
        not news, and reporting it is how a channel stops being read."""
        made, _, _ = _build("gpt-4o")
        self.assertEqual(made, [])


class TestTheReportIsTrue(unittest.TestCase):
    """A notice of the wrong kind is worse than none: it misinforms."""

    def test_a_conversion_is_called_a_conversion(self):
        for name, sent in (("claude-sonnet-4-6", "thinking"),
                           ("gemini-2.5-flash", "thinkingConfig.thinkingBudget")):
            with self.subTest(model=name):
                made, _, _ = _build(name, reasoning="high")
                a, = _for(made, "reasoning")
                self.assertEqual(a.kind, "adapted")
                self.assertEqual(a.sent, sent)

    def test_a_model_swap_is_called_a_swap(self):
        """The loudest case: a different model answers, at a different price,
        and a comparison believing both arms ran the same model is void."""
        made, body, _ = _build("deepseek-chat", reasoning="high")
        a, = _for(made, "reasoning")
        self.assertEqual(a.kind, "swapped")
        self.assertEqual(a.sent, "deepseek-reasoner")
        self.assertEqual(body["model"], "deepseek-reasoner")

    def test_an_overridden_value_names_what_replaced_it(self):
        made, _, _ = _build("claude-sonnet-4-6", reasoning="high",
                            temperature=0.31)
        a, = _for(made, "temperature")
        self.assertEqual((a.kind, a.asked, a.sent), ("declined", 0.31, 1.0))

    def test_asking_to_reason_can_multiply_the_token_ceiling(self):
        """Worth its own notice: it is the caller's bill that changes."""
        made, body, _ = _build("claude-sonnet-4-6", reasoning="high",
                               max_tokens=100)
        a, = _for(made, "max_tokens")
        self.assertEqual(a.kind, "adapted")
        self.assertGreater(body["max_tokens"], 100)


class TestItIsSaidOutLoud(unittest.TestCase):

    def test_a_warning_is_raised(self):
        _, _, caught = _build("deepseek-chat", reasoning="high")
        self.assertTrue(any("deepseek-reasoner" in str(w.message)
                            for w in caught))

    def test_only_once_per_model_and_option(self):
        """Warning fatigue is how a channel like this stops being read: a
        thousand-step loop must not repeat itself a thousand times."""
        _adaptation.reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = Model("gpt-4o", api_key="k", options={"top_k": 37})
            for _ in range(5):
                m.to_request(MSGS, TEXT)
        self.assertEqual(len(caught), 1)

    def test_the_record_survives_on_the_object(self):
        """The outlet that matters: a warning is lost in a log, this is what
        a run record can keep."""
        m = Model("gpt-4o", api_key="k", options={"top_k": 37})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.to_request(MSGS, TEXT)
        self.assertEqual([a.option for a in m.last_adaptations], ["top_k"])


class TestDetectionItself(unittest.TestCase):
    """Two false readings this walker gave before, both worth pinning."""

    def test_a_boolean_is_not_found_inside_a_float(self):
        """`True == 1 == 1.0` in Python: `cache_control=True` was matching
        `temperature=1.0`, and five providers were reported as honouring an
        option none of them implements."""
        made, _, _ = _build("gpt-4o", cache_control=True)
        self.assertEqual([a.kind for a in _for(made, "cache_control")],
                         ["declined"])

    def test_a_value_inside_a_multipart_pair_counts_as_arrived(self):
        """A multipart body is a list of (name, value) tuples; treating a
        pair as one opaque leaf hid the value inside it."""
        found = _adaptation._leaves({"fields": [("quality", "low")]})
        self.assertIn("low", found.values())


if __name__ == "__main__":
    unittest.main()
