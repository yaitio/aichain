"""
`extra` — the escape hatch a closed vocabulary needs.

The model options are a closed set on purpose: a typo in them is unambiguous
and raises. But a closed set with no valve is a cage. Perplexity's
`search_domain_filter` has no universal meaning and never will — nobody else
has a search index — so the answer to "how do I filter by domain?" was "not
through this library", and a caller who needs it drops the whole abstraction
rather than one field.

What goes in travels **verbatim and unchecked**: nothing translates it,
nothing declines it, no notice is emitted. That is the deal, and the library
states it once rather than pretending to supervise. The one thing it still
owes is visibility — a measurement run has to be able to say what its arms
actually sent.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Model                                      # noqa: E402

MSGS = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
TEXT = {"format": {"type": "text"}}


def _build(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.to_request(MSGS, TEXT)[1]


class TestItReachesTheWire(unittest.TestCase):

    def test_a_provider_only_field_travels(self):
        body = _build(Model("sonar", api_key="k", options={
            "extra": {"search_domain_filter": ["arxiv.org"]}}))
        self.assertEqual(body["search_domain_filter"], ["arxiv.org"])

    def test_it_wins_over_the_library_s_own_field(self):
        """A caller reaching for `extra` is overriding. A merge that lost to
        the library would be an escape hatch that does not escape."""
        body = _build(Model("gpt-4o", api_key="k",
                            options={"temperature": 0.1,
                                     "extra": {"temperature": 1.9}}))
        self.assertEqual(body["temperature"], 1.9)

    def test_it_works_on_every_provider(self):
        for name in ("gpt-4o", "claude-fable-5", "gemini-2.5-flash",
                     "deepseek-chat", "sonar"):
            with self.subTest(model=name):
                body = _build(Model(name, api_key="k",
                                    options={"extra": {"x_custom": 1}}))
                self.assertEqual(body["x_custom"], 1)


class TestItIsNotSupervised(unittest.TestCase):
    """The deal, stated as tests so it cannot drift into half-supervision."""

    def test_no_warning(self):
        model = Model("sonar", api_key="k",
                      options={"extra": {"whatever_you_like": True}})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.to_request(MSGS, TEXT)
        self.assertEqual([w for w in caught
                          if "whatever_you_like" in str(w.message)], [])

    def test_no_adaptation_recorded(self):
        """`note_absent` reports an option that left no trace; every `extra`
        field is outside the vocabulary that mechanism speaks, so reporting
        one would be the library commenting on a wheel it handed over."""
        model = Model("gpt-4o", api_key="k",
                      options={"extra": {"nothing_reads_this": 1}})
        _build(model)
        self.assertEqual(model.last_adaptations, [])

    def test_a_typo_in_a_real_option_still_raises(self):
        """The valve must not become a way to smuggle mistakes past the
        closed vocabulary."""
        with self.assertRaises(ValueError):
            Model("gpt-4o", api_key="k", options={"temperatur": 0.5})

    def test_and_the_error_points_at_the_valve(self):
        with self.assertRaises(ValueError) as caught:
            Model("gpt-4o", api_key="k", options={"search_domain_filter": []})
        self.assertIn("extra", str(caught.exception))

    def test_it_must_be_a_dict(self):
        with self.assertRaises(ValueError):
            Model("gpt-4o", api_key="k", options={"extra": ["not", "a", "dict"]})


class TestItIsStillVisible(unittest.TestCase):
    """No warning is not the same as no record. Two arms differing only by an
    `extra` field must not look identical in a run's own account of itself —
    that is the failure the notice channel exists to prevent, arriving by the
    one door that channel does not watch."""

    def test_effective_options_carries_it(self):
        model = Model("gpt-4o", api_key="k",
                      options={"extra": {"search_domain_filter": ["a.com"]}})
        self.assertEqual(model.effective_options["extra"],
                         {"search_domain_filter": ["a.com"]})

    def test_two_arms_are_distinguishable(self):
        plain = Model("gpt-4o", api_key="k")
        tuned = Model("gpt-4o", api_key="k", options={"extra": {"x": 1}})
        self.assertNotEqual(plain.effective_options, tuned.effective_options)


if __name__ == "__main__":
    unittest.main()
