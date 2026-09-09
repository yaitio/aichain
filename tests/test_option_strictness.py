"""
Who decides when an option cannot be honoured.

The library adapts and says so; that is settled. What was not settled is
whether a request carrying an option this provider has no control for should
go anyway. It depends on something the library cannot see:

  * dropping `temperature` gives a different answer — still an answer, and
    the caller can look at it;
  * dropping `seed` or `size` gives a result that **looks** right and is not,
    and nothing downstream will notice.

The second class is `_options.REQUIREMENT`. But even there the cost belongs to
the caller — a thumbnail does not care about its shape, a fixed layout slot
does, and a measurement arm is invalidated outright — so the choice is theirs
and the default is the one that keeps providers swappable.
"""

import unittest
import warnings

from yait_aichain import Model, UnsupportedOption
from yait_aichain.models import _options

MSGS = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
IMAGE = {"format": {"type": "image", "seed": 7}, "modalities": ["image"]}


def _build(model, output):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.to_request(MSGS, output)


class TestTheDefaultKeepsProvidersSwappable(unittest.TestCase):

    def test_an_unhonourable_option_still_sends(self):
        """A library that raises on a provider gap is one you cannot swap a
        provider under, which is the whole promise."""
        m = Model("recraftv3", api_key="k", options={"top_k": 37})
        _build(m, IMAGE)
        self.assertTrue(m.last_adaptations)

    def test_and_it_is_still_on_the_record(self):
        m = Model("recraftv3", api_key="k")
        _build(m, IMAGE)
        self.assertIn("seed", {a.option for a in m.last_adaptations})


class TestTheCallerCanAskToBeStopped(unittest.TestCase):

    def test_requirements_raises_only_for_the_invisible_losses(self):
        """`top_k` is dropped here too, and does not raise at this level: the
        answer is a different answer, not a wrong-shaped one."""
        m = Model("recraftv3", api_key="k", options={"top_k": 37},
                  on_unsupported="requirements")
        with self.assertRaises(UnsupportedOption) as caught:
            _build(m, IMAGE)
        self.assertEqual({a.option for a in caught.exception.adaptations},
                         {"seed"})

    def test_raise_covers_every_loss(self):
        m = Model("recraftv3", api_key="k", options={"top_k": 37},
                  on_unsupported="raise")
        with self.assertRaises(UnsupportedOption) as caught:
            _build(m, IMAGE)
        self.assertEqual({a.option for a in caught.exception.adaptations},
                         {"seed", "top_k"})

    def test_nothing_lost_means_nothing_raised(self):
        m = Model("gpt-4o", api_key="k", options={"temperature": 0.31},
                  on_unsupported="raise")
        _build(m, {"format": {"type": "text"}})
        self.assertEqual(m.last_adaptations, [])

    def test_the_record_survives_the_raise(self):
        """A caller catching this needs to know what happened; a raise that
        leaves the evidence unwritten is the silence the channel exists
        against."""
        m = Model("recraftv3", api_key="k", on_unsupported="raise")
        with self.assertRaises(UnsupportedOption):
            _build(m, IMAGE)
        self.assertTrue(m.last_adaptations)

    def test_a_misspelt_level_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            Model("gpt-4o", api_key="k", on_unsupported="strict")


class TestTheTwoClasses(unittest.TestCase):
    """Every option belongs to exactly one, and the split is the argument the
    strictness levels rest on."""

    def test_the_preferences_leave_a_visible_answer(self):
        for option in ("temperature", "top_p", "top_k", "reasoning",
                       "cache_control", "max_tokens"):
            self.assertFalse(_options.is_requirement(option), option)

    def test_the_requirements_leave_one_that_looks_right(self):
        for option in ("seed", "size", "aspect_ratio", "background",
                       "fidelity", "output_format", "compression"):
            self.assertTrue(_options.is_requirement(option), option)

    def test_an_old_name_lands_in_the_same_class(self):
        """`input_fidelity` is `fidelity`; a rename must not change what a
        caller's strictness setting does."""
        self.assertTrue(_options.is_requirement("input_fidelity"))
        self.assertTrue(_options.is_requirement("output_compression"))


if __name__ == "__main__":
    unittest.main()
