"""
A model option's value is checked where it is written.

`temperature="0.3"`, `max_tokens=-1` and `None`, and 1.7 on a model declared
0–1 went to the provider and came back 400 after a round trip; `True` passed
a [0, 1] range as 1; a build that raised left the previous call's
adaptations; `fidelity="auto"` was 0.5, OpenAI's expensive high.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.models import Model
from yait_aichain.models._adaptation import ADAPTED
from yait_aichain.models._options import canonical_value

MSG = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]


class TestValues(unittest.TestCase):

    def test_wrong_types_and_ranges_are_refused_at_construction(self):
        for opts in ({"temperature": "0.3"}, {"max_tokens": -1}, {"max_tokens": None},
                     {"top_p": 1.5}, {"top_k": 0}, {"temperature": True}, {"max_tokens": 10.5}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                Model("gpt-4o", api_key="k", options=opts)

    def test_a_declared_range_is_clamped_and_said(self):
        m = Model("glm-5.1", api_key="k", options={"temperature": 1.7})
        self.assertEqual(m.temperature, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, body, made = m.build(MSG, {"format": {"type": "text"}})
        self.assertIn(("temperature", ADAPTED), [(a.option, a.kind) for a in made])

    def test_auto_is_not_a_fidelity(self):
        self.assertEqual(canonical_value("fidelity", "auto"), ("auto", None))


if __name__ == "__main__":
    unittest.main()
