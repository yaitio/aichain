"""
A parameter one model takes and another refuses.

`input_fidelity` is the case that prompted this: an edit sent to gpt-image-2
came back HTTP 400, and the caller had no way to know in advance — the
library added nothing to the provider's own refusal. Which models refuse it
was established by asking the API, not by reading the guide: the guide names
only gpt-image-2, while both GPT Image 2.5 models and gpt-image-1-mini refuse
it too. Measured 2026-09-09.
"""

import base64
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yait_aichain import Model

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()
EDIT = [{"role": "user", "parts": [
    {"type": "text", "text": "make it darker"},
    {"type": "image", "source": {"kind": "base64", "mime": "image/png",
                                 "data": PNG}}]}]
FMT = {"format": {"type": "image", "quality": "low",
                  "reference_fidelity": "high"}}

REFUSES = ("gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst",
           "gpt-image-1-mini")
ACCEPTS = ("gpt-image-1.5", "gpt-image-1", "chatgpt-image-latest")


def _body(name):
    from yait_aichain.models import _adaptation
    _adaptation.reset_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = Model(name, api_key="k")
        _, body = m.to_request(EDIT, FMT)
    return repr(body), caught


class TestRejectedParameters(unittest.TestCase):

    def test_models_that_refuse_it_do_not_receive_it(self):
        for name in REFUSES:
            with self.subTest(model=name):
                body, _ = _body(name)
                self.assertNotIn("input_fidelity", body)

    def test_dropping_is_not_silent(self):
        """A parameter the caller set and did not get is worth a line."""
        for name in REFUSES:
            with self.subTest(model=name):
                _, caught = _body(name)
                self.assertTrue(any("reference_fidelity" in str(w.message)
                                    for w in caught))

    def test_models_that_accept_it_still_get_it(self):
        """The cure must not be worse: dropping it everywhere would remove a
        working feature from three models to spare four.

        Counted per option, not in total: the library now reports every
        option it had to change, so a bare warning count here would measure
        whatever else was in the request."""
        for name in ACCEPTS:
            with self.subTest(model=name):
                body, caught = _body(name)
                self.assertIn("input_fidelity", body)
                self.assertEqual(
                    [w for w in caught if "input_fidelity" in str(w.message)],
                    [])

    def test_a_parameter_that_was_never_set_warns_about_nothing(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Model("gpt-image-2", api_key="k").to_request(
                EDIT, {"format": {"type": "image", "quality": "low"}})
        self.assertEqual(
            [w for w in caught if "input_fidelity" in str(w.message)], [])


if __name__ == "__main__":
    unittest.main()
