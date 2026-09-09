"""
Three things can be wrong with an option, and each wants a different answer.

    the name is not one we know       → refuse where it was written
    the provider has no such control  → decline, say so, name the way round
    the value is not one this model
      accepts                         → refuse before the wire, list what is

The third used to be the provider's job, done badly and late: OpenAI answers
"Invalid value: 'ultra-max-supreme'. Supported values are: …" after the round
trip has been spent. The library holds the same list and holds it earlier.

The allowed set belongs to the **model**, not the provider — quality="xhigh"
is fine on gpt-image-2.5-flare and refused by gpt-image-1.5, measured against
the API on 2026-09-09.
"""

import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yait_aichain import Model

MSGS = [{"role": "user", "parts": [{"type": "text", "text": "a cat"}]}]


def _build(name, **fmt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Model(name, api_key="k")
        _, body = m.to_request(MSGS, {"format": {"type": "image", **fmt}})
    return m, body


class TestAValueOutsideTheSet(unittest.TestCase):

    def test_is_refused_before_the_wire(self):
        with self.assertRaises(ValueError) as ctx:
            _build("gpt-image-2.5-flare", quality="ultra-max-supreme")
        self.assertIn("ultra-max-supreme", str(ctx.exception))

    def test_and_the_message_lists_what_is_allowed(self):
        """Being told no is not useful; being told the alternatives is."""
        with self.assertRaises(ValueError) as ctx:
            _build("gpt-image-2.5-flare", quality="ultra-max-supreme")
        for value in ("low", "medium", "high"):
            self.assertIn(repr(value), str(ctx.exception))

    def test_the_set_belongs_to_the_model_not_the_provider(self):
        """Same provider, same option, different answer."""
        _, body = _build("gpt-image-2.5-flare", quality="xhigh")
        self.assertEqual(body["quality"], "xhigh")

        with self.assertRaises(ValueError) as ctx:
            _build("gpt-image-1.5", quality="xhigh")
        self.assertIn("gpt-image-1.5", str(ctx.exception))

    def test_a_value_that_is_fine_goes_out_untouched(self):
        _, body = _build("gpt-image-2.5-flare", quality="high")
        self.assertEqual(body["quality"], "high")


class TestANumberOutsideTheRange(unittest.TestCase):
    """Refusing an enum is right because "high" is not what somebody who
    wrote "xhigh" wanted. A number is different: past the maximum the intent
    is unambiguous, so it is clamped and the clamp is reported."""

    def test_is_clamped_not_refused(self):
        m, body = _build("gpt-image-2.5-flare", output_format="webp",
                         compression=150)
        self.assertEqual(body["output_compression"], 100)

    def test_and_the_clamp_is_reported(self):
        m, _ = _build("gpt-image-2.5-flare", output_format="webp",
                      compression=150)
        note, = [a for a in m.last_adaptations if a.option == "compression"]
        self.assertEqual((note.kind, note.asked, note.sent), ("adapted", 150, 100))

    def test_a_number_inside_the_range_is_left_alone(self):
        m, body = _build("gpt-image-2.5-flare", output_format="webp",
                         compression=50)
        self.assertEqual(body["output_compression"], 50)
        self.assertEqual([a for a in m.last_adaptations
                          if a.option == "compression"], [])


class TestAMultipartBodyCarriesEverythingAsText(unittest.TestCase):
    """A number sent through multipart arrives as a string.

    Compared type-exactly, `strength=0.4` looked absent while it was
    travelling in the request, and the library reported its own delivered
    option as declined. The looser comparison is confined to multipart, so on
    a JSON body an unrelated "1" cannot stand in for an asked 1.
    """

    def _edit(self, **fmt):
        import base64
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        msgs = [{"role": "user", "parts": [
            {"type": "text", "text": "darker"},
            {"type": "image", "source": {"kind": "base64",
                                         "mime": "image/png", "data": png}}]}]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Model("recraftv3", api_key="k")
            _, body = m.to_request(msgs, {"format": {"type": "image", **fmt}})
        return m, dict(body["fields"])

    def test_a_number_that_arrived_is_not_reported_as_declined(self):
        m, fields = self._edit(strength=0.4)
        self.assertEqual(fields["strength"], "0.4")
        self.assertEqual([a for a in m.last_adaptations
                          if a.option == "strength"], [])

    def test_and_one_that_was_clamped_still_is(self):
        m, fields = self._edit(strength=1.7)
        self.assertEqual(fields["strength"], "1.0")
        note, = [a for a in m.last_adaptations if a.option == "strength"]
        self.assertEqual(note.kind, "adapted")


class TestAProviderThatDeclaresNothing(unittest.TestCase):

    def test_leaves_the_value_alone(self):
        """Absence of a declared set is not an empty set: a provider whose
        data says nothing must keep behaving as it did."""
        m, _ = _build("flux-2-pro", aspect_ratio="21:9")
        self.assertEqual([a.kind for a in m.last_adaptations
                          if a.option == "aspect_ratio"], [])


if __name__ == "__main__":
    unittest.main()
