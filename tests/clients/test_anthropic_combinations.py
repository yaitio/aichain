"""
Combinations Claude refuses are adapted with a notice, or refused before the
round trip.

`reasoning="high"` with a json_schema, `top_p` beside the temperature,
`top_k` with thinking, `temperature=1.5`: each was sent, answered 400, and
left `last_adaptations` empty — where the neighbouring branch, tools with a
schema, already refused before the network.
"""

import unittest
import warnings

from yait_aichain import Model

MSGS = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
TEXT = {"modalities": ["text"], "format": {"type": "text"}}


def _build(output=TEXT, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Model("claude-sonnet-4-5", api_key="k", options=options)
        _, body = m.to_request(MSGS, output)
    return body, {(a.option, a.kind) for a in m.last_adaptations}


class TestCombinations(unittest.TestCase):

    def test_reasoning_with_a_schema_is_refused_here(self):
        schema = {"modalities": ["text"], "format": {"type": "json_schema", "schema": {"type": "object"}}}
        with self.assertRaisesRegex(ValueError, "reasoning with output json_schema"):
            _build(schema, reasoning="high")

    def test_top_p_drops_the_temperature(self):
        body, notes = _build(top_p=0.8, temperature=0.3)
        self.assertEqual((body.get("top_p"), "temperature" in body), (0.8, False))
        self.assertIn(("temperature", "adapted"), notes)

    def test_thinking_drops_top_k_and_lifts_top_p(self):
        body, notes = _build(reasoning="high", top_k=20, top_p=0.5)
        self.assertNotIn("top_k", body)
        self.assertEqual(body["top_p"], 0.95)
        self.assertNotIn("temperature", body)
        self.assertIn(("top_k", "declined"), notes)

    def test_temperature_above_one_is_clamped(self):
        body, notes = _build(temperature=1.5)
        self.assertEqual(body["temperature"], 1.0)
        self.assertIn(("temperature", "adapted"), notes)


if __name__ == "__main__":
    unittest.main()
