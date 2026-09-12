"""
One input, every text provider: does the same request mean the same thing?

Provider interchangeability is the library's main promise — "change one word
to switch providers" is on the front page — and the places it has broken have
all been the same shape: a request that works on six providers and quietly
does something else on the seventh. Google entered its image branch on
`modalities` while everyone else was asked with `format.type == "image"`, so
the same call rendered a picture six times and text once. The probe that was
supposed to catch it was not touching the path.

So this asks one question per axis, of every provider at once, and the value
is in the *comparison* rather than in any single answer. A per-provider test
written by whoever wired that provider agrees with itself by construction;
a table asks whether they agree with each other.

Nothing here talks to a network: a request is built and inspected. What a
provider *does* with what it receives is the parameter matrix's job
(`scripts/parameters.py`), established by effect against the live APIs.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Model                                  # noqa: E402
from yait_aichain.models._base import models as _models         # noqa: E402
from yait_aichain.models._data import PROVIDERS                 # noqa: E402

#: One text model per provider. The image-only ones are excluded by name
#: rather than skipped silently — a provider that grows a text model should
#: turn up here rather than stay quietly out of the comparison.
IMAGE_ONLY = {"bfl", "recraft", "reve", "private"}

SYSTEM = {"role": "system", "parts": [{"type": "text",
                                       "text": "You are terse."}]}
USER = {"role": "user", "parts": [{"type": "text", "text": "Say hi."}]}


def _text_models():
    for provider in sorted(PROVIDERS):
        if provider in IMAGE_ONLY:
            continue
        names = [n for n in _models(provider=provider)
                 if "image" not in n and "edit" not in n]
        assert names, f"{provider} has no text model to compare"
        yield provider, names[0]


def _build(name, output, messages=(SYSTEM, USER)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Model(name, api_key="k").to_request(list(messages), output)


def _find(obj, key):
    """Every value stored under *key*, at any depth."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found += _find(v, key)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found += _find(v, key)
    return found


class TestTheSameInputReachesEveryProvider(unittest.TestCase):

    def test_a_system_prompt_is_never_dropped(self):
        """It may be a top-level field or a first message — the providers
        disagree and that is fine. Vanishing is not: an instruction the caller
        wrote and the model never saw changes the answer and leaves no trace."""
        for provider, name in _text_models():
            with self.subTest(provider=provider):
                _, body = _build(name, {"format": {"type": "text"}})
                self.assertIn("You are terse.", str(body),
                              f"{provider}: the system prompt did not travel")

    def test_the_user_text_arrives(self):
        for provider, name in _text_models():
            with self.subTest(provider=provider):
                _, body = _build(name, {"format": {"type": "text"}})
                self.assertIn("Say hi.", str(body))

    def test_json_mode_asks_for_json_somewhere(self):
        """A unified `format: json` is still an open item — the field is the
        provider's own — so this asserts the intent survives, not its
        spelling. The day one provider drops it entirely, this is what
        notices."""
        undelivered = []
        for provider, name in _text_models():
            _, body = _build(name, {"format": {"type": "json"}})
            if "json" not in str(body).lower():
                undelivered.append(provider)
        self.assertEqual(undelivered, [],
                         f"json was asked for and did not reach: {undelivered}")

    def test_a_ceiling_on_the_answer_always_travels(self):
        """Under whichever name — `max_tokens`, `max_completion_tokens`,
        `maxOutputTokens`. An unbounded answer is a bill."""
        for provider, name in _text_models():
            with self.subTest(provider=provider):
                _, body = _build(name, {"format": {"type": "text"}})
                names = ("max_tokens", "max_completion_tokens",
                         "maxOutputTokens")
                self.assertTrue(
                    any(_find(body, n) for n in names),
                    f"{provider}: no ceiling on the answer's length")

    def test_the_model_name_is_on_the_request(self):
        for provider, name in _text_models():
            with self.subTest(provider=provider):
                path, body = _build(name, {"format": {"type": "text"}})
                self.assertIn(name, str(body) + path)


class TestWhereTheyLegitimatelyDiffer(unittest.TestCase):
    """Not every difference is a defect, and pinning the ones that are real
    keeps the tests above honest: a check that has to be loosened until it
    passes everywhere is measuring nothing."""

    def test_the_system_prompt_lands_in_two_different_places(self):
        """Anthropic and Google carry it beside the messages; the
        OpenAI-compatible family carries it as the first message. Both are
        correct, and a caller never sees the difference — which is the
        promise."""
        beside, inside = [], []
        for provider, name in _text_models():
            _, body = _build(name, {"format": {"type": "text"}})
            top = {"system", "system_instruction", "instructions"}
            (beside if top & set(body) else inside).append(provider)
        self.assertTrue(beside, "no provider carries it beside the messages")
        self.assertTrue(inside, "no provider carries it as a message")

    def test_the_ceiling_has_three_names(self):
        spellings = set()
        for provider, name in _text_models():
            _, body = _build(name, {"format": {"type": "text"}})
            for n in ("max_tokens", "max_completion_tokens",
                      "maxOutputTokens"):
                if _find(body, n):
                    spellings.add(n)
        self.assertGreater(len(spellings), 1,
                           "the wire names converged — the test above can be "
                           "tightened to one name")


if __name__ == "__main__":
    unittest.main()
