"""
A listed model is one the provider serves, and an unlisted name says so.

`Model("grok-imagine-image-pro")` matched the `grok-` prefix, was in the
registry, and the xAI API rejected it: a listed model that did not exist, and
nothing in the library could have told. Two halves are pinned here.

Offline: a name the registry does not know, routed by its prefix alone, is
reported once — with the closest listed names, since a retired model and a
typo look the same — and an explicit `provider/name` is not.

Live: every listed model is in the provider's own model list, or carries
`unlisted = "why"` in its data. A provider retiring a model then fails the
nightly run instead of a user's first call, and a marker the provider has
caught up with fails it too, so the reasons do not outlive their facts.
"""

import os
import unittest
import warnings

from yait_aichain.models import Model
from yait_aichain.models import _base
from yait_aichain.models._data import PROVIDERS


def _built(name):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        model = Model(name, api_key="k")
    return model, [str(w.message) for w in seen if "registry" in str(w.message)]


class TestAnUnlistedNameIsSaid(unittest.TestCase):

    def setUp(self):
        _base._UNLISTED_SAID.clear()

    def test_a_listed_name_says_nothing(self):
        model, said = _built("gpt-5.5")
        self.assertTrue(model.listed)
        self.assertEqual(said, [])

    def test_a_name_routed_by_prefix_alone_is_said_with_its_neighbours(self):
        model, said = _built("grok-imagine-image-pro")
        self.assertFalse(model.listed)
        self.assertEqual(len(said), 1)
        self.assertIn("not in the xai registry", said[0])
        self.assertIn("grok-imagine-image", said[0])        # did you mean
        self.assertIn("'xai/grok-imagine-image-pro'", said[0])

    def test_once_per_name(self):
        _built("gpt-5.5-imaginary")
        self.assertEqual(_built("gpt-5.5-imaginary")[1], [])

    def test_an_explicit_provider_is_the_caller_meaning_it(self):
        model, said = _built("openai/ft:gpt-4o:org:abc")
        self.assertFalse(model.listed)
        self.assertEqual(said, [])


class TestEveryMarkerHasAReason(unittest.TestCase):

    def test_unlisted_is_a_sentence(self):
        for key, data in PROVIDERS.items():
            for name, model in data.get("models", {}).items():
                if "unlisted" in model:
                    with self.subTest(model=f"{key}/{name}"):
                        self.assertIsInstance(model["unlisted"], str)
                        self.assertGreater(len(model["unlisted"].split()), 3)


def _keyed(key):
    env = PROVIDERS[key]["provider"].get("env_key")
    return bool(env and os.environ.get(env, "") not in ("", "k", "test-key"))


class TestRegistryAgreesWithProvidersLive(unittest.TestCase):

    def _check(self, key):
        models = PROVIDERS[key].get("models", {})
        if not models or not _keyed(key):
            self.skipTest(f"{key}: no key")
        try:
            served = set(Model(f"{key}/{next(iter(models))}").client.list_models())
        except NotImplementedError:
            self.skipTest(f"{key} has no model list")
        gone = [n for n, m in models.items() if n not in served and "unlisted" not in m]
        caught_up = [n for n, m in models.items() if n in served and "unlisted" in m]
        self.assertEqual(gone, [], f"{key} no longer lists these; probe them, then drop or mark unlisted")
        self.assertEqual(caught_up, [], f"{key} lists these again; drop their unlisted marker")

    def test_every_provider(self):
        for key in sorted(PROVIDERS):
            with self.subTest(provider=key):
                self._check(key)


if __name__ == "__main__":
    unittest.main()
