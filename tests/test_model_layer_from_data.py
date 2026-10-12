"""
What the model layer knows, read from the provider data rather than restated.

The provider keys accepted as a "provider/model" prefix and the prefixes named
in the error for an unknown model were both typed by hand beside the data, and
the error's list had fallen behind it. A frozen ToolCall carried a dict, so
hashing one raised. And `refresh("private")` demanded a key from a provider
whose data says it takes none.
"""

import unittest
from unittest.mock import MagicMock

from yait_aichain.models import Model
from yait_aichain.models._base import _PROVIDER_KEYS, refresh
from yait_aichain.models._calls import ToolCall
from yait_aichain.models._data import PROVIDERS


class TestFromTheData(unittest.TestCase):

    def test_every_provider_is_a_prefix(self):
        self.assertEqual(_PROVIDER_KEYS, frozenset(PROVIDERS))

    def test_the_error_names_every_prefix(self):
        with self.assertRaises(ValueError) as caught:
            Model("nonesuch-1", api_key="k")
        said = str(caught.exception)
        for word in ("claude-", "magistral-", "wan<digit>", "o<digit>", "chatgpt-image-"):
            self.assertIn(word, said)


class TestToolCall(unittest.TestCase):

    def test_a_call_hashes_and_equals_by_its_arguments_too(self):
        a, b = ToolCall("1", "look", {"q": "x"}), ToolCall("1", "look", {"q": "y"})
        self.assertEqual(len({a, ToolCall("1", "look", {"q": "x"})}), 1)
        self.assertNotEqual(a, b)


class TestRefreshAKeylessProvider(unittest.TestCase):

    def test_private_needs_no_key(self):
        import os
        from unittest.mock import patch
        client = MagicMock(list_models=MagicMock(return_value=[]))
        with patch.dict(os.environ, {}, clear=True), \
                patch("yait_aichain.models._base._build_client", return_value=client) as build:
            refresh("private")
        self.assertEqual(build.call_args.args[1], "")


if __name__ == "__main__":
    unittest.main()
