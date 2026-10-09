"""
`search("exa")`, `fetch("local")`, `decide("openai", …)` are called on their
packages, which stay packages.

Found 2026-10-09 on Python 3.10, the first time 3.3.0 ran there: with a
function named `fetch` in `yait_aichain.tools`, the `fetch` package was hidden
behind it, and every path that resolves through the attribute failed —
`mock.patch("yait_aichain.tools.fetch.local.…")` on 3.10 and 3.11 (3.12 changed
how mock resolves a path, so the 3.14 suite passed), `import
yait_aichain.tools.search.exa as m` and attribute access on any version. A
user's own tests patching a tool would have broken on the two oldest Pythons
the library supports. The packages are callable instead; this file holds every
form, and CI runs it on 3.10 to 3.14.
"""

import importlib
import inspect
import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yait_aichain
from yait_aichain.tools import decide, fetch, search

QUESTIONS = {"urgent": {"type": "predicate", "instructions": "Is this urgent?"}}


class TestTheFactoryIsThePackage(unittest.TestCase):

    def test_each_is_its_package_and_callable(self):
        for f, name in ((search, "search"), (fetch, "fetch"), (decide, "decision")):
            with self.subTest(name):
                self.assertIsInstance(f, types.ModuleType)
                self.assertEqual(f.__name__, f"yait_aichain.tools.{name}")
                self.assertTrue(callable(f))

    def test_calling_it_makes_the_tool(self):
        self.assertEqual(type(search("exa", api_key="k")).__name__, "searchExa")
        self.assertEqual(type(fetch("local")).__name__, "fetchLocal")
        self.assertEqual(type(decide("openai", questions=QUESTIONS, api_key="k")).__name__, "decisionOpenAI")

    def test_the_signature_is_the_factorys(self):
        self.assertIn("service", inspect.signature(search).parameters)
        self.assertIn("questions", inspect.signature(decide).parameters)

    def test_every_import_form_resolves(self):
        import yait_aichain.tools.search.exa as exa_module
        import yait_aichain.tools.fetch.local as local_module
        self.assertTrue(hasattr(exa_module, "searchExa"))
        self.assertTrue(hasattr(local_module, "fetchLocal"))
        self.assertIs(yait_aichain.tools.search.exa, exa_module)          # attribute access
        from yait_aichain.tools.decision.openai import decisionOpenAI   # from-import
        self.assertTrue(decisionOpenAI)
        # what Chain.load does with a saved class path
        self.assertTrue(importlib.import_module("yait_aichain.tools.search.brave").searchBrave)

    def test_mock_patch_reaches_inside(self):
        with patch("yait_aichain.tools.fetch.local.assert_safe_url") as p:
            self.assertIs(sys.modules["yait_aichain.tools.fetch.local"].assert_safe_url, p)
        with patch("yait_aichain.tools.search.exa.json") as p:
            self.assertIs(sys.modules["yait_aichain.tools.search.exa"].json, p)


if __name__ == "__main__":
    unittest.main()
