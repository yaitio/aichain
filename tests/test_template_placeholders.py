"""
A placeholder is filled wherever it stands, and an unfilled one is said.

``{"summary": {text}}`` read as an escaped brace and left the document out;
``{θέμα}`` was not a placeholder; a typo sent ``{topic}`` to the model as
written while a Chain raised on the same mistake; a dict went as its Python
repr.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import _template
from yait_aichain._template import placeholder_names, substitute_placeholders


class TestPlaceholders(unittest.TestCase):

    def test_a_placeholder_beside_a_json_brace_is_filled(self):
        self.assertEqual(substitute_placeholders('{"summary": {text}}', {"text": "doc"}),
                         '{"summary": doc}')

    def test_doubled_braces_stay_literal(self):
        self.assertEqual(substitute_placeholders("{{text}}", {"text": "doc"}), "{{text}}")

    def test_a_non_latin_name_is_a_placeholder(self):
        self.assertEqual(substitute_placeholders("Ο {θέμα}", {"θέμα": "x"}), "Ο x")
        self.assertEqual(placeholder_names("{a} {{b}} {β}"), ["a", "β"])

    def test_a_dict_goes_as_json(self):
        self.assertEqual(substitute_placeholders("{d}", {"d": {"ok": True, "n": None}}),
                         '{"ok": true, "n": null}')

    def test_an_unfilled_placeholder_is_said_once_with_the_nearest_name(self):
        _template._warned.discard("topik")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                substitute_placeholders("About {topik}", {"topic": "x"}, warn=True)
        said = [str(w.message) for w in caught if "topik" in str(w.message)]
        self.assertEqual(len(said), 1)
        self.assertIn("{topic}", said[0])


if __name__ == "__main__":
    unittest.main()
