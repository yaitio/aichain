"""
A schema keeps its meaning on the way to each provider.

In OpenAI strict mode an optional `$ref`, `anyOf` or enum became required and
not nullable, so the model had to invent a value, and a free object was a
400. Google kept `additionalProperties` inside `anyOf` and `$defs` (a 400),
read no `$ref`, and turned a union into its first type.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import SchemaNotExpressible, portable_schema


class TestOpenAIStrict(unittest.TestCase):

    def test_an_optional_ref_is_it_or_null(self):
        s = {"type": "object", "$defs": {"A": {"type": "object", "properties": {"x": {"type": "string"}}}},
             "properties": {"a": {"$ref": "#/$defs/A"}, "b": {"type": "string"}}, "required": ["b"]}
        out = portable_schema(s, "openai")
        self.assertEqual(out["properties"]["a"], {"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]})
        self.assertEqual(out["required"], ["a", "b"])

    def test_an_optional_enum_may_be_null(self):
        s = {"type": "object", "properties": {"c": {"type": "string", "enum": ["x", "y"]}}}
        out = portable_schema(s, "openai")
        self.assertIn(None, out["properties"]["c"]["enum"])

    def test_a_free_object_is_refused_by_name(self):
        with self.assertRaises(SchemaNotExpressible):
            portable_schema({"type": "object", "properties": {"meta": {"type": "object"}},
                             "required": ["meta"]}, "openai")


class TestGoogle(unittest.TestCase):

    def _g(self, s):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return portable_schema(s, "google")

    def test_additional_properties_inside_any_of_is_removed(self):
        out = self._g({"anyOf": [{"type": "object", "properties": {"a": {"type": "string"}},
                                  "additionalProperties": False}, {"type": "string"}]})
        self.assertNotIn("additionalProperties", str(out))

    def test_a_local_ref_is_inlined(self):
        out = self._g({"type": "object", "$defs": {"A": {"type": "string"}},
                       "properties": {"a": {"$ref": "#/$defs/A"}}})
        self.assertEqual(out["properties"]["a"], {"type": "string"})
        self.assertNotIn("$defs", out)

    def test_a_union_is_an_any_of(self):
        out = self._g({"type": ["string", "integer", "null"]})
        self.assertEqual(out["anyOf"], [{"type": "string"}, {"type": "integer"}])
        self.assertTrue(out["nullable"])


if __name__ == "__main__":
    unittest.main()
