"""
``tool(input, options, **kwargs)`` reaches ``run`` the way ``run`` declares.

The safe call passed ``(input, options)`` positionally, so options landed in
whatever ``run`` named second, a function tool with no parameters could not
be called, an options dict reached a numeric parameter and ran, and an error
was its message without its type.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain.tools import tool
from yait_aichain.tools._base import Tool


class Pictures(Tool):
    name = "pictures"
    parameters = {"type": "object", "properties": {"input": {"type": "string"}},
                  "required": ["input"]}

    def run(self, input, images=None, options=None):
        return {"input": input, "images": images, "options": options}


class TestCallBinds(unittest.TestCase):

    def test_options_reach_options_not_the_second_parameter(self):
        out = Pictures()("a cat", {"size": "1024x1024"}).output
        self.assertEqual(out, {"input": "a cat", "images": None, "options": {"size": "1024x1024"}})

    def test_a_function_tool_with_no_parameters_runs(self):
        @tool
        def now() -> str:
            """The time."""
            return "noon"
        self.assertEqual(now().output, "noon")

    def test_options_to_a_function_without_them_are_refused(self):
        @tool
        def wc(input: str, unique: bool = False) -> int:
            """Count words.

            Parameters
            ----------
            input : str
                Text.
            unique : bool
                Distinct words only.
            """
            words = input.split()
            return len(set(words)) if unique else len(words)
        result = wc("a a b", {"unique": True})
        self.assertFalse(result.success)
        self.assertIn("unique=", result.error)
        self.assertEqual(wc("a a b", unique=True).output, 2)

    def test_an_error_carries_its_type(self):
        class Missing(Tool):
            name = "missing"
            parameters = {"type": "object", "properties": {}}

            def run(self, input=None, options=None):
                raise KeyError("x")
        self.assertEqual(Missing()().error, "KeyError: 'x'")


if __name__ == "__main__":
    unittest.main()
