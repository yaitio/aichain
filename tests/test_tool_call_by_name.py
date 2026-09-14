"""
A tool with its own parameters is called by name and still never raises.

Found by a Claude agent on held-out set 2: `Domain()(email="a@b.c")` raised a
bare `TypeError`, because `Tool.__call__` took only `input=` and `options=`,
while an Agent and a Chain call `run(**kwargs)` and `MCPTool` already forwarded
keywords. The only safe call left was `.run()` inside a hand-written try.
"""

import unittest

from yait_aichain.tools import Tool


class Domain(Tool):
    name = "email_domain"
    description = "Return the domain of an email address."
    parameters = {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}

    def run(self, email, options=None):
        if "@" not in email:
            raise ValueError("invalid email")
        return email.split("@", 1)[1]


class Echo(Tool):
    name = "echo"
    description = "Echo."
    parameters = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}

    def run(self, input, options=None):
        return input


class TestCallByName(unittest.TestCase):

    def test_a_named_parameter_is_passed_to_run(self):
        result = Domain()(email="ana@example.com")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "example.com")

    def test_an_error_comes_back_as_a_result(self):
        result = Domain()(email="not-an-email")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "invalid email")

    def test_a_missing_required_parameter_is_named(self):
        result = Domain()(other="x")
        self.assertFalse(result.success)
        self.assertIn("'email'", result.error)

    def test_input_and_options_still_work(self):
        self.assertEqual(Echo()(input="hi").output, "hi")
        self.assertFalse(Echo()().success)


if __name__ == "__main__":
    unittest.main()
