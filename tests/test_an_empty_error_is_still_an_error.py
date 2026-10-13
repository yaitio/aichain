"""
A step that fails without a message has still failed.

Success was decided by whether the error's text was truthy, and
`TimeoutError()`, a bare `assert` and `RuntimeError()` render as `""`: under
`stop`, `skip` or `collect` a Chain returned `success=True, error=None` beside
the record of the failure, and a Pool's item read as succeeded.
"""

import unittest

from yait_aichain import Chain, Pool


def _times_out(**_):
    raise TimeoutError()


class TestSilentFailures(unittest.TestCase):

    def test_a_chain_step(self):
        for policy in ("stop", "skip", "collect"):
            with self.subTest(policy=policy):
                result = Chain(steps=[(_times_out, "x")], on_step_error=policy).run()
                self.assertFalse(result.success)
                self.assertIn("TimeoutError", result.error)

    def test_a_pool_item(self):
        from yait_aichain.tools import tool

        @tool(risk="read")
        def waits(x: str = "") -> str:
            """Times out."""
            raise TimeoutError()
        result = Pool(waits, [{"x": "a"}], on_error="collect").run()
        self.assertEqual(result.errors, ["TimeoutError"])


if __name__ == "__main__":
    unittest.main()
