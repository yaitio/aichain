"""
A run that cannot go on ends with a result that says why.

A context overflow read as `llm_error`, like a revoked key; a store that
refused the parked run raised out of `run()` and the paid conversation was
lost; arguments nested 450 deep raised RecursionError on the next turn and
the model never saw why.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent                                        # noqa: E402
from yait_aichain.clients._errors import ContextWindowExceededError, error_from_status  # noqa: E402
from yait_aichain.state import InMemoryStore                                # noqa: E402
from yait_aichain.tools import Wait                                         # noqa: E402
from test_loop import Echo, _calls, _model, _text                           # noqa: E402


class TestReasons(unittest.TestCase):

    def test_a_context_overflow_is_named(self):
        self.assertIsInstance(error_from_status(400, '{"error": {"code": "context_length_exceeded"}}'),
                              ContextWindowExceededError)
        m = _model(_text("x"))
        m.client._post = MagicMock(side_effect=ContextWindowExceededError(400, "maximum context length"))
        self.assertEqual(Agent(m, verbose=0).run("go").stopped_by, "context_overflow")

    def test_a_store_that_refuses_ends_the_run_as_park_failed(self):
        store = InMemoryStore()
        store.save = MagicMock(side_effect=RuntimeError("item too large"))
        agent = Agent(_model(_calls(("c1", "wait", {}))), tools=[Wait(reason="r")], store=store, verbose=0)
        result = agent.run("go")
        self.assertEqual(result.stopped_by, "park_failed")
        self.assertIn("item too large", result.error)

    def test_deep_arguments_are_an_error_the_model_sees(self):
        deep: dict = {}
        cur = deep
        for _ in range(100):
            cur["a"] = {}
            cur = cur["a"]
        sink = []
        agent = Agent(_model(_calls(("c1", "echo", {"value": deep})), _text("ok")),
                      tools=[Echo(sink)], verbose=0)
        result = agent.run("go")
        self.assertEqual(sink, [])
        self.assertIn("nested more than", str(result.journal[0]["observation"]))


if __name__ == "__main__":
    unittest.main()
