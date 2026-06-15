"""
tests.state.test_wait_tools
===========================

The suspend tools: Wait (leaf) and Gate (wrapper), driven through a Chain.
Pure — local tools, no network.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chain import Chain
from tools import Wait, Gate
from tools._base import Tool
from state import SuspendedResult


class _Amount(Tool):
    name = "amount"
    parameters = {"type": "object", "properties": {}, "required": []}
    def run(self, **kw):
        return {"amount": 500}


class TestWait(unittest.TestCase):

    def test_pauses_and_carries_hint(self):
        c = Chain([(_Amount(), "amt"),
                   Wait(reason="approve?", resume_with={"approved": "bool"},
                        hint={"on": "webhook"})])
        res = c.run()
        self.assertIsInstance(res, SuspendedResult)
        self.assertEqual(res.awaiting["reason"], "approve?")
        self.assertEqual(res.awaiting["resume_with"], {"approved": "bool"})
        self.assertEqual(res.awaiting["hint"], {"on": "webhook"})

    def test_signal_becomes_output(self):
        c = Chain([(_Amount(), "amt"), (Wait(), "decision")])
        res = c.run()
        out = c.resume(res.run_id, signal={"approved": True})
        # the signal is the leaf step's output → merged into variables
        self.assertEqual(out, {"approved": True})


class TestGate(unittest.TestCase):

    def _tool(self, ran):
        class _Send(Tool):
            name = "send"
            parameters = {"type": "object", "properties": {}, "required": []}
            def run(self, **kw):
                ran["n"] += 1
                return "sent"
        return _Send()

    def test_approved_runs_wrapped_tool(self):
        ran = {"n": 0}
        c = Chain([(_Amount(), "amt"), (Gate(self._tool(ran)), "result")])
        res = c.run()
        self.assertEqual(res.awaiting["reason"], "Approve running 'send'?")
        out = c.resume(res.run_id, signal={"approved": True})
        self.assertEqual(out, "sent")
        self.assertEqual(ran["n"], 1)

    def test_denied_skips_wrapped_tool(self):
        ran = {"n": 0}
        c = Chain([(_Amount(), "amt"), (Gate(self._tool(ran)), "result")])
        res = c.run()
        out = c.resume(res.run_id, signal={"approved": False})
        self.assertEqual(ran["n"], 0)
        self.assertTrue(out["skipped"])

    def test_custom_decision_key(self):
        ran = {"n": 0}
        gate = Gate(self._tool(ran), decision_key="ok", reason="ship it?")
        c = Chain([(_Amount(), "amt"), (gate, "result")])
        res = c.run()
        self.assertEqual(res.awaiting["resume_with"], {"ok": "bool"})
        c.resume(res.run_id, signal={"ok": True})
        self.assertEqual(ran["n"], 1)


if __name__ == "__main__":
    unittest.main()
