"""
tests.state.test_chain_resume
=============================

Chain ↔ state integration: suspend, persist, resume, idempotency. Pure — the
steps are local tools, no network.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chain import Chain
from tools._base import Tool
from state import Suspend, SuspendedResult, FileStore


class _Amount(Tool):
    name = "amount"
    parameters = {"type": "object", "properties": {}, "required": []}
    def run(self, **kw):
        return {"amount": 500}


class _Approve(Tool):
    name = "approve"
    parameters = {"type": "object", "properties": {}, "required": []}
    def run(self, _signal=None, **kw):
        if _signal is None:
            raise Suspend("Approve refund?", {"approved": "bool"})
        return {"approved": bool(_signal.get("approved"))}


class _Reply(Tool):
    name = "reply"
    parameters = {"type": "object", "properties": {}, "required": []}
    def run(self, **kw):
        return "processed"


def _chain(**kw):
    return Chain([(_Amount(), "amt"), (_Approve(), "appr"), (_Reply(), "reply")], **kw)


class TestChainSuspend(unittest.TestCase):

    def test_run_suspends(self):
        c = _chain()
        res = c.run(variables={"request": "Refund $500"})
        self.assertIsInstance(res, SuspendedResult)
        self.assertFalse(res)
        self.assertEqual(res.awaiting["reason"], "Approve refund?")

    def test_document_persisted_with_statuses(self):
        c = _chain()
        res = c.run()
        doc = c._store.load(res.run_id)
        self.assertEqual(doc["status"], "suspended")
        self.assertEqual([s["status"] for s in doc["steps"]],
                         ["done", "suspended", "pending"])
        self.assertEqual(doc["variables"]["amount"], 500)

    def test_no_suspend_returns_normally(self):
        # A chain without a suspend step behaves exactly as before.
        c = Chain([(_Amount(), "amt"), (_Reply(), "reply")])
        out = c.run()
        self.assertEqual(out, "processed")


class TestChainResume(unittest.TestCase):

    def test_resume_completes(self):
        c = _chain()
        res = c.run(variables={"request": "Refund $500"})
        out = c.resume(res.run_id, signal={"approved": True})
        self.assertEqual(out, "processed")

    def test_done_steps_not_rerun(self):
        # _Amount must run exactly once across run()+resume().
        calls = {"amount": 0}
        class _CountingAmount(_Amount):
            def run(self, **kw):
                calls["amount"] += 1
                return {"amount": 500}
        c = Chain([(_CountingAmount(), "amt"), (_Approve(), "appr"), (_Reply(), "reply")])
        res = c.run()
        c.resume(res.run_id, signal={"approved": True})
        self.assertEqual(calls["amount"], 1)

    def test_store_cleaned_after_completion(self):
        c = _chain()
        res = c.run()
        c.resume(res.run_id, signal={"approved": True})
        self.assertIsNone(c._store.load(res.run_id))

    def test_resume_idempotent_unknown_raises(self):
        c = _chain()
        res = c.run()
        c.resume(res.run_id, signal={"approved": True})
        with self.assertRaises(KeyError):
            c.resume(res.run_id, signal={"approved": True})

    def test_resume_across_filestore(self):
        # Simulate two processes: suspend with one Chain, resume with another,
        # sharing only a FileStore directory.
        with tempfile.TemporaryDirectory() as d:
            res = _chain(store=FileStore(d)).run(variables={"request": "x"})
            self.assertIsInstance(res, SuspendedResult)
            out = _chain(store=FileStore(d)).resume(res.run_id, signal={"approved": True})
            self.assertEqual(out, "processed")


if __name__ == "__main__":
    unittest.main()
