"""
tests.state.test_state
======================

Unit tests for the M2 ``state/`` foundation: RunDocument, the stores,
RunContext, and the Suspend primitives. Pure — no network.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.state import (
    RunContext, Suspend, SuspendedResult,
    InMemoryStore, FileStore, RunDocument, StepStatus,
)


class TestRunDocument(unittest.TestCase):

    def _doc(self):
        return RunDocument.new("chain", ["a", "b", "c"],
                               variables={"x": 1})

    def test_new_all_pending(self):
        d = self._doc()
        self.assertEqual(d.status, StepStatus.RUNNING)
        self.assertEqual(d.first_pending(), 0)
        self.assertEqual(d.variables, {"x": 1})
        self.assertTrue(d.run_id.startswith("run-"))

    def test_cursor_advances_with_done(self):
        d = self._doc()
        d.steps[0]["status"] = StepStatus.DONE
        self.assertEqual(d.first_pending(), 1)
        d.steps[1]["status"] = StepStatus.DONE
        d.steps[2]["status"] = StepStatus.DONE
        self.assertIsNone(d.first_pending())
        self.assertEqual(d.status, StepStatus.DONE)

    def test_status_suspended(self):
        d = self._doc()
        d.steps[0]["status"] = StepStatus.DONE
        d.steps[1]["status"] = StepStatus.SUSPENDED
        d.steps[1]["suspend"] = {"reason": "r", "resume_with": {"ok": "bool"}}
        self.assertEqual(d.status, StepStatus.SUSPENDED)
        self.assertEqual(d.first_pending(), 1)        # resume re-runs the suspended step
        self.assertEqual(d.suspended_step()["name"], "b")

    def test_status_failed_wins(self):
        d = self._doc()
        d.steps[0]["status"] = StepStatus.SUSPENDED
        d.steps[1]["status"] = StepStatus.FAILED
        self.assertEqual(d.status, StepStatus.FAILED)

    def test_round_trip(self):
        d = self._doc()
        d.steps[0]["status"] = StepStatus.DONE
        d.variables["y"] = 2
        d.usage = {"input_tokens": 10}
        again = RunDocument.from_dict(d.to_dict())
        self.assertEqual(again.variables, {"x": 1, "y": 2})
        self.assertEqual(again.status, d.status)
        self.assertEqual(again.usage["input_tokens"], 10)
        self.assertEqual(again.to_dict()["status"], d.status)  # derived field present


class TestInMemoryStore(unittest.TestCase):

    def test_save_load_delete(self):
        s = InMemoryStore()
        s.save("r1", {"variables": {"a": 1}})
        self.assertEqual(s.load("r1")["variables"]["a"], 1)
        s.delete("r1")
        self.assertIsNone(s.load("r1"))

    def test_load_missing_is_none(self):
        self.assertIsNone(InMemoryStore().load("nope"))

    def test_stored_copy_is_isolated(self):
        s = InMemoryStore()
        doc = {"variables": {"a": 1}}
        s.save("r1", doc)
        doc["variables"]["a"] = 99           # mutate caller's copy
        self.assertEqual(s.load("r1")["variables"]["a"], 1)


class TestFileStore(unittest.TestCase):

    def test_round_trip_and_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            s = FileStore(d)
            s.save("r-1", {"status": "suspended", "variables": {"a": 1}})
            # a fresh store over the same dir sees it (survives "restart")
            s2 = FileStore(d)
            self.assertEqual(s2.load("r-1")["status"], "suspended")
            s2.delete("r-1")
            self.assertIsNone(s2.load("r-1"))

    def test_load_missing_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(FileStore(d).load("nope"))


class TestContextAndSuspend(unittest.TestCase):

    def test_run_context(self):
        ctx = RunContext(tenant="acme", metadata={"user_id": "u-42"})
        self.assertEqual(ctx.tenant, "acme")
        self.assertEqual(ctx.get("user_id"), "u-42")
        self.assertIsNone(ctx.get("missing"))

    def test_suspend_carries_fields(self):
        exc = Suspend("approve?", {"approved": "bool"})
        self.assertEqual(exc.reason, "approve?")
        self.assertEqual(exc.resume_with, {"approved": "bool"})

    def test_suspended_result_is_falsy(self):
        r = SuspendedResult(run_id="r1", awaiting={"reason": "x"}, document={})
        self.assertFalse(r)
        self.assertIn("r1", repr(r))


if __name__ == "__main__":
    unittest.main()
