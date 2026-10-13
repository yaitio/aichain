"""
A chain is checkpointed after every step, and resumes only its own runs.

The documentation promised a write after each step; `save` ran only on a
pause, so a process that died in the tail was resumed from the start, every
side effect repeated. A failure skipped before a pause was forgotten after
it (`success=True`). And `resume` took any document: an agent's run, or a
chain whose steps had changed since — the signal reaching another step.
"""

import unittest

from yait_aichain import Chain
from yait_aichain.state import InMemoryStore, StepStatus

calls: list = []


def first(**_):
    calls.append("first")
    return "a"


def dies(**_):
    calls.append("dies")
    raise KeyboardInterrupt          # the process ending, not a step failing


def second(**_):
    calls.append("second")
    return "b"


def broken(**_):
    raise ValueError("bad row")


class TestACrashResumesWhereItDied(unittest.TestCase):

    def test_the_done_steps_are_not_run_again(self):
        calls.clear()
        store = InMemoryStore()
        chain = Chain(steps=[(first, "x"), (dies, "y")], store=store)
        with self.assertRaises(KeyboardInterrupt):
            chain.run()
        (run_id,) = store._runs
        self.assertEqual([s["status"] for s in store.load(run_id)["steps"]][0], StepStatus.DONE)
        calls.clear()
        with self.assertRaises(KeyboardInterrupt):
            chain.resume(run_id)
        self.assertEqual(calls, ["dies"])              # first was not repeated

    def test_a_finished_run_leaves_nothing(self):
        store = InMemoryStore()
        Chain(steps=[(first, "x")], store=store).run()
        self.assertEqual(store._runs, {})


class TestAFailureBeforeAPauseIsRemembered(unittest.TestCase):

    def test_skipped_then_paused_then_resumed_is_not_a_success(self):
        from yait_aichain.tools import Wait
        store = InMemoryStore()
        chain = Chain(steps=[(broken, "x"), Wait(reason="hold")], store=store, on_step_error="skip")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parked = chain.run()
        result = chain.resume(parked.run_id, signal={"ok": True})
        self.assertFalse(result.success)
        self.assertIn("bad row", result.error)


class TestOnlyItsOwnRuns(unittest.TestCase):

    def test_an_agents_run_is_refused_and_kept(self):
        store = InMemoryStore()
        store.save("agent-run", {"run_id": "agent-run", "kind": "agent",
                                 "steps": [{"id": 0, "name": "agent", "status": "suspended"}]})
        with self.assertRaisesRegex(ValueError, "agent run"):
            Chain(steps=[(first, "x")], store=store).resume("agent-run")
        self.assertIsNotNone(store.load("agent-run"))

    def test_a_chain_whose_steps_changed_is_refused_and_kept(self):
        from yait_aichain.tools import Wait
        store = InMemoryStore()
        parked = Chain(steps=[Wait(reason="hold"), (first, "x")], store=store).run()
        with self.assertRaisesRegex(ValueError, "not the ones it parked with"):
            Chain(steps=[Wait(reason="hold"), (second, "x")], store=store).resume(parked.run_id)
        self.assertIsNotNone(store.load(parked.run_id))


if __name__ == "__main__":
    unittest.main()
