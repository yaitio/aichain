"""
What mutation testing found no test guarding in Chain and Pool, 2026-09-13.

Each test here exists because a one-token change to the library survived the
whole suite: a key renamed in the usage a parked run saves, the saved context
ignored on resume, the step kind dropped from an event, `task_key` ignored.
A test that fails when its line breaks is the point; one that passes either
way is noise.
"""

import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Chain, Pool, Tracer                          # noqa: E402
from yait_aichain.models._usage import Usage                          # noqa: E402
from yait_aichain.pool import RUNNING                                 # noqa: E402
from yait_aichain.state import FileStore, RunContext, SuspendedResult, current  # noqa: E402
from yait_aichain.tools import Wait                                   # noqa: E402
from yait_aichain.tools._base import Tool                             # noqa: E402


class Spends:
    """A Skill-shaped step that reports a fixed usage."""

    def __init__(self, name, tokens):
        self.name, self.tokens, self.last_usage = name, tokens, None

    def run(self, variables=None):
        self.last_usage = Usage(input_tokens=self.tokens, output_tokens=self.tokens,
                                total_tokens=2 * self.tokens)
        return f"{self.name} done"


class Tenant(Tool):
    name = "tenant"
    parameters = {"type": "object", "properties": {}}

    def run(self, options=None):
        ctx = current()
        return ctx.tenant if ctx else None


def _paused(store, steps_after, **run_kw):
    chain = Chain(steps=[(Spends("before", 5), "a"), (Wait(reason="ok?"), "approval")] + steps_after,
                  store=store)
    result = chain.run(**run_kw)
    assert isinstance(result, SuspendedResult)
    return result


class TestResume(unittest.TestCase):

    def setUp(self):
        self.store = FileStore(tempfile.mkdtemp())

    def test_usage_before_the_pause_is_counted_after_it(self):
        paused = _paused(self.store, [(Spends("after", 7), "b")])
        fresh = Chain(steps=[(Spends("before", 5), "a"), (Wait(reason="ok?"), "approval"),
                             (Spends("after", 7), "b")], store=self.store)
        fresh.resume(paused.run_id, signal={"approved": True})
        self.assertEqual(fresh.last_usage.input_tokens, 12)
        self.assertEqual(fresh.last_usage.output_tokens, 12)
        self.assertEqual(fresh.last_usage.total_tokens, 24)

    def test_the_saved_context_comes_back_in_another_process(self):
        paused = _paused(self.store, [(Tenant(), "who")], context=RunContext(tenant="acme"))
        fresh = Chain(steps=[(Spends("before", 5), "a"), (Wait(reason="ok?"), "approval"),
                             (Tenant(), "who")], store=self.store)
        self.assertEqual(fresh.resume(paused.run_id, signal={"approved": True}), "acme")

    def test_a_context_passed_to_resume_replaces_the_saved_one(self):
        paused = _paused(self.store, [(Tenant(), "who")], context=RunContext(tenant="acme"))
        fresh = Chain(steps=[(Spends("before", 5), "a"), (Wait(reason="ok?"), "approval"),
                             (Tenant(), "who")], store=self.store)
        out = fresh.resume(paused.run_id, signal={"approved": True}, context=RunContext(tenant="beta"))
        self.assertEqual(out, "beta")

    def test_no_saved_context_means_none(self):
        paused = _paused(self.store, [(Tenant(), "who")])
        self.assertIsNone(Chain(steps=[(Spends("before", 5), "a"), (Wait(reason="ok?"), "approval"),
                                       (Tenant(), "who")], store=self.store)
                          .resume(paused.run_id, signal={"approved": True}))

    def test_a_finished_run_cannot_be_resumed_twice(self):
        paused = _paused(self.store, [(Spends("after", 7), "b")])
        chain = Chain(steps=[(Spends("before", 5), "a"), (Wait(reason="ok?"), "approval"),
                             (Spends("after", 7), "b")], store=self.store)
        chain.resume(paused.run_id, signal={"approved": True})
        with self.assertRaises(KeyError):
            chain.resume(paused.run_id, signal={"approved": True})


class _Result:
    def __init__(self, output):
        self.output = output

    def __bool__(self):
        return True


class Agent:
    """Detected as an agent by class name; records what it was handed."""
    name = "fake-agent"

    def __init__(self):
        self.seen = []
        self.max_cost = "mine"

    def run(self, task, variables=None):
        self.seen.append((task, dict(variables or {})))
        return _Result(f"answered {task}")


class TestChainSteps(unittest.TestCase):

    def test_step_events_carry_the_kind(self):
        tracer = Tracer()
        Chain(steps=[(Spends("s", 1), "a"), (Tenant(), "who")], hooks=[tracer]).run()
        started = [e.payload["kind"] for e in tracer.events if e.type == "step.started"]
        self.assertEqual(started, ["skill", "tool"])

    def test_task_key_names_the_variable_an_agent_reads(self):
        agent = Agent()
        out = Chain(steps=[(agent, "answer", {}, {"task_key": "question"})]).run({"question": "why?"})
        self.assertEqual(out, "answered why?")


class Plain:
    """A Tool-less, Agent-less runner that reports no usage."""
    name = "plain"

    def run(self, variables=None):
        return "ok"


class TestPool(unittest.TestCase):

    def test_an_agent_runner_gets_the_items_variables(self):
        agent = Agent()
        Pool(agent, [{"task": "t1", "lang": "pt"}]).run({"tone": "dry"})
        self.assertEqual(agent.seen, [("t1", {"tone": "dry", "task": "t1", "lang": "pt"})])

    def test_a_pool_without_a_budget_leaves_the_runners_alone(self):
        agent = Agent()
        Pool(agent, [{"task": "t1"}]).run()
        self.assertEqual(agent.max_cost, "mine")

    def test_an_agent_result_without_counts_reports_no_usage(self):
        pool = Pool(Agent(), [{"task": "t1"}])
        pool.run()
        self.assertIsNone(pool.usage)

    def test_an_item_is_marked_running_while_it_runs(self):
        seen = []
        holder = {}

        class Peek(Tool):
            name = "peek"
            parameters = {"type": "object", "properties": {"value": {"type": "string"}}}

            def run(self, value=None, options=None):
                seen.append(holder["pool"].status[RUNNING])
                return value

        pool = Pool(Peek(), [{"value": "a"}], max_flows=1)
        holder["pool"] = pool
        pool.run()
        self.assertEqual(seen, [1])


if __name__ == "__main__":
    unittest.main()
