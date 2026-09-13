"""
Pool's contract, beyond fan-out: what it records, what it shares, what it stops.

`Pool` is one of the two primitives everything else composes, and its own test
file held eleven tests. What it promises is spread across other files — error
words in `test_error_policy_is_one_vocabulary`, money in `test_max_cost`,
blockers in `test_swarm_coordination` — and each of those checks one property.
This file holds the rest of the surface in one place: history and status,
usage per item, agent runners, context crossing into worker threads, beacons,
a shared budget, and what `stop` actually stops.
"""

import os
import sys
import threading
import time
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Budget                                      # noqa: E402
from yait_aichain.agent import beacon                                # noqa: E402
from yait_aichain.models._usage import Usage                         # noqa: E402
from yait_aichain.pool import DONE, FAILED, PENDING, RUNNING, Pool   # noqa: E402
from yait_aichain.state import RunContext, current, using            # noqa: E402
from yait_aichain.tools._base import Tool                            # noqa: E402


class Echo(Tool):
    name = "echo"
    parameters = {"type": "object", "properties": {"value": {"type": "string"},
                                                   "suffix": {"type": "string"}}}

    def run(self, value=None, suffix="", options=None):
        return f"{value}{suffix}"


class FailOn(Tool):
    """Raises for one value; sleeps a little otherwise so ordering is observable."""
    name = "fail_on"
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}}

    def __init__(self, bad, pause=0.0):
        self.bad, self.pause = bad, pause

    def run(self, value=None, options=None):
        if value == self.bad:
            raise KeyError("tenant")
        time.sleep(self.pause)
        return value


class ReadsTenant(Tool):
    name = "reads_tenant"
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}}

    def run(self, value=None, options=None):
        ctx = current()
        return (ctx.tenant if ctx else None, threading.current_thread().name)


class Signals(Tool):
    name = "signals"
    parameters = {"type": "object", "properties": {"value": {"type": "string"},
                                                   "kind": {"type": "string"}}}

    def run(self, value=None, kind="fact", options=None):
        beacon(kind, f"about {value}", source="signals")
        return value


class Counts:
    """A Skill-shaped runner: `run(variables=…)` and a `last_usage` it overwrites."""
    name = "counts"

    def __init__(self):
        self.last_usage = None
        self.max_cost = None
        self.seen_budgets = []

    def run(self, variables=None):
        self.seen_budgets.append(self.max_cost)
        n = len(variables["value"])
        self.last_usage = Usage(input_tokens=n, output_tokens=n, total_tokens=2 * n)
        return variables["value"].upper()


class _Result:
    def __init__(self, ok, output=None, tokens=0, cost=None, error=None):
        self.ok, self.output, self.tokens_used, self.cost, self.error = ok, output, tokens, cost, error

    def __bool__(self):
        return self.ok


class Agent:
    """Detected as an agent by class name, as the real one is."""
    name = "fake-agent"

    def __init__(self, ok=True):
        self.ok = ok
        self.max_cost = None

    def run(self, task, variables=None):
        return _Result(self.ok, output=f"did {task}", tokens=7, error="gave up")


def items(*values):
    return [{"value": v} for v in values]


class TestConstruction(unittest.TestCase):

    def test_repr_names_the_runner_and_the_settings(self):
        r = repr(Pool(Echo(), items("a", "b"), max_flows=3, on_error="skip"))
        for part in ("'echo'", "items=2", "max_flows=3", "on_error='skip'"):
            self.assertIn(part, r)

    def test_name_and_description_are_kept(self):
        p = Pool(Echo(), items("a"), name="fan", description="echoes")
        self.assertEqual((p.name, p.description), ("fan", "echoes"))

    def test_a_number_becomes_a_budget(self):
        self.assertIsInstance(Pool(Echo(), items("a"), max_cost=0.5).max_cost, Budget)

    def test_a_budget_object_is_kept_not_copied(self):
        b = Budget(1.0)
        self.assertIs(Pool(Echo(), items("a"), max_cost=b).max_cost, b)

    def test_every_policy_word_is_accepted(self):
        for word in ("raise", "stop", "skip", "collect"):
            Pool(Echo(), items("a"), on_error=word)


class TestVariablesAndOrder(unittest.TestCase):

    def test_shared_variables_reach_every_item(self):
        self.assertEqual(Pool(Echo(), items("a", "b")).run({"suffix": "!"}), ["a!", "b!"])

    def test_the_item_wins_over_a_shared_variable(self):
        out = Pool(Echo(), [{"value": "a", "suffix": "?"}]).run({"suffix": "!"})
        self.assertEqual(out, ["a?"])

    def test_one_worker_still_returns_in_item_order(self):
        self.assertEqual(Pool(Echo(), items(*"abcde"), max_flows=1).run(), list("abcde"))


class TestHistoryAndStatus(unittest.TestCase):

    def test_before_a_run_everything_is_pending(self):
        p = Pool(Echo(), items("a", "b"))
        self.assertEqual(p.status, {PENDING: 2, RUNNING: 0, DONE: 0, FAILED: 0})

    def test_a_record_carries_what_a_reader_needs(self):
        p = Pool(Echo(), items("a"))
        p.run({"suffix": "!"})
        rec = p.history[0]
        self.assertEqual(rec["index"], 0)
        self.assertEqual(rec["status"], DONE)
        self.assertEqual(rec["output"], "a!")
        self.assertEqual(rec["variables"], {"suffix": "!", "value": "a"})
        self.assertIsNone(rec["error"])
        self.assertGreaterEqual(rec["duration"], 0)

    def test_status_counts_failures(self):
        p = Pool(FailOn("b"), items("a", "b", "c"))
        p.run()
        self.assertEqual(p.status[DONE], 2)
        self.assertEqual(p.status[FAILED], 1)

    def test_history_is_a_copy(self):
        p = Pool(Echo(), items("a"))
        p.run()
        p.history[0]["output"] = "tampered"
        self.assertEqual(p.history[0]["output"], "a")

    def test_a_second_run_starts_a_fresh_history(self):
        p = Pool(FailOn("a"), items("a"))
        p.run()
        p._runner = Echo()
        p.run()
        self.assertEqual(p.history[0]["status"], DONE)
        self.assertIsNone(p.history[0]["error"])


class TestErrorPolicies(unittest.TestCase):

    def test_collect_records_type_and_traceback_not_just_the_message(self):
        p = Pool(FailOn("b"), items("a", "b"))
        self.assertEqual(p.run(), ["a", None])
        failure = p.history[1]["failure"]
        self.assertEqual(failure["type"], "KeyError")
        self.assertIn("Traceback", failure["traceback"])

    def test_skip_names_the_item_in_the_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Pool(FailOn("b"), items("a", "b"), on_error="skip").run()
        self.assertTrue(any("Pool item 1" in str(w.message) for w in caught))

    def test_collect_is_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Pool(FailOn("b"), items("a", "b"), on_error="collect").run()
        self.assertFalse([w for w in caught if "Pool item" in str(w.message)])

    def test_raise_propagates_the_original_type(self):
        with self.assertRaises(KeyError):
            Pool(FailOn("a"), items("a", "b"), on_error="raise").run()

    def test_stop_starts_no_more_items_and_does_not_raise(self):
        values = ["bad"] + [f"v{i}" for i in range(49)]
        p = Pool(FailOn("bad", pause=0.02), items(*values), max_flows=1, on_error="stop")
        out = p.run()
        self.assertIsNone(out[0])
        # Items already running finish; the rest are never begun and stay
        # PENDING — the honest status for work nobody started.
        self.assertGreaterEqual(p.status[PENDING], 40)
        self.assertTrue(all(o is None for o, r in zip(out, p.history) if r["status"] == PENDING))


class TestUsage(unittest.TestCase):

    def test_a_tool_reports_no_usage_so_the_sum_is_none_not_zero(self):
        p = Pool(Echo(), items("a"))
        p.run()
        self.assertIsNone(p.usage)

    def test_usage_is_per_item_even_though_the_runner_is_shared(self):
        p = Pool(Counts(), items("a", "bbb"), max_flows=2)
        self.assertEqual(p.run(), ["A", "BBB"])
        self.assertEqual([r["usage"].total_tokens for r in p.history], [2, 6])
        self.assertEqual(p.usage.total_tokens, 8)

    def test_the_budget_travels_by_reference_to_every_item(self):
        runner, budget = Counts(), Budget(1.0)
        p = Pool(runner, items("a", "b", "c"), max_cost=budget)
        p.run()
        # Each item runs on a shallow copy; the list is shared, so it saw all three.
        self.assertEqual(len(runner.seen_budgets), 3)
        self.assertTrue(all(b is budget for b in runner.seen_budgets))


class TestAgentRunner(unittest.TestCase):

    def test_the_task_goes_in_and_the_output_comes_out(self):
        p = Pool(Agent(), [{"task": "x"}, {"task": "y"}])
        self.assertEqual(p.run(), ["did x", "did y"])
        self.assertEqual(p.usage.total_tokens, 14)

    def test_an_item_without_a_task_fails_and_says_why(self):
        p = Pool(Agent(), [{"value": "no task"}])
        self.assertEqual(p.run(), [None])
        self.assertIn("task", p.history[0]["error"])

    def test_a_failed_agent_result_is_a_failure_not_an_output(self):
        p = Pool(Agent(ok=False), [{"task": "x"}])
        self.assertEqual(p.run(), [None])
        self.assertEqual(p.history[0]["failure"]["type"], "RuntimeError")
        self.assertIn("gave up", p.history[0]["error"])

    def test_the_pool_budget_is_handed_to_the_agent(self):
        agent, budget = Agent(), Budget(2.0)
        Pool(agent, [{"task": "x"}], max_cost=budget).run()
        self.assertIs(agent.max_cost, budget)


class TestContextCrossesIntoWorkers(unittest.TestCase):

    def test_every_worker_sees_the_callers_run_context(self):
        p = Pool(ReadsTenant(), items(*"abcdef"), max_flows=3)
        with using(RunContext(tenant="acme")):
            out = p.run()
        self.assertEqual({tenant for tenant, _ in out}, {"acme"})
        self.assertGreater(len({thread for _, thread in out}), 0)

    def test_outside_a_run_there_is_no_context(self):
        out = Pool(ReadsTenant(), items("a")).run()
        self.assertIsNone(out[0][0])


class TestBeacons(unittest.TestCase):

    def test_a_coordination_beacon_is_collected_and_does_not_stop_the_fan_out(self):
        p = Pool(Signals(), items("a", "b", "c"))
        self.assertEqual(p.run({"kind": "fact"}), ["a", "b", "c"])
        self.assertEqual(len(p.beacons), 3)
        self.assertEqual({b.kind for b in p.beacons}, {"fact"})

    def test_beacons_belong_to_the_last_run(self):
        p = Pool(Signals(), items("a"))
        p.run({"kind": "fact"})
        p._runner = Echo()
        p.run()
        self.assertEqual(p.beacons, [])


if __name__ == "__main__":
    unittest.main()
