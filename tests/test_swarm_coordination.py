"""
Swarm coordination: what a result rests on, and a child that can speak up.

Two items from the plan, both "a field and a comparison, not an architecture".

**The evidence kind decides something now.** `_journal.py` always separated
`CHECK` — a programmatic fact — from `MODEL_CLAIM` — the model's word — and
nothing read the field. Worse than unread: `delegate` returned a worker's
report marked `MODEL_CLAIM`, and the loop wrote the call into the parent's
journal as `CHECK` regardless, so an unverified report entered the record as a
verified fact. A result now says what it rests on.

**Beacons interrupt a wait.** `Pool` fanned out with no channel back. A child
that hit a blocker could fail or finish; the rest of the fan-out went on
spending money on work the blocker had already made pointless.
"""

import os
import sys
import time
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Agent, Pool                                # noqa: E402
from yait_aichain.agent import (ATTENTION, Acceptance, beacon,      # noqa: E402
                                acceptance_from)
from yait_aichain.tools._base import Tool                           # noqa: E402
from test_loop import Echo, _calls, _model, _text                   # noqa: E402


def _entry(kind, action="a", observation="o", outcome="done"):
    return {"outcome": outcome, "action": action, "observation": observation,
            "evidence": {"kind": kind, "detail": ""}}


class TestAcceptanceSaysWhatAResultRestsOn(unittest.TestCase):

    def test_a_checked_result(self):
        self.assertEqual(acceptance_from([_entry("check")]).kind, "checked")

    def test_a_result_resting_only_on_a_claim(self):
        self.assertEqual(acceptance_from([_entry("model_claim")]).kind,
                         "claimed")

    def test_an_answer_with_nothing_behind_it(self):
        """No action at all. Returned, and visibly unsupported."""
        self.assertEqual(acceptance_from([]).kind, "unsupported")

    def test_a_failed_attempt_supports_nothing(self):
        self.assertEqual(
            acceptance_from([_entry("check", outcome="failed")]).kind,
            "unsupported")

    def test_one_check_outweighs_any_number_of_claims(self):
        journal = [_entry("model_claim"), _entry("check"), _entry("model_claim")]
        self.assertEqual(acceptance_from(journal).kind, "checked")


class TestAcceptanceCanBeInvalidated(unittest.TestCase):

    def test_the_same_evidence_still_holds(self):
        accepted = acceptance_from([_entry("check", observation="42 rows")])
        self.assertTrue(accepted.still_holds(
            [_entry("check", observation="42 rows")]))

    def test_changed_evidence_does_not(self):
        """The whole point: accepted on facts that are no longer true is not
        still accepted."""
        accepted = acceptance_from([_entry("check", observation="42 rows")])
        self.assertFalse(accepted.still_holds(
            [_entry("check", observation="41 rows")]))

    def test_a_reworded_claim_does_not_look_like_the_facts_moving(self):
        """The fingerprint covers checked evidence only. Folding claims in
        would let a model re-wording its report masquerade as a change in
        what was verified."""
        before = acceptance_from([_entry("check"),
                                  _entry("model_claim", observation="done")])
        after = [_entry("check"),
                 _entry("model_claim", observation="all finished, promise")]
        self.assertTrue(before.still_holds(after))

    def test_a_caller_can_supersede_it(self):
        accepted = acceptance_from([_entry("check")])
        overruled = accepted.supersede("the user edited the file by hand")
        self.assertFalse(overruled.holds)
        self.assertFalse(overruled.still_holds([_entry("check")]))
        self.assertTrue(accepted.holds, "supersede must not mutate the original")


class TestTheLoopRecordsTheRightKind(unittest.TestCase):

    def test_a_tool_result_is_a_check(self):
        agent = Agent(_model(_calls(("c1", "echo", {"value": "x"})),
                             _text("done")),
                      tools=[Echo()], verbose=0)
        result = agent.run("go")
        self.assertEqual(result.acceptance.kind, "checked")

    def test_an_answer_with_no_action_is_unsupported(self):
        result = Agent(_model(_text("just an answer")), verbose=0).run("go")
        self.assertEqual(result.acceptance.kind, "unsupported")

    def test_a_delegated_report_is_a_claim_not_a_check(self):
        """The defect this fixes: the call ran, so it was recorded as CHECK,
        and the worker's unverified account became a fact in the parent."""
        from yait_aichain.agent._journal import MODEL_CLAIM
        from yait_aichain.agent import Journal

        class Delegate(Tool):
            name = "report"
            parameters = {"type": "object",
                          "properties": {"value": {"type": "string"}},
                          "required": ["value"]}

            def run(self, value, options=None):
                return {"output": "all done", "evidence": MODEL_CLAIM}

        agent = Agent(_model(_calls(("c1", "report", {"value": "x"})),
                             _text("done")),
                      tools=[Delegate()], verbose=0)
        result = agent.run("go")
        kinds = [e["evidence"]["kind"] for e in result.journal
                 if e.get("evidence")]
        self.assertIn("model_claim", kinds)
        self.assertEqual(result.acceptance.kind, "claimed")

    def test_streaming_reaches_the_same_verdict(self):
        """Two paths do not stay the same unless something checks."""
        agent = Agent(_model(_calls(("c1", "echo", {"value": "x"})),
                             _text("done")),
                      tools=[Echo()], verbose=0)
        list(agent.stream("go"))
        self.assertEqual(agent.last_result.acceptance.kind, "checked")


class TestBeacons(unittest.TestCase):

    def test_a_beacon_from_a_tool_reaches_the_result(self):
        class Blocked(Tool):
            name = "probe"
            parameters = {"type": "object",
                          "properties": {"value": {"type": "string"}},
                          "required": ["value"]}

            def run(self, value, options=None):
                beacon("blocker", "credentials expired", source="probe")
                return "stopped"

        result = Agent(_model(_calls(("c1", "probe", {"value": "x"})),
                              _text("done")),
                       tools=[Blocked()], verbose=0).run("go")
        self.assertEqual([b.kind for b in result.beacons], ["blocker"])
        self.assertTrue(result.beacons[0].demands_attention)

    def test_a_wrong_kind_names_the_vocabulary(self):
        with self.assertRaises(ValueError) as caught:
            beacon("urgent", "x")
        for kind in ATTENTION:
            self.assertIn(kind, str(caught.exception))

    def test_a_beacon_nobody_hears_says_so(self):
        """Not a raise — a tool may be run outside any agent — and not
        silence, which is the failure this exists to prevent."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            beacon("fact", "outside any run")
        self.assertTrue(any("nobody is listening" in str(w.message)
                            for w in caught))

    def test_coordination_beacons_do_not_demand_attention(self):
        from yait_aichain.agent import Beacon
        for kind in ("contract", "decision", "fact"):
            self.assertFalse(Beacon(kind, "x").demands_attention)


class TestAPoolLeavesItsWaitOnABlocker(unittest.TestCase):

    def test_it_stops_starting_items(self):
        """Serialised to one worker so the order is deterministic: the second
        item raises a blocker, and nothing after it should begin."""
        started = []

        class Worker(Tool):
            name = "work"
            parameters = {"type": "object",
                          "properties": {"n": {"type": "integer"}},
                          "required": ["n"]}

            def run(self, n, options=None):
                started.append(n)
                if n == 1:
                    beacon("blocker", "upstream schema changed")
                time.sleep(0.01)
                return n

        pool = Pool(runner=Worker(), items=[{"n": i} for i in range(8)],
                    max_flows=1)
        pool.run()
        self.assertIn(1, started)
        self.assertLess(len(started), 8,
                        "every item ran — the blocker interrupted nothing")
        self.assertEqual([b.kind for b in pool.beacons], ["blocker"])

    def test_a_fact_does_not_stop_it(self):
        started = []

        class Worker(Tool):
            name = "work"
            parameters = {"type": "object",
                          "properties": {"n": {"type": "integer"}},
                          "required": ["n"]}

            def run(self, n, options=None):
                started.append(n)
                if n == 1:
                    beacon("fact", "row count is 42")
                return n

        pool = Pool(runner=Worker(), items=[{"n": i} for i in range(5)],
                    max_flows=1)
        self.assertEqual(pool.run(), [0, 1, 2, 3, 4])
        self.assertEqual(len(started), 5)


if __name__ == "__main__":
    unittest.main()
