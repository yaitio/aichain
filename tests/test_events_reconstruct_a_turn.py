"""
The event channel must carry enough to **reconstruct** a turn, not describe it.

It carried a description: `step.started` emitted the tool's name and nothing
else, and the call's arguments and its result reached only two places — the
message list, which is the model's view, and the journal, where the result had
already been rendered to prose for the model (media reduced to
`[returned 2 image]`). Right for a journal. Wrong for the only channel a
program can watch while a run is alive.

The acceptance conditions here are the ones the requirement was written with:
a consumer attached to a run over a tool returning a structured document can
rebuild, for every call — which tool, with which arguments, what came back, in
what order, and whether it failed — without reading the message list and
without parsing prose.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Agent, Tracer                              # noqa: E402
from yait_aichain.tools import Tool                                 # noqa: E402
from test_loop import _calls, _model, _text                         # noqa: E402


class Report(Tool):
    """Returns a structure, not prose — which is the whole point."""
    name        = "report"
    description = "Return rows."
    parameters  = {"type": "object",
                   "properties": {"region": {"type": "string"}},
                   "required": ["region"]}

    def __init__(self, value=None):
        self._value = value

    def run(self, region, options=None):
        if self._value is not None:
            return self._value
        return {"columns": ["month", "revenue"],
                "rows": [["2026-01", 12], ["2026-02", 15]],
                "units": "USD thousands", "notes": ["provisional"]}


def _run(*responses, tools=None):
    tracer = Tracer()
    agent = Agent(_model(*responses), tools=tools or [Report()],
                  hooks=[tracer], verbose=0)
    return agent.run("go"), tracer


class TestOneCallCanBeRebuilt(unittest.TestCase):

    def setUp(self):
        self.result, self.tracer = _run(
            _calls(("c1", "report", {"region": "emea"})), _text("done"))
        self.started = [e for e in self.tracer.events
                        if e.type == "tool_call.started"]
        self.ended = [e for e in self.tracer.events
                      if e.type == "tool_call.ended"]

    def test_which_tool(self):
        self.assertEqual([e.name for e in self.started], ["report"])

    def test_with_which_arguments(self):
        self.assertEqual(self.started[0].payload["arguments"],
                         {"region": "emea"})

    def test_what_came_back_with_its_own_structure(self):
        """Not a rendering of it. Units, column metadata and the source's own
        notes are the reason a consumer wants the raw value — every one of
        them is gone once the result is prose, and none can be recovered
        downstream."""
        result = self.ended[0].payload["result"]
        self.assertEqual(result["units"], "USD thousands")
        self.assertEqual(result["columns"], ["month", "revenue"])
        self.assertEqual(result["notes"], ["provisional"])

    def test_and_whether_it_failed(self):
        self.assertIsNone(self.ended[0].error)

    def test_without_reading_the_message_list(self):
        """Everything above came off the events alone."""
        self.assertTrue(self.result.success)


class TestParallelCallsStayApart(unittest.TestCase):
    """The agent honours every call a provider requests — deliberately. With
    no id per event a consumer watching two in flight cannot pair a result
    with its arguments, and this is the requirement that it stays closed even
    if calls later execute concurrently rather than in sequence."""

    def test_three_calls_pair_up_by_id(self):
        _, tracer = _run(
            _calls(("a", "report", {"region": "emea"}),
                   ("b", "report", {"region": "apac"}),
                   ("c", "report", {"region": "latam"})),
            _text("done"))
        asked = {e.payload["id"]: e.payload["arguments"]["region"]
                 for e in tracer.events if e.type == "tool_call.started"}
        answered = {e.payload["id"]: e.payload["result"]
                    for e in tracer.events if e.type == "tool_call.ended"}
        self.assertEqual(asked, {"a": "emea", "b": "apac", "c": "latam"})
        self.assertEqual(set(answered), set(asked))

    def test_the_order_is_preserved(self):
        _, tracer = _run(
            _calls(("a", "report", {"region": "emea"}),
                   ("b", "report", {"region": "apac"})), _text("d"))
        ids = [e.payload["id"] for e in tracer.events
               if e.type == "tool_call.started"]
        self.assertEqual(ids, ["a", "b"])


class TestARefusalIsNotAnEmptyAnswer(unittest.TestCase):
    """Three outcomes that used to collapse into one. A consumer that cannot
    tell a refusal from an empty answer draws an empty chart for both, and an
    empty chart is a lie about the data."""

    def _end(self, value):
        _, tracer = _run(_calls(("c1", "report", {"region": "x"})),
                         _text("d"), tools=[Report(value)])
        return next(e for e in tracer.events if e.type == "tool_call.ended")

    def test_zero_rows_is_zero_rows(self):
        event = self._end({"rows": [], "columns": ["month", "revenue"]})
        self.assertEqual(event.payload["result"]["rows"], [])
        self.assertIsNone(event.error)

    def test_a_decline_carries_the_source_s_own_words(self):
        event = self._end({"declined": True, "reason": "region not licensed"})
        self.assertEqual(event.payload["result"]["reason"],
                         "region not licensed")
        self.assertIsNone(event.error)

    def test_a_raise_is_the_third_thing(self):
        class Broken(Report):
            def run(self, region, options=None):
                raise RuntimeError("upstream down")
        _, tracer = _run(_calls(("c1", "report", {"region": "x"})),
                         _text("d"), tools=[Broken()])
        event = next(e for e in tracer.events if e.type == "tool_call.ended")
        self.assertIn("upstream down", event.error)


class TestIdentityAndOrder(unittest.TestCase):

    def test_every_event_carries_the_run(self):
        """`run_id` has been a declared field since M3 and the agent never
        filled it. Two concurrent invocations — the ordinary case for the
        serverless target — write into one stream that cannot be
        demultiplexed without it."""
        _, tracer = _run(_calls(("c1", "report", {"region": "x"})),
                         _text("d"))
        ids = {e.run_id for e in tracer.events}
        self.assertEqual(len(ids), 1)
        self.assertTrue(next(iter(ids)))

    def test_two_runs_do_not_share_one(self):
        _, first = _run(_calls(("c1", "report", {"region": "x"})), _text("d"))
        _, second = _run(_calls(("c1", "report", {"region": "y"})), _text("d"))
        self.assertNotEqual(first.events[0].run_id, second.events[0].run_id)

    def test_a_tool_event_says_which_turn_it_belongs_to(self):
        _, tracer = _run(_calls(("a", "report", {"region": "x"})),
                         _calls(("b", "report", {"region": "y"})),
                         _text("done"))
        steps = [e.step for e in tracer.events
                 if e.type == "tool_call.started"]
        self.assertEqual(steps, [1, 2])


class TestTheStreamCarriesTheSameThing(unittest.TestCase):
    """`Agent.stream()` is a view of the hook channel, so whatever a hook can
    rebuild a turn from, a streaming consumer can too — otherwise a UI would
    have to attach a hook *and* walk the stream."""

    def test_a_streamed_tool_event_carries_the_result_and_the_run(self):
        agent = Agent(_model(_calls(("c1", "report", {"region": "emea"})),
                             _text("done")),
                      tools=[Report()], verbose=0)
        events = list(agent.stream("go"))
        ended = next(e for e in events if e.type == "tool_call.ended")
        self.assertEqual(ended.payload["result"]["units"], "USD thousands")
        self.assertEqual(ended.payload["id"], "c1")
        self.assertTrue(ended.run_id)
        self.assertEqual({e.run_id for e in events}, {ended.run_id})


class TestTheOldSpellingStillReachesAHook(unittest.TestCase):
    """Renaming is a breaking change for a hook written against `step.*`, and
    the agent's spelling collided with Chain's own step events. A hook using
    the old method keeps firing, once, with a warning naming the new one."""

    def test_a_hook_written_for_step_started_is_still_called(self):
        import warnings
        from yait_aichain import Hook

        seen = []

        class Old(Hook):
            def step_started(self, event):
                seen.append(event.name)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Agent(_model(_calls(("c1", "report", {"region": "x"})),
                         _text("d")),
                  tools=[Report()], hooks=[Old()], verbose=0).run("go")
        self.assertEqual(seen, ["report"])
        self.assertTrue(any(issubclass(w.category, DeprecationWarning)
                            for w in caught))

    def test_the_new_name_wins_when_both_are_defined(self):
        from yait_aichain import Hook
        seen = []

        class Both(Hook):
            def step_started(self, event):
                seen.append("old")

            def tool_call_started(self, event):
                seen.append("new")

        Agent(_model(_calls(("c1", "report", {"region": "x"})), _text("d")),
              tools=[Report()], hooks=[Both()], verbose=0).run("go")
        self.assertEqual(seen, ["new"])


if __name__ == "__main__":
    unittest.main()
