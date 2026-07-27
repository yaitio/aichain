"""
Tests for mode="goal" — the plan-less loop.

Covers construction (done_when is mandatory, goal-specific budget defaults),
the loop itself (iterating until the condition is met, callable vs string
conditions), the stop rules that make an open-ended loop terminate (iteration
cap, token budget, no progress), the refusal to accept an unearned
final_answer, and suspend/resume of a goal run.
"""

import json
import unittest

from yait_aichain.agent import Agent, Journal
from yait_aichain.state import SuspendedResult
from yait_aichain.tools import Tool


# ── scripted model ─────────────────────────────────────────────────────────────

class _Client:
    def __init__(self, scripted):
        self.scripted, self.i = scripted, 0

    def _auth_headers(self):
        return {}

    def send(self, path, body, headers):
        out = self.scripted[min(self.i, len(self.scripted) - 1)]
        self.i += 1
        return json.dumps({"_c": out,
                           "usage": {"input_tokens": 5, "output_tokens": 5}})


class _Model:
    name = "fake"

    def __init__(self, scripted):
        self.client = _Client(scripted)
        self.prompts: list[str] = []

    def to_request(self, messages, output):
        self.prompts.append("\n".join(
            p.get("text", "") for m in messages for p in m.get("parts", [])))
        return ("/x", {})

    def from_response(self, response, output):
        return response["_c"]


def _act(intent, tool="echo", **kwargs):
    return json.dumps({"intent": intent, "type": "tool",
                       "tool_name": tool, "kwargs": kwargs})


def _assess(outcome="done", met=False, answer="", store_as="", reason=""):
    return json.dumps({"outcome": outcome, "assessment": "…", "reason": reason,
                       "store_as": store_as, "objective_met": met,
                       "final_answer": answer})


class Echo(Tool):
    name = "echo"
    parameters = {"type": "object",
                  "properties": {"text": {"type": "string"}},
                  "required": ["text"]}

    def run(self, text, options=None):
        return f"echo:{text}"


class Boom(Tool):
    name = "boom"
    description = "always fails"
    parameters = {"type": "object", "properties": {}}

    def run(self, options=None):
        raise RuntimeError("nope")


def _agent(scripted, **kw):
    model = _Model(scripted)
    kw.setdefault("done_when", "the answer is in memory")
    return Agent(orchestrator=model, tools=[Echo(), Boom()],
                 mode="goal", **kw), model


# ── construction ───────────────────────────────────────────────────────────────

class TestGoalConstruction(unittest.TestCase):

    def test_done_when_is_mandatory(self):
        with self.assertRaises(ValueError):
            Agent(orchestrator=_Model([]), mode="goal")

    def test_done_when_rejected_outside_goal_mode(self):
        with self.assertRaises(ValueError):
            Agent(orchestrator=_Model([]), done_when="x")

    def test_goal_defaults_are_wider_than_plan_defaults(self):
        a, _ = _agent([])
        self.assertEqual((a.max_steps, a.max_tokens), (50, 250_000))
        plan = Agent(orchestrator=_Model([]))
        self.assertEqual((plan.max_steps, plan.max_tokens), (10, 50_000))

    def test_explicit_budget_wins(self):
        a, _ = _agent([], max_steps=3, max_tokens=99)
        self.assertEqual((a.max_steps, a.max_tokens), (3, 99))

    def test_goal_is_a_valid_mode(self):
        self.assertIn("goal", Agent.MODES)


# ── the loop ───────────────────────────────────────────────────────────────────

class TestGoalLoop(unittest.TestCase):

    def test_runs_until_objective_met(self):
        a, _ = _agent([
            _act("first", text="a"),
            _assess(store_as="a"),                       # not met yet
            _act("second", text="b"),
            _assess(met=True, answer="FINAL", store_as="b"),
        ])
        res = a.run("do the thing")
        self.assertTrue(res.success)
        self.assertEqual(res.output, "FINAL")
        self.assertEqual(res.steps_taken, 2)
        self.assertEqual(res.memory["a"], "echo:a")

    def test_no_planning_call_is_made(self):
        a, model = _agent([_act("only", text="a"), _assess(met=True, answer="F")])
        a.run("task")
        # Two calls only: decide + assess. A planning call would be a third
        # prompt, and would mention "plan".
        self.assertEqual(len(model.prompts), 2)
        self.assertNotIn("execution plan", model.prompts[0])

    def test_iterations_are_journalled_without_a_plan(self):
        a, _ = _agent([_act("only", text="a"), _assess(met=True, answer="F")])
        res = a.run("task")
        self.assertEqual(len(res.journal), 1)
        self.assertEqual(res.journal[0]["intent"], "only")
        self.assertEqual(res.journal[0]["outcome"], "done")

    def test_progress_reaches_the_next_action_prompt(self):
        a, model = _agent([
            _act("gather data", text="a"), _assess(store_as="found"),
            _act("second", text="b"),      _assess(met=True, answer="F"),
        ])
        a.run("task")
        self.assertIn("gather data", model.prompts[2])
        self.assertIn("memory['found']", model.prompts[2])

    def test_refuted_reaches_the_next_action_prompt(self):
        a, model = _agent([
            _act("bad idea", tool="boom"),
            _assess(outcome="refuted", reason="the API has no such endpoint"),
            _act("good idea", text="b"), _assess(met=True, answer="F"),
        ])
        a.run("task")
        self.assertIn("ALREADY RULED OUT", model.prompts[2])
        self.assertIn("no such endpoint", model.prompts[2])

    def test_early_final_answer_is_accepted_without_a_callable(self):
        a, _ = _agent([json.dumps({"intent": "finish", "type": "final_answer",
                                   "answer": "known already"})])
        res = a.run("task")
        self.assertTrue(res.success)
        self.assertEqual(res.output, "known already")
        self.assertEqual(res.journal[0]["evidence"]["kind"], "model_claim")


class TestObservationTrail(unittest.TestCase):
    """The loop must be able to read its own feedback, or it repeats itself."""

    def test_result_is_recorded_on_the_entry(self):
        a, _ = _agent([_act("probe", text="hello"), _assess(met=True, answer="F")])
        res = a.run("task")
        self.assertEqual(res.journal[0]["observation"], "echo:hello")

    def test_observation_reaches_the_next_action_prompt(self):
        a, model = _agent([
            _act("probe", text="500"), _assess(),
            _act("probe again", text="250"), _assess(met=True, answer="F"),
        ])
        a.run("task")
        self.assertIn("echo:500", model.prompts[2])

    def test_failed_attempts_keep_their_observation(self):
        # A failed probe still returned information; dropping it would hide the
        # very feedback the next decision depends on.
        a, model = _agent([
            _act("bad", tool="boom"), _assess(outcome="failed"),
            _act("next", text="b"),   _assess(met=True, answer="F"),
        ])
        a.run("task")
        res_prompt = model.prompts[2]
        self.assertIn("[failed]", res_prompt)
        self.assertIn("nope", res_prompt)

    def test_observations_are_bounded(self):
        from yait_aichain.agent._journal import OBSERVATION_CHARS
        a, _ = _agent([_act("big", text="x" * 5000),
                       _assess(met=True, answer="F")])
        res = a.run("task")
        self.assertLessEqual(len(res.journal[0]["observation"]),
                             OBSERVATION_CHARS)


# ── done_when as a callable ────────────────────────────────────────────────────

class TestCallableCondition(unittest.TestCase):

    def test_callable_ends_the_run_on_a_check(self):
        a, _ = _agent([_act("store it", text="a"), _assess(store_as="answer")],
                      done_when=lambda m: "answer" in m)
        res = a.run("task")
        self.assertTrue(res.success)
        self.assertEqual(res.journal[-1]["evidence"]["kind"], "check")

    def test_unearned_final_answer_is_refuted_not_accepted(self):
        a, _ = _agent([
            json.dumps({"intent": "finish", "type": "final_answer",
                        "answer": "trust me"}),          # condition NOT met
            _act("actually do it", text="a"), _assess(store_as="answer"),
        ], done_when=lambda m: "answer" in m)
        res = a.run("task")
        self.assertTrue(res.success)
        self.assertNotEqual(res.output, "trust me")
        self.assertEqual(res.journal[0]["outcome"], "refuted")

    def test_raising_predicate_does_not_crash_the_run(self):
        def boom(_memory):
            raise KeyError("bad predicate")

        a, _ = _agent([_act("x", text="a"), _assess()], done_when=boom,
                      max_steps=2)
        res = a.run("task")
        self.assertFalse(res.success)          # never satisfied → stops cleanly
        self.assertIsNotNone(res.error)


# ── stop rules ─────────────────────────────────────────────────────────────────

class TestStopRules(unittest.TestCase):

    def test_iteration_cap_stops_the_loop(self):
        a, _ = _agent([_act("again", text="a"), _assess()], max_steps=3)
        res = a.run("task")
        self.assertFalse(res.success)
        self.assertIn("iteration cap", res.error)
        self.assertEqual(res.steps_taken, 3)

    def test_token_budget_stops_the_loop(self):
        a, _ = _agent([_act("again", text="a"), _assess()], max_tokens=30)
        res = a.run("task")
        self.assertFalse(res.success)
        self.assertIn("budget", res.error.lower())

    def test_no_progress_stops_before_the_budget_runs_out(self):
        a, _ = _agent([_act("doomed", tool="boom"), _assess(outcome="failed")],
                      max_steps=50, max_tokens=250_000)
        res = a.run("task")
        self.assertFalse(res.success)
        self.assertIn("No progress", res.error)
        # It gave up on the window, not on the cap.
        self.assertLess(res.steps_taken, 10)

    def test_one_early_failure_does_not_end_the_run(self):
        # The no-progress rule needs a full window; firing on a single failure
        # would kill a run that was about to recover.
        a, _ = _agent([
            _act("shaky", tool="boom"), _assess(outcome="failed"),
            _act("recovered", text="a"), _assess(met=True, answer="F"),
        ])
        res = a.run("task")
        self.assertTrue(res.success)
        self.assertEqual(res.output, "F")

    def test_a_stuck_run_is_visible_in_the_journal(self):
        a, _ = _agent([_act("doomed", tool="boom"), _assess(outcome="failed")],
                      max_steps=50)
        res = a.run("task")
        self.assertFalse(Journal.from_list(res.journal).has_progress(5))
        self.assertTrue(all(e["evidence"]["kind"] == "check"
                            for e in res.journal))


class TestRepetitionDetector(unittest.TestCase):
    """has_progress catches failing; is_repeating catches succeeding pointlessly."""

    def test_identical_intent_is_flagged(self):
        j = Journal()
        for _ in range(5):
            j.append("refine the threshold", action={"type": "tool", "n": 1})
        self.assertTrue(j.is_repeating(5))

    def test_distinct_intents_are_not_flagged(self):
        j = Journal()
        for i in range(5):
            j.append(f"probe midpoint of {i}-999",
                     action={"type": "tool", "kwargs": {"g": i}})
        self.assertFalse(j.is_repeating(5))

    def test_identical_action_is_flagged_even_with_varied_wording(self):
        j = Journal()
        for i in range(5):
            j.append(f"try again, attempt {i}",
                     action={"type": "tool", "tool_name": "p", "kwargs": {"g": 7}})
        self.assertTrue(j.is_repeating(5))

    def test_a_partial_window_is_never_flagged(self):
        j = Journal()
        for _ in range(4):
            j.append("same", action={"type": "tool", "n": 1})
        self.assertFalse(j.is_repeating(5))

    def test_empty_intents_do_not_trigger_it(self):
        j = Journal()
        for i in range(5):
            j.append("", action={"type": "tool", "kwargs": {"g": i}})
        self.assertFalse(j.is_repeating(5))

    def test_a_spinning_run_is_told_so_in_the_prompt(self):
        # Seven identical actions: once the window is full the loop must say so.
        # The scripted model sticks on its last entry, so the pairs are repeated
        # rather than relying on it to cycle.
        a, model = _agent([_act("refine the threshold", text="x"), _assess()] * 8,
                          max_steps=7)
        a.run("task")
        self.assertTrue(any("STOP AND RECONSIDER" in p for p in model.prompts))

    def test_a_healthy_run_is_never_told_so(self):
        a, model = _agent([_act("first", text="a"), _assess(),
                           _act("second", text="b"), _assess(met=True, answer="F")])
        a.run("task")
        self.assertFalse(any("STOP AND RECONSIDER" in p for p in model.prompts))


class TestMemoryRead(unittest.TestCase):
    """A truncated preview must not be a value the agent can never finish."""

    def _tool(self, memory):
        a, _ = _agent([])
        a.memory.update(memory)
        return next(t for t in a.tools if t.name == "memory_read")

    def test_it_is_registered_on_every_agent(self):
        a, _ = _agent([])
        self.assertIn("memory_read", [t.name for t in a.tools])
        plain = Agent(orchestrator=_Model([]))
        self.assertIn("memory_read", [t.name for t in plain.tools])

    def test_it_returns_the_slice_past_the_preview(self):
        body = "".join(str(i % 10) for i in range(3000))
        out  = self._tool({"doc": body}).run("doc", offset=500, length=100)
        self.assertIn(body[500:600], out)

    def test_it_says_how_much_remains(self):
        out = self._tool({"doc": "x" * 3000}).run("doc", offset=0, length=1000)
        self.assertIn("2000 characters remain", out)
        self.assertIn("offset=1000", out)

    def test_it_marks_the_end_of_a_value(self):
        self.assertIn("end of value",
                      self._tool({"doc": "short"}).run("doc"))

    def test_an_unknown_key_lists_what_exists(self):
        out = self._tool({"doc": "x"}).run("nope")
        self.assertIn("no such memory key", out)
        self.assertIn("doc", out)

    def test_the_slice_is_capped(self):
        tool = self._tool({"doc": "x" * 100_000})
        out  = tool.run("doc", offset=0, length=999_999)
        self.assertLessEqual(len(out), tool.MAX_LENGTH + 200)

    def test_a_truncated_preview_advertises_the_tool(self):
        from yait_aichain.agent._prompts import _vars_list
        rendered = _vars_list({"doc": "x" * 9000})
        self.assertIn("memory_read('doc')", rendered)
        self.assertIn("9,000 characters total", rendered)

    def test_a_short_value_is_shown_whole_without_a_notice(self):
        from yait_aichain.agent._prompts import _vars_list
        self.assertNotIn("memory_read", _vars_list({"doc": "short value"}))


class TestSpawnInteraction(unittest.TestCase):
    """Found live: a spawning goal agent crashed, and leaked its own memory."""

    def test_a_child_never_holds_its_parents_memory_reader(self):
        from yait_aichain.agent._agent import _MemoryReadTool
        parent = Agent(orchestrator=_Model([]), allow_spawn=True)
        forwarded = [t for t in parent.tools if t.name != "spawn_agent"]
        child = Agent(orchestrator=_Model([]), tools=forwarded)
        names = [t.name for t in child.tools]
        self.assertEqual(names.count("memory_read"), 1)
        self.assertFalse(any(isinstance(t, _MemoryReadTool) and t._agent is parent
                             for t in child.tools))

    def test_a_child_reads_its_own_memory_not_the_parents(self):
        parent = Agent(orchestrator=_Model([]))
        parent.memory.update({"secret": "parent value"})
        child = Agent(orchestrator=_Model([]), tools=list(parent.tools))
        child.memory.update({"own": "child value"})
        tool = next(t for t in child.tools if t.name == "memory_read")
        self.assertIn("child value", tool.run("own"))
        self.assertIn("no such memory key", tool.run("secret"))

    def test_spawning_from_goal_mode_does_not_raise(self):
        # done_when is a predicate over this agent's objective and memory;
        # neither transfers to a scoped sub-task, so the child must not be
        # constructed in goal mode with nothing to stop it.
        parent = Agent(orchestrator=_Model(['{"steps": []}']), mode="goal",
                       done_when=lambda m: True, allow_spawn=True)
        out = parent.spawn("a sub-task")           # must not raise ValueError
        self.assertIsInstance(out, str)


class TestMemoryViewIsNotStored(unittest.TestCase):
    """Reading memory must not write it back — that is pure churn."""

    def test_a_memory_read_result_is_not_stored_again(self):
        a, _ = _agent([
            _act("look at what I have", tool="memory_read", key="doc"),
            _assess(store_as="copy_of_doc"),
            _act("second", text="b"), _assess(met=True, answer="F"),
        ])
        res = a.run("task", variables={"doc": "the original"})
        self.assertIn("doc", res.memory)
        self.assertNotIn("copy_of_doc", res.memory)

    def test_an_ordinary_result_is_still_stored(self):
        a, _ = _agent([_act("do it", text="x"), _assess(store_as="kept"),
                       _act("second", text="b"), _assess(met=True, answer="F")])
        res = a.run("task")
        self.assertEqual(res.memory["kept"], "echo:x")


# ── suspend / resume ───────────────────────────────────────────────────────────

class Gate(Tool):
    name = "gate"
    description = "waits for an external signal"
    parameters = {"type": "object", "properties": {}}

    def run(self, options=None, _signal=None):
        if _signal is None:
            from yait_aichain.state import Suspend
            raise Suspend("need approval", {"approved": "bool"})
        return f"signal:{_signal}"


class TestGoalSuspendResume(unittest.TestCase):

    def _agent_with_gate(self, scripted):
        model = _Model(scripted)
        return Agent(orchestrator=model, tools=[Echo(), Gate()], mode="goal",
                     done_when="the answer is in memory"), model

    def test_goal_run_suspends_and_resumes(self):
        a, _ = self._agent_with_gate([
            _act("ask a human", tool="gate"),
            _assess(met=True, answer="APPROVED-PATH"),
        ])
        res = a.run("task")
        self.assertIsInstance(res, SuspendedResult)

        final = a.resume(res.run_id, signal={"approved": True})
        self.assertTrue(final.success)
        self.assertEqual(final.output, "APPROVED-PATH")

    def test_journal_survives_a_goal_suspend(self):
        a, _ = self._agent_with_gate([
            _act("first", text="a"), _assess(store_as="a"),
            _act("ask a human", tool="gate"),
            _assess(met=True, answer="DONE"),
        ])
        res = a.run("task")
        self.assertIsInstance(res, SuspendedResult)
        final = a.resume(res.run_id, signal={"approved": True})
        intents = [e["intent"] for e in final.journal]
        self.assertIn("first", intents)          # recorded before the suspend
        self.assertIn("ask a human", intents)     # recorded after the resume


if __name__ == "__main__":
    unittest.main()
