"""
The agent loop — a conversation that can act, over the native tool channel.

What this pins, in the order it matters:

  * the conversation **appends** and never rebuilds — the cacheable prefix
    only grows;
  * schemas travel as the provider's ``tools`` field, not as prompt text —
    the system prompt carries instructions and nothing else;
  * text is the answer, calls are actions — one exit;
  * a ceiling reached is **not** success, and ``stopped_by`` says which fired;
  * ``pool`` / ``delegate`` / ``write_plan`` are synthetic tools on the same
    channel as the real ones, offered only when the configuration earns them.

Mocks speak the chat-completions wire (tool_calls in the message), because
that is what the model under test (gpt-4o) speaks.
"""

import json
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent, step_count, token_budget, cost_budget, check
from yait_aichain.models import Model
from yait_aichain.tools._base import Tool


def _text(content, prompt_tokens=100, completion_tokens=20):
    return json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }).encode()


def _calls(*specs, content=None, prompt_tokens=100, completion_tokens=20):
    """A chat-completions reply asking for tool calls: (id, name, args)."""
    return json.dumps({
        "choices": [{"message": {
            "content": content,
            "tool_calls": [
                {"id": i, "type": "function",
                 "function": {"name": n, "arguments": json.dumps(a)}}
                for i, n, a in specs
            ],
        }}],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }).encode()


def _model(*responses, name="gpt-4o"):
    """A model whose transport replays a script; the last reply repeats."""
    m = Model(name, api_key="k")
    m.sent = []
    call = [0]

    def side(path, body, headers, *a, **k):
        m.sent.append(body)
        i = min(call[0], len(responses) - 1)
        call[0] += 1
        return responses[i]

    def sse(path, body, headers, *a, **k):
        """The same script, delivered as a stream.

        A double that answers only the buffered call stops being a stand-in
        for a provider the moment the agent starts streaming: every tool event
        vanished and the run ended in a network error, which reads as the
        feature being broken rather than the fake being half-written.
        """
        m.sent.append(body)
        i = min(call[0], len(responses) - 1)
        call[0] += 1
        message = (json.loads(responses[i])["choices"][0]["message"])
        if message.get("content"):
            yield {"choices": [{"delta": {"content": message["content"]}}]}
        for index, tc in enumerate(message.get("tool_calls") or []):
            yield {"choices": [{"delta": {"tool_calls": [
                {"index": index, "id": tc["id"],
                 "function": {"name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]}}]}}]}
        yield {"choices": [], "usage": json.loads(responses[i])["usage"]}

    m.client._post = MagicMock(side_effect=side)
    m.client._post_sse = MagicMock(side_effect=sse)
    m.client._auth_headers = MagicMock(return_value={})
    return m


class Echo(Tool):
    name        = "echo"
    description = "Echo a value."
    parameters  = {"type": "object", "properties": {"value": {"type": "string"}},
                   "required": ["value"]}

    def __init__(self, sink=None):
        self._sink = sink if sink is not None else []
        self._lock = threading.Lock()

    def run(self, value, options=None):
        with self._lock:
            self._sink.append(value)
        return f"echo:{value}"


class Boom(Tool):
    name       = "boom"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **kw):
        raise RuntimeError("tool exploded")


_ANSWER = _text("all done")
_ECHO   = _calls(("c1", "echo", {"value": "x"}))


# ── The exit ─────────────────────────────────────────────────────────────────

class TestAnswering(unittest.TestCase):

    def test_a_text_reply_ends_the_run(self):
        res = Agent(_model(_ANSWER), tools=[Echo()]).run("do it")
        self.assertTrue(res.success)
        self.assertEqual(res.output, "all done")
        self.assertEqual(res.stopped_by, "answered")
        self.assertEqual(res.steps_taken, 0)


# ── The wire ─────────────────────────────────────────────────────────────────

class TestNativeChannel(unittest.TestCase):

    def test_schemas_ride_in_the_tools_field_not_the_prompt(self):
        m = _model(_ANSWER)
        Agent(m, tools=[Echo()]).run("do it")
        body = m.sent[0]
        names = [t["function"]["name"] for t in body["tools"]]
        self.assertIn("echo", names)
        # and NOT in the system prompt
        system = body["messages"][0]["content"]
        self.assertNotIn('"parameters"', json.dumps(system))

    def test_the_system_prompt_is_instructions_not_machinery(self):
        m = _model(_ANSWER)
        Agent(m, tools=[Echo()], instructions="Be terse.").run("do it")
        system = m.sent[0]["messages"][0]["content"]
        self.assertIn("Be terse.", system)
        self.assertLess(len(system), 600,
                        "the prompt regrew machinery it no longer needs")

    def test_the_ground_rule_belongs_to_the_loop_owner(self):
        # "Plain text ends the run" is true inside run() and false under an
        # external driver, where a plain reply is an ordinary turn. Measured
        # on a dialogue benchmark as a bias toward polite refusal over action
        # when the run() rule leaked into the driven mode.
        agent = Agent(_model(_ANSWER), tools=[Echo()], instructions="Brief.")
        driven = agent.opening("task")[0]["parts"][0]
        self.assertNotIn("final answer", driven)

        agent.run("task")
        owned = agent.model.sent[0]["messages"][0]["content"]
        self.assertIn("final answer", owned)

    def test_synthetic_tools_are_offered_by_configuration(self):
        def names(agent):
            m = agent.model
            agent.run("x")
            return [t["function"]["name"] for t in m.sent[0]["tools"]]

        base = names(Agent(_model(_ANSWER), tools=[Echo()]))
        self.assertIn("pool", base)
        self.assertNotIn("delegate", base)
        self.assertNotIn("write_plan", base)

        with_team = names(Agent(_model(_ANSWER), tools=[Echo()], team="auto"))
        self.assertIn("delegate", with_team)

        waterfall = names(Agent(_model(_ANSWER), tools=[Echo()],
                                mode="waterfall"))
        self.assertIn("write_plan", waterfall)


# ── The conversation ─────────────────────────────────────────────────────────

class TestConversationGrows(unittest.TestCase):

    def _run_two_steps(self):
        m = _model(_ECHO, _calls(("c2", "echo", {"value": "y"})), _ANSWER)
        Agent(m, tools=[Echo()]).run("do it")
        return m.sent

    def test_each_call_sends_more_than_the_last(self):
        sent = self._run_two_steps()
        lengths = [len(json.dumps(b["messages"])) for b in sent]
        self.assertEqual(lengths, sorted(lengths))
        self.assertGreater(lengths[-1], lengths[0])

    def test_the_earlier_turns_are_still_there_verbatim(self):
        sent = self._run_two_steps()
        first, last = json.dumps(sent[0]["messages"]), json.dumps(sent[-1]["messages"])
        self.assertTrue(last.startswith(first[:-1]),
                        "the prefix was rebuilt, not appended to")

    def test_the_result_comes_back_as_a_tool_turn_with_the_call_id(self):
        sent = self._run_two_steps()
        tool_msgs = [x for x in sent[-1]["messages"] if x.get("role") == "tool"]
        self.assertEqual(tool_msgs[0]["tool_call_id"], "c1")
        self.assertIn("echo:x", tool_msgs[0]["content"])


# ── Stopping ─────────────────────────────────────────────────────────────────

class TestStopConditions(unittest.TestCase):

    def test_a_ceiling_is_not_success(self):
        res = Agent(_model(_ECHO), tools=[Echo()],
                    stop_when=[step_count(2)]).run("do it")
        self.assertFalse(res.success)
        self.assertEqual(res.stopped_by, "step_count")
        self.assertEqual(res.steps_taken, 2)

    def test_a_check_that_passes_is_success(self):
        res = Agent(_model(_ECHO), tools=[Echo()],
                    stop_when=[check(lambda s: s["steps"] >= 2, name="enough")]
                    ).run("do it")
        self.assertTrue(res.success)
        self.assertEqual(res.stopped_by, "check:enough")

    def test_token_budget_counts_what_was_spent(self):
        res = Agent(_model(_ECHO), tools=[Echo()],
                    stop_when=[token_budget(200)]).run("do it")
        self.assertEqual(res.stopped_by, "token_budget")

    def test_cost_budget_counts_money(self):
        big = _calls(("c1", "echo", {"value": "x"}),
                     prompt_tokens=100_000, completion_tokens=100_000)
        res = Agent(_model(big), tools=[Echo()],
                    stop_when=[cost_budget(0.10)]).run("do it")
        self.assertEqual(res.stopped_by, "cost_budget")
        self.assertGreater(res.cost, 0.10)


# ── Executing calls ──────────────────────────────────────────────────────────

class TestToolCalls(unittest.TestCase):

    def test_the_tool_runs_with_its_arguments(self):
        sink = []
        Agent(_model(_ECHO, _ANSWER), tools=[Echo(sink)]).run("do it")
        self.assertEqual(sink, ["x"])

    def test_an_unknown_tool_is_reported_back_not_raised(self):
        m = _model(_calls(("c1", "nope", {})), _ANSWER)
        res = Agent(m, tools=[Echo()]).run("do it")
        self.assertTrue(res.success)
        self.assertIn("no tool named", json.dumps(m.sent[-1]["messages"]))

    def test_a_raising_tool_is_reported_back(self):
        m = _model(_calls(("c1", "boom", {})), _ANSWER)
        res = Agent(m, tools=[Boom()]).run("do it")
        self.assertTrue(res.success)
        self.assertIn("tool exploded", json.dumps(m.sent[-1]["messages"]))

    def test_parallel_calls_all_execute_in_one_step(self):
        sink = []
        m = _model(_calls(("a", "echo", {"value": "1"}),
                          ("b", "echo", {"value": "2"})), _ANSWER)
        res = Agent(m, tools=[Echo(sink)]).run("do it")
        self.assertEqual(sorted(sink), ["1", "2"])
        self.assertEqual(res.steps_taken, 1)      # one decision, two calls
        tool_msgs = [x for x in m.sent[-1]["messages"] if x.get("role") == "tool"]
        self.assertEqual([t["tool_call_id"] for t in tool_msgs], ["a", "b"])


class TestPoolCall(unittest.TestCase):

    def _pool(self, **over):
        return _calls(("p", "pool", {"items": ["a", "b", "c", "d"],
                                     "tool": "echo", "arg": "value", **over}))

    def test_one_call_processes_the_whole_list(self):
        sink = []
        Agent(_model(self._pool(), _ANSWER), tools=[Echo(sink)]).run("do it")
        self.assertEqual(sorted(sink), ["a", "b", "c", "d"])

    def test_dict_entries_spread_across_the_parameters(self):
        sink = []
        call = _calls(("p", "pool", {"items": [{"value": "a"}, {"value": "b"}],
                                     "tool": "echo"}))
        Agent(_model(call, _ANSWER), tools=[Echo(sink)]).run("do it")
        self.assertEqual(sorted(sink), ["a", "b"])

    def test_a_fan_out_where_everything_failed_is_an_error_not_none(self):
        call = _calls(("p", "pool", {"items": ["a", "b"], "tool": "boom"}))
        m = _model(call, _ANSWER)
        Agent(m, tools=[Boom()]).run("do it")
        self.assertIn("fanned-out calls failed", json.dumps(m.sent[-1]["messages"]))

    def test_an_empty_list_is_reported_back(self):
        call = _calls(("p", "pool", {"items": [], "tool": "echo"}))
        m = _model(call, _ANSWER)
        Agent(m, tools=[Echo()]).run("do it")
        self.assertIn("non-empty 'items'", json.dumps(m.sent[-1]["messages"]))


class TestDelegate(unittest.TestCase):

    _DELEGATE = _calls(("d", "delegate", {"task": "sub", "worker": "helper"}))

    def test_without_a_team_the_tool_is_not_declared(self):
        m = _model(_ANSWER)
        Agent(m, tools=[Echo()]).run("do it")
        names = [t["function"]["name"] for t in m.sent[0]["tools"]]
        self.assertNotIn("delegate", names)

    def test_with_a_cast_only_its_members_may_be_named(self):
        helper = Agent(_model(_text("sub done")), tools=[], name="helper")
        m = _model(_calls(("d", "delegate", {"task": "s", "worker": "stranger"})),
                   _ANSWER)
        Agent(m, tools=[Echo()], team=[helper]).run("do it")
        self.assertIn("no worker named", json.dumps(m.sent[-1]["messages"]))

    def test_a_named_worker_runs_and_returns(self):
        helper = Agent(_model(_text("sub done")), tools=[], name="helper")
        m = _model(self._DELEGATE, _ANSWER)
        Agent(m, tools=[Echo()], team=[helper]).run("do it")
        last = json.dumps(m.sent[-1]["messages"])
        self.assertIn("sub done", last)
        self.assertIn("model_claim", last)     # the child's word, so labelled

    def test_the_childs_spend_is_added_to_the_parents(self):
        helper = Agent(_model(_text("sub done")), tools=[], name="helper")
        m = _model(self._DELEGATE, _ANSWER)
        res = Agent(m, tools=[Echo()], team=[helper]).run("do it")
        self.assertGreater(res.tokens_used, 240)

    def test_delegation_depth_is_bounded(self):
        m = _model(_calls(("d", "delegate", {"task": "sub",
                                             "instructions": "deeper"})),
                   _ANSWER)
        res = Agent(m, tools=[], team="auto", _depth=5).run("do it")
        self.assertTrue(res.success)
        self.assertIn("maximum delegation depth",
                      json.dumps(m.sent[-1]["messages"]))


class TestPlannerModel(unittest.TestCase):
    """Think while planning, act without deliberating: the phase boundary is
    the recorded plan, and each side may have its own model."""

    def test_turns_route_by_phase(self):
        planner = _model(_calls(("w", "write_plan", {"items": ["a"]})))
        doer    = _model(_ECHO, _ANSWER)
        Agent(doer, tools=[Echo()], mode="waterfall",
              planner_model=planner).run("do it")
        self.assertEqual(len(planner.sent), 1)      # exactly the plan turn
        self.assertGreaterEqual(len(doer.sent), 2)  # everything after

    def test_planner_model_requires_waterfall(self):
        with self.assertRaises(ValueError):
            Agent(object(), planner_model=object())


class TestWaterfall(unittest.TestCase):

    def test_the_plan_is_recorded_via_the_tool(self):
        m = _model(_calls(("w", "write_plan", {"items": ["one", "two"]})),
                   _ANSWER)
        res = Agent(m, tools=[Echo()], mode="waterfall").run("do it")
        self.assertEqual([s["goal"] for s in res.plan], ["one", "two"])
        self.assertIn("Plan recorded", json.dumps(m.sent[-1]["messages"]))

    def test_a_second_plan_is_refused(self):
        m = _model(_calls(("w1", "write_plan", {"items": ["one"]})),
                   _calls(("w2", "write_plan", {"items": ["two", "three"]})),
                   _ANSWER)
        res = Agent(m, tools=[Echo()], mode="waterfall").run("do it")
        self.assertEqual([s["goal"] for s in res.plan], ["one"])
        self.assertIn("cannot be rewritten", json.dumps(m.sent[-1]["messages"]))

    def test_a_ceiling_with_a_plan_outstanding_leaves_a_trace(self):
        m = _model(_calls(("w", "write_plan", {"items": ["one", "two"]})),
                   _ECHO)
        res = Agent(m, tools=[Echo()], mode="waterfall",
                    stop_when=[step_count(2)]).run("do it")
        self.assertFalse(res.success)
        self.assertTrue([e for e in res.journal if e.get("outcome") == "skipped"])


# ── Journal and accounting ───────────────────────────────────────────────────

class TestJournal(unittest.TestCase):

    def test_every_call_is_recorded(self):
        m = _model(_calls(("a", "echo", {"value": "1"}),
                          ("b", "echo", {"value": "2"})), _ANSWER)
        res = Agent(m, tools=[Echo()]).run("do it")
        done = [e for e in res.journal if e.get("outcome") == "done"]
        self.assertEqual(len(done), 2)

    def test_a_failure_carries_check_evidence(self):
        m = _model(_calls(("c1", "boom", {})), _ANSWER)
        res = Agent(m, tools=[Boom()]).run("do it")
        failed = [e for e in res.journal if e.get("outcome") == "failed"]
        self.assertEqual(failed[0]["evidence"]["kind"], "check")


class TestAccounting(unittest.TestCase):

    def test_tokens_and_cost_both_arrive(self):
        res = Agent(_model(_ECHO, _ANSWER), tools=[Echo()]).run("do it")
        self.assertGreater(res.tokens_used, 0)
        self.assertIsNotNone(res.cost)

    def test_an_unpriced_model_reports_none_not_zero(self):
        m = _model(_ANSWER, name="private/some-local-model")
        self.assertIsNone(Agent(m, tools=[]).run("do it").cost)


class TestConstruction(unittest.TestCase):

    def test_goal_mode_is_gone(self):
        with self.assertRaises(ValueError):
            Agent(object(), mode="goal")

    def test_team_must_be_none_a_list_or_auto(self):
        with self.assertRaises(ValueError):
            Agent(object(), team=42)


if __name__ == "__main__":
    unittest.main()
