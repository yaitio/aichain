"""
Tests for the step boundary: events/hooks, logging routing, the tool
permission matrix, and how a bad call is corrected.

Self-contained: a scripted ``FakeModel`` drives the Agent through the real
``to_request → client.send → from_response`` seam without any network.

Two 1.4.4 behaviours are gone with the 2.0 agent and are no longer tested
here: approval via suspend/resume (there is no suspended run — state lives
in the conversation) and a dedicated repair pass (a rejected call now comes
back to the model as a tool result, which is the same contract through the
ordinary channel).
"""

import json
import logging

import pytest

from yait_aichain import Event, Hook, Tracer, LoggingTracer, PermissionPolicy
from yait_aichain._events import emit
from yait_aichain.agent import Agent
from yait_aichain.skills import Skill
from yait_aichain.chain import Chain
from yait_aichain.tools import Tool
from yait_aichain.models._calls import ToolCall, ToolCallRequest
from yait_aichain.tools._permissions import (
    FINANCIAL, DESTRUCTIVE, WRITE, READ, APPROVE, DENY, ALLOW,
)


# ── Scripted fake model (Model interface) ──────────────────────────────────────

class _FakeClient:
    def __init__(self, scripted):
        self._scripted = scripted
        self.i = 0

    def _auth_headers(self):
        return {}

    def send(self, path, body, headers):
        # The transport carries bytes, so the scripted reply travels by index:
        # a typed ToolCallRequest is not JSON, and pretending otherwise would
        # test a wire this fake does not have.
        i = min(self.i, len(self._scripted) - 1)
        self.i += 1
        return json.dumps(
            {"_i": i, "usage": {"input_tokens": 3, "output_tokens": 4}}
        )


class _FakeModel:
    def __init__(self, scripted, name="fake-orch"):
        self.name = name
        self.client = _FakeClient(scripted)

    def to_request(self, messages, output, tools=None):
        return ("/x", {})

    def from_response(self, response, output):
        return self.client._scripted[response["_i"]]


# ── Tools ──────────────────────────────────────────────────────────────────────

class Echo(Tool):
    name = "echo"
    risk = WRITE
    parameters = {"type": "object",
                  "properties": {"text": {"type": "string"}},
                  "required": ["text"]}

    def run(self, text, options=None):
        return f"echo:{text}"


class Refund(Tool):
    name = "issue_refund"
    risk = FINANCIAL
    parameters = {"type": "object",
                  "properties": {"amount": {"type": "number"}},
                  "required": ["amount"]}

    def run(self, amount, options=None):
        return f"refunded {amount}"


def _call(tool, arguments, call_id="c1"):
    """One native tool call, as the model layer hands it to the Agent."""
    return ToolCallRequest(calls=(ToolCall(id=call_id, name=tool,
                                           arguments=arguments),))


# ── Event / Hook / Tracer unit tests ────────────────────────────────────────────

class TestEvents:
    def test_event_autostamps_ts(self):
        e = Event(type="x")
        assert e.ts > 0 and e.type == "x"

    def test_hook_dispatches_to_named_method(self):
        seen = []

        class H(Hook):
            def tool_call_started(self, e):
                seen.append(e.name)

        emit([H()], Event(type="tool_call.started", name="echo"))
        assert seen == ["echo"]

    def test_tracer_records_all(self):
        tr = Tracer()
        emit([tr], Event(type="a"))
        emit([tr], Event(type="b"))
        assert [e.type for e in tr.events] == ["a", "b"]

    def test_buggy_hook_never_propagates(self):
        tr = Tracer()
        # A raising hook must not stop later hooks or crash emit().
        emit([lambda e: 1 / 0, tr], Event(type="ok"))
        assert [e.type for e in tr.events] == ["ok"]


# ── Permission policy unit tests ────────────────────────────────────────────────

class TestPermissionPolicy:
    def test_defaults(self):
        p = PermissionPolicy()
        assert p.decide_risk(WRITE) == ALLOW
        assert p.decide_risk(FINANCIAL) == APPROVE
        assert p.decide_risk(DESTRUCTIVE) == DENY

    def test_overrides(self):
        p = PermissionPolicy({"financial": "deny", "external": "allow"})
        assert p.decide_risk(FINANCIAL) == DENY
        assert p.decide_risk("external") == ALLOW

    def test_decide_reads_tool_risk(self):
        assert PermissionPolicy().decide(Refund()) == APPROVE

    def test_invalid_decision_rejected(self):
        with pytest.raises(ValueError):
            PermissionPolicy({"write": "maybe"})


# ── Tool risk + arg validation ──────────────────────────────────────────────────

class TestToolContract:
    def test_default_risk(self):
        assert Tool.risk == WRITE

    def test_check_args_missing(self):
        msg = Echo().check_args({})
        assert msg and "missing required" in msg and "echo" in msg

    def test_check_args_ok(self):
        assert Echo().check_args({"text": "hi"}) is None


# ── Agent integration ───────────────────────────────────────────────────────────

class TestAgentEvents:
    def test_lifecycle_and_tool_events(self):
        tr = Tracer()
        ag = Agent(_FakeModel([_call("echo", {"text": "hi"}), "DONE"]),
                   tools=[Echo()], hooks=[tr])
        r = ag.run("hi")
        types = {e.type for e in tr.events}
        assert r.success and r.output == "DONE"
        assert {"run.started", "step.started", "step.ended",
                "llm_call.started", "run.finished"} <= types

    def test_run_finished_carries_usage(self):
        tr = Tracer()
        Agent(_FakeModel([_call("echo", {"text": "x"}), "d"]),
              tools=[Echo()], hooks=[tr]).run("x")
        fin = next(e for e in tr.events if e.type == "run.finished")
        assert fin.usage and fin.usage > 0


class TestAgentPermissions:
    def test_deny_blocks_the_tool_and_tells_the_model(self):
        # A denied call is reported back through the tool channel, so the
        # model can choose something else — it is not a crash and not a
        # silent no-op.
        ran = []

        class Watched(Refund):
            def run(self, amount, options=None):
                ran.append(amount)
                return super().run(amount)

        ag = Agent(_FakeModel([_call("issue_refund", {"amount": 50}), "d"]),
                   tools=[Watched()],
                   permissions=PermissionPolicy({"financial": "deny"}))
        r = ag.run("refund")
        assert r.success
        assert ran == []                                  # never executed
        failed = [e for e in r.journal if e["outcome"] == "failed"]
        assert failed and "denied by policy" in failed[0]["reason"]

    def test_no_policy_runs_unchanged(self):
        ag = Agent(_FakeModel([_call("issue_refund", {"amount": 7}), "done"]),
                   tools=[Refund()])          # no permissions= → no gating
        r = ag.run("refund")
        assert r.success and r.output == "done"
        assert [e["outcome"] for e in r.journal] == ["done"]


class TestBadCallCorrection:
    def test_a_rejected_call_comes_back_as_a_result_and_the_model_retries(self):
        # 1.4.4 ran a dedicated repair pass. Now the schema complaint travels
        # the ordinary tool channel: same contract, one mechanism instead of
        # two, and the model sees exactly what the caller would.
        ag = Agent(_FakeModel([_call("echo", {}),                  # invalid
                               _call("echo", {"text": "fixed"}, "c2"),
                               "repaired"]),
                   tools=[Echo()])
        r = ag.run("repair")
        assert r.success and r.output == "repaired"
        reasons = [e["reason"] for e in r.journal if e["outcome"] == "failed"]
        assert any("missing required" in x for x in reasons)


# ── Skill & Chain hooks ─────────────────────────────────────────────────────────

class TestSkillChainHooks:
    def test_skill_emits_llm_events(self):
        tr = Tracer()
        sk = Skill(model=_FakeModel(["hello"]),
                   input={"messages": [{"role": "user", "parts": ["hi"]}]},
                   hooks=[tr])
        out = sk.run()
        assert out == "hello"
        types = [e.type for e in tr.events]
        assert types == ["llm_call.started", "llm_call.ended"]
        assert tr.events[1].usage == 7        # 3 + 4 from the fake usage

    def test_chain_emits_step_events(self):
        tr = Tracer()
        s1 = Skill(model=_FakeModel(["A"]),
                   input={"messages": [{"role": "user", "parts": ["x"]}]},
                   name="first")
        s2 = Skill(model=_FakeModel(["B"]),
                   input={"messages": [{"role": "user", "parts": ["y"]}]},
                   name="second")
        Chain(steps=[s1, s2], hooks=[tr]).run()
        starts = [e.name for e in tr.events if e.type == "step.started"]
        ends = [e.name for e in tr.events if e.type == "step.ended"]
        assert starts == ["first", "second"] and ends == ["first", "second"]


# ── Logging routing ─────────────────────────────────────────────────────────────

class TestLoggingRouting:
    def test_emits_to_application_handler(self):
        records = []

        class Cap(logging.Handler):
            def emit(self, rec):
                records.append(rec.getMessage())

        pkg = logging.getLogger("yait_aichain")
        cap = Cap()
        pkg.addHandler(cap)
        old_level = pkg.level
        pkg.setLevel(logging.INFO)
        try:
            Agent(_FakeModel([_call("echo", {"text": "hi"}), "d"]),
                  tools=[Echo()], verbose=0).run("hi")
        finally:
            pkg.removeHandler(cap)
            pkg.setLevel(old_level)
        # The library never configures a sink; it emits through named loggers
        # and the application decides where that goes.
        assert any("echo" in m for m in records)
        assert any("[Done]" in m for m in records)
