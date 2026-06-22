"""
Tests for the 1.4.4 step boundary: events/hooks, logging routing, the tool
permission matrix, approval via suspend/resume, and tool-call repair.

Self-contained: a scripted ``FakeModel`` drives the Agent through the real
``to_request → client.send → from_response`` seam without any network.
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
from yait_aichain.tools._permissions import (
    FINANCIAL, DESTRUCTIVE, WRITE, READ, APPROVE, DENY, ALLOW,
)
from yait_aichain.state import SuspendedResult


# ── Scripted fake model (Model interface) ──────────────────────────────────────

class _FakeClient:
    def __init__(self, scripted):
        self._scripted = scripted
        self.i = 0

    def _auth_headers(self):
        return {}

    def send(self, path, body, headers):
        out = self._scripted[min(self.i, len(self._scripted) - 1)]
        self.i += 1
        return json.dumps(
            {"_content": out, "usage": {"input_tokens": 3, "output_tokens": 4}}
        )


class _FakeModel:
    def __init__(self, scripted, name="fake-orch"):
        self.name = name
        self.client = _FakeClient(scripted)

    def to_request(self, messages, output):
        return ("/x", {})

    def from_response(self, response, output):
        return response["_content"]


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


def _plan(tool, goal="do"):
    return json.dumps({"steps": [{"id": 1, "type": "tool",
                                  "tool_name": tool, "goal": goal}]})


def _act(tool, kwargs):
    return json.dumps({"type": "tool", "tool_name": tool, "kwargs": kwargs})


def _refl(decision="final_answer", **kw):
    return json.dumps({"decision": decision, "assessment": "ok", **kw})


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
        ag = Agent(orchestrator=_FakeModel(
            [_plan("echo"), _act("echo", {"text": "hi"}),
             _refl("final_answer", final_answer="DONE")]),
            tools=[Echo()], hooks=[tr])
        r = ag.run("hi")
        types = {e.type for e in tr.events}
        assert r.success and r.output == "DONE"
        assert {"run.started", "step.started", "tool_call.started",
                "tool_call.ended", "llm_call.started", "run.finished"} <= types

    def test_run_finished_carries_usage(self):
        tr = Tracer()
        Agent(orchestrator=_FakeModel(
            [_plan("echo"), _act("echo", {"text": "x"}),
             _refl("final_answer", final_answer="d")]),
            tools=[Echo()], hooks=[tr]).run("x")
        fin = next(e for e in tr.events if e.type == "run.finished")
        assert fin.usage and fin.usage > 0


class TestAgentPermissions:
    def test_deny_returns_result_without_executing(self):
        tr = Tracer()
        ag = Agent(orchestrator=_FakeModel(
            [_plan("issue_refund"), _act("issue_refund", {"amount": 50}),
             _refl("final_answer", final_answer="d")]),
            tools=[Refund()], hooks=[tr],
            permissions=PermissionPolicy({"financial": "deny"}))
        ag.run("refund")
        assert any(e.payload.get("decision") == DENY for e in tr.events)

    def test_approve_suspends_then_resumes(self):
        ag = Agent(orchestrator=_FakeModel(
            [_plan("issue_refund"), _act("issue_refund", {"amount": 99}),
             _refl("final_answer", final_answer="refund done")]),
            tools=[Refund()],
            permissions=PermissionPolicy({"financial": "approve"}))
        r = ag.run("refund 99")
        assert isinstance(r, SuspendedResult)
        assert "Approval required" in r.awaiting["reason"]
        r2 = ag.resume(r.run_id, signal={"approved": True})
        assert r2.success
        assert any("refunded 99" in str(h.get("output")) for h in r2.history)

    def test_rejected_approval_skips_tool(self):
        ag = Agent(orchestrator=_FakeModel(
            [_plan("issue_refund"), _act("issue_refund", {"amount": 99}),
             _refl("final_answer", final_answer="ok")]),
            tools=[Refund()],
            permissions=PermissionPolicy({"financial": "approve"}))
        r = ag.run("refund")
        r2 = ag.resume(r.run_id, signal={"approved": False})
        assert any(isinstance(h.get("output"), dict)
                   and h["output"].get("skipped") for h in r2.history)

    def test_no_policy_runs_unchanged(self):
        ag = Agent(orchestrator=_FakeModel(
            [_plan("issue_refund"), _act("issue_refund", {"amount": 7}),
             _refl("final_answer", final_answer="done")]),
            tools=[Refund()])               # no permissions= → no gating
        r = ag.run("refund")
        assert r.success


class TestToolCallRepair:
    def test_missing_arg_triggers_repair_then_succeeds(self):
        ag = Agent(orchestrator=_FakeModel(
            [_plan("echo"),
             _act("echo", {}),                       # invalid → remediation
             _refl("retry"),
             _act("echo", {"text": "fixed"}),        # corrected
             _refl("final_answer", final_answer="repaired")]),
            tools=[Echo()])
        r = ag.run("repair")
        assert r.success and r.output == "repaired"
        errs = [h.get("exec_error") for h in r.history if h.get("exec_error")]
        assert any("missing required" in (e or "") for e in errs)


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
            Agent(orchestrator=_FakeModel(
                [_plan("echo"), _act("echo", {"text": "hi"}),
                 _refl("final_answer", final_answer="d")]),
                tools=[Echo()], verbose=0).run("hi")
        finally:
            pkg.removeHandler(cap)
            pkg.setLevel(old_level)
        assert any("Agent:" in m for m in records)
        assert any("[Done]" in m for m in records)
