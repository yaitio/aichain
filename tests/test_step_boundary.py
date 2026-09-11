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

    def stream(self, messages, output, tools=None):
        """The same script, delivered as a stream.

        A double that answers only the buffered call stops standing in for a
        provider the moment the agent streams — and the tests that caught this
        are about streaming, so the fake would have failed the feature rather
        than the code. Text is yielded piece by piece; a decision to call a
        tool is not prose and is left on `last_stream_result`, exactly as a
        real client does.
        """
        self.last_stream_usage = {"usage": {"input_tokens": 3,
                                            "output_tokens": 4}}
        i = min(self.client.i, len(self.client._scripted) - 1)
        self.client.i += 1
        reply = self.client._scripted[i]
        self.last_stream_result = reply
        if isinstance(reply, str):
            for piece in (reply[:1], reply[1:]):
                if piece:
                    yield piece
            self.last_stream_result = None


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
        # `tool_call.*`, not `step.*`: Chain emits `step.*` for a chain step
        # (see the Chain test below), so one name meant two things.
        assert {"run.started", "tool_call.started", "tool_call.ended",
                "llm_call.started", "run.finished"} <= types

    def test_run_finished_carries_usage(self):
        tr = Tracer()
        Agent(_FakeModel([_call("echo", {"text": "x"}), "d"]),
              tools=[Echo()], hooks=[tr]).run("x")
        fin = next(e for e in tr.events if e.type == "run.finished")
        assert fin.usage and fin.usage > 0


class TestApproveActuallyGates:
    """The decision that did nothing for four months.

    `_do_tool` compared the policy's answer against "deny" and ignored every
    other value, so `approve` — what the shipped defaults give to external,
    financial, privileged, and to any risk class nobody classified — meant run
    the tool. A policy attached in order to gate spending gated nothing.

    The older tests here are the other half of the story: they asserted what
    `decide()` **returns** and never that the returned decision **happened**.
    Every test below is about the effect, and one of them is written the way
    the missing one should have been.
    """

    def _watched(self, sink):
        class Watched(Refund):
            def run(self, amount, options=None):
                sink.append(amount)
                return super().run(amount)
        return Watched()

    def test_a_gated_tool_does_not_run_when_there_is_nobody_to_ask(self):
        """The headline. A decision whose entire content is "a human should
        see this first" cannot resolve to "go ahead" because no human was
        configured."""
        ran = []
        ag = Agent(_FakeModel([_call("issue_refund", {"amount": 50}), "d"]),
                   tools=[self._watched(ran)],
                   permissions=PermissionPolicy())          # defaults: approve
        result = ag.run("refund")
        assert ran == []
        assert result.success                                # refusal is a result
        failed = [e for e in result.journal if e["outcome"] == "failed"]
        assert failed and "approval" in failed[0]["reason"].lower()

    def test_the_refusal_names_both_ways_out(self):
        """An error a reader cannot act on turns a safety default into a wall
        they route around by removing the policy."""
        ran = []
        ag = Agent(_FakeModel([_call("issue_refund", {"amount": 50}), "d"]),
                   tools=[self._watched(ran)],
                   permissions=PermissionPolicy())
        reason = [e for e in ag.run("refund").journal
                  if e["outcome"] == "failed"][0]["reason"]
        assert "approve=" in reason
        assert "'financial'" in reason and "allow" in reason

    def test_an_approver_saying_yes_lets_it_run(self):
        ran, seen = [], []
        ag = Agent(_FakeModel([_call("issue_refund", {"amount": 50}), "d"]),
                   tools=[self._watched(ran)],
                   permissions=PermissionPolicy(),
                   approve=lambda req: seen.append(req) or True)
        ag.run("refund")
        assert ran == [50]
        assert (seen[0].tool, seen[0].risk) == ("issue_refund", "financial")

    def test_the_approver_is_shown_what_it_would_run_with(self):
        """Approving a name rather than a call is approving nothing: the
        arguments are the whole of what distinguishes a $5 refund from a
        $50 000 one."""
        seen = []
        ag = Agent(_FakeModel([_call("issue_refund", {"amount": 50}), "d"]),
                   tools=[Refund()], permissions=PermissionPolicy(),
                   approve=lambda req: seen.append(req) or True)
        ag.run("refund")
        assert seen[0].arguments == {"amount": 50}

    def test_an_approver_saying_no_blocks_it(self):
        ran = []
        ag = Agent(_FakeModel([_call("issue_refund", {"amount": 50}), "d"]),
                   tools=[self._watched(ran)],
                   permissions=PermissionPolicy(),
                   approve=lambda req: False)
        result = ag.run("refund")
        assert ran == []
        failed = [e for e in result.journal if e["outcome"] == "failed"]
        assert failed and "not approved" in failed[0]["reason"]

    def test_a_refusal_is_a_result_the_model_hears_about(self):
        """Not a crash, and not a silent no-op: a denied call the model is
        never told about is one it will simply make again."""
        ag = Agent(_FakeModel([_call("issue_refund", {"amount": 50}), "done"]),
                   tools=[Refund()], permissions=PermissionPolicy(),
                   approve=lambda req: False)
        result = ag.run("refund")
        assert result.success and result.output == "done"

    def test_an_allowed_class_is_not_asked_about(self):
        """Asking about everything is how an approver stops being read."""
        asked = []
        ag = Agent(_FakeModel([_call("issue_refund", {"amount": 50}), "d"]),
                   tools=[Refund()],
                   permissions=PermissionPolicy({"financial": "allow"}),
                   approve=lambda req: asked.append(req) or True)
        ag.run("refund")
        assert asked == []

    def test_no_policy_means_no_gate_at_all(self):
        """Enforcement stays opt-in. An Agent without `permissions=` behaves
        exactly as it did, approver or not."""
        ran = []
        ag = Agent(_FakeModel([_call("issue_refund", {"amount": 50}), "d"]),
                   tools=[self._watched(ran)])
        ag.run("refund")
        assert ran == [50]


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


class TestApprovalTravelsOnTheChannel:
    """R5's library half: a UI presents a prompt, returns a decision and
    renders the outcome, with no access to the permission layer beyond the
    events and the approver.

    The ordering is the part that had to be built rather than described. The
    gate lived inside the tool call, so a streaming consumer received
    `approval.requested` only after the decision had been made — the record
    read correctly and was useless for the one thing R5 exists for. The gate
    is at the loop boundary now, and the request goes out before anyone is
    asked.
    """

    def _agent(self, approve=None, **kw):
        return Agent(_FakeModel([_call("issue_refund", {"amount": 50}), "done"]),
                     tools=[Refund()], permissions=PermissionPolicy(),
                     approve=approve, **kw)

    def test_the_request_names_the_call_the_tool_and_the_risk(self):
        tr = Tracer()
        self._agent(approve=lambda r: True, hooks=[tr]).run("refund")
        asked = next(e for e in tr.events if e.type == "approval.requested")
        assert asked.payload["tool"] == "issue_refund"
        assert asked.payload["risk"] == "financial"
        assert asked.payload["arguments"] == {"amount": 50}
        assert asked.payload["id"]

    def test_the_decision_carries_the_verdict(self):
        tr = Tracer()
        self._agent(approve=lambda r: True, hooks=[tr]).run("refund")
        decided = next(e for e in tr.events if e.type == "approval.decided")
        assert decided.payload["granted"] is True
        assert decided.payload["id"] == next(
            e.payload["id"] for e in tr.events
            if e.type == "approval.requested")

    def test_a_refusal_can_say_why(self):
        """A UI showing "not approved" and nothing else has thrown away the
        only part a person can act on, and the reason cannot be recovered
        afterwards — it was in the head of whoever clicked no."""
        from yait_aichain.tools import ApprovalDecision
        tr = Tracer()
        self._agent(hooks=[tr],
                    approve=lambda r: ApprovalDecision(False, "over budget")
                    ).run("refund")
        decided = next(e for e in tr.events if e.type == "approval.decided")
        assert decided.payload["granted"] is False
        assert decided.payload["reason"] == "over budget"

    def test_a_denied_call_still_ends(self):
        """The terminal event, so a consumer stops waiting."""
        tr = Tracer()
        self._agent(approve=lambda r: False, hooks=[tr]).run("refund")
        ended = next(e for e in tr.events if e.type == "tool_call.ended")
        assert ended.error and "not approved" in ended.error

    def test_the_missing_approver_is_a_visible_decision(self):
        """Otherwise the call simply fails and nothing says it was governance
        rather than a broken tool."""
        tr = Tracer()
        self._agent(hooks=[tr]).run("refund")
        types = [e.type for e in tr.events]
        assert "approval.requested" in types and "approval.decided" in types
        decided = next(e for e in tr.events if e.type == "approval.decided")
        assert decided.payload["granted"] is False
        assert "no approver" in decided.payload["reason"]

    def test_an_allowed_class_starts_no_conversation(self):
        tr = Tracer()
        Agent(_FakeModel([_call("issue_refund", {"amount": 50}), "d"]),
              tools=[Refund()],
              permissions=PermissionPolicy({"financial": "allow"}),
              hooks=[tr]).run("refund")
        assert not [e for e in tr.events if e.type.startswith("approval.")]

    def test_a_streaming_consumer_sees_the_prompt_before_it_is_answered(self):
        """The requirement that had to be built. A prompt delivered after the
        decision is a record, not a prompt."""
        order = []
        agent = self._agent(
            approve=lambda r: order.append("asked") or True)
        for event in agent.stream("refund"):
            order.append(event.type)
        assert order.index("approval.requested") < order.index("asked")
        assert order.index("asked") < order.index("approval.decided")

    def test_and_in_the_right_place_among_the_actions(self):
        agent = self._agent(approve=lambda r: True)
        types = [e.type for e in agent.stream("refund")]
        assert (types.index("tool_call.started")
                < types.index("approval.requested")
                < types.index("approval.decided")
                < types.index("tool_call.ended"))

    def test_the_human_is_asked_once(self):
        """The gate moved to the loop; leaving a second one inside the tool
        call would put the question to a person twice."""
        asked = []
        self._agent(approve=lambda r: asked.append(r) or True).run("refund")
        assert len(asked) == 1
