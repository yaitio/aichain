"""
_dispatch — the one place a tool call is decided and run.

Before this the rules lived in five places: the agent loop, its `execute()`
seam, `pool` inside the agent, a Chain step and a Pool item — and the last
two only charged the budget, so a refused tool ran there with nothing said
(`tests/test_dispatch_matrix.py` names each path). The invariant this module
exists for: **one place calls `tool.run()`**, and every rule is applied there,
in this order:

    1 resolve    name → tool (the owner's built-ins included)
    2 validate   argument depth, the tool's schema
    3 decide     the policy, on (tool, arguments) → allow / deny / approve
    4 approve    the owner's approver, or a refusal that says why
    5 budget     not begun if it would pass `max_cost`
    6 execute    `tool.run(**arguments)`; `Suspend` passes up untouched
    7 record     what failed, described, for the journal

Two calls, not one, so that `approval.requested` reaches a streaming consumer
*before* the approver is asked — which is what lets a UI put the question to a
person — and so that a run can be parked between the asking and the answer:

    step = dispatcher.prepare(call, state)      # Ready | Denied | NeedsApproval
    if isinstance(step, NeedsApproval):
        step = dispatcher.answer(step, state)   # Ready | Denied
    result, error = dispatcher.execute(call, state)

A refusal is a result, not a crash: it goes back to the model as the call's
error, so the history stays well-formed and the model hears why.

Phase 2 of PLAN-agent-chain.md moves what the agent already did here without
changing it; Chain and Pool come through it in phase 3.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._budget import lend
from .state import Suspend
from .tools._base import charged

#: How deeply a call's arguments may nest. 450 levels raised RecursionError
#: copying the history on the next turn and ended the run unseen.
MAX_ARGUMENT_DEPTH = 64


@dataclass(frozen=True)
class Ready:
    """Nothing stands between the call and its tool."""


@dataclass(frozen=True)
class Denied:
    """The call does not run; *reason* is what the model is told."""
    reason: str


@dataclass(frozen=True)
class NeedsApproval:
    """The policy wants a person; *request* is what they are shown."""
    request: Any


READY = Ready()


def depth(value, level: int = 0) -> int:
    """How deep *value* nests, counted no further than one past the limit."""
    if level > MAX_ARGUMENT_DEPTH:
        return level
    if isinstance(value, dict):
        return max([depth(v, level + 1) for v in value.values()] or [level + 1])
    if isinstance(value, (list, tuple)):
        return max([depth(v, level + 1) for v in value] or [level + 1])
    return level


class Dispatcher:
    """Decides and runs the calls of one owner — an agent today.

    *tools* maps a name to its tool. *builtins* maps the owner's own call
    names (``pool``, ``delegate``, ``write_plan``) to a handler taking the
    call's arguments and the working state. *gated* names, for a built-in that
    runs another tool, the tool the policy is asked about. *run* returns the
    owner's current run, where a pending approval and the last failure live.
    """

    def __init__(self, *, tools: dict, builtins: dict, gated: Callable,
                 permissions, approve, max_cost, emit: Callable, who: str,
                 run: Callable, log_call: Callable) -> None:
        self.tools = tools
        self.builtins = builtins
        self._gated = gated
        self.permissions = permissions
        self.approve = approve
        self.max_cost = max_cost
        self._emit = emit
        self._who = who
        self._run = run
        self._log_call = log_call

    # ── 1 resolve ──────────────────────────────────────────────────────────

    def gated_tool(self, call):
        """The tool the policy decides on: the call's own, or, for a built-in
        that runs another, that one."""
        return self._gated(call)

    # ── 3, 4 decide and approve ────────────────────────────────────────────

    def prepare(self, call, state: dict | None = None):
        """Consult the policy before the tool runs.

        Emits `approval.requested` even when there is nobody to ask: a
        consumer needs to see that a decision was required, and the refusing
        case is where that matters most — otherwise the call simply fails and
        nothing says it was governance rather than a broken tool.
        """
        tool = self.gated_tool(call)
        if tool is None or not self.permissions:
            return READY
        if depth(call.arguments) > MAX_ARGUMENT_DEPTH:
            # Refused before the approver's deep copy would raise on it.
            return Denied(f"tool {tool.name!r}: the arguments are nested more than "
                          f"{MAX_ARGUMENT_DEPTH} levels deep; send a flatter structure")
        from .tools._permissions import ALLOW, APPROVE, DENY, ApprovalRequest, risk_of

        # The call's arguments are the pool's own for `pool`: the policy is
        # asked about the tool it fans out, which takes them one entry at a time.
        arguments = None if call.name == "pool" else dict(call.arguments or {})
        try:
            decision = self.permissions.decide(tool, arguments)
        except TypeError:
            decision = self.permissions.decide(tool)          # a policy of another make
        if decision == ALLOW:
            return READY
        risk = risk_of(tool, arguments)
        if decision == DENY:
            return Denied(f"tool {tool.name!r} denied by policy")
        if decision != APPROVE:
            # A policy of another make answering a word this gate does not
            # know: it was read as "allowed". Anything but yes is no.
            return Denied(f"tool {tool.name!r}: the policy answered {decision!r}, "
                          "which is not a decision")

        run = self._run()
        run.pending = ApprovalRequest(
            # A copy all the way down: an approver that edits what it was
            # shown would change the call it approved.
            tool=tool.name, risk=risk, arguments=copy.deepcopy(dict(call.arguments or {})),
            call_id=getattr(call, "id", "") or "", agent=self._who, run_id=run.id or "")
        self._emit("approval.requested", name=tool.name,
                   step=(state or {}).get("steps"),
                   payload={"id": run.pending.call_id, "tool": tool.name,
                            "risk": risk, "arguments": run.pending.arguments,
                            "agent": self._who})
        return NeedsApproval(run.pending)

    def answer(self, step: NeedsApproval, state: dict | None = None):
        """Ask the approver, then say what it answered: Ready or Denied."""
        run = self._run()
        request, run.pending = step.request, None
        if self.approve is None:
            granted = False
            reason = (f"no approver is attached. Pass Agent(approve=...) to "
                      f"decide per call, or set the policy rule for "
                      f"{request.risk!r} to 'allow' if this class does not "
                      "need gating here.")
        else:
            from .tools._permissions import ApprovalDecision
            reply = self.approve(request)
            # Only a yes that says yes: `bool("denied")` and
            # `bool({"approved": False})` both approved the call.
            if not isinstance(reply, (bool, ApprovalDecision)):
                raise TypeError(f"an approver answers True, False or ApprovalDecision(granted, "
                                f"reason); it answered {type(reply).__name__} {reply!r}")
            granted = bool(reply)
            # An `ApprovalDecision` carries why; a bare False does not, and a
            # UI showing "not approved" and nothing else has thrown away the
            # only part a person can act on.
            reason = getattr(reply, "reason", "") or ("" if granted else "not approved")

        self._emit("approval.decided", name=request.tool,
                   step=(state or {}).get("steps"),
                   payload={"id": request.call_id, "tool": request.tool,
                            "risk": request.risk, "granted": granted,
                            "reason": reason, "agent": self._who})
        if granted:
            return READY
        # A refusal is a result, not a crash: every tool call coming back with
        # something is what keeps the model's history well-formed, and a
        # denied call the model never hears about is one it will simply make
        # again. `tool_call.ended` then carries this as its error.
        head = "needs approval and" if self.approve is None else "was not approved:"
        return Denied(f"tool {request.tool!r} ({request.risk}) {head} {reason}")

    def decide(self, call, state: dict | None = None):
        """`prepare` and, when a person is wanted, `answer` — for a caller with
        no consumer to show the question to first."""
        step = self.prepare(call, state)
        return self.answer(step, state) if isinstance(step, NeedsApproval) else step

    # ── 2, 5, 6, 7 validate, budget, execute, record ───────────────────────

    def execute(self, call, state: dict) -> tuple[Any, str | None]:
        """Run one call: ``(result, error)``, or ``Suspend`` raised for a pause."""
        try:
            if (handler := self.builtins.get(call.name)) is not None:
                return handler(call.arguments, state), None
            return self.run_tool(call), None
        except Suspend:
            # Not a failure, and the one exception this must not flatten: the
            # tool is saying the world has to answer before the run goes on.
            raise
        except Exception as exc:
            from ._errors_policy import describe
            self._run().failure = describe(exc)
            return None, f"{type(exc).__name__}: {exc}"

    def run_tool(self, call) -> Any:
        """The one place a tool runs. No permission check here: the caller
        resolved it before calling, so a second one would put the question to
        a person twice."""
        if depth(call.arguments) > MAX_ARGUMENT_DEPTH:
            # Told to the model, which made it.
            raise ValueError(f"the arguments are nested more than {MAX_ARGUMENT_DEPTH} "
                             "levels deep; send a flatter structure")
        tool = self.tools.get(call.name)
        if tool is None:
            raise ValueError(f"no tool named {call.name!r}. Available: {sorted(self.tools)}")
        kwargs = call.arguments or {}
        if (problem := tool.check_args(kwargs)) is not None:
            raise ValueError(problem)
        self._log_call(tool, kwargs)
        # Under the owner's budget and every one around it.
        with lend(self.max_cost):
            return charged(tool, lambda: tool.run(**kwargs))


__all__ = ["Dispatcher", "Ready", "Denied", "NeedsApproval", "READY",
           "MAX_ARGUMENT_DEPTH", "depth"]
