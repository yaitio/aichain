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

**The rules around a call apply to everything it runs.** While a call
executes, its owner's policy and approver are *governing*: a Chain, a Pool or
an agent started inside it — a tool that runs a pipeline, a `delegate` worker
— decides each of its own calls under the stricter of its own rules and every
governing set around it (deny over approve over allow), and asks the nearest
approver when it has none of its own. An agent's policy that stopped at its
own tool list governed nothing the tool did.

Phase 2 of PLAN-agent-chain.md moves what the agent already did here without
changing it; Chain and Pool come through it in phase 3.
"""

from __future__ import annotations

import contextlib
import contextvars
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
    """Nothing stands between the call and its tool. *call* is the call to run
    — the one asked for, or an interceptor's rewrite of it."""
    call: Any = None


@dataclass(frozen=True)
class Denied:
    """The call does not run; *reason* is what the model is told."""
    reason: str


@dataclass(frozen=True)
class NeedsApproval:
    """The policy wants a person; *request* is what they are shown."""
    request: Any
    call: Any = None


READY = Ready()


@dataclass(frozen=True)
class Block:
    """What an interceptor's ``before`` returns to stop a call.

    Parameters
    ----------
    reason : str
        Why — what the model is told in place of the call's result.
    """
    reason: str


class Interceptor:
    """Acts on a call before and after it runs — where a hook only watches.

    ``before(call, run)`` returns None to leave the call alone, a `Block` to
    stop it, or a `ToolCall` with the same ``id`` and ``name`` and other
    ``arguments`` to run that instead. It runs before the policy, so the
    policy decides on what will actually run. ``after(call, result, run)``
    returns None to leave the result alone, or the result to hand back
    instead. Every change is written down — the journal of an agent's run, an
    ``interceptor.changed`` event — with what it was and what it became, and an
    interceptor that raises in ``before`` blocks the call rather than being
    skipped: a guard that fails open guards nothing. They run in the order
    given. Subclass it, or give any object with either method.
    """

    def before(self, call, run):                      # pragma: no cover - default
        return None

    def after(self, call, result, run):               # pragma: no cover - default
        return None


@dataclass(frozen=True)
class Governor:
    """A policy and its approver, in force around a call. *settled* names the
    tools this call was already decided on — a `pool` asks once for the whole
    list — under every policy then in force, so the items inside it are not
    asked again."""
    permissions: Any
    approve: Any
    settled: frozenset = frozenset()


_GOVERNING: contextvars.ContextVar = contextvars.ContextVar("yait_aichain_governing",
                                                            default=())

#: The id of the call executing now, so a run started inside it can name it.
_CALLING: contextvars.ContextVar = contextvars.ContextVar("yait_aichain_calling", default=None)


def calling() -> str | None:
    """The id of the tool call this code runs inside, if any."""
    return _CALLING.get()


@contextlib.contextmanager
def governing(permissions, approve, settled: frozenset = frozenset()):
    """Make *permissions* and *approve* hold for every call inside the block."""
    if not permissions and not settled:
        yield
        return
    token = _GOVERNING.set(_GOVERNING.get() + (Governor(permissions, approve, settled),))
    try:
        yield
    finally:
        _GOVERNING.reset(token)


def _decide(permissions, tool, arguments):
    try:
        return permissions.decide(tool, arguments)
    except TypeError:
        return permissions.decide(tool)          # a policy of another make


def _short(value, limit: int = 160) -> str:
    import json
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def takes_run(tool) -> bool:
    """Whether *tool* asks for the run it is called in: a keyword-only ``run``
    parameter on its ``run()``. Only that form is reserved — an ordinary
    argument named ``run`` stays the model's."""
    import inspect
    try:
        target = getattr(tool, "__wrapped__", None) or tool.run
        param = inspect.signature(target).parameters.get("run")
    except (TypeError, ValueError):
        return False
    return param is not None and param.kind == param.KEYWORD_ONLY


#: What `policy=` takes, as a dict: the rules by risk class, the decision for
#: a class no rule names, who answers when a rule says "approve", and the
#: interceptors that act on each call. One place for everything that decides
#: what happens to a call.
POLICY_KEYS = ("rules", "default", "approve", "interceptors")

#: The words `policy["approve"]` takes besides a function.
APPROVE_WORDS = ("park", "deny")


class ApprovalParked(Suspend):
    """A call the policy wants a person for, under ``approve: "park"``: the
    run parks with the question, and ``resume(run_id, signal={"approved":
    True})`` runs the call — or ``False``, with a ``"reason"``, refuses it."""
    approval = True


def split_policy(policy, *, approve=None, interceptors=None, where: str = "policy"):
    """``policy=`` as every primitive takes it, split into its three parts:
    ``(permissions, approve, interceptors)``.

    A `PermissionPolicy` is the rules alone. A dict of risk classes is the
    rules too. ``{"rules": …, "default": …, "approve": …, "interceptors": …}``
    is the whole of it, its keys checked — a misspelt ``"aprove"`` refused
    rather than ignored. *approve* and *interceptors* are the separate
    parameters the dict replaced, honoured beside it, never both.
    """
    from .tools._permissions import APPROVE, PermissionPolicy
    rules_given = False
    if policy is None:
        permissions = None
    elif hasattr(policy, "decide"):
        permissions = policy
    elif isinstance(policy, dict) and not (set(policy) & set(POLICY_KEYS)):
        permissions = PermissionPolicy(policy)
    elif isinstance(policy, dict):
        if unknown := sorted(set(policy) - set(POLICY_KEYS)):
            raise ValueError(f"{where} keys are {list(POLICY_KEYS)}; got {unknown}")
        rules_given = "rules" in policy or "default" in policy
        permissions = (PermissionPolicy(policy.get("rules"), default=policy.get("default", APPROVE))
                       if rules_given else None)
        for name, given in (("approve", approve), ("interceptors", interceptors)):
            if name in policy and given is not None:
                raise ValueError(f"{name} is given twice — in {where} and as {name}=; keep the "
                                 f"one in {where}")
        approve = policy.get("approve", approve)
        interceptors = policy.get("interceptors", interceptors)
    else:
        raise TypeError(f"{where} is a PermissionPolicy or a dict, not {type(policy).__name__}")
    if approve is not None and not callable(approve) and approve not in APPROVE_WORDS:
        raise ValueError(f"{where}['approve'] is 'park', 'deny' or a function; got {approve!r}")
    if interceptors is not None and not isinstance(interceptors, (list, tuple)):
        raise ValueError(f"{where}['interceptors'] is a list; got {type(interceptors).__name__}")
    return permissions, approve, list(interceptors or [])


def as_policy(policy):
    """``policy=`` as Chain and Pool take it: a `PermissionPolicy`, a dict of
    rules, or ``{"rules": {...}, "default": ...}``; None for no policy of its own."""
    if policy is None or hasattr(policy, "decide"):
        return policy
    if isinstance(policy, dict):
        from .tools._permissions import APPROVE, PermissionPolicy
        if "rules" in policy or "default" in policy:
            unknown = sorted(set(policy) - {"rules", "default"})
            if unknown:
                raise ValueError(f"policy keys are 'rules' and 'default'; got {unknown}")
            return PermissionPolicy(policy.get("rules"), default=policy.get("default", APPROVE))
        return PermissionPolicy(policy)
    raise TypeError(f"policy is a PermissionPolicy or a dict of rules, not {type(policy).__name__}")


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
                 run: Callable, log_call: Callable, interceptors=(),
                 note: Callable | None = None) -> None:
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
        self.interceptors = list(interceptors or ())
        self._note = note or (lambda text: None)

    # ── interceptors ───────────────────────────────────────────────────────

    def _changed(self, text: str) -> None:
        self._emit("interceptor.changed", payload={"note": text, "agent": self._who})
        self._note(text)

    def _before(self, call):
        """The call after every interceptor's ``before``, or a Denied."""
        for each in self.interceptors:
            hook = getattr(each, "before", None)
            if hook is None:
                continue
            label = getattr(each, "name", None) or type(each).__name__
            try:
                out = hook(call, self._run())
            except Exception as exc:
                return Denied(f"interceptor {label} failed before {call.name!r}, so the call "
                              f"was blocked: {type(exc).__name__}: {exc}")
            if out is None:
                continue
            if isinstance(out, Block):
                return Denied(f"blocked by interceptor {label}: {out.reason}")
            if getattr(out, "name", None) != call.name or getattr(out, "id", None) != call.id:
                return Denied(f"interceptor {label} turned {call.name!r} into another call "
                              f"({getattr(out, 'name', out)!r}); it may change the arguments only")
            if dict(out.arguments or {}) != dict(call.arguments or {}):
                self._changed(f"interceptor {label} changed the arguments of {call.name!r}: "
                              f"{_short(call.arguments)} → {_short(out.arguments)}")
            call = out
        return call

    def _after(self, call, result):
        """The result after every interceptor's ``after``."""
        for each in self.interceptors:
            hook = getattr(each, "after", None)
            if hook is None:
                continue
            label = getattr(each, "name", None) or type(each).__name__
            out = hook(call, result, self._run())
            if out is not None and out is not result:
                self._changed(f"interceptor {label} changed the result of {call.name!r}: "
                              f"{_short(result)} → {_short(out)}")
                result = out
        return result

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
        intercepted = self._before(call)
        if isinstance(intercepted, Denied):
            return intercepted
        call = intercepted
        tool = self.gated_tool(call)
        around = _GOVERNING.get()
        if tool is not None:
            # What was settled for this tool settled it under every policy
            # around the call that settled it, so those are not asked again.
            cut = max((i for i, g in enumerate(around) if tool.name in g.settled), default=-1)
            around = around[cut + 1:]
        policies = [g.permissions for g in around if g.permissions] + (
            [self.permissions] if self.permissions else [])
        if tool is None or not policies:
            return Ready(call)
        if depth(call.arguments) > MAX_ARGUMENT_DEPTH:
            # Refused before the approver's deep copy would raise on it.
            return Denied(f"tool {tool.name!r}: the arguments are nested more than "
                          f"{MAX_ARGUMENT_DEPTH} levels deep; send a flatter structure")
        from .tools._permissions import ALLOW, APPROVE, DENY, ApprovalRequest, risk_of

        # The call's arguments are the pool's own for `pool`: the policy is
        # asked about the tool it fans out, which takes them one entry at a time.
        arguments = None if call.name == "pool" else dict(call.arguments or {})
        # The strictest word among every policy in force: a refusal anywhere
        # refuses, a wish for a person anywhere asks one.
        decisions = [_decide(p, tool, arguments) for p in policies]
        decision = (next((d for d in decisions if d not in (ALLOW, APPROVE)), None)
                    or (APPROVE if APPROVE in decisions else ALLOW))
        if decision == ALLOW:
            return Ready(call)
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
        return NeedsApproval(run.pending, call)

    def answer(self, step: NeedsApproval, state: dict | None = None):
        """Ask the approver, then say what it answered: Ready or Denied."""
        run = self._run()
        request, run.pending = step.request, None
        approve = self.approve if self.approve is not None else next(
            (g.approve for g in reversed(_GOVERNING.get()) if g.approve is not None), None)
        if approve == "park" and self.approve != "park":
            # A lead's "park" waits in the lead's run, which is where the store
            # is; a worker or a chain inside it has nowhere to park, so a call
            # of theirs that needs a person is refused, and says why.
            approve = "deny"
            parked_elsewhere = True
        else:
            parked_elsewhere = False
        if approve == "park":
            # A person answers later, maybe in another process: the run waits
            # with the question, and `resume()` brings the answer.
            raise ApprovalParked(
                reason=f"approve {request.tool!r} ({request.risk})",
                resume_with={"approved": "bool", "reason": "str"},
                hint={"approval": True, "tool": request.tool, "risk": request.risk,
                      "arguments": request.arguments})
        if approve == "deny":
            granted, reason = False, (
                "it needs a person, and the run that can wait for one is the agent around "
                "this one; give this one approve= of its own" if parked_elsewhere else
                f"the policy refuses every call that needs approval ({request.risk!r} here)")
        elif approve is None:
            granted = False
            reason = (f"no approver is attached. Pass Agent(approve=...) to "
                      f"decide per call, or set the policy rule for "
                      f"{request.risk!r} to 'allow' if this class does not "
                      "need gating here.")
        else:
            from .tools._permissions import ApprovalDecision
            reply = approve(request)
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
            return Ready(step.call)
        # A refusal is a result, not a crash: every tool call coming back with
        # something is what keeps the model's history well-formed, and a
        # denied call the model never hears about is one it will simply make
        # again. `tool_call.ended` then carries this as its error.
        head = "needs approval and" if approve in (None, "deny") else "was not approved:"
        return Denied(f"tool {request.tool!r} ({request.risk}) {head} {reason}")

    def decide(self, call, state: dict | None = None):
        """`prepare` and, when a person is wanted, `answer` — for a caller with
        no consumer to show the question to first."""
        step = self.prepare(call, state)
        return self.answer(step, state) if isinstance(step, NeedsApproval) else step

    # ── 2, 5, 6, 7 validate, budget, execute, record ───────────────────────

    def execute(self, call, state: dict) -> tuple[Any, str | None]:
        """Run one call: ``(result, error)``, or ``Suspend`` raised for a pause."""
        token = _CALLING.set(getattr(call, "id", None) or None)
        try:
            if (handler := self.builtins.get(call.name)) is not None:
                # A built-in that runs another tool was decided on that tool,
                # once, before it began: the decision covers what it runs.
                gated = self.gated_tool(call)
                settled = frozenset({gated.name}) if gated is not None else frozenset()
                # Under the owner's budget too: a worker's spend is the
                # team's, and a lead's `max_cost` bounded only its own calls.
                with governing(self.permissions, self.approve, settled), lend(self.max_cost):
                    result = handler(call.arguments, state)
            else:
                result = self.run_tool(call)
            return self._after(call, result), None
        except Suspend:
            # Not a failure, and the one exception this must not flatten: the
            # tool is saying the world has to answer before the run goes on.
            raise
        except Exception as exc:
            from ._errors_policy import describe
            self._run().failure = describe(exc)
            return None, f"{type(exc).__name__}: {exc}"
        finally:
            _CALLING.reset(token)

    def run_tool(self, call, extra: dict | None = None, check: bool = True) -> Any:
        """The one place a tool runs. No permission check here: the caller
        resolved it before calling, so a second one would put the question to
        a person twice.

        *check* holds the arguments to the tool's schema — the schema a model
        is shown, which is for calls a model makes. A Chain or Pool fills
        arguments from its own variables, credentials among them, and passes
        ``check=False``, as it always did."""
        if depth(call.arguments) > MAX_ARGUMENT_DEPTH:
            # Told to the model, which made it.
            raise ValueError(f"the arguments are nested more than {MAX_ARGUMENT_DEPTH} "
                             "levels deep; send a flatter structure")
        tool = self.tools.get(call.name)
        if tool is None:
            raise ValueError(f"no tool named {call.name!r}. Available: {sorted(self.tools)}")
        kwargs = call.arguments or {}
        if check and (problem := tool.check_args(kwargs)) is not None:
            raise ValueError(problem)
        self._log_call(tool, kwargs)
        # *extra* is the library's own, never the model's: a Chain resuming a
        # step hands the signal as `_signal`, which no schema declares.
        call_kwargs = {**kwargs, **(extra or {})}
        if takes_run(tool):
            # The run this call belongs to: its id, its depth, its hooks.
            call_kwargs["run"] = self._run()
        # Under the owner's budget and every one around it, and under its rules.
        with lend(self.max_cost), governing(self.permissions, self.approve):
            return charged(tool, lambda: tool.run(**call_kwargs))


__all__ = ["Dispatcher", "Ready", "Denied", "NeedsApproval", "READY", "Block", "Interceptor",
           "ApprovalParked", "split_policy", "POLICY_KEYS",
           "governing", "calling",
           "as_policy",
           "MAX_ARGUMENT_DEPTH", "depth"]
