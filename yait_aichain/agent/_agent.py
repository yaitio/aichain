"""
agent._agent
============

``Agent`` — a conversation that can act.

The whole algorithm::

    messages = [system, task]
    loop:
        reply = Skill(model, messages, tools).run()
        if the reply is text          → done, it is the answer
        for each requested call:
            result = execute(call)
            append the call and its result to the conversation
        if a stop condition fires     → stop

Tool calling is **native**: schemas travel as the provider's ``tools`` field
and the reply comes back as typed calls, not as JSON the model was asked to
write into text. The previous protocol cost 13,473 characters of prompt
machinery per turn and a 42%-correct call rate against 86% for a native agent
on the same tasks — acting was a special case of talking, so weak models
talked.

Everything the agent can *do* is a tool. Real tools are the caller's; three
synthetic ones expose the library's own machinery through the same channel:

``pool``        repeat one tool (or a prompt) over a list, in parallel
``delegate``    hand a scoped sub-task to a worker — only when ``team`` is set
``write_plan``  waterfall's step-0 plan — only when ``mode="waterfall"``

Two parameters carry the design space: ``mode`` — is the sequence frozen;
``team`` — who does the work. The conversation is held with the world, not
with itself, and it only ever appends — which is what keeps the cacheable
prefix stable.
"""

from __future__ import annotations

import json
import contextlib
import copy
import dataclasses
import logging
import time
import uuid
import warnings
from collections.abc import Callable
from typing import Any

from .._budget import Budget, as_budget, lend, start_run
from ..clients._errors import ContextWindowExceededError, TruncatedResponseError
from ..models import Model
from ..models._adaptation import Adaptation
from ..skills import Skill
from ..utilities._base import Utility
from ..state import StateStore
from ..tools._base import charged
from ..tools._permissions import ApprovalRequest, PermissionPolicy

from .._console import console
from .. import _hints
from ._swarm import acceptance_from, collecting_beacons
from ._journal import Journal, evidence as _evidence, CHECK, MODEL_CLAIM, \
                      DONE as _J_DONE, FAILED as _J_FAILED, SKIPPED as _J_SKIPPED
from ._result  import AgentResult
from .         import _prompts as prompts
from .._events import Event, emit
from ..models._calls import (ToolCallRequest, dangling_calls,
                             tool_result_turn)
from ..pool    import Pool
from ..state   import RunContext, Suspend, SuspendedResult, current, using

_logger = logging.getLogger("yait_aichain.agent")


# ── Stop conditions ──────────────────────────────────────────────────────────
#
# Everything that can end a run other than the model simply answering. Two
# kinds, and they must not be confused: a *check* that passes is a success
# with harness-verified evidence, a *ceiling* that is reached is a failure.

def step_count(n: int):
    """Stop after *n* decisions. A ceiling: reaching it is not success.

    Parameters
    ----------
    n : int
        Model turns allowed.
    """
    def _c(state):
        return "step_count" if state["steps"] >= n else None
    _c.kind, _c.name = "ceiling", "step_count"
    # `spec` is what lets a ceiling survive `Chain.save()`: the condition
    # itself is a closure and a file cannot hold one. A `check` has no
    # spec on purpose — its predicate is the caller's function, and
    # pretending it round-trips would be worse than saying it does not.
    _c.spec = {"kind": "step_count", "value": n}
    return _c


def token_budget(n: int):
    """Stop once *n* tokens have been spent. A ceiling.

    Parameters
    ----------
    n : int
        Tokens allowed, input and output together.
    """
    def _c(state):
        return "token_budget" if state["tokens"] >= n else None
    _c.kind, _c.name = "ceiling", "token_budget"
    _c.spec = {"kind": "token_budget", "value": n}
    return _c


def cost_budget(usd: float):
    """
    Stop once *usd* has been spent. A ceiling.

    Bounds "do not begin another step", not "never exceed by a cent": the
    length of a reply is not known before it is paid for.

    Parameters
    ----------
    usd : float
        Dollars allowed.
    """
    def _c(state):
        return "cost_budget" if (state["cost"] or 0.0) >= usd else None
    _c.kind, _c.name = "ceiling", "cost_budget"
    _c.spec = {"kind": "cost_budget", "value": usd}
    return _c


def nudge(predicate, message, name: str = "nudge"):
    """
    A condition that **speaks** instead of ending the run.

    `stop_when` knew two kinds, and both ended the run: a ceiling ("hit it,
    fail") and a check ("met it, succeed"). A stall needs neither. It needs
    "change approach", and the loop had no way to say that — so a stuck run
    either spun until a ceiling stopped it or was stopped early by a ceiling
    set too low to let a real run finish.

    When *predicate(state)* is true a user turn carrying *message* is
    appended and the loop carries on. It enters the conversation as an
    ordinary, logged message — not an invisible edit to the system prompt —
    so a reader of the transcript can see exactly what the model was told
    and when. *message* may be a string or a callable of `state`.

    It fires **once per streak**: while the predicate stays true it stays
    quiet, and it re-arms when the predicate clears. A reminder repeated
    every turn is how a reminder stops being read.

    Parameters
    ----------
    predicate : callable
        Given the run's state; true when the model should be told something.
    message : str | callable
        What it is told — a string, or a function of the state.
    name : str, optional
        How the nudge is named in events.
    """
    def _c(state):
        return f"nudge:{name}" if predicate(state) else None
    _c.kind, _c.name, _c.message = "nudge", name, message
    return _c


#: The risk classes that change something. A call of one of these is an
#: effect; any later call that succeeds — a read, a test run — is a look at
#: what it did.
_EFFECTS = frozenset({"write", "external", "financial", "destructive"})


def verified(message: str | None = None, name: str = "verified"):
    """
    Before the run answers, ask once for a check of what it changed.

    The commonest agent failure there is: "done" with nothing run. When the
    model answers after a call that changed something — a write, an external
    call, a delete — and no call has succeeded since to look at the result,
    its answer is held, *message* is put to it as a user turn, and the loop
    goes on. Once per run: a model that answers again is believed, and the
    record says it was asked.

    Only a call clears it — a read, a test run, anything that succeeds after
    the change. The model saying it checked does not, because that is the
    claim the gate exists to test.

    Parameters
    ----------
    message : str | None, optional
        What the model is told; a default names the change.
    name : str, optional
        How the nudge is reported in its event.
    """
    def _c(state):
        return state.get("unverified")
    _c.kind, _c.name = "verify", name
    _c.message = message or (lambda state: (
        f"You changed something with {state.get('unverified')!r} and have not looked at the "
        "result since. Check it — read it back, or run the check that shows it works — "
        "before you answer."))
    return _c


def stalled(last_k: int = 5):
    """Nudge when the last *last_k* attempts moved nothing.

    `Journal.has_progress` has existed, been documented, and been called
    from no library code at all — available only to someone writing their
    own loop. It reads the journal, not the task metric, so it has nothing to
    overfit to and applies to every task, verifiable or not.

    Parameters
    ----------
    last_k : int, optional
        Attempts that must all have moved nothing.
    """
    def _stalled(state):
        journal = state.get("journal")
        return journal is not None and not journal.has_progress(last_k)

    def _say(state):
        text = (f"The last {last_k} attempts did not move the task forward. "
                "Do not repeat them; change the approach, or conclude from "
                "what you already have.")
        ruled_out = state["journal"].do_not_redo()
        return f"{text}\n\nAlready ruled out:\n{ruled_out}" if ruled_out else text

    return nudge(_stalled, _say, name="stalled")


def repeating(last_k: int = 5):
    """Nudge when the last *last_k* attempts were the same move.

    The other half of a stall: `has_progress` catches a run that is failing,
    this catches one that is succeeding pointlessly — clean results that
    carry no new information. Like `stalled`, it existed in the journal and
    nothing called it.

    Parameters
    ----------
    last_k : int, optional
        Attempts that must all have been the same move.
    """
    def _repeating(state):
        journal = state.get("journal")
        return journal is not None and journal.is_repeating(last_k)

    return nudge(_repeating,
                 f"The last {last_k} actions were the same. Their results "
                 "will not change. Either conclude from what they returned "
                 "or try something different.",
                 name="repeating")


def check(fn, name: str = "check"):
    """
    Stop when *fn* says the objective is met — and call that a **success**.

    The only completion the harness can verify itself. Everything else the
    agent reports about its own progress is the model's word for it.

    Parameters
    ----------
    fn : callable
        Given the run's state; returns true when the objective is met.
    name : str, optional
        How the stop is reported: ``stopped_by="check:<name>"``.
    """
    def _c(state):
        # A raise is the loop's to report — as `condition_error`, the way a
        # ceiling's or a nudge's is — where it was a DEBUG line and "not yet".
        return f"check:{name}" if fn(state) else None
    _c.kind, _c.name = "check", name
    return _c


# ── Synthetic tool schemas ───────────────────────────────────────────────────

_POOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "pool",
        "description": (
            "Repeat ONE tool over every entry of a list, in parallel, in one "
            "step. Use this whenever the same work applies to each item — "
            "calling the tool once per item spends a whole turn on each. "
            "Give either `tool` (run that tool per entry) or `prompt` (run "
            "the model per entry; {item} is the current entry)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                # An array parameter must say what its elements are: Google
                # rejects the whole request when `items` is missing, before
                # the first step. Declared as objects because that is the
                # form that carries named arguments; a plain value still
                # works at runtime and fills `arg`.
                "items": {"type": "array",
                          "items": {"type": "object"},
                          "description": "one entry per call; an object entry "
                                         "is spread as the tool's named "
                                         "arguments, a plain value fills `arg`"},
                "tool": {"type": "string",
                         "description": "name of the tool to run per entry"},
                "arg": {"type": "string",
                        "description": "parameter a plain entry fills "
                                       "(defaults to the tool's first)"},
                "prompt": {"type": "string",
                           "description": "instead of `tool`: a prompt run "
                                          "per entry, {item} = the entry"},
                "max_flows": {"type": "integer"},
            },
            "required": ["items"],
        },
    },
}

_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_plan",
        "description": "Record the plan: a short ordered list of what this "
                       "task requires. Call once, first, then hold to it.",
        "parameters": {
            "type": "object",
            "properties": {"items": {"type": "array",
                                     "items": {"type": "string"}}},
            "required": ["items"],
        },
    },
}


def _delegate_schema(team) -> dict:
    props: dict = {"task": {"type": "string",
                            "description": "a self-contained task"}}
    required = ["task"]
    if isinstance(team, list):
        names = [getattr(w, "name", None) or f"worker{i}"
                 for i, w in enumerate(team)]
        props["worker"] = {"type": "string", "enum": names,
                           "description": "which worker takes it"}
        required.append("worker")
    else:                                              # "auto"
        props["instructions"] = {"type": "string",
                                 "description": "who this worker should be"}
    return {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": (
                "Hand a scoped sub-task to a worker. The worker runs its own "
                "conversation and returns a conclusion — use it to keep bulk "
                "work out of this one."
            ),
            "parameters": {"type": "object", "properties": props,
                           "required": required},
        },
    }


# ── Agent ────────────────────────────────────────────────────────────────────

#: The most entries one `pool` call fans out to, and the most at once. A
#: model sets both; unbounded, a thousand items were a thousand threads.
MAX_POOL_ITEMS = 200
MAX_POOL_FLOWS = 8

#: Returned by `_ask_permission` when a request has gone out and the answer
#: has not been asked for yet. A string would be a denial and None a pass, so
#: the third state needs a value of its own.
_AWAITING = object()



def _is_utility(obj) -> bool:
    return isinstance(obj, Utility)


def _utility_tools(u) -> str:
    names = list(getattr(u, "TOOLS", ()) or ())
    kind = type(u).__name__
    if not names:
        return f"Give the agent {kind}.tools, the tools it has."
    return (f"Its tools are {', '.join(names)} — give the agent those, e.g. tools=[x.{names[0]}], "
            f"or all of them with tools=x.tools.")


def _check_tools(tools: list) -> None:
    """
    Refuse, by name, what cannot be a tool and two tools the model cannot tell
    apart — here, before the first call, rather than as an ``AttributeError``
    on the first turn or a tool silently replaced by its namesake.
    """
    seen: dict = {}
    for t in tools:
        owner = getattr(t, "__self__", None)
        if owner is not None and _is_utility(owner):
            raise TypeError(f"{type(owner).__name__}.{t.__name__} is a method of a utility, not a tool — "
                            f"code calls it. {_utility_tools(owner)}")
        if _is_utility(t):
            raise TypeError(f"{type(t).__name__} is a utility, not a tool. {_utility_tools(t)}")
        if not (getattr(t, "name", None) and callable(getattr(t, "schema", None))
                and callable(getattr(t, "run", None))):
            kind = type(t).__name__
            hint = (" A Skill is one call to a model — a step for a Chain, not a tool for an Agent."
                    if isinstance(t, Skill) else "")
            raise TypeError(f"{kind} is not a tool: an Agent needs a name, schema() and run() — "
                            f"a Tool subclass, or a tool from a factory such as search('exa').{hint}")
        if t.name in _RESERVED:
            # The loop answers these names itself, so a tool of one never ran.
            raise ValueError(f"{t.name!r} is a name the agent's own tools use "
                             f"({', '.join(sorted(_RESERVED))}); give the tool another name")
        if t.name in seen:
            raise ValueError(f"two tools are named {t.name!r} ({type(seen[t.name]).__name__} and "
                             f"{type(t).__name__}); the model could call only one of them. Give one "
                             f"of them name=… to tell them apart — Vectors(name=…), "
                             f"Shell(prefix=…) and MCP(prefix=…) for a utility's tools.")
        seen[t.name] = t


class _Unset:
    """No signal given — distinct from a signal of None."""

    def __repr__(self) -> str:
        return "<required>"


_UNSET = _Unset()

#: Names the loop's synthetic tools hold.
_RESERVED = frozenset({"pool", "delegate", "write_plan"})

class Agent:
    """
    A loop that holds one growing conversation and acts on the world.

    Parameters
    ----------
    model : Model
        The model that drives the loop.
    tools : list[Tool], optional
        What it may call.
    instructions : str, optional
        The standing brief — goes into the stable prefix.
    mode : "agile" | "waterfall"
        ``"agile"`` decides at every step. ``"waterfall"`` writes a plan at
        step 0 (via the ``write_plan`` tool) and holds to it.
    team : list[Agent] | "auto" | None
        Who else may do work. ``None`` removes ``delegate`` from the tool set
        entirely; a list restricts it to those workers; ``"auto"`` lets the
        model describe new ones.
    planner_model : Model, optional
        ``waterfall`` only: the model that writes the plan. The freeze makes
        the plan the highest-leverage tokens of the run — everything after
        merely executes it — so it may deserve deliberation the execution
        turns do not need. Turns before the plan is recorded go to this model;
        every turn after goes to ``model``. Think while planning, act without
        deliberating.
    stop_when : list, optional
        Conditions from this module. Defaults to ``[step_count(30)]`` —
        a backstop, not a plan.
    hooks : list | None, optional
        Callables given an ``Event`` at every step boundary; ``stream()`` yields
        the same events.
    permissions : PermissionPolicy | None, optional
        Maps a tool's ``risk`` class to ``allow``, ``approve`` or ``deny``,
        checked before the tool runs. With none, every tool runs.
    approve : callable, optional
        Asked when the policy says ``approve``: given an ``ApprovalRequest``,
        returns ``True``, ``False`` or ``ApprovalDecision(False, reason)``.
        With ``approve`` in the policy and none attached, the call is refused.
    store : StateStore, optional
        Where a run parked by a ``wait()`` or ``gate()`` in the agent's tools
        is kept until ``resume()``. Required when such a tool is given.
    max_cost : float | Budget | None, optional
        A ceiling in dollars for the whole run; a model call that would begin
        past it ends the run with ``success=False`` and the reason in
        ``.error``.
    name : str | None, optional
        How this agent is named as a worker in another agent's ``team`` and in
        events.
    description : str | None, optional
        What it does — what a lead agent reads when choosing a worker.
    verbose : int, optional
        ``0`` is silent, ``1`` logs each step, ``2`` logs everything, through the
        ``yait_aichain`` logger.
    """

    MODES = ("agile", "waterfall")

    @_hints.refuse("Agent")
    def __init__(
        self,
        model:         Model,
        tools:         list | None = None,
        instructions:  str = "",
        mode:          str = "agile",
        team:          list[Agent] | str | None = None,
        planner_model: Model | None = None,
        stop_when:     list | None = None,
        hooks:         list | None = None,
        permissions:   PermissionPolicy | None = None,
        approve:       Callable[[ApprovalRequest], Any] | None = None,
        store:         StateStore | None = None,
        max_cost:      float | Budget | None = None,
        name:          str | None = None,
        description:   str | None = None,
        verbose:       int = 0,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}; got {mode!r}")
        if team is not None and team != "auto" and not isinstance(team, (list, tuple)):
            raise ValueError(
                "team must be None, a list of agents, or \"auto\"; "
                f"got {type(team).__name__}"
            )

        self.model        = model
        self.tools        = list(tools or [])
        _check_tools(self.tools)
        self.instructions = instructions
        self.mode         = mode
        self.team         = list(team) if isinstance(team, (list, tuple)) else team
        if planner_model is not None and mode != "waterfall":
            raise ValueError("planner_model applies to mode='waterfall' only — "
                             "no other mode has a planning phase")
        self.planner_model = planner_model
        self.stop_when    = list(stop_when) if stop_when is not None else [step_count(30)]
        if not any(getattr(c, "kind", "ceiling") == "ceiling" for c in self.stop_when):
            # `stop_when=[check(...)]` replaced the default step_count(30), and
            # a check that never passes ran the loop without end.
            warnings.warn("stop_when has no ceiling, so step_count(30) is added; give one of "
                          "your own — step_count, token_budget, cost_budget — to choose",
                          stacklevel=3)
            self.stop_when.append(step_count(30))
        if any(getattr(c, "name", "") == "cost_budget" for c in self.stop_when):
            from ..models._usage import _price_of
            if _price_of(getattr(model, "name", ""), getattr(model, "_provider", None)) is None:
                warnings.warn(f"cost_budget on {getattr(model, 'name', model)!r}, which has no "
                              "price: its cost is never known, so the budget never fires",
                              stacklevel=3)
        self.hooks        = list(hooks or [])

        #: The result of the last :meth:`stream`, once the generator has run
        #: to the end. `run()` returns its result; a generator cannot, so it
        #: is left here rather than mixed into a stream of events.
        self.last_result: AgentResult | None = None

        # `Event.run_id` has been a declared field since M3 and the agent
        # never filled it. Without it two concurrent invocations — which is
        # the ordinary case for the serverless target — write into one stream
        # that cannot be demultiplexed afterwards.
        self._run_id: str | None = None

        # Whether a consumer is walking this run. Set for the duration of
        # `stream()` and never by `run()`: streaming costs the fallback chain,
        # and paying that for an observer who cannot see the pieces anyway
        # would be trading reliability for nothing.
        self._streaming = False

        #: The approval request between going out and being answered.
        self._pending = None
        self.permissions  = permissions
        # Who answers when the policy says "approve". A callable taking an
        # ApprovalRequest and returning truthy to proceed. The library cannot
        # ask a human; it can only make sure one is asked when a policy said
        # to. Absent, an `approve` decision refuses — see `_permit`.
        self.approve      = approve
        # Where a parked run is kept. A tool that declares `suspends` can end
        # a turn with "the world must answer this before I go on", and the
        # conversation — which is the whole of an agent's state — has to
        # survive until it does. Demanded at construction rather than at the
        # pause, because a pause nobody can resume is otherwise discovered at
        # the worst possible moment: after the model asked.
        self.store        = store
        if (parked := [t.name or type(t).__name__ for t in self.tools
                       if getattr(t, "suspends", False)]) and store is None:
            raise ValueError(
                f"tool(s) {parked} can park this run and there is nowhere to "
                "keep it. Pass store=InMemoryStore() to resume inside this "
                "process, or store=FileStore(path) — or your own StateStore — "
                "to resume from another one. A serverless invocation cannot "
                "hold the wait, so the run has to outlive it.")
        #: The document id while a parked run is carried, so a second pause
        #: overwrites the first instead of leaking a second run id.
        self._doc_id: str | None = None
        # A ceiling in money for everything this agent spends, shared with the
        # Skills it builds and with any worker it delegates to. `cost_budget`
        # in `stop_when` stops the loop between turns; this stops a call from
        # beginning, and the two compose — one is a decision, the other a
        # limit.
        self.max_cost = as_budget(max_cost)
        self._own_budget = self.max_cost        # see _budget.start_run
        self.name         = name
        self.description  = description
        self.verbose      = verbose
        # How deep in a delegation this agent sits, and how deep it may go.
        # Not parameters: only `delegate` sets them, on the worker it makes.
        self._depth       = 0
        self._max_depth   = 5

        self._tool_map   = {t.name: t for t in self.tools}
        self._worker_map = ({(getattr(w, "name", None) or f"worker{i}"): w
                             for i, w in enumerate(self.team)}
                            if isinstance(self.team, list) else {})
        if isinstance(self.team, list) and len(self._worker_map) != len(self.team):
            names = [getattr(w, "name", None) or f"worker{i}" for i, w in enumerate(self.team)]
            raise ValueError(f"two workers share a name in {names}; delegate picks a worker by "
                             "name, and the model could reach only one of them")

    # ── Public API ───────────────────────────────────────────────────────

    def __getattr__(self, name):
        raise _hints.missing(self, name)

    def run(self, task: str, variables: dict | None = None) -> AgentResult:
        """
        Execute *task* and return an :class:`AgentResult`.

        *variables* seeds the conversation with data the caller already holds —
        this is how a ``Chain`` or ``Pool`` step hands its accumulated values
        down. There is one place state lives, and it is the conversation.

        Parameters
        ----------
        task : str
            What to do, in words.
        variables : dict | None, optional
            Data the caller already holds, given to the model with the task.
        """
        return self._fork()._run(task, variables)

    def _run(self, task: str, variables: dict | None = None) -> AgentResult:
        journal, messages, state = self._begin(task, variables)
        self._emit("run.started", payload={"task": task, "mode": self.mode})
        with collecting_beacons() as board:
            result = self._loop(messages, state, journal)
        return self._finished(result, journal, board)

    def stream(self, task: str, variables: dict | None = None):
        """
        Run *task* and yield each :class:`~.._events.Event` as it happens.

        ``run()`` is untouched and returns the same result it always did;
        this is the same loop walked instead of exhausted. Three things are
        worth knowing before reaching for it:

        * **Events, not tokens.** A turn in an agent loop is usually a tool
          call, not prose, so token deltas would be empty for most of a run
          and would arrive interleaved with decisions in no useful order.
          What a caller actually wants to show is what the agent is *doing* —
          which is what the event channel already carries. `Skill.stream` is
          the one for text.
        * **The result is not yielded**, because a stream of one type is
          easier to consume than a stream of two. It lands on
          ``last_result`` when the generator finishes, with the journal
          attached, exactly as ``run()`` would have returned it.
        * **Abandoning the generator abandons the run.** A `break` out of the
          loop leaves the agent mid-turn; the temporary hook is still removed
          (that much is guaranteed), but no result is produced and no
          `run.finished` is emitted. Exhaust it, or accept that.

        Parameters
        ----------
        task : str
            What to do, in words.
        variables : dict | None, optional
            Data the caller already holds, given to the model with the task.
        """
        run = self._fork()
        self.last_result = None
        # A context of its own, copied when the stream is made — not when it
        # is first read. A generator runs in whichever context advances it:
        # Starlette advances each step in a fresh copy, so the beacon board
        # was gone from the second step on; and two streams advanced in turn
        # in one thread set and reset the same variables out of order.
        from .._bound import bound
        return bound(run._stream(task, variables),
                     on_end=lambda: setattr(self, "last_result", run.last_result))

    def _stream(self, task: str, variables: dict | None = None):
        journal, messages, state = self._begin(task, variables)
        self.last_result = None

        sink: list = []
        # Appended for the duration and removed in `finally`: a hook left
        # behind on the instance would go on collecting into a list nobody
        # drains, which is a leak that only shows up under load.
        collect = sink.append          # bound once, so `remove` finds it
        self.hooks.append(collect)
        self._streaming = True
        try:
            with collecting_beacons() as board:
                self._emit("run.started", payload={"task": task, "mode": self.mode})
                while sink:
                    yield sink.pop(0)
                result = yield from self._loop_events(messages, state, journal, sink=sink)
                result = self._finished(result, journal, board)
            while sink:
                yield sink.pop(0)
            self.last_result = result
        finally:
            self._streaming = False
            with contextlib.suppress(ValueError):    # already gone; fine
                self.hooks.remove(collect)

    def arm(self, reason: str = "", resume_with: dict | None = None, *,
            hint: dict | None = None,
            variables: dict | None = None) -> SuspendedResult:
        """
        Create the run **without starting it**, and return it parked.

        The run exists before it has a task: a ``run_id``, a document in the
        store, and a declared shape for the signal that will start it. That is
        a subscription — what a trigger waits on, and what a person is shown a
        form for — and it is the same object a pause mid-run produces, so
        "an email arrived", "the human pressed start" and "the gate was
        approved" all arrive as ``resume(run_id, signal)``.

        Nothing is scheduled here. ``hint`` is recorded for whatever does the
        waking: ``{"on": "email.received"}``, ``{"wake_at": "..."}``.

        The signal carries the task: a string is the task, a dict gives it
        under ``"task"`` and every other key seeds the conversation the way
        ``run(task, variables=...)`` does.

        Parameters
        ----------
        reason : str, optional
            What the run waits for, shown to whoever starts it.
        resume_with : dict | None, optional
            The shape of the signal that starts it.
        hint : dict | None, optional
            Recorded for whatever does the waking; the engine schedules nothing.
        variables : dict | None, optional
            Data the run starts with, beside what the signal brings.
        """
        if self.store is None:
            raise ValueError(
                "arming needs somewhere to keep the run: pass "
                "Agent(store=FileStore('runs/')) — or InMemoryStore() when the "
                "signal arrives in this same process.")

        awaiting = {"reason": reason, "resume_with": dict(resume_with or {}),
                    "hint": hint, "at": "start"}
        doc = self._save_parked(
            {"messages": [], "journal": [],
             "state": self._portable_state(self.new_state()),
             "variables": dict(variables or {})},
            awaiting)
        self._emit("run.armed", payload={"run_id": doc.run_id,
                                         "agent": self.name or "", **awaiting})
        return SuspendedResult(run_id=doc.run_id, awaiting=awaiting,
                               document=doc.to_dict())

    def resume(self, run_id: str, signal=_UNSET) -> AgentResult | SuspendedResult:
        """
        Continue a run this agent parked, delivering *signal* as the answer.

        The pause and this call are the same seam the externally driven path
        already has: a parked run is a history whose last tool call has no
        result, and *signal* is that result. Nothing is restored that was not
        written down — the conversation **is** the state, which is why 2.0
        removed the agent's snapshot mechanism and why none is needed here.

        Idempotent in the way a trigger needs: a ``run_id`` the store no
        longer holds raises ``KeyError``, so an at-least-once delivery can
        read that as "already handled" instead of running the tail twice.

        Returns an :class:`AgentResult` when the run finishes, or another
        :class:`~..state.SuspendedResult` when it parks again — under the same
        ``run_id``, so a caller tracks one identifier for the whole
        conversation however many times it waits.

        Parameters
        ----------
        run_id : str
            The parked run, as the ``SuspendedResult`` named it.
        signal : any, optional
            The answer the run waited for — the result of the tool call that
            parked it. For a run made by ``arm()``, a string is the task, or a dict
            gives it under ``"task"``.
        """
        if signal is _UNSET:
            # A run kept after a failure is retried with no answer; a run
            # waiting for one is not resumed without it — taken without an
            # answer, the parked question was gone: the run paused again with
            # nobody to ask and its document deleted.
            waiting = ((((self.store.load(run_id) if self.store is not None else None) or {})
                        .get("steps") or [{}])[0].get("suspend") or {})
            if waiting and waiting.get("at") != "after":
                raise TypeError("resume needs the answer the run waited for: "
                                "agent.resume(run_id, signal=…)")
            signal = None
        return self._fork()._resume(run_id, signal)

    def _resume(self, run_id: str, signal=None) -> AgentResult | SuspendedResult:
        from ..state import RunDocument

        if self.store is None:
            raise ValueError(f"agent {self.name or ''!r} has no store, so it "
                             "never parked a run. Pass Agent(store=...).")
        # Claimed, not read: a second delivery of the same signal finds
        # nothing and gets the KeyError below, where loading then deleting at
        # the end let two deliveries both run the approved action.
        raw = self.store.take(run_id)
        if raw is None:
            raise KeyError(
                f"no suspended run {run_id!r} in the store — it has already "
                "been resumed, or it was never parked here.")

        doc = RunDocument.from_dict(raw)
        # The tenant the run was parked under, restored: the signal may arrive
        # in another process, and a key resolved from the context there must
        # be the parked run's tenant's, not whoever happens to be current.
        context = RunContext.from_dict(doc.context) if doc.context else None
        with using(context):
            return self._resume_doc(doc, run_id, signal)

    def _resume_doc(self, doc, run_id: str, signal):
        parked   = doc.steps[0].get("suspend") or {}

        if parked.get("at") == "start":
            # Armed, never run: there is no tool to complete and no history to
            # continue. The signal is the task, and this is an ordinary run
            # that happens to have been waiting for its input.
            task, seeded = self._task_from_signal(signal, parked, doc)
            journal, messages, state = self._begin(task, seeded)
            self._doc_id = run_id
            self._emit("run.started", payload={"task": task, "mode": self.mode,
                                               "run_id": run_id})
            with collecting_beacons() as board:
                result = self._loop(messages, state, journal)
            if _interrupted(result):
                # Nothing has acted yet, so the run is armed again as it was.
                self.store.save(run_id, doc.to_dict())
            return self._finished(result, journal, board)

        messages = list(doc.variables.get("messages") or [])
        journal  = Journal.from_list(doc.variables.get("journal"))
        state    = self.new_state()
        state.update(doc.variables.get("state") or {})
        start_run(self, spent=state.get("cost") or 0.0)  # the parked part was spent already

        self._run_id = doc.variables.get("run_id") or uuid.uuid4().hex
        self._doc_id = run_id
        call_id      = parked.get("call_id") or ""
        if parked.get("at") == "after":
            # Interrupted after the parked call was answered: its result is in
            # the history, so the run continues from there and nothing is
            # completed — or charged, or refunded — a second time.
            self._emit("run.resumed", step=state.get("steps"),
                       payload={"run_id": run_id, "after": parked.get("reason", ""),
                                "agent": self.name or ""})
            return self._continue(messages, state, journal, run_id)
        self._emit("run.resumed", name=parked.get("tool"),
                   step=state.get("steps"),
                   payload={"run_id": run_id, "tool": parked.get("tool"),
                            "signal": signal, "agent": self.name or ""})

        # The parked tool is *completed*, not bypassed: it is re-invoked with
        # the signal, exactly as a Chain re-invokes its suspended step. For a
        # `Wait` that returns the signal; for a `Gate` it is where the guarded
        # action finally runs, or does not. Answering the call from here
        # instead would make every gate a rubber stamp.
        result, error = self._complete(parked, signal)
        messages.append(tool_result_turn(call_id,
                                         prompts.result_message(result, error)))
        journal.append(
            "",
            action      = _safe({"name": parked.get("tool"),
                                 "arguments": parked.get("arguments") or {}}),
            outcome     = _J_FAILED if error else _J_DONE,
            evidence    = _evidence(CHECK, error or
                                    "resumed with an external signal"),
            reason      = error or "",
            observation = prompts.observation_text(result, error),
            step        = state.get("steps"),
        )
        self._emit("tool_call.ended", name=parked.get("tool"),
                   step=state.get("steps"),
                   payload={"id": call_id, "result": result,
                            "agent": self.name or ""},
                   error=error)
        return self._continue(messages, state, journal, run_id)

    def _continue(self, messages, state, journal, run_id: str):
        """The rest of a resumed run. A failure the run can be retried from —
        the model's call failing, a reply cut off — keeps the run in the store
        with what has happened so far, the parked action's result included, so
        a retry picks up after it instead of losing a run a person approved."""
        with collecting_beacons() as board:
            result = self._loop(messages, state, journal)
        if _interrupted(result):
            self._save_parked(
                {"messages": messages, "state": self._portable_state(state),
                 "journal": journal.to_list(), "run_id": self._run_id},
                {"at": "after", "reason": result.error or result.stopped_by or ""},
                run_id=run_id)
            self._doc_id = None              # kept on purpose: `_finished` must not delete it
        return self._finished(result, journal, board)

    # ── Externally driven: one turn at a time ────────────────────────────
    #
    # ``run()`` owns the loop. Some callers own it instead — a serverless
    # invocation that must be one step, or a harness that executes the tools
    # itself. The same machinery, exposed.

    def new_state(self) -> dict:
        """Fresh run state — the counters ``stop_when`` conditions read."""
        return {"steps": 0, "tokens": 0, "cost": None, "plan": None,
                "adaptations": []}

    def opening(self, task: str, variables: dict | None = None,
                *, ground_rules: bool = False) -> list:
        """
        The two messages a run starts from: the stable prefix and the task.

        ``ground_rules`` injects the "plain text ends the run" rule, and it
        belongs to whoever owns the loop: ``run()`` passes True because in its
        loop a text reply really does end the run. An external driver calling
        ``opening()``/``step()`` owns its own loop — there a plain reply is an
        ordinary turn, so the default is clean instructions and the driver
        states its own framing.

        Parameters
        ----------
        task : str
            What to do, in words.
        variables : dict | None, optional
            Data the caller already holds, given with the task.
        ground_rules : bool, optional
            Add the rule that a plain-text reply ends the run. ``run()`` sets it;
            a driver that owns its own loop usually does not.
        """
        opening = task
        if variables:
            # Without its secrets: a token a chain carried for a tool went to
            # the model's provider in this block.
            from ..state._secret import public
            opening = (f"{task}\n\nGIVEN:\n"
                       + json.dumps(public(variables), indent=1, ensure_ascii=False,
                                    default=str))
        return [
            {"role": "system",
             "parts": [prompts.system_message(self.instructions, self.mode,
                                              ground_rules=ground_rules)]},
            {"role": "user", "parts": [opening]},
        ]

    def step(self, messages: list, state: dict | None = None):
        """
        Ask for the next decision and return it **without executing it**.

        Returns a :class:`~models._calls.ToolCallRequest` when the model asked
        to act, or a string — the answer. The caller appends the reply (via
        ``.as_turn()``) and whatever the world answered, then calls again.

        **The obligation in that sentence is now checked.** A history holding
        a tool call nobody answered is malformed, and the providers disagree
        about it in the worst way: three reject the request, Google is looser,
        and a self-hosted OpenAI-compatible server validates nothing and
        templates it through, so the model meets its own unanswered call and
        improvises — differently each time. Provider interchangeability is
        this library's main promise and that is a place it did not hold.

        Raising here buys three things and costs one. It makes the providers
        agree; it turns an intermittent, unbounded behaviour into the same
        named failure every time, since the condition that produces a
        dangling call is itself intermittent; and it fails locally, before
        the network, pointing at the caller rather than at a provider's
        wording. What it does not buy is a better score — on a benchmark a
        crash can rank below a model muddling through. This is an instrument
        for reliability, not for accuracy.

        Parameters
        ----------
        messages : list
            The conversation so far — ``opening()``'s two messages, then each
            reply and each tool result in turn.
        state : dict | None, optional
            The run's working state between calls, kept by the caller; the plan in
            ``waterfall`` mode lives here.
        """
        if unanswered := dangling_calls(messages):
            raise ValueError(
                f"agent {self.name or ''!r}: the history has tool call(s) "
                f"{unanswered} with no result. Append "
                "`tool_result_turn(call.id, result)` for every call in the "
                "previous reply — failed ones included, with the error as "
                "the result — before calling step() again. `run()` does this "
                "for you; an external driver owns it.")
        state = state if state is not None else self.new_state()
        reply = self._ask(messages, state)
        if isinstance(reply, ToolCallRequest):
            # Counted here as run() counts it: a driver on this seam had a
            # step_count that never moved, and so no ceiling at all.
            state["steps"] = (state.get("steps") or 0) + 1
        return reply

    def stopped(self, state: dict) -> str | None:
        """The name of the terminal condition that holds, or None — what
        ``run()`` asks after every turn, for a driver of ``step()`` to ask too.

        Parameters
        ----------
        state : dict
            The state passed to ``step()``.
        """
        fired = self._fired(state)
        return fired[0] if fired else None

    def nudges(self, state: dict) -> list:
        """The messages of nudges that fire now, once per streak — what
        ``run()`` appends as user turns, for a driver of ``step()`` to append.

        Parameters
        ----------
        state : dict
            The state passed to ``step()``.
        """
        return self._nudges(state)

    def execute(self, call, state: dict | None = None):
        """Run one :class:`ToolCall`. Returns ``(result, error)``: a tool that
        fails is an error, not an exception, and so is a call the permission
        policy refuses — the same policy, approver included, that ``run()``
        applies, because this is the seam a serverless driver runs tools
        through and a policy it skipped would govern nothing there. The one
        exception that passes is ``Suspend`` — a ``wait()`` or ``gate()``
        asking the run to park, which the driver handles as a pause.

        Parameters
        ----------
        call : ToolCall
            One call from a ``ToolCallRequest`` that ``step()`` returned.
        state : dict | None, optional
            The same working state passed to ``step()``.
        """
        return self._fork()._execute_gated(call, state)

    def _execute_gated(self, call, state: dict | None = None):
        # The permission gate keeps its pending request on the instance between
        # asking and answering; a fork keeps two drivers' requests apart.
        state = state if state is not None else self.new_state()
        denial = self._ask_permission(call, state)
        if denial is _AWAITING:
            denial = self._resolve_permission(state)
        if denial:
            return None, denial
        return self._execute(call, state)

    # ── The loop ─────────────────────────────────────────────────────────

    def _fork(self) -> Agent:
        """This agent for one run: the same configuration, its own run state.

        A run keeps its id, its parked document, its pending approval, its
        stream flag and the hooks it emits to; all of them lived on the agent,
        and one agent serves concurrent runs — a module-level agent in a warm
        Lambda, a Pool's items, a `delegate` worker. Approvals crossed (a call
        approved for one run executed in another), one tenant's stream carried
        another's events, and a run read another's id. A shallow copy keeps the
        configuration shared — tools, model, policy, store, a shared `Budget` —
        and gives each run the rest.
        """
        run = copy.copy(self)
        run.hooks = list(self.hooks)
        run._run_id = run._doc_id = run._pending = None
        run._streaming = False
        run.last_result = None
        return run

    def _begin(self, task: str, variables: dict | None):
        """A run's first moments, for `run()` and `stream()` alike: a fresh
        journal, the opening messages, working state, a new run id — and no
        document of an earlier run, which a finishing run would otherwise
        delete or overwrite."""
        self._run_id = uuid.uuid4().hex
        self._doc_id = None
        start_run(self)
        return Journal(), self.opening(task, variables, ground_rules=True), self.new_state()

    def _loop(self, messages: list, state: dict, journal: Journal) -> AgentResult:
        """Drive the loop to its end and return the result.

        One loop, driven two ways. ``run()`` exhausts it and takes the value;
        ``stream()`` walks it and hands each event on as it appears. A second
        copy of this loop for the streaming case is exactly the mistake 2.0
        undid when `Agent` stopped hand-rolling its own call path — two paths
        do not stay the same, and the one nobody watches is where the defect
        lives.
        """
        walk = self._loop_events(messages, state, journal)
        try:
            while True:
                next(walk)
        except StopIteration as done:
            return done.value

    def _loop_events(self, messages: list, state: dict, journal: Journal,
                     sink: list | None = None):
        """The loop itself, yielding what it has emitted as it goes.

        Events are not produced twice. They go out through ``_emit`` exactly
        as they always have — every hook a caller installed still fires, in
        the same order — and a streaming caller adds one more hook that
        appends to *sink*, which is drained at each boundary. So the stream is
        a view of the existing event channel rather than a second vocabulary
        beside it.

        The granularity is the turn, not the token: everything a turn emitted
        is handed over when the turn is done. Within one model call there is
        nothing to interleave, because the loop is synchronous by design —
        no threads, no async, because the target is Lambda.
        """
        def drain():
            while sink:
                yield sink.pop(0)

        while True:
            try:
                # Walked, not called: the text deltas of a streamed turn reach
                # the consumer as they arrive, where they were handed over in
                # one batch after the model finished.
                asking = self._ask_events(messages, state)
                while True:
                    try:
                        next(asking)
                    except StopIteration as done:
                        reply = done.value
                        break
                    yield from drain()
            except Exception as exc:
                yield from drain()
                # A reply cut at the token ceiling is its own stop: the fix is
                # a larger max_tokens, not a retry or a different model.
                cut = isinstance(exc, TruncatedResponseError)
                # Its own stop: the history is longer than the model can read,
                # and that is not the same failure as a revoked key.
                full = isinstance(exc, ContextWindowExceededError)
                return self._result(state, False, None,
                                    "truncated" if cut else "context_overflow" if full else "llm_error",
                                    error=f"{type(exc).__name__}: {exc}")
            yield from drain()

            # Text is the answer — one exit, not two spellings of it — unless
            # a `verified()` gate holds it once for a check.
            if not isinstance(reply, ToolCallRequest):
                if state.get("truncated"):
                    # Cut at max_tokens: kept as the output, because half an
                    # answer can be worth reading, but not called an answer.
                    # It ended the run as `answered`, `success=True`.
                    self._log(1, "\n[Done] ✗ the answer was cut at max_tokens")
                    return self._result(state, False, str(reply), "truncated",
                                        error="the answer was cut at max_tokens; raise max_tokens "
                                              "for the whole of it")
                held = self._verify_gate(state)
                if held is None:
                    self._log(1, f"\n[Done] ✓ answered · {state['steps']} step(s)")
                    return self._result(state, True, str(reply), "answered")
                messages.append({"role": "assistant", "parts": [{"type": "text", "text": str(reply)}]})
                messages.append({"role": "user", "parts": [{"type": "text", "text": held}]})
                yield from drain()
                continue

            state["steps"] += 1
            messages.append(reply.as_turn())

            # Providers may request several calls in one turn; all of them run,
            # each with its own result turn and journal entry. Executing only
            # the first would silently narrow the channel.
            for call in reply.calls:
                # Everything a program needs to rebuild this call, rather
                # than a description of it. `id` is what tells two calls in
                # one turn apart — a provider may request several, and the
                # agent honours all of them — and the raw result is the only
                # form in which "no rows" and "the tool declined" are still
                # different things. `prompts.observation_text` is the model's
                # view and stays the model's view.
                # A copy: a hook that masks a token in the payload changed
                # the argument the tool then ran with.
                self._emit("tool_call.started", name=call.name,
                           step=state["steps"],
                           payload={"id": call.id or "",
                                    "arguments": _copy(call.arguments or {}),
                                    "agent": self.name or ""})
                yield from drain()

                denial = self._ask_permission(call, state)
                if denial is _AWAITING:
                    # The request reaches the consumer *here*, before anyone
                    # is asked — which is the whole point of splitting it.
                    yield from drain()
                    try:
                        denial = self._resolve_permission(state)
                    except Exception as exc:
                        messages.append(tool_result_turn(call.id, prompts.result_message(
                            None, f"approval failed: {type(exc).__name__}: {exc}")))
                        yield from drain()
                        return self._callback_failed(state, "the approver", exc)
                    yield from drain()

                try:
                    result, error = ((None, denial) if denial
                                     else self._execute(call, state))
                except Suspend as susp:
                    try:
                        parked = self._park(susp, call, reply, messages,
                                            state, journal)
                    except Exception as exc:
                        # The store refused the document — a DynamoDB item
                        # over 400 KB after twenty calls. run() raised it raw
                        # and the paid conversation was gone; it comes back
                        # as a result, history and cost with it.
                        yield from drain()
                        return self._result(state, False, None, "park_failed",
                                            error=f"the run could not be parked: "
                                                  f"{type(exc).__name__}: {exc}")
                    yield from drain()
                    return parked
                failure, self._failure = self._failure, None
                if not error:
                    # What a `verified()` gate reads: the last change not yet
                    # looked at. Any later call that succeeds clears it.
                    from ..tools._permissions import risk_of
                    acted = self._gated_tool(call)
                    if acted is not None and risk_of(acted, dict(call.arguments or {})) in _EFFECTS:
                        state["unverified"] = call.name
                    elif state.get("unverified"):
                        state["unverified"] = None
                self._emit("tool_call.ended", name=call.name,
                           step=state["steps"],
                           payload={"id": call.id or "", "result": _copy(result),
                                    "agent": self.name or "",
                                    # The type and the traceback, which the
                                    # error string had dropped.
                                    **({"exception": failure} if failure else {})},
                           error=error)
                yield from drain()

                journal.append(
                    # No intent: an intent is what an attempt was *for* — a plan
                    # step's goal — and this loop has none. The tool's name
                    # here would make ten different searches read as one
                    # search repeated to `repeating()`; the action carries it.
                    "",
                    action      = _action(call),
                    outcome     = _J_FAILED if error else _J_DONE,
                    # A call that *ran* is a fact, but what it returned is
                    # only as good as its source. `delegate` hands back a
                    # worker's own account and marks it MODEL_CLAIM; this
                    # recorded the call as CHECK regardless, so an unverified
                    # report entered the parent's journal as a programmatic
                    # fact — the distinction existed and decided nothing. A
                    # failure stays CHECK: that the call failed is verified.
                    evidence    = _evidence(
                        MODEL_CLAIM if (not error and isinstance(result, dict)
                                        and result.get("evidence") == MODEL_CLAIM)
                        else CHECK,
                        error or "executed"),
                    reason      = error or "",
                    # The journal is a written record; an image cannot go in
                    # it, so media is named rather than embedded.
                    observation = prompts.observation_text(result, error),
                    step        = state["steps"],
                )
                messages.append(tool_result_turn(
                    call.id, prompts.result_message(result, error), is_error=bool(error)))

            state["journal"] = journal
            try:
                fired = self._fired(state)
            except Exception as exc:
                yield from drain()
                return self._callback_failed(state, "a stop condition", exc)
            if fired:
                kind = getattr(fired[1], "kind", "ceiling")
                if kind == "check":
                    self._log(1, f"\n[Done] ✓ {fired[0]} · {state['steps']} step(s)")
                    return self._result(state, True, None, fired[0])
                self._log(1, f"\n[Stop] ✗ {fired[0]} · {state['steps']} step(s)")
                self._note_unfinished(journal, state, fired[0])
                return self._result(
                    state, False, None, fired[0],
                    error=f"stopped by {fired[0]} after {state['steps']} step(s)")

            # After the terminal conditions, never before: a ceiling that has
            # been reached ends the run whatever a nudge would have said, so a
            # nudge listed first in `stop_when` cannot talk the loop past its
            # own budget.
            try:
                notes = self._nudges(state)
            except Exception as exc:
                yield from drain()
                return self._callback_failed(state, "a nudge", exc)
            for note in notes:
                messages.append({"role": "user",
                                 "parts": [{"type": "text", "text": note}]})
                yield from drain()

    _failure: dict | None = None

    def _callback_failed(self, state: dict, where: str, exc: BaseException) -> AgentResult:
        """One handling for every caller-supplied callback that raises — a check,
        a ceiling, a nudge, an approver. They were handled three ways: a check
        swallowed it, a ceiling and a nudge raised out of run(), and so did an
        approver, with no result and no `run.finished`."""
        _logger.warning("%s raised; the run stops", where, exc_info=exc)
        return self._result(state, False, None, "condition_error",
                            error=f"{where} raised {type(exc).__name__}: {exc}")

    def _fired(self, state: dict):
        """The first terminal condition that fired. Nudges are not terminal."""
        for cond in self.stop_when:
            if getattr(cond, "kind", "ceiling") in ("nudge", "verify"):
                continue
            if fired := cond(state):
                return fired, cond
        return None

    def _verify_gate(self, state: dict) -> str | None:
        """The message a `verified()` gate holds an answer for, or None."""
        latched = state.get("nudged")
        if not isinstance(latched, set):
            latched = state["nudged"] = set(latched or ())
        for i, cond in enumerate(self.stop_when):
            if getattr(cond, "kind", None) != "verify":
                continue
            key = f"{i}:{cond.name}"
            if key in latched or not cond(state):
                continue
            latched.add(key)
            message = cond.message(state) if callable(cond.message) else cond.message
            self._emit("nudge.fired", step=state.get("steps"),
                       payload={"name": cond.name, "message": message})
            return message
        return None

    def _nudges(self, state: dict) -> list:
        """Messages from nudges that fired this turn — once per streak."""
        # Keyed by place, not name: two nudges of one name — every nudge is
        # "nudge" unless named, `stalled(3)` beside `stalled(8)` — shared one
        # latch, and the second never spoke. A list after a resume.
        latched = state.get("nudged")
        if not isinstance(latched, set):
            latched = state["nudged"] = set(latched or ())
        notes = []
        for i, cond in enumerate(self.stop_when):
            if getattr(cond, "kind", None) != "nudge":
                continue
            key = f"{i}:{cond.name}"
            if not cond(state):
                latched.discard(key)              # cleared: re-arm
                continue
            if key in latched:
                continue                          # said already, still true
            latched.add(key)
            message = (cond.message(state) if callable(cond.message)
                       else cond.message)
            self._emit("nudge.fired", step=state.get("steps"),
                       payload={"name": cond.name, "message": message})
            notes.append(message)
        return notes

    # ── One model call ───────────────────────────────────────────────────

    def _tool_schemas(self) -> list:
        """Real tools plus the synthetic ones this configuration earns."""
        schemas = [t.schema() for t in self.tools]
        if self.tools:
            schemas.append(_POOL_SCHEMA)
        if self.team:
            schemas.append(_delegate_schema(self.team))
        if self.mode == "waterfall":
            schemas.append(_PLAN_SCHEMA)
        return schemas

    def _ask(self, messages: list, state: dict):
        """One turn, to its end — for ``step()``, which has nobody to stream to."""
        asking = self._ask_events(messages, state)
        while True:
            try:
                next(asking)
            except StopIteration as done:
                return done.value

    def _ask_events(self, messages: list, state: dict):
        """
        One turn of the conversation, through ``Skill``, yielding after each
        event it emits so a stream can hand it on.

        Delegating instead of hand-rolling the call is what makes cost,
        caching and native tool calling arrive at all: they were added to the
        primitive and would never reach a private copy of it.
        """
        _t0 = time.monotonic()
        # The planning phase, when it exists and has its own model, deserves
        # deliberation; execution turns do not. The phase boundary is the
        # recorded plan.
        model = (self.planner_model
                 if self.planner_model is not None and state.get("plan") is None
                 else self.model)
        skill = Skill(
            model  = model,
            input  = {"messages": messages},
            # Stamped, not passed through: a Skill emits `llm_call.*` and
            # knows nothing of the run it is inside, so those events reached
            # the stream with no run_id and a consumer could not tell them
            # from another invocation's. Identity belongs to whoever owns the
            # run, and that is the agent.
            hooks  = self._stamping_hooks(),
            max_cost = self.max_cost,
            _tools = self._tool_schemas(),
            # Transient provider failures (rate limit / 5xx / network) retry
            # inside the call instead of costing the whole turn. The reference
            # agent this design was measured against retries every call; a
            # loop that doesn't pays for each blip with a lost decision.
            max_retries = 2,
        )
        reply = yield from self._reply(skill)
        # In the state, where both exits read it: `run()` ends on it, and a
        # driver of `step()` gets a string back and needs to know it is half.
        state["truncated"] = bool(getattr(skill, "last_truncated", False))
        # Kept per run, deduplicated: a loop asks the same thing of the same
        # model every turn, so without this the list grows by a copy a step.
        for a in skill.last_adaptations:
            # As plain data: the state of the step() seam is documented as
            # JSON-serialisable, and a dataclass in it broke `json.dumps` on
            # the first adaptation — and with `default=str` the `not in`
            # check never matched, so the list grew every step.
            entry = dataclasses.asdict(a) if dataclasses.is_dataclass(a) else a
            if entry not in state["adaptations"]:
                state["adaptations"].append(entry)
        usage = skill.last_usage
        state["tokens"] += getattr(usage, "total_tokens", 0) or 0
        if (c := getattr(usage, "cost", None)) is not None:
            state["cost"] = (state["cost"] or 0.0) + c
        self._log(2, f"     +{getattr(usage, 'total_tokens', 0):,} tokens "
                     f"({time.monotonic() - _t0:.1f}s)")
        return reply

    # ── Executing one call ───────────────────────────────────────────────

    def _execute(self, call, state: dict) -> tuple[Any, str | None]:
        """Run one call: ``(result, error)``, or ``Suspend`` raised for a pause."""
        try:
            if call.name == "pool":
                return self._do_pool(call.arguments, state), None
            if call.name == "delegate":
                return self._do_delegate(call.arguments, state), None
            if call.name == "write_plan":
                return self._do_plan(call.arguments, state), None
            return self._do_tool(call, state), None
        except Suspend:
            # Not a failure, and the one exception this must not flatten: the
            # tool is saying the world has to answer before the run goes on.
            # Flattened into a tool result, the pause would read to the model
            # as an error and the run would carry on as if nothing was owed.
            raise
        except Exception as exc:
            from .._errors_policy import describe
            self._failure = describe(exc)
            return None, f"{type(exc).__name__}: {exc}"

    def _do_tool(self, call, state: dict | None = None) -> Any:
        if _depth(call.arguments) > MAX_ARGUMENT_DEPTH:
            # Told to the model, which made it: 450 levels raised RecursionError
            # copying the history on the next turn and ended the run unseen.
            raise ValueError(f"the arguments are nested more than {MAX_ARGUMENT_DEPTH} levels deep; "
                             "send a flatter structure")
        tool = self._tool_map.get(call.name)
        if tool is None:
            raise ValueError(f"no tool named {call.name!r}. "
                             f"Available: {sorted(self._tool_map)}")
        kwargs = call.arguments or {}
        # No permission check here: the loop resolved it before calling, so
        # a second one would put the question to a person twice.
        if (problem := tool.check_args(kwargs)) is not None:
            raise ValueError(problem)
        if self.verbose or _logger.isEnabledFor(logging.INFO):   # serialised only to be shown
            self._log(1, f"  ⚙  {tool.name}({_safe(kwargs)})")
        # Under this agent's budget and every one around it.
        with lend(self.max_cost):
            return charged(tool, lambda: tool.run(**kwargs))

    def _gated_tool(self, call):
        """The tool a call will run, which the policy decides on. For `pool`
        it is the tool the pool names — the fan-out runs that tool, once per
        item, and asking about `pool` itself asked about nothing. One approval
        covers the whole list, which the request carries as its arguments."""
        if call.name == "pool":
            return self._tool_map.get((call.arguments or {}).get("tool") or "")
        return self._tool_map.get(call.name)

    def _ask_permission(self, call, state: dict | None = None):
        """Consult the policy before a tool runs. Returns a denial, or None.

        **At the loop boundary, not inside the call**, and split from the
        answer: the loop hands `approval.requested` to a streaming consumer
        *before* the approver is asked, which is what lets a UI show a person
        the question. Inside the call, the request and the decision would
        arrive together, after the fact.

        Emitted even when there is nobody to ask. A consumer needs to see
        that a decision was required, and the refusing case is where that
        matters most: otherwise the call simply fails and nothing says it was
        governance rather than a broken tool.
        """
        tool = self._gated_tool(call)
        if tool is None or not self.permissions:
            return None
        if _depth(call.arguments) > MAX_ARGUMENT_DEPTH:
            # Refused before the approver's deep copy would raise on it.
            return (f"tool {tool.name!r}: the arguments are nested more than "
                    f"{MAX_ARGUMENT_DEPTH} levels deep; send a flatter structure")
        from ..tools._permissions import ALLOW, APPROVE, DENY, ApprovalRequest

        # The call's arguments are the pool's own for `pool`: the policy is asked
        # about the tool it fans out, which takes them one entry at a time.
        arguments = None if call.name == "pool" else dict(call.arguments or {})
        try:
            decision = self.permissions.decide(tool, arguments)
        except TypeError:
            decision = self.permissions.decide(tool)          # a policy of another make
        if decision == ALLOW:
            return None
        from ..tools._permissions import risk_of
        risk = risk_of(tool, arguments)
        if decision == DENY:
            return f"tool {tool.name!r} denied by policy"
        if decision != APPROVE:
            # A policy of another make answering a word this gate does not
            # know: it was read as "allowed". Anything but yes is no.
            return f"tool {tool.name!r}: the policy answered {decision!r}, which is not a decision"

        self._pending = ApprovalRequest(
            # A copy all the way down: an approver that edits what it was
            # shown would change the call it approved.
            tool=tool.name, risk=risk, arguments=copy.deepcopy(dict(call.arguments or {})),
            call_id=getattr(call, "id", "") or "", agent=self.name or "", run_id=self._run_id or "")
        self._emit("approval.requested", name=tool.name,
                   step=(state or {}).get("steps"),
                   payload={"id": self._pending.call_id, "tool": tool.name,
                            "risk": risk, "arguments": self._pending.arguments,
                            "agent": self.name or ""})
        return _AWAITING

    def _resolve_permission(self, state: dict | None = None):
        """Ask, then say what the answer was. Returns a denial, or None."""
        request = self._pending
        self._pending = None
        if self.approve is None:
            granted = False
            reason  = (f"no approver is attached. Pass Agent(approve=...) to "
                       f"decide per call, or set the policy rule for "
                       f"{request.risk!r} to 'allow' if this class does not "
                       "need gating here.")
        else:
            from ..tools._permissions import ApprovalDecision
            answer  = self.approve(request)
            # Only a yes that says yes: `bool("denied")` and
            # `bool({"approved": False})` both approved the call.
            if not isinstance(answer, (bool, ApprovalDecision)):
                raise TypeError(f"an approver answers True, False or ApprovalDecision(granted, "
                                f"reason); it answered {type(answer).__name__} {answer!r}")
            granted = bool(answer)
            # An `ApprovalDecision` carries why; a bare False does not, and a
            # UI showing "not approved" and nothing else has thrown away the
            # only part a person can act on.
            reason  = getattr(answer, "reason", "") or (
                "" if granted else "not approved")

        self._emit("approval.decided", name=request.tool,
                   step=(state or {}).get("steps"),
                   payload={"id": request.call_id, "tool": request.tool,
                            "risk": request.risk, "granted": granted,
                            "reason": reason, "agent": self.name or ""})
        if granted:
            return None
        # A refusal is a result, not a crash: every tool call coming back with
        # something is what keeps the model's history well-formed, and a
        # denied call the model never hears about is one it will simply make
        # again. `tool_call.ended` then carries this as its error, which is
        # the terminal event a consumer needs in order to stop waiting.
        head = ("needs approval and" if self.approve is None
                else "was not approved:")
        return f"tool {request.tool!r} ({request.risk}) {head} {reason}"

    def _do_plan(self, args: dict, state: dict) -> str:
        items = args.get("items") or []
        if not isinstance(items, list):
            items = [items]                  # one item, not a string split into characters
        if state["plan"] is not None:
            # "You may not rewrite it" — and a rewrite would also invalidate
            # the prefix the freeze exists to keep cacheable.
            return "The plan is already recorded and cannot be rewritten."
        state["plan"] = [str(i) for i in items]
        self._log(1, f"  ▣  plan: {len(items)} item(s)")
        return "Plan recorded. Begin."

    def _do_pool(self, args: dict, state: dict) -> Any:
        """
        One call repeated over a list.

        Every other call is exactly one execution. Without this, a model asked
        to "read every document" reads one and reports success — measured at
        2 of 8, silently.

        The pool is no way around the policy: the loop asked about the tool
        it names before this ran (see `_gated_tool`), and each item is checked
        against that tool's schema before any of them starts. Its spend is the
        run's — under `max_cost`, in `state`, and so in the result.
        """
        items = args.get("items")
        if not isinstance(items, (list, tuple)) or not items:
            raise ValueError("pool needs a non-empty 'items' list")
        if len(items) > MAX_POOL_ITEMS:
            raise ValueError(f"pool takes at most {MAX_POOL_ITEMS} items in one call; "
                             f"got {len(items)} — split the list")

        max_flows = max(1, min(int(args.get("max_flows") or 4), len(items), MAX_POOL_FLOWS))

        if args.get("tool"):
            tool = self._tool_map.get(args["tool"])
            if tool is None:
                raise ValueError(f"pool tool {args['tool']!r} not found. "
                                 f"Available: {sorted(self._tool_map)}")
            arg = args.get("arg") or next(
                iter(tool.parameters.get("properties", {})), "input")
            # A dict entry is a bundle of named arguments; wrapping it under
            # one name would pass the whole dict as a single argument and fail
            # every call on what it never received.
            pool_items = [it if (isinstance(it, dict) and not args.get("arg"))
                          else {arg: it} for it in items]
            for i, item in enumerate(pool_items):
                if (problem := tool.check_args(item)) is not None:
                    raise ValueError(f"pool item {i}: {problem}")
            runner = tool
        elif args.get("prompt"):
            runner = Skill(self.model, prompt=args["prompt"], max_cost=self.max_cost,
                           hooks=self._stamping_hooks(), max_retries=2)
            pool_items = [{"item": it} for it in items]
        else:
            raise ValueError("pool needs either 'tool' or 'prompt'")

        self._log(1, f"  ⇉  pool ×{len(items)} (max_flows={max_flows})")
        results = Pool(runner, items=pool_items, max_flows=max_flows,
                       on_error="collect", max_cost=self.max_cost).run()
        if results.usage is not None:
            state["tokens"] += results.tokens_used
            if results.cost is not None:
                state["cost"] = (state["cost"] or 0.0) + results.cost
        errors = [e for e in results.errors if e is not None]
        if len(errors) == len(pool_items):
            raise RuntimeError(f"all {len(pool_items)} fanned-out calls failed. "
                               f"First: {errors[0]}")
        if errors:
            self._log(1, f"  ⚠  {len(errors)}/{len(pool_items)} entries failed; "
                         f"first: {errors[0]}")
        # A list, which is what a model reads: the result object rendered to
        # its repr, `PoolResult(ok, items=3)`, told it nothing it had asked for.
        return [out if err is None else {"error": err}
                for out, err in zip(results.output, results.errors)]

    def _do_delegate(self, args: dict, state: dict) -> Any:
        """
        Hand a scoped sub-task down. First of all a **context-management**
        mechanism: the child burns its own conversation and returns one
        turn's worth of conclusion.
        """
        if not self.team:
            raise ValueError("this agent has no team; delegate is unavailable "
                             "unless team= is set")
        if self._depth >= self._max_depth:
            raise RuntimeError(f"maximum delegation depth {self._max_depth} reached")

        task = args.get("task") or ""
        if isinstance(self.team, list):
            worker = self._worker_map.get(args.get("worker", ""))
            if worker is None:
                raise ValueError(f"no worker named {args.get('worker')!r}. "
                                 f"Available: {sorted(self._worker_map)}")
            # A run of its own, one level down: a listed worker sat at depth 0
            # whatever delegated to it, and the depth guard never fired.
            worker = worker._fork()
            worker._depth, worker._max_depth = self._depth + 1, self._max_depth
        else:                                            # "auto"
            worker = Agent(
                # Without the tools that park a run: a worker's pause has
                # nowhere to go — `delegate` waits for an answer, not a resume.
                self.model, tools=[t for t in self.tools if not getattr(t, "suspends", False)],
                instructions=args.get("instructions", ""),
                stop_when=self.stop_when, hooks=self.hooks,
                # The approver travels with the policy: a worker that
                # inherited the rules and not the answerer would refuse every
                # gated call, which reads as the policy being stricter for
                # children than for their parent.
                permissions=self.permissions, approve=self.approve,
                max_cost=self.max_cost,
                verbose=self.verbose,
            )
            worker._depth, worker._max_depth = self._depth + 1, self._max_depth

        self._log(1, f"  ▽  delegate → {getattr(worker, 'name', None) or 'worker'}")
        child = worker.run(task)
        state["tokens"] += child.tokens_used
        if child.cost is not None:
            state["cost"] = (state["cost"] or 0.0) + child.cost
        # What the child says it did is the child's word for it. Delegation
        # multiplies the places where "it says it did" stands in for "we
        # checked"; naming the class keeps them countable.
        return {"worker": getattr(worker, "name", None) or "worker",
                "stopped_by": child.stopped_by,
                "success": child.success,
                "output": child.output,
                "error": child.error,
                # The worker's board already forwarded these to ours, so a
                # waiting Pool above has seen them. They are repeated here
                # because the model that delegated is waiting too, and the
                # tool result is the only channel it reads.
                "beacons": [b.to_dict() for b in (child.beacons or [])],
                "evidence": MODEL_CLAIM}

    # ── Result ───────────────────────────────────────────────────────────

    def _result(self, state, success, output, stopped_by, error=None) -> AgentResult:
        return AgentResult(
            success=success, output=output, mode=self.mode,
            steps_taken=state["steps"], tokens_used=state["tokens"],
            cost=state["cost"], stopped_by=stopped_by, error=error,
            plan=[{"goal": g} for g in (state["plan"] or [])],
            adaptations=[Adaptation(**a) if isinstance(a, dict) else a
                         for a in state.get("adaptations") or []],
        )

    def _finished(self, result, journal, board):
        """Attach the run's records and announce the end — unless it parked.

        A :class:`~..state.SuspendedResult` is not an end: ``run.suspended``
        went out when the run parked, the document is in the store, and none
        of the closing records apply to a run that is still owed an answer.
        """
        if isinstance(result, SuspendedResult):
            return result
        result.journal = journal.to_list()
        result.beacons = board.all()
        result.acceptance = acceptance_from(result.journal)
        self._emit("run.finished", payload={"success": result.success,
                                            "stopped_by": result.stopped_by},
                   usage=result.tokens_used, error=result.error)
        if self._doc_id is not None:
            # The parked run reached its end; the store holds suspended runs
            # only, and a document left behind is a run a trigger can wake a
            # second time.
            self.store.delete(self._doc_id)
            self._doc_id = None
        return result

    def _park(self, susp, call, reply, messages, state, journal):
        """Write the parked run to the store and return a SuspendedResult.

        Two things are settled here rather than left to the resume. The other
        calls of the same turn — a provider may ask for several — are answered
        as not executed, because a history with calls nobody answered is the
        one shape every provider handles differently. And the parked call is
        deliberately left **unanswered**: that gap is what the run is waiting
        for, and `resume()` fills exactly it.
        """
        at = next(i for i, c in enumerate(reply.calls) if c is call)
        for other in reply.calls[at + 1:]:
            messages.append(tool_result_turn(
                other.id,
                prompts.result_message(None, f"not executed: the run paused at "
                                             f"{call.name!r} before this call")))

        # The question with what it is about: a person asked to approve saw a
        # reason and not the call it was for.
        awaiting = {"reason": susp.reason, "resume_with": dict(susp.resume_with),
                    "tool": call.name, "arguments": _copy(dict(call.arguments or {}))}
        doc = self._save_parked(
            {"messages": messages, "state": self._portable_state(state),
             "journal": journal.to_list(), "run_id": self._run_id},
            {**awaiting, "hint": susp.hint, "call_id": call.id,
             "tool": call.name, "arguments": dict(call.arguments or {})},
            run_id=self._doc_id)                 # parked again: one identifier
        self._doc_id = doc.run_id

        self._emit("run.suspended", name=call.name, step=state.get("steps"),
                   payload={"run_id": doc.run_id, "tool": call.name,
                            "hint": susp.hint, "agent": self.name or "",
                            **awaiting})
        self._log(1, f"\n[Paused] … {susp.reason or call.name} · {doc.run_id}")
        return SuspendedResult(run_id=doc.run_id, awaiting=awaiting,
                               document=doc.to_dict())

    def _save_parked(self, variables: dict, suspend: dict, run_id=None):
        """Store a run document whose one step is suspended on *suspend*."""
        from ..state import RunDocument, StepStatus

        doc = RunDocument.new("agent", [self.name or "agent"], variables=variables)
        if run_id is not None:
            doc.run_id = run_id
        if (context := current()) is not None:
            doc.context = context.to_dict()
        doc.steps[0]["status"]  = StepStatus.SUSPENDED
        doc.steps[0]["suspend"] = suspend
        self.store.save(doc.run_id, doc.to_dict())
        return doc

    @staticmethod
    def _task_from_signal(signal, parked: dict, doc):
        """What an armed run was waiting for: its task, and its variables."""
        seeded = dict(doc.variables.get("variables") or {})
        if isinstance(signal, str) and signal.strip():
            return signal, seeded
        if isinstance(signal, dict) and str(signal.get("task") or "").strip():
            seeded.update({k: v for k, v in signal.items() if k != "task"})
            return str(signal["task"]), seeded
        raise ValueError(
            "an armed run is started by its task: pass resume(run_id, "
            "signal=\"...\") or signal={\"task\": \"...\", ...}, where the "
            "other keys seed the conversation. This run was armed asking for "
            f"{parked.get('resume_with') or '{}'}"
            + (f" — {parked['reason']}" if parked.get("reason") else "") + ".")

    def _complete(self, parked: dict, signal):
        """Run the parked tool with the signal. Returns ``(result, error)``.

        A tool that parks and a tool that resumes are the same object called
        twice — `_signal=None` the first time, the world's answer the second.
        A second `Suspend` from here would be a tool that cannot be completed
        by the signal it asked for, so it is reported rather than parked
        again: the run would otherwise wait for the same answer forever.
        """
        tool = self._tool_map.get(parked.get("tool"))
        if tool is None:
            return None, (f"no tool named {parked.get('tool')!r} on this agent "
                          "any more; the run was parked by a different "
                          "configuration")
        try:
            return tool.run(_signal=signal, **(parked.get("arguments") or {})), None
        except Suspend as susp:
            return None, (f"tool {parked.get('tool')!r} asked to pause again on "
                          f"resume ({susp.reason or 'no reason given'}); a "
                          "signal must complete the tool that asked for it")
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _portable_state(state: dict) -> dict:
        """The counters that survive a pause, and only those.

        The journal travels beside this, in its own serialised form. What is
        dropped is dropped on purpose: `adaptations` are per-request notices
        about what went on the wire, and do not describe the conversation,
        which is what resumes. The nudge latch is kept, so a resumed run does
        not repeat a nudge it already gave.
        """
        out = {k: state.get(k) for k in ("steps", "tokens", "cost", "plan")}
        # The latch travels: dropped, a resumed run said its nudge again.
        out["nudged"] = sorted(state.get("nudged") or ())
        return out

    def _note_unfinished(self, journal, state, why: str) -> None:
        """A ceiling reached with a plan outstanding is a plan not finished."""
        if not state.get("plan"):
            return
        journal.append(
            "plan not completed",
            outcome  = _J_SKIPPED,
            evidence = _evidence(MODEL_CLAIM, f"run ended via {why}"),
            reason   = (f"{len(state['plan'])} planned item(s) were outstanding "
                        f"when the run stopped."),
        )

    # ── Plumbing ─────────────────────────────────────────────────────────

    def _reply(self, skill):
        """One model turn — streamed when somebody is watching it.

        `run()` buffers, because it has nobody to show prose to and a
        buffered call keeps the fallback chain. `stream()` streams, because a
        reader is in front of the last turn and today it arrives in one piece
        after a silence as long as the model takes.

        The text goes out on the **same** channel as everything else, bracketed
        so a consumer can open a block, append to it and close it:
        ``text.started`` on the first piece, a ``text.delta`` per piece, and
        ``text.ended`` when the turn is done — all carrying one ``id``. A turn
        that asks for a tool and says nothing produces none of the three,
        rather than an empty pair a consumer has to filter.
        """
        if not self._streaming:
            return skill.run()

        message_id = uuid.uuid4().hex[:12]
        opened = False
        try:
            for piece in skill.stream():
                if not opened:
                    self._emit("text.started", payload={"id": message_id})
                    opened = True
                self._emit("text.delta",
                           payload={"id": message_id, "text": piece})
                yield
        finally:
            # A block opened is closed, whatever ended it: a consumer left with
            # `text.started` and no `text.ended` keeps a block open for ever.
            if opened:
                self._emit("text.ended", payload={"id": message_id})
        # `stream()` assembles as well as yields, so the decision this turn
        # reached is the same object `run()` would have returned — a
        # ToolCallRequest when the model asked to act, the text otherwise.
        return skill.last_result

    def _stamping_hooks(self) -> list:
        """The caller's hooks, each seeing this run's id on every event."""
        if not self.hooks or self._run_id is None:
            return self.hooks
        import dataclasses

        run_id = self._run_id

        def stamp(event):
            if getattr(event, "run_id", None) is None:
                event = dataclasses.replace(event, run_id=run_id)
            for hook in self.hooks:
                emit([hook], event)

        return [stamp]

    def _emit(self, etype: str, **fields) -> None:
        if self.hooks:
            fields.setdefault("name", self.name)
            emit(self.hooks, Event(type=etype, run_id=self._run_id, **fields))

    def _log(self, level: int, message: str = "") -> None:
        # This agent's verbosity, not the process's: the line goes to the
        # console when this agent was asked to speak, and to the package
        # logger otherwise, for whoever configured it.
        lvl = logging.INFO if level == 1 else logging.DEBUG
        if (self.verbose or 0) >= level:
            console().log(lvl, "%s", message)
        else:
            _logger.log(lvl, "%s", message)

    def __repr__(self) -> str:
        team = ("auto" if self.team == "auto"
                else f"{len(self.team)} worker(s)" if self.team else "none")
        return (f"Agent(mode={self.mode!r}, tools={len(self.tools)}, "
                f"team={team})")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _interrupted(result) -> bool:
    """A run that stopped on the model rather than on its own: worth a retry."""
    return (isinstance(result, AgentResult) and not result.success
            and result.stopped_by in ("llm_error", "truncated"))


def _copy(value):
    """A copy for an event; the value itself when it cannot be copied."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


#: The deepest a call's arguments may nest.
MAX_ARGUMENT_DEPTH = 64


def _depth(value, level: int = 0) -> int:
    """How deep *value* nests, counted no further than one past the limit."""
    if level > MAX_ARGUMENT_DEPTH:
        return level
    if isinstance(value, dict):
        return max([_depth(v, level + 1) for v in value.values()] or [level + 1])
    if isinstance(value, (list, tuple)):
        return max([_depth(v, level + 1) for v in value] or [level + 1])
    return level


def _action(call) -> str:
    """A call as the journal records it: the name and arguments, shortened for
    reading, with a hash of the whole arguments — two calls that differed only
    past the cut read as one repeated, and `repeating()` fired on them."""
    import hashlib
    full = json.dumps(call.arguments or {}, ensure_ascii=False, default=str, sort_keys=True)
    digest = hashlib.sha256(full.encode()).hexdigest()[:12]
    return f"{_safe({'name': call.name, 'arguments': call.arguments})} #{digest}"


def _safe(value, limit: int = 200):
    """A short rendering for logs and the journal."""
    text = json.dumps(value, ensure_ascii=False, default=str) \
        if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "…"
