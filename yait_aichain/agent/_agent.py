"""
agent._agent — the ``Agent`` class and its public loop: ``run``, ``stream``,
``step``, ``execute`` and ``resume``.

Everything the agent can do is a tool, its own machinery included (``pool``,
``delegate``, ``write_plan``). The conversation only ever appends, which keeps
the cacheable prefix stable. See docs/agents/overview.md.
"""

from __future__ import annotations

import json
import contextlib
import logging
import uuid
import warnings
from collections.abc import Callable
from typing import Any

from .._budget import Budget, as_budget, start_run
from ..models import Model
from ..skills import Skill
from ..utilities._base import Utility
from ..state._run import AgentRun, active, opened
from .._dispatch import Denied, Dispatcher, calling
from .._dispatch import MAX_ARGUMENT_DEPTH  # noqa: F401  (named in the release notes)
from ..state import StateStore
from ..tools._permissions import ApprovalRequest, PermissionPolicy

from .._console import console
from .. import _hints
from ._swarm import collecting_beacons
from ._journal import Journal, evidence as _evidence, CHECK, DONE as _J_DONE, FAILED as _J_FAILED
from ._result  import AgentResult
from .         import _prompts as prompts
from .._events import Event, emit
from ..models._calls import (ToolCallRequest, dangling_calls,
                             tool_result_turn)
from ..state   import RunContext, SuspendedResult, using

_logger = logging.getLogger("yait_aichain.agent")

# The pieces this module is cut into. Every name that lived here is still
# importable from here: callers, tests and saved chains name them.
from ._helpers import _interrupted, _copy, _action, _safe  # noqa: F401
from ._stops import (step_count, token_budget, cost_budget, nudge, verified,  # noqa: F401
                     stalled, repeating, check, _EFFECTS)
from ._builtins import (_POOL_SCHEMA, _PLAN_SCHEMA, DELEGATE_WHEN,  # noqa: F401
                        _delegate_schema, _team_parts, MAX_POOL_ITEMS, MAX_POOL_FLOWS,
                        _RESERVED, _Builtin, _DRAFTING, _Builtins)
from ._loop import _Loop


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
    """Refuse, by name and before the first call, what cannot be a tool and two
    tools the model cannot tell apart."""
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
            # The loop answers these names itself, so a tool of one would never run.
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
class Agent(_Loop, _Builtins):
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
        ``"agile"`` decides at every step; ``"waterfall"`` plans at step 0 and holds to it.
    team : dict | list[Agent] | None
        ``{"members": [agents, chains], "extend": bool, "max": N, "max_delegations": N}``;
        a list is the members, ``None`` removes ``delegate``; ``"auto"`` is deprecated.
    planner_model : Model, optional
        ``waterfall`` only: the model that writes the plan; ``model`` runs it.
    stop_when : list, optional
        Conditions from this module. Defaults to ``[step_count(30)]``.
    hooks : list | None, optional
        Callables given an ``Event`` at every step boundary.
    policy : dict | PermissionPolicy | None, optional
        ``{"rules": {risk: "allow"|"approve"|"deny"}, "default": ..., "approve":
        "park"|"deny"|fn, "interceptors": [...]}``, or the rules alone; none runs all.
    permissions : PermissionPolicy | None, optional
        Deprecated; goes in 5.0, use ``policy={"rules": ...}``.
    approve : callable, optional
        Deprecated; goes in 5.0, use ``policy={"approve": ...}``.
    store : StateStore, optional
        Where a parked run is kept until ``resume()``; needed by ``wait()``/``gate()``.
    max_cost : float | Budget | None, optional
        A ceiling in dollars for the run; past it the run ends ``success=False``.
    name : str | None, optional
        This agent's name as a worker in another's ``team`` and in events.
    description : str | None, optional
        What it does — what a lead reads when choosing a worker.
    verbose : int, optional
        ``0`` silent, ``1`` each step, ``2`` everything, to the ``yait_aichain`` logger.
    interceptors : list | None, optional
        Deprecated; goes in 5.0, use ``policy={"interceptors": [...]}``.
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
        interceptors:  list | None = None,
        policy:        dict | PermissionPolicy | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}; got {mode!r}")

        self.model        = model
        self.tools        = list(tools or [])
        _check_tools(self.tools)
        self.instructions = instructions
        self.mode         = mode
        # One form whatever was given — see `_team_parts`.
        self.team         = _team_parts(team)
        #: Members the lead added, by the instructions that made them, so one
        #: role is made once and given the work again.
        self._extended: dict = {}
        if planner_model is not None and mode != "waterfall":
            raise ValueError("planner_model applies to mode='waterfall' only — "
                             "no other mode has a planning phase")
        self.planner_model = planner_model
        self.stop_when    = list(stop_when) if stop_when is not None else [step_count(30)]
        if not any(getattr(c, "kind", "ceiling") == "ceiling" for c in self.stop_when):
            # A `stop_when` given replaces the default step_count(30), and a
            # check that never passes would run the loop without end.
            warnings.warn("stop_when has no ceiling, so step_count(30) is added; give one of "
                          "your own — step_count, token_budget, cost_budget — to choose",
                          stacklevel=3)
            self.stop_when.append(step_count(30))
        if any(getattr(c, "name", "") == "cost_budget" for c in self.stop_when):
            # Two money limits that work differently: this one ends the run
            # between turns, `max_cost` refuses to begin a call past it, on
            # every primitive and shared between them. One is enough.
            warnings.warn("cost_budget(...) in stop_when goes in 5.0; give max_cost= instead — it "
                          "stops a call before it would pass the limit, where cost_budget "
                          "stops the run after a turn already has", FutureWarning, stacklevel=3)
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

        # Everything one invocation owns — its id, a parked document, a
        # pending approval, whether it streams, the hooks it reports to —
        # lives on its `AgentRun` (state/_run.py), not here: one agent serves
        # concurrent runs. Outside a run, `_idle` stands in for the external
        # seams (`step()`, a direct `_do_delegate`).
        self._idle: AgentRun | None = None
        for old, value in (("permissions", permissions), ("approve", approve),
                           ("interceptors", interceptors)):
            if value is not None:
                warnings.warn(f"Agent({old}=...) goes in 5.0; say it in policy="
                              + ("{'rules': ...}" if old == "permissions" else f"{{{old!r}: ...}}"),
                              FutureWarning, stacklevel=3)
        if policy is not None and permissions is not None:
            raise ValueError("the rules are given twice — policy= and permissions=; keep policy=")
        from .._dispatch import split_policy
        rules, approve, interceptors = split_policy(policy, approve=approve,
                                                    interceptors=interceptors,
                                                    where="Agent(policy=)")
        # The rules, who answers an "approve" — a function, "park" or "deny" —
        # and what acts on each call: the three parts of `policy=`, kept apart
        # because each is read in its own place.
        self.permissions  = rules if rules is not None else permissions
        self.approve      = approve
        # Where a parked run is kept. A tool that declares `suspends` can end
        # a turn with "the world must answer this before I go on", and the
        # conversation — which is the whole of an agent's state — has to
        # survive until it does. Demanded at construction rather than at the
        # pause, because a pause nobody can resume is otherwise discovered at
        # the worst possible moment: after the model asked.
        self.store        = store
        if self.approve == "park" and store is None:
            raise ValueError("policy approve='park' parks a run until a person answers, and "
                             "there is nowhere to keep it: pass store=InMemoryStore() or "
                             "store=FileStore(path)")
        if (parked := [t.name or type(t).__name__ for t in self.tools
                       if getattr(t, "suspends", False)]) and store is None:
            raise ValueError(
                f"tool(s) {parked} can park this run and there is nowhere to "
                "keep it. Pass store=InMemoryStore() to resume inside this "
                "process, or store=FileStore(path) — or your own StateStore — "
                "to resume from another one. A serverless invocation cannot "
                "hold the wait, so the run has to outlive it.")
        # A ceiling in money for everything this agent spends, shared with the
        # Skills it builds and with any worker it delegates to. `cost_budget`
        # in `stop_when` stops the loop between turns; this stops a call from
        # beginning, and the two compose — one is a decision, the other a
        # limit.
        self.max_cost = as_budget(max_cost)
        # Its own only when made here from a number: a Budget handed in — an
        # agent's, to the Skill of each turn — is shared, and start_run must
        # not replace it, or the ceiling holds for one turn at a time.
        self._own_budget = None if isinstance(max_cost, Budget) else self.max_cost
        self.name         = name
        self.description  = description
        self.verbose      = verbose
        self.interceptors = interceptors
        # How deep in a delegation this agent's runs start, and how deep they
        # may go. Not parameters: a run carries its own depth (`AgentRun`),
        # and `delegate` hands a worker its lead's plus one.
        self._depth       = 0
        self._max_depth   = 5

        self._tool_map   = {t.name: t for t in self.tools}
        members = (self.team or {}).get("members") or []
        self._worker_map = {(getattr(w, "name", None) or f"worker{i}"): w
                            for i, w in enumerate(members)}
        if len(self._worker_map) != len(members):
            names = [getattr(w, "name", None) or f"worker{i}" for i, w in enumerate(members)]
            raise ValueError(f"two workers share a name in {names}; delegate picks a worker by "
                             "name, and the model could reach only one of them")
        if members:
            blind = [getattr(w, "name", None) or f"worker{i}"
                     for i, w in enumerate(members)
                     if not getattr(w, "name", None) or not getattr(w, "description", None)]
            if blind:
                warnings.warn(
                    f"worker(s) {blind} have no name or no description; the lead picks a "
                    "worker by what these say, so without them the choice is a guess. "
                    "Give each worker Agent(name=..., description=...).", stacklevel=2)

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
        with self._opened():
            return self._run(task, variables)

    def _run(self, task: str, variables: dict | None = None) -> AgentResult:
        journal, messages, state = self._begin(task, variables)
        self._emit("run.started", payload={"task": task, "mode": self.mode})
        with collecting_beacons() as board:
            result = self._loop(messages, state, journal)
        return self._finished(result, journal, board)

    def stream(self, task: str, variables: dict | None = None):
        """
        Run *task* and yield each :class:`~.._events.Event` as it happens.

        The same loop as ``run()``, walked instead of exhausted. It yields events, not
        tokens — ``Skill.stream`` is the one for text. The result is not yielded: it
        lands on ``last_result`` when the generator finishes. Abandoning the generator
        abandons the run, with no result and no ``run.finished``.

        Parameters
        ----------
        task : str
            What to do, in words.
        variables : dict | None, optional
            Data the caller already holds, given to the model with the task.
        """
        self.last_result = None
        ended: dict = {}
        # A context of its own, copied when the stream is made, not when it is
        # first read: a generator runs in whichever context advances it (see
        # `_bound`). The run is opened inside it, so it is this stream's from
        # the first step.
        from .._bound import bound
        return bound(self._stream(task, variables, ended),
                     on_end=lambda: setattr(self, "last_result", ended.get("result")))

    def _stream(self, task: str, variables: dict | None = None, ended: dict | None = None):
        with self._opened() as run:
            journal, messages, state = self._begin(task, variables)

            sink: list = []
            # On the run's own hooks, which end with it: a collector left on
            # the agent would go on collecting into a list nobody drains.
            run.hooks.append(sink.append)
            run.streaming = True
            with collecting_beacons() as board:
                self._emit("run.started", payload={"task": task, "mode": self.mode})
                while sink:
                    yield sink.pop(0)
                result = yield from self._loop_events(messages, state, journal, sink=sink)
                result = self._finished(result, journal, board)
            while sink:
                yield sink.pop(0)
            if ended is not None:
                ended["result"] = result

    def arm(self, reason: str = "", resume_with: dict | None = None, *,
            hint: dict | None = None,
            variables: dict | None = None) -> SuspendedResult:
        """
        Create the run **without starting it**, and return it parked.

        ``resume(run_id, signal)`` starts it: a string is the task, or a dict gives
        it under ``"task"`` and its other keys seed the conversation.

        Parameters
        ----------
        reason : str, optional
            What the run waits for, shown to whoever starts it.
        resume_with : dict | None, optional
            The shape of the signal that starts it.
        hint : dict | None, optional
            Recorded for whatever does the waking, such as ``{"on": "email.received"}``.
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
             "state": AgentRun.portable(self.new_state()),
             "variables": dict(variables or {})},
            awaiting)
        self._emit("run.armed", payload={"run_id": doc.run_id,
                                         "agent": self.name or "", **awaiting})
        return SuspendedResult(run_id=doc.run_id, awaiting=awaiting,
                               document=doc.to_dict())

    def resume(self, run_id: str, signal=_UNSET) -> AgentResult | SuspendedResult:
        """
        Continue a run this agent parked, delivering *signal* as the answer.

        Parameters
        ----------
        run_id : str
            The parked run, as the ``SuspendedResult`` named it.
        signal : any, optional
            The result of the tool call that parked the run; for an ``arm()`` run,
            the task (a string, or a dict with ``"task"``).

        Returns
        -------
        AgentResult | SuspendedResult
            The finished run, or the run parked again under the same ``run_id``.

        Raises
        ------
        KeyError
            *run_id* is not in the store — already resumed, or never parked.
        """
        if signal is _UNSET:
            # A run kept after a failure is retried with no answer; a run
            # waiting for one is not resumed without it: taken without an
            # answer, the parked question would be lost, the run would pause
            # again with nobody to ask, and its document would be deleted.
            waiting = ((((self.store.load(run_id) if self.store is not None else None) or {})
                        .get("steps") or [{}])[0].get("suspend") or {})
            if waiting and waiting.get("at") != "after":
                raise TypeError("resume needs the answer the run waited for: "
                                "agent.resume(run_id, signal=…)")
            signal = None
        with self._opened():
            return self._resume(run_id, signal)

    def _resume(self, run_id: str, signal=None) -> AgentResult | SuspendedResult:
        from ..state import RunDocument

        if self.store is None:
            raise ValueError(f"agent {self.name or ''!r} has no store, so it "
                             "never parked a run. Pass Agent(store=...).")
        # Claimed, not read: a second delivery of the same signal finds
        # nothing and gets the KeyError below, so two deliveries cannot both
        # run the approved action.
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
            self._live.doc_id = run_id
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

        self._live.id = doc.variables.get("run_id") or uuid.uuid4().hex
        self._live.doc_id = run_id
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
        if not error and result is not None:
            self._live.evidence = result
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
        """The rest of a resumed run. A retryable failure keeps the run in the store with
        what has happened, the parked action's result included, so a retry picks up
        after it instead of losing a run a person approved."""
        with collecting_beacons() as board:
            result = self._loop(messages, state, journal)
        if _interrupted(result):
            self._save_parked(
                self._live.parked(messages, state, journal),
                {"at": "after", "reason": result.error or result.stopped_by or ""},
                run_id=run_id)
            self._live.doc_id = None         # kept on purpose: `_finished` must not delete it
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

        Parameters
        ----------
        task : str
            What to do, in words.
        variables : dict | None, optional
            Data the caller already holds, given with the task.
        ground_rules : bool, optional
            Add the rule that a plain-text reply ends the run. ``run()`` sets it; a
            driver that owns its own loop usually does not.
        """
        opening = task
        if variables:
            # Without its secrets, or a token a chain carries for a tool would
            # go to the model's provider in this block.
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

        Returns a :class:`~models._calls.ToolCallRequest` when the model asked to act,
        or a string — the answer. The caller appends the reply (``.as_turn()``) and a
        result for every call, then calls again; :meth:`owns` names the calls that go
        to :meth:`execute`.

        Parameters
        ----------
        messages : list
            The conversation so far — ``opening()``'s two messages, then each reply
            and each tool result in turn.
        state : dict | None, optional
            The run's working state, kept by the caller; holds the ``waterfall`` plan.

        Raises
        ------
        ValueError
            When a tool call already in *messages* has no result.
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
            # Counted here as run() counts it, or a driver on this seam would
            # have a step_count that never moves, and so no ceiling at all.
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

    def owns(self, call) -> bool:
        """Whether *call* is one of the agent's own — ``pool``, ``write_plan``,
        ``delegate`` — which exist only inside this library and so go to
        :meth:`execute`, never to the driver's environment. The tools the
        agent was given are the driver's to place: a tool declared by schema
        alone runs in the world, one with a body may run in either.

        Parameters
        ----------
        call : ToolCall
            One call from a ``ToolCallRequest`` that ``step()`` returned.
        """
        return getattr(call, "name", call) in _RESERVED

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
        with self._opened():
            return self._execute_gated(call, state)

    def _execute_gated(self, call, state: dict | None = None):
        # The permission gate keeps its pending request on the run between
        # asking and answering; a run of its own keeps two drivers' apart.
        state = state if state is not None else self.new_state()
        step = self._dispatch.decide(call, state)
        if isinstance(step, Denied):
            return None, step.reason
        return self._dispatch.execute(step.call or call, state)

    # ── The loop ─────────────────────────────────────────────────────────

    def _opened(self, depth: int | None = None, parent_id: str | None = None,
                hooks: list | None = None):
        """Open a run of this agent: its id, hooks and depth are its own, while the
        configuration (tools, model, policy, store, a shared `Budget`) stays here."""
        waiting = active(self)
        if waiting is not None and waiting.id is None and depth is None:
            # Opened for this run by whoever starts it — `delegate`, with the
            # depth and the parent — and not begun yet: this is that run.
            return contextlib.nullcontext(waiting)
        own = list(self.hooks)
        return opened(AgentRun(
            # A worker reports to its lead's hooks as well as its own.
            owner=self, hooks=own + [h for h in (hooks or ()) if h not in own],
            depth=self._depth if depth is None else depth,
            max_depth=self._max_depth, parent_id=parent_id,
            parent_call_id=calling()))

    @property
    def _live(self) -> AgentRun:
        """This agent's current run, or, outside one, an idle stand-in with the agent's
        own hooks."""
        run = active(self)
        if run is not None:
            return run
        if self._idle is None:
            self._idle = AgentRun(owner=self, hooks=self.hooks,
                                  depth=self._depth, max_depth=self._max_depth)
        return self._idle

    @property
    def _dispatch(self) -> Dispatcher:
        """Where this agent's calls are decided and run (`_dispatch.py`).
        Built per use from the configuration, so a changed policy or approver
        on the agent is the one asked."""
        return Dispatcher(
            tools=self._tool_map,
            builtins={"pool": self._do_pool, "delegate": self._do_delegate,
                      "write_plan": self._do_plan},
            gated=self._gated_tool,
            permissions=self.permissions, approve=self.approve, max_cost=self.max_cost,
            emit=self._emit, who=self.name or "", run=lambda: self._live,
            log_call=self._log_call, interceptors=self.interceptors, note=self._noted)

    def _noted(self, text: str) -> None:
        """An interceptor's change, in the journal of the run it changed."""
        journal = getattr(self._live, "journal", None)
        if journal is not None:
            journal.append("interceptor", outcome=_J_DONE,
                           evidence=_evidence(CHECK, text), reason=text)

    def _gated_tool(self, call):
        """The tool a call will run, which the policy decides on: for `pool`, the tool
        it fans out, one approval covering the whole list."""
        if call.name == "pool":
            return self._tool_map.get((call.arguments or {}).get("tool") or "")
        if call.name in _DRAFTING:
            # Asked about like any tool, as `draft`: allowed unless a policy
            # says otherwise.
            return _DRAFTING[call.name]
        return self._tool_map.get(call.name)

    def _log_call(self, tool, kwargs) -> None:
        if self.verbose or _logger.isEnabledFor(logging.INFO):   # serialised only to be shown
            self._log(1, f"  ⚙  {tool.name}({_safe(kwargs)})")

    def _stamping_hooks(self) -> list:
        """The caller's hooks, each seeing this run's id on every event."""
        hooks, run_id = self._live.hooks, self._live.id
        parent_call_id = self._live.parent_call_id
        if not hooks or run_id is None:
            return hooks
        import dataclasses

        def stamp(event):
            if getattr(event, "run_id", None) is None:
                event = dataclasses.replace(event, run_id=run_id)
            if parent_call_id and "parent_call_id" not in (event.payload or {}):
                event = dataclasses.replace(event, payload={**(event.payload or {}),
                                                            "parent_call_id": parent_call_id})
            for hook in hooks:
                emit([hook], event)

        return [stamp]

    def _emit(self, etype: str, **fields) -> None:
        live = self._live
        if live.hooks:
            fields.setdefault("name", self.name)
            if live.parent_call_id:
                # Where this run sits in its lead's: the call it runs inside.
                fields["payload"] = {**(fields.get("payload") or {}),
                                     "parent_call_id": live.parent_call_id}
            emit(live.hooks, Event(type=etype, run_id=live.id, **fields))

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
        team = ("none" if not self.team else
                f"{len(self.team['members'])} member(s)"
                + (", extends" if self.team["extend"] else ""))
        return (f"Agent(mode={self.mode!r}, tools={len(self.tools)}, "
                f"team={team})")


# ── Helpers ──────────────────────────────────────────────────────────────────

