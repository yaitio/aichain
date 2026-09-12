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
import logging
import time
import uuid
from typing import Any

from ._journal import Journal, evidence as _evidence, CHECK, MODEL_CLAIM, \
                      DONE as _J_DONE, FAILED as _J_FAILED, SKIPPED as _J_SKIPPED
from ._result  import AgentResult
from .         import _prompts as prompts
from .._events import Event, emit
from ..models._calls import (ToolCallRequest, dangling_calls,
                             tool_result_turn)
from ..pool    import Pool
from ..tools._base import Tool

_logger = logging.getLogger("yait_aichain.agent")


# ── Stop conditions ──────────────────────────────────────────────────────────
#
# Everything that can end a run other than the model simply answering. Two
# kinds, and they must not be confused: a *check* that passes is a success
# with harness-verified evidence, a *ceiling* that is reached is a failure.

def step_count(n: int):
    """Stop after *n* decisions. A ceiling: reaching it is not success."""
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
    """Stop once *n* tokens have been spent. A ceiling."""
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
    """
    def _c(state):
        return "cost_budget" if (state["cost"] or 0.0) >= usd else None
    _c.kind, _c.name = "ceiling", "cost_budget"
    _c.spec = {"kind": "cost_budget", "value": usd}
    return _c


def check(fn, name: str = "check"):
    """
    Stop when *fn* says the objective is met — and call that a **success**.

    The only completion the harness can verify itself. Everything else the
    agent reports about its own progress is the model's word for it.
    """
    def _c(state):
        try:
            return f"check:{name}" if fn(state) else None
        except Exception as exc:                       # pragma: no cover
            _logger.debug("stop check %s raised: %s", name, exc)
            return None
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

#: Returned by `_ask_permission` when a request has gone out and the answer
#: has not been asked for yet. A string would be a denial and None a pass, so
#: the third state needs a value of its own.
_AWAITING = object()


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
    """

    MODES = ("agile", "waterfall")

    def __init__(
        self,
        model,
        tools:        "list | None" = None,
        instructions: str           = "",
        mode:         str           = "agile",
        team                        = None,
        planner_model               = None,
        stop_when:    "list | None" = None,
        hooks:        "list | None" = None,
        permissions                 = None,
        approve                     = None,
        max_cost                    = None,
        name:         "str | None"  = None,
        description:  "str | None"  = None,
        verbose:      int           = 0,
        _depth:       int           = 0,
        _max_depth:   int           = 5,
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
        self.instructions = instructions
        self.mode         = mode
        self.team         = list(team) if isinstance(team, (list, tuple)) else team
        if planner_model is not None and mode != "waterfall":
            raise ValueError("planner_model applies to mode='waterfall' only — "
                             "no other mode has a planning phase")
        self.planner_model = planner_model
        self.stop_when    = list(stop_when) if stop_when is not None else [step_count(30)]
        self.hooks        = list(hooks or [])

        #: The result of the last :meth:`stream`, once the generator has run
        #: to the end. `run()` returns its result; a generator cannot, so it
        #: is left here rather than mixed into a stream of events.
        self.last_result: "AgentResult | None" = None

        # `Event.run_id` has been a declared field since M3 and the agent
        # never filled it. Without it two concurrent invocations — which is
        # the ordinary case for the serverless target — write into one stream
        # that cannot be demultiplexed afterwards.
        self._run_id: "str | None" = None

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
        # A ceiling in money for everything this agent spends, shared with the
        # Skills it builds and with any worker it delegates to. `cost_budget`
        # in `stop_when` stops the loop between turns; this stops a call from
        # beginning, and the two compose — one is a decision, the other a
        # limit.
        from .._budget import as_budget
        self.max_cost = as_budget(max_cost)
        self.name         = name
        self.description  = description
        self.verbose      = verbose
        self._depth       = _depth
        self._max_depth   = _max_depth

        self._tool_map   = {t.name: t for t in self.tools}
        self._worker_map = ({(getattr(w, "name", None) or f"worker{i}"): w
                             for i, w in enumerate(self.team)}
                            if isinstance(self.team, list) else {})
        if verbose:
            _install_console_handler(verbose)

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, task: str, variables: "dict | None" = None) -> AgentResult:
        """
        Execute *task* and return an :class:`AgentResult`.

        *variables* seeds the conversation with data the caller already holds —
        this is how a ``Chain`` or ``Pool`` step hands its accumulated values
        down. There is one place state lives, and it is the conversation.
        """
        journal  = Journal()
        messages = self.opening(task, variables, ground_rules=True)
        state    = self.new_state()
        self._run_id = uuid.uuid4().hex
        self._emit("run.started", payload={"task": task, "mode": self.mode})

        result = self._loop(messages, state, journal)
        result.journal = journal.to_list()
        self._emit("run.finished", payload={"success": result.success,
                                            "stopped_by": result.stopped_by},
                   usage=result.tokens_used, error=result.error)
        return result

    def stream(self, task: str, variables: "dict | None" = None):
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
        """
        journal  = Journal()
        messages = self.opening(task, variables, ground_rules=True)
        state    = self.new_state()
        self._run_id = uuid.uuid4().hex
        self.last_result = None

        sink: list = []
        # Appended for the duration and removed in `finally`: a hook left
        # behind on the instance would go on collecting into a list nobody
        # drains, which is a leak that only shows up under load.
        collect = sink.append          # bound once, so `remove` finds it
        self.hooks.append(collect)
        self._streaming = True
        try:
            self._emit("run.started", payload={"task": task, "mode": self.mode})
            while sink:
                yield sink.pop(0)

            walk = self._loop_events(messages, state, journal, sink=sink)
            while True:
                try:
                    yield next(walk)
                except StopIteration as done:
                    result = done.value
                    break

            result.journal = journal.to_list()
            self._emit("run.finished",
                       payload={"success": result.success,
                                "stopped_by": result.stopped_by},
                       usage=result.tokens_used, error=result.error)
            while sink:
                yield sink.pop(0)
            self.last_result = result
        finally:
            self._streaming = False
            try:
                self.hooks.remove(collect)
            except ValueError:                       # already gone; fine
                pass

    # ── Externally driven: one turn at a time ────────────────────────────
    #
    # ``run()`` owns the loop. Some callers own it instead — a serverless
    # invocation that must be one step, or a harness that executes the tools
    # itself. The same machinery, exposed.

    def new_state(self) -> dict:
        """Fresh run state — the counters ``stop_when`` conditions read."""
        return {"steps": 0, "tokens": 0, "cost": None, "plan": None,
                "adaptations": []}

    def opening(self, task: str, variables: "dict | None" = None,
                *, ground_rules: bool = False) -> list:
        """
        The two messages a run starts from: the stable prefix and the task.

        ``ground_rules`` injects the "plain text ends the run" rule, and it
        belongs to whoever owns the loop: ``run()`` passes True because in its
        loop a text reply really does end the run. An external driver calling
        ``opening()``/``step()`` owns its own loop — there a plain reply is an
        ordinary turn, so the default is clean instructions and the driver
        states its own framing.
        """
        opening = task
        if variables:
            opening = (f"{task}\n\nGIVEN:\n"
                       + json.dumps(variables, indent=1, ensure_ascii=False,
                                    default=str))
        return [
            {"role": "system",
             "parts": [prompts.system_message(self.instructions, self.mode,
                                              ground_rules=ground_rules)]},
            {"role": "user", "parts": [opening]},
        ]

    def step(self, messages: list, state: "dict | None" = None):
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
        """
        if unanswered := dangling_calls(messages):
            raise ValueError(
                f"agent {self.name or ''!r}: the history has tool call(s) "
                f"{unanswered} with no result. Append "
                "`tool_result_turn(call.id, result)` for every call in the "
                "previous reply — failed ones included, with the error as "
                "the result — before calling step() again. `run()` does this "
                "for you; an external driver owns it.")
        return self._ask(messages, state if state is not None else self.new_state())

    def execute(self, call, state: "dict | None" = None):
        """Run one :class:`ToolCall`. Returns ``(result, error)`` — never raises."""
        return self._execute(call, state if state is not None else self.new_state())

    # ── The loop ─────────────────────────────────────────────────────────

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
        while True:
            try:
                next(walk)
            except StopIteration as done:
                return done.value

    def _loop_events(self, messages: list, state: dict, journal: Journal,
                     sink: "list | None" = None):
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
                reply = self._ask(messages, state)
            except Exception as exc:
                yield from drain()
                return self._result(state, False, None, "llm_error",
                                    error=f"{type(exc).__name__}: {exc}")
            yield from drain()

            # Text is the answer — one exit, not two spellings of it.
            if not isinstance(reply, ToolCallRequest):
                self._log(1, f"\n[Done] ✓ answered · {state['steps']} step(s)")
                return self._result(state, True, str(reply), "answered")

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
                self._emit("tool_call.started", name=call.name,
                           step=state["steps"],
                           payload={"id": call.id or "",
                                    "arguments": call.arguments or {},
                                    "agent": self.name or ""})
                yield from drain()

                denial = self._ask_permission(call, state)
                if denial is _AWAITING:
                    # The request reaches the consumer *here*, before anyone
                    # is asked — which is the whole point of splitting it.
                    yield from drain()
                    denial = self._resolve_permission(state)
                    yield from drain()

                result, error = ((None, denial) if denial
                                 else self._execute(call, state))
                self._emit("tool_call.ended", name=call.name,
                           step=state["steps"],
                           payload={"id": call.id or "", "result": result,
                                    "agent": self.name or ""},
                           error=error)
                yield from drain()

                journal.append(
                    call.name,
                    action      = _safe({"name": call.name,
                                         "arguments": call.arguments}),
                    outcome     = _J_FAILED if error else _J_DONE,
                    evidence    = _evidence(CHECK, error or "executed"),
                    reason      = error or "",
                    # The journal is a written record; an image cannot go in
                    # it, so media is named rather than embedded.
                    observation = prompts.observation_text(result, error),
                    step        = state["steps"],
                )
                messages.append(tool_result_turn(
                    call.id or call.name,
                    prompts.result_message(result, error)))

            if fired := self._fired(state):
                kind = getattr(fired[1], "kind", "ceiling")
                if kind == "check":
                    self._log(1, f"\n[Done] ✓ {fired[0]} · {state['steps']} step(s)")
                    return self._result(state, True, None, fired[0])
                self._log(1, f"\n[Stop] ✗ {fired[0]} · {state['steps']} step(s)")
                self._note_unfinished(journal, state, fired[0])
                return self._result(
                    state, False, None, fired[0],
                    error=f"stopped by {fired[0]} after {state['steps']} step(s)")

    def _fired(self, state: dict):
        for cond in self.stop_when:
            if fired := cond(state):
                return fired, cond
        return None

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
        """
        One turn of the conversation, through ``Skill``.

        Delegating instead of hand-rolling the call is what makes cost,
        caching and native tool calling arrive at all: they were added to the
        primitive and would never reach a private copy of it.
        """
        from ..skills import Skill
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
        reply = self._reply(skill)
        # Kept per run, deduplicated: a loop asks the same thing of the same
        # model every turn, so without this the list grows by a copy a step.
        for a in skill.last_adaptations:
            if a not in state["adaptations"]:
                state["adaptations"].append(a)
        usage = skill.last_usage
        state["tokens"] += getattr(usage, "total_tokens", 0) or 0
        if (c := getattr(usage, "cost", None)) is not None:
            state["cost"] = (state["cost"] or 0.0) + c
        self._log(2, f"     +{getattr(usage, 'total_tokens', 0):,} tokens "
                     f"({time.monotonic() - _t0:.1f}s)")
        return reply

    # ── Executing one call ───────────────────────────────────────────────

    def _execute(self, call, state: dict) -> "tuple[Any, str | None]":
        """Run one call. Returns ``(result, error)`` — never raises."""
        try:
            if call.name == "pool":
                return self._do_pool(call.arguments), None
            if call.name == "delegate":
                return self._do_delegate(call.arguments, state), None
            if call.name == "write_plan":
                return self._do_plan(call.arguments, state), None
            return self._do_tool(call, state), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _do_tool(self, call, state: "dict | None" = None) -> Any:
        tool = self._tool_map.get(call.name)
        if tool is None:
            raise ValueError(f"no tool named {call.name!r}. "
                             f"Available: {sorted(self._tool_map)}")
        kwargs = call.arguments or {}
        # No permission check here: the loop resolved it before calling, so
        # a second one would put the question to a person twice.
        if (problem := tool.check_args(kwargs)) is not None:
            raise ValueError(problem)
        self._log(1, f"  ⚙  {tool.name}({_safe(kwargs)})")
        return tool.run(**kwargs)

    def _ask_permission(self, call, state: "dict | None" = None):
        """Consult the policy before a tool runs. Returns a denial, or None.

        **At the loop boundary, not inside the call.** It lived in
        `_do_tool` until 2026-09-11, which put the approver's question and the
        approver's answer in the same instant as far as anyone outside could
        tell: a streaming consumer received `approval.requested` only after
        the decision had already been made, because the events it produced
        were drained at the next boundary — after `tool_call.ended`. The
        record read correctly and was useless for the one thing R5 exists
        for, which is showing a person the prompt. Split in two here so the
        loop can hand the request over *before* asking.

        Emitted even when there is nobody to ask. A consumer needs to see
        that a decision was required, and the refusing case is where that
        matters most: otherwise the call simply fails and nothing says it was
        governance rather than a broken tool.
        """
        tool = self._tool_map.get(call.name)
        if tool is None or not self.permissions:
            return None
        from ..tools._permissions import ALLOW, APPROVE, DENY, ApprovalRequest

        decision = self.permissions.decide(tool)
        if decision == ALLOW:
            return None
        risk = getattr(tool, "risk", "write")
        if decision == DENY:
            return f"tool {tool.name!r} denied by policy"
        if decision != APPROVE:                       # unreachable via the
            return None                               # policy's own validation

        self._pending = ApprovalRequest(
            tool=tool.name, risk=risk, arguments=dict(call.arguments or {}),
            call_id=getattr(call, "id", "") or "", agent=self.name or "")
        self._emit("approval.requested", name=tool.name,
                   step=(state or {}).get("steps"),
                   payload={"id": self._pending.call_id, "tool": tool.name,
                            "risk": risk, "arguments": self._pending.arguments,
                            "agent": self.name or ""})
        return _AWAITING

    def _resolve_permission(self, state: "dict | None" = None):
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
            answer  = self.approve(request)
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
        if state["plan"] is not None:
            # "You may not rewrite it" — and a rewrite would also invalidate
            # the prefix the freeze exists to keep cacheable.
            return "The plan is already recorded and cannot be rewritten."
        state["plan"] = [str(i) for i in items]
        self._log(1, f"  ▣  plan: {len(items)} item(s)")
        return "Plan recorded. Begin."

    def _do_pool(self, args: dict) -> Any:
        """
        One call repeated over a list.

        Every other call is exactly one execution. Without this, a model asked
        to "read every document" reads one and reports success — measured at
        2 of 8, silently.
        """
        items = args.get("items")
        if not isinstance(items, (list, tuple)) or not items:
            raise ValueError("pool needs a non-empty 'items' list")

        max_flows = max(1, min(int(args.get("max_flows") or 4), len(items)))

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
        elif args.get("prompt"):
            tool = _MappedSkill(self, args["prompt"])
            pool_items = [{"item": it} for it in items]
        else:
            raise ValueError("pool needs either 'tool' or 'prompt'")

        self._log(1, f"  ⇉  pool ×{len(items)} (max_flows={max_flows})")
        pool    = Pool(tool, items=pool_items, max_flows=max_flows,
                       on_error="collect")
        results = pool.run()
        errors  = [str(h.get("error")) for h in pool.history if h.get("error")]
        if errors and len(errors) == len(pool_items):
            raise RuntimeError(f"all {len(pool_items)} fanned-out calls failed. "
                               f"First: {errors[0]}")
        if errors:
            self._log(1, f"  ⚠  {len(errors)}/{len(pool_items)} entries failed; "
                         f"first: {errors[0]}")
        return results

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
        else:                                            # "auto"
            worker = Agent(
                self.model, tools=self.tools,
                instructions=args.get("instructions", ""),
                stop_when=self.stop_when, hooks=self.hooks,
                # The approver travels with the policy: a worker that
                # inherited the rules and not the answerer would refuse every
                # gated call, which reads as the policy being stricter for
                # children than for their parent.
                permissions=self.permissions, approve=self.approve,
                max_cost=self.max_cost,
                verbose=self.verbose,
                _depth=self._depth + 1, _max_depth=self._max_depth,
            )

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
                "evidence": MODEL_CLAIM}

    # ── Result ───────────────────────────────────────────────────────────

    def _result(self, state, success, output, stopped_by, error=None) -> AgentResult:
        return AgentResult(
            success=success, output=output, mode=self.mode,
            steps_taken=state["steps"], tokens_used=state["tokens"],
            cost=state["cost"], stopped_by=stopped_by, error=error,
            plan=[{"goal": g} for g in (state["plan"] or [])],
            adaptations=list(state.get("adaptations") or []),
        )

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
        for piece in skill.stream():
            if not opened:
                self._emit("text.started", payload={"id": message_id})
                opened = True
            self._emit("text.delta",
                       payload={"id": message_id, "text": piece})
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
        _logger.log(logging.INFO if level == 1 else logging.DEBUG, "%s", message)

    def __repr__(self) -> str:
        team = ("auto" if self.team == "auto"
                else f"{len(self.team)} worker(s)" if self.team else "none")
        return (f"Agent(mode={self.mode!r}, tools={len(self.tools)}, "
                f"team={team})")


# ── Helpers ──────────────────────────────────────────────────────────────────

class _MappedSkill(Tool):
    """Runs one ``Skill`` per pool entry."""

    name        = "_mapped_skill"
    description = "Internal: run the model over one pool entry."
    parameters  = {"type": "object", "properties": {"item": {}}}
    risk        = "read"

    def __init__(self, agent: "Agent", prompt: str) -> None:
        self._agent  = agent
        self._prompt = prompt

    def run(self, item=None, options=None):
        from ..skills import Skill
        text  = (self._prompt or "{item}").replace("{item}", str(item))
        skill = Skill(model=self._agent.model,
                      input={"messages": [{"role": "user", "parts": [text]}]})
        return skill.run()


def _safe(value, limit: int = 200):
    """A short rendering for logs and the journal."""
    text = json.dumps(value, ensure_ascii=False, default=str) \
        if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "…"


def _install_console_handler(verbose: int) -> None:
    """Route ``verbose`` output through the package logger, once."""
    pkg   = logging.getLogger("yait_aichain")
    level = logging.INFO if verbose == 1 else logging.DEBUG
    if not any(getattr(h, "_aichain_console", False) for h in pkg.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._aichain_console = True                # type: ignore[attr-defined]
        pkg.addHandler(handler)
    if pkg.level == logging.NOTSET or pkg.level > level:
        pkg.setLevel(level)
