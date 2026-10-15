"""
chain._chain — the ``Chain`` class: building, running, pausing and resuming a
pipeline of steps.

The step forms, how each kind of step reads the variables, and the history
record are described in docs/pipelines/chain.md; saving is in ``_files``,
input filling in ``_inputs``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import SuspendedResult

from .._budget import as_budget, lend, start_run

import logging
import warnings

_skip_log = logging.getLogger("yait_aichain.chain")

from .. import _hints
from ..models._usage import Usage
from .._events import Event, emit
from ._result import ChainResult
from ..tools._base import Record
from ..state._secret import public

# One vocabulary with Pool — see `_errors_policy`. `collect` is a silent
# `skip`: a chain of a hundred generated steps that expects a few to fail
# should not have to choose between a warning per failure and no record.
from .._errors_policy import POLICIES as _VALID_ON_STEP_ERROR, describe as _describe

from ._inputs import _AGENT_DEFAULT_OPTIONS


# The pieces this module is cut into; every name that lived here is still
# importable from here.
from ._inputs import (_MISSING, _usage_to_dict, _build_tool_kwargs, _callable_name,  # noqa: F401
                      _build_call_kwargs, _runner_name, _lookup, _resolve, _type_name,
                      _sources)
from ._files import _import_tool, _tool_spec, _refuse_function_tool, _Files  # noqa: F401
from ._flow import _Flow
from .._runners import invoke
from ._step import Step
from .._runners import is_tool as _is_tool, is_agent as _is_agent


class Chain(_Flow, _Files):
    """
    Steps in a fixed order, sharing one set of variables.

    A step is a ``Skill``, a tool, an ``Agent``, or a function, in any order; each
    gets the variables its prompt, schema or signature names, and its output is
    stored under its key, read whole or by path (``"page.body"``).

    ::

        chain = Chain(steps=[(summarise, "summary"), (translate, "french")], max_cost=0.05)
        result = chain.run(variables={"text": "…"})
        result["french"], result.output, result.cost

    Parameters
    ----------
    steps : list
        Runners, or tuples ``(runner, key[, input_map[, options]])`` or ``(runner,
        input_map)``; an Agent's *options* take ``task_key`` and ``output_field``.
    variables : dict | None, optional
        Default variables for every ``run()``; call-time values win.
    name : str | None, optional
        A label, shown in events and when the chain is a step elsewhere.
    description : str | None, optional
        What the chain does.
    on_step_error : str, optional
        ``"raise"`` (default), ``"stop"`` without raising, ``"skip"`` with a
        warning, or ``"collect"`` silently.
    store : StateStore, optional
        Where a paused run is kept; a persistent one to resume in another process.
    hooks : list | None, optional
        Callables given an ``Event`` at every step boundary.
    max_cost : float | Budget | None, optional
        A ceiling in dollars shared by every step; past it, ``BudgetExceeded``.
    policy : PermissionPolicy | dict | None, optional
        The rules a tool step is decided under; inside an agent, the stricter wins.
    approve : callable | None, optional
        Answers a step the policy wants a person for; absent, the step is refused.
    interceptors : list | None, optional
        Act on a tool step before and after it runs, as on an ``Agent``.
    inputs : dict | None, optional
        Descriptions for inputs the steps do not describe, read by :meth:`input_schema`.

    Raises
    ------
    ValueError
        If *steps* is empty, a tuple has an unsupported shape, or *on_step_error*
        is not one of the four.
    """

    @_hints.refuse("Chain")
    def __init__(
        self,
        steps,
        variables:      dict | None = None,
        name:           str  | None = None,
        description:    str  | None = None,
        on_step_error:  str         = "raise",
        store=None,
        hooks:          list | None = None,
        max_cost=None,
        policy=None,
        approve=None,
        interceptors=None,
        inputs=None,
    ) -> None:
        if not steps:
            raise ValueError("Chain requires at least one step.")
        from .._dispatch import split_policy
        import warnings as _warnings
        for old, value in (("approve", approve), ("interceptors", interceptors)):
            if value is not None:
                _warnings.warn(f"Chain({old}=...) goes in 5.0; say it in policy={{{old!r}: ...}}",
                              FutureWarning, stacklevel=3)
        self.policy, self.approve, self.interceptors = split_policy(
            policy, approve=approve, interceptors=interceptors, where="Chain(policy=)")
        if self.approve == "park":
            raise ValueError("Chain: approve='park' parks an agent's run; a chain cannot "
                             "be parked on a person's answer yet — give a function, or 'deny'")
        if inputs is not None and not (isinstance(inputs, dict)
                                       and all(isinstance(v, str) for v in inputs.values())):
            raise ValueError("inputs maps an input's name to its description, a str")
        self.inputs = dict(inputs or {})
        if on_step_error not in _VALID_ON_STEP_ERROR:
            raise ValueError(
                f"on_step_error must be one of {sorted(_VALID_ON_STEP_ERROR)}; "
                f"got {on_step_error!r}"
            )
        self._steps, self._extra = self._normalise(steps)
        self.variables      = variables or {}
        self._check_flow()
        self.name           = name
        self.description    = description
        self.on_step_error  = on_step_error
        self.hooks          = list(hooks or [])     # observability hooks
        # A store is always present (default: process-local in-memory) so the
        # engine has one uniform path; pass store= for persistence (FileStore,
        # S3, …). It holds suspended runs for resume.
        from ..state import InMemoryStore
        # `is not None`: a store of one's own with a `__len__` is falsy when
        # empty, and must not be replaced by an in-memory one.
        self._store = store if store is not None else InMemoryStore()
        # A Wait is a pause for a human, and a human answers in another
        # process — `resume()` is typically a separate command, sometimes the
        # next day. The default store is process-local, so that pause is
        # unresumable and only says so at resume time, as
        # "No suspended run ... in the store". Warn where the mistake is made
        # rather than where it surfaces. The default is left alone: a chain
        # with no Wait has nothing to persist.
        # A normalised step is (runner, output_key, input_map, kind, options).
        if store is None and any(
            getattr(st[0], "suspends", False) for st in self._steps
        ):
            import warnings
            warnings.warn(
                "This chain has a Wait step but no store, so the run is kept "
                "in memory and cannot be resumed from another process. Pass "
                "store=FileStore('runs/') (or another persistent store) if "
                "resume() runs outside this process.",
                RuntimeWarning, stacklevel=2,
            )
        self._history: list[dict] = []
        self._accumulated: dict   = {}     # snapshot of accumulated vars after last run()
        # Summed token usage of the last run() across Skill steps; None until
        # the first run or when no step reported usage. Reading is optional.
        # One ceiling for the whole chain, lent to each step for the duration
        # of a run — see `_lend_budget`. Shared rather than divided: a step
        # cannot be given "its share" when nobody knows in advance which step
        # will be the expensive one.
        self.max_cost = as_budget(max_cost)
        self._own_budget = self.max_cost        # see _budget.start_run

        self.last_usage: Usage | None = None
        # Per-request RunContext for the current run() / resume() (tenant +
        # metadata); set while a run is in flight, restored on resume.
        self.context = None

    def _store_output(self, idx: int, name: str, key: str, output, accumulated: dict) -> None:
        """An output under its key, a dict read whole or by path (``"page.body"``). For
        now a dict's fields are also spread, said once per step unless ``Step(spread=)``
        decides, and a step keyed by its runner's name also writes ``"result"``."""
        extra = self._extra[idx]
        accumulated[key] = output
        if isinstance(output, dict) and not isinstance(output, Record):
            spread = extra["spread"]
            if spread is None:
                spread = True
                # Once per step of this chain: a loop says it once, and a new
                # chain is not silenced by an old one that had its id.
                said = self.__dict__.setdefault("_spread_said", set())
                if idx not in said:
                    said.add(idx)
                    warnings.warn(
                        f"step {idx} ({name!r}) returned a dict: it is stored under {key!r}, and "
                        f"its fields are also spread among the chain's variables, which ends in "
                        f"5.0. Read {key + '.<field>'!r}, or say Step(..., spread=True) to keep it.",
                        FutureWarning, stacklevel=5)
            if spread:
                accumulated.update(output)
                accumulated[key] = output        # the dict wins over a field of its own name
        elif extra["default_key"] and key != "result":
            accumulated["result"] = output

    def _run_tool(self, runner, idx: int, kwargs: dict, signal, run_id):
        """A tool step, decided and run where every tool call is (`_dispatch.py`): this
        chain's rules and any around it, the tool's schema, the budget."""
        from .._dispatch import Denied, Dispatcher
        from ..models._calls import ToolCall
        from ..state._run import Run
        run = Run(owner=self, id=run_id, hooks=self.hooks)
        dispatcher = Dispatcher(
            tools={runner.name: runner}, builtins={}, gated=lambda call: runner,
            permissions=self.policy, approve=self.approve, max_cost=self.max_cost,
            emit=self._emit, who=self.name or "", run=lambda: run,
            log_call=lambda tool, kwargs: None, interceptors=self.interceptors)
        call = ToolCall(id=f"step_{idx}", name=runner.name, arguments=kwargs)
        step = dispatcher.decide(call)
        if isinstance(step, Denied):
            raise PermissionError(step.reason)
        call = step.call or call
        result = dispatcher.run_tool(call, extra=None if signal is None else {"_signal": signal},
                                     check=False)
        return dispatcher._after(call, result)

    def _emit(self, etype: str, **fields) -> None:
        """Dispatch a step-boundary :class:`Event` to registered hooks."""
        emit(self.hooks, Event(type=etype, **fields))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        variables:     dict | None = None,
        on_step_error: str  | None = None,
        *,
        context=None,
    ) -> ChainResult:
        """
        Execute all steps in order and return a :class:`ChainResult`.

        Parameters
        ----------
        variables : dict | None, optional
            Variables for this invocation, merged over ``self.variables``.
        on_step_error : str | None, optional
            Overrides the chain's ``on_step_error`` for this call; deprecated, goes in
            5.0 — set it on the chain or on a ``Step``.
        context : RunContext, optional
            The tenant and metadata of this request, ambient for every step;
            ``Model(api_key=callable)`` resolves its key from it.

        Returns
        -------
        ChainResult
            ``.output`` the last step's, ``result["key"]`` any step's, ``.success``
            whether all completed; a ``SuspendedResult`` when a step paused the run.

        Raises
        ------
        Exception
            A step's own exception, when ``on_step_error`` is ``"raise"``.
        ValueError
            If the *on_step_error* override is not recognised.
        """
        if on_step_error is not None:
            warnings.warn("on_step_error= on run() and resume() goes in 5.0; put it on the chain — "
                          "Chain(..., on_step_error=...) — or on a step, Step(..., on_error=...)",
                          FutureWarning, stacklevel=2)
        _on_error = self.on_step_error if on_step_error is None else on_step_error
        if _on_error not in _VALID_ON_STEP_ERROR:
            raise ValueError(
                f"on_step_error must be one of {sorted(_VALID_ON_STEP_ERROR)}; "
                f"got {_on_error!r}"
            )

        from ..state import RunDocument
        start_run(self)
        accumulated = {**self.variables, **(variables or {})}
        step_names  = [getattr(r, "name", None) or _runner_name(r) or f"step_{i}"
                       for i, (r, *_rest) in enumerate(self._steps)]
        doc = RunDocument.new("chain", step_names, variables=public(accumulated),
                              definition={"steps": self._signature()})
        from ..state import current, using
        # No context given is the one already in force — `with using(ctx):
        # chain.run()`, a chain inside a chain step, a chain in a Pool — and
        # the run document keeps it, so a resume in another process has it.
        context = context if context is not None else current()
        from .._events import running
        with using(context), lend(self.max_cost), running(doc.run_id):
            return self._run_from(doc, accumulated, start_idx=0, signal=None,
                                  usage_in=None, on_error=_on_error,
                                  context=context)

    def arm(self, reason: str = "", resume_with: dict | None = None, *,
            hint: dict | None = None, variables: dict | None = None,
            context=None) -> SuspendedResult:
        """
        Create the run **without starting it**, and return it parked.

        The run has a ``run_id`` and a document in the store before it runs;
        ``resume(run_id, signal)`` starts it, and the signal seeds the variables.

        Parameters
        ----------
        reason : str, optional
            What the run waits for, shown to whoever starts it.
        resume_with : dict | None, optional
            The shape of the signal that starts it — a JSON-Schema-like description.
        hint : dict | None, optional
            Recorded for whatever does the waking (``{"on": "order.created"}``).
        variables : dict | None, optional
            Variables the run is seeded with before the signal adds its own.
        context : RunContext, optional
            The request's tenant and metadata, kept with the run.
        """
        from ..state import RunDocument, StepStatus, SuspendedResult

        if not self._steps:
            raise ValueError("a chain with no steps has nothing to arm")

        accumulated = {**self.variables, **(variables or {})}
        step_names  = [getattr(r, "name", None) or _runner_name(r) or f"step_{i}"
                       for i, (r, *_rest) in enumerate(self._steps)]
        doc = RunDocument.new("chain", step_names, variables=public(accumulated),
                              definition={"steps": self._signature()})
        doc.context = context.to_dict() if context is not None else None

        awaiting = {"reason": reason, "resume_with": dict(resume_with or {}),
                    "hint": hint, "at": "start"}
        doc.steps[0]["status"]  = StepStatus.SUSPENDED
        doc.steps[0]["suspend"] = awaiting
        self._store.save(doc.run_id, doc.to_dict())
        self._emit("run.armed", step=0, run_id=doc.run_id, payload={"run_id": doc.run_id, **awaiting})
        return SuspendedResult(run_id=doc.run_id, awaiting=awaiting,
                               document=doc.to_dict())

    def resume(
        self,
        run_id:        str,
        signal=None,
        *,
        on_step_error: str | None = None,
        context=None,
    ) -> ChainResult | None:
        """
        Resume a parked run: the suspended step is re-invoked with *signal*, and steps
        already done are never re-run.

        Parameters
        ----------
        run_id : str
            The parked run, as the ``SuspendedResult`` named it.
        signal : any, optional
            The answer the run waited for. A dict is merged into the variables.
        on_step_error : str | None, optional
            Overrides the chain's policy for the rest of this run; deprecated, goes in
            5.0 — set it on the chain or on a ``Step``.
        context : RunContext, optional
            The tenant and metadata for the resumed part; the stored one when omitted.

        Raises
        ------
        KeyError
            *run_id* is not in the store — already resumed, or never parked.
        """
        if on_step_error is not None:
            warnings.warn("on_step_error= on run() and resume() goes in 5.0; put it on the chain — "
                          "Chain(..., on_step_error=...) — or on a step, Step(..., on_error=...)",
                          FutureWarning, stacklevel=2)
        _on_error = self.on_step_error if on_step_error is None else on_step_error
        if _on_error not in _VALID_ON_STEP_ERROR:
            raise ValueError(
                f"on_step_error must be one of {sorted(_VALID_ON_STEP_ERROR)}; "
                f"got {_on_error!r}"
            )

        from ..state import RunDocument, RunContext
        from ..models._usage import Usage

        # Claimed, not read: two deliveries of one signal cannot both run the
        # step a person approved. The second finds nothing.
        raw = self._store.take(run_id)
        if raw is None:
            raise KeyError(
                f"No suspended run {run_id!r} in the store "
                f"(already resumed/completed, or unknown id)."
            )
        doc   = RunDocument.from_dict(raw)
        # Whose run is this? A document of another kind — an agent's — or of
        # a chain whose steps have changed since it parked would be run, and
        # deleted, by the wrong code: the signal reaching another step, or an
        # IndexError. Refused, and put back as it was.
        then = (doc.definition or {}).get("steps")
        problem = (f"it is an {doc.kind} run" if doc.kind != "chain" else
                   "this chain's steps are not the ones it parked with "
                   f"({then} then, {self._signature()} now)"
                   if then is not None and then != self._signature() else None)
        if problem:
            self._store.save(run_id, raw)
            raise ValueError(f"cannot resume {run_id!r} with this chain: {problem}")
        # Restore the run's context unless the caller supplied a new one.
        if context is None and doc.context is not None:
            context = RunContext.from_dict(doc.context)
        if context is None:
            from ..state import current
            context = current()
        from ..state import StepStatus
        # FAILED: kept after a failure; RUNNING: the process died inside it.
        failed = next((st for st in doc.steps
                       if st["status"] in (StepStatus.FAILED, StepStatus.RUNNING)), None)
        if doc.suspended_step() is None:
            # Not parked: kept after a failure, or checkpointed by a process
            # that died. Resuming retries from the first step not done, never
            # the ones already done — an approved refund among them.
            if failed is not None:
                failed["status"] = StepStatus.PENDING
            signal = None
        start = doc.first_pending()
        if start is None:
            return None   # nothing left to run → idempotent no-op

        accumulated = dict(doc.variables)
        if ((doc.suspended_step() or {}).get("suspend") or {}).get("at") == "start":
            # Armed rather than paused inside a step: nothing has run, so there
            # is no suspended tool to complete. The signal is the run's input.
            if signal is not None and not isinstance(signal, dict):
                raise TypeError(
                    f"an armed chain starts from variables, so the signal must "
                    f"be a dict; got {type(signal).__name__}. The shape it was "
                    f"armed with is "
                    f"{((doc.suspended_step() or {}).get('suspend') or {}).get('resume_with')}.")
            accumulated.update(signal or {})
            signal = None
        u = doc.usage or {}
        from dataclasses import fields as _fields
        usage_in = Usage(**{f.name: u[f.name] for f in _fields(Usage) if f.name in u}) if u else None
        start_run(self, spent=u.get("cost") or 0.0)     # the parked part was spent already
        from ..state import using
        from .._events import running
        with using(context), lend(self.max_cost), running(doc.run_id):
            return self._run_from(doc, accumulated, start_idx=start,
                                  signal=signal,
                                  usage_in=usage_in, on_error=_on_error,
                                  context=context, resumed=True)

    def _signature(self) -> list:
        """What each step is, as a document can record it: its name, or the
        qualified name of the function or class it is — a run resumed by a
        chain with other steps would hand the signal to the wrong one."""
        out = []
        for runner, *_rest in self._steps:
            out.append(getattr(runner, "name", None) or getattr(runner, "__qualname__", None)
                       or type(runner).__qualname__)
        return out

    def _park(self, doc, idx, awaiting, accumulated, usage_total, history):
        """Persist the run as suspended at *idx* and return a SuspendedResult."""
        from ..state import StepStatus, SuspendedResult
        doc.steps[idx]["status"]  = StepStatus.SUSPENDED
        doc.steps[idx]["suspend"] = awaiting
        doc.variables = public(accumulated)
        doc.usage     = _usage_to_dict(usage_total)
        self._history     = history
        self._accumulated = dict(accumulated)
        self.last_usage   = usage_total
        self._store.save(doc.run_id, doc.to_dict())
        self._emit("run.suspended", step=idx, name=self.name, run_id=doc.run_id,
                   payload={"run_id": doc.run_id, **awaiting})
        return SuspendedResult(run_id=doc.run_id, awaiting=awaiting,
                               document=doc.to_dict())

    def _run_from(self, doc, accumulated, *, start_idx, signal, usage_in,
                  on_error, context, resumed: bool = False) -> ChainResult:
        """The step loop for ``run()`` (from step 0) and ``resume()`` (from the suspended
        step, which gets *signal* as ``_signal=``)."""
        from ..state import StepStatus, Suspend

        # Expose the per-request context for the duration of the run and persist
        # it in the document so it survives suspend/resume.
        self.context = context
        doc.context  = context.to_dict() if context is not None else None

        history:     list[dict] = []
        last_output: str | dict | None = None
        usage_total: Usage | None = usage_in
        checkpoints = [True]                     # off once a document will not serialise

        def checkpoint():
            """Save the document as it stands, so a process that dies mid-run resumes from
            the step it died in rather than re-running the ones before it."""
            if not checkpoints[0]:
                return
            doc.variables = public(accumulated)
            doc.usage     = _usage_to_dict(usage_total)
            try:
                self._store.save(doc.run_id, doc.to_dict())
            except ValueError as exc:
                checkpoints[0] = False
                warnings.warn(f"Chain run {doc.run_id}: not checkpointed after each step — {exc}",
                              RuntimeWarning, stacklevel=3)

        self._emit("run.resumed" if resumed else "run.started", step=start_idx, name=self.name,
                   run_id=doc.run_id, payload={"steps": len(self._steps)})
        for idx in range(start_idx, len(self._steps)):
            runner, output_key, input_map, kind, options = self._steps[idx]
            step_input  = dict(accumulated)          # snapshot before the step
            name        = getattr(runner, "name", None) or _runner_name(runner) or f"step_{idx}"
            step_signal = signal if idx == start_idx else None
            doc.steps[idx]["status"] = StepStatus.RUNNING

            self._emit("step.started", step=idx, name=name, run_id=doc.run_id,
                       payload={"kind": kind})
            run_usage = None

            try:
                if kind == "call":
                    args, kwargs = _build_call_kwargs(runner, accumulated, input_map)
                    output = runner(*args, **kwargs)

                elif kind == "tool":
                    kwargs = _build_tool_kwargs(runner, accumulated, input_map)
                    output = self._run_tool(runner, idx, kwargs, step_signal, doc.run_id)

                elif kind == "agent":
                    # An agent can park its own run, but the chain does not
                    # carry that pause: the pause belongs in the chain as a
                    # step when the chain is what waits.
                    task_key = options.get("task_key", _AGENT_DEFAULT_OPTIONS["task_key"])
                    if not accumulated.get(task_key, ""):
                        raise ValueError(
                            f"Agent step {idx} ({name!r}): accumulated variable "
                            f"{task_key!r} is empty or missing — an Agent step reads its "
                            f"task from {task_key!r}: store the previous step under the key "
                            f"{task_key!r}, e.g. (question_skill, {task_key!r}), or pass "
                            f"run(variables={{{task_key!r}: ...}})."
                        )
                    done = invoke(runner, accumulated, task_key=task_key,
                                  output_field=options.get(
                                      "output_field", _AGENT_DEFAULT_OPTIONS["output_field"]),
                                  label=f"Agent step {idx} ({name!r})")
                    if done.parked is not None:
                        raise RuntimeError(
                            f"Agent step {idx} ({name!r}) parked its own run "
                            f"({done.parked.awaiting.get('reason') or 'no reason given'}) "
                            f"and a chain cannot carry a child's pause. Put the "
                            f"Wait/Gate in this chain as its own step, or drive "
                            f"that agent outside the chain and resume it with "
                            f"agent.resume({done.parked.run_id!r}, signal=...)."
                        )
                    output = done.output
                    # The agent reports on its result, not through `last_usage`,
                    # which the accumulation below reads.
                    if done.usage is not None:
                        usage_total = done.usage if usage_total is None else usage_total + done.usage

                else:
                    values = dict(accumulated)
                    for dst, src in (input_map or {}).items():
                        # Resolved as a tool's are: a template filled, a name
                        # the chain lacks an error.
                        values[dst] = _resolve(src, accumulated, f"{name}.{dst}", top=True)
                    done = invoke(runner, values, mirror=True,
                                  label=f"Chain step {idx} ({name!r})")
                    run_usage = done.usage
                    if done.parked is not None:
                        raise RuntimeError(
                            f"Chain step {idx} ({name!r}) parked its own run "
                            f"({done.parked.awaiting.get('reason') or 'no reason given'}) and a chain "
                            "cannot carry a child's pause: put the Wait/Gate in this chain as "
                            "its own step, or run that chain on its own and resume it with "
                            f"its resume({done.parked.run_id!r}, signal=...).")
                    output = done.output

            except Suspend as susp:
                # Only a tool step is handed the signal on resume, so only a
                # tool may pause here: a function or a Skill raising Suspend
                # would be re-run without it and pause again, forever.
                if kind != "tool":
                    raise RuntimeError(
                        f"Chain step {idx} ({name!r}) asked to pause, and only a tool step can: "
                        "it is the one that is given the signal on resume. Use Wait or Gate "
                        "as the step.") from susp
                # A suspend tool paused the run: park the document and return a
                # SuspendedResult; on resume the engine re-invokes this step.
                awaiting = {"reason":      susp.reason,
                            "resume_with": susp.resume_with,
                            "hint":        susp.hint}
                return self._park(doc, idx, awaiting, accumulated,
                                  usage_total, history)

            except Exception as exc:
                history.append({
                    "step":       idx,
                    "kind":       kind,
                    "name":       name,
                    "input":      public(step_input),
                    "output":     None,
                    "output_key": output_key,
                    "options":    options,
                    # The type and the traceback, not just the message:
                    # `KeyError('tenant')` renders as `'tenant'`, which reads
                    # as a value rather than a fault.
                    "error":      str(exc) or type(exc).__name__,
                    "failure":    _describe(exc),
                })

                step_on_error = self._extra[idx]["on_error"] or on_error
                if step_on_error in ("raise", "stop"):
                    # Terminal exit, with the step marked failed. A resumed run
                    # is kept so: the store already gave it up to this resume,
                    # and a transient failure after an approved step would
                    # otherwise lose the run — resume() retries from this step.
                    doc.steps[idx]["status"] = StepStatus.FAILED
                    if resumed:
                        doc.variables = public(accumulated)
                        doc.usage     = _usage_to_dict(usage_total)
                        self._store.save(doc.run_id, doc.to_dict())
                    else:
                        self._store.delete(doc.run_id)     # its checkpoints
                    self._history     = history
                    self._accumulated = dict(accumulated)
                    self.last_usage   = usage_total
                    if step_on_error == "raise":
                        raise
                    return self._result(doc, history, accumulated, usage_total, last_output)

                # skip / collect: record a terminal SKIPPED status so resume
                # does not re-run this step (first_pending would land on it).
                doc.steps[idx]["status"] = StepStatus.SKIPPED
                doc.steps[idx].pop("suspend", None)
                doc.failures.append({"step": idx, "name": name, "type": type(exc).__name__,
                                     "message": str(exc)})
                checkpoint()
                if step_on_error == "skip":
                    # Logged, every time: a warning is shown once per place in
                    # the code, and that place is inside the library.
                    _skip_log.warning(
                        f"Chain step {idx} ({name!r}) failed and was skipped: "
                        f"{type(exc).__name__}: {exc}. Downstream steps that "
                        f"read {output_key!r} will receive a stale or absent "
                        "value."
                    )
                continue

            history.append({
                "step":       idx,
                "kind":       kind,
                "name":       name,
                "input":      public(step_input),
                "output":     output,
                "output_key": output_key,
                "options":    options,
                **({"default_key": True} if self._extra[idx]["default_key"] else {}),
            })

            # A Skill's or a chain's from its own run; a shared runner's
            # `last_usage` is whichever thread wrote it last.
            step_usage = run_usage if kind not in ("call", "tool", "agent") \
                else getattr(runner, "last_usage", None) if kind != "agent" else None
            if step_usage:
                usage_total = step_usage if usage_total is None else usage_total + step_usage

            self._store_output(idx, name, output_key, output, accumulated)

            last_output = output

            # Checkpoint: the step is done; variables reflect its output.
            doc.steps[idx]["status"] = StepStatus.DONE
            doc.steps[idx].pop("suspend", None)
            checkpoint()

            self._emit("step.ended", step=idx, name=name, run_id=doc.run_id,
                       payload={"kind": kind})

        # Completed run: a finished run leaves no parked document (no-op when
        # the run never suspended and was therefore never stored).
        self._store.delete(doc.run_id)
        self._history     = history
        self._accumulated = dict(accumulated)
        self.last_usage   = usage_total
        result = self._result(doc, history, accumulated, usage_total, last_output)
        self._emit("run.finished", name=self.name, run_id=doc.run_id,
                   payload={"success": result.success}, cost=result.cost,
                   usage=usage_total.total_tokens if usage_total else None, error=result.error)
        return result

    @staticmethod
    def _result(doc, history, accumulated, usage_total, last_output) -> ChainResult:
        # By the failure record, not by its message: `TimeoutError()` and a
        # bare `assert` have an empty one, which would read as success.
        # Failures from before a pause come from the document, where they
        # were kept; the history of this process starts at the resume.
        seen = {r["step"] for r in history if r.get("failure") or r.get("error")}
        failed = ([{"step": f["step"], "name": f["name"], "error": f["message"] or f["type"]}
                   for f in doc.failures if f["step"] not in seen]
                  + [r for r in history if r.get("failure") or r.get("error")])
        failed.sort(key=lambda r: r["step"])
        error = (f"step {failed[0]['step']} ({failed[0]['name']}): {failed[0]['error']}"
                 if failed else None)
        renamed = {}
        for r in history:
            if r.get("default_key") and r.get("output_key") != "result" and not (
                    isinstance(r.get("output"), dict) and not isinstance(r.get("output"), Record)):
                renamed["result"] = r["output_key"]
        return ChainResult(output=last_output, success=not failed, error=error, renamed=renamed,
                           history=list(history), variables=dict(accumulated),
                           usage=usage_total, run_id=doc.run_id)

    @property
    def accumulated(self) -> dict:
        """
        The variables after the most recent ``run()`` — initial ones, step outputs and
        keys tools wrote; deprecated, goes in 5.0 — read ``result.variables``.

        A shallow copy; empty before the first ``run()``.
        """
        warnings.warn("chain.accumulated is the last run's result.variables; read it from "
                      "the result — it goes in 5.0", FutureWarning, stacklevel=2)
        return dict(self._accumulated)

    @property
    def history(self) -> list[dict]:
        """
        Step records from the most recent ``run()``; deprecated, goes in 5.0 — read
        ``result.history``.

        Each record holds ``step`` (0-based), ``kind`` (``"skill"``, ``"tool"``,
        ``"agent"`` or ``"call"``), ``name``, ``input`` (the variables before the
        step), ``output`` and ``output_key``. A copy; empty before the first run.
        """
        warnings.warn("chain.history is the last run's result.history; read it from the "
                      "result — it goes in 5.0", FutureWarning, stacklevel=2)
        return list(self._history)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(steps) -> tuple[list, list]:
        """*steps* as ``(runner, output_key, input_map, kind, options)`` tuples, and beside
        them what each says beyond that: its ``on_error``, ``spread``, and given key."""
        result: list = []
        extras: list = []
        for item in steps:
            extra: dict = {"on_error": None, "spread": None, "default_key": False}
            if isinstance(item, Step):
                runner, key = item.runner, item.out
                input_map, options = dict(item.inputs or {}), dict(item.options or {})
                extra.update(on_error=item.on_error, spread=item.spread)
                item = None
            elif isinstance(item, list):
                item = tuple(item)                 # a step from YAML: [runner, "key"]
            if isinstance(item, tuple):
                if len(item) == 2 and isinstance(item[1], dict):
                    runner, input_map = item       # (runner, inputs): key is the name
                    key, options = None, {}
                elif len(item) == 2:
                    runner, key = item
                    input_map   = {}
                    options     = {}
                elif len(item) == 3:
                    runner, key, input_map = item
                    options = {}
                    if not isinstance(input_map, dict):
                        raise ValueError(
                            "The third element of a step tuple must be a dict "
                            f"(input_map); got {input_map!r}"
                        )
                elif len(item) == 4:
                    runner, key, input_map, options = item
                    if not isinstance(input_map, dict):
                        raise ValueError(
                            "The third element of a step tuple must be a dict "
                            f"(input_map); got {input_map!r}"
                        )
                    if not isinstance(options, dict):
                        raise ValueError(
                            "The fourth element of a step tuple must be a dict "
                            f"(options); got {options!r}"
                        )
                else:
                    raise ValueError(
                        "Step tuples must have 2, 3, or 4 elements: "
                        "(runner, output_key), (runner, input_map), "
                        "(runner, output_key, input_map), or "
                        "(runner, output_key, input_map, options); "
                        f"got a {len(item)}-element tuple"
                    )
                if not isinstance(key, (str, type(None))):
                    raise ValueError(
                        f"output_key must be a str; got {key!r}"
                    )
            elif item is not None:
                runner    = item
                key       = None
                input_map = {}
                options   = {}

            if key is None:
                # Named after its runner, so unnamed steps do not overwrite
                # one shared "result". Still written there too for now; see
                # `_store_output`.
                key = _runner_name(runner)
                extra["default_key"] = True
                if key is None:
                    named = getattr(runner, "name", None)
                    why = (f"its runner's name {named!r} is not one a key can be "
                           "(letters, digits and _)" if isinstance(named, str) and named
                           else "its runner no name")
                    warnings.warn(
                        f"a step running {type(runner).__name__} has no key and {why}, "
                        "so its output goes to 'result', which the next such step "
                        "overwrites; from 5.0 this is an error. Give it one: (runner, 'key'), "
                        "or name the runner.", FutureWarning, stacklevel=4)
                    key, extra["default_key"] = "result", False

            if extra["spread"] is None:
                from ..tools.general._wait import Gate, Wait
                if isinstance(runner, (Wait, Gate)):
                    # A pause's answer is the variables its `resume_with`
                    # declares, so spreading it is what the step is for, and
                    # it says nothing.
                    extra["spread"] = True

            if _is_tool(runner):
                kind = "tool"
            elif _is_agent(runner):
                kind = "agent"
            elif callable(runner) and not hasattr(runner, "run"):
                # A function or a utility's method — `kb.upsert` — called with
                # the arguments its signature names.
                kind = "call"
            elif callable(getattr(runner, "run", None)):
                kind = "skill"
            else:
                # Refused here, not only when run() reaches it.
                raise TypeError(f"a chain step is a Skill, a tool, an Agent, a Chain or a "
                                f"function; got {type(runner).__name__} {runner!r}")

            if kind == "agent":
                if input_map:
                    raise ValueError(
                        f"an Agent step takes no input_map — it reads its task from one variable "
                        f"(options={{'task_key': …}}) and is given all of them; got {input_map}")
                if unknown := sorted(set(options) - set(_AGENT_DEFAULT_OPTIONS)):
                    raise ValueError(f"an Agent step's options are {sorted(_AGENT_DEFAULT_OPTIONS)}; "
                                     f"got {unknown}")
                from ..agent._result import AgentResult
                field = options.get("output_field", _AGENT_DEFAULT_OPTIONS["output_field"])
                if field not in AgentResult.__dataclass_fields__:
                    raise ValueError(f"output_field {field!r} is not a field of AgentResult; "
                                     f"it has {sorted(AgentResult.__dataclass_fields__)}")
            elif options:
                raise ValueError(f"step options are an Agent step's only ({sorted(_AGENT_DEFAULT_OPTIONS)}); "
                                 f"this {kind} step was given {sorted(options)}, which nothing would read")

            result.append((runner, key, input_map, kind, options))
            extras.append(extra)

        # Two steps named alike would write one key: the later wins, which is
        # the overwrite the runner's name is meant to end.
        seen: dict = {}
        for i, ((runner, key, *_), extra) in enumerate(zip(result, extras)):
            if extra["default_key"] and key in seen:
                raise ValueError(f"steps {seen[key]} and {i} both write {key!r} — two steps run "
                                 f"{key!r} with no key of their own; give one a key: "
                                 f"({key}, '{key}_2')")
            seen.setdefault(key, i)
        return result, extras

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        step_names = [
            f"[{kind[0].upper()}]{getattr(runner, 'name', None) or type(runner).__name__}→{key}"
            for runner, key, _, kind, _opts in self._steps
        ]
        parts = [f"steps=[{', '.join(step_names)}]"]
        if self.name:
            parts.append(f"name={self.name!r}")
        if self.on_step_error != "raise":
            parts.append(f"on_step_error={self.on_step_error!r}")
        return f"Chain({', '.join(parts)})"
