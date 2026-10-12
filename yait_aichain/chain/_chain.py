"""
chain._chain
============

``Chain`` — a sequential pipeline of :class:`~skills.Skill`,
:class:`~tools.Tool` and :class:`~agent.Agent` instances, plain functions and
utilities' methods.

A Chain wires steps together so each step's output automatically feeds the
next step's input as a named variable.  The kinds of step can be freely mixed
in any order; a pause for a person or a signal is a step too, made with
``wait(...)`` or ``gate(tool, ...)``.

Step specification
------------------
Each element of *steps* may be any of the following forms:

  * ``runner``
        Bare Skill, tool, Agent, or function.  Output stored under
        ``"result"``.

  * ``(runner, output_key: str)``
        String output stored under *output_key*.

  * ``(runner, output_key: str, input_map: dict)``
        Same as above **plus** an *input_map* that fills the step's
        arguments from the accumulated variables.

        ``input_map`` has the form ``{"param_name": value}``, where *value*
        is a variable's name or a ``"{template}"``.  ``{"source": "url"}``
        passes ``accumulated["url"]`` as the ``source`` argument;
        ``{"text": "Title: {french}"}`` passes the template filled in.

  * ``(runner, output_key: str, input_map: dict, options: dict)``
        Same as above **plus** an *options* dict for step-level settings.
        Currently meaningful only for Agent steps:

        ``task_key``     (str,  default ``"task"``)
            Name of the accumulated variable that holds the agent's task
            string.  The task is read from ``accumulated[task_key]``
            before the agent runs.

        ``output_field`` (str,  default ``"output"``)
            Which field of the :class:`~agent.AgentResult` to extract as
            the step's output.  Use ``"output"`` for the final answer
            (default), or any other ``AgentResult`` attribute name.

Variable flow
-------------
1. The chain starts with *initial variables* — merged from
   ``Chain.variables`` and ``run(variables=...)``, call-time values win.

2. Before each step the *accumulated variable dict* is inspected to build
   the step's input:

   * **Skill** — the full accumulated dict is passed as
     ``skill.run(variables=accumulated)``.  The skill's template engine
     picks the placeholders it needs.

   * **Tool**  — kwargs are assembled by matching the tool's declared
     ``parameters["properties"]`` keys against the accumulated dict
     (with optional *input_map* renaming).  Only declared parameters that
     have a matching accumulated variable are forwarded, so extra variables
     are silently ignored and never cause unexpected keyword errors.

   * **Function / method** — called with the arguments its signature names,
     filled the same way; a required one the chain cannot fill is an error.

   * **Agent** — ``agent.run(task=accumulated[task_key],
     variables=accumulated)`` is called.  The task string is read from
     ``accumulated[task_key]`` (default key: ``"task"``).  The
     ``AgentResult`` field named by ``output_field`` (default:
     ``"output"``) is extracted as the step's output value.

3. After each step the output is merged back into the accumulated dict:

     * ``str`` output  →  ``accumulated[output_key] = output``
     * ``dict`` output →  ``accumulated.update(output)``

4. ``run()`` returns a :class:`ChainResult`: ``.output`` is the final step's
   output and ``result["key"]`` any step's.

History
-------
After each ``run()`` call, ``chain.history`` holds one record per step::

    {
        "step":       int,                    # 0-based index
        "kind":       "skill"|"tool"|"agent"|"call", # type of runner
        "name":       str,                    # runner.name or "step_N"
        "input":      dict,                   # accumulated vars before step
        "output":     str|dict,               # raw output from the runner
        "output_key": str,                    # variable name for str output
        "options":    dict,                   # step-level options (agents)
    }
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import SuspendedResult

from .._require import require

import importlib
import os
import warnings

from .. import _hints
from ..models._usage import Usage
from .._events import Event, emit
from ._result import ChainResult
from ..tools._base import Record
from .._template import _PLACEHOLDER_RE, substitute_placeholders

# One vocabulary with Pool — see `_errors_policy`. `collect` is new here and
# is a silent `skip`: a chain of a hundred generated steps that expects a few
# to fail should not have to choose between a warning per failure and no
# record at all.
from .._errors_policy import POLICIES as _VALID_ON_STEP_ERROR, describe as _describe

# Default options applied to every Agent step unless overridden.
_AGENT_DEFAULT_OPTIONS: dict = {
    "task_key":     "task",
    "output_field": "output",
}


def _require_safe_tool_class(class_path: str, trusted: bool) -> None:
    """
    Guard against arbitrary-code-execution when loading a chain file.

    A serialised chain names tool classes by import path; importing one runs
    its module/constructor. Unless *trusted*, only classes under the
    ``yait_aichain`` package are allowed, so a tampered or untrusted file
    cannot name (e.g.) ``subprocess.Popen`` and have it instantiated.
    """
    if trusted:
        return
    module_path = class_path.rsplit(".", 1)[0]
    if module_path != "yait_aichain" and not module_path.startswith("yait_aichain."):
        raise ValueError(
            f"Refusing to import tool class {class_path!r} from outside "
            f"'yait_aichain' while loading an untrusted chain file. Pass "
            f"trusted=True to Chain.load() only for files you fully control."
        )


def _usage_to_dict(usage) -> dict:
    """Serialise a ``Usage`` (or None) for the run document."""
    if not usage:
        return {}
    return {
        "input_tokens":  usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens":  usage.total_tokens,
        "cost":          usage.cost,
    }


from .._runners import is_tool as _is_tool, is_agent as _is_agent


def _build_tool_kwargs(tool, accumulated: dict, input_map: dict) -> dict:
    """
    Build the ``**kwargs`` dict to pass to ``tool.run()``.

    A parameter the step's *input_map* does not mention takes the variable of
    the same name, when there is one, and is otherwise left to the tool's own
    default. A parameter it does mention is filled from its value:

      * ``"draft"`` — a variable's name: that variable's value;
      * ``"Draw {topic} in watercolor"`` — a template: the variables
        substituted, as a Skill substitutes its prompt;
      * ``"{candidates}"`` — one placeholder alone: the variable's value itself,
        so a list stays a list;
      * ``{"query": "{question}", "top_n": 5}`` — a dict, resolved key by key:
        templates are filled, and anything else — a number, a plain string — is
        a literal.

    A name or a template naming a variable the chain does not have, and a
    parameter the tool does not take, are errors that say which.

    Parameters
    ----------
    tool       : Tool instance whose ``.parameters`` schema is inspected.
    accumulated: Current accumulated variable dict.
    input_map  : ``{"tool_param": value}`` — see above.
    """
    props  = tool.parameters.get("properties", {})
    name   = getattr(tool, "name", None) or type(tool).__name__
    if props and (unknown := [k for k in input_map if k not in props]):
        raise ValueError(f"the step maps {unknown} into {name!r}, which takes {sorted(props)}")
    kwargs = {}
    for param in props:
        if param in input_map:
            kwargs[param] = _resolve(input_map[param], accumulated, f"{name}.{param}", top=True)
        elif param in accumulated:
            kwargs[param] = accumulated[param]
    return kwargs


def _callable_name(f) -> str:
    owner = getattr(f, "__self__", None)
    return (f"{type(owner).__name__}.{f.__name__}" if owner is not None and hasattr(f, "__name__")
            else getattr(f, "__name__", None) or type(f).__name__)


def _build_call_kwargs(f, accumulated: dict, input_map: dict) -> dict:
    """
    The arguments for a function or a method step: each parameter its
    signature names, filled as a tool's are (see `_build_tool_kwargs`); a
    required one the chain cannot fill is an error naming it.
    """
    import inspect
    params = {n: p for n, p in inspect.signature(f).parameters.items()
              if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    where = _callable_name(f)
    if unknown := [k for k in input_map if k not in params]:
        raise ValueError(f"the step maps {unknown} into {where!r}, which takes {list(params)}")
    kwargs = {}
    for name, p in params.items():
        if name in input_map:
            kwargs[name] = _resolve(input_map[name], accumulated, f"{where}.{name}", top=True)
        elif name in accumulated:
            kwargs[name] = accumulated[name]
        elif p.default is p.empty:
            raise ValueError(f"{where} needs {name!r}, and the chain has no variable of that name — "
                             f"map one: ({where}, key, {{{name!r}: 'variable'}}); it has {sorted(accumulated)}")
    return kwargs


def _refuse_function_tool(t) -> None:
    """A @tool function is code: a saved chain could name it and never bring it back."""
    from ..tools._base import FunctionTool
    if isinstance(t, FunctionTool):
        raise TypeError(f"tool {t.name!r} is a @tool function, and a function cannot be saved — "
                        "it is code, not configuration. Build this chain in code, or make the "
                        "tool a Tool subclass whose settings can be written down.")


def _resolve(value, accumulated: dict, where: str, top: bool = False):
    """One value of a step's input_map, filled from the chain's variables."""
    if isinstance(value, str):
        names = _PLACEHOLDER_RE.findall(value)
        if not names:
            if not top:
                return value                                  # a literal inside a dict
            if value not in accumulated:
                raise ValueError(f"{where} is mapped to the variable {value!r}, which the chain does "
                                 f"not have; it has {sorted(accumulated)}")
            return accumulated[value]
        if missing := [n for n in dict.fromkeys(names) if n not in accumulated]:
            raise ValueError(f"{where}: the template {value!r} names {missing}, which the chain does "
                             f"not have; it has {sorted(accumulated)}")
        if (alone := _PLACEHOLDER_RE.fullmatch(value)) is not None:
            return accumulated[alone.group(1)]                # the value itself, not its text
        return substitute_placeholders(value, accumulated)
    if isinstance(value, dict):
        return {k: _resolve(v, accumulated, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, accumulated, where) for v in value]
    return value                                              # a number, a bool: a literal


class Chain:
    """
    Steps in a fixed order, sharing one set of variables.

    A step is a ``Skill``, a tool, an ``Agent``, or a plain function or a
    utility's method, in any order. A Skill reads any variable in its prompt;
    a tool or a function gets the arguments its schema or signature names,
    filled from the variables or from the step's mapping. Each step's output
    is stored under its key; a plain dict a step returns is merged into the
    variables, so one step can write several.

    ::

        chain = Chain(steps=[
            (summarise,          "summary"),
            (translate,          "french"),
            (count_words,        "length", {"text": "Title: {french}"}),
            (shout,              "loud"),
        ], max_cost=0.05)
        result = chain.run(variables={"text": "…"})
        result["french"], result.output, result.cost

    Parameters
    ----------
    steps : list
        The steps, in order. Each is a runner on its own, or a tuple:
        ``(runner, output_key)``; ``(runner, output_key, input_map)``, where
        *input_map* maps an argument name to a variable's name or to a
        ``"{template}"`` the chain fills in; or ``(runner, output_key,
        input_map, options)``. For an Agent step, *options* takes
        ``task_key`` (the variable holding its task, default ``"task"``) and
        ``output_field`` (the ``AgentResult`` attribute stored, default
        ``"output"``). A bare runner is stored under ``"result"``.
    variables : dict | None, optional
        Default variables for every ``run()``; call-time values win.
    name : str | None, optional
        A label, shown in events and when the chain is a step elsewhere.
    description : str | None, optional
        What the chain does.
    on_step_error : str, optional
        What a failed step does to the run: ``"raise"`` (the default)
        propagates it, ``"stop"`` ends the run without raising, ``"skip"``
        records it, warns and goes on, ``"collect"`` does the same without
        the warning.
    store : StateStore, optional
        Where a paused run is kept. The default is process-local, so a run
        that waits for a person or a webhook needs ``FileStore(...)`` or
        another persistent store to be resumed from another process.
    hooks : list | None, optional
        Callables given an ``Event`` at every step boundary.
    max_cost : float | Budget | None, optional
        A ceiling in dollars for the whole chain, shared by every step; a
        step that would begin past it is refused with ``BudgetExceeded``.

    Raises
    ------
    ValueError
        If *steps* is empty, a tuple has an unsupported shape, or
        *on_step_error* is not one of the four.
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
    ) -> None:
        if not steps:
            raise ValueError("Chain requires at least one step.")
        if on_step_error not in _VALID_ON_STEP_ERROR:
            raise ValueError(
                f"on_step_error must be one of {sorted(_VALID_ON_STEP_ERROR)}; "
                f"got {on_step_error!r}"
            )
        self._steps         = self._normalise(steps)
        self.variables      = variables or {}
        self.name           = name
        self.description    = description
        self.on_step_error  = on_step_error
        self.hooks          = list(hooks or [])     # observability hooks
        # A store is always present (default: process-local in-memory) so the
        # engine has one uniform path; pass store= for persistence (FileStore,
        # S3, …). It holds suspended runs for resume.
        from ..state import InMemoryStore
        self._store = store or InMemoryStore()
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
        from .._budget import as_budget
        self.max_cost = as_budget(max_cost)

        self.last_usage: Usage | None = None
        # Per-request RunContext for the current run() / resume() (tenant +
        # metadata); set while a run is in flight, restored on resume.
        self.context = None

    def _emit(self, etype: str, **fields) -> None:
        """Dispatch a step-boundary :class:`Event` to registered hooks."""
        if self.hooks:
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
            Variables for this invocation.  Merged with ``self.variables``;
            call-time values win.
        on_step_error : str | None, optional
            Override the instance-level ``on_step_error`` for this call only.
            Must be ``"raise"``, ``"stop"``, ``"skip"`` or ``"collect"`` when
            provided.
        context : RunContext, optional
            The tenant and metadata of this request, ambient for every step; a
            ``Model(api_key=callable)`` resolves its key from it.

        Returns
        -------
        ChainResult
            ``.output`` is the last successful step's output, ``result["key"]``
            any step's output by its key, ``.success`` whether every step
            completed. A :class:`~state.SuspendedResult` instead when a step
            paused the run.

        Raises
        ------
        RuntimeError, clients._base.APIError, …
            Any step exception, when ``on_step_error`` is ``"raise"``
            (the default).
        ValueError
            If *on_step_error* override value is not recognised.
        """
        _on_error = self.on_step_error if on_step_error is None else on_step_error
        if _on_error not in _VALID_ON_STEP_ERROR:
            raise ValueError(
                f"on_step_error must be one of {sorted(_VALID_ON_STEP_ERROR)}; "
                f"got {_on_error!r}"
            )

        from ..state import RunDocument
        accumulated = {**self.variables, **(variables or {})}
        step_names  = [getattr(r, "name", None) or f"step_{i}"
                       for i, (r, *_rest) in enumerate(self._steps)]
        doc = RunDocument.new("chain", step_names, variables=accumulated)
        from ..state import using
        with using(context), self._lend_budget():
            return self._run_from(doc, accumulated, start_idx=0, signal=None,
                                  usage_in=None, on_error=_on_error,
                                  context=context)

    def arm(self, reason: str = "", resume_with: dict | None = None, *,
            hint: dict | None = None, variables: dict | None = None,
            context=None) -> SuspendedResult:
        """
        Create the run **without starting it**, and return it parked.

        The run exists before it runs: it has a ``run_id``, a durable document
        in the store, the variables it was seeded with, and a declared shape
        for the signal that will start it. That is a subscription — the thing
        a trigger waits on and a person is shown a form for — expressed as the
        same object every other pause produces, so a webhook, a cron tick and
        a human clicking *start* all end in ``resume(run_id, signal)``.

        The engine still schedules nothing. ``hint`` is recorded for whatever
        does: ``{"on": "order.created"}``, ``{"wake_at": "2026-06-15T09:00"}``.

        On resume the signal **seeds the variables** rather than being handed
        to a step, because nothing has run yet and there is no suspended tool
        to complete.

        Parameters
        ----------
        reason : str, optional
            What the run waits for, shown to whoever starts it.
        resume_with : dict | None, optional
            The shape of the signal that starts it — a JSON-Schema-like description.
        hint : dict | None, optional
            Recorded for whatever does the waking: ``{"on": "order.created"}``,
            ``{"wake_at": "…"}``. The engine schedules nothing.
        variables : dict | None, optional
            Variables the run is seeded with before the signal adds its own.
        context : RunContext, optional
            The request's tenant and metadata, kept with the run.
        """
        from ..state import RunDocument, StepStatus, SuspendedResult

        if not self._steps:
            raise ValueError("a chain with no steps has nothing to arm")

        accumulated = {**self.variables, **(variables or {})}
        step_names  = [getattr(r, "name", None) or f"step_{i}"
                       for i, (r, *_rest) in enumerate(self._steps)]
        doc = RunDocument.new("chain", step_names, variables=accumulated)
        doc.context = context.to_dict() if context is not None else None

        awaiting = {"reason": reason, "resume_with": dict(resume_with or {}),
                    "hint": hint, "at": "start"}
        doc.steps[0]["status"]  = StepStatus.SUSPENDED
        doc.steps[0]["suspend"] = awaiting
        self._store.save(doc.run_id, doc.to_dict())
        self._emit("run.armed", step=0, payload={"run_id": doc.run_id, **awaiting})
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
        Resume a previously suspended run.

        Loads the run document from the store, re-invokes the suspended step
        with *signal*, and continues from the first step that is not yet
        ``done`` — already-completed steps are never re-run (their outputs are
        in the document's variables).

        Idempotent: a ``run_id`` no longer in the store (already resumed to
        completion, or unknown) raises ``KeyError`` so an at-least-once trigger
        can treat that as "already handled".

        Parameters
        ----------
        run_id : str
            The parked run, as the ``SuspendedResult`` named it.
        signal : any, optional
            The answer the run waited for. A dict is merged into the variables.
        on_step_error : str | None, optional
            Overrides the chain's policy for the rest of this run.
        context : RunContext, optional
            The tenant and metadata for the resumed part; the stored one when
            omitted.
        """
        _on_error = self.on_step_error if on_step_error is None else on_step_error
        if _on_error not in _VALID_ON_STEP_ERROR:
            raise ValueError(
                f"on_step_error must be one of {sorted(_VALID_ON_STEP_ERROR)}; "
                f"got {_on_error!r}"
            )

        from ..state import RunDocument, RunContext
        from ..models._usage import Usage

        raw = self._store.load(run_id)
        if raw is None:
            raise KeyError(
                f"No suspended run {run_id!r} in the store "
                f"(already resumed/completed, or unknown id)."
            )
        doc   = RunDocument.from_dict(raw)
        # Restore the run's context unless the caller supplied a new one.
        if context is None and doc.context is not None:
            context = RunContext.from_dict(doc.context)
        start = doc.first_pending()
        if start is None or doc.suspended_step() is None:
            return None   # not actually suspended → idempotent no-op

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
        usage_in = Usage(
            input_tokens  = u.get("input_tokens", 0),
            output_tokens = u.get("output_tokens", 0),
            total_tokens  = u.get("total_tokens", 0),
            cost          = u.get("cost"),
        ) if u else None
        from ..state import using
        with using(context), self._lend_budget():
            return self._run_from(doc, accumulated, start_idx=start,
                                  signal=signal,
                                  usage_in=usage_in, on_error=_on_error,
                                  context=context)

    def _lend_budget(self):
        """Put this chain's budget on every step that can hold one.

        Lent, not injected at construction: the steps arrive already built,
        and a caller who set a budget on the chain means the chain's whole
        spend rather than each step's. Restored afterwards, because a Skill
        reused in two chains must not keep the first one's ceiling — that
        would be the chain quietly editing an object it does not own.
        """
        import contextlib

        @contextlib.contextmanager
        def _lent():
            if self.max_cost is None:
                yield
                return
            touched = []
            for entry in self._steps:
                runner = entry[0]
                if hasattr(runner, "max_cost"):
                    touched.append((runner, runner.max_cost))
                    runner.max_cost = self.max_cost
            try:
                yield
            finally:
                for runner, previous in touched:
                    runner.max_cost = previous
        return _lent()

    def _park(self, doc, idx, awaiting, accumulated, usage_total, history):
        """Persist the run as suspended at *idx* and return a SuspendedResult."""
        from ..state import StepStatus, SuspendedResult
        doc.steps[idx]["status"]  = StepStatus.SUSPENDED
        doc.steps[idx]["suspend"] = awaiting
        doc.variables = dict(accumulated)
        doc.usage     = _usage_to_dict(usage_total)
        self._history     = history
        self._accumulated = dict(accumulated)
        self.last_usage   = usage_total
        self._store.save(doc.run_id, doc.to_dict())
        return SuspendedResult(run_id=doc.run_id, awaiting=awaiting,
                               document=doc.to_dict())

    def _run_from(self, doc, accumulated, *, start_idx, signal, usage_in,
                  on_error, context) -> ChainResult:
        """
        Shared step loop for ``run()`` (from step 0) and ``resume()`` (from the
        suspended step). On ``resume``, *signal* is delivered to the step at
        *start_idx* (the suspended tool) via ``_signal=``.
        """
        from ..state import StepStatus, Suspend, SuspendedResult

        # Expose the per-request context for the duration of the run and persist
        # it in the document so it survives suspend/resume.
        self.context = context
        doc.context  = context.to_dict() if context is not None else None

        history:     list[dict] = []
        last_output: str | dict | None = None
        usage_total: Usage | None = usage_in

        for idx in range(start_idx, len(self._steps)):
            runner, output_key, input_map, kind, options = self._steps[idx]
            step_input  = dict(accumulated)          # snapshot before the step
            name        = getattr(runner, "name", None) or f"step_{idx}"
            step_signal = signal if idx == start_idx else None

            self._emit("step.started", step=idx, name=name, payload={"kind": kind})

            try:
                if kind == "call":
                    output = runner(**_build_call_kwargs(runner, accumulated, input_map))

                elif kind == "tool":
                    kwargs = _build_tool_kwargs(runner, accumulated, input_map)
                    if step_signal is not None:
                        output = runner.run(_signal=step_signal, **kwargs)
                    else:
                        output = runner.run(**kwargs)

                elif kind == "agent":
                    # An agent step always starts a run. An agent can park
                    # its own run, but the chain does not carry that pause:
                    # two levels of resume would need the child's run id
                    # threaded through this document, and the pause belongs
                    # in the chain as a step when the chain is what waits.
                    task_key = options.get("task_key", _AGENT_DEFAULT_OPTIONS["task_key"])
                    task     = accumulated.get(task_key, "")
                    if not task:
                        raise ValueError(
                            f"Agent step {idx} ({name!r}): accumulated variable "
                            f"{task_key!r} is empty or missing — an Agent step reads its "
                            f"task from {task_key!r}: store the previous step under the key "
                            f"{task_key!r}, e.g. (question_skill, {task_key!r}), or pass "
                            f"run(variables={{{task_key!r}: ...}})."
                        )
                    agent_result = runner.run(task=task, variables=accumulated)

                    if isinstance(agent_result, SuspendedResult):
                        raise RuntimeError(
                            f"Agent step {idx} ({name!r}) parked its own run "
                            f"({agent_result.awaiting.get('reason') or 'no reason given'}) "
                            f"and a chain cannot carry a child's pause. Put the "
                            f"Wait/Gate in this chain as its own step, or drive "
                            f"that agent outside the chain and resume it with "
                            f"agent.resume({agent_result.run_id!r}, signal=...)."
                        )
                    if not agent_result:
                        raise RuntimeError(
                            f"Agent step {idx} ({name!r}) failed: "
                            f"{getattr(agent_result, 'error', 'no result returned')}"
                        )

                    output_field = options.get(
                        "output_field", _AGENT_DEFAULT_OPTIONS["output_field"]
                    )
                    output = getattr(agent_result, output_field, None)

                    # Fold the agent's token usage into the chain total. An
                    # Agent has no ``last_usage`` (the generic accumulation
                    # below would miss it); it reports tokens on AgentResult.
                    _atok = getattr(agent_result, "tokens_used", 0) or 0
                    if _atok:
                        usage_total = (Usage(total_tokens=_atok) if usage_total is None
                                       else usage_total + Usage(total_tokens=_atok))

                else:
                    # Skill: pass full accumulated dict, honouring input_map renames.
                    if input_map:
                        skill_vars = dict(accumulated)
                        for dst, src in input_map.items():
                            if src in accumulated:
                                skill_vars[dst] = accumulated[src]
                        output = runner.run(variables=skill_vars)
                    else:
                        output = runner.run(variables=accumulated)
                    # A chain used as a step contributes its output, and its
                    # failure is this step's failure — the same contract an
                    # agent step has.
                    if isinstance(output, ChainResult):
                        if not output:
                            raise RuntimeError(
                                f"Chain step {idx} ({name!r}) failed: {output.error}")
                        output = output.output

            except Suspend as susp:
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
                    "input":      step_input,
                    "output":     None,
                    "output_key": output_key,
                    "options":    options,
                    # The type and the traceback, not just the message:
                    # `KeyError('tenant')` renders as `'tenant'`, which reads
                    # as a value rather than a fault.
                    "error":      str(exc),
                    "failure":    _describe(exc),
                })

                if on_error in ("raise", "stop"):
                    # Terminal exit: mark the step failed and drop any parked
                    # document so a re-resume can't re-run the already-attempted
                    # steps (a duplicate trigger becomes a no-op / KeyError).
                    doc.steps[idx]["status"] = StepStatus.FAILED
                    self._store.delete(doc.run_id)
                    self._history     = history
                    self._accumulated = dict(accumulated)
                    self.last_usage   = usage_total
                    if on_error == "raise":
                        raise
                    return self._result(doc, history, accumulated, usage_total, last_output)

                # skip / collect: record a terminal SKIPPED status so resume
                # does not
                # re-run this step (first_pending would otherwise land on it).
                doc.steps[idx]["status"] = StepStatus.SKIPPED
                doc.steps[idx].pop("suspend", None)
                doc.variables = dict(accumulated)
                if on_error == "skip":
                    warnings.warn(
                        f"Chain step {idx} ({name!r}) failed and was skipped: "
                        f"{type(exc).__name__}: {exc}. Downstream steps that "
                        f"read {output_key!r} will receive a stale or absent "
                        "value.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue

            history.append({
                "step":       idx,
                "kind":       kind,
                "name":       name,
                "input":      step_input,
                "output":     output,
                "output_key": output_key,
                "options":    options,
            })

            step_usage = getattr(runner, "last_usage", None)
            if step_usage:
                usage_total = step_usage if usage_total is None else usage_total + step_usage

            # A plain dict is fields for the variables (a REST reply); a Record
            # is one result, kept whole under its key so two steps' `results`
            # or `cost` do not overwrite each other.
            if isinstance(output, dict) and not isinstance(output, Record):
                accumulated.update(output)
            else:
                accumulated[output_key] = output

            last_output = output

            # Checkpoint: the step is done; variables reflect its output.
            doc.steps[idx]["status"] = StepStatus.DONE
            doc.steps[idx].pop("suspend", None)
            doc.variables = dict(accumulated)

            self._emit("step.ended", step=idx, name=name, payload={"kind": kind})

        # Completed run: a finished run leaves no parked document (no-op when
        # the run never suspended and was therefore never stored).
        self._store.delete(doc.run_id)
        self._history     = history
        self._accumulated = dict(accumulated)
        self.last_usage   = usage_total
        return self._result(doc, history, accumulated, usage_total, last_output)

    @staticmethod
    def _result(doc, history, accumulated, usage_total, last_output) -> ChainResult:
        failed = [r for r in history if r.get("error")]
        error = (f"step {failed[0]['step']} ({failed[0]['name']}): {failed[0]['error']}"
                 if failed else None)
        return ChainResult(output=last_output, success=not failed, error=error,
                           history=list(history), variables=dict(accumulated),
                           usage=usage_total, run_id=doc.run_id)

    @property
    def accumulated(self) -> dict:
        """
        Snapshot of the full accumulated variable dict after the most recent
        ``run()`` call.  Includes all initial variables, step outputs, and any
        keys written by Tools that return dicts.

        Returns a shallow copy; modifying it does not affect internal state.
        Returns an empty dict before the first ``run()``.

        Typical use — assembling a sectional document after all steps complete::

            chain.run(variables=initial_vars)
            document = assemble_document(chain.accumulated, sections)
        """
        return dict(self._accumulated)

    @property
    def history(self) -> list[dict]:
        """
        Step records from the most recent ``run()`` call.

        Returns a shallow copy; modifying it does not affect the chain's state.
        Returns an empty list before the first ``run()``.

        Each record contains:

        ``step``       — 0-based step index
        ``kind``       — ``"skill"``, ``"tool"``, ``"agent"`` or ``"call"``
        ``name``       — runner name (or ``"step_N"`` if unnamed)
        ``input``      — accumulated variable dict *before* this step ran
        ``output``     — raw output (``str`` or ``dict``)
        ``output_key`` — variable name used to store a ``str`` output
        """
        return list(self._history)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise this chain to a YAML file at *path*.

        The file stores everything needed to recreate the chain:

        * Chain-level metadata — ``name``, ``description``, ``variables``
        * Each step's ``output_key`` and ``input_map``
        * For **Skill** steps — the model name, input template, output spec,
          variables, and options (same fields as ``Skill.save()``)
        * For **Tool** steps — the fully-qualified class path
          (``"module.ClassName"``) and any serialisable ``init_args``

        API keys are **never** written to the file; :meth:`load` resolves
        them from environment variables at load time.

        Parameters
        ----------
        path : str
            Destination file path.  Parent directories are created
            automatically if they do not exist.

        Raises
        ------
        ImportError
            If PyYAML is not installed.

        Example
        -------
        ::

            chain.save("chains/analyse_and_translate.yaml")
        """
        yaml = require("yaml", package="pyyaml", who="Chain.save()", extra="yaml")

        steps_data = []
        for runner, output_key, input_map, kind, options in self._steps:
            if kind == "call":
                raise TypeError(f"step {_callable_name(runner)!r} is a function or a utility's method, and "
                                "cannot be saved — it is code, or a method of a live object, not "
                                "configuration. Build this chain in code.")
            if kind == "skill":
                skill_data: dict = {
                    "model_name": runner.model.name,
                }
                if runner.name:
                    skill_data["name"] = runner.name
                if runner.description:
                    skill_data["description"] = runner.description
                skill_data["input"]  = runner._input
                skill_data["output"] = runner._output
                if runner.variables:
                    skill_data["variables"] = runner.variables
                if runner.options:
                    skill_data["options"] = runner.options

                step: dict = {
                    "kind":       "skill",
                    "output_key": output_key,
                    "skill":      skill_data,
                }

            elif kind == "agent":
                cls        = type(runner)
                class_path = f"{cls.__module__}.{cls.__qualname__}"
                # Serialise the Agent's constructor fields, read directly
                # rather than through `getattr(..., default)`: a fallback
                # turns a missing attribute into a plausible value and a
                # saved chain into a different one, in silence.
                agent_data: dict = {
                    "class":       class_path,
                    "model":       runner.model.name,
                    "mode":        runner.mode,
                    "verbose":     runner.verbose,
                }
                if runner.instructions:
                    agent_data["instructions"] = runner.instructions
                if runner.name:
                    agent_data["name"] = runner.name
                if runner.description:
                    agent_data["description"] = runner.description
                # `stop_when` holds closures — a step ceiling is a function, not
                # a number — so what survives is what can be named. A condition
                # that cannot be reconstructed is dropped **loudly**: a chain
                # that silently comes back without its budget is the quiet
                # failure this whole repair is about.
                kept, lost = [], []
                for cond in (runner.stop_when or []):
                    spec = getattr(cond, "spec", None)
                    (kept if spec else lost).append(spec or cond)
                if kept:
                    agent_data["stop_when"] = kept
                if lost:
                    import warnings as _w
                    _w.warn(
                        f"Agent {runner.name or class_path!r}: "
                        f"{len(lost)} stop condition(s) cannot be serialised "
                        "and will not come back on load; rebuild them in code.",
                        RuntimeWarning, stacklevel=2)
                # Tool list — store class paths only (no API keys)
                if getattr(runner, "tools", None):
                    for t in runner.tools:
                        _refuse_function_tool(t)
                    agent_data["tools"] = [
                        f"{type(t).__module__}.{type(t).__qualname__}"
                        for t in runner.tools
                    ]

                step = {
                    "kind":       "agent",
                    "output_key": output_key,
                    "agent":      agent_data,
                }

            else:  # tool
                _refuse_function_tool(runner)
                cls       = type(runner)
                class_path = f"{cls.__module__}.{cls.__qualname__}"
                init_args  = runner._serialise_init_args() \
                    if hasattr(runner, "_serialise_init_args") else {}

                step = {
                    "kind":       "tool",
                    "output_key": output_key,
                    "tool": {
                        "class":     class_path,
                        "init_args": init_args,
                    },
                }
                # A name or a description given to the factory, over the
                # class's: kept for every tool, whatever its constructor takes.
                for attr in ("name", "description"):
                    value = getattr(runner, attr, None)
                    if value and value != getattr(cls, attr, None):
                        step["tool"][attr] = value

            if input_map:
                step["input_map"] = input_map
            if options:
                step["options"] = options

            steps_data.append(step)

        data: dict = {"steps": steps_data}
        if self.name:
            data["name"] = self.name
        if self.description:
            data["description"] = self.description
        if self.variables:
            data["variables"] = self.variables
        if self.on_step_error != "raise":
            data["on_step_error"] = self.on_step_error
        # Reorder so name/description/variables appear before steps
        ordered: dict = {}
        for k in ("name", "description", "on_step_error", "variables", "steps"):
            if k in data:
                ordered[k] = data[k]

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(ordered, fh, allow_unicode=True, sort_keys=False,
                      default_flow_style=False)

    @classmethod
    def load(
        cls,
        path: str,
        api_key: str | None = None,
        *,
        trusted: bool = False,
    ) -> Chain:
        """
        Load a chain from a YAML file previously created by :meth:`save`.

        .. warning::
            A chain file names tool classes by import path and instantiates
            them — loading one therefore imports and runs code. By default
            only classes under ``yait_aichain`` are allowed; pass
            ``trusted=True`` ONLY for files you created or fully control.

        Each step is reconstructed from the serialised data:

        * **Skill** steps — the model is re-created via
          :class:`~models.Model` (API key resolved from the environment
          unless *api_key* is supplied), then a :class:`~skills.Skill` is
          instantiated from the stored template.
        * **Tool** steps — the class is imported by its stored module path
          and instantiated with the stored ``init_args``; API keys are
          resolved from environment variables automatically.

        Parameters
        ----------
        path : str
            The YAML file ``save()`` wrote.
        api_key : str | None, optional
            A key for every Skill step's model; each reads its own provider's from
            the environment otherwise.
        trusted : bool, optional
            Allow tool classes outside ``yait_aichain`` to be imported. Loading a
            chain file runs the code it names; pass ``True`` only for files you
            wrote or control.

        Returns
        -------
        Chain
            A fully initialised :class:`Chain` ready to call :meth:`run`.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ImportError
            If PyYAML is not installed, or a tool's module cannot be imported.
        ValueError
            If a step's ``kind`` is unrecognised, or a required API key is
            missing.

        Example
        -------
        ::

            from yait_aichain.chain import Chain

            chain  = Chain.load("chains/analyse_and_translate.yaml")
            result = chain.run(variables={"topic": "AI safety",
                                          "language": "French"})
        """
        yaml = require("yaml", package="pyyaml", who="Chain.load()", extra="yaml")

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        # Local imports to avoid circular references at module load time
        from ..models._base import Model as _Model
        from ..skills._skill import Skill as _Skill

        steps = []
        for step_data in data.get("steps", []):
            kind       = step_data["kind"]
            output_key = step_data.get("output_key", "result")
            input_map  = step_data.get("input_map", {})
            options    = step_data.get("options", {})

            if kind == "skill":
                sd    = step_data["skill"]
                model = _Model(sd["model_name"], api_key=api_key)
                skill = _Skill(
                    model       = model,
                    input       = sd["input"],
                    output      = sd["output"],
                    variables   = sd.get("variables"),
                    options     = sd.get("options"),
                    name        = sd.get("name"),
                    description = sd.get("description"),
                )
                entry: tuple | object = (skill, output_key, input_map, options)

            elif kind == "agent":
                from ..agent._agent import Agent as _Agent

                ad = step_data["agent"]

                # Reconstruct tools from class paths
                tool_instances = []
                for tool_path in ad.get("tools", []):
                    _require_safe_tool_class(tool_path, trusted)
                    t_mod, t_cls = tool_path.rsplit(".", 1)
                    try:
                        t_module = importlib.import_module(t_mod)
                    except ModuleNotFoundError as exc:
                        raise ImportError(
                            f"Cannot import tool module '{t_mod}': {exc}"
                        ) from exc
                    tool_instances.append(getattr(t_module, t_cls)())

                from ..agent import cost_budget, step_count, token_budget
                _BUILDERS = {"step_count": step_count,
                             "token_budget": token_budget,
                             "cost_budget": cost_budget}
                stop_when = []
                for spec in ad.get("stop_when", []):
                    build = _BUILDERS.get(spec.get("kind"))
                    if build is None:                # a `check` is a closure
                        continue                     # and cannot come back
                    stop_when.append(build(spec["value"]))

                # An agent step with no `model` cannot be rebuilt; refuse it
                # rather than guess one.
                if "model" not in ad:
                    raise ValueError(
                        "This chain file has an agent step with no model name. "
                        "Rebuild the chain in code and save it again.")
                agent = _Agent(
                    _Model(ad["model"], api_key=api_key),
                    tools        = tool_instances or None,
                    mode         = ad.get("mode", "agile"),
                    stop_when    = stop_when or None,
                    verbose      = ad.get("verbose", 0),
                    instructions = ad.get("instructions", ""),
                    name         = ad.get("name"),
                    description  = ad.get("description"),
                )
                entry = (agent, output_key, input_map, options)

            elif kind == "tool":
                td         = step_data["tool"]
                class_path = td["class"]
                init_args  = td.get("init_args") or {}

                # Import the tool class dynamically (allowlisted unless trusted)
                _require_safe_tool_class(class_path, trusted)
                module_path, class_name = class_path.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                except ModuleNotFoundError as exc:
                    raise ImportError(
                        f"Cannot import tool module '{module_path}': {exc}"
                    ) from exc
                try:
                    tool_cls = getattr(module, class_name)
                except AttributeError:
                    raise ImportError(
                        f"Module '{module_path}' has no class '{class_name}'"
                    )

                # A class can name what it will not take from an untrusted
                # file — an endpoint's URL and the variable its token is read
                # from both come from the file.
                refusal = getattr(tool_cls, "_untrusted_load_refusal", None)
                if not trusted and refusal and (why := refusal(init_args)):
                    raise ValueError(f"Refusing to load {class_path}: {why}")
                tool  = tool_cls(**init_args)
                for attr in ("name", "description"):
                    if td.get(attr):
                        setattr(tool, attr, td[attr])
                entry = (tool, output_key, input_map, options)

            else:
                raise ValueError(
                    f"Unknown step kind {kind!r}. "
                    f"Expected 'skill', 'tool', or 'agent'."
                )

            steps.append(entry)

        return cls(
            steps          = steps,
            variables      = data.get("variables"),
            name           = data.get("name"),
            description    = data.get("description"),
            on_step_error  = data.get("on_step_error", "raise"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(steps) -> list:
        """
        Validate and normalise *steps* into a list of
        ``(runner, output_key, input_map, kind, options)`` 5-tuples.
        """
        result = []
        for item in steps:
            if isinstance(item, tuple):
                if len(item) == 2:
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
                        "(runner, output_key), "
                        "(runner, output_key, input_map), or "
                        "(runner, output_key, input_map, options); "
                        f"got a {len(item)}-element tuple"
                    )
                if not isinstance(key, str):
                    raise ValueError(
                        f"output_key must be a str; got {key!r}"
                    )
            else:
                runner    = item
                key       = "result"
                input_map = {}
                options   = {}

            if _is_tool(runner):
                kind = "tool"
            elif _is_agent(runner):
                kind = "agent"
            elif callable(runner) and not hasattr(runner, "run"):
                # A function or a utility's method — `kb.upsert` — called with
                # the arguments its signature names.
                kind = "call"
            else:
                kind = "skill"

            result.append((runner, key, input_map, kind, options))

        return result

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
