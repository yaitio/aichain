"""
pool._pool
==========

``Pool`` — parallel execution of one runner across many items.

Where ``Chain`` runs steps sequentially (each feeds the next), ``Pool``
runs the same runner against every item at the same time using a thread
pool.  Order of items is preserved in the result even though execution
is concurrent.

Design
------
  runner   — one Skill, Tool, Chain, or Agent
  items    — list of variable dicts, one per task
  max_flows — maximum number of parallel worker threads

Every item is merged with the shared ``variables`` passed to ``run()``
and then handed to the runner independently.  No state flows between items.

Runner dispatch
---------------
  Skill / Chain  → ``runner.run(variables=merged)``
  Tool           → ``runner.run(**kwargs)``  where kwargs are the item
                   variables that match the tool's declared parameters
  Agent          → ``runner.run(task=merged["task"], variables=merged)``

Status codes (integers)
-----------------------
  PENDING = 0   queued, not yet started
  RUNNING = 1   a worker thread is executing this item right now
  DONE    = 2   completed successfully
  FAILED  = 3   completed with an error

Thread safety
-------------
All writes to ``_history`` are protected by a ``threading.Lock`` so
``pool.history`` and ``pool.status`` can be read safely from outside
while the pool is still running (e.g. for progress monitoring).
"""

from __future__ import annotations

import copy
import threading
import time
import logging
import contextvars
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._result import PoolResult

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

_skip_log = logging.getLogger("yait_aichain.pool")

PENDING = 0
RUNNING = 1
DONE    = 2
FAILED  = 3
#: Never started, and never will this run: a stop, a raise, an attention
#: beacon. Such items stayed PENDING for ever, and a polling loop waiting for
#: DONE + FAILED to reach the count waited for ever too.
CANCELLED = 4

# One vocabulary with Chain — see `_errors_policy`. `stop` is new here: it
# means "start no more items", which is what a fan-out can offer where a
# sequence stops running steps. The difference between the two is the
# difference between a sequence and a fan-out, not between two vocabularies.
from .._budget import as_budget, lend, start_run
from .._errors_policy import POLICIES as _VALID_ON_ERROR, describe as _describe
from .. import _hints
from ..state import Suspend


# ---------------------------------------------------------------------------
# Runner-type helpers  (same lazy checks as chain._chain)
# ---------------------------------------------------------------------------

from .._runners import is_tool as _is_tool, is_agent as _is_agent


def _build_tool_kwargs(tool, variables: dict) -> dict:
    """Match item variables to the tool's declared parameter names."""
    props = tool.parameters.get("properties", {})
    return {k: variables[k] for k in props if k in variables}


def _check_tool_kwargs(tool, variables: dict) -> None:
    """
    Refuse a fan-out that could not pass the runner anything it needs.

    A Tool that declares no matching ``parameters`` gets called with nothing —
    and if its ``run()`` requires arguments, that is a TypeError per item, which
    the default ``on_error="collect"`` turns straight back into ``None``. The
    caller receives a full-length list of ``None`` with no exception and no
    warning, indistinguishable from "every item legitimately returned nothing".
    That shipped: a cookbook recipe's first two stages silently did nothing.

    This is a misdeclared tool, not a per-item runtime failure — it fails
    identically for every item, before any work happens. So it is checked in
    the caller's thread, ahead of the fan-out, where no error policy can
    swallow it. Raising it from inside a worker only changed which exception
    was being turned into ``None``.
    """
    if not variables or _build_tool_kwargs(tool, variables):
        return                                 # nothing to pass, or it fits
    import inspect
    try:
        sig = inspect.signature(tool.run)
    except (TypeError, ValueError):            # builtins, C-implemented run
        return
    needed = [n for n, p in sig.parameters.items()
              if p.default is inspect.Parameter.empty
              and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
              and n != "self"]
    if not needed:
        return
    name = getattr(tool, "name", None) or type(tool).__name__
    raise TypeError(
        f"Pool cannot call {name!r}: its run() requires {needed}, but the tool "
        f"declares no matching `parameters`, so nothing can be passed to it. "
        f"Item keys available: {sorted(variables)}.\n"
        f"Declare them on the tool, e.g.:\n"
        f"    parameters = {{'type': 'object', 'properties': {{"
        + ", ".join(f"'{n}': {{'type': 'string'}}" for n in needed)
        + "}}"
    )


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

class Pool:
    """
    Parallel execution of one runner across many items.

    Parameters
    ----------
    runner : Skill | Tool | Chain | Agent
        The runner applied to every item.
    items : list[dict]
        One variable dict per task.  Each dict is merged with the shared
        ``variables`` passed to :meth:`run` (item values win on conflicts).
    max_flows : int, optional
        Maximum number of parallel worker threads (default 10).
    on_error : ``"raise"`` | ``"collect"`` | ``"skip"``
        How to handle a task that raises an exception:

        ``"raise"``    — re-raise immediately; other tasks may still finish.
        ``"collect"``  — record the error; result for that item is ``None``.
        ``"skip"``     — same as ``"collect"`` but emits a ``RuntimeWarning``.
    name : str | None, optional
        A label for this pool, kept on it as ``pool.name``.
    description : str | None, optional
        Kept on the pool as ``pool.description`` for your own code. The library
        reads nothing from it — a pool is not offered to a model, and it is
        not saved — so it changes no behaviour.
    max_cost : float | Budget | None, optional
        One ceiling in dollars for every item; an item that would begin past
        it fails with the budget's reason, handled by ``on_error``.
    policy : PermissionPolicy | dict | None, optional
        The rules a tool runner is decided under, item by item — a
        ``PermissionPolicy``, a dict of rules, or ``{"rules": {...},
        "default": ...}``. A refused item fails under ``on_error``. Inside an
        agent's tool call the agent's rules hold too, the stricter word winning.
    approve : callable | None, optional
        Answers an item the policy wants a person for, as ``Agent(approve=)``
        does. Absent, such an item is refused — or asked of the approver of an
        agent this pool runs inside.
    interceptors : list | None, optional
        Act on each tool item before and after it runs, as on an ``Agent``.

    Examples
    --------
    Parse every file in a directory::

        from yait_aichain.tools import fetch
        from yait_aichain.pool import Pool

        files = [{"source": f} for f in os.listdir("./docs")]
        pool  = Pool(fetch("local"), items=files, max_flows=5)
        texts = pool.run()   # list[str], same order as files

    Fetch 20 URLs in parallel::

        from yait_aichain.tools import fetch
        from yait_aichain.pool import Pool

        pool    = Pool(fetch("local"), items=[{"source": u} for u in urls])
        pages   = pool.run()

    Shared variables merged into every item::

        pool = Pool(
            summarize_skill,
            items     = [{"article": a} for a in articles],
            max_flows = 8,
        )
        summaries = pool.run(variables={"language": "English", "length": "short"})

    Use a Chain as runner (per-item multi-step pipeline)::

        per_item = Chain(steps=[fetch_tool, analyze_skill])
        pool     = Pool(per_item, items=[{"url": u} for u in urls])
        results  = pool.run()

    Concurrency note
    ----------------
    All worker threads call ``run()`` on the **same** runner instance.
    Skill, Tool, and Chain runners are safe: their ``run()`` keeps state
    local (Chain's ``history``/``accumulated`` attributes reflect the most
    recently finished item — use the Pool results as the source of truth).
    An **Agent** runner keeps the run in flight on the instance — the run id
    its events are stamped with, ``last_result``, a parked run's document — so
    concurrent items overwrite each other's. Each item's result is its own;
    for events and parking that can be told apart, give each item its own
    Agent.
    """

    @_hints.refuse("Pool")
    def __init__(
        self,
        runner,
        items:       list[dict],
        max_flows:   int       = 10,
        on_error:    str       = "collect",
        name:        str | None = None,
        description: str | None = None,
        max_cost=None,
        policy=None,
        approve=None,
        interceptors=None,
    ) -> None:
        if not callable(getattr(runner, "run", None)):
            # A plain function had no `run`, and every item failed into a list
            # of None under the default on_error.
            raise TypeError(f"Pool runs a Skill, a Tool, a Chain or an Agent — something with "
                            f"run(); got {type(runner).__name__}. Wrap a function with @tool")
        if isinstance(items, (str, bytes)) or not hasattr(items, "__iter__"):
            raise TypeError(f"Pool items is a list of dicts, not a {type(items).__name__}")
        items = list(items)
        if bad := [i for i, it in enumerate(items) if not isinstance(it, Mapping)]:
            raise TypeError(f"Pool items are dicts of variables; item {bad[0]} is a "
                            f"{type(items[bad[0]]).__name__} — e.g. [{{'text': t}} for t in texts]")
        if isinstance(max_flows, bool) or not isinstance(max_flows, int) or max_flows < 1:
            raise ValueError(f"max_flows is a whole number of at least 1, not {max_flows!r}")
        if not items:
            raise ValueError("Pool requires at least one item.")
        if on_error not in _VALID_ON_ERROR:
            raise ValueError(
                f"on_error must be one of {sorted(_VALID_ON_ERROR)}; "
                f"got {on_error!r}"
            )

        self._runner    = runner
        self._beacons: list = []
        # One ceiling for the whole fan-out, and this is where sharing earns
        # its keep: items run concurrently, so a per-item copy of a number
        # would let every worker spend the full amount. `Budget` takes a lock
        # for exactly this.
        self.max_cost = as_budget(max_cost)
        self._own_budget = self.max_cost        # see _budget.start_run
        self._items     = list(items)
        self._max_flows = max_flows
        self._on_error  = on_error
        self.name       = name
        self.description = description
        from .._dispatch import split_policy
        import warnings as _warnings
        for old, value in (("approve", approve), ("interceptors", interceptors)):
            if value is not None:
                _warnings.warn(f"Pool({old}=...) goes in 5.0; say it in policy={{{old!r}: ...}}",
                              FutureWarning, stacklevel=3)
        self.policy, self.approve, self.interceptors = split_policy(
            policy, approve=approve, interceptors=interceptors, where="Pool(policy=)")
        if self.approve == "park":
            raise ValueError("Pool: approve='park' parks an agent's run; a pool cannot "
                             "be parked on a person's answer yet — give a function, or 'deny'")

        self._lock: threading.Lock = threading.Lock()
        self._history: list[dict]  = self._init_history()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, variables: dict | None = None) -> PoolResult:
        """
        Execute the runner for every item in parallel.

        Parameters
        ----------
        variables : dict | None, optional
            Shared variables merged into every item's variable dict.
            Item-level values take precedence on key conflicts.

        Returns
        -------
        PoolResult
            The outputs in item order — iterable and indexable — with
            ``.success``, ``.error``, ``.history``, ``.usage`` and ``.cost``.
            A failed or unstarted item's output is ``None``.

        Raises
        ------
        Exception
            The first task exception when ``on_error="raise"``.
        """
        start_run(self)
        shared = variables or {}

        # Pre-flight, in this thread: a runner that cannot be handed what the
        # items carry fails for every item, so say so once and now rather than
        # letting on_error turn it into a list of None.
        if _is_tool(self._runner):
            for item in self._items:
                _check_tool_kwargs(self._runner, {**shared, **item})

        # Reset history before each run
        self._history = self._init_history()

        results: list = [None] * len(self._items)

        # A board for the whole fan-out. Each item's context is copied at
        # submit time, so every worker posts to this one — and the wait below
        # is where a blocker is noticed, which is the wait it exists to end.
        from ..agent._swarm import collecting_beacons
        # The budget too is ambient for the fan-out: copied into each worker
        # with the context, and charged by every Skill inside an item.
        self._stop = threading.Event()
        self._stop_reason = ""
        with collecting_beacons() as board, lend(self.max_cost):
            self._board = board
            executor = ThreadPoolExecutor(max_workers=self._max_flows)
            future_to_idx: dict = {}
            finished = False
            try:
                for i, item in enumerate(self._items):
                    merged = {**shared, **item}
                    # `contextvars` do not cross a thread boundary on their
                    # own, so the current run's context is copied into each
                    # worker explicitly. Without this a fan-out under a tenant
                    # would resolve every item's API key as if no run were in
                    # flight — and the failure would look like a missing key
                    # rather than a lost context.
                    future = executor.submit(
                        contextvars.copy_context().run, self._run_one, i, merged)
                    future_to_idx[future] = i

                def stop_starting(why: str):
                    # Items in flight finish: killing a call mid-request costs
                    # the tokens anyway and loses the answer, so stopping can
                    # only honestly mean not beginning more. The event stops
                    # a worker that is free between the decision and here.
                    self._stop_reason = self._stop_reason or why
                    self._stop.set()
                    for pending in future_to_idx:
                        pending.cancel()

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    if future.cancelled():
                        continue
                    if board.attention():
                        # A blocker, a question or a changed contract from any
                        # item, or from anything it delegated to: the rest of
                        # the fan-out would be spent on work it made pointless.
                        stop_starting("an attention beacon was raised")
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        if self._on_error == "raise":
                            stop_starting(f"item {idx} raised")
                            raise
                        if self._on_error == "stop":
                            stop_starting(f"item {idx} failed and on_error='stop'")
                        if self._on_error == "skip":
                            name = getattr(self._runner, "name", None) or type(self._runner).__name__
                            # Logged, every time: a warning shows once per
                            # place in the code, and the place was ours.
                            _skip_log.warning(f"Pool item {idx} ({name!r}) failed and was skipped: "
                                              f"{type(exc).__name__}: {exc}")
                        # "collect", "skip" and "stop" all leave results[idx] None
                finished = True
            finally:
                if not finished:
                    # A raise — on_error="raise", Ctrl-C — waited for the whole
                    # queue: a thousand paid calls after the first one failed.
                    self._stop_reason = self._stop_reason or "the run was interrupted"
                    self._stop.set()
                executor.shutdown(wait=finished, cancel_futures=not finished)
                self._cancel_unstarted()
        self._beacons = board.all()

        from ._result import PoolResult
        history = self.history
        self._board = None
        return PoolResult(results, success=all(r["status"] == DONE for r in history),
                          history=history, usage=self.usage, beacons=self._beacons)

    @property
    def beacons(self) -> list:
        """Beacons raised by any item during the last run, attention or not."""
        return list(self._beacons)

    @property
    def history(self) -> list[dict]:
        """
        One record per item from the most recent :meth:`run` call.

        Each record:
          ``index``     — position in the original items list
          ``variables`` — the merged variable dict passed to the runner
          ``status``    — int: PENDING(0) RUNNING(1) DONE(2) FAILED(3)
          ``output``    — runner output, or ``None`` if not yet done
          ``error``     — error message string, or ``None``
          ``duration``  — seconds the task took, or ``None`` if not done

        Thread-safe: safe to read while the pool is still running.
        """
        with self._lock:
            return [dict(r) for r in self._history]

    @property
    def usage(self):
        """Summed usage across items, or ``None`` when nothing reported any.

        Items that cannot report — a plain ``Tool`` — contribute nothing
        rather than a zero, so the sum stays the bill for the calls that
        actually happened.
        """
        with self._lock:
            parts = [r["usage"] for r in self._history if r.get("usage")]
        return sum(parts) if parts else None

    @property
    def status(self) -> dict:
        """
        Count of tasks per status code.

        Returns
        -------
        dict
            ``{PENDING: N, RUNNING: N, DONE: N, FAILED: N}``

        Thread-safe: safe to poll while the pool is still running::

            while pool.status[DONE] + pool.status[FAILED] < len(items):
                print(pool.status)
                time.sleep(0.5)
        """
        with self._lock:
            counts = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0, CANCELLED: 0}
            for record in self._history:
                counts[record["status"]] += 1
            return counts

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_history(self) -> list[dict]:
        """Build fresh history with every item in PENDING state."""
        return [
            {
                "index":     i,
                "variables": item,
                "status":    PENDING,
                "output":    None,
                "error":     None,
                "duration":  None,
                "usage":     None,
            }
            for i, item in enumerate(self._items)
        ]

    _stop_reason: str = ""
    _stop: threading.Event | None = None
    _board = None

    def _cancel_unstarted(self) -> None:
        with self._lock:
            for record in self._history:
                if record["status"] == PENDING:
                    record.update(status=CANCELLED,
                                  error=f"not started: {self._stop_reason or 'the pool stopped'}")

    def _run_one(self, index: int, merged: dict) -> Any:
        """
        Execute the runner for one item.  Called inside a worker thread.

        Updates ``_history[index]`` at every status transition.
        Raises the original exception after recording it as FAILED.
        """
        # Checked before starting, by the worker itself: a free worker took
        # the next item in the moment between a blocker and the cancel.
        if (self._stop is not None and self._stop.is_set()) or (
                self._board is not None and self._board.attention()):
            with self._lock:
                self._history[index].update(
                    status=CANCELLED,
                    error=f"not started: {self._stop_reason or 'an attention beacon was raised'}")
            return None
        with self._lock:
            self._history[index]["status"]    = RUNNING
            self._history[index]["variables"] = merged

        start = time.monotonic()
        try:
            try:
                output, usage = self._dispatch(merged)
            except Suspend as susp:
                # A fan-out has no single run to park: the other items are in
                # flight and there is no identifier the world could resume.
                # Said here, with the shapes that do work, rather than arriving
                # as a bare "Suspend: approve?" beside nine results.
                reason = susp.reason or "no reason given"
                raise RuntimeError(
                    f"a Pool item asked to pause ({reason}) and a fan-out "
                    "cannot park. Put the Gate around the Pool — one Chain "
                    "step that pauses, then the fan-out — or give the item's "
                    "Agent a store, so it parks its own run and returns a "
                    "SuspendedResult as this item's output."
                ) from susp
            duration = round(time.monotonic() - start, 3)

            with self._lock:
                self._history[index].update({
                    "status":   DONE,
                    "output":   output,
                    "duration": duration,
                    "usage":    usage,
                })
            return output

        except Exception as exc:
            duration = round(time.monotonic() - start, 3)
            with self._lock:
                self._history[index].update({
                    "status":   FAILED,
                    # Type and traceback beside the message: a
                    # `KeyError('tenant')` renders as `'tenant'`, which in a
                    # run record reads as a value rather than a fault.
                    # Never empty: `TimeoutError()` and a bare `assert` have
                    # no message, and an empty error read as no error.
                    "error":    str(exc) or type(exc).__name__,
                    "failure":  _describe(exc),
                    "duration": duration,
                })
            raise

    def _run_tool(self, runner, kwargs: dict):
        """One item of a tool runner, decided and run where every tool call is
        (`_dispatch.py`). Before, an item was only charged — a tool an agent's
        policy refused ran when a pool inside its tool fanned it out."""
        from .._dispatch import Denied, Dispatcher
        from ..models._calls import ToolCall
        from ..state._run import Run
        run = Run(owner=self)
        dispatcher = Dispatcher(
            tools={runner.name: runner}, builtins={}, gated=lambda call: runner,
            permissions=self.policy, approve=self.approve, max_cost=self.max_cost,
            emit=lambda *a, **k: None, who=self.name or "", run=lambda: run,
            log_call=lambda tool, kwargs: None, interceptors=self.interceptors)
        call = ToolCall(id="", name=runner.name, arguments=kwargs)
        step = dispatcher.decide(call)
        if isinstance(step, Denied):
            raise PermissionError(step.reason)
        call = step.call or call
        return dispatcher._after(call, dispatcher.run_tool(call, check=False))

    def _dispatch(self, variables: dict) -> tuple[Any, Any]:
        """Route the call to the correct runner interface.

        Returns ``(output, usage)``. Usage is per item because that is the
        only place it can be attributed: one shared runner serves every item,
        and ``last_usage`` on it is overwritten by whichever thread finishes
        next. Reading it after ``run()`` races; the runner is copied instead,
        so each item reads its own counter.
        """
        from ..models._usage import Usage
        runner = self._runner

        if _is_tool(runner):
            kwargs = _build_tool_kwargs(runner, variables)
            return self._run_tool(runner, kwargs), None

        if _is_agent(runner):
            task = variables.get("task", "")
            if not task:
                raise ValueError(
                    "Agent runner requires a 'task' key in the item variables."
                )
            result = runner.run(task=task, variables=variables)
            # A parked agent is falsy but did not fail: what the docstring
            # above promises is the SuspendedResult as the item's output, with
            # its run_id, not "Agent failed: unknown error" and an orphan
            # document in the store.
            from ..state import SuspendedResult
            if isinstance(result, SuspendedResult):
                return result, None
            if not result:
                raise RuntimeError(
                    f"Agent failed: {getattr(result, 'error', 'unknown error')}"
                )
            # An Agent reports on its result, not through ``last_usage``; only
            # the total is available, so the split is left at zero rather than
            # invented. Same treatment as Chain gives an Agent step.
            tokens = getattr(result, "tokens_used", 0) or 0
            cost   = getattr(result, "cost", None)
            usage  = Usage(total_tokens=tokens, cost=cost) if (tokens or cost) else None
            return result.output, usage

        # Skill or Chain — both expose run(variables=...) and report on
        # ``last_usage``. The copy is shallow: configuration is shared, the
        # usage counter is not.
        own = copy.copy(runner)
        output = own.run(variables=variables)
        # A chain as the runner contributes its output; a chain that did not
        # complete is a failed item, exactly as a failed agent is.
        from ..chain._result import ChainResult
        if isinstance(output, ChainResult):
            if not output:
                raise RuntimeError(f"Chain failed: {output.error}")
            return output.output, output.usage
        return output, getattr(own, "last_usage", None)

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        runner_name = getattr(self._runner, "name", None) or \
                      type(self._runner).__name__
        return (
            f"Pool(runner={runner_name!r}, items={len(self._items)}, "
            f"max_flows={self._max_flows}, on_error={self._on_error!r})"
        )
