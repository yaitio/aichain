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

import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

PENDING = 0
RUNNING = 1
DONE    = 2
FAILED  = 3

_VALID_ON_ERROR = frozenset({"raise", "collect", "skip"})


# ---------------------------------------------------------------------------
# Runner-type helpers  (same lazy checks as chain._chain)
# ---------------------------------------------------------------------------

def _is_tool(runner) -> bool:
    """True for any Tool subclass regardless of where it is defined."""
    return any(cls.__name__ == "Tool" for cls in type(runner).__mro__)


def _is_agent(runner) -> bool:
    """True for any Agent subclass regardless of where it is defined."""
    return any(cls.__name__ == "Agent" for cls in type(runner).__mro__)


def _build_tool_kwargs(tool, variables: dict) -> dict:
    """Match item variables to the tool's declared parameter names."""
    props = tool.parameters.get("properties", {})
    return {k: variables[k] for k in props if k in variables}


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
    description : str | None, optional

    Examples
    --------
    Parse every file in a directory::

        from tools.convert import convertToMD
        from pool import Pool

        files = [{"source": f} for f in os.listdir("./docs")]
        pool  = Pool(convertToMD(), items=files, max_flows=5)
        texts = pool.run()   # list[str], same order as files

    Fetch 20 URLs in parallel::

        from tools.convert import convertToMD
        from pool import Pool

        pool    = Pool(convertToMD(), items=[{"source": u} for u in urls])
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
    An **Agent** runner, however, shares its ``memory`` across items: every
    item resets and writes the same ``AgentMemory``.  Do not rely on agent
    memory contents when running agents through a Pool; give each run its
    own Agent instance if isolated memory matters.
    """

    def __init__(
        self,
        runner,
        items:       list[dict],
        max_flows:   int       = 10,
        on_error:    str       = "collect",
        name:        str | None = None,
        description: str | None = None,
    ) -> None:
        if not items:
            raise ValueError("Pool requires at least one item.")
        if on_error not in _VALID_ON_ERROR:
            raise ValueError(
                f"on_error must be one of {sorted(_VALID_ON_ERROR)}; "
                f"got {on_error!r}"
            )

        self._runner    = runner
        self._items     = list(items)
        self._max_flows = max_flows
        self._on_error  = on_error
        self.name       = name
        self.description = description

        self._lock: threading.Lock = threading.Lock()
        self._history: list[dict]  = self._init_history()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, variables: dict | None = None) -> list:
        """
        Execute the runner for every item in parallel.

        Parameters
        ----------
        variables : dict | None, optional
            Shared variables merged into every item's variable dict.
            Item-level values take precedence on key conflicts.

        Returns
        -------
        list
            Outputs in the same order as *items*.
            Failed items produce ``None`` when ``on_error`` is
            ``"collect"`` or ``"skip"``.

        Raises
        ------
        Exception
            The first task exception when ``on_error="raise"``.
        """
        shared = variables or {}

        # Reset history before each run
        self._history = self._init_history()

        results: list = [None] * len(self._items)

        with ThreadPoolExecutor(max_workers=self._max_flows) as executor:
            future_to_idx: dict = {}

            for i, item in enumerate(self._items):
                merged = {**shared, **item}
                future = executor.submit(self._run_one, i, merged)
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    if self._on_error == "raise":
                        raise
                    if self._on_error == "skip":
                        name = getattr(self._runner, "name", None) or \
                               type(self._runner).__name__
                        warnings.warn(
                            f"Pool item {idx} ({name!r}) failed and was "
                            f"skipped: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    # "collect" or "skip": leave results[idx] as None

        return results

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
            counts = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0}
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
            }
            for i, item in enumerate(self._items)
        ]

    def _run_one(self, index: int, merged: dict) -> Any:
        """
        Execute the runner for one item.  Called inside a worker thread.

        Updates ``_history[index]`` at every status transition.
        Raises the original exception after recording it as FAILED.
        """
        with self._lock:
            self._history[index]["status"]    = RUNNING
            self._history[index]["variables"] = merged

        start = time.monotonic()
        try:
            output = self._dispatch(merged)
            duration = round(time.monotonic() - start, 3)

            with self._lock:
                self._history[index].update({
                    "status":   DONE,
                    "output":   output,
                    "duration": duration,
                })
            return output

        except Exception as exc:
            duration = round(time.monotonic() - start, 3)
            with self._lock:
                self._history[index].update({
                    "status":   FAILED,
                    "error":    str(exc),
                    "duration": duration,
                })
            raise

    def _dispatch(self, variables: dict) -> Any:
        """Route the call to the correct runner interface."""
        runner = self._runner

        if _is_tool(runner):
            kwargs = _build_tool_kwargs(runner, variables)
            return runner.run(**kwargs)

        if _is_agent(runner):
            task = variables.get("task", "")
            if not task:
                raise ValueError(
                    "Agent runner requires a 'task' key in the item variables."
                )
            result = runner.run(task=task, variables=variables)
            if not result:
                raise RuntimeError(
                    f"Agent failed: {getattr(result, 'error', 'unknown error')}"
                )
            return result.output

        # Skill or Chain — both expose run(variables=...)
        return runner.run(variables=variables)

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        runner_name = getattr(self._runner, "name", None) or \
                      type(self._runner).__name__
        return (
            f"Pool(runner={runner_name!r}, items={len(self._items)}, "
            f"max_flows={self._max_flows}, on_error={self._on_error!r})"
        )
