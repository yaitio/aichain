"""
pool
====

Parallel execution of one runner across many items.

Where ``Chain`` runs steps sequentially, ``Pool`` fires the same runner
for every item at the same time using a thread pool.  Results are returned
in the same order as the input items regardless of completion order.

Public API
----------
``Pool(runner, items, *, max_flows, on_error)``
    Run *runner* in parallel for every dict in *items*.

``PENDING = 0``  ``RUNNING = 1``  ``DONE = 2``  ``FAILED = 3``
    Integer status constants for reading ``pool.history`` and ``pool.status``.

Runner support
--------------
  Skill  → ``runner.run(variables=merged)``
  Tool   → ``runner.run(**kwargs)``          kwargs matched from item variables
  Chain  → ``runner.run(variables=merged)``
  Agent  → ``runner.run(task=merged["task"], variables=merged)``

Result and memory
-----------------
``pool.run(variables={})``  → ``list``
    One output per item, same order as *items*.
    Failed items produce ``None`` (unless ``on_error="raise"``).

``pool.history``  → ``list[dict]``
    One record per item:
      ``index``     — position in items list
      ``variables`` — merged variable dict passed to the runner
      ``status``    — int  (PENDING / RUNNING / DONE / FAILED)
      ``output``    — runner output, or ``None``
      ``error``     — error message, or ``None``
      ``duration``  — seconds, or ``None`` if not yet finished

``pool.status``  → ``dict``
    ``{PENDING: N, RUNNING: N, DONE: N, FAILED: N}``
    Thread-safe — safe to poll while the pool is running.

on_error
--------
  ``"raise"``    re-raise the first exception immediately
  ``"collect"``  record the error; result for that item is ``None``
  ``"skip"``     same as collect but emits a ``RuntimeWarning``

Examples
--------
Parse files in parallel::

    import os
    from tools.convert import convertToMD
    from pool import Pool

    files  = [{"source": f} for f in os.listdir("./documents")]
    pool   = Pool(convertToMD(), items=files, max_flows=5)
    texts  = pool.run()
    # texts[i] is the Markdown content of files[i]

Fetch 20 URLs, shared variable merged in::

    from tools.convert import convertToMD
    from pool import Pool

    pool  = Pool(
        convertToMD(),
        items     = [{"source": url} for url in urls],
        max_flows = 10,
        on_error  = "collect",
    )
    pages = pool.run(variables={"format": "markdown"})

Progress monitoring while running::

    import threading, time
    from pool import Pool, DONE, FAILED

    pool   = Pool(my_skill, items=items, max_flows=8)
    thread = threading.Thread(target=pool.run)
    thread.start()

    while thread.is_alive():
        s = pool.status
        print(f"done={s[DONE]}  running={s[RUNNING]}  failed={s[FAILED]}")
        time.sleep(0.5)

    thread.join()
    results = pool.run()   # or read pool.history for outputs

Use a Chain as runner (multi-step pipeline per item)::

    from chain import Chain
    from pool  import Pool

    per_item = Chain(steps=[fetch_tool, summarize_skill])
    pool     = Pool(per_item, items=[{"url": u} for u in urls])
    results  = pool.run()
"""

from ._pool import Pool, PENDING, RUNNING, DONE, FAILED

__all__ = [
    "Pool",
    "PENDING",
    "RUNNING",
    "DONE",
    "FAILED",
]
