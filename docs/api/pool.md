# API: Pool

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../pipelines/pool.md).

<!-- g:api -->
[`Pool`](#pool) · [`PoolResult`](#poolresult) · [`PENDING`](#pending) · [`RUNNING`](#running) · [`DONE`](#done) · [`FAILED`](#failed)

---

## `Pool`

```python
Pool(
    runner,
    items: list[dict],
    max_flows: int = 10,
    on_error: str = 'collect',
    name: str | None = None,
    description: str | None = None,
    max_cost=None,
)
```

Parallel execution of one runner across many items.

**Parameters**

- `runner` (*Skill | Tool | Chain | Agent*) — The runner applied to every item.
- `items` (*list[dict]*) — One variable dict per task.  Each dict is merged with the shared
  `variables` passed to `run` (item values win on conflicts).
- `max_flows` (*int, optional*, default `10`) — Maximum number of parallel worker threads (default 10).
- `on_error` (*`"raise"` | `"collect"` | `"skip"`*, default `'collect'`) — How to handle a task that raises an exception:

  `"raise"`    — re-raise immediately; other tasks may still finish.
  `"collect"`  — record the error; result for that item is `None`.
  `"skip"`     — same as `"collect"` but emits a `RuntimeWarning`.
- `name` (*str | None, optional*, default `None`)
- `description` (*str | None, optional*, default `None`)
- `max_cost` (*float | Budget | None, optional*, default `None`) — One ceiling in dollars for every item; an item that would begin past
  it fails with the budget's reason, handled by `on_error`.

**Examples**

Parse every file in a directory:

```python
from yait_aichain.tools import fetch
from yait_aichain.pool import Pool

files = [{"source": f} for f in os.listdir("./docs")]
pool  = Pool(fetch("local"), items=files, max_flows=5)
texts = pool.run()   # list[str], same order as files
```

Fetch 20 URLs in parallel:

```python
from yait_aichain.tools import fetch
from yait_aichain.pool import Pool

pool    = Pool(fetch("local"), items=[{"source": u} for u in urls])
pages   = pool.run()
```

Shared variables merged into every item:

```python
pool = Pool(
    summarize_skill,
    items     = [{"article": a} for a in articles],
    max_flows = 8,
)
summaries = pool.run(variables={"language": "English", "length": "short"})
```

Use a Chain as runner (per-item multi-step pipeline):

```python
per_item = Chain(steps=[fetch_tool, analyze_skill])
pool     = Pool(per_item, items=[{"url": u} for u in urls])
results  = pool.run()
```

Concurrency note
----------------
All worker threads call `run()` on the **same** runner instance.
Skill, Tool, and Chain runners are safe: their `run()` keeps state
local (Chain's `history`/`accumulated` attributes reflect the most
recently finished item — use the Pool results as the source of truth).
An **Agent** runner, however, shares its `memory` across items: every
item resets and writes the same `AgentMemory`.  Do not rely on agent
memory contents when running agents through a Pool; give each run its
own Agent instance if isolated memory matters.

### `Pool.run()`

```python
Pool.run(variables: dict | None = None) -> 'PoolResult'
```

Execute the runner for every item in parallel.

**Parameters**

- `variables` (*dict | None, optional*, default `None`) — Shared variables merged into every item's variable dict.
  Item-level values take precedence on key conflicts.

**Returns**

- `PoolResult` — The outputs in item order — iterable and indexable — with
  `.success`, `.error`, `.history`, `.usage` and `.cost`.
  A failed or unstarted item's output is `None`.

**Raises**

- `Exception` — The first task exception when `on_error="raise"`.

### `Pool.beacons`

Beacons raised by any item during the last run, attention or not.

### `Pool.history`

One record per item from the most recent `run` call.

Each record:
  `index`     — position in the original items list
  `variables` — the merged variable dict passed to the runner
  `status`    — int: PENDING(0) RUNNING(1) DONE(2) FAILED(3)
  `output`    — runner output, or `None` if not yet done
  `error`     — error message string, or `None`
  `duration`  — seconds the task took, or `None` if not done

Thread-safe: safe to read while the pool is still running.

### `Pool.usage`

Summed usage across items, or `None` when nothing reported any.

Items that cannot report — a plain `Tool` — contribute nothing
rather than a zero, so the sum stays the bill for the calls that
actually happened.

### `Pool.status`

Count of tasks per status code.

**Returns**

- `dict` — `{PENDING: N, RUNNING: N, DONE: N, FAILED: N}`
- `Thread-safe: safe to poll while the pool is still running::` — while pool.status[DONE] + pool.status[FAILED] < len(items):
      print(pool.status)
      time.sleep(0.5)

---

## `PoolResult`

```python
PoolResult(
    outputs: list,
    *,
    success: bool,
    history: list,
    usage: Any = None,
    beacons: 'list | None' = None,
)
```

The outcome of one pool run.

**Attributes**

- `output` (*list*) — One output per item, in item order; `None` for an item that failed
  or was never started.
- `success` (*bool*) — `True` when every item completed.
- `error` (*str | None*) — The first failure, as `"item N: message"`; `None` on success.
- `errors` (*list[str | None]*) — One entry per item.
- `history` (*list[dict]*) — One record per item: status, output, error, failure, duration, usage.
- `usage` (*Usage | None*, default `None`) — Summed across the items that reported usage.
- `beacons` (*list*, default `None`) — Beacons raised by any item.

**Parameters**

- `outputs` (*list*) — One output per item, in the items' order.
- `success` (*bool*) — Whether every item succeeded.
- `history` (*list*) — One record per item: its status, output, error and duration.
- `usage` (*Usage, optional*, default `None`) — Tokens and cost across the items.
- `beacons` (*list | None, optional*, default `None`) — Beacons the items raised.

---

## `PENDING`

`0`

---

## `RUNNING`

`1`

---

## `DONE`

`2`

---

## `FAILED`

`3`
<!-- /g:api -->
