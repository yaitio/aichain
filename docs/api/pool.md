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
    policy=None,
    approve=None,
    interceptors=None,
    hooks: list | None = None,
)
```

Parallel execution of one runner across many items.

**Parameters**

- `runner` (*Skill | Tool | Chain | Agent*) — The runner applied to every item; an Agent takes its task from `"task"`.
- `items` (*list[dict]*) — One variable dict per task, merged over the shared `variables` of `run`.
- `max_flows` (*int, optional*, default `10`) — Maximum number of parallel worker threads (default 10).
- `on_error` (*`"raise"` | `"stop"` | `"skip"` | `"collect"`*, default `'collect'`) — A failing item propagates, stops new items, warns, or is recorded
  silently (the default); a failed item's output is `None`.
- `name` (*str | None, optional*, default `None`) — A label for this pool, kept on it as `pool.name`.
- `description` (*str | None, optional*, default `None`) — Kept as `pool.description` for your own code; the library reads nothing from it.
- `max_cost` (*float | Budget | None, optional*, default `None`) — One ceiling in dollars shared by every item; past it an item fails under `on_error`.
- `policy` (*PermissionPolicy | dict | None, optional*, default `None`) — The rules a tool runner is decided under, item by item; inside an agent,
  the stricter wins.
- `approve` (*callable | None, optional*, default `None`) — Answers an item the policy wants a person for; absent, the item is refused.
- `interceptors` (*list | None, optional*, default `None`) — Act on each tool item before and after it runs, as on an `Agent`.
- `hooks` (*list | None, optional*, default `None`) — Callables given an `Event` — `run.*` for the fan-out, `step.*` per item.

**Examples**

```python
pool = Pool(summarize_skill, items=[{"article": a} for a in articles], max_flows=8)
summaries = pool.run(variables={"language": "English"})
```

Concurrent items share the runner instance: give each item its own Agent when
its events or parked runs must be told apart.

### `Pool.run()`

```python
Pool.run(variables: dict | None = None) -> PoolResult
```

Execute the runner for every item in parallel.

**Parameters**

- `variables` (*dict | None, optional*, default `None`) — Shared variables merged into every item; an item's own values win.

**Returns**

- `PoolResult` — The outputs in item order (`None` for a failed or unstarted item),
  with `.success`, `.error`, `.history`, `.usage` and `.cost`.

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
    beacons: list | None = None,
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
