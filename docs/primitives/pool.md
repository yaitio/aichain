# Pool

A **Pool** runs the same runner — a [Skill](skills.md) or a [Chain](chain.md) —
across many inputs **in parallel**. Total wall-clock time is the slowest item,
not the sum. Results come back in the same order as the inputs.

---

## Quick start

```python
import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.pool   import Pool

summarise = Skill(
    model = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": ["Summarise {topic} in one line."]}]},
)

pool = Pool(summarise, items=[{"topic": t} for t in ["gravity", "entropy", "AI"]], max_flows=10)
results = pool.run()          # list, one entry per item, same order

print(results)
print(pool.status)            # {0: 0, 1: 0, 2: 3, 3: 0}  → 3 DONE
```

Each item is a **variable dict** fed to the runner. The pool runs up to
`max_flows` of them concurrently.

▶ Skill across items: [`examples/11_pool.py`](../../examples/11_pool.py) ·
Chain as the runner: [`examples/12_pool_chain.py`](../../examples/12_pool_chain.py) ·
Deep dive ↓

---

## Common gotchas

- **`items` are variable dicts**, not positional args — each is merged into the
  runner's variables for that flow.
- **Results preserve input order** even though execution is concurrent.
  `results[i]` corresponds to `items[i]`.
- **A failed item is `None`** in the results (with `on_error="collect"`/`"skip"`);
  use the results list — not live `status` — as the source of truth after the run.
- **`max_flows` is the concurrency cap**, not the item count. 1000 items with
  `max_flows=10` run ten at a time.

---

## Reference

### Constructor

```python
Pool(runner, items, max_flows=10, on_error="collect", name=None, description=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `runner` | `Skill` \| `Chain` | — | What to run for each item. |
| `items` | `list[dict]` | — | One variable dict per parallel flow. |
| `max_flows` | `int` | `10` | Max concurrent flows (worker threads). |
| `on_error` | `str` | `"collect"` | How a failing item is handled (below). |
| `name`, `description` | `str` \| `None` | `None` | Labels. |

### `run()`

```python
results = pool.run(variables=None) -> list
```

Returns a list the same length and order as `items`; entry `i` is the runner's
output for `items[i]` (or `None` if that item failed under
`collect`/`skip`). The optional `variables` dict is **merged into every item** —
handy for values shared across all flows.

```python
results = pool.run(variables={"language": "English"})   # every item gets language=English
```

### `on_error` modes

| Mode | Behaviour |
|---|---|
| `"raise"` | The first item error aborts the whole pool and propagates. |
| `"collect"` (default) | Record the error; that item's result is `None`; other items continue. |
| `"skip"` | Like `collect`, but also emit a warning. |

### Monitoring: `status` and `history`

Both are safe to read from another thread while the pool runs.

```python
pool.status   # {PENDING: N, RUNNING: N, DONE: N, FAILED: N} — live counts
pool.history  # one record per item: its status, result, and error (if any)
```

Status codes: `PENDING=0`, `RUNNING=1`, `DONE=2`, `FAILED=3` (importable from
`yait_aichain.pool`). Poll `status` for a live progress bar; read `history` or
the `run()` results for the outcomes.

### Any runner works

The runner can be a whole **Chain**, so each item runs its own multi-step
pipeline independently and in parallel:

```python
from yait_aichain.chain import Chain

pipeline = Chain(steps=[(fetch, "page", {"source": "url"}), (summarise, "summary")])
pool = Pool(pipeline, items=[{"url": u} for u in urls], max_flows=5)
results = pool.run()
```

---

## See also

- [Skill](skills.md) / [Chain](chain.md) — the runners a Pool parallelises.
- [Agent](../agents/overview.md) — for tasks that need planning rather than fan-out.
