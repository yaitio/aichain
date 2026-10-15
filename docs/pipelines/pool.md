# Pool

A Pool runs one Skill or Chain over many inputs at once. Fifty documents to
summarise, a thousand rows to classify, twenty pages to translate: the work is
the same, only the input changes, and running it one after another wastes the
time each call spends waiting for the network.

```python
pool = Pool(summarise, items=[{"text": t} for t in documents], max_flows=10)
results = pool.run()
results[0], results.success, results.cost
```

Total time is the slowest item, not the sum. Results come back in the order
of the inputs, whatever order they finished in.

→ Every parameter: [API: Pool](../api/pool.md)

---

## An item is a set of variables

Each item is a dict, given to the runner as its variables — `{"text": …}` fills
`{text}` in the Skill's prompt. Variables passed to `pool.run(variables=…)` go
to every item, for what they share: a language, a tone, a date.

The runner can be a whole Chain, so each item runs its own pipeline — fetch,
then summarise — independently of the others. A Chain or an Agent runner
contributes its `.output` to the results.

A Tool runner is called with the item's variables that match its declared
`parameters`; a Pool refuses to start when the tool's `run()` needs arguments
none of them can pass, rather than return a list of `None`. An Agent runner
takes its task from the item's `"task"`, and an agent that parks returns its
`SuspendedResult` as that item's output.

## Concurrency is a cap, not a count

`max_flows` is how many items are in flight at once, not how many there are:
a thousand items with `max_flows=10` run ten at a time. The right number is
set by the provider's rate limit, not by the machine — past it, calls start
coming back 429 and the retries cost more time than the parallelism saved.

## When an item fails

<!-- g:policy -->
| Mode | Behaviour |
|---|---|
| `"raise"` | The first failing item propagates out of `run()`. |
| `"stop"` | Start no further items; items already running finish. Every failed or unstarted item is `None` in the results. |
| `"skip"` | That item's result is `None`, a `RuntimeWarning` names it, the others continue. |
| `"collect"` (default) | That item's result is `None` and its error is in `history`; the others continue, silently. |
<!-- /g:policy -->

The default differs from a Chain's, and for a reason: in a chain a later step
depends on an earlier one, so one failure invalidates the run; in a pool the
items are independent, and one bad document should not cost you the other
forty-nine. A failed item is `None` in the results and its error is in
`results.history`.

## Watching it run

`pool.status` counts the items pending, running, done and failed, and is safe
to read from another thread while the pool works — enough for a progress bar.
After the run, read the results, not the status.

`pool.history` holds one record per item: its `index`, `variables`, `status`,
`output`, `error` and `duration`. An item a stop, a raise or an attention
beacon kept from starting is `CANCELLED`, with the reason in `error`, so a
loop waiting for every item to finish does end.

## One budget for all of it

`max_cost=` is one ceiling in dollars shared by every item. The items run at
the same time, so a per-item copy would give each the full amount; a shared
`Budget` is decremented by all of them and is safe under the concurrency. An
item that would begin past it fails with the budget's reason, handled by
`on_error`, and the first item always runs.
→ [Cost and budget](../control/cost-and-budget.md)

---

▶ [`examples/11_pool.py`](../../examples/11_pool.py) ·
[`examples/12_pool_chain.py`](../../examples/12_pool_chain.py)
