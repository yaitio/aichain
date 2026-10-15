# API: Chain

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../pipelines/chain.md).

<!-- g:api -->
[`Chain`](#chain) · [`ChainResult`](#chainresult)

---

## `Chain`

```python
Chain(
    steps,
    variables: dict | None = None,
    name: str | None = None,
    description: str | None = None,
    on_step_error: str = 'raise',
    store=None,
    hooks: list | None = None,
    max_cost=None,
    policy=None,
    approve=None,
    interceptors=None,
)
```

Steps in a fixed order, sharing one set of variables.

A step is a `Skill`, a tool, an `Agent`, or a plain function or a
utility's method, in any order. A Skill reads any variable in its prompt;
a tool or a function gets the arguments its schema or signature names,
filled from the variables or from the step's mapping. Each step's output
is stored under its key; a plain dict a step returns is merged into the
variables, so one step can write several.

```python
chain = Chain(steps=[
    (summarise,          "summary"),
    (translate,          "french"),
    (count_words,        "length", {"text": "Title: {french}"}),
    (shout,              "loud"),
], max_cost=0.05)
result = chain.run(variables={"text": "…"})
result["french"], result.output, result.cost
```

**Parameters**

- `steps` (*list*) — The steps, in order. Each is a runner on its own, or a tuple:
  `(runner, output_key)`; `(runner, output_key, input_map)`, where
  *input_map* maps an argument name to a variable's name or to a
  `"{template}"` the chain fills in; or `(runner, output_key,
  input_map, options)`. For an Agent step, *options* takes
  `task_key` (the variable holding its task, default `"task"`) and
  `output_field` (the `AgentResult` attribute stored, default
  `"output"`). A bare runner is stored under `"result"`.
- `variables` (*dict | None, optional*, default `None`) — Default variables for every `run()`; call-time values win.
- `name` (*str | None, optional*, default `None`) — A label, shown in events and when the chain is a step elsewhere.
- `description` (*str | None, optional*, default `None`) — What the chain does.
- `on_step_error` (*str, optional*, default `'raise'`) — What a failed step does to the run: `"raise"` (the default)
  propagates it, `"stop"` ends the run without raising, `"skip"`
  records it, warns and goes on, `"collect"` does the same without
  the warning.
- `store` (*StateStore, optional*, default `None`) — Where a paused run is kept. The default is process-local, so a run
  that waits for a person or a webhook needs `FileStore(...)` or
  another persistent store to be resumed from another process.
- `hooks` (*list | None, optional*, default `None`) — Callables given an `Event` at every step boundary.
- `max_cost` (*float | Budget | None, optional*, default `None`) — A ceiling in dollars for the whole chain, shared by every step; a
  step that would begin past it is refused with `BudgetExceeded`.
- `policy` (*PermissionPolicy | dict | None, optional*, default `None`) — The rules a tool step is decided under — a `PermissionPolicy`, a
  dict of rules, or `{"rules": {...}, "default": ...}`. A refused step
  fails the way any step fails, under `on_step_error`. Inside an
  agent's tool call the agent's rules hold too, and the stricter word
  of the two is the one taken.
- `approve` (*callable | None, optional*, default `None`) — Answers a step the policy wants a person for: given an
  `ApprovalRequest`, returns `True`, `False` or an
  `ApprovalDecision`. Absent, such a step is refused — or asked of the
  approver of an agent this chain runs inside.
- `interceptors` (*list | None, optional*, default `None`) — Act on a tool step before and after it runs, as on an `Agent`; each
  change goes out as an `interceptor.changed` event.

**Raises**

- `ValueError` — If *steps* is empty, a tuple has an unsupported shape, or
  *on_step_error* is not one of the four.

### `Chain.run()`

```python
Chain.run(
    variables: dict | None = None,
    on_step_error: str | None = None,
    *,
    context=None,
) -> ChainResult
```

Execute all steps in order and return a `ChainResult`.

**Parameters**

- `variables` (*dict | None, optional*, default `None`) — Variables for this invocation.  Merged with `self.variables`;
  call-time values win.
- `on_step_error` (*str | None, optional*, default `None`) — Override the instance-level `on_step_error` for this call only.
  Must be `"raise"`, `"stop"`, `"skip"` or `"collect"` when
  provided.
- `context` (*RunContext, optional*, default `None`) — The tenant and metadata of this request, ambient for every step; a
  `Model(api_key=callable)` resolves its key from it.

**Returns**

- `ChainResult` — `.output` is the last successful step's output, `result["key"]`
  any step's output by its key, `.success` whether every step
  completed. A `SuspendedResult` instead when a step
  paused the run.

**Raises**

- `RuntimeError, clients._base.APIError, …` — Any step exception, when `on_step_error` is `"raise"`
  (the default).
- `ValueError` — If *on_step_error* override value is not recognised.

### `Chain.arm()`

```python
Chain.arm(
    reason: str = '',
    resume_with: dict | None = None,
    *,
    hint: dict | None = None,
    variables: dict | None = None,
    context=None,
) -> SuspendedResult
```

Create the run **without starting it**, and return it parked.

The run exists before it runs: it has a `run_id`, a durable document
in the store, the variables it was seeded with, and a declared shape
for the signal that will start it. That is a subscription — the thing
a trigger waits on and a person is shown a form for — expressed as the
same object every other pause produces, so a webhook, a cron tick and
a human clicking *start* all end in `resume(run_id, signal)`.

The engine still schedules nothing. `hint` is recorded for whatever
does: `{"on": "order.created"}`, `{"wake_at": "2026-06-15T09:00"}`.

On resume the signal **seeds the variables** rather than being handed
to a step, because nothing has run yet and there is no suspended tool
to complete.

**Parameters**

- `reason` (*str, optional*, default `''`) — What the run waits for, shown to whoever starts it.
- `resume_with` (*dict | None, optional*, default `None`) — The shape of the signal that starts it — a JSON-Schema-like description.
- `hint` (*dict | None, optional*, default `None`) — Recorded for whatever does the waking: `{"on": "order.created"}`,
  `{"wake_at": "…"}`. The engine schedules nothing.
- `variables` (*dict | None, optional*, default `None`) — Variables the run is seeded with before the signal adds its own.
- `context` (*RunContext, optional*, default `None`) — The request's tenant and metadata, kept with the run.

### `Chain.resume()`

```python
Chain.resume(
    run_id: str,
    signal=None,
    *,
    on_step_error: str | None = None,
    context=None,
) -> ChainResult | None
```

Resume a previously suspended run.

Loads the run document from the store, re-invokes the suspended step
with *signal*, and continues from the first step that is not yet
`done` — already-completed steps are never re-run (their outputs are
in the document's variables).

Idempotent: a `run_id` no longer in the store (already resumed to
completion, or unknown) raises `KeyError` so an at-least-once trigger
can treat that as "already handled".

**Parameters**

- `run_id` (*str*) — The parked run, as the `SuspendedResult` named it.
- `signal` (*any, optional*, default `None`) — The answer the run waited for. A dict is merged into the variables.
- `on_step_error` (*str | None, optional*, default `None`) — Overrides the chain's policy for the rest of this run.
- `context` (*RunContext, optional*, default `None`) — The tenant and metadata for the resumed part; the stored one when
  omitted.

### `Chain.accumulated`

Snapshot of the full accumulated variable dict after the most recent
`run()` call.  Includes all initial variables, step outputs, and any
keys written by Tools that return dicts.

Returns a shallow copy; modifying it does not affect internal state.
Returns an empty dict before the first `run()`.

Typical use — assembling a sectional document after all steps complete:

```python
chain.run(variables=initial_vars)
document = assemble_document(chain.accumulated, sections)
```

### `Chain.history`

Step records from the most recent `run()` call.

Returns a shallow copy; modifying it does not affect the chain's state.
Returns an empty list before the first `run()`.

Each record contains:

`step`       — 0-based step index
`kind`       — `"skill"`, `"tool"`, `"agent"` or `"call"`
`name`       — runner name (or `"step_N"` if unnamed)
`input`      — accumulated variable dict *before* this step ran
`output`     — raw output (`str` or `dict`)
`output_key` — variable name used to store a `str` output

### `Chain.save()`

```python
Chain.save(path: str)
```

Serialise this chain to a YAML file at *path*.

The file stores everything needed to recreate the chain:

* Chain-level metadata — `name`, `description`, `variables`
* Each step's `output_key` and `input_map`
* For **Skill** steps — the model name, input template, output spec,
  variables, and options (same fields as `Skill.save()`)
* For **Tool** steps — the fully-qualified class path
  (`"module.ClassName"`) and any serialisable `init_args`

API keys are **never** written to the file; `load` resolves
them from environment variables at load time.

**Parameters**

- `path` (*str*) — Destination file path.  Parent directories are created
  automatically if they do not exist.

**Raises**

- `ImportError` — If PyYAML is not installed.

**Examples**

```python
chain.save("chains/analyse_and_translate.yaml")
```

### `Chain.load()`

```python
Chain.load(path: str, api_key: str | None = None, *, trusted: bool = False) -> Chain
```

Load a chain from a YAML file previously created by `save`.

.. warning:

```python
A chain file names tool classes by import path and instantiates
them — loading one therefore imports and runs code. By default
only classes under ``yait_aichain`` are allowed; pass
``trusted=True`` ONLY for files you created or fully control.
```

Each step is reconstructed from the serialised data:

* **Skill** steps — the model is re-created via
  `Model` (API key resolved from the environment
  unless *api_key* is supplied), then a `Skill` is
  instantiated from the stored template.
* **Tool** steps — the class is imported by its stored module path
  and instantiated with the stored `init_args`; API keys are
  resolved from environment variables automatically.

**Parameters**

- `path` (*str*) — The YAML file `save()` wrote.
- `api_key` (*str | None, optional*, default `None`) — A key for every Skill step's model; each reads its own provider's from
  the environment otherwise.
- `trusted` (*bool, optional*, default `False`) — Allow tool classes outside `yait_aichain` to be imported. Loading a
  chain file runs the code it names; pass `True` only for files you
  wrote or control.

**Returns**

- `Chain` — A fully initialised `Chain` ready to call `run`.

**Raises**

- `FileNotFoundError` — If *path* does not exist.
- `ImportError` — If PyYAML is not installed, or a tool's module cannot be imported.
- `ValueError` — If a step's `kind` is unrecognised, or a required API key is
  missing.

**Examples**

```python
from yait_aichain.chain import Chain

chain  = Chain.load("chains/analyse_and_translate.yaml")
result = chain.run(variables={"topic": "AI safety",
                              "language": "French"})
```

---

## `ChainResult`

```python
ChainResult(
    output: Any,
    success: bool,
    error: str | None = None,
    history: list = <factory>,
    variables: dict = <factory>,
    usage: Any = None,
    run_id: str = '',
)
```

The outcome of one chain run.

**Attributes**

- `output` (*str | dict | None*) — The last successful step's output.
- `success` (*bool*) — `True` when every step completed. A step failure under
  `on_step_error="skip"`, `"collect"` or `"stop"` leaves it
  `False` — the run carried on, but not everything happened.
- `error` (*str | None*, default `None`) — The first failure, as `"step N (name): message"`; `None` on success.
- `history` (*list[dict]*, default `<factory>`) — One record per step that ran: input, output, and `failure` when it failed.
- `variables` (*dict*, default `<factory>`) — Every variable at the end — the initial ones and each step's output
  under its key. `result["key"]` reads from here.
- `usage` (*Usage | None*, default `None`) — Tokens and cost summed across the steps that reported them.
- `run_id` (*str*, default `''`) — The run document's id.

### `ChainResult.cost`

Dollars across every step that reported a price, agents included;
None when no step could be priced — never an invented zero.

### `ChainResult.evidence`

What the output rests on: `"check"` when every step that ran was a
function or a tool — a programmatic fact — and `"model_claim"` when a
Skill or an Agent produced part of it, so it is a model's word.

The same two words the agent's journal uses, so a caller deciding
whether to re-check a result reads one vocabulary for both.
<!-- /g:api -->
