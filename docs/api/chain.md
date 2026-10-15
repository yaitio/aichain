# API: Chain

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../pipelines/chain.md).

<!-- g:api -->
[`Chain`](#chain) · [`ChainResult`](#chainresult) · [`Step`](#step)

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
    inputs=None,
)
```

Steps in a fixed order, sharing one set of variables.

A step is a `Skill`, a tool, an `Agent`, or a function, in any order; each
gets the variables its prompt, schema or signature names, and its output is
stored under its key, read whole or by path (`"page.body"`).

```python
chain = Chain(steps=[(summarise, "summary"), (translate, "french")], max_cost=0.05)
result = chain.run(variables={"text": "…"})
result["french"], result.output, result.cost
```

**Parameters**

- `steps` (*list*) — Runners, or tuples `(runner, key[, input_map[, options]])` or `(runner,
  input_map)`; an Agent's *options* take `task_key` and `output_field`.
- `variables` (*dict | None, optional*, default `None`) — Default variables for every `run()`; call-time values win.
- `name` (*str | None, optional*, default `None`) — A label, shown in events and when the chain is a step elsewhere.
- `description` (*str | None, optional*, default `None`) — What the chain does.
- `on_step_error` (*str, optional*, default `'raise'`) — `"raise"` (default), `"stop"` without raising, `"skip"` with a
  warning, or `"collect"` silently.
- `store` (*StateStore, optional*, default `None`) — Where a paused run is kept; a persistent one to resume in another process.
- `hooks` (*list | None, optional*, default `None`) — Callables given an `Event` at every step boundary.
- `max_cost` (*float | Budget | None, optional*, default `None`) — A ceiling in dollars shared by every step; past it, `BudgetExceeded`.
- `policy` (*PermissionPolicy | dict | None, optional*, default `None`) — The rules a tool step is decided under; inside an agent, the stricter wins.
- `approve` (*callable | None, optional*, default `None`) — Answers a step the policy wants a person for; absent, the step is refused.
- `interceptors` (*list | None, optional*, default `None`) — Act on a tool step before and after it runs, as on an `Agent`.
- `inputs` (*dict | None, optional*, default `None`) — Descriptions for inputs the steps do not describe, read by `input_schema`.

**Raises**

- `ValueError` — If *steps* is empty, a tuple has an unsupported shape, or *on_step_error*
  is not one of the four.

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

- `variables` (*dict | None, optional*, default `None`) — Variables for this invocation, merged over `self.variables`.
- `on_step_error` (*str | None, optional*, default `None`) — Overrides the chain's `on_step_error` for this call; deprecated, goes in
  5.0 — set it on the chain or on a `Step`.
- `context` (*RunContext, optional*, default `None`) — The tenant and metadata of this request, ambient for every step;
  `Model(api_key=callable)` resolves its key from it.

**Returns**

- `ChainResult` — `.output` the last step's, `result["key"]` any step's, `.success`
  whether all completed; a `SuspendedResult` when a step paused the run.

**Raises**

- `Exception` — A step's own exception, when `on_step_error` is `"raise"`.
- `ValueError` — If the *on_step_error* override is not recognised.

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

The run has a `run_id` and a document in the store before it runs;
`resume(run_id, signal)` starts it, and the signal seeds the variables.

**Parameters**

- `reason` (*str, optional*, default `''`) — What the run waits for, shown to whoever starts it.
- `resume_with` (*dict | None, optional*, default `None`) — The shape of the signal that starts it — a JSON-Schema-like description.
- `hint` (*dict | None, optional*, default `None`) — Recorded for whatever does the waking (`{"on": "order.created"}`).
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

Resume a parked run: the suspended step is re-invoked with *signal*, and steps
already done are never re-run.

**Parameters**

- `run_id` (*str*) — The parked run, as the `SuspendedResult` named it.
- `signal` (*any, optional*, default `None`) — The answer the run waited for. A dict is merged into the variables.
- `on_step_error` (*str | None, optional*, default `None`) — Overrides the chain's policy for the rest of this run; deprecated, goes in
  5.0 — set it on the chain or on a `Step`.
- `context` (*RunContext, optional*, default `None`) — The tenant and metadata for the resumed part; the stored one when omitted.

**Raises**

- `KeyError` — *run_id* is not in the store — already resumed, or never parked.

### `Chain.accumulated`

The variables after the most recent `run()` — initial ones, step outputs and
keys tools wrote; deprecated, goes in 5.0 — read `result.variables`.

A shallow copy; empty before the first `run()`.

### `Chain.history`

Step records from the most recent `run()`; deprecated, goes in 5.0 — read
`result.history`.

Each record holds `step` (0-based), `kind` (`"skill"`, `"tool"`,
`"agent"` or `"call"`), `name`, `input` (the variables before the
step), `output` and `output_key`. A copy; empty before the first run.

### `Chain.input_schema()`

```python
Chain.input_schema() -> dict
```

What a run needs from its caller: every input some step reads that no
earlier step writes, with its type, its description, and whether it is
required or has a default.

Read from what the steps already say — a function's signature and
docstring, a tool's schema, a Skill's template placeholders, an Agent's
task, a chain's own schema — with `variables=` as defaults and
`inputs=` as descriptions. An input nothing describes is said, since
a caller or a model filling it has only its name to go on:

```python
{"url": {"type": "string", "description": "Article URL", "required": True},
 "language": {"type": "string", "default": "English"}}
```

### `Chain.save()`

```python
Chain.save(path: str)
```

Serialise this chain to a YAML file at *path* — metadata, each step's key and
input map, a Skill's model and template, a tool's class path and init args.
API keys are never written; `load` reads them from the environment.

**Parameters**

- `path` (*str*) — Destination file path; parent directories are created.

**Raises**

- `ImportError` — If PyYAML is not installed.

### `Chain.load()`

```python
Chain.load(path: str, api_key: str | None = None, *, trusted: bool = False) -> Chain
```

Load a chain from a YAML file `save` wrote; loading imports and runs the
tool classes it names, so only `yait_aichain` classes unless *trusted*.

**Parameters**

- `path` (*str*) — The YAML file `save()` wrote.
- `api_key` (*str | None, optional*, default `None`) — A key for every Skill step's model; otherwise each reads its own from the environment.
- `trusted` (*bool, optional*, default `False`) — Allow tool classes outside `yait_aichain` — only for files you wrote or control.

**Returns**

- `Chain` — The chain, ready to `run`.

**Raises**

- `FileNotFoundError` — If *path* does not exist.
- `ImportError` — If PyYAML is missing, or a tool's module cannot be imported.
- `ValueError` — If a step's `kind` is unrecognised, or a required API key is missing.

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
    renamed: dict = <factory>,
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
- `renamed` (*dict*, default `<factory>`) — An output's old keys mapped to its key now — `{"result": "fetch"}`;
  reading an old key still works and warns.

### `ChainResult.cost`

Dollars across every step that reported a price, agents included;
None when no step could be priced — never an invented zero.

### `ChainResult.evidence`

What the output rests on: `"check"` when every step that ran was a
function or a tool — a programmatic fact — and `"model_claim"` when a
Skill or an Agent produced part of it, so it is a model's word.

The same two words the agent's journal uses, so a caller deciding
whether to re-check a result reads one vocabulary for both.

---

## `Step`

```python
Step(
    runner: Any,
    out: str | None = None,
    inputs: dict | None = None,
    on_error: str | None = None,
    spread: bool | None = None,
    options: dict = <factory>,
)
```

One step of a `Chain`.

**Parameters**

- `runner` (*Skill | Tool | Agent | Chain | callable*) — What runs.
- `out` (*str | None, optional*, default `None`) — The key the output is stored under; the runner's name by default.
- `inputs` (*dict | None, optional*, default `None`) — Where each input comes from — a variable's name, a path into one
  (`"page.body"`), or a template.
- `on_error` (*str | None, optional*, default `None`) — This step's error policy, over the chain's: `"raise"`, `"stop"`,
  `"skip"` or `"collect"`.
- `spread` (*bool | None, optional*, default `None`) — `True` also puts a dict output's fields among the chain's variables;
  `False` keeps the dict under its key alone; `None` spreads, with a warning.
- `options` (*dict | None, optional*, default `<factory>`) — An Agent step's options: `task_key`, `output_field`.
<!-- /g:api -->
