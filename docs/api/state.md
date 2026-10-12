# API: State — pause and resume

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../pipelines/pause-and-resume.md).

<!-- g:api -->
[`RunContext`](#runcontext) · [`current`](#current) · [`using`](#using) · [`Suspend`](#suspend) · [`SuspendedResult`](#suspendedresult) · [`StateStore`](#statestore) · [`InMemoryStore`](#inmemorystore) · [`FileStore`](#filestore) · [`RunDocument`](#rundocument) · [`StepStatus`](#stepstatus)

---

## `RunContext`

```python
RunContext(tenant: 'str | None' = None, metadata: dict = <factory>)
```

Per-request context for one `run()` / `resume()`.

**Parameters**

- `tenant` (*str | None*, default `None`) — Logical tenant / account this run belongs to.
- `metadata` (*dict*, default `<factory>`) — Arbitrary non-secret data (user id, request id, locale, trace ids …),
  available via `chain.context` / `agent.context` for logging and
  routing.

### `RunContext.get()`

```python
RunContext.get(key: str, default=None)
```

Convenience read from `metadata`.

**Parameters**

- `key` (*str*) — A key of `metadata`.
- `default` (*any, optional*, default `None`) — Returned when it is absent.

---

## `current()`

```python
current()
```

The `RunContext` of the run in flight, or None outside one.

---

## `using()`

```python
using(context)
```

Make *context* the current one for the duration of a block.

```python
with using(RunContext(tenant="acme")):
    chain.run()
```

**Parameters**

- `context` (*RunContext*) — The context every call inside the block sees.

---

## `Suspend`

```python
Suspend(
    reason: str = '',
    resume_with: 'dict | None' = None,
    *,
    hint: 'dict | None' = None,
)
```

Raised by a suspend tool to pause the current run.

**Parameters**

- `reason` (*str*, default `''`) — Human-readable reason for the pause (shown to whoever must act).
- `resume_with` (*dict | None*, default `None`) — Schema/description of the signal expected on resume, e.g.
  `{"approved": "bool"}`. Informational — the engine does not enforce
  it; the tool validates the signal on resume.
- `hint` (*dict | None*, default `None`) — Optional hint for an external scheduler/trigger about how/when to
  resume (e.g. `{"wake_at": "2026-06-15T09:00"}` or
  `{"on": "payment.confirmed"}`). The engine only parks the run and
  records the hint; it never schedules anything itself.

---

## `SuspendedResult`

```python
SuspendedResult(run_id: str, awaiting: dict = <factory>, document: dict = <factory>)
```

Returned by `run()` / `resume()` when the run paused instead of
finishing. Falsy, so `if not result:` distinguishes it from a completed
run, mirroring `AgentResult`.

**Attributes**

- `run_id` (*str*) — Identifier to resume with: `chain.resume(run_id, signal=...)`.
- `awaiting` (*dict*, default `<factory>`) — `{"reason": ..., "resume_with": ...}` describing what is awaited.
- `document` (*dict*, default `<factory>`) — The full self-contained run document (also saved in the store). Exposed
  so callers can persist/transport it themselves if they prefer.

---

## `StateStore`

```python
StateStore()
```

Abstract persistence for run documents, keyed by `run_id`.

### `StateStore.load()`

```python
StateStore.load(run_id: str) -> 'dict | None'
```

Return the stored document, or `None` if there is none.

---

## `InMemoryStore`

```python
InMemoryStore()
```

Process-local store (the default). Survives suspend→resume **within one
process** only; for cross-process / serverless use a shared store
(`FileStore` or a custom S3/Dynamo backend).

### `InMemoryStore.load()`

```python
InMemoryStore.load(run_id: str) -> 'dict | None'
```

Return the stored document, or `None` if there is none.

---

## `FileStore`

```python
FileStore(directory: str)
```

Store each run as `<dir>/<run_id>.json` (atomic write). Survives a
process restart, so it is the simplest persistent backend for serverless
when the directory is on shared storage (EFS, a mounted volume, …).

**Parameters**

- `directory` (*str*) — Where the files go; created if it does not exist.

### `FileStore.load()`

```python
FileStore.load(run_id: str) -> 'dict | None'
```

Return the stored document, or `None` if there is none.

---

## `RunDocument`

```python
RunDocument(
    run_id: str,
    kind: str,
    variables: dict = <factory>,
    steps: list[dict] = <factory>,
    usage: dict = <factory>,
    definition: 'dict | None' = None,
    context: 'dict | None' = None,
)
```

One run's durable state (see module docstring).

### `RunDocument.new()`

```python
RunDocument.new(
    kind: str,
    step_names: list[str],
    *,
    variables: 'dict | None' = None,
    definition: 'dict | None' = None,
) -> 'RunDocument'
```

Start a fresh document: all steps `pending`, seeded variables.

### `RunDocument.status`

Aggregate run status, derived from the step statuses.

### `RunDocument.first_pending()`

```python
RunDocument.first_pending() -> 'int | None'
```

Index of the first non-terminal step (the resume cursor).

Terminal steps — `done`, `skipped` and `failed` — were already
attempted and are never re-run, so the cursor advances past them; only
`pending` / `running` / `suspended` steps are resumable.

---

## `StepStatus`

```python
StepStatus()
```
<!-- /g:api -->
