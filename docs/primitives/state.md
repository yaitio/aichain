# State — suspend & resume

A run can **pause** until an external signal arrives — a human approval, a
webhook, a cron tick, a queue message — and **resume** later, even in a
*different process*. The paused run is parked in a **store**; `resume(run_id,
signal)` continues from exactly where it stopped, without re-running completed
steps. This is the serverless core of the library.

It applies to both [Chain](chain.md) and [Agent](../agents/overview.md), and is
entirely additive: a run with no pause tool behaves exactly as before.

---

## Quick start

Put a `Wait` tool where the run should pause; resume it with the awaited data:

```python
from yait_aichain.chain import Chain
from yait_aichain.tools import Wait
from yait_aichain.state import SuspendedResult

chain = Chain(steps=[
    (draft, "draft"),
    Wait(reason="A human must approve the reply", resume_with={"reply": "str"}),
    (send, "confirmation"),
])

result = chain.run(variables={"complaint": "My order arrived late."})

if isinstance(result, SuspendedResult):
    print(result.awaiting["reason"])          # why it paused
    print(result.document["variables"]["draft"])  # work done so far
    # ...a human reviews, then:
    final = chain.resume(result.run_id, signal={"reply": "Approved text"})
```

▶ Human-in-the-loop: [`examples/17_chain_human_input.py`](../../examples/17_chain_human_input.py) ·
External trigger (cross-process): [`examples/18_agent_external_trigger.py`](../../examples/18_agent_external_trigger.py) ·
Deep dive ↓

---

## Common gotchas

- **`run()` returns a `SuspendedResult` when it pauses** — and it's *falsy*. So
  `if result:` is false on a pause; check `isinstance(result, SuspendedResult)`
  (or inspect `result.run_id`).
- **Default store is in-memory** — fine within one process, but a restart loses
  the run. For a webhook/cron in another process, give a persistent store
  (`FileStore`, or your own).
- **Resume is idempotent by construction.** The store holds only *suspended*
  runs; a completed run is deleted. Resuming a `run_id` that already finished
  raises `KeyError` — a duplicate trigger can't double-execute.
- **Completed steps are never re-run.** Their outputs are already in the saved
  run document; resume re-enters only the paused step with your `signal`.

---

## Reference

### How it works

A run carries a self-contained **document**: the variables produced so far, each
step's status, and enough definition to continue. When a step raises the
internal `Suspend`, the engine saves that document to the store and returns a
`SuspendedResult`. `resume(run_id, signal)` loads the document, feeds `signal`
into the suspended step, and runs forward to completion (or the next pause),
deleting the document from the store when the run finishes.

Because step outputs are *named* and stored, there is no replay: resume picks up
at the cursor, not from the top.

### `Wait` — a pause point

A leaf tool that pauses the run on first reach and, on resume, returns the
signal as its output.

```python
Wait(reason="...", resume_with={"reply": "str"})
```

| Argument | Purpose |
|---|---|
| `reason` | Human-readable why-paused, surfaced in `SuspendedResult.awaiting`. |
| `resume_with` | A `{field: type}` hint describing the signal you'll send back. Documentation only — not enforced. |
| `name` | Optional step/tool name. |
| `hint` | Optional free-form hint for whatever will resume the run (e.g. a cron expression). The library records it; it never schedules anything itself. |

On resume, `Wait`'s output is the `signal` dict — so a later step can read what
the human/webhook provided.

### `Gate` — approval around any tool

Wraps **any** tool behind an external decision. On first reach the run pauses;
on resume, if the signal grants approval the wrapped tool runs with its normal
arguments, otherwise it's skipped.

```python
from yait_aichain.tools import Gate

Gate(IssueRefund(), reason="A manager must approve the refund",
     resume_with={"approved": "bool"})
# resume with: signal={"approved": True}
```

| Argument | Default | Purpose |
|---|---|---|
| `tool` | — | The tool to run only after approval. |
| `reason` | `""` | Why approval is needed. |
| `resume_with` | `{}` | Shape of the expected signal. |
| `decision_key` | `"approved"` | Which signal key is read as the approve/skip boolean. |
| `name`, `hint` | — | As for `Wait`. |

`Gate` mirrors the wrapped tool's `parameters`, so it drops into a Chain or an
Agent's toolset transparently.

### Stores

A store is where suspended runs live. Every Chain/Agent always has one (default
in-memory), so the engine has a single uniform path.

| Store | Use |
|---|---|
| `InMemoryStore()` | Default. Process-local; lost on restart. Fine for same-process human-in-the-loop. |
| `FileStore(dir)` | Persists each run to `<dir>/<run_id>.json` (atomic writes). Survives restart; shareable between processes on the same disk. |
| Subclass `StateStore` | Implement `save` / `load` / `delete` over S3, DynamoDB, Redis, any KV — for real serverless / multi-host. |

```python
from yait_aichain.state import InMemoryStore, FileStore

Chain(steps=[...], store=FileStore("runs/"))
Agent(orchestrator=..., tools=[...], store=FileStore("runs/"))
```

### `SuspendedResult`

Returned by `run()` (and `resume()`) when the run pauses.

| Attribute | Meaning |
|---|---|
| `run_id` | Pass to `resume(run_id, signal=...)` to continue. |
| `awaiting` | Dict describing the pause (`reason`, and the `resume_with` hint). |
| `document` | The run snapshot — `variables` done so far, step `status`, definition. |

It is **falsy**, so an accidental `if result:` treats a pause as "not done".

### `resume()`

```python
final = chain.resume(run_id, signal={...})        # Chain
result = agent.resume(run_id, signal={...})       # Agent
```

Loads the parked document, injects `signal` into the suspended step, and runs to
completion — returning the final value (or another `SuspendedResult` if it
pauses again). The document is deleted from the store on completion. A fresh
Chain/Agent built in another process resumes fine as long as it shares the same
store.

### `RunContext` — per-request context

Pass tenant + metadata (tracing ids, request info — **not** secrets; API keys
stay on `Model`/`Tool`) into a run:

```python
from yait_aichain.state import RunContext

chain.run(variables={...}, context=RunContext(tenant="acme", metadata={"req": "r-42"}))
chain.context.tenant          # "acme" — available for the duration of the run
chain.context.get("req")      # "r-42"
```

`RunContext(tenant=None, metadata={})` is a frozen value object with a
convenience `.get(key, default)` reader. It is exposed as `chain.context` /
`agent.context` while the run is in flight and is **persisted in the run
document**, so `resume()` restores it (even in another process — pass it again
to override). `Agent.run()` / `Agent.resume()` accept `context=` the same way.

### The cross-process pattern

The whole point: the two halves can be separate serverless invocations that
share nothing but the store.

```python
# Invocation 1 — start; it parks at the gate
def start():
    agent = build_agent(store=FileStore("/srv/runs"))
    res = agent.run("Issue a refund for order #123.")
    return res.run_id                      # persist this id somewhere

# Invocation 2 — a webhook delivers the decision later
def approval_webhook(run_id, approved):
    agent = build_agent(store=FileStore("/srv/runs"))   # fresh agent, same store
    return agent.resume(run_id, signal={"approved": approved})
```

The trigger needs only the `run_id`, the shared store, and the signal — no
reference to the original object.

---

## See also

- [Chain](chain.md) — `store=`, `run()` pausing, `resume()`.
- [Agent](../agents/overview.md) — the same suspend/resume on autonomous runs.
- [Tools](tools.md) — `Wait` and `Gate` alongside the other built-in tools.
