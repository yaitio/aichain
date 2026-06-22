# Observability & control (the step boundary)

*Since 1.4.4.*

An agent is a control plane around a model: the model *proposes* actions; the
harness *validates, authorizes, executes, records, and returns observations*.
The step boundary makes every one of those steps legible and governable from
**outside** the model — without touching `run()`.

Four mechanisms, all opt-in and additive:

| Mechanism | What it gives you | API |
|---|---|---|
| **Logging** | human/ops diagnostics, routable anywhere | `logging` + `verbose=` |
| **Hooks / events** | a machine-readable stream at every boundary | `hooks=[...]`, `Event`, `Tracer` |
| **Permission matrix** | gate risky tools before they run | `risk` on `Tool`, `PermissionPolicy` |
| **Tool-call repair** | self-correct malformed tool calls | automatic |

---

## Logging

The library emits through named loggers (`yait_aichain.agent`, `.chain`,
`.skills`, …) and never configures a sink itself — a `NullHandler` on the root
`yait_aichain` logger keeps it silent until the application attaches a handler.

```python
import logging
logging.getLogger("yait_aichain").addHandler(logging.StreamHandler())
logging.getLogger("yait_aichain").setLevel(logging.INFO)
```

`verbose=1` / `verbose=2` on an `Agent` is a convenience that attaches a console
handler and raises the level (DEBUG at `2`) — the old behavior, now routed
through `logging` so the same records can go to a file, syslog, CloudWatch, or a
custom handler. Secrets (API keys, auth headers) are never logged.

**Logging is the diagnostic channel.** To drive product behavior — push a status
into a chat UI, write a trace to a database — use the event stream below, which
carries structure (`usage`, `cost`, `decision`) rather than text.

---

## Hooks & events

A **hook** is any callable `hook(event) -> None` passed to `Agent`, `Chain`, or
`Skill` via `hooks=[...]`. The engine emits an [`Event`](#event-fields) at every
boundary. A hook that raises is logged at DEBUG and never crashes the run, and a
hook cannot change behavior — observation only.

```python
from yait_aichain import Tracer

tracer = Tracer()                       # records every event into .events
agent = Agent(orchestrator=Model("gpt-4o-mini"), tools=[...], hooks=[tracer])
agent.run("…")

for e in tracer.events:
    print(e.type, e.name, e.usage, e.duration)
```

Event types:

- `run.started` · `run.finished` · `run.suspended` · `run.resumed`
- `step.started` · `step.ended`
- `llm_call.started` · `llm_call.ended`
- `tool_call.started` · `tool_call.ended`

### Convenience bases

- `Tracer` — records every event into `.events`.
- `LoggingTracer(logger, level)` — logs every event.
- `Hook` — subclass and implement only the boundaries you care about; an event
  of type `"tool_call.started"` dispatches to `tool_call_started(event)`.

```python
from yait_aichain import Hook

class CostGuard(Hook):
    def llm_call_ended(self, e):
        if e.cost: print(f"+${e.cost:.4f}")
    def run_finished(self, e):
        print(f"run done: {e.usage} tokens")

agent = Agent(orchestrator=Model("gpt-4o-mini"), hooks=[CostGuard()])
```

### Event fields

`type`, `run_id`, `step`, `name` (tool/model), `payload`, `usage` (token delta),
`cost` (USD delta), `duration` (seconds, on `*.ended`), `error`, `ts`.

---

## Permission matrix

A tool declares a **risk class** as data; a `PermissionPolicy` maps it to a
runtime decision the harness enforces *before* the tool runs.

```python
from yait_aichain.tools import Tool, FINANCIAL

class IssueRefund(Tool):
    name = "issue_refund"
    risk = FINANCIAL                    # read | draft | write | external |
    ...                                 # financial | destructive | privileged
```

```python
from yait_aichain import PermissionPolicy

policy = PermissionPolicy({"financial": "approve", "destructive": "deny"})
agent  = Agent(orchestrator=Model("gpt-4o-mini"),
               tools=[IssueRefund()], permissions=policy)
```

Decisions:

- **`allow`** — run the tool.
- **`approve`** — pause for an external approval, reusing suspend/resume; the
  agent returns a `SuspendedResult`. Resume with the decision:
  ```python
  result = agent.run("Refund order #123")          # SuspendedResult
  result = agent.resume(result.run_id, signal={"approved": True})
  ```
- **`deny`** — never run; the tool call still returns a (denial) result.

Shipped defaults gate `external` / `financial` / `privileged` behind approval and
deny `destructive`; `read` / `draft` / `write` run. **Enforcement is opt-in** —
an `Agent` without `permissions=` behaves exactly as before, and unmarked tools
default to `write` (allowed), so you tag only the risky tools.

The model never decides its own permission — the policy lives outside it.

---

## Tool-call repair

Before a tool runs, its arguments are validated against the tool's
`parameters` schema. A malformed call (a missing required argument) does not
crash the run — it returns a **model-readable remediation message** as the step's
observation, so the orchestrator corrects the call within the step's attempt
budget (`max_attempts`).

This upholds the harness invariant: **every tool call returns a result** — on
success, denial, validation failure, or error.

---

## Serverless note

Approval (`approve`) and human-in-the-loop both ride the existing
suspend/resume + `Store` machinery (see [state](../primitives/state.md)). The
run parks in the store; a separate process resumes it with only the `run_id` and
a shared store — the cross-process pattern from
[`examples/18_agent_external_trigger.py`](../../examples/18_agent_external_trigger.py)
and [`examples/20_observability.py`](../../examples/20_observability.py).
