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

## The attempt journal

*Since 1.5.2.* Every run keeps an **append-only, typed record of what was
attempted** — separate from `AgentMemory`, which holds the *data* the agent
works with. It rides on `AgentResult.journal` and is preserved across
suspend/resume.

```python
res = agent.run("…")
for e in res.journal:
    print(e["seq"], e["outcome"], e["evidence"]["kind"], e["intent"])
```

Each entry carries `seq` (the primary key), `intent`, `action`, `outcome`,
`evidence`, `reason`, `artifact`, `observation`, `tokens`, and — for the
plan-driven modes — `step` / `attempt`.

**Outcomes:** `done` · `failed` (may be retried) · `refuted` (established as not
viable) · `skipped` (blocked by the permission policy).

**Evidence is typed, and the asymmetry is the point:**

| kind | meaning |
|---|---|
| `check` | a programmatic fact — a tool raised, a permission denied. We *know*. |
| `model_claim` | the orchestrator asserts it worked. Nothing verified it. |

A failure is normally a `check`; a success is normally a `model_claim`. A run
whose `done` entries are all `model_claim` has proven nothing — and the journal
says so rather than hiding it behind a green result.

### The observation trail

`observation` is a bounded excerpt (240 chars) of **what came back**; the full
value lives in memory under `artifact`. `Journal.progress_summary()` renders the
recent trail — every outcome, not just successes — and in
[goal mode](../agents/overview.md#goal-mode) it is fed straight back into the
next action prompt.

This is not cosmetic. A record of what was *tried*, without what it *returned*,
is not enough to decide a next move: an agent that cannot see its own
observations will re-issue the same action forever. Failed attempts stay in the
trail for the same reason — a probe that failed still returned information.

### Do not redo

Entries recorded as `refuted` are rendered back into the next action prompt as an
`ALREADY RULED OUT` block, so a long run stops re-attempting what it has already
ruled out. The block is omitted entirely when nothing has been refuted.

### Detecting a stuck run

```python
from yait_aichain.agent import Journal
j = Journal.from_list(res.journal)
j.has_progress(5)      # False → only failed/skipped lately: the agent is spinning
j.refuted()            # what was ruled out, and why
j.done()               # what landed, and on what evidence
```

`has_progress()` is what makes an open-ended loop *stoppable*: a budget alone
would let an agent spin in place until the tokens run out.

### Spinning while succeeding

*Since 1.6.0.* `has_progress()` catches a run that is **failing**. It cannot
catch one that is **succeeding pointlessly** — an agent re-issuing a call that
returns cleanly every time records an honest `done` on every iteration, and
nothing in the outcomes says otherwise.

```python
j.is_repeating(5)      # True → the last 5 attempts were the same move
```

It fires only on exact repetition: the same stated intent throughout the
window, or the identical action re-issued. In [goal mode](../agents/overview.md#goal-mode)
the result is written into the next action prompt as a `STOP AND RECONSIDER`
block — **surfaced, never enforced**, since an agent legitimately circling a
hard sub-problem must not be cut off.

> A fuzzier rule, flagging intents that merely *resemble* each other, was
> measured against three real runs and rejected: an agent bisecting past its
> instrument's resolution scored a median word-overlap of 0.73 between
> consecutive intents, while a binary search that finished correctly scored
> 0.64. Not separable — restating "probe the midpoint of X–Y" every turn is
> what a healthy search looks like.

### Checkpoints and crash recovery

*Since 1.5.2.* The run is checkpointed to the `Store` **after every committed
step** — not only when it suspends. An unplanned death (crash, OOM, function
timeout) is therefore recoverable: a brand-new `Agent` instance sharing the
store picks the run up with `resume(run_id)`, restoring memory, the plan cursor
and the journal.

```python
agent = Agent(orchestrator=..., tools=[...], store=FileStore("./runs"))
agent.run(task)          # process dies mid-run

# …restart, different process…
Agent(orchestrator=..., tools=[...], store=FileStore("./runs")).resume(run_id)
```

Suspend/resume is now the special case of the same mechanism — a step that is
additionally marked `suspended` with a pending action to re-run against the
external signal.

> **Re-execution caveat.** Recovery restarts the step that was *in flight* when
> the process died — its completion was never recorded, so it must be retried
> (at-least-once). A side-effecting tool can therefore run twice. Gate such
> tools with a `PermissionPolicy` (an approval pauses before the effect) or make
> them idempotent. Committed steps are never re-run.

## Serverless note

Approval (`approve`) and human-in-the-loop both ride the existing
suspend/resume + `Store` machinery (see [state](../primitives/state.md)). The
run parks in the store; a separate process resumes it with only the `run_id` and
a shared store — the cross-process pattern from
[`examples/18_agent_external_trigger.py`](../../examples/18_agent_external_trigger.py)
and [`examples/20_observability.py`](../../examples/20_observability.py).
