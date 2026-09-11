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
agent = Agent(model=Model("gpt-4o-mini"), tools=[...], hooks=[tracer])
agent.run("…")

for e in tracer.events:
    print(e.type, e.name, e.usage, e.duration)
```

Event types:

| type | emitted by | carries |
|---|---|---|
| `run.started` · `run.finished` | Agent, Chain | the task and mode; the verdict and usage |
| `tool_call.started` · `tool_call.ended` | Agent | one tool call — see below |
| `step.started` · `step.ended` | **Chain** | one chain step, `payload["kind"]` |
| `llm_call.started` · `llm_call.ended` | Skill | model name, tokens, cost, duration |

### Enough to rebuild a turn, not to describe it

`tool_call.*` carries what a program needs to reconstruct the call, because a
rendering of a result cannot be recovered downstream — units, column metadata,
and the difference between *no rows* and *the tool declined* are all gone once
it is prose, and a consumer that cannot tell a refusal from an empty answer
draws an empty chart for both.

```python
class Watch(Hook):
    def tool_call_started(self, e):
        print(e.name, e.payload["arguments"], e.payload["id"])

    def tool_call_ended(self, e):
        if e.error:
            print(e.name, "failed:", e.error)
        else:
            render(e.payload["result"])      # the tool's own value, verbatim
```

`payload["id"]` is the provider's id for the call. A model may ask for several
in one turn and the agent honours all of them, so the id is how a result pairs
with its arguments — and it stays the answer if calls ever execute
concurrently rather than in sequence.

**The result travels by value**, which makes an `Event` no longer uniformly
small: a thousand-row result is about 90 KB. A handle instead would need a
lifetime and somewhere to live, which is hostile to the serverless target —
the process that issued it may be gone. `Event.__repr__` omits the payload, so
`LoggingTracer` stays readable.

`run_id` is on every event of a run, including the `llm_call.*` a `Skill`
emits inside it: two concurrent invocations write into one stream, and without
it that stream cannot be demultiplexed afterwards. `step` says which turn a
tool call belongs to.

> **Renamed in 2.7.0.** The agent emitted its tool calls as `step.*` while
> this page documented `tool_call.*`, and `Chain` emits `step.*` for a chain
> step — one name meaning two things depending on which primitive a hook was
> attached to. A `Hook` subclass with `step_started` / `step_ended` still
> fires, once, with a `DeprecationWarning` naming the new method.

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

agent = Agent(model=Model("gpt-4o-mini"), hooks=[CostGuard()])
```

### Event fields

`type`, `run_id`, `step`, `name` (tool/model), `payload`, `usage` (token delta),
`cost` (USD delta), `duration` (seconds, on `*.ended`), `error`, `ts`.

---

## Watching a run as it happens

Hooks are a push channel: you hand the agent a callable and it calls you.
`stream()` is the same events pulled instead:

```python
agent = Agent(model=Model("gpt-4o"), tools=[...])

for event in agent.stream("audit the invoices"):
    print(event.type, event.payload)

result = agent.last_result      # the AgentResult run() would have returned
```

It is the **same loop**, walked rather than exhausted — not a second
implementation — so a run behaves identically whichever way it is driven, and
hooks you installed yourself still fire in the same order. Events emitted by
`Skill` (`llm_call.started`, `llm_call.ended`) arrive too, because the stream
is a view of the hook channel rather than a vocabulary of its own.

Three things to know:

* **Events, not tokens.** A turn is usually a tool call rather than prose, so
  token deltas would be empty for most of a run and would interleave with
  decisions in no useful order. What a caller wants to show is what the agent
  is *doing*. For text, `Skill.stream()` is the one.
* **The result is not yielded.** A stream of one type is easier to consume
  than a stream of two, so the result lands on `last_result` when the
  generator finishes, journal attached.
* **Abandoning the generator abandons the run.** Breaking out of the loop
  leaves the agent mid-turn: no result, no `run.finished`. The temporary hook
  is still removed — that much is guaranteed — but nothing else is.

The granularity is the turn, not the token: everything a turn emitted is
handed over when the turn ends. The loop is synchronous by design (no threads,
no async — the target is Lambda), so within one model call there is nothing to
interleave.

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

def ask(request):                        # ApprovalRequest
    answer = input(f"{request.tool}({request.arguments}) — run it? [y/N] ")
    return answer.strip().lower() == "y"

policy = PermissionPolicy({"financial": "approve", "destructive": "deny"})
agent  = Agent(Model("gpt-4o-mini"), tools=[IssueRefund()],
               permissions=policy, approve=ask)
```

Decisions:

- **`allow`** — run the tool.
- **`approve`** — ask `approve=` and run only on a yes. The callable is handed
  an `ApprovalRequest` carrying the tool's name, its risk class, **the
  arguments it would run with**, the call id and the asking agent's name —
  approving a name rather than a call is approving nothing, since the
  arguments are what separate a $5 refund from a $50,000 one.

  **With no `approve=` attached, the call is refused.** A decision whose whole
  content is "a human should see this first" cannot resolve to "go ahead"
  because no human was configured. The refusal names both ways out: attach an
  approver, or set that risk class to `allow` if it does not need gating here.
- **`deny`** — never run; the tool call still returns a (denial) result.

A refusal — policy or approver — comes back through the tool channel as a
result, not as a crash: the model is told and can choose something else. A
denied call it never hears about is one it will simply make again.

Shipped defaults gate `external` / `financial` / `privileged` behind approval and
deny `destructive`; `read` / `draft` / `write` run. **Enforcement is opt-in** —
an `Agent` without `permissions=` behaves exactly as before, and unmarked tools
default to `write` (allowed), so you tag only the risky tools.

A delegated worker inherits the approver along with the policy: rules without
an answerer would refuse every gated call, which reads as the policy being
stricter for children than for their parent.

> **Changed in 2.6.0.** `approve` used to be consulted and discarded — the
> agent acted on `deny` and on nothing else, so a gated tool ran. This page
> also documented an `agent.resume(...)` flow for it; `Agent.resume()` was
> removed in 2.0, when the agent's state became the conversation, so that
> example could not have run either. If you attached a policy relying on the
> old behaviour, pass `approve=` or relax the rule.

The model never decides its own permission — the policy lives outside it.

---

## Tool-call repair

Before a tool runs, its arguments are validated against the tool's
`parameters` schema. A malformed call (a missing required argument) does not
crash the run — it returns a **model-readable remediation message** as the step's
observation, so the orchestrator corrects the call within the step's attempt
budget (a `stop_when` ceiling).

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

**`Chain` only.** A chain is checkpointed to its `Store` after every committed
step, so an unplanned death — crash, OOM, function timeout — is recoverable: a
brand-new `Chain` sharing the store picks the run up with `resume(run_id)`.

```python
from yait_aichain.state import FileStore

chain = Chain(steps=[...], store=FileStore("./runs"))
chain.run(variables)     # process dies mid-run

# …restart, different process…
Chain(steps=[...], store=FileStore("./runs")).resume(run_id)
```

The agent has no equivalent and needs none: its state **is** the conversation,
a message list the caller can persist and hand back to a fresh `Agent`. There
is no `Agent(store=...)` and no `Agent.resume()` — both were removed in
`2.0.0`, and this section documented them for four minor versions after.

## Serverless note

Approval (`approve`) and human-in-the-loop both ride the existing
suspend/resume + `Store` machinery (see [state](../primitives/state.md)). The
run parks in the store; a separate process resumes it with only the `run_id` and
a shared store — the cross-process pattern from
[`examples/18_agent_external_trigger.py`](../../examples/18_agent_external_trigger.py)
and [`examples/20_observability.py`](../../examples/20_observability.py).
