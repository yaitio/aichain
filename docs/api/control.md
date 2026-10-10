# API: Control — budget, permissions, events

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../control/cost-and-budget.md).

<!-- g:api -->
[`Budget`](#budget) · [`BudgetExceeded`](#budgetexceeded) · [`PermissionPolicy`](#permissionpolicy) · [`ApprovalRequest`](#approvalrequest) · [`ApprovalDecision`](#approvaldecision) · [`Event`](#event) · [`Hook`](#hook) · [`Tracer`](#tracer) · [`LoggingTracer`](#loggingtracer) · [`RISK_CLASSES`](#risk_classes) · [`DECISIONS`](#decisions)

---

## `Budget`

```python
Budget(limit: float)
```

A ceiling in dollars, and what has been spent against it.

Thread-safe because `Pool` runs its items concurrently and a budget that
can be overspent by a race is not a budget.

```python
budget = Budget(0.50)
Chain(steps=[...], max_cost=budget).run()
budget.spent        # what it cost
budget.remaining    # what is left
```

**Parameters**

- `limit` (*float*) — The ceiling, in dollars; must be positive.

### `Budget.check()`

```python
Budget.check(what: str = '')
```

Refuse to begin another call when there is nothing left.

**Parameters**

- `what` (*str, optional*, default `''`) — What was about to begin, for the error's message.

### `Budget.charge()`

```python
Budget.charge(amount: 'float | None')
```

Record what a call cost. `None` is free — a provider that does
not price its answers cannot be billed against a ceiling, and
inventing a number would be worse than admitting it.

**Parameters**

- `amount` (*float | None*) — What the call cost; `None` when it could not be priced.

---

## `BudgetExceeded`

```python
BudgetExceeded(spent: float, limit: float, what: str = '')
```

A run stopped because its money ceiling was reached.

A raise rather than a partial result. Returning what was finished with a
flag set is how a truncated run gets read as a complete one — the same
silent-success shape this library has spent its 2.x line removing — and a
caller who wants the partial output can catch this and read
`last_usage`.

**Parameters**

- `spent` (*float*) — Dollars spent when the call was refused.
- `limit` (*float*) — The ceiling.
- `what` (*str, optional*, default `''`) — What was refused.

---

## `PermissionPolicy`

```python
PermissionPolicy(rules: 'dict | None' = None, *, default: str = 'approve')
```

Maps a tool's risk class to a runtime decision (`allow`/`approve`/
`deny`). The model never decides its own permission — the harness consults
this policy, which lives outside the model.

**Parameters**

- `rules` (*dict[str, str] | None*, default `None`) — Overrides merged over the shipped defaults, e.g.
  `{"external": "allow", "financial": "deny"}`.
- `default` (*str*, default `'approve'`) — Decision for a risk class not present in the rules (default
  `"approve"` — unknown/unclassified risk is gated, not silently run).

**Examples**

>>> policy = PermissionPolicy({"financial": "approve", "destructive": "deny"})
>>> policy.decide_risk("financial")
'approve'

### `PermissionPolicy.decide_risk()`

```python
PermissionPolicy.decide_risk(risk: str) -> str
```

Return the decision for a bare *risk* class string.

**Parameters**

- `risk` (*str*) — A risk class.

### `PermissionPolicy.decide()`

```python
PermissionPolicy.decide(tool) -> str
```

Return the decision for *tool* based on its `risk` attribute.

**Parameters**

- `tool` (*Tool*) — The tool about to run; its `risk` is read.

---

## `ApprovalRequest`

```python
ApprovalRequest(
    tool: str,
    risk: str,
    arguments: dict,
    call_id: str = '',
    agent: str = '',
)
```

What an approver is shown before the tool runs.

Enough to decide with, and no more: the tool's own name and risk class,
the arguments it would run with, and the call id so a request can be
paired with the event stream and with the result that follows. Deliberately
not the Tool object — an approver that can reach in and run it is not a
gate.

**Attributes**

- `tool` (*the tool's name.*)
- `risk` (*its risk class, which is what the policy actually judged.*)
- `arguments` (*what it would be called with.*)
- `call_id` (*the provider's id for this call, when there is one.*, default `''`)
- `agent` (*the name of the agent asking, which matters once a team is*, default `''`) — delegating — the approver needs to know who wants this.

---

## `ApprovalDecision`

```python
ApprovalDecision(granted: bool, reason: str = '')
```

An approver's answer, when a bare yes/no is not enough.

Returning `True` or `False` stays the short form and is what most
approvers will use. This exists for the other half of the requirement: a
refusal that says *why*. A UI that shows "the tool was not approved" and
nothing else has thrown away the only part a person can act on, and the
reason cannot be recovered afterwards — it lived in the head of whoever
clicked no.

**Attributes**

- `granted` (*whether the call may run.*)
- `reason` (*one sentence, carried into the event stream and into the*, default `''`) — denial the model is told about.

---

## `Event`

```python
Event(
    type: str,
    run_id: 'str | None' = None,
    step: 'int | None' = None,
    name: 'str | None' = None,
    payload: dict = <factory>,
    usage: 'int | None' = None,
    cost: 'float | None' = None,
    duration: 'float | None' = None,
    error: 'str | None' = None,
    ts: float = <factory>,
)
```

One operational event at a step boundary. Carries only operational fields —
never hidden chain-of-thought.

**Attributes**

- `type` (*str*) — Dotted event name, e.g. `"llm_call.started"`, `"tool_call.ended"`,
  `"tool_call.started"`, `"step.started"` (a **Chain** step;
  the agent's tool events were spelled this way before 2.7.0).
- `run_id` (*str | None*, default `None`) — Identifier of the run this event belongs to.
- `step` (*int | None*, default `None`) — Zero-based step index within the run, when applicable.
- `name` (*str | None*, default `None`) — Tool or model name, when applicable.
- `payload` (*dict*, default `<factory>`) — Event-specific extra fields. For `tool_call.*` this carries what a
  program needs to **reconstruct** the turn rather than describe it:
  `id` (so concurrent calls in one turn can be told apart),
  `arguments` on started, and the tool's **raw** result on ended.

  The result travels by value, and that is a deliberate choice about
  what an `Event` is. A rendering of it cannot be recovered
  downstream — units, column metadata, the difference between *no rows*
  and *the tool declined* are all gone once it is prose — and a handle
  instead would need a lifetime and a place to live, which is hostile to
  the serverless niche this library targets: the process that issued the
  handle may be gone. Events are therefore no longer uniformly small;
  `Event.__repr__` omits the payload so a log stays readable.
- `usage` (*int | None*, default `None`) — Token delta attributable to this event, when known.
- `cost` (*float | None*, default `None`) — Estimated USD cost delta, when known.
- `duration` (*float | None*, default `None`) — Wall-clock seconds the operation took (set on `*.ended` events).
- `error` (*str | None*, default `None`) — Error message when the operation failed.
- `ts` (*float*, default `<factory>`) — Unix timestamp when the event was created (filled automatically).

---

## `Hook`

```python
Hook()
```

Optional convenience base for a hook.

Subclass and implement any subset of boundary methods; an event of type
`"tool_call.started"` is dispatched to `tool_call_started(event)`. The
raw callable form (a plain `def hook(event): ...`) works too — `Hook` is
only ergonomics.

### `Hook()`

```python
Hook(event: 'Event')
```

Call self as a function.

---

## `Tracer`

```python
Tracer()
```

A hook that records every event into `.events` for later inspection.

### `Tracer()`

```python
Tracer(event: 'Event')
```

Call self as a function.

---

## `LoggingTracer`

```python
LoggingTracer(logger: 'logging.Logger | None' = None, level: int = 20)
```

A hook that logs every event to a `logging` logger (default INFO).

**Parameters**

- `logger` (*logging.Logger | None, optional*, default `None`) — Where to log; `yait_aichain.trace` by default.
- `level` (*int, optional*, default `20`) — The level each event is logged at; `INFO` by default.

### `LoggingTracer()`

```python
LoggingTracer(event: 'Event')
```

Call self as a function.

---

## `RISK_CLASSES`

`['destructive', 'draft', 'external', 'financial', 'privileged', 'read', 'write']`

---

## `DECISIONS`

`['allow', 'approve', 'deny']`
<!-- /g:api -->
