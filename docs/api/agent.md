# API: Agent

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../agents/overview.md).

<!-- g:api -->
[`Agent`](#agent) · [`AgentResult`](#agentresult) · [`step_count`](#step_count) · [`token_budget`](#token_budget) · [`cost_budget`](#cost_budget) · [`check`](#check) · [`nudge`](#nudge) · [`stalled`](#stalled) · [`repeating`](#repeating) · [`verified`](#verified) · [`Acceptance`](#acceptance) · [`acceptance_from`](#acceptance_from) · [`beacon`](#beacon) · [`Beacon`](#beacon) · [`ATTENTION`](#attention) · [`COORDINATION`](#coordination) · [`Journal`](#journal) · [`JournalEntry`](#journalentry) · [`evidence`](#evidence)

---

## `Agent`

```python
Agent(
    model: Model,
    tools: list | None = None,
    instructions: str = '',
    mode: str = 'agile',
    team: list[Agent] | str | None = None,
    planner_model: Model | None = None,
    stop_when: list | None = None,
    hooks: list | None = None,
    permissions: PermissionPolicy | None = None,
    approve: Callable[[ApprovalRequest], Any] | None = None,
    store: StateStore | None = None,
    max_cost: float | Budget | None = None,
    name: str | None = None,
    description: str | None = None,
    verbose: int = 0,
    interceptors: list | None = None,
    policy: dict | PermissionPolicy | None = None,
)
```

A loop that holds one growing conversation and acts on the world.

**Parameters**

- `model` (*Model*) — The model that drives the loop.
- `tools` (*list[Tool], optional*, default `None`) — What it may call.
- `instructions` (*str, optional*, default `''`) — The standing brief — goes into the stable prefix.
- `mode` (*"agile" | "waterfall"*, default `'agile'`) — `"agile"` decides at every step; `"waterfall"` plans at step 0 and holds to it.
- `team` (*dict | list[Agent] | None*, default `None`) — `{"members": [agents, chains], "extend": bool, "max": N, "max_delegations": N}`;
  a list is the members, `None` removes `delegate`; `"auto"` is deprecated.
- `planner_model` (*Model, optional*, default `None`) — `waterfall` only: the model that writes the plan; `model` runs it.
- `stop_when` (*list, optional*, default `None`) — Conditions from this module. Defaults to `[step_count(30)]`.
- `hooks` (*list | None, optional*, default `None`) — Callables given an `Event` at every step boundary.
- `policy` (*dict | PermissionPolicy | None, optional*, default `None`) — `{"rules": {risk: "allow"|"approve"|"deny"}, "default": ..., "approve":
  "park"|"deny"|fn, "interceptors": [...]}`, or the rules alone; none runs all.
- `permissions` (*PermissionPolicy | None, optional*, default `None`) — Deprecated; goes in 5.0, use `policy={"rules": ...}`.
- `approve` (*callable, optional*, default `None`) — Deprecated; goes in 5.0, use `policy={"approve": ...}`.
- `store` (*StateStore, optional*, default `None`) — Where a parked run is kept until `resume()`; needed by `wait()`/`gate()`.
- `max_cost` (*float | Budget | None, optional*, default `None`) — A ceiling in dollars for the run; past it the run ends `success=False`.
- `name` (*str | None, optional*, default `None`) — This agent's name as a worker in another's `team` and in events.
- `description` (*str | None, optional*, default `None`) — What it does — what a lead reads when choosing a worker.
- `verbose` (*int, optional*, default `0`) — `0` silent, `1` each step, `2` everything, to the `yait_aichain` logger.
- `interceptors` (*list | None, optional*, default `None`) — Deprecated; goes in 5.0, use `policy={"interceptors": [...]}`.

### `Agent.run()`

```python
Agent.run(task: str, variables: dict | None = None) -> AgentResult
```

Execute *task* and return an `AgentResult`.

*variables* seeds the conversation with data the caller already holds —
this is how a `Chain` or `Pool` step hands its accumulated values
down. There is one place state lives, and it is the conversation.

**Parameters**

- `task` (*str*) — What to do, in words.
- `variables` (*dict | None, optional*, default `None`) — Data the caller already holds, given to the model with the task.

### `Agent.stream()`

```python
Agent.stream(task: str, variables: dict | None = None)
```

Run *task* and yield each `Event` as it happens.

The same loop as `run()`, walked instead of exhausted. It yields events, not
tokens — `Skill.stream` is the one for text. The result is not yielded: it
lands on `last_result` when the generator finishes. Abandoning the generator
abandons the run, with no result and no `run.finished`.

**Parameters**

- `task` (*str*) — What to do, in words.
- `variables` (*dict | None, optional*, default `None`) — Data the caller already holds, given to the model with the task.

### `Agent.arm()`

```python
Agent.arm(
    reason: str = '',
    resume_with: dict | None = None,
    *,
    hint: dict | None = None,
    variables: dict | None = None,
) -> SuspendedResult
```

Create the run **without starting it**, and return it parked.

`resume(run_id, signal)` starts it: a string is the task, or a dict gives
it under `"task"` and its other keys seed the conversation.

**Parameters**

- `reason` (*str, optional*, default `''`) — What the run waits for, shown to whoever starts it.
- `resume_with` (*dict | None, optional*, default `None`) — The shape of the signal that starts it.
- `hint` (*dict | None, optional*, default `None`) — Recorded for whatever does the waking, such as `{"on": "email.received"}`.
- `variables` (*dict | None, optional*, default `None`) — Data the run starts with, beside what the signal brings.

### `Agent.resume()`

```python
Agent.resume(run_id: str, signal=<required>) -> AgentResult | SuspendedResult
```

Continue a run this agent parked, delivering *signal* as the answer.

**Parameters**

- `run_id` (*str*) — The parked run, as the `SuspendedResult` named it.
- `signal` (*any, optional*, default `<required>`) — The result of the tool call that parked the run; for an `arm()` run,
  the task (a string, or a dict with `"task"`).

**Returns**

- `AgentResult | SuspendedResult` — The finished run, or the run parked again under the same `run_id`.

**Raises**

- `KeyError` — *run_id* is not in the store — already resumed, or never parked.

### `Agent.new_state()`

```python
Agent.new_state() -> dict
```

Fresh run state — the counters `stop_when` conditions read.

### `Agent.opening()`

```python
Agent.opening(
    task: str,
    variables: dict | None = None,
    *,
    ground_rules: bool = False,
) -> list
```

The two messages a run starts from: the stable prefix and the task.

**Parameters**

- `task` (*str*) — What to do, in words.
- `variables` (*dict | None, optional*, default `None`) — Data the caller already holds, given with the task.
- `ground_rules` (*bool, optional*, default `False`) — Add the rule that a plain-text reply ends the run. `run()` sets it; a
  driver that owns its own loop usually does not.

### `Agent.step()`

```python
Agent.step(messages: list, state: dict | None = None)
```

Ask for the next decision and return it **without executing it**.

Returns a `ToolCallRequest` when the model asked to act,
or a string — the answer. The caller appends the reply (`.as_turn()`) and a
result for every call, then calls again; `owns` names the calls that go
to `execute`.

**Parameters**

- `messages` (*list*) — The conversation so far — `opening()`'s two messages, then each reply
  and each tool result in turn.
- `state` (*dict | None, optional*, default `None`) — The run's working state, kept by the caller; holds the `waterfall` plan.

**Raises**

- `ValueError` — When a tool call already in *messages* has no result.

### `Agent.stopped()`

```python
Agent.stopped(state: dict) -> str | None
```

The name of the terminal condition that holds, or None — what
`run()` asks after every turn, for a driver of `step()` to ask too.

**Parameters**

- `state` (*dict*) — The state passed to `step()`.

### `Agent.nudges()`

```python
Agent.nudges(state: dict) -> list
```

The messages of nudges that fire now, once per streak — what
`run()` appends as user turns, for a driver of `step()` to append.

**Parameters**

- `state` (*dict*) — The state passed to `step()`.

### `Agent.owns()`

```python
Agent.owns(call) -> bool
```

Whether *call* is one of the agent's own — `pool`, `write_plan`,
`delegate` — which exist only inside this library and so go to
`execute`, never to the driver's environment. The tools the
agent was given are the driver's to place: a tool declared by schema
alone runs in the world, one with a body may run in either.

**Parameters**

- `call` (*ToolCall*) — One call from a `ToolCallRequest` that `step()` returned.

### `Agent.execute()`

```python
Agent.execute(call, state: dict | None = None)
```

Run one `ToolCall`. Returns `(result, error)`: a tool that
fails is an error, not an exception, and so is a call the permission
policy refuses — the same policy, approver included, that `run()`
applies, because this is the seam a serverless driver runs tools
through and a policy it skipped would govern nothing there. The one
exception that passes is `Suspend` — a `wait()` or `gate()`
asking the run to park, which the driver handles as a pause.

**Parameters**

- `call` (*ToolCall*) — One call from a `ToolCallRequest` that `step()` returned.
- `state` (*dict | None, optional*, default `None`) — The same working state passed to `step()`.

---

## `AgentResult`

```python
AgentResult(
    success: bool,
    output: Any,
    mode: str,
    steps_taken: int,
    tokens_used: int,
    cost: float | None = None,
    stopped_by: str | None = None,
    output_from: str = 'answer',
    plan: list[dict] = <factory>,
    adaptations: list = <factory>,
    history: list[dict] = <factory>,
    error: str | None = None,
    journal: list[dict] = <factory>,
    acceptance: object | None = None,
    beacons: list = <factory>,
)
```

What `run` returns: the answer, how the run ended, and what it
cost. `bool(result)` is `result.success`.

**Attributes**

- `success` (*bool*) — `True` when the agent answered, or a `check()` passed.
- `output` (*str | dict | None*) — The answer. Never `None` on success; on a failure, `None` or the
  text cut at `max_tokens`.
- `output_from` (*str*, default `'answer'`) — Where *output* came from: `"answer"` (the model answered), and for
  a run a `check()` ended, `"check"` (its `output=`),
  `"check_answer"` (its `answer=True`), `"tool_result"` (the last
  tool call that succeeded) or `"none"` (only the check's name).
- `mode` (*str*) — `"agile"` or `"waterfall"`.
- `steps_taken` (*int*) — Model turns taken.
- `tokens_used` (*int*) — Tokens across every model call of the run, input and output together.
- `cost` (*float | None*, default `None`) — Dollars, resumed parts included; `None` when nothing could be priced.
- `stopped_by` (*str | None*, default `None`) — `"answered"`, `"check:<name>"`, `"step_count"`, `"token_budget"`, `"cost_budget"`,
  `"truncated"` (cut at `max_tokens`; not a success), `"context_overflow"`, `"park_failed"`,
  `"llm_error"`, `"condition_error"` (a caller's callable raised), or an error's name.
- `plan` (*list[dict]*, default `<factory>`) — In `waterfall` mode, the plan it wrote — `{"goal": …}` per step.
- `adaptations` (*list*, default `<factory>`) — Each `Adaptation` the library made to fit the provider.
- `history` (*list[dict]*, default `<factory>`) — Empty for an agent, kept so every result has the same fields.
- `error` (*str | None*, default `None`) — Why it failed, when it did.
- `journal` (*list[dict]*, default `<factory>`) — What was attempted, each with its outcome and evidence (`check` or `model_claim`).
- `acceptance` (*Acceptance | None*, default `None`) — What the answer rests on — `checked`, `claimed` or `unsupported`.
- `beacons` (*list*, default `<factory>`) — Beacons raised during the run, delegated workers' included.

**Examples**

```python
result = agent.run("Summarise fusion energy in French")
print(result.output if result else result.error, result.cost)
```

### `AgentResult.usage`

Tokens and cost as a `Usage`, the field `ChainResult` and
`PoolResult` carry too. An agent sums only the total across its
turns, so the input/output split is left at zero rather than invented.

---

## `step_count()`

```python
step_count(n: int)
```

Stop after *n* decisions. A ceiling: reaching it is not success.

**Parameters**

- `n` (*int*) — Model turns allowed.

---

## `token_budget()`

```python
token_budget(n: int)
```

Stop once *n* tokens have been spent. A ceiling.

**Parameters**

- `n` (*int*) — Tokens allowed, input and output together.

---

## `cost_budget()`

```python
cost_budget(usd: float)
```

Stop once *usd* has been spent. A ceiling.

Bounds "do not begin another step", not "never exceed by a cent": the
length of a reply is not known before it is paid for.

**Parameters**

- `usd` (*float*) — Dollars allowed.

---

## `check()`

```python
check(fn, name: str = 'check', *, output=None, answer: bool = False)
```

Stop when *fn* says the objective is met — a **success**, the only one the
harness verifies itself. The output is what *output* returns, else with *answer*
the model's closing reply, else the last tool result; `output_from` says which.

**Parameters**

- `fn` (*callable*) — Given the run's state; true when the objective is met.
- `name` (*str, optional*, default `'check'`) — Reported as `stopped_by="check:<name>"`.
- `output` (*callable | None, optional*, default `None`) — Given the run's state; returns the output.
- `answer` (*bool, optional*, default `False`) — One more model reply, without tools, as the output; it costs a call.

---

## `nudge()`

```python
nudge(predicate, message, name: str = 'nudge')
```

A condition that **speaks** instead of ending the run.

When *predicate(state)* is true, *message* is appended as a user turn and the
loop carries on — once per streak, re-armed when the predicate clears.

**Parameters**

- `predicate` (*callable*) — Given the run's state; true when the model should be told something.
- `message` (*str | callable*) — What it is told — a string, or a function of the state.
- `name` (*str, optional*, default `'nudge'`) — How the nudge is named in events.

---

## `stalled()`

```python
stalled(last_k: int = 5)
```

Nudge when the last *last_k* attempts moved nothing.

`Journal.has_progress` has existed, been documented, and been called
from no library code at all — available only to someone writing their
own loop. It reads the journal, not the task metric, so it has nothing to
overfit to and applies to every task, verifiable or not.

**Parameters**

- `last_k` (*int, optional*, default `5`) — Attempts that must all have moved nothing.

---

## `repeating()`

```python
repeating(last_k: int = 5)
```

Nudge when the last *last_k* attempts were the same move.

The other half of a stall: `has_progress` catches a run that is failing,
this catches one that is succeeding pointlessly — clean results that
carry no new information. Like `stalled`, it existed in the journal and
nothing called it.

**Parameters**

- `last_k` (*int, optional*, default `5`) — Attempts that must all have been the same move.

---

## `verified()`

```python
verified(message: str | None = None, name: str = 'verified')
```

Before the run answers, ask once for a check of what it changed.

An answer after a `write`/`external`/`financial`/`destructive` call is
held and *message* is put to the model, unless a later call succeeded. Once per
run; only a call clears it, never the model saying it checked.

**Parameters**

- `message` (*str | None, optional*, default `None`) — What the model is told; a default names the change.
- `name` (*str, optional*, default `'verified'`) — How the nudge is reported in its event.

---

## `Acceptance`

```python
Acceptance(
    kind: str,
    fingerprint: str,
    checks: tuple = (),
    claims: tuple = (),
    superseded: str = '',
)
```

What a finished result rests on, and whether that still holds.

**Attributes**

- `kind` (*str*) — `checked`, `claimed` or `unsupported`.
- `fingerprint` (*str*) — A hash of the checked evidence, so a later look can tell it changed.
- `checks` (*tuple*, default `()`) — The checks it rests on.
- `claims` (*tuple*, default `()`) — The model's claims it rests on.
- `superseded` (*str*, default `''`) — Why it no longer holds; empty while it does.

### `Acceptance.still_holds()`

```python
Acceptance.still_holds(journal) -> bool
```

Recompute over fresh evidence and compare.

*journal* is evidence obtained **again** — the same actions re-run,
the same files re-read. If what the checks observed has changed, the
result was accepted on facts that are no longer true.

**Parameters**

- `journal` (*list | Journal*) — Evidence gathered again, after the run.

### `Acceptance.supersede()`

```python
Acceptance.supersede(reason: str) -> Acceptance
```

The caller intervened; the acceptance must be re-earned.

**Parameters**

- `reason` (*str*) — Why it must be re-earned.

---

## `acceptance_from()`

```python
acceptance_from(journal) -> Acceptance
```

Derive an `Acceptance` from a run's journal.

Only completed attempts count, and only their evidence *kind* decides the
category. The fingerprint covers the **checked** evidence alone: a claim
changing does not change what was verified, and folding claims in would
let a model re-wording its report look like the facts moving.

**Parameters**

- `journal` (*list | Journal*) — A run's journal.

---

## `beacon()`

```python
beacon(kind: str, message: str, *, source: str = '') -> Beacon
```

Signal whoever is waiting on this run, from anywhere inside it.

**Parameters**

- `kind` (*str*) — `blocker`, `question` and `contract_change` demand attention and
  end a parent's wait early; `contract`, `decision` and `fact` are
  coordination records.
- `message` (*str*) — What it says.
- `source` (*str, optional*, default `''`) — Who raised it.

---

## `Beacon`

```python
Beacon(kind: str, message: str, source: str = '', ts: float = <factory>)
```

One signal from a child to whoever is waiting on it.

---

## `ATTENTION`

`['blocker', 'contract_change', 'question']`

---

## `COORDINATION`

`['contract', 'decision', 'fact']`

---

## `Journal`

```python
Journal(entries: list[JournalEntry] | None = None)
```

Append-only log of attempts, with the three views the loop needs.

### `Journal.append()`

```python
Journal.append(
    intent: str = '',
    *,
    action: dict | None = None,
    outcome: str = 'done',
    evidence: dict | None = None,
    reason: str = '',
    artifact: str | None = None,
    observation: str = '',
    step: int | None = None,
    attempt: int | None = None,
    tokens: int = 0,
) -> JournalEntry
```

Record one attempt and return the stored entry.

### `Journal.done()`

```python
Journal.done() -> list[JournalEntry]
```

Entries that finished — the progress record.

### `Journal.refuted()`

```python
Journal.refuted() -> list[JournalEntry]
```

Approaches already ruled out — the "do not redo" set.

### `Journal.has_progress()`

```python
Journal.has_progress(last_k: int = 5) -> bool
```

True when anything actually moved in the last *last_k* entries.

Movement means a `done` (work landed) or a `refuted` (the search
space shrank). A tail of only `failed` / `skipped` means the agent is
spinning — the stop rule a goal loop needs, since a budget alone would
let it spin until the tokens run out.

The verdict needs a **full window**: with fewer than *last_k* entries
there is not yet enough evidence to call a run stuck, and reporting one
would let a single early failure end the run before it had a chance to
recover.

### `Journal.is_repeating()`

```python
Journal.is_repeating(last_k: int = 5) -> bool
```

True when the last *last_k* attempts stopped carrying new information.

It fires when every attempt in a full window states the same intent or issues
the identical action — exact repetition only, so a working search is not
interrupted. It reports and does not stop; `has_progress` catches failing.

**Parameters**

- `last_k` (*int, optional*, default `5`) — The window, in entries; fewer entries than this is never repeating.

### `Journal.do_not_redo()`

```python
Journal.do_not_redo(limit: int = 10) -> str
```

Render the refuted set for the action prompt. Empty string when nothing
has been ruled out, so the block is simply omitted.

---

## `JournalEntry`

```python
JournalEntry(
    seq: int,
    intent: str = '',
    action: dict = <factory>,
    outcome: str = 'done',
    evidence: dict | None = None,
    reason: str = '',
    artifact: str | None = None,
    observation: str = '',
    observation_sha: str = '',
    step: int | None = None,
    attempt: int | None = None,
    tokens: int = 0,
    ts: float = <factory>,
)
```

One recorded attempt.

**Attributes**

- `seq` (*monotonic sequence number — the primary key.*)
- `intent` (*what this attempt was trying to achieve (a plan step's goal in*, default `''`) — waterfall/agile, the iteration's intent in a goal loop).
- `action` (*the action that was executed (tool call / skill call).*, default `<factory>`)
- `outcome` (*one of `done` / `failed` / `refuted` / `skipped`.*, default `'done'`)
- `evidence` (*`{"kind": "check"|"model_claim", "detail": str}` or `None`.*, default `None`)
- `reason` (*why it failed or was refuted — the text fed back as "do not redo".*, default `''`)
- `artifact` (*reference (a memory key / path) to a bulky result; the result*, default `None`) — itself is NOT copied into the journal.
- `observation` (*a short excerpt of what came back. An attempt record that*, default `''`) — says only what was *tried* is not enough to decide the next move — the
  loop must be able to read its own feedback. Truncated to
  `OBSERVATION_CHARS`; the full value lives under *artifact*.
- `observation_sha` (*a hash of the whole observation, which is what an*, default `''`) — acceptance's fingerprint compares — not the excerpt.
- `step, attempt` (*plan coordinates; `None` in a goal loop.*)
- `tokens` (*tokens spent on this attempt.*, default `0`)
- `ts` (*unix timestamp.*, default `<factory>`)

---

## `evidence()`

```python
evidence(kind: str, detail: str = '') -> dict
```

Build an evidence record, validating *kind*.

**Parameters**

- `kind` (*str*) — `check` for a fact code verified, or `model_claim` for the
  model's word that nothing verified.
- `detail` (*str, optional*, default `''`) — What the evidence is.
<!-- /g:api -->
