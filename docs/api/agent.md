# API: Agent

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../agents/overview.md).

<!-- g:api -->
[`Agent`](#agent) · [`AgentResult`](#agentresult) · [`step_count`](#step_count) · [`token_budget`](#token_budget) · [`cost_budget`](#cost_budget) · [`check`](#check) · [`nudge`](#nudge) · [`stalled`](#stalled) · [`repeating`](#repeating) · [`Acceptance`](#acceptance) · [`acceptance_from`](#acceptance_from) · [`beacon`](#beacon) · [`Beacon`](#beacon) · [`ATTENTION`](#attention) · [`COORDINATION`](#coordination) · [`Journal`](#journal) · [`JournalEntry`](#journalentry) · [`evidence`](#evidence) · [`AgentMemory`](#agentmemory) · [`MemoryBackend`](#memorybackend) · [`InMemoryBackend`](#inmemorybackend) · [`FileBackend`](#filebackend)

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
)
```

A loop that holds one growing conversation and acts on the world.

**Parameters**

- `model` (*Model*) — The model that drives the loop.
- `tools` (*list[Tool], optional*, default `None`) — What it may call.
- `instructions` (*str, optional*, default `''`) — The standing brief — goes into the stable prefix.
- `mode` (*"agile" | "waterfall"*, default `'agile'`) — `"agile"` decides at every step. `"waterfall"` writes a plan at
  step 0 (via the `write_plan` tool) and holds to it.
- `team` (*list[Agent] | "auto" | None*, default `None`) — Who else may do work. `None` removes `delegate` from the tool set
  entirely; a list restricts it to those workers; `"auto"` lets the
  model describe new ones.
- `planner_model` (*Model, optional*, default `None`) — `waterfall` only: the model that writes the plan. The freeze makes
  the plan the highest-leverage tokens of the run — everything after
  merely executes it — so it may deserve deliberation the execution
  turns do not need. Turns before the plan is recorded go to this model;
  every turn after goes to `model`. Think while planning, act without
  deliberating.
- `stop_when` (*list, optional*, default `None`) — Conditions from this module. Defaults to `[step_count(30)]` —
  a backstop, not a plan.
- `hooks` (*list | None, optional*, default `None`) — Callables given an `Event` at every step boundary; `stream()` yields
  the same events.
- `permissions` (*PermissionPolicy | None, optional*, default `None`) — Maps a tool's `risk` class to `allow`, `approve` or `deny`,
  checked before the tool runs. With none, every tool runs.
- `approve` (*callable, optional*, default `None`) — Asked when the policy says `approve`: given an `ApprovalRequest`,
  returns `True`, `False` or `ApprovalDecision(False, reason)`.
  With `approve` in the policy and none attached, the call is refused.
- `store` (*StateStore, optional*, default `None`) — Where a run parked by a `wait()` or `gate()` in the agent's tools
  is kept until `resume()`. Required when such a tool is given.
- `max_cost` (*float | Budget | None, optional*, default `None`) — A ceiling in dollars for the whole run; a model call that would begin
  past it ends the run with `success=False` and the reason in
  `.error`.
- `name` (*str | None, optional*, default `None`) — How this agent is named as a worker in another agent's `team` and in
  events.
- `description` (*str | None, optional*, default `None`) — What it does — what a lead agent reads when choosing a worker.
- `verbose` (*int, optional*, default `0`) — `0` is silent, `1` logs each step, `2` logs everything, through the
  `yait_aichain` logger.

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

`run()` is untouched and returns the same result it always did;
this is the same loop walked instead of exhausted. Three things are
worth knowing before reaching for it:

* **Events, not tokens.** A turn in an agent loop is usually a tool
  call, not prose, so token deltas would be empty for most of a run
  and would arrive interleaved with decisions in no useful order.
  What a caller actually wants to show is what the agent is *doing* —
  which is what the event channel already carries. `Skill.stream` is
  the one for text.
* **The result is not yielded**, because a stream of one type is
  easier to consume than a stream of two. It lands on
  `last_result` when the generator finishes, with the journal
  attached, exactly as `run()` would have returned it.
* **Abandoning the generator abandons the run.** A `break` out of the
  loop leaves the agent mid-turn; the temporary hook is still removed
  (that much is guaranteed), but no result is produced and no
  `run.finished` is emitted. Exhaust it, or accept that.

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

The run exists before it has a task: a `run_id`, a document in the
store, and a declared shape for the signal that will start it. That is
a subscription — what a trigger waits on, and what a person is shown a
form for — and it is the same object a pause mid-run produces, so
"an email arrived", "the human pressed start" and "the gate was
approved" all arrive as `resume(run_id, signal)`.

Nothing is scheduled here. `hint` is recorded for whatever does the
waking: `{"on": "email.received"}`, `{"wake_at": "..."}`.

The signal carries the task: a string is the task, a dict gives it
under `"task"` and every other key seeds the conversation the way
`run(task, variables=...)` does.

**Parameters**

- `reason` (*str, optional*, default `''`) — What the run waits for, shown to whoever starts it.
- `resume_with` (*dict | None, optional*, default `None`) — The shape of the signal that starts it.
- `hint` (*dict | None, optional*, default `None`) — Recorded for whatever does the waking; the engine schedules nothing.
- `variables` (*dict | None, optional*, default `None`) — Data the run starts with, beside what the signal brings.

### `Agent.resume()`

```python
Agent.resume(run_id: str, signal=None) -> AgentResult | SuspendedResult
```

Continue a run this agent parked, delivering *signal* as the answer.

The pause and this call are the same seam the externally driven path
already has: a parked run is a history whose last tool call has no
result, and *signal* is that result. Nothing is restored that was not
written down — the conversation **is** the state, which is why 2.0
removed the agent's snapshot mechanism and why none is needed here.

Idempotent in the way a trigger needs: a `run_id` the store no
longer holds raises `KeyError`, so an at-least-once delivery can
read that as "already handled" instead of running the tail twice.

Returns an `AgentResult` when the run finishes, or another
`SuspendedResult` when it parks again — under the same
`run_id`, so a caller tracks one identifier for the whole
conversation however many times it waits.

**Parameters**

- `run_id` (*str*) — The parked run, as the `SuspendedResult` named it.
- `signal` (*any, optional*, default `None`) — The answer the run waited for — the result of the tool call that
  parked it. For a run made by `arm()`, a string is the task, or a dict
  gives it under `"task"`.

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

`ground_rules` injects the "plain text ends the run" rule, and it
belongs to whoever owns the loop: `run()` passes True because in its
loop a text reply really does end the run. An external driver calling
`opening()`/`step()` owns its own loop — there a plain reply is an
ordinary turn, so the default is clean instructions and the driver
states its own framing.

**Parameters**

- `task` (*str*) — What to do, in words.
- `variables` (*dict | None, optional*, default `None`) — Data the caller already holds, given with the task.
- `ground_rules` (*bool, optional*, default `False`) — Add the rule that a plain-text reply ends the run. `run()` sets it;
  a driver that owns its own loop usually does not.

### `Agent.step()`

```python
Agent.step(messages: list, state: dict | None = None)
```

Ask for the next decision and return it **without executing it**.

Returns a `ToolCallRequest` when the model asked
to act, or a string — the answer. The caller appends the reply (via
`.as_turn()`) and whatever the world answered, then calls again.

**The obligation in that sentence is now checked.** A history holding
a tool call nobody answered is malformed, and the providers disagree
about it in the worst way: three reject the request, Google is looser,
and a self-hosted OpenAI-compatible server validates nothing and
templates it through, so the model meets its own unanswered call and
improvises — differently each time. Provider interchangeability is
this library's main promise and that is a place it did not hold.

Raising here buys three things and costs one. It makes the providers
agree; it turns an intermittent, unbounded behaviour into the same
named failure every time, since the condition that produces a
dangling call is itself intermittent; and it fails locally, before
the network, pointing at the caller rather than at a provider's
wording. What it does not buy is a better score — on a benchmark a
crash can rank below a model muddling through. This is an instrument
for reliability, not for accuracy.

**Parameters**

- `messages` (*list*) — The conversation so far — `opening()`'s two messages, then each
  reply and each tool result in turn.
- `state` (*dict | None, optional*, default `None`) — The run's working state between calls, kept by the caller; the plan in
  `waterfall` mode lives here.

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
    plan: list[dict] = <factory>,
    adaptations: list = <factory>,
    history: list[dict] = <factory>,
    memory: dict = <factory>,
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
- `output` (*str | dict | None*) — The answer. `None` when *success* is `False`.
- `mode` (*str*) — `"agile"` or `"waterfall"`.
- `steps_taken` (*int*) — Model turns taken.
- `tokens_used` (*int*) — Tokens across every model call of the run, input and output together.
- `cost` (*float | None*, default `None`) — Dollars, priced per model from the registry; `None` when nothing
  could be priced, never an invented zero. A run picked up with
  `resume()` includes what it spent before the pause.
- `stopped_by` (*str | None*, default `None`) — What ended the run: `"answered"`, `"check:<name>"`,
  `"step_count"`, `"token_budget"`, `"cost_budget"`,
  `"truncated"` (a reply cut at `max_tokens` — raise it),
  `"llm_error"` (the model call failed otherwise),
  `"condition_error"` (a check, ceiling, nudge or approver of the
  caller's raised; `error` says which), or the name of an error.
- `plan` (*list[dict]*, default `<factory>`) — In `waterfall` mode, the plan it wrote — `{"goal": …}` per step.
- `adaptations` (*list*, default `<factory>`) — What the library changed in this run's requests to fit the provider,
  each an `Adaptation`; empty when nothing was.
- `history` (*list[dict]*, default `<factory>`) — Empty for an agent, kept so every result has the same fields; an
  agent's record is `journal`.
- `memory` (*dict*, default `<factory>`) — Always empty; the agent keeps no memory store. Removed in 5.0.
- `error` (*str | None*, default `None`) — Why it failed, when it did.
- `journal` (*list[dict]*, default `<factory>`) — What was attempted, in order, each entry with its outcome and the
  evidence behind it — `check` for a fact the code verified,
  `model_claim` for the model's word.
- `acceptance` (*Acceptance | None*, default `None`) — What the answer rests on: `checked`, `claimed` or `unsupported`,
  with a fingerprint of the evidence, so `still_holds()` can tell when
  it has changed.
- `beacons` (*list*, default `<factory>`) — Beacons raised during the run, including by delegated workers — a
  blocker, a question, a contract that changed.

**Examples**

```python
result = agent.run("Research fusion energy and summarise it in French")
if result:
    print(result.output, result.cost, result.stopped_by)
else:
    print("failed:", result.error)
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
check(fn, name: str = 'check')
```

Stop when *fn* says the objective is met — and call that a **success**.

The only completion the harness can verify itself. Everything else the
agent reports about its own progress is the model's word for it.

**Parameters**

- `fn` (*callable*) — Given the run's state; returns true when the objective is met.
- `name` (*str, optional*, default `'check'`) — How the stop is reported: `stopped_by="check:<name>"`.

---

## `nudge()`

```python
nudge(predicate, message, name: str = 'nudge')
```

A condition that **speaks** instead of ending the run.

`stop_when` knew two kinds, and both ended the run: a ceiling ("hit it,
fail") and a check ("met it, succeed"). A stall needs neither. It needs
"change approach", and the loop had no way to say that — so a stuck run
either spun until a ceiling stopped it or was stopped early by a ceiling
set too low to let a real run finish.

When *predicate(state)* is true a user turn carrying *message* is
appended and the loop carries on. It enters the conversation as an
ordinary, logged message — not an invisible edit to the system prompt —
so a reader of the transcript can see exactly what the model was told
and when. *message* may be a string or a callable of `state`.

It fires **once per streak**: while the predicate stays true it stays
quiet, and it re-arms when the predicate clears. A reminder repeated
every turn is how a reminder stops being read.

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

Signal whoever is waiting on this run.

Callable from anywhere inside a run — a tool, a nested agent, a pool item
— without a parameter, because the board is ambient for the duration of
the run, exactly as `RunContext` is.

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

`has_progress` catches a run that is *failing*; this catches one
that is *succeeding pointlessly*. An agent can keep issuing calls that
return cleanly — bisecting past the resolution of its own instrument,
re-asking a question already answered — and every entry is an honest
`done`. Nothing in the outcomes distinguishes that from real work.

Two signals, both requiring a full window and both deliberately strict,
because a false positive would interrupt an agent that is genuinely
working:

* every attempt in the window states the same intent, or
* every attempt in the window issues the identical action.

This reports; it does not stop. The judgement of whether a repeated
approach is still worth pursuing belongs to the caller.

**What this deliberately does not do.** A fuzzier version — flagging
intents that merely *resemble* each other — was measured against three
real runs and rejected. An agent bisecting past its instrument's
resolution scored a median word-overlap of 0.73 between consecutive
intents; a textbook binary search that finished correctly scored 0.64.
The two are not separable, because restating "probe the midpoint of
X–Y" every turn is what a healthy search looks like. Any threshold in
that gap is fitted to one example and would interrupt working runs, so
only exact repetition is reported.

### `Journal.progress_summary()`

```python
Journal.progress_summary(limit: int = 12) -> str
```

Render the observation trail for the next action prompt.

Every recent attempt appears, whatever its outcome, with **what was
tried** and **what came back**. Both halves are needed. A summary of
intents alone is useless to a loop that has to decide its next move from
feedback; and an observation without the candidate that produced it is
just as bad — "4 problems remain" cannot be acted on when the draft that
had 4 problems is nowhere to be seen. Failures earn their place here
too: a failed probe still returned information.

Both halves are bounded, so a genuinely large artifact will still be
clipped. That is what `artifact` is for: put it in memory, which the
action prompt renders separately and does not roll off after
*limit* entries.

Successes are additionally marked by evidence kind, so the model can see
which of its "achievements" nothing ever verified. What has been ruled
out is rendered separately by `do_not_redo`.

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

---

## `AgentMemory`

```python
AgentMemory(initial: dict | None = None, backend: MemoryBackend | None = None)
```

A key-value store with a pluggable backend.

Stores any Python value: strings, dicts, lists, numbers.  Nothing reads
it on its own — the agent loop does not use it — so what goes in comes
out only where your own code asks for it.

**Parameters**

- `initial` (*dict | None, optional*, default `None`) — Seed the memory with these key/value pairs.  Applied *on top of*
  any state loaded from the backend.
- `backend` (*MemoryBackend | None, optional*, default `None`) — Persistence backend.  When omitted, an `InMemoryBackend` is
  used (volatile, gone with the process).  Supply a
  `FileBackend` to persist state across processes.

**Examples**

Ephemeral memory (default):

```python
memory = AgentMemory({"language": "French", "topic": "fusion energy"})
```

File-backed memory:

```python
from yait_aichain.agent import AgentMemory, FileBackend

memory = AgentMemory(backend=FileBackend("~/.my_state.json"))
memory.set("topic", "fusion energy")
memory.flush()   # write it to the file
```

### `AgentMemory.set()`

```python
AgentMemory.set(key: str, value: Any)
```

Store *value* under *key*, overwriting any existing entry.

### `AgentMemory.get()`

```python
AgentMemory.get(key: str, default: Any = None) -> Any
```

Return the value for *key*, or *default* if not present.

### `AgentMemory.update()`

```python
AgentMemory.update(data: dict)
```

Merge *data* into the store (existing keys are overwritten).

### `AgentMemory.delete()`

```python
AgentMemory.delete(key: str)
```

Remove *key* from the store.  Silent no-op if absent.

### `AgentMemory.reset()`

```python
AgentMemory.reset()
```

Restore the in-process store to its construction-time baseline.

The baseline is the state loaded from the backend plus any `initial`
seed.  Call it to discard scratch values while keeping persistent
state (FileBackend contents, `initial`).  The backend itself is
never touched.

### `AgentMemory.clear()`

```python
AgentMemory.clear()
```

Remove all entries from the in-process store, the baseline, and the
backend.  This is a destructive, explicit operation; `reset`
is the one that keeps persistent state.

### `AgentMemory.flush()`

```python
AgentMemory.flush()
```

Persist the current in-process state to the backend without clearing.

Call it to checkpoint the current state.  The flushed state also
becomes the new baseline, so a later `reset` on the same
instance returns to the checkpoint instead of the construction-time
snapshot.

### `AgentMemory.all()`

```python
AgentMemory.all() -> dict
```

Return a shallow copy of the full store as a plain dict.

---

## `MemoryBackend`

```python
MemoryBackend()
```

Abstract base for memory persistence backends.

Subclasses must implement `load`, `save`, and `clear`.

### `MemoryBackend.load()`

```python
MemoryBackend.load() -> dict
```

Return the persisted key-value store as a plain dict.

### `MemoryBackend.save()`

```python
MemoryBackend.save(data: dict)
```

Persist *data*, overwriting any previously saved state.

### `MemoryBackend.clear()`

```python
MemoryBackend.clear()
```

Erase all persisted data.

---

## `InMemoryBackend`

```python
InMemoryBackend()
```

Volatile in-process backend.

State lives only in `self._data`; nothing is written to disk.

This is the default backend: with no `initial` seed the baseline is
empty, so each new `AgentMemory` starts with a clean store.

### `InMemoryBackend.load()`

```python
InMemoryBackend.load() -> dict
```

Return a copy of the in-process store.

### `InMemoryBackend.save()`

```python
InMemoryBackend.save(data: dict)
```

Replace the in-process store with *data*.

### `InMemoryBackend.clear()`

```python
InMemoryBackend.clear()
```

Wipe the in-process store.

---

## `FileBackend`

```python
FileBackend(path: str)
```

Durable JSON-file backend.

State is written to a UTF-8 JSON file via an atomic rename so that a
crash during `save()` never leaves a corrupt file.

**Parameters**

- `path` (*str*) — File path where the JSON state is stored.  `~` is expanded to the
  user's home directory.

**Examples**

```python
backend = FileBackend("~/.agent_memory.json")
memory  = AgentMemory(backend=backend)

# Persist the current state:
memory.flush()

# After reset() the same instance starts from the flushed
# checkpoint.  A new process restores the state the same way:
memory2 = AgentMemory(backend=FileBackend("~/.agent_memory.json"))
```

### `FileBackend.load()`

```python
FileBackend.load() -> dict
```

Load and return the JSON state, or `{}` on any error.

### `FileBackend.save()`

```python
FileBackend.save(data: dict)
```

Write *data* to the backing file atomically.

Creates parent directories as needed.  Writes to a temporary file
in the same directory, then renames it over the target path.

### `FileBackend.clear()`

```python
FileBackend.clear()
```

Delete the backing file if it exists.
<!-- /g:api -->
