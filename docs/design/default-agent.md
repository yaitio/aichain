# Design — the default agent

Status: **specification**, not shipped. Written August 2026 after benchmarking
the three existing modes against a task with a machine-checkable answer
(`cookbook/reconcile`).

One agent, the shape the industry already runs, built out of this library's own
primitives. Everything unusual is removed so that a reader who has used any
other agent framework recognises it immediately.

## What it is

A conversation that can act — held with **the world**, not with itself. Every
second turn is a fact the model did not have, which is why the loop converges
instead of circling. When the other side stops supplying new facts it degenerates
into self-talk and spins: measured, three identical `memory_read` calls returning
the same value until the budget ran out.

```
messages = [system, task]
loop:
    reply = Skill(model, messages, tools).run()
    if reply has no action:      → done, reply is the answer
    result = execute(reply.action)
    messages += [reply, result]
    if stop_when(...):           → stop
```

That is the whole algorithm. No plan, no modes, no reflection pass.

## Why this shape

Verified against the alternatives during the benchmark:

* **The loop is what works.** It is the core of ReAct, Claude Code, OpenClaw,
  Cursor, and the Vercel AI SDK — whose agent class is literally called
  `ToolLoopAgent` and is the *only* implementation of its own `Agent` interface.
* **Three modes is a menu, not a feature.** `mode=` must be chosen before the
  task is understood — before anyone knows how many documents there are or
  whether the plan will need revising. On a task with an unknown item count,
  `waterfall`'s precondition is violated by construction and no setting saves it.
* **Rebuilding the prompt each step defeats prompt caching.** The current agent
  returns exactly two messages (system + one composed user) from each of its
  five prompt builders, and nothing ever appends. The prefix changes every step,
  so the cache never hits — on an agent that makes dozens of near-identical
  calls in a row. Caching is implemented and measured at −89% on warm calls;
  the agent is the one caller that cannot use it.

## What it is made of

| primitive | role |
|---|---|
| `Skill` | the loop itself — one growing conversation |
| `Tool` | what the model calls; the result is appended as the next turn |
| `Pool` | one action that repeats over a list — fan-out |
| `Chain` | one action that runs a fixed, developer-authored sequence |
| `Agent` | one action that delegates a scoped sub-task with its own conversation |

The agent composes the library instead of reimplementing it. The current one
imports only `Tool` and hand-rolls the rest, which is why cost accounting
arrived years late, prompt caching never arrived, and a structured-output schema
could not be expressed at all.

## Actions

The model either replies in plain text — that reply is the answer, and the run
ends — or asks for exactly one action:

```jsonc
{"type": "tool",  "name": "...", "arguments": {...}}
{"type": "pool",  "over": "...", "runner": {...}, "max_flows": 4}
{"type": "agent", "task": "...", "worker": "..."}     // offered only when `team` is set
```

There is deliberately no `answer` action: answering *is* the absence of an
action, one exit rather than two spellings of it.

`pool` exists because every other action is exactly **one** call. A loop asked
to "read every document" without it does one document and reports success —
measured at 2 of 8, with no warning. The list length is read at execution time,
so nothing has to be known in advance.

`agent` is gated by `team`, and the gate is validated before execution:
`team=None` — the action is not in the vocabulary at all; `team=[a, b]` — it
may only name a member of the cast; `team="auto"` — it may describe a new
worker.

There is no `chain` action and no registry to go with it. The registry already
exists — it is `tools`: a developer-authored `Chain` is wrapped as a `Tool` and
listed like everything else. One registration mechanism, not two.

## Configuration

Two parameters carry the whole design space, and they are orthogonal: `mode`
answers *is the sequence frozen*, `team` answers *who does the work*.

```python
mode = "agile" | "waterfall"        # default "agile"
team = None | [agents] | "auto"     # default None
```

| `mode` | `team` | what it is |
|---|---|---|
| `agile` | — | the loop |
| `waterfall` | — | a plan written at step 0, then held |
| `agile` | `[a, b]` | the loop routing work across a given cast |
| `waterfall` | `[a, b]` | a plan executed across a given cast |
| `agile` | `"auto"` | the loop spawning workers as it needs them |
| `waterfall` | `"auto"` | designs a team at step 0, then runs it |

Six combinations, each a distinct requirement — this table is the logic to
implement, not a comparison with anyone. Delegation here is always **call with
return**: the parent hands a scoped task down and receives the result. Handing
the whole conversation over to another agent with no return is a different
mechanism, and it is deliberately not part of this design.

`goal` disappears: it *was* the loop without a plan, which is what `agile` now
means. They were distinguished only by the plan, and a plan you may rewrite is
a to-do list, not control flow.

Everything else is model, tools, instructions, and one stop list:

```python
Agent(
    model        = Model("gpt-5.6-terra"),
    tools        = [...],
    instructions = "...",
    stop_when    = [step_count(30), token_budget(150_000), cost_budget(0.50)],
)
```

The ordinary exit is built into the loop and needs no condition: the model
replies without asking for an action, and that reply is the answer. `stop_when`
holds **everything else** that may end a run, in one readable list. Today those
are scattered — `max_steps`, `max_tokens`, `max_attempts`, `done_when` — and
which one ended a run can only be recovered by reading the code.

Three ways a run ends, and conflating them is what made the benchmark unreadable:

| | meaning | result |
|---|---|---|
| answered | the model replied without asking for an action | **success** |
| `check(fn)` | a harness-verified condition was met | **success**, with `check` evidence |
| `step_count`, `token_budget`, `cost_budget` | a ceiling was reached | **`success=False`** |

Whichever fired is named in `AgentResult.stopped_by` — `"answered"`,
`"check:<name>"`, `"step_count"`, `"cost_budget"` — so "finished" and "gave
out" can never again look identical from the outside.

`cost_budget` is the honest one to ask for and the one to caveat: output length
is not known before a call, so it bounds "do not begin another step", not "never
exceed by a cent".

**Dropped from configuration:**

* `max_attempts` — it governed retries of a *plan step*. A loop has no plan
  steps: a call that fails comes back as the next turn and the model decides.
* `done_when` — the key goes with `mode="goal"`. Its string form was a
  completion the *model* judged, which the loop now answers by simply replying
  without asking for an action. Its callable form was a completion the *harness*
  checked — the strongest evidence class there is — and that is not lost: it
  becomes one more entry in `stop_when`, alongside the ceilings rather than
  special-cased beside them.
* preview limits — nothing is truncated while state lives in the conversation.
  They return only with eviction, and then as eviction settings.
* `allow_spawn` — becomes `team="auto"`. It is not a new parameter, only the
  same capability under an honest name and switched on.
* `executors` — a worker's model is the worker's own property: members of
  `team=[...]` arrive carrying their models, and workers spawned under
  `"auto"` inherit the parent's. A price-aware roster for the composer stays
  an open item.

No wall-clock budget: `timeout` and `retries` are already `Model` options and
belong at the transport, where a caller can set them.

`AgentResult` carries `tokens_used`, `cost` and `stopped_by`. Cost is not
derivable from tokens: they are summed input and output, and output costs five
to six times input at frontier rates. Both come from `Skill`'s `Usage`, so
there is one implementation.

## Durability

The run state **is** the conversation. A checkpoint is `messages` plus the
journal plus the counters; resume is: load, continue the loop. That is strictly
simpler than the current agent's parked document — plan, memory, cursor and a
pending action, four things that must stay mutually consistent — and the
serverless requirement of *one invocation, one step* falls out naturally:
execute one action per invocation, persist the appended turns.

## Context

Everything lives in one growing conversation. It grows monotonically — nothing
is discarded — so the design is bounded by the context window. Three answers, in
the order to reach for them:

1. **Caching.** Does not shorten anything; makes the repeated prefix ~10× cheaper
   on reads. Free and automatic, because a conversation that only ever appends
   has a stable prefix.
2. **Delegation.** The `agent` action is first of all a **context-management**
   mechanism, not a way to parallelise. A sub-agent reads fifty documents in its
   own conversation and returns three conclusions; the parent gains one turn
   instead of fifty. This is why Claude Code ships a Task tool.
3. **Eviction to memory.** Only when the first two are not enough. Old turns move
   out and a pointer stays. This is where `AgentMemory` earns its place — and
   where preview limits come back, as eviction settings.

The current agent pays for (3) always, uses (1) never — the prompt is rebuilt
each step, so the cache cannot hit — and has (2) switched off by default since
1.0.0. That combination produced a livelock on a 3 KB task.

## Observability

The **journal** is kept, and it is the only part of the current agent that
survives unchanged. Every action produces one typed entry: what was attempted,
what happened, and the evidence class — `check` (the tool raised, or did not)
versus `model_claim` (only the model says so).

Without a reflection pass this gets *better*, not worse: outcomes come from
execution facts rather than the model's own assessment, so more entries carry
`check` and fewer carry `model_claim`.

The non-negotiable property: **a run that did less than it claimed leaves a
trace.** Every defect found during the benchmark was silent — truncation with no
notice, a fan-out returning a list of `None`, plan steps dropped, a tool step
answered in prose. Not one produced an error.

A sub-agent's return enters the parent's journal as `model_claim`, together
with the child's `stopped_by`: the parent can see whether the child answered or
hit a ceiling, but must not treat the child's words as verified. Delegation
multiplies the places where "it says it did" can stand in for "we checked" —
this is the line that keeps them countable.

## What is deliberately absent

**Reflection after every step.** The model sees the previous result as the next
turn; a separate assessment call is a second opinion on something already in
front of it. It was roughly half of goal mode's bill — $0.23 across 38 calls,
against $0.04 across 5 for a composed pipeline at the same recall.

**External memory by default.** State lives in the conversation until it does
not fit. `AgentMemory` becomes an eviction mechanism, not a permanent layer.
The layer cost us a livelock on a **3 KB** task: eight documents were bundled
into one value, the prompt showed 500 characters of it and said "read the rest
with `memory_read`", the agent did, the result was not persisted, and the next
prompt said the same thing again. Three identical calls, then the budget ran out.

**Modes.** See below — they turn out not to be architectures at all.

## The plan is a tool, not a structure

A plan may be genuinely useful: it answers "did I cover all of it", it makes
cost estimable before the run, and it lets a human read the decomposition before
it executes. The benchmark's central failure was agents doing part of the work
and reporting success — a checklist is what makes that answerable.

But it must be **context, not control flow**. Today's plan is a state machine:
`step_idx` walks the list and one step is exactly one call, which is why "read
every document" read two of eight. As a checklist the model keeps, the loop stays
free to spend as many calls on an item as the item needs.

So it ships as a `Tool` — `plan.write`, `plan.read`, `plan.check_off` — not as a
framework structure. The base loop does not change at all; an agent given the
tool behaves like plan-and-execute when the task calls for it.

Placement follows from caching, and it differs by mode:

* **`agile` has no plan of the framework's making.** The decision is taken at
  every step. If the model keeps a checklist through the plan tool, that
  checklist lives in the conversation tail like any other tool result — it is
  working state, and working state never goes into the prefix.
* **`waterfall`'s plan is written once at step 0 and frozen**, so it may be
  pinned into the prefix: one invalidation when it lands, stable ever after.
  That is exactly what makes the freeze pay for itself.
* **Progress goes in neither.** Checking off "item 2 done" inside the prefix
  would invalidate it on every step and cancel the cache. Progress is read off
  the conversation — what was called is what was done.

### Which is why the modes collapse

The axis is not "mode". It is *when the sequence is fixed, and by whom*:

| | plan | written by | fixed when |
|---|---|---|---|
| `Chain` | yes, rigid | **the developer** | before the run |
| `waterfall` | yes, rigid | **the model** | at step 0 |
| default (= `agile`) | none | — | decided at every step |

`agile` disappears. "Revise the plan" and "decide the next step from what just
happened" are the same act, not two modes; every loop does it by construction and
no setting grants or withholds it.

The freeze in `waterfall` is **temporal, not authorial** — the model writes the
plan once and does not rewrite it. An earlier draft of this document justified
read-only with human sign-off; that was an invented story. Nothing about
`waterfall` requires a human.

`Chain` stays a separate primitive despite being the same shape, for a reason
that is not authorship: **it spends no model call on planning.** Its sequence is
deterministic, runnable and testable offline. `waterfall` pays one call at step 0
and gets a sequence that is generated text, so two runs may differ. A developer
who already knows the order should not pay a model to guess it.

Whether the middle row earns its place is **an open empirical question**, not a
settled design: `waterfall` has never completed an honest run. Three defects of
ours broke it, and it now aborts on `replan`.

## Nothing new is needed

The primitives are already closed under composition:

| | accepts |
|---|---|
| `Chain` | `Skill`, `Tool`, `Agent` — in any order |
| `Pool` | `Tool`, `Skill`, `Chain`, `Agent` |
| `Agent` | a loop that may call any of them |

So a branch of one call is a `Chain` of length 1, and a branch of three steps is
a `Chain` of length 3. `scaffold` is `Pool(Chain(...), items)` with a plan on
top — its reviewers do the *same* work with a different **parameter** (the remit),
which is `Pool` exactly, not some heterogeneous-parallel construct.

Two constructs were proposed during this design and both were withdrawn once the
library was actually read: an `agent` step type with nested budgets and memory
isolation (`spawn` already exists), and a `parallel` primitive for
heterogeneous branches (`Pool` already covers it — the heterogeneity is in the
data, not the structure).

**The one real gap is access.** The Agent cannot reach what is already built:

* no `chain` action in the plan;
* a `pool` runner may only be `tool` or `skill`, though `Pool` itself accepts
  `Chain` and `Agent`;
* `spawn` has been off by default since 1.0.0.

That is the shape of every defect found this session: **a capability built and
unreachable.**

## Prompt ownership

One rule decides what text the library may put in front of the model:
**a prompt belongs to whoever's behaviour it describes.**

| text | owner | mechanism |
|---|---|---|
| "plain text ends the run" | **the loop's owner** | injected by `run()` only; an external driver states its own framing |
| "write the plan first, hold to it" | the library, with `mode="waterfall"` | `_PLAN_RULE` |
| what `pool`/`delegate`/`write_plan` do | the library — its own machinery | synthetic tool descriptions, on the tool channel |
| result and error rendering | the library | `result_message` |
| role, domain, policy, tone | **the caller** | `instructions` |
| tool schemas | nobody — not prompt text at all | the provider's `tools` field |

The library's entire prompt surface is ~350 characters. Every line of the
boundary was bought with a measurement: the exit rule imposed on an external
driver biased a dialogue agent toward polite refusal; schemas as prompt text
cost a 42%-correct call rate against 86% native; the unconditional plan rule
outscored both always-thinking and think-on-demand configurations.

Two placement corollaries:

* **Deliberation is placed by the harness, not self-diagnosed.** A thinking
  `planner_model` on the one plan turn beat reasoning-everywhere and beat an
  expert-on-demand — because the agent that most needs advice does not
  experience its confusion as uncertainty, so it never asks. Unconditional
  placement lands; optional placement waits for a self-diagnosis that does
  not come.
* **The driver's framing is the driver's.** The τ² adapter states "asking the
  user for details is a normal part of the conversation"; the library does not
  know it is in a dialogue and must not pretend to.

## Validation

Anything the model authored is checked before it runs. During the benchmark the
model authored three compositions and all three were wrong, silently: a `skill`
runner where a tool call was required (returning four sentences explaining it
could not call the tool), a JSON schema the provider rejected, and a reference
to a memory key nothing filled. All three are catchable by reading the action,
with no model call at all.

## Open questions

* **Whether `waterfall` earns its row.** Empirical, not settled: it has never
  completed an honest run — three defects of ours broke it before it could be
  measured. The freeze buys a cacheable plan and a predictable bill; whether
  that beats deciding at every step is for the benchmark to answer.
* **Whether `team="auto"` beats `team=[...]`.** Never measured: there is no run
  yet where a model-composed cast and a human-composed cast face the same task
  with equal access to the data.
* **Who composes.** Prompted, as here, or trained, as Sakana Fugu's TRINITY.
  Theirs will assign roles better. The advantage here is that the composition
  is readable, checkable and priced.
* **Eviction design.** What moves out when the conversation outgrows the
  window, what the pointer left behind looks like, and where the preview limits
  return as eviction settings.
* **A price-aware roster.** For the composer to trade capability against cost
  it must see prices — they are in the registry already; what it is shown, and
  in which prompt, is unspecified.
