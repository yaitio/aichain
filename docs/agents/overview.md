# Agent

An **Agent** is a conversation that can act. You give it a task; it is asked
what to do, the world answers, the answer is appended, and the loop goes round
until it replies without asking for anything more. Use it when the *sequence*
of steps is not known upfront. For a fixed pipeline, use a
[Chain](../primitives/chain.md) — it spends no model call deciding an order you
already know.

---

## Quick start

```python
import os
from yait_aichain.models import Model
from yait_aichain.agent  import Agent, step_count, cost_budget
from yait_aichain.tools  import PerplexitySearchTool

agent = Agent(
    Model("claude-opus-5", api_key=os.getenv("ANTHROPIC_API_KEY")),
    tools     = [PerplexitySearchTool()],
    stop_when = [step_count(8), cost_budget(0.50)],
)

result = agent.run("Compare the top 3 managed vector databases and return a Markdown table.")
print(result.output)
print(result.stopped_by,                       # "answered" — or the ceiling that fired
      result.steps_taken, "steps",
      f"{result.tokens_used:,} tokens", f"${result.cost or 0:.4f}")
```

▶ One tool: [`examples/13_agent.py`](../../examples/13_agent.py) ·
Multiple tools: [`examples/14_agent_tools.py`](../../examples/14_agent_tools.py) ·
Delegation: [`examples/15_agent_orchestrator.py`](../../examples/15_agent_orchestrator.py) ·
A verified stop condition: [`examples/22_goal_mode.py`](../../examples/22_goal_mode.py)

---

## Common gotchas

- **`run()` never raises.** It returns an `AgentResult`; check `result.success`
  (and `bool(result)` *is* `success`). On failure, read `result.error`.
- **Answering is the absence of an action.** The ordinary exit is the model
  replying without asking for a tool. There is no separate "finish" action.
- **A ceiling reached is not success.** `stop_when` holds everything else that
  can end a run, and `result.stopped_by` says which fired — `"answered"`,
  `"check:<name>"`, `"step_count"`, `"cost_budget"`. Without that field a run
  that finished and a run that ran out look identical from the outside.
- **`check()` is the strong one.** A callable the harness evaluates itself, so
  the run ends on a fact rather than on the model's word for it.
- **Repeating work over a list needs `pool`.** Every other action is exactly one
  call, so "read every document" without it reads one — silently.

---

## Reference

### Constructor

```python
Agent(model, tools=None, instructions="", mode="agile", team=None,
      stop_when=None, hooks=None, permissions=None,
      name=None, description=None, verbose=0)
```

| Parameter | Default | Description |
|---|---|---|
| `model` | — | The `Model` that drives the loop. |
| `tools` | `None` | Tools it may call. |
| `instructions` | `""` | The standing brief. Goes in the stable prefix, so it is cached from the second call onward. |
| `mode` | `"agile"` | `"agile"` decides at every step; `"waterfall"` writes a plan at step 0 and holds to it. |
| `team` | `None` | `None` — nobody else. `[agents]` — delegate only to these. `"auto"` — describe and spawn workers as needed. |
| `stop_when` | `[step_count(30)]` | Everything that can end a run other than answering. |
| `hooks` | `None` | Observability sinks ([Observability](observability.md)). |
| `permissions` | `None` | Tool governance. |
| `verbose` | `0` | `0` silent · `1` per-action status · `2` token breakdowns. |

`agent.run(task, variables=None)` executes and returns an `AgentResult`.
`variables` seeds the conversation with data the caller already holds — this is
how a `Chain` or `Pool` step hands its accumulated values down.

### The loop

```
messages = [system, task]
loop:
    reply = ask the model
    if the reply has no action  → done, the reply is the answer
    result = execute(reply.action)
    messages += [reply, result]
    if a stop condition fires   → stop
```

That is all of it. The conversation is held with **the world**, not with
itself: every second turn is a fact the model did not have, which is why the
loop converges instead of circling.

The conversation only ever **appends** — nothing is rebuilt — so the cacheable
prefix grows and prompt caching works without being configured.

### Actions

The model either replies in plain text, which ends the run, or asks for exactly
one action:

| Action | What it does |
|---|---|
| `tool` | Call one tool with arguments. |
| `pool` | Repeat one runner over a list, in parallel. The only action that is more than one call. |
| `agent` | Delegate a scoped sub-task. Offered only when `team` is set. |

### Externally driven

`run()` owns the loop. When something else needs to — a serverless invocation
that must be one step, or a harness that executes the tools itself — the same
machinery is exposed:

```python
messages = agent.opening(task)
state    = agent.new_state()
decision = agent.step(messages, state)      # decide, do not execute
result, error = agent.execute(decision["action"], state)
```

Reflection also assigns a `store_as` key: a snake_case name where the step
output lands in memory for later steps to reference.

### Modes

| Mode | Plan can change? | Use when |
|---|---|---|
| `waterfall` (default) | No (retries only) | The path is predictable. |
| `agile` | Yes, via `replan` | The path is exploratory; later steps depend on what early ones reveal. |
| `goal` | There is no plan | The steps cannot be known in advance at all. |

#### Goal mode

*Since 1.6.0.* A plan is a bet that you know the steps up front. When you don't
— when step 4 is unknowable until step 3 answers — **that is what `"agile"`
already is**: it decides one action at a time from what it has learned, with
nothing written in advance.

```python
from yait_aichain.agent import Agent, step_count, check

agent = Agent(
    model     = Model("claude-sonnet-4-6"),
    tools     = [Probe(), RecordAnswer()],
    mode      = "agile",
    stop_when = [check(lambda state: answer_recorded()), step_count(50)],
)
result = agent.run("Find the combination and record it.")
```

Each iteration the model sees the objective and the conversation so far —
every attempt *with what came back* — then picks one action. The
[journal](observability.md#the-attempt-journal) is not a side-effect here, it
is the record of that.

**Stopping.** An open-ended loop needs stop rules a plan gives for free, and
they all live in `stop_when`:

| Rule | Meaning |
|---|---|
| `check(fn)` | Success — the harness verified a condition. |
| `step_count(n)` | Iteration ceiling. `success=False`: a ceiling reached is not an answer. |
| `token_budget(n)` / `cost_budget(x)` | Spending ceilings, same verdict. |

> The loop is only as good as the model driving it. On
> [`examples/22_goal_mode.py`](../../examples/22_goal_mode.py) (find a number in
> 1–1000, 15 iterations): `claude-sonnet-4-6` ran a clean binary search and
> finished in 9 probes; `gpt-4o-mini` bisected for a while, then degenerated
> into +1 scanning and hit the cap. Same harness both times — which is exactly
> why the stop rules exist.

> **Corrected 2026-09-11.** This section documented a third mode, `"goal"`,
> with a required `done_when` and its own defaults for `max_steps` /
> `max_tokens`; the modes are `"agile"` and `"waterfall"`, and the ceilings
> are entries in `stop_when`. It also listed two automatic stop rules — "no
> progress" over a 5-attempt window, and repetition — that **no library code
> fires**: `Journal.has_progress()` and `Journal.is_repeating()` exist and are
> called from nowhere, so a run that stalls today spins until a ceiling stops
> it. Making them fire is the `nudge` work in the plan, not a thing the
> library does now.

### `AgentResult`

`run()` never raises — inspect the result:

| Field | Description |
|---|---|
| `success: bool` | `True` on completion; `bool(result)` mirrors it. |
| `output` | The final answer (`None` on failure). |
| `steps_taken: int` | Plan steps executed. |
| `tokens_used: int` | Total tokens across all LLM calls. |
| `cost: float \| None` | Estimated USD for those calls, priced per model. `None` when nothing is priceable — a self-hosted model has no price per token, and this says so rather than reporting `0.0`. Not derivable from `tokens_used`, which sums input and output while output costs several times more. Covers this invocation; a `resume()` reports only the resumed leg. |
| `plan: list[dict]` | The final plan (may differ from the first in agile mode). |
| `history: list[dict]` | Per-attempt trace: action, output, reflection, tokens. |
| `memory: dict` | Memory snapshot at the end. |
| `error: str \| None` | Failure reason. |

```python
for rec in result.history:
    print(f"step {rec['step']+1} [{rec['action_type']}] {rec['step_goal']}"
          f" → {rec['reflection']['assessment']} ({rec['tokens']:,} tok)")
```

### Suspend & resume

**This is a `Chain` capability, not an agent one.** A chain can pause for an
external signal and continue later, in another process, through `Wait`/`Gate`
and a persistent store — see [State](../primitives/state.md).

The agent deliberately has none of that. `2.0.0` removed its suspend/resume
because its state **is** the conversation: a message list, serialisable by
construction, which a caller can put away and hand back to a new `Agent`
without the library owning a store. There is no `Agent(store=...)` and no
`Agent.resume()`.

What that leaves for a human in the loop is `approve=`, which decides
**before** the tool runs rather than by pausing after the model asked:

```python
agent = Agent(Model("gpt-4o"), tools=[IssueRefund()],
              permissions=PermissionPolicy({"financial": "approve"}),
              approve=lambda req: ask_the_manager(req.tool, req.arguments))
```

See [Observability](observability.md#permission-matrix) for the request an
approver is handed and what happens when there is none.

> **Corrected in 2.6.0.** This section previously showed `Agent(store=...)`,
> `agent.resume(...)` and a `SuspendedResult` from `agent.run()` — none of
> which have existed since `2.0.0`. A `Gate` tool handed to an agent does not
> pause it either: the `Suspend` it raises is caught like any other tool
> failure and reported to the model as an error.

### Sub-agents

With `team="auto"`, the orchestrator gets a `delegate` tool and can hand a
sub-task to a fresh agent (its own tools/model), then use the result. Pass a
list of agents instead to delegate to named workers. Good for fan-out research
— one worker per topic.

A worker inherits the parent's permissions **and its approver**: rules without
an answerer would refuse every gated call, which reads as the policy being
stricter for children than for their parent.

> **Corrected in 2.6.0.** `allow_spawn=True` and `spawn_agent` are the older
> names; the parameter is `team=` and the tool is `delegate`. See
> [design/default-agent.md](../design/default-agent.md), which recorded the
> rename while this page kept the old spelling.

### When to reach for an Agent

| Situation | Use |
|---|---|
| One model call, known prompt | [Skill](../primitives/skills.md) |
| Fixed sequence with known data flow | [Chain](../primitives/chain.md) |
| Search → read → cross-reference → reason | **Agent** |
| Exploratory; next step depends on what you learned | **Agent (agile)** |
| No plan is possible; only a finish line | **Agent (goal)** |

An agent can also be one step *inside* a Chain — it handles the open-ended
phase, the Chain handles the rest. See [Agent as a Chain step](agent-as-chain-step.md).

---

## See also

- [Configuration](configuration.md) — tools, executors, persona, modes in depth.
- [Memory](memory.md) — shared step state and persistent cross-run memory.
- [Observability & control](observability.md) — hooks, events, the permission matrix, tool-call repair.
- [State](../primitives/state.md) — suspend/resume, stores, `Wait`/`Gate`.
- [Agent as a Chain step](agent-as-chain-step.md).
