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
— when step 4 is unknowable until step 3 answers — `mode="goal"` drops the
planning phase entirely: you give an objective and a **done condition**, and the
agent decides one action at a time from what it has learned.

```python
agent = Agent(
    orchestrator = Model("claude-sonnet-4-6"),
    tools        = [Probe(), RecordAnswer()],
    mode         = "goal",
    done_when    = lambda memory: "answer" in memory,   # the harness checks it
)
result = agent.run("Find the combination and record it.")
```

Each iteration the orchestrator sees the objective, the **observation trail**
(recent attempts *with what came back*) and what has been
[ruled out](observability.md#do-not-redo) — then picks one action. The
[journal](observability.md#the-attempt-journal) is not a side-effect here, it is
the loop's working memory.

`done_when` is required, and should be a **callable** where possible: a callable
lets the run finish on a `check`, a string only ever on a `model_claim`. See
[Configuration](configuration.md#done_when--goal-mode-only-required).

**Sub-agents.** A goal-mode agent with `team=` delegates in the plan-driven
mode, not in goal mode: `done_when` is a predicate over the parent's objective
and memory, and a scoped sub-task is exactly what a plan is for.

**Stopping.** An open-ended loop needs stop rules a plan gives for free:

| Rule | Meaning |
|---|---|
| `done_when` met | Success — the only exit that reports `success=True`. |
| `max_steps` | Iteration cap (default `50`). |
| `max_tokens` | Budget cap (default `250_000`). |
| No progress | The last `NO_PROGRESS_WINDOW` (5) attempts all failed — stop rather than burn the rest of the budget proving it again. |
| Repetition | The last `REPEAT_WINDOW` (5) attempts were the same move. Not a stop: the prompt says so and the model decides. |

The last one is the reason a goal run terminates in practice: a budget alone
lets an agent spin in place until the tokens run out. The verdict needs a full
window, so an early failure never ends a run that was about to recover.

> The loop is only as good as the orchestrator driving it. On
> [`examples/22_goal_mode.py`](../../examples/22_goal_mode.py) (find a number in
> 1–1000, 15 iterations): `claude-sonnet-4-6` ran a clean binary search and
> finished in 9 probes; `gpt-4o-mini` bisected for a while, then degenerated
> into +1 scanning and hit the cap. Same harness both times — which is exactly
> why the stop rules exist.

### Budgets

| Limit | Default | When hit |
|---|---|---|
| `max_steps` | `10` (`50` in goal mode) | The plan is truncated; never more than this many steps (iterations in goal mode). |
| `max_attempts` | `3` | Retries per step are capped, then the loop moves on. |
| `max_tokens` | `50_000` (`250_000` in goal mode) | Total across plan + actions + executions + reflections; the agent stops cleanly. |

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
