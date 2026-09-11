# Agent configuration

Every parameter that shapes how an Agent decides, acts, and spends.

```python
from yait_aichain.models import Model
from yait_aichain.agent  import Agent, step_count, token_budget
from yait_aichain.tools  import PerplexitySearchTool, MarkItDownTool

agent = Agent(
    model        = Model("claude-opus-4-6"),
    tools        = [PerplexitySearchTool(), MarkItDownTool()],
    mode         = "agile",
    stop_when    = [step_count(12), token_budget(80_000)],
    instructions = "You are a senior market intelligence analyst…",
    verbose      = 1,
    name         = "market_research_agent",
)
```

> **Rewritten 2026-09-11.** This page documented the pre-`2.0.0` agent —
> `orchestrator`, `executors`, `goal` mode, `done_when`, `max_steps`,
> `max_attempts`, `max_tokens`, `persona`, `memory`. Every one of those is
> either renamed or gone, every decision is recorded in
> [design/default-agent.md](../design/default-agent.md), and none of it
> reached this page: an example copied from here raised `TypeError` on its
> first line. What follows is checked against the constructor by
> `tests/test_docs_promise_what_exists.py`.

---

## Required

### `model` — the brain

The model that decides what happens next, every turn. **Use a capable
reasoning model**: the loop is only as good as the model driving it.

```python
Agent(model=Model("claude-opus-4-6"))
```

It is the first positional parameter, so `Agent(Model("gpt-4o"))` is the same
thing.

---

## Tools

### `tools: list[Tool] | None`

What the agent can do. Schemas travel as the provider's native `tools` field,
not as prompt text, and a reply asking for one comes back as a typed
`ToolCall`.

An agent with no tools still runs — it just answers.

### `team: list[Agent] | "auto" | None`

Delegation. `"auto"` gives the model a `delegate` tool that hands a sub-task
to a fresh agent inheriting the parent's model, tools, permissions and
approver; a list delegates to named workers, each carrying its own model.

There is no `executors=`: a worker's model is the worker's own property.

---

## Mode

### `mode: "agile" | "waterfall"` — default `"agile"`

- **`"agile"`** decides at every step, with nothing written in advance.
- **`"waterfall"`** writes a plan first and then follows it. The plan is
  frozen: it cannot be rewritten mid-run, which is what makes a run readable
  afterwards.

There is no `"goal"` mode. Its ordinary exit — the model replying without
asking for an action — is built into the loop and needs no condition, and its
`done_when` split cleanly in two: the string form was a completion the *model*
judged, which is now just that ordinary exit; the callable form was a
completion the *harness* checked, which is `check(fn)` in `stop_when` below.

### `planner_model: Model | None` — `waterfall` only

A different, usually stronger model for writing the plan. Passing it in any
other mode raises: a planner with no planning phase is a parameter that
silently does nothing.

---

## Stopping

### `stop_when: list | None`

Everything that can end a run except the ordinary exit, in one readable list.

```python
from yait_aichain.agent import step_count, token_budget, cost_budget, check

stop_when = [step_count(12), token_budget(80_000), cost_budget(0.50),
             check(lambda state: Path("report.md").exists())]
```

Three ways a run ends, and conflating them is what made a benchmark
unreadable:

| | meaning | result |
|---|---|---|
| answered | the model replied without asking for an action | **success** |
| `check(fn)` | a condition the harness verified | **success**, with `check` evidence |
| `step_count`, `token_budget`, `cost_budget` | a ceiling was reached | **`success=False`** |

Whichever fired is named in `AgentResult.stopped_by`, so "finished" and "gave
out" can never look identical from the outside.

`cost_budget` is the honest one to ask for and the one to caveat: output
length is not known before a call, so it bounds "do not begin another step",
not "never exceed by a cent".

There is no `max_attempts`: it governed retries of a *plan step*, and a loop
has no plan steps — a call that fails comes back as the next turn and the
model decides. Transport-level retries are `Model` options, where a caller
can set them.

---

## Instructions

### `instructions: str | None`

Who the agent is and how it should work — the stable part of the system
prompt. Formerly `persona`.

```python
instructions = ("You are a senior market intelligence analyst. Prefer primary "
                "sources. State what you could not verify.")
```

Whether the loop honours a given instruction is, today, unmeasured — see the
compliance work in the plan. An instruction is not a mechanism: where the
behaviour matters, express it as one.

---

## Governance

### `permissions: PermissionPolicy | None` and `approve: callable | None`

A tool declares a risk class as data; the policy maps it to `allow`,
`approve` or `deny` before the tool runs. `approve` asks `approve=` and runs
only on a yes — **with no approver attached the call is refused.** See
[Observability](observability.md#permission-matrix).

### `hooks: list | None`

Callables receiving an `Event` at each step boundary. The same events are
available as an iterator through `agent.stream(task)`.

---

## Memory

There is no `memory=` parameter, and the agent has no memory subsystem. Its
state **is** the conversation: a message list, serialisable by construction,
which is also why `2.0.0` removed suspend/resume from the agent. `AgentMemory`
survives as a standalone store you can use from your own tools; nothing in the
loop reads or writes it.

---

## Verbosity

### `verbose: int` — default `0`

| Value | Output |
|---|---|
| `0` | Silent. Use for production / inside a Chain. |
| `1` | One status line per step, final summary with token count. |
| `2` | Everything in level 1 plus full action payloads (tool kwargs), output previews, per-call token breakdowns. |

`2` is for debugging — it prints a lot.

---

## Labels

### `name: str | None`

Human-readable identifier. Shown in the header line at `verbose >= 1`, used in
`repr`, carried on every `Event`, and used as the step name when the agent
runs inside a Chain.

### `description: str | None`

Free-text description. Purely informational.

---

## Task and variables

These are the inputs to `agent.run()`, not to the constructor:

```python
result = agent.run(
    task      = "Compare the top 3 managed vector databases.",
    variables = {"audience": "C-level IT decision makers", "horizon": "12 months"},
)
```

### `task: str`

What to accomplish. Be specific about **outputs** — the shape of what you want
— not about the steps. The agent chooses the steps.

Good:

> "Compare the top 3 managed vector databases. For each: name, estimated
> market share, and main differentiator. Return a Markdown table."

Less good:

> "Do some research on vector databases."

### `variables: dict | None`

Data the caller already holds, appended to the opening message under `GIVEN:`
so the model starts with it. This is how a `Chain` or `Pool` step hands its
accumulated values down.

---

## See also

- [Overview](overview.md) — what the loop does
- [Observability](observability.md) — events, permissions, the journal
- [Agent as a Chain step](agent-as-chain-step.md) — composition and persistence
