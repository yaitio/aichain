# Agent

An Agent is a conversation that can act. It is given a task and tools; each
turn the model either calls tools or answers. Tool results go back into the
conversation, and the loop asks again. Use it when the next step depends on
what the last one returned — when you could not have written the steps down in
advance. When you could, a [Chain](../pipelines/chain.md) does the same work
without paying a model to rediscover the order.

```python
agent = Agent(Model("gpt-6-luna"), tools=[search("exa"), fetch("local")],
              stop_when=[step_count(10)], max_cost=0.50)
result = agent.run("Compare the three largest managed vector databases.")
result.output, result.stopped_by, result.cost
```

→ Every parameter: [API: Agent](../api/agent.md)

---

## One loop, and the reply decides

```text
messages = [instructions, task]
loop:
    reply = the model
    no tool calls in it?   → done; the reply is the answer
    run each call, append the reply and every result
    a stop condition fires? → stop
```

That is the whole of it. Control flow is a pure function of the reply: tool
calls continue, prose ends. The model chooses what to say and what to call,
never whether the loop goes on, which is why a run cannot wander off into a
state the code does not know about.

The conversation only grows. Nothing is rebuilt between turns, so the prefix a
provider can cache grows with it, and every second message is something the
world said rather than the model — which is what makes the loop converge
instead of circling.

## The agent's state is its conversation

There is no memory store beside it, no plan object to keep in step, no
snapshot to save. Everything the agent knows is in the message list: every
attempt, with what came back. A message list is data, so a run can park and
resume in another process, a driver can run it one turn per serverless call,
and a reader can see exactly what the model was told and when.

## Tools, and three it may be given

The agent offers each tool's name, description and schema in the provider's
own `tools` field, and the calls come back typed — not as JSON the model was
asked to write into its text, which weak models get wrong. It checks every
call against the schema before it runs: a missing argument or an unknown key
comes back to the model as a message it can correct, not as a crash. Beside
your tools it may get three of its own:

| Tool | When it is offered | What it does |
|---|---|---|
| `pool` | whenever it has tools | repeats one of them, or a prompt, over a list in parallel — "read every document" in one turn instead of one turn each |
| `write_plan` | `mode="waterfall"` | records the plan the run then holds to |
| `delegate` | `team=` is set | hands a scoped sub-task to a worker with a conversation of its own |

## How hard the task is picks the mode

| | `agile` (default) | `waterfall` | `team=` |
|---|---|---|---|
| Decides | at every turn, from what the last one returned | once, up front, in a written plan | which worker does which part |
| Suits | exploration: step four is unknowable until step three answers | a path that can be designed once — and a stronger `planner_model` can design it | work that splits into independent parts |
| Costs | the cheapest loop | one planning turn, possibly on a dearer model | a conversation per worker |

`planner_model` is where the money goes in `waterfall`: the plan is the
highest-leverage tokens of the run, everything after merely executes it. Think
while planning, act without deliberating.

Which mode wins depends on the model, and the difference is large. In a
τ²-bench study of five harnesses on the same 184 tasks, with the same models
and the same system prompt, a plain loop scored best with open-weight models
(0.560) and this library's `waterfall` scored best with proprietary ones
(0.745), where the plain loop fell to fourth. A harness is a tax on a weak
model and an asset on a strong one; measure on yours before settling.
→ [Eval](../control/eval.md)

`team=[historian, chemist]` delegates only to the members named, each read by
its `name` and `description`; `team={"members": [...], "extend": True, "max": 6}`
also lets the lead add a member when none fits — one role made once and reused,
within `max`; `"max_delegations": 10` bounds the delegations of one run. A
member may be a `Chain` as well as an agent: the lead reads its inputs from its
`input_schema()` and gives them by name, and a chain that does exactly the job
is the deterministic choice over an agent doing it by hand. Members are held to
the lead: its rules, its budget, its hooks. A worker's report comes back as the
model's word, not as a checked fact, and the result says so, with what it cost
and how many steps it took — and a member with nothing the lead lacks that
finished in one step is said to have cost more than doing it would have.

```python
reader = Chain([(fetch("local"), "page"), summarise], name="reader",
               description="Reads a web page and summarises it")
lead = Agent(Model("claude-haiku-5-5"), team={"members": [reader], "max_delegations": 5})
```

## How a run ends

The ordinary end is an answer — a reply with no tool calls. Everything else
that can end a run is in `stop_when`, and there are three kinds, kept apart
because they mean different things:

- **A ceiling** — `step_count(n)`, `token_budget(n)` — ends
  the run as a **failure**. Reaching the limit is not an answer.
- **A check** — `check(fn)` — ends it as a **success** on a fact your code
  verified. It is the only completion the harness can confirm itself;
  everything else is the model's word for it. Its output is what
  `check(fn, output=…)` returns; with `answer=True`, one more reply from the
  model, which costs a call; otherwise the result of the last tool call that
  succeeded, as the tool returned it — a dict stays a dict.
  `result.output_from` says which: `"check"`, `"check_answer"`,
  `"tool_result"`.
- **A nudge** — `stalled(k)`, `repeating(k)` — ends nothing. When the last *k*
  attempts moved nothing, or were the same move, the model is told so in an
  ordinary, logged message, and the loop goes on.

The default is `[step_count(30)]`: a backstop, not a plan. `result.stopped_by`
says which ended the run — `"answered"`, `"check:<name>"`, `"step_count"`, … —
because a run that finished and a run that ran out of budget otherwise look
identical from outside.

The loop is only as good as the model driving it. Asked to find a number
between 1 and 1000 in fifteen probes
([`examples/28_agent_stop.py`](../../examples/28_agent_stop.py)), one model ran a
clean binary search and finished in nine; a smaller one bisected for a while,
then fell into counting up by one and hit the ceiling. Same harness both
times — which is what the stop conditions are for.

## What the result says

`run()` does not raise. It returns an `AgentResult`: `bool(result)` is
`.success`, `.error` says why it failed, `.cost` and `.usage` what it spent,
`.adaptations` what the library changed to fit the provider. `.journal` is
what was attempted, each entry marked `check` when code verified it and
`model_claim` when only the model said so, and `.acceptance` sums that up:
`checked`, `claimed` or `unsupported`. An unsupported answer is still returned
— the point is that you can see it is one.

## Around the loop

- **Permissions** — a tool's `risk` and the agent's `policy` decide, before a
  call runs, whether it runs, asks a person or never runs.
  → [Permissions](../control/permissions.md)
- **Money** — `max_cost=` ends the run before a call that would pass it.
  → [Cost and budget](../control/cost-and-budget.md)
- **Pausing** — a `wait()` or `gate()` among the tools lets the model park the
  run until the world answers; `store=` keeps it.
  → [Pause and resume](../pipelines/pause-and-resume.md)
- **Watching** — every turn, call and approval is an event, and `stream()`
  yields them as they happen. → [Observability](../control/observability.md)
- **Driving it yourself** — `step()` decides without executing, for a harness
  or a serverless function that owns the loop.
  → [Driving the loop](driving-the-loop.md)
- **Inside a pipeline** — an agent is a Chain step, and reads its task from the
  chain's variables. → [An agent as a chain step](agent-as-chain-step.md)

---

▶ [`examples/27_agent.py`](../../examples/27_agent.py) ·
[`examples/28_agent_stop.py`](../../examples/28_agent_stop.py) ·
[`examples/29_agent_stream.py`](../../examples/29_agent_stream.py) ·
[`examples/30_agent_team.py`](../../examples/30_agent_team.py) ·
[`examples/32_agent_pause.py`](../../examples/32_agent_pause.py) ·
[`examples/33_agent_step.py`](../../examples/33_agent_step.py)
