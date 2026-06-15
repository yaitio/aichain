# Agent

An **Agent** is an autonomous execution engine. You describe a task in natural
language; the agent plans the steps, calls the right tools, reflects on each
result, and stops when it has the answer. Use it when the *sequence* of steps
isn't known upfront — research, multi-source reasoning, exploratory work. For a
fixed pipeline, use a [Chain](../primitives/chain.md).

---

## Quick start

```python
import os
from yait_aichain.models import Model
from yait_aichain.agent  import Agent
from yait_aichain.tools  import PerplexitySearchTool

agent = Agent(
    orchestrator = Model("claude-opus-4-8", api_key=os.getenv("ANTHROPIC_API_KEY")),
    tools        = [PerplexitySearchTool()],
    max_steps    = 8,
)

result = agent.run("Compare the top 3 managed vector databases and return a Markdown table.")
if result:
    print(result.output, "·", result.steps_taken, "steps ·", result.tokens_used, "tokens")
else:
    print("Failed:", result.error)
```

▶ One tool: [`examples/13_agent.py`](../../examples/13_agent.py) ·
Multiple tools: [`examples/14_agent_tools.py`](../../examples/14_agent_tools.py) ·
Sub-agents: [`examples/15_agent_orchestrator.py`](../../examples/15_agent_orchestrator.py) ·
Deep dive ↓

---

## Common gotchas

- **`run()` never raises.** It returns an `AgentResult`; check `result.success`
  (and `bool(result)` *is* `success`). On failure, read `result.error`.
- **It can also pause.** With a `Wait`/`Gate` tool, `run()` returns a
  `SuspendedResult` instead — resume with `agent.resume(run_id, signal)`. See
  [Suspend & resume](#suspend--resume).
- **Budgets stop runaway loops.** `max_steps`, `max_attempts`, and `max_tokens`
  each cap a different axis; the agent stops cleanly when any is hit.
- **`agile` mode can replan.** Use it for exploratory tasks; `waterfall` (the
  default) keeps the original plan and only retries.

---

## Reference

### Constructor

```python
Agent(orchestrator, tools=None, executors=None, mode="waterfall",
      max_steps=10, max_attempts=3, max_tokens=50_000, memory=None,
      verbose=0, name=None, description=None, persona=None,
      allow_spawn=False, store=None)
```

| Parameter | Default | Description |
|---|---|---|
| `orchestrator` | — | The `Model` that plans, decides actions, and reflects. |
| `tools` | `None` | Tools the agent may call. |
| `executors` | `[orchestrator]` | Models available to run "skill"-type steps (the orchestrator picks one). |
| `mode` | `"waterfall"` | `"waterfall"` or `"agile"` (see [Modes](#modes)). |
| `max_steps` | `10` | Hard cap on executed plan steps. |
| `max_attempts` | `3` | Retries per step. |
| `max_tokens` | `50_000` | Total token budget across all LLM calls. |
| `memory` | `AgentMemory()` | Working memory; pass a persistent one to carry state across runs ([Memory](memory.md)). |
| `persona` | `None` | System persona steering tone/role. |
| `allow_spawn` | `False` | Let the agent spawn sub-agents via a `spawn_agent` tool. |
| `store` | `InMemoryStore()` | Where suspended runs are parked ([Suspend & resume](#suspend--resume)). |
| `verbose` | `0` | `0` silent · `1` per-step status · `2` full payloads + token breakdowns. |

`agent.run(task, variables=None)` executes; `agent.resume(run_id, signal=None)`
continues a suspended run. Both return an `AgentResult` (or a `SuspendedResult`
on pause).

### The loop: plan → act → reflect

The orchestrator plans **once**, then for each step makes structured calls:

```
1. PLAN     (once)   task + tools + memory → ordered steps, each tagged tool|skill
   for each step:
2. ACTION            step goal + context → concrete tool kwargs OR a skill prompt
3. EXECUTE           run the tool / call the skill
4. REFLECT           assess → continue | retry | replan | stop | final_answer
```

Every orchestrator call is structured JSON; the agent parses it robustly (plain
JSON, fenced ```` ```json ````, or the first `{...}` block).

### Reflection decisions

| Decision | Effect |
|---|---|
| `continue` | Success — move to the next step. |
| `retry` | Retry the same step (capped by `max_attempts`). |
| `replan` | *(agile only)* Produce a revised plan; optionally jump to an earlier step. |
| `stop` | Fatal — return `success=False`. |
| `final_answer` | Done — return the answer now, skipping remaining steps. |

Reflection also assigns a `store_as` key: a snake_case name where the step
output lands in memory for later steps to reference.

### Modes

| Mode | Plan can change? | Use when |
|---|---|---|
| `waterfall` (default) | No (retries only) | The path is predictable. |
| `agile` | Yes, via `replan` | The path is exploratory; later steps depend on what early ones reveal. |

### Budgets

| Limit | Default | When hit |
|---|---|---|
| `max_steps` | `10` | The plan is truncated; never more than this many steps. |
| `max_attempts` | `3` | Retries per step are capped, then the loop moves on. |
| `max_tokens` | `50_000` | Total across plan + actions + executions + reflections; the agent stops cleanly. |

### `AgentResult`

`run()` never raises — inspect the result:

| Field | Description |
|---|---|
| `success: bool` | `True` on completion; `bool(result)` mirrors it. |
| `output` | The final answer (`None` on failure). |
| `steps_taken: int` | Plan steps executed. |
| `tokens_used: int` | Total tokens across all LLM calls. |
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

An agent run can pause for an external signal and continue later — even in
another process. Give the agent a `Wait`/`Gate` tool and (for cross-process) a
persistent `store`:

```python
from yait_aichain.tools import Gate
from yait_aichain.state import FileStore, SuspendedResult

agent = Agent(
    orchestrator = Model("gpt-4o"),
    tools        = [Gate(IssueRefund(), reason="Manager must approve",
                         resume_with={"approved": "bool"})],
    store        = FileStore("runs/"),
)

res = agent.run("Issue a refund for order #123.")
if isinstance(res, SuspendedResult):
    # ...later, a webhook delivers the decision (possibly another process)...
    final = agent.resume(res.run_id, signal={"approved": True})
```

`resume()` restores memory, the plan, and the cursor, then runs to completion.
See [State](../primitives/state.md) for stores, `Wait`/`Gate`, and the
cross-process pattern.

### Sub-agents

With `allow_spawn=True`, the orchestrator gets a `spawn_agent` tool and can
delegate a sub-task to a fresh agent (its own tools/model), then use the result.
Good for fan-out research — one sub-agent per topic.

### When to reach for an Agent

| Situation | Use |
|---|---|
| One model call, known prompt | [Skill](../primitives/skills.md) |
| Fixed sequence with known data flow | [Chain](../primitives/chain.md) |
| Search → read → cross-reference → reason | **Agent** |
| Exploratory; next step depends on what you learned | **Agent (agile)** |

An agent can also be one step *inside* a Chain — it handles the open-ended
phase, the Chain handles the rest. See [Agent as a Chain step](agent-as-chain-step.md).

---

## See also

- [Configuration](configuration.md) — tools, executors, persona, modes in depth.
- [Memory](memory.md) — shared step state and persistent cross-run memory.
- [State](../primitives/state.md) — suspend/resume, stores, `Wait`/`Gate`.
- [Agent as a Chain step](agent-as-chain-step.md).
