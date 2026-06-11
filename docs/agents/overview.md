# Agent overview

An **Agent** is an autonomous execution engine. You describe a task in natural language; the agent figures out the steps, calls the right tools, reflects on what it got back, and stops when it has the answer.

```python
from models import Model
from agent  import Agent
from tools  import PerplexitySearchTool, MarkItDownTool

agent = Agent(
    orchestrator = Model("claude-opus-4-6"),
    tools        = [PerplexitySearchTool(), MarkItDownTool()],
    mode         = "agile",
    max_steps    = 8,
    verbose      = 1,
)

result = agent.run("Research the top 3 ERP vendors in Kazakhstan and return a Markdown table.")

if result:
    print(result.output)
else:
    print("Failed:", result.error)
```

Use an Agent when the work requires **planning**, **multiple tool calls**, and **judgement about intermediate results** — research tasks, multi-step transformations, anything where the sequence of steps isn't known upfront.

For fixed pipelines with known data flow, use a [Chain](../primitives/chain.md) instead.

---

## The three phases per step

For every step, the **orchestrator model** makes three structured LLM calls:

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│   1. PLAN                                                 │
│      (once, up front)                                     │
│      task + tools + memory → ordered list of steps       │
│                                                           │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│                                                           │
│    for each step in plan:                                 │
│                                                           │
│    2. ACTION                                              │
│       step goal + context → tool call OR skill prompt    │
│                                                           │
│    3. EXECUTE                                             │
│       run the tool / call the skill                       │
│                                                           │
│    4. REFLECT                                             │
│       what happened? → continue | retry | replan | stop  │
│                       | final_answer                      │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

- **Plan** is called once at the start. It returns an ordered list of steps, each tagged `tool` or `skill`.
- **Action** turns the step's natural-language goal into a concrete call: exact tool kwargs, or a system/user prompt for a skill.
- **Reflect** assesses the output and decides what happens next.

Every orchestrator call uses structured JSON. The agent parses it robustly — plain JSON, fenced `` ```json ``, or the first `{...}` block in the response.

---

## Reflection decisions

After every step's execution, reflection picks one of five decisions:

| Decision | Effect |
|---|---|
| `continue` | Success. Move to the next step. |
| `retry` | Something went wrong; retry the same step. Capped by `max_attempts`. |
| `replan` | (agile only) Produce a revised plan and optionally jump back to an earlier step. |
| `stop` | Fatal error. Agent returns `success=False`. |
| `final_answer` | Agent has everything it needs. Return that answer immediately, skipping any remaining steps. |

Reflection also picks a **`store_as`** key — a snake_case variable name where the step's output lands in memory. Downstream steps reference it by that name.

---

## Waterfall vs agile

Two execution modes:

| Mode | Plan can change? | Use when |
|---|---|---|
| `waterfall` | No | The path is predictable (fixed research → synthesis → output). Retries still allowed. |
| `agile` | Yes, via `replan` decisions | The path is exploratory. Early steps may reveal that later steps need to change — agent adapts based on what it learns. |

In agile mode, `replan` returns a new step list and a `goto_step` index. The agent jumps to that step and continues. The same `max_tokens` budget caps overall spend either way.

---

## Early exit: `final_answer`

Two shortcuts let the agent return before running every step:

1. **Action-level** — during action determination, the orchestrator can return `{"type": "final_answer", "answer": "..."}` directly. The step doesn't execute; the answer is returned immediately.
2. **Reflection-level** — after a step executes, reflection can decide `"final_answer"` and provide the final output.

Both count the tokens consumed up to that point and return `success=True`.

---

## Budgets

Three independent limits:

| Limit | Default | What happens when hit |
|---|---|---|
| `max_steps` | `10` | The plan is truncated; the agent never runs more than this many distinct steps. |
| `max_attempts` | `3` | Retries per step are capped. After the cap, the loop moves on (step recorded as failed). |
| `max_tokens` | `50_000` | Total token budget across plan + actions + executions + reflections. Agent stops cleanly when exceeded. |

Token usage is extracted from every raw provider response and accumulated in `result.tokens_used`.

---

## AgentResult

`agent.run()` **never raises**. Check the return value instead:

```python
result = agent.run("…")

if result:                    # bool(result) == result.success
    print(result.output)
else:
    print("Failed:", result.error)
```

| Field | Description |
|---|---|
| `success: bool` | `True` on successful completion. |
| `output: Any` | The final answer (`None` on failure). |
| `mode: str` | `"waterfall"` or `"agile"`. |
| `steps_taken: int` | How many plan steps were executed. |
| `tokens_used: int` | Total tokens across all LLM calls. |
| `plan: list[dict]` | The final plan (may differ from the initial plan in agile mode). |
| `history: list[dict]` | Full per-attempt trace — action, output, reflection, tokens. |
| `memory: dict` | Snapshot of the agent's memory at the end. |
| `error: str \| None` | Failure reason. |

Inspecting the trace:

```python
for rec in result.history:
    print(f"Step {rec['step']+1} attempt {rec['attempt']} [{rec['action_type']}] {rec['step_goal']}")
    print(f"  → assessment: {rec['reflection']['assessment']}")
    print(f"  → stored as : {rec['stored_as']}")
    print(f"  → tokens    : {rec['tokens']:,}")
```

---

## Verbosity

Three levels:

| `verbose` | Output |
|---|---|
| `0` (default) | Silent. |
| `1` | Plan overview, one status line per step, final summary. |
| `2` | Everything in level 1 plus full action payloads (tool kwargs / skill prompts), output previews, per-call token breakdowns, reflection reasoning. |

Example at `verbose=1`:

```
╔══ Agent: research_agent │ mode=agile │ budget=40,000 tokens ══════════════
║  Task   : Research top 3 ERP vendors in Kazakhstan…
║  Tools  : ['perplexity_search', 'markitdown']
║  Execs  : ['claude-opus-4-6']
╚══════════════════════════════════════════════════════════════════════

[Plan] Generating plan…
[Plan] 4 step(s) · +612 tokens
        1.  ⚙  perplexity_search       Search for ERP market share KZ
        2.  ⚙  perplexity_search       Find vendor differentiators
        3.  ◆  claude-opus-4-6         Synthesise findings
        4.  ◆  claude-opus-4-6         Format as Markdown table

[Step 1/4]  Search for ERP market share KZ
  ⚙  perplexity_search  {"query": "ERP vendors market share Kazakhstan 2025"}
  →  SAP holds approx. 38%, 1C ~30%, Oracle ~12%, others…
  ↳  success  ·  →  next step  ·  stored as 'market_share_data'  ·  +890 tokens
…
[Done] ✓ All steps complete · 4 step(s) · 18,420 tokens total
```

---

## When to reach for an Agent

| Situation | Use Agent? |
|---|---|
| Single model call with known prompt | No — use [Skill](../primitives/skills.md). |
| Fixed sequence of Skills/Tools with known data flow | No — use [Chain](../primitives/chain.md). |
| Task that needs searching, reading, cross-referencing, and reasoning across sources | **Yes**. |
| Open-ended exploration where the right next step depends on what you just learned | **Yes, agile mode**. |
| Long document generation with a predictable structure | Consider Chain with [sectional pattern](../primitives/chain.md); use Agent only for the research phase. |

Agents can also be embedded **inside** a Chain — the agent handles the open-ended phase; the Chain handles the rest. See [Agent as Chain step](agent-as-chain-step.md).

---

## See also

- **Constructor parameters, tools, executors, persona** → [Configuration](configuration.md)
- **Shared state across steps; persistent memory across runs** → [Memory](memory.md)
- **Use an Agent as one step in a Chain** → [Agent as Chain step](agent-as-chain-step.md)
