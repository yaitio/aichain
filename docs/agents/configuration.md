# Agent configuration

Every knob that shapes how an Agent plans, acts, and spends its budget.

```python
from models import Model
from agent  import Agent
from tools  import PerplexitySearchTool, MarkItDownTool

agent = Agent(
    orchestrator = Model("claude-opus-4-6"),
    executors    = [Model("gpt-4o"), Model("gemini-2.5-flash")],
    tools        = [PerplexitySearchTool(), MarkItDownTool()],
    mode         = "agile",
    max_steps    = 12,
    max_attempts = 3,
    max_tokens   = 80_000,
    persona      = "You are a senior market intelligence analyst…",
    verbose      = 1,
    name         = "market_research_agent",
)
```

---

## Required

### `orchestrator` — the brain

The model that does planning, action determination, and reflection. **Always use a capable reasoning model** — the orchestrator makes every decision about what happens next.

```python
orchestrator = Model("claude-opus-4-6")   # recommended
orchestrator = Model("gpt-4o")
orchestrator = Model("o3")
orchestrator = Model("gemini-2.5-pro")
```

A weak orchestrator produces weak plans and poor decisions. This is the most important choice you make.

---

## Tools and executors

### `tools: list[Tool] | None`

Tools the agent is allowed to call. The orchestrator reads each tool's `name`, `description`, and `parameters` schema to decide how and when to use it.

```python
tools = [PerplexitySearchTool(), MarkItDownTool(), WeasyPrintTool()]
```

Keep the list focused. Giving the agent 15 similar tools is usually worse than giving it 3 well-differentiated ones — it spends planning tokens deciding between near-duplicates.

### `executors: list[Model] | None`

Models available for LLM **skill** steps — the generation/reasoning work that isn't tool-based.

```python
agent = Agent(
    orchestrator = Model("claude-opus-4-6"),    # strong planner
    executors    = [Model("gpt-4o-mini")],       # cheap executor
)
```

When `executors` is omitted, the orchestrator handles both roles. Splitting them is a real cost lever: the orchestrator only runs plan/action/reflect calls (small, structured); executors run the heavy generation work. Put a strong reasoner on one side and a cheap, fast model on the other.

The orchestrator can ask for a specific executor by name in its action response — if multiple executors are provided, it picks the one best suited to the step.

---

## Mode and limits

### `mode: "waterfall" | "agile"` — default `"waterfall"`

See [Overview: waterfall vs agile](overview.md#waterfall-vs-agile).

- `"waterfall"` — plan is fixed; reflection can `continue`/`retry`/`stop`/`final_answer`.
- `"agile"` — reflection can also `replan`, producing a revised step list and optionally jumping back.

### `max_steps: int` — default `10`

Upper bound on how many distinct plan steps the agent will ever run. If the plan returned by the orchestrator is longer, it is truncated. If replanning produces more steps, the new plan is also truncated.

### `max_attempts: int` — default `3`

Retries **per step**. After the cap, the step is recorded as failed and the loop advances (in waterfall) or the orchestrator decides what to do (in agile).

### `max_tokens: int` — default `50_000`

Total token budget across **all** LLM calls — planning, every action determination, every skill execution, every reflection. When exceeded, the agent stops cleanly with whatever it has. Tokens are extracted from every raw provider response; OpenAI, Anthropic, Google, xAI, and Perplexity are all supported.

Rough budgeting:

- Short research task (3–5 steps): 15 000 – 30 000 tokens.
- Deep research (8–12 steps): 40 000 – 80 000 tokens.
- Long-document research phase: 80 000 – 150 000 tokens.

---

## Persona

### `persona: str | None`

Identity / domain context prepended to **every** orchestrator system prompt (planning, action, and reflection).

```python
agent = Agent(
    orchestrator = Model("gpt-4o"),
    persona      = (
        "You are a senior financial analyst specialising in tech equities. "
        "Always cite data sources and flag information older than 30 days. "
        "Prefer primary sources over aggregated news."
    ),
    tools        = [...],
)
```

A persona shapes how the agent plans and reflects, not just how it writes. A "market intelligence director" persona produces different search queries and different next-step decisions than a "general research assistant" persona.

Keep it declarative — describe who the agent is and how it thinks, not step-by-step instructions (that's what `task` is for).

---

## Memory

### `memory: AgentMemory | None`

Custom memory instance. Omit for a fresh in-process memory per `run()` call.

```python
from agent import AgentMemory, FileBackend

memory = AgentMemory(backend=FileBackend("~/.my_agent.json"))
agent  = Agent(..., memory=memory)
```

See [Memory](memory.md) for pre-populated memory, persistent memory across runs, and custom backends.

---

## Verbosity

### `verbose: int` — default `0`

| Value | Output |
|---|---|
| `0` | Silent. Use for production / inside a Chain. |
| `1` | Plan overview, one status line per step, final summary with token count. |
| `2` | Everything in level 1 plus full action payloads (tool kwargs, skill prompts), output previews, per-call token breakdowns, and reflection reasoning. |

`2` is for debugging — it prints a lot.

---

## Labels

### `name: str | None`

Human-readable identifier. Shown in the header line at `verbose >= 1`, used in `repr`, and used as the step name when the agent runs inside a Chain.

### `description: str | None`

Free-text description. Purely informational.

---

## Task and initial variables

These are the only inputs to `agent.run()`, not the constructor:

```python
result = agent.run(
    task      = "Research the top 3 ERP vendors in Kazakhstan.",
    variables = {
        "language":  "English",
        "audience":  "C-level IT decision makers",
        "horizon":   "12 months",
    },
)
```

### `task: str`

The natural-language description of what to accomplish. Be specific about **outputs** — the shape of what you want — not the steps to get there. The agent plans the steps.

Good:

> "Research the top 3 ERP vendors in Kazakhstan. For each: name, estimated market share, and main differentiator. Return a Markdown table."

Less good:

> "Do some research on ERP in Kazakhstan."

### `variables: dict | None`

Key/value pairs seeded into memory **before** the first step. The orchestrator sees them in every prompt; any action (tool kwargs or skill prompts) can reference them via `{placeholder}`.

---

## Full configuration example

A Phase-1 research agent — strong orchestrator, cheap executor, single search tool, agile mode with a healthy token budget, verbose progress for monitoring:

```python
from models import Model
from agent  import Agent
from tools  import PerplexitySearchTool

agent = Agent(
    orchestrator = Model("claude-opus-4-6",
                          options={"reasoning": "medium"}),
    executors    = [Model("claude-sonnet-4-6")],
    tools        = [PerplexitySearchTool()],

    mode         = "agile",
    max_steps    = 14,
    max_attempts = 3,
    max_tokens   = 90_000,

    persona      = (
        "You are a senior market intelligence director. "
        "Prefer Perplexity for live, citable facts. "
        "Search in the local language when the target geography is non-English. "
        "Produce a structured research brief, not a polished report — "
        "the report will be written downstream."
    ),

    verbose      = 1,
    name         = "phase_1_research",
)
```

The Chain that consumes its output is shown in [Agent as Chain step](agent-as-chain-step.md).

---

## See also

- **The three phases & execution flow** → [Overview](overview.md)
- **Shared state & persistence** → [Memory](memory.md)
- **Embedding an Agent in a Chain** → [Agent as Chain step](agent-as-chain-step.md)
