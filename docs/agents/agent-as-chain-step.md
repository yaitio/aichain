# Agent as a Chain step

An Agent can run standalone — `agent.run(task="…")` — or sit inside a Chain as one step among many. Embedding it inside a Chain is the recommended pattern whenever you want the agent's output to flow into deterministic downstream work (formatting, translation, rendering, saving to disk).

Think of the Chain as the spine of the pipeline and the Agent as the **research / open-ended phase** inside it.

```python
Chain
  ├─ step 1   Tool / Skill  (deterministic pre-work)
  ├─ step 2   Agent         ← plans, searches, reflects — unknown path
  ├─ step 3   Skill         (deterministic synthesis)
  └─ step 4   Tool          (export, translate, store)
```

---

## The minimal integration

```python
from yait_aichain.chain import Chain
from yait_aichain.agent import Agent

chain = Chain(steps=[
    my_agent,   # bare Agent — task read from accumulated["task"] by default
])

chain.run(variables={"task": "Compare the top 3 managed vector databases."})
```

The Chain treats the Agent like any other step: it passes the accumulated variable dict in, and the agent's final `output` goes back into the dict.

---

## Step syntax

Agents use the 4-tuple step form, same as everything else, but add an **options** dict for agent-specific settings:

```python
(agent, output_key, input_map, options)
```

| Element | Type | Purpose |
|---|---|---|
| `agent` | `Agent` | The agent instance. |
| `output_key` | `str` | Name under which the agent's output lands in accumulated vars. |
| `input_map` | `dict` | Optional renaming. Rarely needed for agents — the orchestrator handles variable access through memory. |
| `options` | `dict` | Agent-only settings (see below). |

### `options`

| Key | Default | Meaning |
|---|---|---|
| `task_key` | `"task"` | Which accumulated variable holds the agent's task string. The chain reads it as `agent.run(task=accumulated[task_key])`. |
| `output_field` | `"output"` | Which field of `AgentResult` becomes the step's output. Usually `"output"`; sometimes `"memory"`. |

Example:

```python
chain = Chain(steps=[
    (research_agent, "brief", {}, {
        "task_key":     "research_task",
        "output_field": "output",
    }),
])

chain.run(variables={
    "research_task": "Investigate the competitive landscape in segment X.",
    "language":      "English",
})
```

---

## How the Agent sees the Chain

When the Chain invokes an Agent step, it calls:

```python
agent.run(
    task      = accumulated[task_key],   # the agent's task string
    variables = accumulated,             # everything else becomes memory
)
```

So the agent gets:

- A task (natural language) — from one specific accumulated variable.
- Its initial memory — the **entire** accumulated dict. Every other step's output, every initial variable, every tool-produced value is available.

This is the cleanest way to give an Agent the context it needs: just populate the accumulated dict with earlier steps and it's automatically available as the agent's memory.

---

## How the Chain sees the Agent

After the Agent runs, the Chain looks at the `AgentResult` and extracts one field:

```python
output = getattr(agent_result, options["output_field"], None)
```

- `output_field="output"` (default) → the agent's final answer.
- `output_field="memory"` → the entire final memory dict (useful when the agent was asked to produce *multiple* named outputs).

If the Agent fails (`AgentResult.success == False`), the Chain raises a `RuntimeError` with the agent's error message. It's handled by the Chain's `on_step_error` mode (`raise`/`stop`/`skip`) like any other step error.

### Two output patterns

**Single string output** — the common case:

```python
(agent, "research_brief", {}, {}),
```

`accumulated["research_brief"]` now holds the agent's final answer. A downstream Skill can reference it as `{research_brief}`.

**Multiple named outputs via `memory`:**

```python
(agent, "_discarded", {}, {"output_field": "memory"}),
```

The agent's **final memory dict** is returned. Because dict outputs are *merged* into accumulated vars, every key the agent stored (`market_share_data`, `vendor_profiles`, …) becomes its own accumulated variable. The `output_key` in this case is ignored in practice, because the output is a dict and gets merged.

This is how you let one agent phase contribute several named outputs to downstream Skill prompts.

---

## Canonical pattern: research phase + report phase

The most common reason to embed an Agent: open-ended research followed by deterministic document generation.

```python
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.agent  import Agent, step_count, token_budget
from yait_aichain.chain  import Chain
from yait_aichain.tools  import PerplexitySearchTool, WeasyprintTool

# Phase 1 — Agent: gather everything we need
research_agent = Agent(
    model        = Model("claude-opus-4-6"),
    tools        = [PerplexitySearchTool()],
    mode         = "agile",
    stop_when    = [step_count(12), token_budget(80_000)],
    instructions = "You are a senior market intelligence director…",
)

# Phase 2 — Skill: turn the research brief into a structured report
report_skill = Skill(
    model = Model("claude-opus-4-6"),
    input = {"messages": [
        {"role": "system", "parts": [{"type": "text",
            "text": "You are a senior analyst. Write the full report."}]},
        {"role": "user",   "parts": [{"type": "text",
            "text": "Research brief:\n{research_brief}\n\nTask: {report_task}"}]},
    ]},
    output = {"modalities": ["text"], "format": {"type": "text"}},
)

# Phase 3 — Tool: export to PDF
pdf_tool = WeasyprintTool()

chain = Chain(steps=[
    (research_agent, "research_brief", {}, {"task_key": "research_task"}),
    (report_skill,   "report_markdown"),
    (pdf_tool,       "pdf_path", {"html": "report_markdown"}),
])

chain.run(variables={
    "research_task": "Compare the top 3 managed vector databases; cite sources.",
    "report_task":   "Deliver a 5-page market briefing in English.",
})
```

What flows where:

```
accumulated at start:
  {research_task: "...", report_task: "..."}

  ↓ Agent step
      task     = accumulated["research_task"]
      memory   = (entire accumulated dict)
      returns  = final answer string

accumulated after Agent:
  {research_task, report_task, research_brief}

  ↓ Skill step
      sees {research_brief} and {report_task} in its prompt
      returns the full report text

accumulated after Skill:
  {research_task, report_task, research_brief, report_markdown}

  ↓ Tool step (input_map renames report_markdown → html)
      returns the output PDF path
```

---

## Verbose inside a Chain

Setting `verbose=1` on the Agent keeps the agent's progress log visible even when run from inside a Chain. Useful for the research phase of long pipelines, where you want to monitor what the agent is doing without adding any other observability.

```python
research_agent = Agent(
    model        = Model("claude-opus-4-6"),
    tools        = [...],
    verbose      = 1,   # agent prints its own progress; chain stays silent
)
```

Set `verbose=0` in production — `result.journal` and the event stream (`hooks=`) are what post-hoc inspection reads.

---

## Persistence inside Chain.save()

When you `chain.save(...)`, Agent steps are serialised too:

- the `model` name
- `mode`, `verbose`
- `instructions`, `name`, `description`
- `stop_when` — the conditions that can be named (`step_count`, `token_budget`,
  …). A condition built from your own closure cannot be written down; it is
  dropped **with a `RuntimeWarning`** at save time rather than coming back
  silently absent
- tool **class paths** (no API keys, no state)

At load time the tools are instantiated from their class paths and the model is
rebuilt with `Model(name, api_key=...)`. Keys come from the environment unless
`Chain.load(path, api_key=...)` overrides them.

Not serialised — set them on the loaded chain's agent step before `run()`:

- `permissions`, `approve`, `hooks` — they are code, not configuration
- `team`, `planner_model`, `max_cost`
- `check(fn)` conditions and anything else that wraps a callable

---

## When **not** to use an Agent inside a Chain

If the sequence of work is fully deterministic — e.g. "fetch URL → clean → summarise → translate" — use a plain Chain of Skills and Tools. Agents add turns and token cost that isn't warranted when the path is knowable upfront.

Rule of thumb: the moment you catch yourself writing `if/else` logic to decide what the next step should be, that is the decision an Agent's loop makes for you.

---

## See also

- **The loop and its stop conditions** → [Overview](overview.md)
- **Model, tools, stop conditions, permissions** → [Configuration](configuration.md)
- **Seeding memory and persisting it across runs** → [Memory](memory.md)
- **Full Chain step syntax** → [Chain](../primitives/chain.md)
