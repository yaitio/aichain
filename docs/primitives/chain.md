# Chain

A **Chain** is an ordered sequence of steps — Skills, Tools, and Agents. All
steps share one **accumulated variable dict**: each step's output flows forward
as a named variable the next step can read. No glue code.

---

## Quick start

```python
import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain

summarise = Skill(model=Model("gpt-4o"), name="summary",
                  input={"messages": [{"role": "user", "parts": ["Summarise: {article}"]}]})
translate = Skill(model=Model("claude-sonnet-4-6"), name="final",
                  input={"messages": [{"role": "user", "parts": ["Translate to {language}: {summary}"]}]})

chain = Chain(steps=[(summarise, "summary"), (translate, "final")])
result = chain.run(variables={"article": "...", "language": "French"})
print(chain.history)   # full per-step audit trail
```

The summariser writes `summary`; the translator reads `{summary}` automatically.

▶ Runnable: [`examples/08_chain.py`](../../examples/08_chain.py) ·
Tool + Skill: [`examples/09_chain_tool_skill.py`](../../examples/09_chain_tool_skill.py) ·
Deep dive ↓

---

## Common gotchas

- **Outputs are named, not positional.** `(skill, "summary")` stores the result
  under `summary`; later steps reference `{summary}`. A bare step stores under
  `"result"`.
- **A Tool reads only its declared parameters** from the accumulated dict; a
  Skill reads the whole dict as its variable namespace.
- **A `dict` step output is merged** into the accumulated dict (many keys at
  once); a `str` output is stored under the one `output_key`.
- **A chain can pause.** If a step suspends (a `Wait`/`Gate` tool), `run()`
  returns a `SuspendedResult` instead of the final value — resume it later with
  `chain.resume(run_id, signal)`. See [Suspend & resume](#suspend--resume).

---

## Reference

### Step syntax

Each element of `steps` takes one of four shapes, simplest to most explicit:

**1. Bare runner** — output stored under `"result"`:

```python
Chain(steps=[summariser, translator])
```

**2. `(runner, output_key)`** — name the output so later steps can read it:

```python
Chain(steps=[(summariser, "summary"), (translator, "final")])
```

**3. `(runner, output_key, input_map)`** — remap accumulated variables into the
step. `input_map` is `{dst: src}`:

```python
Chain(steps=[
    (fetch_url,  "raw",     {"source": "url"}),    # Tool: accumulated["url"] → param "source"
    (summariser, "summary", {"article": "raw"}),   # Skill: alias raw → {article}
])
```

- **Tool** — `dst` is the tool's parameter name; `src` is the accumulated key to read.
- **Skill** — `dst` is the template variable name; `src` is the accumulated key to copy in (non-destructive).

**4. `(runner, output_key, input_map, options)`** — currently for **Agent** steps:

| Option | Default | Meaning |
|---|---|---|
| `task_key` | `"task"` | Accumulated variable the agent reads its task from. |
| `output_field` | `"output"` | Which `AgentResult` field to store as the step output. |

### Variable flow

The accumulated dict is the spine of the pipeline:

```
initial: {"article": "...", "language": "French"}
              │
        ┌─────▼──────┐
        │ summariser │  reads {article} → writes accumulated["summary"]
        └─────┬──────┘
              │  {article, language, summary}
        ┌─────▼──────┐
        │ translator │  reads {summary}, {language} → writes accumulated["final"]
        └─────┬──────┘
        final output: accumulated["final"]
```

| Runner | Reads | Writes |
|---|---|---|
| **Skill** | The whole accumulated dict as its variable namespace. | Output under `output_key`. |
| **Tool** | Only kwargs matching its `parameters` (optionally renamed via `input_map`); extras ignored. | `str` → `output_key`; `dict` → **merged** into accumulated (many keys). |
| **Agent** | Task from `accumulated[task_key]`; full dict passed as variables. | The `output_field` attribute under `output_key`. |

After each step: `str` output → `accumulated[output_key] = output`; `dict`
output → `accumulated.update(output)`.

### Initial variables

Merge order, later wins: `Chain(variables=...)` (defaults) then
`chain.run(variables=...)` (per-call).

```python
chain = Chain(steps=[...], variables={"language": "English"})
chain.run()                                   # English
chain.run(variables={"language": "Spanish"})  # Spanish
```

### Inspecting a run

```python
result = chain.run(variables={...})
chain.accumulated   # full variable dict after the run (initial + every step output)
chain.history       # one record per step: step, kind, name, input, output, output_key, options
```

Both are shallow copies — mutating them doesn't affect the chain. `accumulated`
is how you fan several step outputs into a final assembler (each step writes a
distinct key, then you read them all at once).

### Error handling

By default a step exception propagates and stops the run. Two alternatives:

| Mode | Behaviour |
|---|---|
| `"raise"` (default) | Propagate the exception. |
| `"stop"` | Record the error in `history`, return the last successful output (or `None`), don't raise. |
| `"skip"` | Record the error, emit a `RuntimeWarning`, continue. Downstream steps may see missing variables — your responsibility. |

```python
chain = Chain(steps=[...], on_step_error="stop")
chain.run(..., on_step_error="raise")   # per-call override
```

### Suspend & resume

A chain can **pause** mid-pipeline until an external signal arrives — a human
approval, a webhook, a cron tick — and **resume** later, even in a different
process. This is the durable-run mechanism (see [State](state.md) for the full
model).

You pause by putting a `Wait` or `Gate` tool in the chain, and you make the run
survivable by giving the chain a persistent `store`:

```python
from yait_aichain.chain import Chain
from yait_aichain.tools import Wait
from yait_aichain.state import FileStore, SuspendedResult

chain = Chain(
    steps = [(draft, "draft"),
             Wait(reason="A human must approve the draft", resume_with={"reply": "str"}),
             (send, "confirmation")],
    store = FileStore("runs/"),     # survives a restart; omit for in-memory
)

result = chain.run(variables={"complaint": "..."})
if isinstance(result, SuspendedResult):
    # ...later, possibly another process sharing the same store...
    final = chain.resume(result.run_id, signal={"reply": "approved text"})
```

| Method / arg | Purpose |
|---|---|
| `Chain(store=...)` | Where suspended runs are parked. Default `InMemoryStore` (process-local); `FileStore(dir)` survives restart; subclass `StateStore` for S3/DB. |
| `chain.run(..., context=RunContext(...))` | Per-request tenant + metadata threaded through the run (not secrets). |
| `result` from `run()` | The final value, **or** a falsy `SuspendedResult` (`run_id`, `awaiting`) if a step paused. |
| `chain.resume(run_id, signal)` | Continue from the paused step; completed steps are **not** re-run. Idempotent — a duplicate resume of a finished run raises `KeyError`. |

### Mixing Skills, Tools, and Agents

All three compose as steps:

```python
from yait_aichain.chain  import Chain
from yait_aichain.agent  import Agent
from yait_aichain.tools  import MarkItDownTool

chain = Chain(steps=[
    (MarkItDownTool(),        "article", {"source": "url"}),             # Tool
    (summariser,              "summary"),                                 # Skill
    (Agent(orchestrator=...), "analysis", {}, {"task_key": "summary"}),   # Agent
    (report_skill,            "report"),                                  # Skill
])
```

An Agent inside a Chain is just another step — its final `output` becomes the
next variable.

### Save and load

A Chain serialises to YAML; every step is recreated at load time with keys
resolved from the environment (never written to disk).

```python
chain.save("chains/pipeline.yaml")
loaded = Chain.load("chains/pipeline.yaml")
# loaded = Chain.load("chains/pipeline.yaml", api_key="sk-...")
```

Stored: chain-level `name`/`description`/`variables`/`on_step_error`; per step
`kind`/`output_key`/`input_map`/`options`; and each runner's own definition
(Skill → model name + template; Tool → class path + init args; Agent →
orchestrator + tools). **Not** stored: API keys.

---

## See also

- [Skill](skills.md), [Tool](tools.md), [Agent](../agents/overview.md) — the step runners.
- [State](state.md) — suspend/resume, stores, `Wait`/`Gate`, `RunContext`.
- [Agent as a Chain step](../agents/agent-as-chain-step.md).
