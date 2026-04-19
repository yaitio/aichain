# Chain

A **Chain** is an ordered sequence of steps. Each step is a Skill, Tool, or Agent. Every step shares one **accumulated variable dict** — outputs flow forward automatically as named variables. No glue code.

```python
from chain import Chain

chain = Chain(steps=[
    summariser,
    translator,
])

chain.run(variables={"article": "...", "language": "French"})
```

---

## Step syntax

Each element of `steps` can take four shapes, from simplest to most explicit:

### 1. Bare runner

```python
Chain(steps=[summariser, translator])
```

String output is stored under `"result"` (the default key). Fine for a two-step pipeline where you only care about the final output.

### 2. `(runner, output_key)`

```python
Chain(steps=[
    (summariser, "summary"),
    (translator, "final"),
])
```

The summariser's output is stored under `accumulated["summary"]`, so the translator's prompt can reference `{summary}`.

### 3. `(runner, output_key, input_map)`

```python
Chain(steps=[
    (fetch_url,  "raw",     {"source": "url"}),    # Tool: rename accumulated["url"] → source
    (summariser, "summary", {"article": "raw"}),   # Skill: alias raw → article
])
```

`input_map` is a `{dst: src}` dict. Interpretation depends on the runner:

- **Tool** — `dst` is the tool's parameter name; `src` is the accumulated variable to read.
- **Skill** — `dst` is the variable name the template uses; `src` is the accumulated variable to copy into it. Non-destructive: the original accumulated key stays intact.

### 4. `(runner, output_key, input_map, options)`

```python
Chain(steps=[
    (researcher, "brief", {}, {"task_key": "research_task", "output_field": "output"}),
])
```

Currently used only by **Agent** steps. Recognised keys:

| Option | Default | Meaning |
|---|---|---|
| `task_key` | `"task"` | Accumulated variable the agent reads its task from. |
| `output_field` | `"output"` | Which `AgentResult` attribute to write as the step output. |

---

## Variable flow

The accumulated dict is the spine of every pipeline.

```
initial variables: {"article": "...", "language": "French"}
                          │
                    ┌─────▼──────┐
                    │ summariser │  reads {article}
                    │  (Skill)   │  writes → accumulated["summary"]
                    └─────┬──────┘
                          │  accumulated: {article, language, summary}
                    ┌─────▼──────┐
                    │ translator │  reads {summary} and {language}
                    │  (Skill)   │  writes → accumulated["final"]
                    └─────┬──────┘
                          │
                    final output: accumulated["final"]
```

### How each runner sees it

| Runner | What it reads | What it writes |
|---|---|---|
| **Skill** | Full accumulated dict as its variable namespace. Any `{placeholder}` in the prompt is filled from it. | Output stored under `output_key`. |
| **Tool** | Only the kwargs that match its declared `parameters` keys (optionally renamed via `input_map`). Extra variables are silently ignored. | `str` output stored under `output_key`; `dict` output **merged** into the accumulated dict (multiple keys at once). |
| **Agent** | Task string read from `accumulated[task_key]`; the full accumulated dict is passed as `variables`. | The field named by `output_field` (default `"output"`) is stored under `output_key`. |

### After each step

- Output is `str` → `accumulated[output_key] = output`
- Output is `dict` → `accumulated.update(output)`

The dict case is how a single step produces multiple named outputs.

---

## Initial variables

Merge order (later wins):

1. `Chain(variables={...})` — defaults at construction.
2. `chain.run(variables={...})` — per-call overrides.

```python
chain = Chain(steps=[...], variables={"language": "English"})

chain.run()                                  # language = English
chain.run(variables={"language": "Spanish"}) # language = Spanish
```

---

## Accessing intermediate state

```python
result = chain.run(variables={...})

chain.accumulated   # full variable dict after the run — every initial var + every step output
chain.history       # one record per step: step, kind, name, input, output, output_key, options
```

Both are shallow copies; mutating them does not affect the chain's internal state.

`chain.accumulated` is essential for **sectional document generation** — each section writes its content into a distinct key, then a post-run assembler reads them all at once:

```python
chain.run(variables=initial_vars)
document = assemble_document(chain.accumulated, sections)
```

---

## Error handling

By default, any exception raised inside a step propagates and stops the run:

```python
Chain(steps=[...], on_step_error="raise")   # default
```

Two alternatives:

| Mode | Behaviour |
|---|---|
| `"stop"` | Record the error in `history`, return the last successful output (or `None`), do not raise. |
| `"skip"` | Record the error in `history`, issue a `RuntimeWarning`, continue with the next step. Downstream steps may get stale or missing variables — your responsibility. |

```python
chain = Chain(steps=[...], on_step_error="stop")
chain.run(variables={...})          # instance default
chain.run(..., on_step_error="raise")   # override per call
```

---

## Mixing Skills, Tools, and Agents

All three compose freely:

```python
from chain import Chain
from agent import Agent
from skills import Skill
from tools  import MarkItDownTool, PerplexitySearchTool

chain = Chain(steps=[
    (MarkItDownTool(),        "article", {"source": "url"}),        # Tool
    (summariser,              "summary"),                            # Skill
    (Agent(orchestrator=...), "analysis", {}, {"task_key": "summary"}),  # Agent
    (report_skill,            "report"),                             # Skill
])
```

An Agent inside a Chain is just another step — its final `output` becomes the next variable.

---

## Save and load

A Chain serialises to YAML. Every Skill, Tool, and Agent step is recreated from scratch at load time — API keys resolved from the environment.

```python
chain.save("chains/summarise_and_translate.yaml")

from chain import Chain
chain = Chain.load("chains/summarise_and_translate.yaml")
```

What's stored:

- Chain-level: `name`, `description`, `variables`, `on_step_error`
- Per step: `kind`, `output_key`, `input_map`, `options`
- **Skill step** — `model_name`, `input`, `output`, `variables`, `options`, `name`, `description`
- **Tool step** — fully-qualified class path and serialisable `init_args`
- **Agent step** — orchestrator model name, mode, limits, persona, and the class paths of its tools

What is **not** stored: API keys. They're pulled from environment variables at load time (or you can override with `Chain.load(path, api_key=...)`).

Example YAML:

```yaml
name: summarise_and_translate
variables:
  language: French
steps:
  - kind: skill
    output_key: summary
    skill:
      model_name: gpt-4o
      input:
        messages:
          - role: user
            parts:
              - type: text
                text: "Summarise: {article}"
      output: {modalities: [text], format: {type: text}}
  - kind: skill
    output_key: final
    skill:
      model_name: gpt-4o
      input: { ... }
      output: { ... }
  - kind: tool
    output_key: raw_markdown
    input_map: {source: url}
    tool:
      class: tools.markitdown.MarkItDownTool
      init_args: {}
```

---

## Common recipes

### Fan-in: one tool's dict feeds several skills

```python
class FetchEverythingTool(Tool):
    def run(self, url: str) -> dict:
        return {"title": ..., "body": ..., "author": ...}

Chain(steps=[
    (FetchEverythingTool(), "_", {"url": "url"}),   # merges title/body/author into accumulated
    (title_critic,   "title_feedback"),
    (body_summariser,"summary"),
])
```

### Passing previous outputs by a different name

A skill's prompt uses `{current_section_content}` but the chain stored the same value under `body_content`:

```python
(summariser, "summary", {"current_section_content": "body_content"}),
```

### Conditionally skipping a step

Use `on_step_error="skip"` and let the step itself raise on an "I have nothing to do" check.

---

## See also

- **Step runners** → [Skill](skills.md), [Tool](tools.md), [Agent](../agents/overview.md)
- **Agents as Chain steps** → [Agent as Chain step](../agents/agent-as-chain-step.md)
- **Persistence details** → this page's Save-and-load section
