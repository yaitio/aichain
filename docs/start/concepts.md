# Concepts

Six ideas, each on one screen. Everything else in the documentation is detail
of one of them.

```text
           your code ─────────────── or ─────────────── your coding agent
               │                                               │
               ▼                                               ▼
   ┌───────────────────────── pipelines ──────────────────────────┐
   │   Skill        one model call                                │
   │   Chain        steps in order:  Skill · tool · Agent · fn    │
   │   Pool         one Skill or Chain over many inputs at once   │
   └──────────────────────────────┬───────────────────────────────┘
                                  │ a step can be
   ┌──────────────────────────────▼───────────────────────────────┐
   │   Agent        a loop: the model calls tools or answers      │
   └──────────────────────────────┬───────────────────────────────┘
                                  │ calls
   ┌──────────────────────────────▼───────────────────────────────┐
   │   tools        search("exa") · @tool functions · kb.query    │
   │   utilities    Vectors · Shell · REST · MCP — code calls the │
   │                methods, an agent gets only the tools         │
   └──────────────────────────────┬───────────────────────────────┘
                                  │ all of it runs on
   ┌──────────────────────────────▼───────────────────────────────┐
   │   Model        a name: "claude-haiku-5-5", "gpt-6-luna", …   │
   └──────────────────────────────────────────────────────────────┘
          and around every run: cost, budget, permissions, events
```

---

## Model — the name is the provider

```python
Model("claude-haiku-5-5")         # Anthropic
Model("gpt-6-luna")               # OpenAI
Model("gemini-3-flash-preview")   # Google
Model("private/llama3.3")         # your own server — no key, nothing leaves it
```

A Model formats one request and reads one response. It knows nothing about
prompts or pipelines, which is why changing the name changes the provider and
nothing else. When the new provider cannot do what the code asked — a
`top_k`, a `seed`, an image size — the result says what was adapted or
dropped. Text and vision are models. Image generation is a tool, because a
model here is what can be an agent's mind, and an image model cannot.

→ [Models](../models/models.md) · [what each provider does with each option](../reference/parameters.md)

---

## Skill — one model call

```python
summarise = Skill(Model("claude-haiku-5-5"), prompt="Summarise in one line: {text}")
summarise.run(variables={"text": "…"})          # a str
```

One model, one prompt template, one output format: `text`, `json`, or
`json_schema` validated against your schema. `run()` returns the value itself.
A Skill saves to YAML and loads back as it was; keys are never written.

→ [Skill](../pipelines/skills.md)

---

## Chain, Pool — pipelines you can list

```python
chain = Chain(steps=[
    (summarise,                 "summary"),
    (translate,                 "french"),
    (count_words,               "length", {"text": "{french}"}),   # a tool, its argument templated
    (shout,                     "loud"),                           # a plain function
])
result = chain.run(variables={"text": "…"})
result["french"], result.output, result.cost
```

A Chain runs steps in order and keeps one set of variables. A Skill reads any
of them in its prompt; a tool or a function gets the ones its parameters
name, or what the step's mapping gives it — a variable's name, or a
`"{template}"` the chain fills in. A dict a step returns is merged into the
variables. A Pool runs one Skill or Chain over many inputs, in parallel, and
total time is the slowest item rather than the sum.

When the steps can be listed, a Chain is cheaper, faster and more predictable
than an Agent that rediscovers them.

→ [Chain](../pipelines/chain.md) · [Pool](../pipelines/pool.md) · [Pause and resume](../pipelines/pause-and-resume.md)

---

## Agent — when the next step depends on the last

```python
agent = Agent(Model("gpt-6-luna"), tools=[search("exa"), fetch("local")],
              stop_when=[step_count(10)])
agent.run("What changed in Python 3.14?").output
```

One loop: each turn the model calls tools or answers. The reply decides —
tool calls continue, prose ends the turn — so the model chooses content, never
control flow. `stop_when` decides how a run ends: a ceiling (`step_count`,
`token_budget`) ends it as a failure, `check(fn)` ends it on a
fact your code verified, and `stalled(k)` / `repeating(k)` speak to the model
and let it go on.

How hard the task is picks the mode. `agile` decides at every turn.
`waterfall` writes a plan first — from a stronger `planner_model` if you like —
and holds to it. `team=` lets it hand work to agents with conversations of
their own. An agent's state is its conversation, a message list, so a run can
park on a `wait()` and `agent.resume()` it in another process.

→ [Agent](../agents/overview.md)

---

## Tools and utilities — what can act

```python
search("exa").run("vector databases")            # a built-in tool, made by service name

@tool(risk="read")
def word_count(text: str) -> int:
    """Count the words in a text.

    text: the text to count.
    """
    return len(text.split())                     # any function, made a tool

sh = Shell("notes")                              # a utility: a folder
Agent(model, tools=[sh.browse, sh.read])         # the agent may read, not write or run
```

A **tool** is anything a chain, an agent and your code can all call, with
`.run()`. What a tool *is* — the service, the model, the folder — is set when
it is made; what one call asks goes in `.run()`. A setting never appears in
both.

A **utility** is an object your code holds: `Vectors`, `Shell`, `REST`, `MCP`.
Your code calls its methods directly — `kb.upsert(records)`. Some of its
members are tools — `kb.query`, `sh.read`, each REST endpoint — and those are
what an agent gets, one by one. An agent handed a utility's method refuses it
by name.

→ [Tools](../tools/index.md) · [Utilities](../utilities/index.md)

---

## Control — what every run reports

Every result carries `usage` and `cost`, priced from the registry.
`max_cost=` is one ceiling in dollars shared by every step of a Skill, Chain,
Pool or Agent. A tool's `risk` class and a `PermissionPolicy` decide whether it
runs, asks a person first, or never runs. Every step emits an event a tracer
or a UI can read, and `eval/` runs the same cases through several models,
prompts or modes and reports accuracy, `Pass^k` and cost side by side.

→ [Cost and budget](../control/cost-and-budget.md) · [Permissions](../control/permissions.md) · [Observability](../control/observability.md) · [Eval](../control/eval.md)

---

## Which one

| The task | Use |
|---|---|
| One model call — summarise, extract, classify, translate | `Skill` |
| Known steps in a fixed order | `Chain` |
| The same work over many inputs | `Pool` |
| The next step depends on what the last one returned | `Agent` with tools |
| A step that is plain code | a function, as a chain step |
| Something a model may call | `@tool`, or a built-in tool by name |
| An outside API, server, folder or vector store | a utility, and its tools to the agent |
| A picture | `image(service, model=…)` |
| A run that waits for a person or a webhook | `wait()` / `gate(tool)` with `store=` |
