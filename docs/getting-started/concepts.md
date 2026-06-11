# Concepts

aichain is a gateway: one interface in, any AI provider out. The five building blocks below are what make that gateway programmable.

---

## The five building blocks

```
┌─────────────────────────────────────────────────────────┐
│                         Agent                           │
│   plans → acts → reflects → (replans) → final answer   │
│                                                         │
│   uses Tools and Skills at each step                    │
└─────────────────────────────────────────────────────────┘
                          │ uses
┌─────────────────────────────────────────────────────────┐
│                         Chain                           │
│   step 1 → step 2 → step 3 → ... → final output        │
│                                                         │
│   each step is a Skill, Tool, or Agent                  │
│   outputs flow forward as named variables               │
└─────────────────────────────────────────────────────────┘
              │                    │
    ┌─────────────────┐   ┌────────────────┐
    │      Skill      │   │      Tool      │
    │                 │   │                │
    │  Model + prompt │   │  Python func   │
    │  + output spec  │   │  + JSON Schema │
    └─────────────────┘   └────────────────┘
              │
    ┌─────────────────┐
    │      Model      │
    │                 │
    │  provider API   │
    │  auto-detected  │
    └─────────────────┘
```

---

## Model

A **Model** wraps one provider's API. You create one by name — the library detects the provider automatically:

```python
Model("gpt-4o")            # → OpenAIModel
Model("claude-sonnet-4-6") # → AnthropicModel
Model("gemini-2.0-flash")  # → GoogleAIModel
Model("grok-3")            # → XAIModel
Model("sonar-pro")         # → PerplexityModel
```

A Model does exactly two things: format a request and parse a response. It knows nothing about prompts, pipelines, or tools. That separation is intentional — you can swap the model in any Skill or Chain without touching the logic around it.

---

## Skill

A **Skill** is a single reusable task: one Model + one prompt template + one output format.

```python
Skill(model=..., input={...}, output={...})
```

The prompt template uses `{placeholder}` tokens:

```
"Translate {text} into {language}."
```

At runtime, `skill.run(variables={"text": "...", "language": "French"})` fills in the placeholders and calls the model.

Three output formats are supported:

| Format | What you get back |
|---|---|
| `{"type": "text"}` | A plain string |
| `{"type": "json"}` | A Python dict (model returns JSON, Skill parses it) |
| `{"type": "json_schema", ...}` | A validated Python dict matching your schema |

Skills can be saved to YAML and reloaded later — useful for sharing prompts across projects without embedding them in code.

---

## Tool

A **Tool** is a Python function with a declared JSON Schema interface. It does something in the real world: searches the web, fetches a URL, converts a file, calls an external API.

```python
class MyTool(Tool):
    name        = "my_tool"
    description = "Does something useful."
    parameters  = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The input."},
        },
        "required": ["query"],
    }

    def run(self, query: str) -> str:
        return do_something(query)
```

Tools have two call styles:

- `tool.run(query="...")` — returns the raw result, raises on error
- `tool(query="...")` — returns a `ToolResult`; never raises

The `ToolResult` wrapper is what Chain and Agent use internally. The declared `parameters` schema is what Agent uses to decide how to call the tool.

---

## Chain

A **Chain** is an ordered sequence of Steps. Each step is a Skill, Tool, or Agent.

The key design: every step shares a single **accumulated variable dict**. Each step reads variables it needs from that dict and writes its output back into it.

```
initial variables: {"article": "...", "language": "Spanish"}
                          │
                    ┌─────▼──────┐
                    │ summariser │  reads {article}
                    │  (Skill)   │  writes → accumulated["summary"]
                    └─────┬──────┘
                          │  accumulated: {article, language, summary}
                    ┌─────▼──────┐
                    │ translator │  reads {summary} and {language}
                    │  (Skill)   │  writes → accumulated["translation"]
                    └─────┬──────┘
                          │
                    final output: accumulated["translation"]
```

This means you never write glue code to pass data between steps. If a skill's prompt contains `{summary}`, it will automatically receive the value that the previous step stored under that name.

When a Tool or Agent returns a **dict**, all keys are merged into the accumulated dict at once — a single step can produce multiple named outputs.

---

## Agent

An **Agent** solves tasks that require planning, multiple tool calls, and reasoning about intermediate results.

The orchestrator (a Model) drives three phases per step:

1. **Plan** — produces an ordered list of steps to achieve the goal
2. **Act** — for the current step, decides which tool to call or what to ask the model
3. **Reflect** — assesses the result, decides: continue / retry / replan / stop

Two execution modes:

| Mode | Behaviour |
|---|---|
| `waterfall` | Fixed plan. Steps execute in order. Retries on failure, stops if max attempts exceeded. |
| `agile` | Same, but the reflect phase can also trigger replanning. The agent can revise its plan based on what it has learned. |

Agents can be used standalone or as a step inside a Chain (`kind="agent"`).

---

## Which primitive should I use?

| Situation | Use |
|---|---|
| Single model call | **Skill** |
| Fixed sequence of steps with known data flow | **Chain** of Skills and Tools |
| Need to call an external API or service | **Tool** in a Chain |
| Task requires searching, reading, and reasoning across multiple sources | **Agent** |
| Long document that would exceed the model's output limit | **Chain** with sectional pattern |
| Research phase followed by document generation | **Agent** inside a **Chain** |

---

## Variable flow in detail

The accumulated dict is the spine of every pipeline. Understanding it removes 90% of potential confusion.

**Before the first step**, the dict is seeded with the Chain's default variables merged with any variables you pass to `run()`. Call-time values win on conflict.

**After each step:**

- If the step output is a `str` → `accumulated[output_key] = output`
- If the step output is a `dict` → `accumulated.update(output)`

**Skills** receive the full accumulated dict as their variable namespace. Any `{placeholder}` in the prompt is filled from it.

**Tools** receive only the kwargs that match their declared `parameters` keys — extra variables are silently ignored.

**Agents** receive `accumulated[task_key]` as their task string and the full accumulated dict as their variables.

After `chain.run()` completes, `chain.accumulated` holds the final state of the entire dict — all initial variables plus every step's output.

---

## Persistence

Both Skill and Chain can be serialised to YAML and reloaded:

```python
skill.save("skills/translator.yaml")
skill = Skill.load("skills/translator.yaml")

chain.save("chains/summarise_and_translate.yaml")
chain = Chain.load("chains/summarise_and_translate.yaml")
```

API keys are **never** written to YAML. They are resolved from environment variables when the file is loaded.
