# Quickstart

Four steps, simplest first — from a single model call to an autonomous agent.
The interface is the same for every provider.

---

## 1. Single model call

Ask any model a question. The provider is detected from the model name; text
output is the default, so a Skill is just a model plus a prompt.

```python
from yait_aichain.models import Model
from yait_aichain.skills import Skill

skill = Skill(
    model = Model("claude-sonnet-4-6"),
    input = {"messages": [{"role": "user", "parts": ["What is {topic}?"]}]},
)

print(skill.run(variables={"topic": "the CAP theorem"}))
```

A `part` can be a plain string (shorthand) or `{"type": "text", "text": "…"}`.

Swap the model name — nothing else changes:

```python
Model("gpt-4o")            # OpenAI
Model("gemini-2.5-flash")  # Google
Model("grok-3")            # xAI
Model("sonar-pro")         # Perplexity (web search built in)
```

---

## 2. Two-step pipeline

Summarise an article, then translate the summary. Step 1's output flows into
step 2 automatically.

```python
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain

model = Model("gpt-4o")

summariser = Skill(
    model = model,
    input = {"messages": [{"role": "user", "parts": ["Summarise in 3 sentences:\n{article}"]}]},
    name  = "summary",
)

translator = Skill(
    model = model,
    input = {"messages": [{"role": "user", "parts": ["Translate into {language}:\n{summary}"]}]},
    name  = "final",
)

pipeline = Chain(steps=[(summariser, "summary"), (translator, "final")])

print(pipeline.run(variables={
    "article":  "Artificial intelligence is reshaping how software is built...",
    "language": "Spanish",
}))
```

The summariser stores its output under `"summary"`; the translator reads it as
`{summary}`. (A bare step with no key stores under `"result"`.)

---

## 3. Web search + summarise

Mix a Tool and a Skill: fetch live search results, then summarise them.

```python
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.tools  import PerplexitySearchTool
from yait_aichain.chain  import Chain

summariser = Skill(
    model = Model("claude-haiku-4-5-20251001"),
    input = {"messages": [{"role": "user", "parts": [
        "Summarise these search results for: {query}\n\n{search_results}"]}]},
)

pipeline = Chain(steps=[
    (PerplexitySearchTool(), "search_results", {"input": "query"}),
    (summariser,             "summary"),
])

print(pipeline.run(variables={"query": "best managed vector databases 2025"}))
```

`{"input": "query"}` tells the tool to read its `input` parameter from the
accumulated variable `"query"`.

---

## 4. Autonomous agent

For tasks that need planning, multiple tool calls, and reasoning about what came
back.

```python
from yait_aichain.models import Model
from yait_aichain.agent  import Agent
from yait_aichain.tools  import PerplexitySearchTool

agent = Agent(
    orchestrator = Model("claude-opus-4-8"),
    tools        = [PerplexitySearchTool()],
    mode         = "agile",   # can replan mid-task
    max_steps    = 10,
    verbose      = 1,         # progress in the terminal
)

result = agent.run(
    "Compare the top 3 managed vector databases: pricing model, hosting, "
    "and standout feature. Return a Markdown table."
)

print(result.output if result else f"Agent failed: {result.error}")
```

---

## Next steps

- **Understand the design** → [Concepts](concepts.md)
- **All Skill options** (JSON output, schemas, save/load) → [Skills](../primitives/skills.md)
- **All Chain options** (error handling, history, input_map) → [Chain](../primitives/chain.md)
- **All Agent options** (modes, memory, persona) → [Agent](../agents/overview.md)
- **Suspend & resume** → [State](../primitives/state.md)
- **Available tools** → [Tools reference](../tools-reference/)
