# Quickstart

Four progressively more complex examples — from a single routed model call to an autonomous research agent. All use the same interface regardless of provider.

---

## 1. Single model call

Ask any model a question. The provider is detected automatically from the model name.

```python
from models import Model
from skills import Skill

skill = Skill(
    model  = Model("claude-sonnet-4-6"),
    input  = {
        "messages": [
            {"role": "system", "parts": [{"type": "text", "text": "Be concise."}]},
            {"role": "user",   "parts": [{"type": "text", "text": "What is {topic}?"}]},
        ]
    },
    output = {"modalities": ["text"], "format": {"type": "text"}},
)

answer = skill.run(variables={"topic": "the CAP theorem"})
print(answer)
```

Swap the model name — nothing else changes:

```python
Model("gpt-4o")           # OpenAI
Model("gemini-2.0-flash") # Google
Model("grok-3")           # xAI
Model("sonar-pro")        # Perplexity (with web search)
```

---

## 2. Two-step pipeline

Summarise an article, then translate the summary. The output of step 1 flows automatically into step 2.

```python
from models import Model
from skills import Skill
from chain  import Chain

model = Model("gpt-4o")

summariser = Skill(
    model  = model,
    input  = {"messages": [
        {"role": "system", "parts": [{"type": "text",
            "text": "Summarise the article in exactly 3 sentences."}]},
        {"role": "user",   "parts": [{"type": "text", "text": "{article}"}]},
    ]},
    output = {"modalities": ["text"], "format": {"type": "text"}},
    name   = "summariser",
)

translator = Skill(
    model  = model,
    input  = {"messages": [
        {"role": "system", "parts": [{"type": "text",
            "text": "Translate the text into {language}."}]},
        {"role": "user",   "parts": [{"type": "text", "text": "{result}"}]},
    ]},
    output = {"modalities": ["text"], "format": {"type": "text"}},
    name   = "translator",
)

pipeline = Chain(steps=[summariser, translator])

output = pipeline.run(variables={
    "article":  "Artificial intelligence is reshaping...",
    "language": "Spanish",
})

print(output)
```

`{result}` in the translator prompt refers to the summariser's output. That name is the default output key. You can use any name:

```python
Chain(steps=[
    (summariser, "summary"),   # stored as accumulated["summary"]
    (translator, "final"),     # reads {summary} from the prompt template
])
```

---

## 3. Web search + summarise

Fetch live search results, then summarise them. Mix a Tool and a Skill in the same pipeline.

```python
from models import Model
from skills import Skill
from tools  import PerplexitySearchTool
from chain  import Chain

search = PerplexitySearchTool()

summariser = Skill(
    model  = Model("claude-haiku-4-5-20251001"),
    input  = {"messages": [
        {"role": "system", "parts": [{"type": "text",
            "text": "You are a research analyst. Summarise the search results below."}]},
        {"role": "user",   "parts": [{"type": "text",
            "text": "Query: {query}\n\nResults:\n{search_results}"}]},
    ]},
    output = {"modalities": ["text"], "format": {"type": "text"}},
)

pipeline = Chain(steps=[
    (search,     "search_results", {"query": "query"}),
    (summariser, "summary"),
])

result = pipeline.run(variables={
    "query": "SAP ERP market in Kazakhstan 2025",
})

print(result)
```

The `input_map` `{"query": "query"}` tells the tool to read its `query` parameter from the accumulated variable called `"query"`.

---

## 4. Autonomous agent

For tasks that require planning, multiple tool calls, and reasoning about intermediate results.

```python
from models import Model
from agent  import Agent
from tools  import PerplexitySearchTool, MarkItDownTool

agent = Agent(
    orchestrator = Model("claude-opus-4-6"),
    tools        = [PerplexitySearchTool(), MarkItDownTool()],
    mode         = "agile",       # can replan mid-task
    max_steps    = 10,
    verbose      = 1,             # shows progress in the terminal
)

result = agent.run(
    task = (
        "Research the top 3 ERP vendors in Kazakhstan. "
        "For each: name, estimated market share, and main differentiator. "
        "Return a structured Markdown table."
    )
)

if result:
    print(result.output)
else:
    print(f"Agent failed: {result.error}")
```

---

## Next steps

- **Understand the design** → [Concepts](concepts.md)
- **All Skill options** (JSON output, schemas, save/load) → [Skills](../primitives/skills.md)
- **All Chain options** (error handling, history, input_map) → [Chain](../primitives/chain.md)
- **All Agent options** (modes, memory, persona) → [Agent](../agents/overview.md)
- **Available tools** → [Tools reference](../tools-reference/)
