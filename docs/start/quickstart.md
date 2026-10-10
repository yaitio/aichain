# Quickstart

Five minutes, four steps: one model call, a pipeline, a tool, an agent. Each
builds on the one before. You need Python 3.10 or later and one provider's key
— this page uses Anthropic's; any other works by changing the model's name.

```bash
pip install yait-aichain
export ANTHROPIC_API_KEY="sk-ant-…"
```

---

## 1. One model call

```python
from yait_aichain import Model, Skill

summarise = Skill(Model("claude-haiku-5-5"), prompt="Summarise in one sentence: {text}")
print(summarise.run(variables={"text": "Large language models predict the next token…"}))
print(summarise.last_usage.cost)            # what it cost, in dollars
```

A Skill is a template: building it calls nothing, `run()` does. Change
`"claude-haiku-5-5"` to `"gpt-6-luna"` and set `OPENAI_API_KEY` — nothing else
changes. → [Skill](../pipelines/skills.md)

## 2. Two steps that pass a value along

```python
from yait_aichain import Chain

translate = Skill(Model("claude-haiku-5-5"), prompt="Translate into {language}: {summary}")

chain = Chain(steps=[(summarise, "summary"), (translate, "translation")])
result = chain.run(variables={"text": "…", "language": "French"})
print(result["summary"])
print(result.output, result.cost)           # the last step, and the whole run's cost
```

The first step's output is stored as `summary`, and the second step's prompt
reads `{summary}`. No glue code. → [Chain](../pipelines/chain.md)

## 3. A tool

```python
from yait_aichain.tools import fetch, tool

@tool(risk="read")
def word_count(text: str) -> int:
    """Count the words in a text.

    text: the text to count.
    """
    return len(text.split())

chain = Chain(steps=[
    (fetch("local"), "page",    {"url": "link"}),       # reads a page on this machine, no key
    (summarise,      "summary", {"text": "page"}),
    (word_count,     "length",  {"text": "summary"}),
])
print(chain.run(variables={"link": "https://example.com"})["length"])
```

`fetch("local")` reads a page by URL; any function becomes a tool with
`@tool`. The third element of a step says where its arguments come from.
→ [Tools](../tools/index.md)

## 4. An agent

When the next step depends on what the last one found, let a model choose:

```python
from yait_aichain import Agent
from yait_aichain.agent import step_count

agent = Agent(Model("claude-haiku-5-5"), tools=[fetch("local"), word_count],
              stop_when=[step_count(8)])
result = agent.run("Read https://example.com and tell me how many words its main text has.")
print(result.output, result.stopped_by, result.cost)
```

Every agent gets a ceiling. `stopped_by` says whether it answered or ran out.
→ [Agent](../agents/overview.md)

---

## Next

- [Concepts](concepts.md) — the six ideas behind all of this, one screen each.
- [Build with your coding agent](../with-your-agent.md) — let Claude Code, Codex
  or Cursor write the next one.
- [`examples/`](../../examples/) — twenty-four programs, each runnable.
