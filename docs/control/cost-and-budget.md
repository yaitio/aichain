# Cost and budget

Every result says what it cost. Every primitive takes a ceiling in dollars.

```python
from yait_aichain import Chain, Model, Skill

model = Model("claude-haiku-5-5")
chain = Chain(steps=[
    (Skill(model, prompt="Summarise: {text}"), "summary"),
    (Skill(model, prompt="Translate into French: {summary}"), "french"),
], max_cost=0.05)

result = chain.run(variables={"text": "…"})
result.cost              # dollars, e.g. 0.000412
result.usage             # input, output, cached and image tokens
```

---

## What a call costs

Prices live in the model registry, one entry per model, read from each
provider's own pricing page with the date it was read. A result's `usage`
counts, separately:

| Field | |
|---|---|
| `input_tokens` | the prompt, fresh |
| `output_tokens` | the reply, thinking tokens included where the provider bills them |
| `cache_read_tokens` | a reused prompt prefix, billed far below fresh input |
| `cache_write_tokens` | a prefix stored for reuse, billed above it |
| `image_input_tokens` | images sent in, where they cost more than text |

`cost` is those counts at the registry's rates: a cache read at the model's own
rate where it has one, the long-context rate past its threshold, an image model
by the picture. A model the registry has no price for reports `cost=None`,
never `0` — an unknown price is not a free call.

What it does not include, said rather than hidden: Perplexity's per-request
search fee, DeepSeek's off-peak discount, and a tool's own price — speech,
search and embeddings are billed by their services, outside these numbers.

→ [Model registry](../reference/model-registry.md)

## A ceiling in dollars

`max_cost=` on a `Skill`, `Chain`, `Pool` or `Agent` refuses to **begin** a
model call once that much is spent. It bounds beginning, not exceeding: the
call in flight when the line is crossed finishes and is counted.

A number is a ceiling **per run**: each `run()` starts a fresh one, so an
agent built once — at module level, in a warm Lambda — gives every request the
whole amount, and a run picked up with `resume()` starts from what it had
already spent before the pause. A `Budget` object, below, is the one that
spends down across runs.

| | Over budget |
|---|---|
| `Skill` | raises `BudgetExceeded` — what was spent and the limit |
| `Chain` | the step that would begin is refused with `BudgetExceeded`, handled by `on_step_error` like any failed step — raised by default |
| `Pool` | each item that would begin fails with the budget's reason, handled by `on_error`; the first item always runs |
| `Agent` | the run ends with `success=False`, `stopped_by="llm_error"`, the reason in `.error` |

## One budget, several runs

```python
from yait_aichain import Budget

budget = Budget(0.50)
Pool(summarise, items=documents, max_cost=budget).run()
Agent(model, tools=[…], max_cost=budget).run("…")
budget.spent, budget.remaining
```

A `Budget` passed to several primitives is shared between them, and safe under
a `Pool`'s parallel items — a budget a race can overspend is not a budget.

## Choosing a model by price

```python
from yait_aichain.models import registry

registry.models("anthropic", "text-to-text")     # the names
```

The registry is data: each model's capabilities and price are in
`models/providers/*.toml`, and the [model registry](../reference/model-registry.md)
page is generated from it.
