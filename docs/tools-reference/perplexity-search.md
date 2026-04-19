# `perplexity_search` — `PerplexitySearchTool`

Web search powered by the **Perplexity Search API**. Returns rich text snippets extracted directly from page content — the best-in-class choice when an agent needs substantive excerpts without a separate URL-fetch step.

```python
from tools import PerplexitySearchTool

tool   = PerplexitySearchTool()
result = tool(query="nuclear fusion breakthroughs 2025", max_results=5)

print(result.output)
```

---

## Requirements

| | |
|---|---|
| Env var | `PERPLEXITY_API_KEY` |
| Get a key | <https://www.perplexity.ai/settings/api> |
| Endpoint | `POST https://api.perplexity.ai/search` |

At construction time, the tool raises `ValueError` if no key is available — either pass `api_key=` or set `PERPLEXITY_API_KEY` in the environment.

---

## Constructor

```python
PerplexitySearchTool(api_key: str | None = None)
```

- `api_key` — Perplexity key (`pplx-…`). Falls back to `PERPLEXITY_API_KEY`.

---

## Parameters

| Name | Type | Required | Default | Notes |
|---|---|---|---|---|
| `query` | `string` | ✓ | — | The search query. |
| `max_results` | `integer` | | `10` | 1–20. |
| `search_recency_filter` | `string` | | — | One of `hour`, `day`, `week`, `month`, `year`. |
| `search_domain_filter` | `list[string]` | | — | Restrict to specific domains (up to 20). |
| `country` | `string` | | — | ISO 3166-1 alpha-2 (e.g. `US`, `GB`, `DE`). |
| `search_after_date_filter` | `string` | | — | `MM/DD/YYYY`. |
| `search_before_date_filter` | `string` | | — | `MM/DD/YYYY`. |
| `search_language_filter` | `list[string]` | | — | ISO 639-1 codes (e.g. `["en", "fr"]`). |

---

## Output shape

A human-readable plain-text string:

```
Search results for "nuclear fusion breakthroughs 2025" (5 results):

[1] Page Title
    URL: https://example.com/page
    Date: 2025-11-03
    Snippet: Detailed excerpt from the page content…

[2] Another Title
    URL: https://another.com/article
    Snippet: …
```

Plain text is deliberate — it keeps the output visible in an agent's memory preview and avoids JSON parsing by downstream skills.

---

## Usage

### Direct call (errors wrapped in `ToolResult`)

```python
tool   = PerplexitySearchTool()
result = tool(
    query                  = "GPT-5 benchmark results",
    max_results            = 8,
    search_recency_filter  = "month",
    search_domain_filter   = ["openai.com", "arxiv.org"],
)

if result:
    print(result.output)
else:
    print("Error:", result.error)
```

### Run-style (raises on error)

```python
text = tool.run(
    query       = "ERP vendors market share Kazakhstan 2025",
    max_results = 10,
    country     = "KZ",
)
```

### In a Chain

```python
from chain import Chain
from skills import Skill
from tools import PerplexitySearchTool
from models import Model

query_skill = Skill(
    model  = Model("gpt-4o-mini"),
    input  = {"messages": [{"role": "user", "parts": [
        {"type": "text", "text": "Formulate a precise search query for: {topic}"}
    ]}]},
    output = {"modalities": ["text"], "format": {"type": "text"}},
)

chain = Chain(steps=[
    (query_skill,              "search_query"),
    (PerplexitySearchTool(),   "search_results", {"query": "search_query"}),
    (analysis_skill,           "report"),
])

chain.run(variables={"topic": "AI regulation in the EU (2025)"})
```

### In an Agent

```python
from agent import Agent
from models import Model
from tools import PerplexitySearchTool

agent = Agent(
    orchestrator = Model("claude-opus-4-6"),
    tools        = [PerplexitySearchTool()],
    mode         = "agile",
    max_steps    = 10,
    persona      = "You are a research analyst. Prefer primary sources.",
)

result = agent.run("Find and compare the top 3 cloud ERP vendors in 2025.")
```

---

## Notes

- Unlike `brave_search`, Perplexity returns **snippets with substantive content**, not just ranked links — ideal for a single-hop research agent.
- On non-2xx responses the tool raises `RuntimeError`. When called via the `tool(…)` style, the error is captured in `ToolResult.error`.

---

## See also

- [`brave_search`](brave-search.md) — alternative web search (ranked links).
- [`openai_web_search`](openai-web-search.md) — OpenAI Responses API with built-in search.
- [`serp_api_search`](serp-api.md) — any engine, any locale.
