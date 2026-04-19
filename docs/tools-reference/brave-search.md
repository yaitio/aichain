# `brave_search` — `BraveSearchTool`

Web search via the **Brave Search API**. Returns ranked links with short description snippets — the tool to reach for when you want **URLs an agent can feed straight into `markitdown`**.

```python
from tools import BraveSearchTool

tool   = BraveSearchTool()
result = tool(query="Python asyncio tutorial", count=5)

print(result.output)
```

---

## Requirements

| | |
|---|---|
| Env var | `BRAVE_SEARCH_API_KEY` |
| Get a key | <https://api-dashboard.search.brave.com> |
| Endpoint | `POST https://api.search.brave.com/res/v1/web/search` |
| Auth header | `x-subscription-token` |

`ValueError` at construction time if the key isn't found.

---

## Constructor

```python
BraveSearchTool(api_key: str | None = None)
```

---

## Parameters

| Name | Type | Required | Default | Notes |
|---|---|---|---|---|
| `query` | `string` | ✓ | — | 1–400 characters. |
| `count` | `integer` | | `10` | 1–20. |
| `country` | `string` | | `US` | ISO 3166-1 alpha-2 (`GB`, `DE`, …). |
| `search_lang` | `string` | | `en` | Language code. |
| `safesearch` | `string` | | `moderate` | `off` / `moderate` / `strict`. |
| `freshness` | `string` | | — | `pd`, `pw`, `pm`, `py`, or `YYYY-MM-DDtoYYYY-MM-DD`. |
| `extra_snippets` | `boolean` | | `false` | Up to 5 extra excerpts per result. |
| `result_filter` | `string` | | — | Comma-separated list: `discussions, faq, infobox, news, query, summarizer, videos, web, locations`. |

---

## Output shape

Plain-text numbered list:

```
Search results for "Python asyncio tutorial" (5 results):

[1] Page Title
    URL: https://example.com/page
    Summary: Brief excerpt describing the page content.
    Age: 2025-09-12

[2] Another Title
    URL: https://another.com/article
    Summary: …
    · extra snippet 1
    · extra snippet 2
```

Plain text keeps URLs visible in the agent's memory preview, so the orchestrator can copy a specific URL into a follow-up tool call (e.g. `markitdown(source=…)`).

---

## Usage

### Direct call

```python
tool = BraveSearchTool()

text = tool.run(
    query          = "climate change solutions",
    count          = 10,
    country        = "GB",
    freshness      = "pw",
    extra_snippets = True,
)
```

### Brave → MarkItDown — canonical pattern

```python
from agent import Agent
from models import Model
from tools import BraveSearchTool, MarkItDownTool

agent = Agent(
    orchestrator = Model("claude-opus-4-6"),
    tools        = [BraveSearchTool(), MarkItDownTool()],
    mode         = "agile",
    max_steps    = 8,
)

agent.run(
    "Find the top 3 blog posts from 2025 about GPU availability; "
    "for each, fetch the full page as Markdown and extract the key claims."
)
```

The agent learns: search first, then fetch the URLs it surfaces.

---

## Notes

- Brave returns **ranked links**, not substantive content snippets. Pair with `markitdown` for deep reads.
- If you want a single-hop search with real content, prefer [`perplexity_search`](perplexity-search.md).

---

## See also

- [`perplexity_search`](perplexity-search.md), [`serp_api_search`](serp-api.md), [`openai_web_search`](openai-web-search.md)
- [`markitdown`](markitdown.md) — convert URLs Brave returned into readable text.
