<!-- g:tool -->
# `searchBrave` — `BraveSearchTool`

Web search powered by the Brave Search API.

| | |
|---|---|
| Import | `from yait_aichain.tools import BraveSearchTool` |
| Risk class | `write` |
| Also exported as | `searchBrave` |
| Key | `BRAVE_SEARCH_API_KEY` |

```python
BraveSearchTool(
    api_key: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The search query (required, 1–400 characters). |
| `options.max_results` | `integer` |  | Number of results to return (1–20, default 10). |
| `options.country` | `string` |  | Two-letter ISO 3166-1 country code that biases results (e.g. 'US', 'GB', 'DE'). Defaults to 'US'. |
| `options.language` | `string` |  | Language code for results (e.g. 'en', 'de', 'fr'). Defaults to 'en'. |
| `options.safe_search` | `string` |  | Adult-content filter level. Defaults to 'moderate'. One of `off`, `moderate`, `strict`. |
| `options.freshness` | `string` |  | Filter results by page age. Use 'pd' (past day), 'pw' (past week), 'pm' (past month), 'py' (past year), or a date range 'YYYY-MM-DDtoYYYY-MM-DD'. |
| `options.extra_snippets` | `boolean` |  | When True, up to 5 additional excerpt snippets are included per result. Defaults to False. |
| `options.result_filter` | `string` |  | Comma-separated list of result types to include. Allowed values: discussions, faq, infobox, news, query, summarizer, videos, web, locations. Defaults to all types. |

```python
result = BraveSearchTool()(
    input="…",
    options={"max_results": …},
)
```
<!-- /g:tool -->

Ranked links with short description snippets — the tool to reach for when you
want **URLs an agent can hand straight to `convertToMD`**.

```python
from yait_aichain.tools import BraveSearchTool

tool   = BraveSearchTool()
result = tool(input="Python asyncio tutorial", options={"max_results": 5})

print(result.output)
```

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

Plain text keeps the URLs visible in the agent's conversation, so the model can
pass one to a follow-up call such as `convertToMD`.

---

## Usage

### Direct call

```python
tool = BraveSearchTool()

text = tool.run(
    input   = "climate change solutions",
    options = {"max_results": 10, "country": "GB",
               "freshness": "pw", "extra_snippets": True},
)
```

### Search, then read — the canonical pair

```python
from yait_aichain.agent import Agent, step_count
from yait_aichain.models import Model
from yait_aichain.tools import BraveSearchTool, MarkItDownTool

agent = Agent(
    Model("claude-opus-4-6"),
    tools     = [BraveSearchTool(), MarkItDownTool()],
    stop_when = [step_count(8)],
)

agent.run(
    "Find the top 3 blog posts from 2025 about GPU availability; "
    "for each, fetch the full page as Markdown and extract the key claims."
)
```

The model searches first, then fetches the URLs the search surfaced.

---

## Notes

- Brave returns **ranked links**, not substantive content. Pair it with
  `convertToMD` for deep reads.
- For a single-hop search with real content, prefer
  [`searchPerplexity`](perplexity-search.md).

---

## See also

- [`searchPerplexity`](perplexity-search.md), [`searchSerp`](serp-api.md), [`searchOpenAI`](openai-web-search.md)
- [`convertToMD`](markitdown.md) — turn the URLs Brave returns into readable text.
