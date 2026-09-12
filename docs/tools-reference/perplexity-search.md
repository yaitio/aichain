<!-- g:tool -->
# `searchPerplexity` — `PerplexitySearchTool`

Web search powered by the Perplexity Search API.

| | |
|---|---|
| Import | `from yait_aichain.tools import PerplexitySearchTool` |
| Risk class | `write` |
| Also exported as | `searchPerplexity` |
| Key | `PERPLEXITY_API_KEY` |

```python
PerplexitySearchTool(
    api_key: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The search query (required). |
| `options.max_results` | `integer` |  | Maximum number of results to return (1–20, default 10). |
| `options.recency` | `string` |  | Filter results by how recently they were published. Use 'day' for breaking news, 'week' for recent events, 'month' for the past month, 'year' for the past year. One of `hour`, `day`, `week`, `month`, `year`. |
| `options.domains` | `list[string]` |  | Restrict results to specific domains (e.g. ["openai.com", "arxiv.org"]). Up to 20 domains. |
| `options.country` | `string` |  | Two-letter ISO 3166-1 country code to localise results (e.g. 'US', 'GB', 'DE'). |
| `options.language` | `string` |  | ISO 639-1 language code (or list of codes) for results (e.g. 'en', 'fr'). Up to 20 languages. |
| `options.after_date` | `string` |  | Only return results published after this date. Format: MM/DD/YYYY (e.g. '01/01/2025'). |
| `options.before_date` | `string` |  | Only return results published before this date. Format: MM/DD/YYYY (e.g. '12/31/2025'). |

```python
result = PerplexitySearchTool()(
    input="…",
    options={"max_results": …},
)
```
<!-- /g:tool -->

Returns rich text snippets extracted from page content — the choice when an
agent needs substantive excerpts without a separate fetch step.

```python
from yait_aichain.tools import PerplexitySearchTool

tool   = PerplexitySearchTool()
result = tool(input="nuclear fusion breakthroughs 2025", options={"max_results": 5})

print(result.output)
```

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

Plain text is deliberate — it stays readable in an agent's conversation and
needs no JSON parsing downstream.

---

## Usage

### Call-style (errors wrapped in `ToolResult`)

```python
tool   = PerplexitySearchTool()
result = tool(
    input   = "GPT-5 benchmark results",
    options = {"max_results": 8, "recency": "month",
               "domains": ["openai.com", "arxiv.org"]},
)

if result:
    print(result.output)
else:
    print("Error:", result.error)
```

### Run-style (raises on error)

```python
tool = PerplexitySearchTool()
text = tool.run(input="managed vector databases comparison 2025",
                options={"max_results": 10, "country": "KZ"})
```

### In a Chain

A Tool step receives the accumulated variables that match its parameters —
`input` and `options` — so the third element renames a variable onto `input`.

```python
from yait_aichain.chain import Chain
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.tools import PerplexitySearchTool

query_skill = Skill(Model("gpt-4o-mini"),
                    prompt="Write one precise web search query for: {topic}")

chain = Chain(steps=[
    (query_skill,            "search_query"),
    (PerplexitySearchTool(), "search_results", {"input": "search_query"}),
    (analysis_skill,         "report"),
])

chain.run(variables={"topic": "AI regulation in the EU (2025)"})
```

### In an Agent

```python
from yait_aichain.agent import Agent, step_count
from yait_aichain.models import Model
from yait_aichain.tools import PerplexitySearchTool

agent = Agent(
    Model("claude-opus-4-6"),
    tools        = [PerplexitySearchTool()],
    stop_when    = [step_count(10)],
    instructions = "You are a research analyst. Prefer primary sources.",
)

result = agent.run("Find and compare the top 3 cloud ERP vendors in 2025.")
```

---

## Notes

- Unlike `searchBrave`, Perplexity returns **snippets with substantive content**,
  not just ranked links — suited to a single-hop research agent.
- On a non-2xx response the tool raises `RuntimeError`; called as `tool(…)`,
  the error is captured in `ToolResult.error`.
- An option the schema does not declare is refused before the request, with
  the declared ones named — see [Tools](../primitives/tools.md).

---

## See also

- [`searchBrave`](brave-search.md) — ranked links.
- [`searchOpenAI`](openai-web-search.md) — a synthesized answer with citations.
- [`searchSerp`](serp-api.md) — any engine, any locale.
