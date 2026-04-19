# `serp_api_search` — `SerpApiTool`

Multi-engine search via **SerpAPI**. One tool, 50+ underlying engines: Google, Bing, Yahoo, DuckDuckGo, Baidu, Yandex, Google News, Google Scholar, Google Shopping, Google Maps, Google Jobs, and more. Pick the engine with a single parameter.

```python
from tools import SerpApiTool

tool   = SerpApiTool()
result = tool(query="Python asyncio tutorial", num=5)          # Google by default

print(result.output)
```

---

## Requirements

| | |
|---|---|
| Env var | `SERPAPI_API_KEY` |
| Get a key | <https://serpapi.com/manage-api-key> |
| Endpoint | `GET https://serpapi.com/search` |

`ValueError` at construction time if the key isn't found.

---

## Constructor

```python
SerpApiTool(api_key: str | None = None)
```

---

## Parameters

| Name | Type | Required | Default | Notes |
|---|---|---|---|---|
| `query` | `string` | ✓ | — | The search query. |
| `engine` | `string` | | `google` | See list below. |
| `num` | `integer` | | `10` | Up to 100. |
| `location` | `string` | | — | e.g. `"Austin, Texas"`. Google/Bing/Yahoo. |
| `gl` | `string` | | — | Country (Google only). |
| `hl` | `string` | | — | Interface language (Google only). |
| `safe` | `string` | | — | `active` / `off` (Google only). |
| `tbs` | `string` | | — | Time filter: `qdr:d`, `qdr:w`, `qdr:m`, `qdr:y` (Google only). |
| `start` | `integer` | | `0` | Pagination offset (Google only). |
| `no_cache` | `boolean` | | `false` | Bypass SerpAPI's cache. |

### Supported `engine` values

`google`, `bing`, `yahoo`, `duckduckgo`, `baidu`, `yandex`, `google_news`, `bing_news`, `google_images`, `google_shopping`, `google_maps`, `google_jobs`, `google_scholar` — plus any other engine SerpAPI supports.

---

## Output shape

```
Search results for "Python asyncio tutorial" via google (5 results):

[1] Page Title
    URL: https://example.com/page
    Date: Mar 15, 2025
    Source: Example.com
    Summary: Brief excerpt…
```

If the query returns nothing, the output ends with `(no results found)`.

---

## Usage

### Cross-engine comparisons

```python
tool = SerpApiTool()

print(tool.run(query="climate policy 2025", engine="google",     num=5))
print(tool.run(query="climate policy 2025", engine="bing",       num=5))
print(tool.run(query="climate policy 2025", engine="duckduckgo", num=5))
```

### Localised / non-Latin search

```python
tool.run(query="人工智能 2025", engine="baidu")
tool.run(query="искусственный интеллект",  engine="yandex", num=10)
tool.run(query="climate",    engine="google", gl="gb", hl="en", location="London, UK")
```

### Time-scoped research

```python
tool.run(query="GPT-5 benchmarks",  engine="google", tbs="qdr:w")   # past week
tool.run(query="AI regulation",     engine="google_news", num=20)    # news only
```

### In an Agent

```python
from agent import Agent
from models import Model
from tools import SerpApiTool, MarkItDownTool

agent = Agent(
    orchestrator = Model("claude-opus-4-6"),
    tools        = [SerpApiTool(), MarkItDownTool()],
    mode         = "agile",
    max_steps    = 10,
)

agent.run(
    "Compare how Google News and Bing News cover the EU AI Act. "
    "Cite 5 articles from each and summarise the framing differences."
)
```

---

## Notes

- SerpAPI charges per request regardless of engine. Use `no_cache=True` sparingly.
- Google News and Bing News return items under `news_results`; the tool handles both transparently.

---

## See also

- [`perplexity_search`](perplexity-search.md), [`brave_search`](brave-search.md), [`openai_web_search`](openai-web-search.md)
- [`markitdown`](markitdown.md) — fetch any URL as Markdown.
