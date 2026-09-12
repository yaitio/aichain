<!-- g:tool -->
# `searchSerp` — `SerpApiTool`

Multi-engine web search powered by SerpAPI.

| | |
|---|---|
| Import | `from yait_aichain.tools import SerpApiTool` |
| Risk class | `write` |
| Also exported as | `searchSerp` |
| Key | `SERPAPI_API_KEY` |

```python
SerpApiTool(
    api_key: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The search query (required). |
| `options.engine` | `string` |  | The search engine to use. Common values: 'google' (default), 'bing', 'yahoo', 'duckduckgo', 'baidu', 'yandex', 'google_news', 'bing_news', 'google_images', 'google_shopping', 'google_scholar', 'google_maps', 'google_jobs'. Defaults to 'google'. |
| `options.max_results` | `integer` |  | Number of results to return (default 10, max 100). |
| `options.location` | `string` |  | Geographic location to use for localised results (e.g. 'Austin, Texas', 'Paris, France'). Supported by Google, Bing, and Yahoo. |
| `options.country` | `string` |  | Two-letter ISO 3166-1 country code that biases results (e.g. 'us', 'gb', 'de'). Google engine only. |
| `options.language` | `string` |  | Language code for the interface/results (e.g. 'en', 'fr', 'de'). Google engine only. |
| `options.safe` | `string` |  | Safe search filter. 'active' enables filtering; 'off' disables it. Google engine only. One of `active`, `off`. |
| `options.time_filter` | `string` |  | Advanced time-based search filter. Use 'qdr:d' (past day), 'qdr:w' (past week), 'qdr:m' (past month), 'qdr:y' (past year). Google engine only. |
| `options.start` | `integer` |  | Offset for pagination — the index of the first result to return. Default 0. Google engine only. |
| `options.no_cache` | `boolean` |  | When True, forces a live search and bypasses SerpAPI's cached results. Useful for real-time data. Default False. |

```python
result = SerpApiTool()(
    input="…",
    options={"engine": …},
)
```
<!-- /g:tool -->

One tool, many engines: Google, Bing, Yahoo, DuckDuckGo, Baidu, Yandex, Google
News, Google Scholar, Google Shopping, Google Maps, Google Jobs and more —
chosen with `options["engine"]`.

```python
from yait_aichain.tools import SerpApiTool

tool   = SerpApiTool()
result = tool(input="Python asyncio tutorial", options={"max_results": 5})   # Google by default

print(result.output)
```

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

for engine in ("google", "bing", "duckduckgo"):
    print(tool.run(input="climate policy 2025",
                   options={"engine": engine, "max_results": 5}))
```

### Localised and non-Latin search

```python
tool = SerpApiTool()

tool.run(input="人工智能 2025", options={"engine": "baidu"})
tool.run(input="artificial intelligence", options={"engine": "yandex", "max_results": 10})
tool.run(input="climate", options={"engine": "google", "country": "gb",
                                   "language": "en", "location": "London, UK"})
```

### Time-scoped research

```python
tool = SerpApiTool()

tool.run(input="GPT-5 benchmarks", options={"engine": "google", "time_filter": "qdr:w"})  # past week
tool.run(input="AI regulation",    options={"engine": "google_news", "max_results": 20})
```

### In an Agent

```python
from yait_aichain.agent import Agent, step_count
from yait_aichain.models import Model
from yait_aichain.tools import SerpApiTool, MarkItDownTool

agent = Agent(
    Model("claude-opus-4-6"),
    tools     = [SerpApiTool(), MarkItDownTool()],
    stop_when = [step_count(10)],
)

agent.run(
    "Compare how Google News and Bing News cover the EU AI Act. "
    "Cite 5 articles from each and summarise the framing differences."
)
```

---

## Notes

- SerpAPI charges per request whatever the engine. Use `no_cache` sparingly.
- `country`, `language` and `time_filter` apply to the Google engine only.
- Google News and Bing News return items under `news_results`; the tool reads both.

---

## See also

- [`searchPerplexity`](perplexity-search.md), [`searchBrave`](brave-search.md), [`searchOpenAI`](openai-web-search.md)
- [`convertToMD`](markitdown.md) — fetch any URL as Markdown.
