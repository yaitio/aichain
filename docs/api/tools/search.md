# API: search

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../tools/search.md).

<!-- g:tool -->
## `search()`

```python
search(
    service: str,
    *,
    api_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> yait_aichain.tools.search._base.Search
```

A web search tool for *service* — one of `SERVICES`.

`name` and `description` default to the provider's own and override
them; `api_key` defaults to the provider's environment variable. Every
other setting of a search is an option of the call, so nothing else is
taken here.

```python
search("exa").run("AI and leadership", {"domains": ["hbr.org"]})["results"]
Agent(model, tools=[search("perplexity")])
```

---

## The classes behind it

What the factory returns, with the schema a model is shown.

### `searchPerplexity` — `PerplexitySearchTool`

Web search powered by the Perplexity Search API — each result with a substantive passage from the page.

| | |
|---|---|
| Import | `from yait_aichain.tools import PerplexitySearchTool` |
| Risk class | `read` |
| Also exported as | `searchPerplexity` |
| Key | `PERPLEXITY_API_KEY` |

```python
PerplexitySearchTool(
    api_key: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The search query, in natural language. |
| `options.max_results` | `integer` |  | How many results to return (default 10). |
| `options.recency` | `string` |  | Only pages published within this window. One of `hour`, `day`, `week`, `month`, `year`. |
| `options.start_date` | `string` |  | Only pages published on or after this date, YYYY-MM-DD. |
| `options.end_date` | `string` |  | Only pages published on or before this date, YYYY-MM-DD. |
| `options.country` | `string` |  | ISO 3166-1 alpha-2 country code, e.g. 'GB'. |
| `options.language` | `string` |  | ISO 639-1 language code, e.g. 'en'. |
| `options.domains` | `list[string]` |  | Only these sites — a domain, or a domain with a path prefix. |
| `options.after_date` | `string` |  | Perplexity's own MM/DD/YYYY lower bound; wins over start_date. |
| `options.before_date` | `string` |  | Perplexity's own MM/DD/YYYY upper bound; wins over end_date. |

```python
result = PerplexitySearchTool()(
    input="…",
    options={"max_results": …},
)
```

### `searchBrave` — `BraveSearchTool`

Web search powered by the Brave Search API — titles, URLs, the page's description and, with `extra_snippets`, more passages from it.

| | |
|---|---|
| Import | `from yait_aichain.tools import BraveSearchTool` |
| Risk class | `read` |
| Also exported as | `searchBrave` |
| Key | `BRAVE_SEARCH_API_KEY` |

```python
BraveSearchTool(
    api_key: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The search query, in natural language. |
| `options.max_results` | `integer` |  | How many results to return (default 10). |
| `options.recency` | `string` |  | Only pages published within this window. One of `hour`, `day`, `week`, `month`, `year`. |
| `options.start_date` | `string` |  | Only pages published on or after this date, YYYY-MM-DD. |
| `options.end_date` | `string` |  | Only pages published on or before this date, YYYY-MM-DD. |
| `options.country` | `string` |  | ISO 3166-1 alpha-2 country code, e.g. 'GB'. |
| `options.language` | `string` |  | ISO 639-1 language code, e.g. 'en'. |
| `options.domains` | `list[string]` |  | Only these sites — a domain, or a domain with a path prefix. |
| `options.safe_search` | `string` |  | Adult-content filter. Default 'moderate'. One of `off`, `moderate`, `strict`. |
| `options.extra_snippets` | `boolean` |  | Up to five more passages from each page. Default false. |
| `options.result_filter` | `string` |  | Comma-separated result types: discussions, faq, infobox, news, query, summarizer, videos, web, locations. |
| `options.freshness` | `string` |  | Brave's own time filter — 'pd', 'pw', 'pm', 'py' or 'YYYY-MM-DDtoYYYY-MM-DD'; wins over recency and the dates. |

```python
result = BraveSearchTool()(
    input="…",
    options={"max_results": …},
)
```

### `searchExa`

Web search powered by the Exa Search API — results with the passages that answer the query.

| | |
|---|---|
| Import | `from yait_aichain.tools import searchExa` |
| Risk class | `read` |
| Key | `EXA_API_KEY` |

```python
searchExa(
    api_key: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The search query, in natural language. |
| `options.max_results` | `integer` |  | How many results to return (default 10). |
| `options.recency` | `string` |  | Only pages published within this window. One of `hour`, `day`, `week`, `month`, `year`. |
| `options.start_date` | `string` |  | Only pages published on or after this date, YYYY-MM-DD. |
| `options.end_date` | `string` |  | Only pages published on or before this date, YYYY-MM-DD. |
| `options.country` | `string` |  | ISO 3166-1 alpha-2 country code, e.g. 'GB'. |
| `options.language` | `string` |  | ISO 639-1 language code, e.g. 'en'. Not applied here: Exa has no language filter. |
| `options.domains` | `list[string]` |  | Only these sites — a domain, or a domain with a path prefix. |
| `options.exclude_domains` | `list[string]` |  | Leave out results from these domains. |
| `options.category` | `string` |  | Focus on one kind of page. One of `company`, `publication`, `news`, `personal site`, `financial report`, `people`. |
| `options.search_type` | `string` |  | 'auto' (default) balances quality and speed; 'instant'/'fast' trade quality for latency; the 'deep' modes search iteratively and cost more. One of `instant`, `fast`, `auto`, `deep-lite`, `deep`, `deep-reasoning`. |
| `options.highlight_chars` | `integer` |  | Most characters of highlights per result (default 1000); 0 turns them off. |
| `options.text_chars` | `integer` |  | Also return each page's text, cut to this many characters. Off by default — highlights usually carry the answer. |

```python
result = searchExa()(
    input="…",
    options={"max_results": …},
)
```

### `searchFirecrawl`

Web search powered by Firecrawl — results with a description and, on request, each page's content as Markdown.

| | |
|---|---|
| Import | `from yait_aichain.tools import searchFirecrawl` |
| Risk class | `read` |
| Key | `FIRECRAWL_API_KEY` |

```python
searchFirecrawl(
    api_key: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The search query, in natural language. |
| `options.max_results` | `integer` |  | How many results to return (default 10). |
| `options.recency` | `string` |  | Only pages published within this window. One of `hour`, `day`, `week`, `month`, `year`. |
| `options.start_date` | `string` |  | Only pages published on or after this date, YYYY-MM-DD. |
| `options.end_date` | `string` |  | Only pages published on or before this date, YYYY-MM-DD. |
| `options.country` | `string` |  | ISO 3166-1 alpha-2 country code, e.g. 'GB'. |
| `options.language` | `string` |  | ISO 639-1 language code, e.g. 'en'. Not applied here: Firecrawl has no language filter. |
| `options.domains` | `list[string]` |  | Only these sites — a domain, or a domain with a path prefix. |
| `options.exclude_domains` | `list[string]` |  | Leave out results from these host names. Not with domains. |
| `options.categories` | `list[string]` |  | Restrict to research sites, PDFs, the developer index or the government index ('gov' combines with nothing else). |
| `options.sources` | `list[string]` |  | Where to search: 'web' (default) and/or 'news'. |
| `options.location` | `string` |  | A place to search from, e.g. 'Berlin,Germany'. |
| `options.safe` | `boolean` |  | Filter explicit content. |
| `options.tbs` | `string` |  | Firecrawl's raw time filter, e.g. 'qdr:w' or 'sbd:1,qdr:m'; wins over recency and the dates. |
| `options.text_chars` | `integer` |  | Also return each page's content as Markdown, cut to this many characters. Scraping costs extra credits. |

```python
result = searchFirecrawl()(
    input="…",
    options={"max_results": …},
)
```

### `searchTavily`

Web search powered by Tavily — each page with the passages chosen for the query, and on request the page itself.

| | |
|---|---|
| Import | `from yait_aichain.tools import searchTavily` |
| Risk class | `read` |
| Key | `TAVILY_API_KEY` |

```python
searchTavily(
    api_key: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The search query, in natural language. |
| `options.max_results` | `integer` |  | How many results to return (default 10). |
| `options.recency` | `string` |  | Only pages published within this window. One of `hour`, `day`, `week`, `month`, `year`. |
| `options.start_date` | `string` |  | Only pages published on or after this date, YYYY-MM-DD. |
| `options.end_date` | `string` |  | Only pages published on or before this date, YYYY-MM-DD. |
| `options.country` | `string` |  | ISO 3166-1 alpha-2 country code, e.g. 'GB'. |
| `options.language` | `string` |  | ISO 639-1 language code, e.g. 'en'. |
| `options.domains` | `list[string]` |  | Only these sites — a domain, or a domain with a path prefix. |
| `options.topic` | `string` |  | What to search: 'general' (default), 'news' or 'finance'. One of `general`, `news`, `finance`. |
| `options.search_depth` | `string` |  | 'basic' (default); 'advanced' is more relevant and costs two credits. One of `ultra-fast`, `fast`, `basic`, `advanced`. |
| `options.chunks_per_source` | `integer` |  | Passages per page, 1 to 3 (default 3). |
| `options.exclude_domains` | `list[string]` |  | Leave out results from these domains. |
| `options.exact_match` | `boolean` |  | Only pages containing the quoted phrases of the query exactly. |
| `options.safe_search` | `boolean` |  | Filter adult content. |
| `options.filter_by_published_date` | `boolean` |  | Drop pages outside the date window — and pages with no date. |
| `options.filter_by_language` | `boolean` |  | Make language a filter rather than a ranking boost. |
| `options.text_chars` | `integer` |  | Also return each page as Markdown, cut to this many characters. |

```python
result = searchTavily()(
    input="…",
    options={"max_results": …},
)
```

### `searchParallel`

Web search powered by Parallel — the excerpts of each page that serve the objective of the search.

| | |
|---|---|
| Import | `from yait_aichain.tools import searchParallel` |
| Risk class | `read` |
| Key | `PARALLEL_API_KEY` |

```python
searchParallel(
    api_key: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The search query, in natural language. |
| `options.max_results` | `integer` |  | How many results to return (default 10). |
| `options.recency` | `string` |  | Only pages published within this window. One of `hour`, `day`, `week`, `month`, `year`. |
| `options.start_date` | `string` |  | Only pages published on or after this date, YYYY-MM-DD. |
| `options.end_date` | `string` |  | Only pages published on or before this date, YYYY-MM-DD. Not applied here: Parallel filters from a start date only. |
| `options.country` | `string` |  | ISO 3166-1 alpha-2 country code, e.g. 'GB'. |
| `options.language` | `string` |  | ISO 639-1 language code, e.g. 'en'. Not applied here: Parallel has no language filter. |
| `options.domains` | `list[string]` |  | Only these sites — a domain, or a domain with a path prefix. |
| `options.objective` | `string` |  | The goal of the search in natural language; default: the query. |
| `options.search_queries` | `list[string]` |  | Short keyword queries, three to six words, to run beside the query. |
| `options.mode` | `string` |  | 'advanced' (default) is the most thorough; 'turbo' the fastest. One of `turbo`, `fast`, `basic`, `advanced`. |
| `options.exclude_domains` | `list[string]` |  | Leave out these domains or domain/path prefixes. Not with domains. |
| `options.max_chars_total` | `integer` |  | Upper bound on characters of excerpts across all results. |
| `options.max_age_seconds` | `integer` |  | Fetch a page live when the cached copy is older than this (600 or more). Slower. |

```python
result = searchParallel()(
    input="…",
    options={"max_results": …},
)
```
<!-- /g:tool -->
