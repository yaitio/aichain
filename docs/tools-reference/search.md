<!-- g:tool -->
# Web search

## `searchPerplexity` — `PerplexitySearchTool`

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

## `searchBrave` — `BraveSearchTool`

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

## `searchSerp` — `SerpApiTool`

Web search through SerpAPI — Google by default, or any engine it supports.

| | |
|---|---|
| Import | `from yait_aichain.tools import SerpApiTool` |
| Risk class | `read` |
| Also exported as | `searchSerp` |
| Key | `SERPAPI_API_KEY` |

```python
SerpApiTool(
    api_key: str | None = None,
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
| `options.engine` | `string` |  | The engine: 'google' (default), 'bing', 'duckduckgo', 'baidu', 'yandex', 'google_news', 'google_scholar' and others. |
| `options.location` | `string` |  | A place to search from, e.g. 'Austin, Texas'. |
| `options.safe` | `string` |  | Google SafeSearch. One of `active`, `off`. |
| `options.start` | `integer` |  | Offset of the first result, for paging. |
| `options.no_cache` | `boolean` |  | Bypass SerpAPI's cache. |
| `options.time_filter` | `string` |  | A raw Google 'tbs', e.g. 'qdr:w'; wins over recency and the dates. |

```python
result = SerpApiTool()(
    input="…",
    options={"max_results": …},
)
```

## `searchExa`

Web search powered by the Exa Search API — results with the passages that answer the query.

| | |
|---|---|
| Import | `from yait_aichain.tools import searchExa` |
| Risk class | `read` |
| Key | `EXA_API_KEY` |

```python
searchExa(
    api_key: str | None = None,
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

## `searchFirecrawl`

Web search powered by Firecrawl — results with a description and, on request, each page's content as Markdown.

| | |
|---|---|
| Import | `from yait_aichain.tools import searchFirecrawl` |
| Risk class | `read` |
| Key | `FIRECRAWL_API_KEY` |

```python
searchFirecrawl(
    api_key: str | None = None,
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

## `searchTavily`

Web search powered by Tavily — each page with the passages chosen for the query, and on request the page itself.

| | |
|---|---|
| Import | `from yait_aichain.tools import searchTavily` |
| Risk class | `read` |
| Key | `TAVILY_API_KEY` |

```python
searchTavily(
    api_key: str | None = None,
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
<!-- /g:tool -->

## One format, six providers

The providers sell the same thing — ranked pages for a query, filtered by
time, place and site — and spell every part of it differently. The library
holds one vocabulary of options and one record of results, and each provider
tool translates them, the way the [decision tools](decision.md) work.

| Tool | Strength | Results | Key |
|---|---|---|---|
| `searchPerplexity` | a substantive passage from each page; dates on every result | ≤ 20 | `PERPLEXITY_API_KEY` |
| `searchBrave` | ranked links and descriptions; more passages with `extra_snippets` | ≤ 20 | `BRAVE_SEARCH_API_KEY` |
| `searchSerp` | Google — or Bing, DuckDuckGo, Baidu, Yandex — through SerpAPI | ≤ 100 | `SERPAPI_API_KEY` |
| `searchExa` | the passages that answer the query; the page's own text on request | ≤ 100 | `EXA_API_KEY` |
| `searchFirecrawl` | a scraper first: each page as Markdown in the same call; news and PDF/research/gov indexes | ≤ 100 | `FIRECRAWL_API_KEY` |
| `searchTavily` | up to three passages chosen for the query per page; news and finance topics; page as Markdown on request | ≤ 20 | `TAVILY_API_KEY` |

## Options

| Option | Meaning | Perplexity | Brave | SerpAPI (Google) | Exa | Firecrawl | Tavily |
|---|---|---|---|---|---|---|---|
| `max_results` | how many | `max_results` | `count` | `num` | `numResults` | `limit` | native |
| `recency` | `hour`, `day`, `week`, `month`, `year` | native | `freshness`; `hour` widened to a day, said so | `tbs=qdr:` | a start date | `tbs=qdr:` — **not enforced**, said so | `time_range`; `hour` widened to a day, said so |
| `start_date`, `end_date` | `YYYY-MM-DD` | MM/DD/YYYY filters | `freshness` range | `tbs=cdr:` | ISO dates | `tbs=cdr:` — **not enforced**, said so | native; undated pages pass unless `filter_by_published_date` |
| `country` | ISO alpha-2 | native | native | `gl` | `userLocation` | native | the country's name; general topic only, said so |
| `language` | ISO 639-1 | native | `search_lang` | `hl` | **not applied** — said so | **not applied** — said so | a ranking boost; a filter with `filter_by_language` |
| `domains` | a domain or a domain with a path | native, ≤ 20 | `site:` in the query | `site:` in the query | `includeDomains` | `includeDomains`; with a path, `site:` — said so | `include_domains` |

`recency` and `start_date` bound the same side; the later is kept. Each
provider keeps its own options beside these — Brave `extra_snippets`,
`safe_search`, `result_filter`, `freshness`; SerpAPI `engine`, `location`,
`safe`, `start`, `no_cache`, `time_filter`; Perplexity `after_date`,
`before_date`; Exa `category`, `search_type`, `exclude_domains`,
`highlight_chars`, `text_chars`; Firecrawl `exclude_domains`, `categories`,
`sources`, `location`, `safe`, `tbs`, `text_chars`; Tavily `topic`,
`search_depth`, `chunks_per_source`, `exclude_domains`, `exact_match`,
`safe_search`, `filter_by_published_date`, `filter_by_language`, `text_chars`. A provider's own time option wins over the
standard ones.

**Nothing is dropped in silence.** An option a provider cannot honour, or
honours differently, is reported three ways: a warning, a `Note:` line at the
end of the text the model reads, and an entry in `notices`. On SerpAPI the
time, country and language filters apply to the Google engines; on another
engine they are reported as not applied.

## Results

```python
from yait_aichain.tools import searchPerplexity

tool  = searchPerplexity()
found = tool.search("AI and leadership", {"domains": ["hbr.org"],
                                          "start_date": "2026-09-28", "end_date": "2026-10-04"})
found["results"][0]
# {"title": "...", "url": "https://hbr.org/2026/10/...", "date": "2026-10-02",
#  "author": "", "snippets": ["..."], "text": ""}
found["notices"]            # [] — or what was not applied, and why

tool.run(input="AI and leadership", options={...})   # the same, as numbered text for a model
```

- **`search()`** is for code: `{"query", "provider", "results", "notices"}`,
  every result with the same six keys from every provider.
- **`run()`** is what an agent calls: the same results as numbered text —
  title, URL, `Published: … · Author: …`, one line per passage, then any notes.
- **`date`** is the publication date as the provider reports it, normalised to
  a day (`Sep 30, 2026` and Google's `3 days ago` included), or empty. It is
  not a fact about the page: providers date pages wrongly — Brave's `page_age`
  is when it saw a page, so its article date is preferred; Perplexity has dated
  an April article to September. When the date matters, read it from the page:
  `convertToMD().last_metadata["published"]`.
- **`text`** is the page's own text where the provider returned it — Exa,
  Firecrawl and Tavily, with `text_chars`.

## Provider notes

- **Exa's date filter lets undated pages through** — 31 of 73 results in a
  one-week window carried no date (2026-10-06). A window is a preference there,
  not a guarantee.
- **Firecrawl does not enforce its time filter** — measured 2026-10-08: a
  28.09–04.10 range returned exactly the unfiltered results, and with
  `hbr.org` all five fell between 2025-11 and 2026-08. The filter is sent and
  every such search carries a notice. Its web results carry no date unless the
  page is scraped (`text_chars`), so ask for text when the date matters.
- **Exa's highlights are capped** at 1 000 characters a result by default
  (`highlight_chars`; `0` turns them off): uncapped they were 5 000–8 000
  characters each, most of every page, in the agent's context.
- **Search finds what is relevant, not everything that was published.** For a
  complete list of a site's new pages, its own feed or sitemap is the better
  source; a search is the top-up.

## A new provider

Subclass `Search`; set `name`, `description`, `max_results`, `_ENV_KEY`,
`_PROVIDER`, `extra_options` and `declines` (standard options it cannot honour,
with the reason); implement `_search(query, opts, window, notices)` returning
records built with `self.record(...)`. The options, the window, the notices,
the text and the errors are the base's.
