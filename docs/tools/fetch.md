<!-- g:tool -->
# Fetch: link → page

## `fetchFirecrawl`

A web page's text by Firecrawl Scrape — the main content, as Markdown, with the page's metadata.

| | |
|---|---|
| Import | `from yait_aichain.tools import fetchFirecrawl` |
| Risk class | `read` |
| Key | `FIRECRAWL_API_KEY` |

```python
fetchFirecrawl(
    api_key: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `url` | `string` | ✓ | The page's URL, http or https. |
| `options.max_chars` | `integer` |  | Most characters of text per page (default 20 000). |
| `options.only_main_content` | `boolean` |  | Drop navigation, headers and footers (default true). |

## `fetchTavily`

Web pages' text by Tavily Extract — up to 20 URLs a call, as Markdown.

| | |
|---|---|
| Import | `from yait_aichain.tools import fetchTavily` |
| Risk class | `read` |
| Key | `TAVILY_API_KEY` |

```python
fetchTavily(
    api_key: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `url` | `string` | ✓ | The page's URL, http or https. |
| `options.max_chars` | `integer` |  | Most characters of text per page (default 20 000). |
| `options.extract_depth` | `string` |  | 'basic' (default); 'advanced' reaches tables and embedded content at twice the price. One of `basic`, `advanced`. |

## `fetchExa`

Web pages' text by Exa Contents — up to 100 URLs a call, with title, date and author.

| | |
|---|---|
| Import | `from yait_aichain.tools import fetchExa` |
| Risk class | `read` |
| Key | `EXA_API_KEY` |

```python
fetchExa(
    api_key: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `url` | `string` | ✓ | The page's URL, http or https. |
| `options.max_chars` | `integer` |  | Most characters of text per page (default 20 000). |
| `options.max_age_hours` | `integer` |  | Use a cached copy younger than this (0 fetches fresh, -1 cache only; at most 720). |

## `fetchParallel`

Web pages' full content by Parallel Extract — up to 20 URLs a call, with title and date.

| | |
|---|---|
| Import | `from yait_aichain.tools import fetchParallel` |
| Risk class | `read` |
| Key | `PARALLEL_API_KEY` |

```python
fetchParallel(
    api_key: str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `url` | `string` | ✓ | The page's URL, http or https. |
| `options.max_chars` | `integer` |  | Most characters of text per page (default 20 000). |
| `options.max_age_seconds` | `integer` |  | Fetch a page live when the cached copy is older than this (600 or more). Slower. |

## `fetchLocal`

A web page's text fetched from this machine and converted by MarkItDown — free.

| | |
|---|---|
| Import | `from yait_aichain.tools import fetchLocal` |
| Risk class | `read` |

```python
fetchLocal(
    timeout = (10.0, 30.0),
    headers: dict | None = None,
    respect_robots: bool = True,
    robots_agents = ('*', 'ClaudeBot', 'Claude-User', 'anthropic-ai'),
    *,
    name: str | None = None,
    description: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `url` | `string` | ✓ | The page's URL, http or https. |
| `options.max_chars` | `integer` |  | Most characters of text per page (default 20 000). |
<!-- /g:tool -->

## One record, five ways to get it

A search finds the link; a fetch brings back the page. Four providers sell
this as an API and spell every part of it differently; the fifth runs on your
machine and costs nothing. The library holds one record of a page and each
tool translates, the way the [search tools](search.md) work.

| Tool | URLs a call | Title | Date | Author | Price a page | Key |
|---|---|---|---|---|---|---|
| `fetchFirecrawl` | 1 | ✓ | where the page declares it | where the page declares it | 1 credit (≈ $0.0038 on Hobby); a PDF per PDF page | `FIRECRAWL_API_KEY` |
| `fetchTavily` | ≤ 20 | ✓ | — | — | 1 credit per 5 URLs ($0.0016) | `TAVILY_API_KEY` |
| `fetchExa` | ≤ 100 | ✓ | ✓ | ✓ | $0.001 | `EXA_API_KEY` |
| `fetchParallel` | ≤ 20 | ✓ | ✓ | — | $0.001 | `PARALLEL_API_KEY` |
| `fetchLocal` | 1 | ✓ | from the page's `<head>` | from the page's `<head>` | free | — |

A longer list is split into calls of what the provider takes; the pages come
back in the order asked, duplicates once.

**`fetchLocal` is not installed by default.** It runs
[MarkItDown](https://github.com/microsoft/markitdown) on this machine, and MarkItDown comes only with the
extra:

```bash
pip install "yait-aichain[fetch]"
```

Constructing it without MarkItDown raises `ImportError` with that line.

## Pages

```python
from yait_aichain.tools import fetch

found = fetch("exa").run(["https://example.com", "https://example.invalid/x"])
found["pages"][0]
# {"url": "https://example.com", "final_url": "https://example.com",
#  "title": "Example Domain", "published": "", "author": "",
#  "text": "This domain is for use in documentation examples...", "error": ""}
found["pages"][1]["error"]  # "CRAWL_LIVECRAWL_TIMEOUT (HTTP 504)"
found["cost"]               # 0.002 — read from the answer where the provider reports it
found["cost_basis"]         # "reported by Exa"

str(found)                  # what a model reads: each page, or why it could not be fetched

Agent(model, tools=[search("perplexity"), fetch("firecrawl")])
```

`fetch(service)` makes the tool — `firecrawl`, `tavily`, `exa`, `parallel` or
`local` — with `api_key`, `name` and `description`; `local` takes `timeout`,
`headers`, `respect_robots` and `robots_agents` instead of a key. `run()` takes
one URL or a list and returns a `Record`: the dict above for code, and for a
model each page as text and each failed one as `Could not fetch …: reason`.

A page that could not be fetched is still a page, with `error` set and no
text: one bad URL does not fail the call, and an agent reads why. An API error
(a bad key, no credits) raises. `fetch()` on a tool is `run()` under its 3.2
name.

`published` is a day, `YYYY-MM-DD`, or empty — never guessed.

## Options

| Option | Meaning | Where it goes |
|---|---|---|
| `max_chars` | most characters of text per page, default 20 000 | Exa `text.maxCharacters`, Parallel `max_chars_per_result`; cut here for the rest |
| `only_main_content` | Firecrawl: drop navigation, headers, footers (default true) | `onlyMainContent` |
| `extract_depth` | Tavily: `basic` or `advanced` (tables, embedded content, twice the price) | native |
| `max_age_hours` | Exa: how old a cached copy may be; `0` fetches fresh | `maxAgeHours` |
| `max_age_seconds` | Parallel: fetch live when the cached copy is older (≥ 600) | `fetch_policy` |

## robots.txt and `fetchLocal`

`fetchLocal` reads a site's `robots.txt` once per host before it fetches a
page, and skips the page — no request — when it disallows any of
`robots_agents`: `*` and the Claude crawlers (`ClaudeBot`, `Claude-User`,
`anthropic-ai`) by default, since a page an agent reads is usually read for a
model. Per [RFC 9309](https://www.rfc-editor.org/rfc/rfc9309), a `robots.txt`
that answers 4xx allows everything and one that answers 5xx disallows
everything. `respect_robots=False` turns the check off — the caller's decision
about the sites they read. No User-Agent is invented; pass `headers=` to set
one.

```python
fetchLocal(respect_robots=True, robots_agents=("*", "ClaudeBot"), timeout=(10, 30), headers=None)
```

The provider tools fetch from the provider's own servers under the provider's
own policy; the library does not check `robots.txt` for them.

## What each reaches

Measured 2026-10-08 on 34 articles from six business publishers, fetched three
ways:

- **Firecrawl** returned the full text of all 34, including pages behind bot
  protection and articles readers see only with a subscription — how, the
  measurement did not establish. About half of what came back was still page
  furniture, `only_main_content` notwithstanding.
- **MarkItDown on this machine** returned 21 in full: what the site serves an
  ordinary client. The whole page is converted, navigation and footers
  included.
- **Exa and Tavily**, on the articles behind a paywall, returned the public
  teaser rather than the article.

## A new provider

Subclass `Fetch`; set `name`, `description`, `max_batch`, `_ENV_KEY`,
`_PROVIDER`, `extra_options`, `_LIST_PRICE` and `_PRICE_SOURCE`; implement
`_fetch(urls, opts, notices, cost)` returning `{url: page}` built with
`self.page(...)`, and fill `cost` where the answer says what the call cost.
Batching, order, `max_chars`, the text and the errors are the base's.
