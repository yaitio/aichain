# API: fetch

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../tools/fetch.md).

<!-- g:tool -->
## `fetch()`

```python
fetch(
    service: str,
    *,
    api_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
    **settings,
) -> yait_aichain.tools.fetch._base.Fetch
```

A fetch tool for *service* — one of `SERVICES`.

`name` and `description` default to the provider's own and override
them; `api_key` defaults to the provider's environment variable.
`local` takes `timeout`, `headers`, `respect_robots` and
`robots_agents` as well, and no key. Everything else is an option of the
call.

```python
fetch("firecrawl").run(["https://example.com/a", "https://example.com/b"])["pages"]
fetch("local", respect_robots=True).run("https://example.com")
```

---

## The classes behind it

What the factory returns, with the schema a model is shown.

### `fetchFirecrawl`

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
| `url` | `list[string]` | ✓ | The pages' URLs, http or https — one or several. |
| `options.max_chars` | `integer` |  | Most characters of text per page (default 20 000). |
| `options.only_main_content` | `boolean` |  | Drop navigation, headers and footers (default true). |

### `fetchTavily`

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
| `url` | `list[string]` | ✓ | The pages' URLs, http or https — one or several. |
| `options.max_chars` | `integer` |  | Most characters of text per page (default 20 000). |
| `options.extract_depth` | `string` |  | 'basic' (default); 'advanced' reaches tables and embedded content at twice the price. One of `basic`, `advanced`. |

### `fetchExa`

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
| `url` | `list[string]` | ✓ | The pages' URLs, http or https — one or several. |
| `options.max_chars` | `integer` |  | Most characters of text per page (default 20 000). |
| `options.max_age_hours` | `integer` |  | Use a cached copy younger than this (0 fetches fresh, -1 cache only; at most 720). |

### `fetchParallel`

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
| `url` | `list[string]` | ✓ | The pages' URLs, http or https — one or several. |
| `options.max_chars` | `integer` |  | Most characters of text per page (default 20 000). |
| `options.max_age_seconds` | `integer` |  | Fetch a page live when the cached copy is older than this (600 or more). Slower. |

### `fetchLocal`

A web page's text fetched from this machine and converted to Markdown — free.

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
| `url` | `list[string]` | ✓ | The pages' URLs, http or https — one or several. |
| `options.max_chars` | `integer` |  | Most characters of text per page (default 20 000). |
<!-- /g:tool -->
