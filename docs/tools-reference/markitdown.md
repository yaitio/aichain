<!-- g:tool -->
# `convertToMD` — `MarkItDownTool`

Convert a file or URL to Markdown text.

| | |
|---|---|
| Import | `from yait_aichain.tools import MarkItDownTool` |
| Risk class | `write` |
| Also exported as | `convertToMD` |

```python
MarkItDownTool(
    llm_client: Any = None,
    llm_model: str | None = None,
    enable_builtins: bool | None = None,
    enable_plugins: bool | None = None,
    timeout: float | tuple[float, float] | None = (10.0, 30.0),
    headers: dict | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Absolute or relative file path, or a URL, to convert to Markdown. |
| `options.output_path` | `string` |  | Optional file path to write the result. Parent dirs created automatically. |
| `options.metadata` | `boolean` |  | For a web page, start the result with its title, publication date, author and canonical URL as front matter. Default false. |

```python
result = MarkItDownTool()(
    input="…",
    options={"output_path": …},
)
```
<!-- /g:tool -->

Converts **any file or URL to Markdown** with Microsoft's
[MarkItDown](https://github.com/microsoft/markitdown). The standard companion to
any search tool that returns links.

```python
from yait_aichain.tools import MarkItDownTool

tool     = MarkItDownTool()
markdown = tool.run(input="report.pdf")
```

---

## Supported inputs

| Category | Formats |
|---|---|
| Documents | PDF, DOCX, PPTX, XLSX, XLS, ODP, ODT, ODS |
| Web | HTML pages, URLs |
| Text / data | Plain text, CSV, JSON, XML, YAML, RST, EPUB |
| Code | Any source file → fenced code block |
| Images | JPEG, PNG, GIF, BMP, TIFF, WEBP *(LLM client required for descriptions)* |
| Audio | WAV, MP3, M4A, FLAC, OGG *(LLM client required for transcription)* |
| Archives | ZIP (recursively converts contents) |

---

## Installation

```bash
pip install "yait-aichain[convert]"     # or: pip install markitdown
```

No key required. Without `markitdown` installed, the tool raises `ImportError`
on first use.

---

## Usage

### Basic (no LLM)

```python
tool = MarkItDownTool()

# Call-style — errors captured in ToolResult
result = tool(input="slides.pptx")
if result:
    print(result.output)

# Run-style — raises on error
markdown = tool.run(input="https://example.com/article")

# Save to a file as well
tool.run(input="data.xlsx", options={"output_path": "exports/data.md"})
```

### With an LLM (image descriptions, audio transcription)

```python
import openai
from yait_aichain.tools import MarkItDownTool

tool   = MarkItDownTool(llm_client=openai.OpenAI(), llm_model="gpt-4o")
result = tool(input="architecture_diagram.png")
```

### In a Chain — fetch, then summarise

```python
from yait_aichain.chain import Chain
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.tools import MarkItDownTool

summariser = Skill(Model("gpt-4o-mini"), prompt="Summarise in 200 words:\n\n{article}")

chain = Chain(steps=[
    (MarkItDownTool(), "article", {"input": "url"}),
    (summariser,       "summary"),
])

chain.run(variables={"url": "https://example.com/article"})
```

### In an Agent

With a search tool this is the canonical two-tool research loop:

```python
from yait_aichain.agent import Agent, step_count
from yait_aichain.models import Model
from yait_aichain.tools import BraveSearchTool, MarkItDownTool

agent = Agent(
    Model("claude-opus-4-6"),
    tools     = [BraveSearchTool(), MarkItDownTool()],
    stop_when = [step_count(10)],
)
```

### Web pages: headers, timeout and the date

MarkItDown fetches a URL with `requests` and keeps only the page body. Three
things the tool adds for the web:

```python
tool = MarkItDownTool(
    timeout=(10, 30),                         # connect, read — the default
    headers={"User-Agent": "Mozilla/5.0 …"},  # sent with every request
)
text = tool.run(input="https://blogs.lse.ac.uk/businessreview/…",
                options={"metadata": True})
tool.last_metadata
# {"title": "…", "published": "2026-09-28T09:26:06+01:00",
#  "author": "…", "site": "LSE Business Review", "canonical": "…", "url": "…"}
```

- **`timeout`** — without one a server that holds the connection and never
  answers hangs the call (www.mckinsey.com, 2026-10-06). `None` waits forever.
- **`headers`** — sites disagree on who they let in, so this is per site and
  the default is unchanged. Measured 2026-10-07: blogs.lse.ac.uk answers 403 to
  the default `python-requests` agent and 200 to a browser's; www.bcg.com does
  the reverse.
- **`last_metadata`** — what the page's `<head>` says: JSON-LD first, then
  Open Graph, Dublin Core and the other common meta tags. A field the page
  does not carry is left out. MarkItDown drops the head, and on some sites
  (library.hbs.edu) the publication date is nowhere else. `{"metadata": true}`
  also puts it on top of the Markdown as front matter, for a model to read.

---

## Notes

- The `MarkItDown` instance is created on first `run()` and reused.
- Parent directories for `output_path` are created as needed.
- Output paths are confined by `AICHAIN_OUTPUT_ROOT` when it is set.

---

## See also

- [Search tools](search.md): `searchPerplexity`, `searchBrave`, `searchSerp`, `searchExa`
- [`convertToHTML`](mistletoe.md) — the other direction: Markdown → HTML / LaTeX.
