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
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Absolute or relative file path, or a URL, to convert to Markdown. |
| `options.output_path` | `string` |  | Optional file path to write the result. Parent dirs created automatically. |

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

---

## Notes

- The `MarkItDown` instance is created on first `run()` and reused.
- Parent directories for `output_path` are created as needed.
- Output paths are confined by `AICHAIN_OUTPUT_ROOT` when it is set.

---

## See also

- Search tools: [`searchBrave`](brave-search.md), [`searchSerp`](serp-api.md), [`searchPerplexity`](perplexity-search.md)
- [`convertToHTML`](mistletoe.md) — the other direction: Markdown → HTML / LaTeX.
