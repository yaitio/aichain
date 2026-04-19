# `markitdown` — `MarkItDownTool`

Convert **any file or URL to Markdown** using Microsoft's [MarkItDown](https://github.com/microsoft/markitdown) library. The standard companion to any link-returning search tool.

```python
from tools import MarkItDownTool

tool     = MarkItDownTool()
markdown = tool.run(source="report.pdf")
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
pip install markitdown
```

No env var required. If `markitdown` isn't installed, the tool raises `ImportError` on first use.

---

## Constructor

```python
MarkItDownTool(
    llm_client:      Any        = None,
    llm_model:       str | None = None,
    enable_builtins: bool | None = None,
    enable_plugins:  bool | None = None,
)
```

- `llm_client` — OpenAI-compatible client used to describe images and transcribe audio.
- `llm_model` — model name for that client, e.g. `"gpt-4o"`. Required when `llm_client` is set.
- `enable_builtins` / `enable_plugins` — passed through to `MarkItDown(...)`.

---

## Parameters

| Name | Type | Required | Notes |
|---|---|---|---|
| `source` | `string` | ✓ | File path (absolute or relative) **or** URL. |
| `output_path` | `string` | | Save Markdown to this path. Parent directories are created. |

---

## Usage

### Basic (no LLM)

```python
tool = MarkItDownTool()

# Call-style — errors captured in ToolResult
result = tool(source="slides.pptx")
if result:
    print(result.output)

# Direct — raises on error
markdown = tool.run(source="https://example.com/article")

# Save directly
tool.run(source="data.xlsx", output_path="exports/data.md")
```

### With LLM (image descriptions / audio transcription)

```python
import openai
from tools import MarkItDownTool

tool = MarkItDownTool(
    llm_client = openai.OpenAI(),
    llm_model  = "gpt-4o",
)
result = tool(source="architecture_diagram.png")
```

### In a Chain — search → fetch → summarise

```python
from chain import Chain
from tools import BraveSearchTool, MarkItDownTool
from skills import Skill
from models import Model

summariser = Skill(
    model  = Model("gpt-4o-mini"),
    input  = {"messages": [{"role": "user", "parts": [
        {"type": "text", "text": "Summarise in 200 words:\n\n{article}"}
    ]}]},
    output = {"modalities": ["text"], "format": {"type": "text"}},
)

chain = Chain(steps=[
    (BraveSearchTool(),  "search_results"),
    # (orchestrator step picks a URL — in an agent; for a chain you'd
    # structure the search tool to return a specific URL variable.)
    (MarkItDownTool(),   "article", {"source": "target_url"}),
    (summariser,         "summary"),
])
```

### In an Agent

Together with a search tool, this is the canonical 2-tool research loop:

```python
from agent import Agent
from models import Model
from tools import BraveSearchTool, MarkItDownTool

agent = Agent(
    orchestrator = Model("claude-opus-4-6"),
    tools        = [BraveSearchTool(), MarkItDownTool()],
    mode         = "agile",
    max_steps    = 10,
)
```

---

## Notes

- The `MarkItDown` instance is lazy-initialised on first `run()` and reused.
- Parent directories for `output_path` are created with `os.makedirs(..., exist_ok=True)`.
- Pairs naturally with any of the search tools — feed the URL returned in search output straight into `source=`.

---

## See also

- Search tools: [`brave_search`](brave-search.md), [`serp_api_search`](serp-api.md), [`perplexity_search`](perplexity-search.md)
- [`mistletoe`](mistletoe.md) — go the other direction: Markdown → HTML / LaTeX.
