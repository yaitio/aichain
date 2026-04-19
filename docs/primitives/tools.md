# Tool

A **Tool** is a Python function with a declared JSON Schema interface. It does something in the real world: search the web, fetch a URL, render PDF, call an external API, convert a file.

Skills reason about text. Tools actually **do** things.

```python
from tools import PerplexitySearchTool

search = PerplexitySearchTool()
result = search(query="latest ERP market analysis Kazakhstan 2025")

if result:
    print(result.output)
```

---

## The contract

Every Tool subclass sets three class-level attributes and implements `run()`:

```python
from tools import Tool

class ReverseTextTool(Tool):
    name        = "reverse_text"
    description = "Reverse the characters in a string."
    parameters  = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to reverse."},
        },
        "required": ["text"],
    }

    def run(self, text: str) -> str:
        return text[::-1]
```

| Attribute | Purpose |
|---|---|
| `name` | Machine-readable identifier. Used as the function name in OpenAI / Anthropic tool-use schemas. |
| `description` | One sentence. This is what an LLM reads to decide whether to use the tool. |
| `parameters` | JSON Schema `object` describing accepted kwargs. |
| `run(**kwargs)` | The actual logic. |

That is the complete surface. Everything else — validation, error wrapping, schema export — comes from the base class.

---

## Two call styles

Tools expose two entry points with different error semantics:

### `tool(**kwargs)` — safe

Returns a `ToolResult`. Never raises.

```python
result = search(query="AI agents")

if result:
    print(result.output)
else:
    print("Error:", result.error)
```

`ToolResult` has three fields:

| Field | Meaning |
|---|---|
| `success: bool` | `True` if `run()` completed without exception. |
| `output: Any` | The raw return value of `run()` (or `None` on failure). |
| `error: str \| None` | Human-readable error message (or `None` on success). |

`bool(result)` returns `result.success`, so `if result:` works naturally.

This is the style used internally by `Chain` and `Agent`.

### `tool.run(**kwargs)` — raw

Returns the bare output; propagates any exception.

```python
raw = search.run(query="AI agents")   # raises on error
```

Use this when you want normal Python error handling.

---

## The parameters schema

The schema must be a JSON Schema `object` with a `"properties"` dict and an optional `"required"` list:

```python
parameters = {
    "type": "object",
    "properties": {
        "source": {
            "type":        "string",
            "description": "File path or URL to process.",
        },
        "output_path": {
            "type":        "string",
            "description": "Optional path to write the result.",
        },
    },
    "required": ["source"],
}
```

Two things use this schema:

1. **`tool.__call__`** — validates that every required key is present in kwargs before calling `run()`.
2. **`Agent`** — serialises it to the provider's tool-use format so the LLM knows how to call the tool.

### OpenAI / Anthropic function-calling

Every Tool exposes a ready-to-use schema for direct use with provider SDKs:

```python
tool.schema()
# {
#   "type": "function",
#   "function": {
#     "name":        "reverse_text",
#     "description": "Reverse the characters in a string.",
#     "parameters":  { ... }
#   }
# }
```

```python
import openai
client = openai.OpenAI()
client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    tools=[tool.schema()],
)
```

---

## Using a Tool in a Chain

Tools slot into a `Chain` like Skills, with one important difference: they receive only the kwargs that match their declared parameters.

```python
from chain import Chain
from tools import MarkItDownTool

fetch = MarkItDownTool()

chain = Chain(steps=[
    (fetch, "content", {"source": "url"}),   # input_map renames accumulated["url"] → source
    (summariser, "summary"),
])

chain.run(variables={"url": "https://example.com/article"})
```

The third tuple element — `input_map` — lets you rename an accumulated variable to match the tool's parameter name:

```python
(fetch, "content", {"source": "url"})
#                    ^      ^
#                    |      accumulated variable to read
#                    tool parameter to fill
```

See [Chain](chain.md) for the full step-tuple syntax.

### Dict-valued tools

A Tool whose `run()` returns a **dict** writes multiple named outputs at once — the dict is merged into the accumulated variables:

```python
class FetchStatsTool(Tool):
    name = "fetch_stats"
    ...
    def run(self, url: str) -> dict:
        return {"stats_raw": "...", "stats_count": 42}

# After this step, both stats_raw and stats_count are in accumulated vars.
```

This is the only way a single step can produce several output keys.

---

## Built-in tools

| Tool | Class | Purpose |
|---|---|---|
| Perplexity search | `PerplexitySearchTool` | Live web search via Perplexity Sonar. |
| Brave search | `BraveSearchTool` | Web search via Brave Search API. |
| SerpAPI | `SerpApiTool` | Google search via SerpAPI. |
| OpenAI web search | `OpenAIWebSearchTool` | Web search via OpenAI Responses API. |
| MarkItDown | `MarkItDownTool` | URL or file → Markdown. |
| Mistletoe | `MistletoeTool` | Markdown → HTML. |
| WeasyPrint | `WeasyPrintTool` | HTML → PDF. |
| DeepL translate | `DeepLTranslateTool` | Translate text via DeepL. |
| DeepL rephrase | `DeepLRephraseTool` | Rephrase text via DeepL. |
| Section context | `SectionContextTool` | Rolling context window for sectional document generation. |
| Late | `LateTool` | Social-media scheduling via Late. |

See **[Tools reference](../tools-reference/)** for per-tool parameters and examples.

---

## Writing a custom Tool

A real-world example — fetch the current weather for a city:

```python
import json
import urllib3

from tools import Tool

class WeatherTool(Tool):
    name        = "get_weather"
    description = "Get the current temperature (°C) for a city."
    parameters  = {
        "type": "object",
        "properties": {
            "city":    {"type": "string", "description": "City name."},
            "country": {"type": "string", "description": "ISO country code."},
        },
        "required": ["city"],
    }

    def __init__(self, api_key: str | None = None):
        import os
        self._api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        if not self._api_key:
            raise ValueError("Set OPENWEATHER_API_KEY or pass api_key=...")
        self._http = urllib3.PoolManager()

    def run(self, city: str, country: str = "") -> dict:
        q   = f"{city},{country}" if country else city
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?q={q}&units=metric&appid={self._api_key}")
        resp = self._http.request("GET", url)
        if resp.status != 200:
            raise RuntimeError(f"Weather API returned {resp.status}")
        data = json.loads(resp.data.decode("utf-8"))
        return {
            "temperature_c": data["main"]["temp"],
            "description":   data["weather"][0]["description"],
        }
```

Three things to notice:

1. **Constructor** — reads its API key from the environment, matching the pattern used by every built-in tool.
2. **Return type** — a dict, so this tool writes both `temperature_c` and `description` into a Chain's accumulated variables in one step.
3. **Errors** — raised as exceptions. The `__call__` wrapper catches them and returns a failed `ToolResult`.

---

## See also

- **Use a Tool in a pipeline** → [Chain](chain.md)
- **Let an LLM choose tools** → [Agent](../agents/overview.md)
- **All built-in tools** → [Tools reference](../tools-reference/)
