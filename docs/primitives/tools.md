# Tool

A **Tool** is a Python callable with a declared JSON-Schema interface that does
something in the real world: search the web, fetch and convert a file, embed
text, call a REST API, pause a run for approval. Skills reason about text; Tools
act.

---

## Quick start

Use a built-in tool, or define your own in a few lines:

```python
import os
from yait_aichain.tools import PerplexitySearchTool, Tool

# built-in — call it, get a ToolResult (never raises)
search = PerplexitySearchTool()                 # reads PERPLEXITY_API_KEY
result = search(input="best managed vector databases 2025")
if result:
    print(result.output)

# custom — name + description + parameters + run()
class ReverseTool(Tool):
    name        = "reverse"
    description = "Reverse a string."
    parameters  = {"type": "object",
                   "properties": {"input": {"type": "string"}},
                   "required": ["input"]}
    def run(self, input, options=None):
        return input[::-1]

print(ReverseTool()(input="hello").output)       # 'olleh'
```

▶ Built-in convert tool: [`examples/05_tool_convert.py`](../../examples/05_tool_convert.py) ·
Custom tool: [`examples/07_tool_custom.py`](../../examples/07_tool_custom.py) ·
Deep dive ↓

---

## Common gotchas

- **Two call styles, different error behaviour.** `tool(...)` returns a
  `ToolResult` and **never raises**; `tool.run(...)` returns the bare value and
  **propagates** exceptions. Chains and Agents use the safe form internally.
- **`bool(result)` is `result.success`**, so `if result:` works — but a failed
  call is *falsy*, so don't forget the `else`.
- **The standard shape is `run(self, input, options=None)`** — one primary
  input. Tools that need several named fields declare them in `parameters` and
  take them as kwargs (see [Tool shapes](#tool-shapes)); those are driven by
  Chain/Agent, not by the single-input `tool(input=…)` form.
- **A `dict` return merges** into a Chain's accumulated variables (many keys at
  once); a `str` return is stored under one key.

---

## Reference

### The contract

A Tool sets three class attributes and implements `run()`:

```python
from yait_aichain.tools import Tool

class WordCountTool(Tool):
    name        = "word_count"                       # function name in tool-use schemas
    description = "Count the words in a text."        # what an LLM reads to decide to use it
    parameters  = {                                   # JSON Schema for the arguments
        "type": "object",
        "properties": {"input": {"type": "string", "description": "Text to count."}},
        "required": ["input"],
    }
    def run(self, input, options=None) -> int:
        return len(input.split())
```

Validation, error wrapping, and schema export come from the base class — you
write only `run()`.

### Tool shapes

| Shape | `run` signature | Called as | Use for |
|---|---|---|---|
| **Single-input** | `run(self, input, options=None)` | `tool(input=…)` / `tool.run(input=…)` | Most tools — one main argument plus an optional `options` dict. All built-ins use this. |
| **Multi-parameter** | `run(self, a, b, …)` | inside a Chain/Agent (matched kwargs), or `tool.run(a=…, b=…)` | Tools that produce several named outputs or need several inputs. |

In a Chain/Agent, the engine reads the tool's declared `parameters`, pulls the
matching values from the accumulated variables, and calls `run(**kwargs)`. The
single-input safe form `tool(input, options)` is the convenience for standalone
use.

### Two call styles

```python
result = tool(input="…")        # safe  → ToolResult; never raises
if result: print(result.output)
else:      print(result.error)

raw = tool.run(input="…")       # raw   → bare output; raises on error
```

`ToolResult` has `success: bool`, `output: Any`, `error: str | None`, and is
truthy when `success`.

### Provider function-calling schema

Every tool exposes an OpenAI/Anthropic-ready schema via `tool.schema()` — useful
if you drive a provider SDK directly:

```python
tool.schema()
# {"type": "function", "function": {"name": …, "description": …, "parameters": {…}}}
```

An [Agent](../agents/overview.md) does this for you.

### Using a Tool in a Chain

A tool slots into a Chain step. It receives only the kwargs matching its
declared `parameters`; `input_map` renames an accumulated variable to a
parameter name:

```python
from yait_aichain.chain  import Chain
from yait_aichain.tools  import MarkItDownTool

chain = Chain(steps=[
    (MarkItDownTool(), "content", {"source": "url"}),   # accumulated["url"] → param "source"
    (summariser,       "summary"),
])
chain.run(variables={"url": "https://example.com/article"})
```

A tool whose `run()` returns a **dict** writes several named outputs at once —
the dict is merged into the accumulated variables. See [Chain](chain.md) for the
full step syntax.

### Built-in tools

| Group | Tools |
|---|---|
| **Web search** | `PerplexitySearchTool`, `BraveSearchTool`, `OpenAIWebSearchTool`, `SerpApiTool` (functional forms: `searchPerplexity`, `searchBrave`, `searchOpenAI`, `searchSerp`) |
| **Convert** | `convertToMD` / `MarkItDownTool` (→ Markdown), `convertToHTML` / `MistletoeTool` (→ HTML), `convertToPDF` / `WeasyprintTool` (→ PDF), `convertToText` |
| **Speech** | `convertToSpeech`, `TTS` (`ttsOpenAI/Google/XAI/Qwen`), `STT` (`sttOpenAI/Google/XAI/Qwen`) |
| **Embeddings** | `Embedding` + `EmbeddingOpenAI/Cohere/Voyage/Google/Qwen` |
| **Vector DB** | `VectorDB`, `VectorStore` |
| **HTTP** | `RestApiTool` — call any REST endpoint as a tool |
| **Suspend** | `Wait`, `Gate` — pause a run for an external signal ([State](state.md)) |

Per-tool parameters and examples: [Tools reference](../tools-reference/).

### Wait & Gate

`Wait` and `Gate` are tools that *pause* a run instead of returning immediately —
the basis of suspend/resume. Drop a `Wait` into a Chain to pause for human
input, or wrap any tool in a `Gate` to require approval before it runs. Full
treatment in [State — suspend & resume](state.md).

### Writing a custom Tool

A multi-parameter tool that fetches weather (returns a dict, so it writes two
variables in one Chain step):

```python
import os, json, urllib3
from yait_aichain.tools import Tool

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

    def __init__(self, api_key=None):
        self._api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        if not self._api_key:
            raise ValueError("Set OPENWEATHER_API_KEY or pass api_key=...")
        self._http = urllib3.PoolManager()

    def run(self, city, country="") -> dict:
        q = f"{city},{country}" if country else city
        r = self._http.request("GET",
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={q}&units=metric&appid={self._api_key}")
        if r.status != 200:
            raise RuntimeError(f"Weather API returned {r.status}")
        d = json.loads(r.data.decode())
        return {"temperature_c": d["main"]["temp"], "description": d["weather"][0]["description"]}
```

Notes: the constructor reads its key from the environment (the built-in
pattern); `run` declares its parameters as kwargs (so Chain/Agent can call it);
and it raises on error — the safe `__call__` wrapper turns that into a failed
`ToolResult`.

---

## See also

- [Chain](chain.md) — use Tools as pipeline steps.
- [Agent](../agents/overview.md) — let an LLM choose and call tools.
- [State](state.md) — `Wait` / `Gate` and suspend/resume.
- [Tools reference](../tools-reference/) — every built-in tool in detail.
