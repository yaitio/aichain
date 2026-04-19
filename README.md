# aichain

**One interface. Eight providers. Text, images, agents.**

Route any AI task — chat, reasoning, image generation, vision, web search, speech, embeddings — to OpenAI, Anthropic, Google, xAI, Perplexity, Kimi, DeepSeek, or Qwen through a single Python library. Swap models without touching your logic. Build agents and pipelines in minutes.

```python
from models import Model
from skills import Skill

skill = Skill(
    model  = Model("claude-opus-4-6"),       # swap to "gpt-4o" with one change
    input  = {"messages": [{"role": "user", "parts": [
        {"type": "text", "text": "Summarise this in three bullets:\n\n{text}"}
    ]}]},
    output = {"modalities": ["text"], "format": {"type": "text"}},
)

result = skill.run(variables={"text": "..."})
```

---

## Why aichain?

**One model interface for every provider.** OpenAI, Anthropic, Google AI, xAI, Perplexity, Kimi, DeepSeek, and Qwen all speak different APIs. `Model("gpt-4o")`, `Model("claude-opus-4-6")`, `Model("gemini-2.5-pro")` — same call, same output, same downstream code.

**Universal reasoning.** Set `reasoning="high"` once; the library maps it to `budget_tokens` for Anthropic, `thinkingBudget` for Google, `reasoning_effort` for OpenAI and xAI, `thinking` for Kimi, `enable_thinking` for Qwen, and a model switch for DeepSeek.

**Built for pipelines.** Chain skills, tools, and agents together. Each step's output feeds the next. Mix models freely across steps.

**Agents that actually reflect.** The built-in Agent loop plans, acts, and reflects on every step — choosing to continue, retry, replan, or stop. `"agile"` mode lets it revise the plan mid-run.

---

## Install

```bash
pip install aichain
```

Install extras for the tools you need:

```bash
pip install markitdown          # file/URL → Markdown
pip install weasyprint          # HTML → PDF
pip install mistletoe           # Markdown → HTML / LaTeX
pip install pyyaml              # save & load Skills and Chains
pip install fastmcp             # MCP server integration
```

Set the API keys for the providers you use:

```bash
export OPENAI_API_KEY="sk-…"
export ANTHROPIC_API_KEY="sk-ant-…"
export GOOGLE_AI_API_KEY="AIza…"
export XAI_API_KEY="xai-…"
export PERPLEXITY_API_KEY="pplx-…"
export MOONSHOT_API_KEY="sk-…"        # Kimi
export DEEPSEEK_API_KEY="sk-…"        # DeepSeek
export DASHSCOPE_API_KEY="sk-…"       # Qwen (Alibaba DashScope)
```

---

## Eight providers, one syntax

```python
from models import Model

gpt      = Model("gpt-4.1")
claude   = Model("claude-sonnet-4-6")
gemini   = Model("gemini-2.5-flash")
grok     = Model("grok-3")
sonar    = Model("sonar-pro")
kimi     = Model("kimi-k2.5")
deepseek = Model("deepseek-chat")
qwen     = Model("qwen-max")
```

Full model list: [model registry →](docs/reference/model-registry.md)

---

## Core concepts

### Skill — run a prompt against any model

```python
from models import Model
from skills import Skill

translator = Skill(
    model  = Model("gpt-4.1-mini"),
    input  = {"messages": [{"role": "user", "parts": [
        {"type": "text", "text": "Translate to {language}:\n\n{text}"}
    ]}]},
    output = {"modalities": ["text"], "format": {"type": "text"}},
)

result = translator.run(variables={"text": "Hello world", "language": "French"})
# → "Bonjour le monde"
```

### Chain — wire steps together

```python
from chain import Chain

chain = Chain(steps=[
    (search_skill,    "search_results"),
    (convert_skill,   "article",   {"source": "target_url"}),
    (summarise_skill, "summary"),
])

result = chain.run(variables={"query": "fusion energy 2025"})
```

### Agent — plan, act, reflect

```python
from agent import Agent
from models import Model
from tools import PerplexitySearchTool, MarkItDownTool

agent = Agent(
    orchestrator = Model("claude-opus-4-6"),
    tools        = [PerplexitySearchTool(), MarkItDownTool()],
    mode         = "agile",
    max_steps    = 10,
)

result = agent.run("Compare the top 3 vector databases in 2025.")
if result:
    print(result.output)
    print(f"Used {result.tokens_used:,} tokens in {result.steps_taken} steps.")
```

Two execution modes:
- **`waterfall`** — fixed plan, executes in order; reflection can retry or stop on fatal failure.
- **`agile`** — same structure, but reflection can revise the remaining plan mid-run.

### Universal reasoning

```python
from models import Model

# Same option — each provider handles it natively
model = Model("claude-opus-4-6", options={"reasoning": "high"})
model = Model("gpt-4o",          options={"reasoning": "medium"})
model = Model("gemini-2.5-pro",  options={"reasoning": "low"})
```

### Image generation

```python
from models import Model
from skills import Skill

skill = Skill(
    model         = Model("dall-e-3"),          # or "gpt-image-1", "wanx2.1-t2i-turbo"
    input         = {"messages": [{"role": "user", "parts": [
        {"type": "text", "text": "A photorealistic red apple on a wooden table"}
    ]}]},
    output_format = "image",
)

result = skill.run()
# result → {"base64": "…", "mime_type": "image/png", "url": None, "revised_prompt": "…"}
```

### Vision (image input)

```python
skill = Skill(
    model  = Model("gpt-4o"),
    input  = {"messages": [{"role": "user", "parts": [
        {"type": "image", "source": {"kind": "url", "url": "https://…/photo.jpg", "mime": "image/jpeg"}},
        {"type": "text",  "text": "What is in this image?"},
    ]}]},
    output_format = "text",
)
```

---

## Built-in tools

### Search

| Tool | What it does |
|---|---|
| `PerplexitySearchTool` | Web search with rich content snippets |
| `BraveSearchTool` | Web search — ranked links |
| `SerpApiTool` | 50+ search engines via SerpAPI |
| `OpenAIWebSearchTool` | Synthesised answers via OpenAI Responses API |

### Convert

| Tool | What it does |
|---|---|
| `MarkItDownTool` | PDF, DOCX, PPTX, URLs → Markdown |
| `MistletoeTool` | Markdown → HTML / LaTeX |
| `WeasyprintTool` | HTML → PDF |
| `TTS(provider)` | Text → audio file (OpenAI, Google, xAI, Qwen) |
| `STT(provider)` | Audio file → transcript (OpenAI, Google, xAI, Qwen) |

### Embeddings

```python
from tools import Embedding

embedder = Embedding("text-embedding-3-small")   # or "embed-english-v3.0", "voyage-3", …
result   = embedder.run("The quick brown fox")
# result.vector → list[float]
```

Providers: OpenAI · Cohere · Voyage · Google · Qwen

### Services

| Tool | What it does |
|---|---|
| `DeepLTranslateTool` | Translate text (30+ languages) |
| `DeepLRephraseTool` | Rephrase for style and tone |
| `LateAccountsTool` | List connected social accounts (Late API) |
| `LatePublishTool` | Publish to 14 social platforms |
| `ImgbbUploadTool` | base64 image → permanent public HTTPS URL |

### Infrastructure

| Tool | What it does |
|---|---|
| `RestApiTool` | Universal REST endpoint — drop any API call into a Chain or Agent |
| `MCPTool` / `MCPTools` | Model Context Protocol — connect any MCP server's tools |
| `SectionContextTool` | Rolling-context queue manager for long-document generation |

---

## Persist and reload

```python
# Save a skill
skill.save("skills/translator.yaml")

# Load anywhere — API keys come from env vars, not the file
from skills import Skill
skill = Skill.load("skills/translator.yaml")

# Same for chains
chain.save("chains/research_pipeline.yaml")
chain = Chain.load("chains/research_pipeline.yaml")
```

---

## Testing

The library ships with a full unit-test suite — no real API keys required for unit tests; live integration tests are skipped unless real keys are present.

```bash
python3 -m pytest tests/clients/ tests/models/ tests/skills/ tests/chain/ tests/agent/
```

**Coverage:**

| Area | Tests | What's covered |
|---|---|---|
| Clients (8 providers) | auth headers, base URLs, `list_models` parsing, invalid-key rejection | `tests/clients/` |
| Models — text-to-text | factory routing, `to_request` / `from_response` per provider, JSON output, reasoning | `tests/models/` |
| Models — text-to-image | DALL-E 3, GPT-Image-1, Grok-Imagine, Gemini image, Wanx | `tests/models/` |
| Models — image-to-text | Vision messages (URL + base64) for all 6 vision providers | `tests/models/` |
| Models — features | Reasoning translation, JSON schema, registry helpers, MIME detection | `tests/models/` |
| Skill | Init, `run()`, variable substitution, multi-provider, no-mutation | `tests/skills/` |
| Chain | Init, step execution, variable flow, history, precedence, reset | `tests/chain/` |
| Agent | Memory + FileBackend, AgentResult, init, helpers, `_execute_action`, `run()`, agile replan, spawn | `tests/agent/` |

---

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started/index.md) | Install, quickstart, core concepts |
| [Primitives](docs/primitives/models.md) | Model, Skill, Tool, Chain |
| [Agents](docs/agents/overview.md) | Agent loop, memory, configuration |
| [Tools reference](docs/tools-reference/index.md) | All built-in tools |
| [Reference](docs/reference/model-registry.md) | Model registry, env vars, YAML schema |
| [Cookbooks](docs/cookbooks/index.md) | Research agent, long-doc, translate & publish, RAG, and more |

---

## Supported providers and models

| Provider | Text | Vision | Image gen | Reasoning | Env var |
|---|---|---|---|---|---|
| **OpenAI** | GPT-5, GPT-4.1, GPT-4o, o3, o4-mini | ✓ | DALL-E 3, GPT-Image-1 | `reasoning_effort` | `OPENAI_API_KEY` |
| **Anthropic** | Claude Opus / Sonnet / Haiku 4 | ✓ | — | `budget_tokens` | `ANTHROPIC_API_KEY` |
| **Google AI** | Gemini 2.5 Pro / Flash, 2.0 Flash | ✓ | Gemini image models | `thinkingBudget` | `GOOGLE_AI_API_KEY` |
| **xAI** | Grok 4, Grok 3 | ✓ | Grok-Imagine | `reasoning_effort` | `XAI_API_KEY` |
| **Perplexity** | Sonar Pro, Sonar, Deep Research | — | — | built-in | `PERPLEXITY_API_KEY` |
| **Kimi** | K2.5, K2, K2 Turbo, K2 Thinking | ✓ K2.5 | — | `thinking` | `MOONSHOT_API_KEY` |
| **DeepSeek** | DeepSeek-V3 (chat), DeepSeek-R1 (reasoner) | — | — | model switch | `DEEPSEEK_API_KEY` |
| **Qwen** | Qwen-Max, Qwen3, QwQ | ✓ QwenVL | Wanx image models | `enable_thinking` | `DASHSCOPE_API_KEY` |

---

## License

MIT
