# aichain

**The simplest way to build AI pipelines. 8 providers. 1 interface. Zero lock-in.**

```python
from yait_aichain import Model, Skill

skill = Skill(
    model  = Model("claude-sonnet-4-6"),   # change this one word to switch providers
    input  = {"messages": [{"role": "user", "parts": ["Summarise: {text}"]}]},
)

result = skill.run(variables={"text": "..."})
```

Change `"claude-sonnet-4-6"` to `"gpt-4o"`, `"gemini-2.5-pro"`, or `"grok-3"` — nothing else changes.

---

## Why aichain?

Every major AI library makes you choose: LangChain is too complex, LlamaIndex is RAG-only, CrewAI is agents-only, AutoGen requires a PhD to configure.

**aichain covers the full stack with the simplest interface:**

| Need | aichain primitive |
|---|---|
| Call any LLM | `Skill` |
| Chain steps together | `Chain` |
| Run tasks in parallel | `Pool` |
| Autonomous reasoning | `Agent` |
| Vector search + RAG | `VectorDB` + `vectorQuery` |
| Rerank results | `Reranker` |
| Call any tool or MCP server | `Tool` / `MCPTools` |

All of these work identically across **57 models from 8 providers** — with one line to swap any of them.

---

## Install

```bash
pip install yait-aichain
```

Optional extras:

```bash
pip install markitdown   # file/URL → Markdown
pip install pyyaml       # save & load Skills and Chains
pip install fastmcp      # MCP server integration
```

API keys — only for the providers you use:

```bash
export ANTHROPIC_API_KEY="sk-ant-…"
export OPENAI_API_KEY="sk-…"
export GOOGLE_AI_API_KEY="AIza…"
export XAI_API_KEY="xai-…"
export PERPLEXITY_API_KEY="pplx-…"
export MOONSHOT_API_KEY="sk-…"      # Kimi
export DEEPSEEK_API_KEY="sk-…"
export DASHSCOPE_API_KEY="sk-…"     # Qwen
export COHERE_API_KEY="…"           # embeddings + reranking
export VOYAGE_API_KEY="…"           # embeddings + reranking
```

---

## 8 providers, one syntax

```python
from yait_aichain.models import Model

Model("claude-sonnet-4-6")   # Anthropic
Model("gpt-4o")              # OpenAI
Model("gemini-2.5-flash")    # Google
Model("grok-3")              # xAI
Model("sonar-pro")           # Perplexity
Model("kimi-k2.5")           # Kimi
Model("deepseek-chat")       # DeepSeek
Model("qwen-max")            # Qwen
```

57 models total. Full list: [model registry →](docs/reference/model-registry.md)

---

## Core concepts

### Skill — one prompt, any model

```python
skill  = Skill(model=Model("gpt-4o-mini"), input={...})
result = skill.run(variables={"topic": "neural networks"})
```

### Chain — sequential steps, automatic variable flow

```python
chain = Chain(steps=[
    (fetch_tool,    "page"),
    (summarise,     "summary"),
    (translate,     "result"),
])
result = chain.run(variables={"url": "https://…", "language": "French"})
print(chain.history)   # full audit trail
```

### Pool — parallel execution

```python
pool    = Pool(summarise_skill, items=[{"text": t} for t in documents], max_flows=10)
results = pool.run()                    # all documents processed simultaneously

print(pool.status)   # {PENDING: 0, RUNNING: 0, DONE: 50, FAILED: 0}
print(pool.history)  # per-item: status, output, error, duration
```

### Agent — autonomous reasoning

```python
agent  = Agent(
    orchestrator = Model("claude-opus-4-6"),
    tools        = [searchPerplexity(), convertToMD()],
    mode         = "agile",
    max_steps    = 10,
)
result = agent.run("Compare the top 3 vector databases.")
print(result.output)
print(f"steps={result.steps_taken}  tokens={result.tokens_used:,}")
```

### Full RAG pipeline

```python
from yait_aichain.tools.embedding import Embedding
from yait_aichain.tools.vectordb  import VectorDB, vectorChunk, vectorUpsert, vectorQuery
from yait_aichain.tools.reranking import Reranker

store    = VectorDB("chroma", "docs", embedder=Embedding("cohere/embed-v4.0"))
reranker = Reranker("cohere/rerank-v3.5")

# Ingest
chunks = vectorChunk(max_chars=800).run(my_document)
vectorUpsert(store).run([{"id": f"c{i}", **c} for i, c in enumerate(chunks)])

# Query → rerank → answer
pipeline = Chain(steps=[
    (vectorQuery(store),  "candidates", {"input": "{question}", "options": {"n": 20}}),
    (reranker,            "context",    {"input": "{candidates}",
                                         "options": {"query": "{question}", "top_n": 5}}),
    answer_skill,
])
answer = pipeline.run(variables={"question": "How does KV caching work?"})
```

Vector DB providers: **Chroma · Qdrant · Pinecone**
Reranking providers: **Cohere · Voyage · Qwen**

---

## Cost, routing & resilience

These are all **opt-in** — the minimal program above is unchanged.

**Token usage & cost** — every result carries normalised usage:

```python
skill.run()
skill.last_usage.input_tokens   # 1240
skill.last_usage.total_tokens   # 1310
skill.last_usage.cost           # 0.0032  (USD, None if the model has no price)

chain.run()
chain.last_usage.cost           # summed across all Skill steps
```

**Explicit provider routing** — pick the provider with a `provider/model`
prefix; it also unlocks custom / fine-tuned names the auto-detector can't
recognise:

```python
Model("openai/gpt-4o")                 # same as Model("gpt-4o")
Model("openai/ft:gpt-4o:acme:42")      # custom name, explicit provider
```

**Fallback chain** — pass a list; a transient failure (rate limit / server /
network) advances to the next model. A real error (bad key, bad request)
still raises immediately:

```python
skill = Skill(
    model = [Model("claude-sonnet-4-6"), Model("gpt-4o")],  # primary, then backup
    input = {"messages": [{"role": "user", "parts": ["..."]}]},
)
```

**Typed errors** — catch a specific failure mode, or `APIError` for all:

```python
from yait_aichain import RateLimitError, AuthenticationError, APIError

try:
    skill.run()
except RateLimitError as e:
    wait(e.retry_after)        # honours the Retry-After header
except AuthenticationError:
    ...
```

**Keep the model list fresh** — diff the registry against a provider's live
roster:

```python
from yait_aichain.models import registry
registry.refresh("openai")     # → {"new": [...], "removed": [...], ...}
```

---

## Built-in tools

### Search
`searchPerplexity` · `searchBrave` · `searchSerp` · `searchOpenAI`

### Convert
`convertToMD` · `convertToHTML` · `convertToPDF` · `TTS(provider)` · `STT(provider)`

### Embeddings
```python
Embedding("openai/text-embedding-3-small")
Embedding("cohere/embed-v4.0")
Embedding("voyage/voyage-3-large")
```

---

## Examples

→ **[examples/](examples/README.md)** — 16 focused examples, one concept each

| # | File | What it shows |
|---|---|---|
| 01 | `01_skill.py` | The minimum viable aichain program |
| 02 | `02_skill_models.py` | Same prompt, Claude + GPT + Gemini |
| 03 | `03_skill_multimodal.py` | Text → image → vision, three providers |
| 04 | `04_skill_save_load.py` | Save to YAML, reload anywhere |
| 05 | `05_tool_convert.py` | URL → Markdown |
| 06 | `06_tool_mcp.py` | Connect an MCP server, discover + call tools |
| 07 | `07_tool_custom.py` | Build your own Tool, plug into a Chain |
| 08 | `08_chain.py` | GPT writes → Claude reviews |
| 09 | `09_chain_tool_skill.py` | Fetch page → summarise |
| 10 | `10_chain_save_load.py` | Save/reload a full pipeline |
| 11 | `11_pool.py` | 5 topics, all in parallel |
| 12 | `12_pool_chain.py` | Chain-per-item, all in parallel |
| 13 | `13_agent.py` | Autonomous agent, one tool |
| 14 | `14_agent_tools.py` | Agent picks its own tools |
| 15 | `15_agent_orchestrator.py` | Orchestrator spawns sub-agents |
| 16 | `16_debug.py` | Inspect Chain history, Pool status, Agent steps |

---


## Persist and reload

```python
skill.save("skills/translator.yaml")
skill = Skill.load("skills/translator.yaml")   # API key from env, not file

chain.save("chains/research.yaml")
chain = Chain.load("chains/research.yaml")
```

---

## Supported providers

| Provider | Text | Vision | Image gen | Env var |
|---|---|---|---|---|
| **Anthropic** | Claude Opus / Sonnet / Haiku 4 | ✓ | — | `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT-5.5, GPT-5.4, GPT-4o | ✓ | GPT-Image-1 | `OPENAI_API_KEY` |
| **Google** | Gemini 2.5 Pro / Flash, 3.x | ✓ | Gemini image models | `GOOGLE_AI_API_KEY` |
| **xAI** | Grok 4, Grok 3 | ✓ | Grok-Imagine | `XAI_API_KEY` |
| **Perplexity** | Sonar Pro, Sonar, Deep Research | — | — | `PERPLEXITY_API_KEY` |
| **Kimi** | K2.5, K2, K2 Turbo, K2 Thinking | ✓ | — | `MOONSHOT_API_KEY` |
| **DeepSeek** | DeepSeek-V3, DeepSeek-R1 | — | — | `DEEPSEEK_API_KEY` |
| **Qwen** | Qwen-Max, Qwen3, QwQ | ✓ | Wanx image models | `DASHSCOPE_API_KEY` |

**Embedding:** OpenAI · Cohere · Voyage · Google · Qwen  
**Reranking:** Cohere · Voyage · Qwen  
**Vector DB:** Chroma · Qdrant · Pinecone

---

## License

MIT

---

## Changelog

### 1.2.3

Two-tier model layer: **format is code (by API family), provider is data.**
Changing provider — one word in `Model("…")` — changes only data, never the
request format or any model code. Behaviour is byte-for-byte unchanged
(guarded by characterisation tests).

- `Model` is a single, thin, data-driven class: it resolves the provider from
  the name and delegates the wire format to the matching family client. The
  provider is exposed as `model._provider`.
- Provider settings, model capabilities and prices live in data — one file per
  provider under `models/providers/*.toml`.
- The protocol layer (`clients/`) owns the wire format. Five family clients
  cover all eight providers: `OpenAIClient` (openai, xai, kimi, deepseek),
  `PerplexityClient`, `QwenClient`, `AnthropicClient`, `GoogleClient`.
- Removed the per-provider `Model` subclasses and client classes; use
  `Model("name")` (or an explicit `provider/` prefix for custom names).
- Fixed Qwen region endpoints (`us`, `hk`) and base URL.

### 1.2.1

- Removed leaked application code from the library (`skills/summarise.py`,
  `tools/section_context.py`) — an app-specific pipeline that did not belong in
  the general-purpose library.

### 1.2.0

Mechanism 1 — "LLM layer as data". All additive; the minimal program is
unchanged.

- `Usage` on every result — normalised `input_tokens` / `output_tokens` /
  `total_tokens` across providers; additive; `skill.last_usage` and
  `chain.last_usage` sum across steps.
- Cost estimation from a per-model price table (`usage.cost`; `None` when a
  model is unpriced).
- Exception hierarchy under `APIError` (`RateLimitError`,
  `AuthenticationError`, `InvalidRequestError`, `NotFoundError`,
  `ServerError`, `NetworkError`).
- `provider/model` routing — `Model("openai/gpt-4o")`; unlocks custom /
  fine-tuned names.
- Model fallback chain — `Skill(model=[primary, backup, …])` advances on a
  transient failure; a non-transient failure propagates immediately.
- `registry.refresh(provider)` — diffs the registry against the provider's
  live `list_models()`.

### 1.1.0

Foundation repaired: five features that never worked in the installed package
are revived, plus fragility closed across the library. 69 regression tests.

- Fixed `Chain.load()`, Agent inside `Chain`, the Qwen embedder/reranker, and
  Agent persistent memory — each previously crashed or was dead code.
- Honest `success=False` on token-budget exhaustion and exhausted step retries.
- Hardened LLM-response and agent-JSON parsing; timeouts on all search tools
  and the MCP bridge; `delete(ids=[])` no longer wipes a collection.
- Chunker contract fixes; thread-safe `Chain.run()`; fixed
  `from yait_aichain.tools import *`.
- POST retry policy (429/503); Anthropic auto-raises `max_tokens` above the
  thinking budget; gpt-5 / o-series parameter correctness; Chroma v2; Qdrant
  string IDs; batched VectorDB `upsert`; safe prompt templating.
- Security: the Google API key moved from the query string to the
  `x-goog-api-key` header.

### 1.0.0

Initial release — `Skill`, `Chain`, `Pool`, `Agent`, `Tool` / `MCPTools`,
`VectorDB`, `Reranker`, and 8 providers (Anthropic, OpenAI, Google, xAI,
Perplexity, Kimi, DeepSeek, Qwen).
