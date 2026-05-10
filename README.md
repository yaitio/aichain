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
| **OpenAI** | GPT-5, GPT-4.1, GPT-4o, o1, o3, o4-mini | ✓ | DALL-E 3, GPT-Image-1 | `OPENAI_API_KEY` |
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
