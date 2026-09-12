# aichain

**The simplest way to build AI pipelines. 11 cloud providers + your own private server. 1 interface. Zero lock-in.**

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

All of these work identically across **88 models from 11 cloud providers** — plus any open-weight model you host yourself, where the data never leaves your perimeter — with one line to swap any of them.

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

## 12 providers, one syntax

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
Model("private/llama3.3")    # your own server — vLLM, Ollama, LM Studio, …
```

88 cloud models total (full list: [model registry →](docs/reference/model-registry.md)).
The `private` provider needs no key and accepts whatever your server serves —
the prompts and the answers stay inside your perimeter
([private models →](docs/getting-started/private-models.md)).

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
from yait_aichain.agent import Agent, step_count

agent  = Agent(
    model        = Model("claude-opus-4-8"),
    tools        = [searchPerplexity(), convertToMD()],
    mode         = "agile",
    stop_when    = [step_count(10)],
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
from yait_aichain import (RateLimitError, AuthenticationError,
                          InsufficientCreditsError, APIError)

try:
    skill.run()
except RateLimitError as e:
    wait(e.retry_after)        # honours the Retry-After header
except InsufficientCreditsError:
    ...                        # out of credits/quota — top up (not a key problem)
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

→ **[examples/](examples/README.md)** — one concept per file

<!-- examples:start -->
| # | File | What it shows |
|---|---|---|
| 01 | [`01_skill.py`](examples/01_skill.py) | Run a prompt against a model. |
| 02 | [`02_skill_models.py`](examples/02_skill_models.py) | Same prompt, three providers. |
| 03 | [`03_skill_multimodal.py`](examples/03_skill_multimodal.py) | Text → Image → Text, three different providers. |
| 04 | [`04_skill_save_load.py`](examples/04_skill_save_load.py) | Save a skill to YAML, reload and run it. |
| 05 | [`05_tool_convert.py`](examples/05_tool_convert.py) | Convert a URL or file to Markdown. |
| 06 | [`06_tool_mcp.py`](examples/06_tool_mcp.py) | Connect to an MCP server, discover tools, run them. |
| 07 | [`07_tool_custom.py`](examples/07_tool_custom.py) | Define your own tool and use it in a Chain. |
| 08 | [`08_chain.py`](examples/08_chain.py) | Two skills in sequence, two different providers. |
| 09 | [`09_chain_tool_skill.py`](examples/09_chain_tool_skill.py) | Tool + Skill in one chain. |
| 10 | [`10_chain_save_load.py`](examples/10_chain_save_load.py) | Save a chain to YAML, reload and run it. |
| 11 | [`11_pool.py`](examples/11_pool.py) | Run the same skill in parallel for multiple inputs. |
| 12 | [`12_pool_chain.py`](examples/12_pool_chain.py) | Chain as a Pool runner. |
| 13 | [`13_agent.py`](examples/13_agent.py) | Basic agent with one tool. |
| 14 | [`14_agent_tools.py`](examples/14_agent_tools.py) | Agent with multiple tools, picks autonomously. |
| 15 | [`15_agent_orchestrator.py`](examples/15_agent_orchestrator.py) | Orchestrator agent spawns sub-agents. |
| 16 | [`16_debug.py`](examples/16_debug.py) | Inspect intermediate steps in Chain, Pool, and Agent. |
| 17 | [`17_chain_human_input.py`](examples/17_chain_human_input.py) | Human-in-the-loop: pause a chain for manual input. |
| 18 | [`18_chain_external_trigger.py`](examples/18_chain_external_trigger.py) | Pause a run until an EXTERNAL trigger resumes it. |
| 19 | [`19_image_edit.py`](examples/19_image_edit.py) | Image → image (editing) across four providers, one Skill. |
| 20 | [`20_observability.py`](examples/20_observability.py) | The step boundary: hooks, events, and an approval gate. |
| 21 | [`21_multi_turn.py`](examples/21_multi_turn.py) | Directed multi-turn reasoning in ONE Skill. |
| 22 | [`22_goal_mode.py`](examples/22_goal_mode.py) | A stop condition the harness can verify |
| 23 | [`23_private_model.py`](examples/23_private_model.py) | Run a model on hardware you control. |
| 24 | [`24_scaffold.py`](examples/24_scaffold.py) | Scaffold — the model designs the team, then runs it |
<!-- examples:end -->

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
| **Anthropic** | Claude Fable 5, Opus / Sonnet / Haiku 4 | ✓ | — | `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT-5.5, GPT-5.4, GPT-4o | ✓ | ChatGPT-Image, GPT-Image-2 | `OPENAI_API_KEY` |
| **Google** | Gemini 2.5 Pro / Flash, 3.x | ✓ | Gemini image models | `GOOGLE_AI_API_KEY` |
| **xAI** | Grok 4, Grok 3 | ✓ | Grok-Imagine | `XAI_API_KEY` |
| **Perplexity** | Sonar Pro, Sonar, Deep Research | — | — | `PERPLEXITY_API_KEY` |
| **Kimi** | K2.7 Code, K2.6, K2.5, K2 Thinking | ✓ | — | `MOONSHOT_API_KEY` |
| **DeepSeek** | DeepSeek-V3, DeepSeek-R1 | — | — | `DEEPSEEK_API_KEY` |
| **Qwen** | Qwen-Max, Qwen3, QwQ | ✓ | Wan 2.2 image | `DASHSCOPE_API_KEY` |
| **Recraft** | — | — | Recraft V3 (raster + vector) | `RECRAFT_API_TOKEN` |
| **BFL (FLUX)** | — | — | FLUX.2, FLUX Kontext | `BFL_API_KEY` |
| **Reve** | — | — | Reve Image (create / edit / remix) | `REVE_API_KEY` |

**Image editing — instruction edit, preserves the subject:** OpenAI · Google · xAI · Qwen · FLUX Kontext (place / restyle / recompose while keeping the original object)  
**Image-to-image — whole-image variation (restyle, not subject-preserving):** Recraft (`imageToImage`, `strength`-controlled)  
**Embedding:** OpenAI · Cohere · Voyage · Google · Qwen  
**Reranking:** Cohere · Voyage · Qwen  
**Vector DB:** Chroma · Qdrant · Pinecone

---

## License

MIT

---

## Changelog

Every release is in **[CHANGELOG.md](CHANGELOG.md)** — what changed, and why it
changed. Releases before `2.0` are recorded incompletely; the file says which
ones and why they were not reconstructed.
