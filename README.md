# aichain

**The simplest way to build AI pipelines. 10 providers. 1 interface. Zero lock-in.**

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

All of these work identically across **77 models from 10 providers** — with one line to swap any of them.

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

## 10 providers, one syntax

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

77 models total. Full list: [model registry →](docs/reference/model-registry.md)

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
    orchestrator = Model("claude-opus-4-8"),
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

→ **[examples/](examples/README.md)** — 18 focused examples, one concept each

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
| 17 | `17_chain_human_input.py` | Pause a chain for human approval, then resume |
| 18 | `18_agent_external_trigger.py` | Suspend an agent; a webhook resumes it (cross-process) |
| 19 | `19_image_edit.py` | Edit one product photo across four image providers |

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

### 1.4.0

**Multimodal: image-to-image (image editing).** Restyle / recompose / edit an
existing image across four providers behind the same `Skill` — an input image
part plus an image-output model *is* an edit; swap the provider by changing the
model name, nothing else.

```python
Skill(
    model  = Model("gpt-image-1.5"),     # or gemini-3.1-flash-image / grok-imagine-image / qwen-image-edit
    input  = {"messages": [{"role": "user", "parts": [
        {"type": "image", "source": {"kind": "file", "path": "product.png"}},
        "Place this product on a marble kitchen counter, soft morning light",
    ]}]},
    output = {"modalities": ["image"], "format": {"type": "image"}},
).run()
```

- **Six image providers** edit: OpenAI (`gpt-image-*`, `chatgpt-image-latest`,
  multipart `/v1/images/edits`), Google (Gemini image, conversational edit), xAI
  (`grok-imagine-*`, JSON edits), Qwen (`qwen-image-edit` series, synchronous
  multimodal-generation), plus two dedicated image houses — **Recraft**
  (`recraftv3`, multipart imageToImage) and **Black Forest Labs / FLUX**
  (`flux-kontext-*`, async submit→poll→download). New `image-to-image` task in
  the registry.
- **+2 providers, now 10 / 77 models.** Recraft (`RECRAFT_API_TOKEN`) and BFL
  (`BFL_API_KEY`) join as image specialists — `Model("flux-kontext-pro")` /
  `Model("recraftv3")` resolve and edit like any other.
- **Multiple reference images for edits.** Pass several image parts to compose /
  restyle with references — OpenAI (`image[]`, ≤16), xAI (`images`, ≤3), Qwen and
  Gemini all accept multi-image input.
- **Empty image responses raise a clear error.** When a provider returns no image
  (blocked / refused / text-instead-of-image), the call raises a descriptive
  `ValueError` instead of silently returning `base64=None` (which crashed callers
  on decode).
- **Local files as input.** A media source `{"kind": "file", "path": "..."}` is
  read and base64-encoded automatically (MIME inferred), so you can pass a path
  straight into any vision or edit call.

### 1.3.6

Added **Kimi K2.7 Code** (`kimi-k2.7-code`) — Moonshot's coding-focused model
($0.95 / $4.00 per 1M input/output). It runs with Thinking enabled by default
on the Kimi API; the client never disables it, so it works out of the box. For
non-coding tasks, `kimi-k2.6` stays the recommendation.

### 1.3.5

Model registry refresh — Google image models.

- Gemini image generation is now GA: `gemini-3.1-flash-image` and
  `gemini-3-pro-image` (the `-preview` suffixes are gone), plus the new
  `gemini-2.5-flash-image`. These are the migration target for Google's
  discontinued Imagen 4 endpoints (`imagen-4.0-*`), which this library never
  referenced.

### 1.3.4

Agent engine, token accounting, and transport fixes from the audit.

- **Token budget** is enforced *within* a step (after the action and execution
  calls, not only between steps), so one step can no longer overshoot
  `max_tokens`; agile replans are capped; tokens from an unparseable
  orchestrator reply are still counted.
- **Honest success**: a run fails if *any* executed step ended in an execution
  error (not only the last), and the final output is the last executed step's
  output (even `None`) rather than a stale earlier value.
- **Agent LLM calls go through the `send()` seam** (consistent with Skill;
  async providers work).
- **Usage**: `Skill.last_usage` resets each run (no stale value after a
  failure); `chain.last_usage` includes Agent-step tokens; `NetworkError` is
  retried within a model when `max_retries > 0`.
- **Robustness**: token extraction tolerates `usage: null` / non-numeric; the
  DeepSeek-reasoner gate matches the name exactly; Google embeddings accept
  `GOOGLE_API_KEY` or `GOOGLE_AI_API_KEY`.
- **Unified HTTP transport**: one `make_http()` factory builds the urllib3
  manager for both model and tool clients and honours `HTTPS_PROXY` /
  `HTTP_PROXY` — a proxy now applies to tool traffic (search, fetch, REST, …),
  not just LLM calls.

### 1.3.3

Durable-run (suspend/resume) correctness fixes from the audit.

- Resume no longer re-runs already-attempted steps: skipped/failed steps get a
  terminal status and the resume cursor skips past them; a terminal error during
  a resumed run clears the parked document, so a duplicate trigger is a no-op
  (no double side effects).
- A nested `Agent` that pauses (`Wait`/`Gate`) inside a `Chain` now propagates
  the suspension up — `chain.run()` returns a `SuspendedResult` instead of
  reporting a failed step, and `chain.resume()` continues the child agent run.
- `RunContext` is now real: exposed as `chain.context` / `agent.context` during
  a run, persisted in the run document, and restored on `resume`
  (`Agent.run` / `Agent.resume` accept `context=` too).
- `FileStore`: a non-JSON-serialisable variable/output at a suspend point raises
  a clear error instead of a raw `TypeError`.

### 1.3.2

Correctness & safety fixes from a code audit.

- Qwen image: a terminal task failure (FAILED / timeout / no result) now raises
  a non-retryable `TaskFailedError`, so a retry or model fallback can't silently
  submit a second billable generation.
- SSRF hardening: `RestApiTool` now blocks private / loopback / metadata
  targets (opt out with `AICHAIN_ALLOW_PRIVATE_URLS`); the Qwen TTS audio
  download is guarded and no longer follows redirects.
- VectorDB: `ChromaBackend.delete` refuses an empty `ids`+`filter` (which would
  wipe the whole collection), matching Qdrant.
- `FileStore`: `fsync` before the atomic rename (durable parked runs); a corrupt
  run file raises a clear error instead of a raw `JSONDecodeError`.
- `convertToHTML` writes through the confined output path, consistent with the
  other convert tools.

### 1.3.1

Image-generation fixes and additions.

- Qwen text-to-image now works. DashScope serves the `wan` models only through
  its native *asynchronous* task API, so a new request lifecycle (submit → poll
  → download) runs behind a `send()` seam on the client and returns the image in
  the same shape the synchronous path does. Registry updated to the available
  `wan2.2-t2i-flash` / `wan2.2-t2i-plus` (the old `wanx2.1-*` ids 404'd).
- New OpenAI image models: `chatgpt-image-latest` (the always-current model used
  by ChatGPT — fast, recommended default) and `gpt-image-2`.
- Transparent backgrounds for `gpt-image-*` / `chatgpt-image-*`: `background`
  and `output_format` are forwarded from the output format (`background:
  "transparent"`, `output_format: "png"`), so these models can emit real PNG
  alpha. The `response_format` param is correctly omitted for the
  `chatgpt-image-*` family (it returns base64 natively and rejects the param).

### 1.3.0

Durable, resumable runs — the serverless core. A Chain or Agent can **pause**
until an external signal arrives (a human, a webhook, a cron tick) and
**resume** later, even in a different process. All additive; existing programs
are unchanged.

- `Wait` / `Gate` suspend tools — pause a run until a signal arrives; on resume
  the signal drives the step. `Wait` is a leaf (its output is the signal);
  `Gate` wraps any tool behind an approval decision.
- `Chain.resume(run_id, signal)` / `Agent.resume(run_id, signal)` — continue a
  suspended run from where it paused; completed steps are not re-run.
- Self-contained run documents in a pluggable `StateStore`: `InMemoryStore`
  (default, process-local) and `FileStore` (survives restart); subclass for
  S3/DynamoDB/any KV. The store holds only suspended runs and a resume is
  idempotent — a duplicate trigger is a no-op.
- `run()` returns a falsy `SuspendedResult` (carrying `run_id` and `awaiting`)
  when paused instead of a final result. `RunContext` (tenant, metadata) can be
  passed to `run()` for per-request context.

### 1.2.6

- Security hardening: SSRF guard on outbound URL tools (private/loopback ranges
  blocked; opt out with `AICHAIN_ALLOW_PRIVATE_URLS=1`), URL-scheme allow-list,
  and output-path confinement (`AICHAIN_OUTPUT_ROOT`). Safe-class checks on
  tool instantiation during chain load.

### 1.2.5

- Correctness fixes across the model and tooling layer: response parsing,
  search/MCP timeouts, table chunking when a header exceeds the chunk size,
  and assorted edge cases surfaced by analysis.
- Refreshed the model registry to current provider catalogues and June 2026
  pricing.

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
