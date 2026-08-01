# aichain

**The simplest way to build AI pipelines. 11 cloud providers + your own local server. 1 interface. Zero lock-in.**

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

All of these work identically across **88 models from 11 cloud providers** — plus any model your own local server (vLLM, Ollama, LM Studio, …) serves — with one line to swap any of them.

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
Model("local/llama3.3")      # your own server — vLLM, Ollama, LM Studio, …
```

88 cloud models total (full list: [model registry →](docs/reference/model-registry.md));
the local provider accepts whatever your server serves
([local models →](docs/primitives/local-models.md)).

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

### 1.6.0

**Goal mode** — a third agent mode, for tasks where no plan can be written in
advance because each step depends on what the last one returned.

```python
agent = Agent(orchestrator=Model("claude-sonnet-4-6"), tools=[...],
              mode="goal", done_when=lambda memory: "answer" in memory)
```

- No planning phase. Each iteration the orchestrator sees the objective, the
  observation trail, and what has been ruled out — and picks one action.
- `done_when` is **required** and is best given as a callable: a callable lets
  the run finish on a `check` the harness performed, a string only ever on the
  model's own claim. An unearned `final_answer` is recorded as `refuted` and the
  loop continues.
- Stop rules an open-ended loop needs: `done_when` met, iteration cap (50),
  token budget (250 000), and **no progress** — five consecutive failed attempts
  end the run instead of spending the rest of the budget proving it again.
- Suspend/resume, permissions, checkpoints and the journal all work unchanged.

**Journal entries now carry the observation.** A record of what was *attempted*,
without what it *returned*, cannot drive a next decision — an agent that cannot
read its own feedback re-issues the same action forever. Entries keep a bounded
excerpt of the result (the full value stays in memory), and the trail rendered
into the prompt includes failures, since a failed probe still returned
information. Applies to all modes.

**`memory_read` — a built-in tool on every agent.** Memory previews in the
prompt are truncated so one large value cannot crowd out everything else, but
until now there was no way past that limit: an agent could store a document,
see its first 500 characters on every subsequent turn, and never reach the
rest. Measured on a research task, that failure is total — the agent searched
15 times, stored 48 sources and never wrote an answer, because it could not
read what it had gathered. `memory_read(key, offset, length)` pages through a
stored value, and a truncated preview now says so and names the tool. Prompt
stays bounded; access does not.

**Three fixes found by running the agent, not by reading it.**

- Spawning from a goal-mode agent raised `ValueError`. `spawn()` forwarded
  `mode` but not `done_when`, and goal mode requires one — so `allow_spawn=True`
  crashed the moment the orchestrator delegated. A child now falls back to the
  plan-driven mode: `done_when` is a predicate over *this* agent's objective and
  memory, and neither transfers to a scoped sub-task.
- A spawned child could read its **parent's** memory. `spawn()` forwards the
  parent's tool list, which now contains an agent-bound `memory_read`; the child
  prepended its own and ended up with two tools of the same name, one pointed at
  the wrong memory. A foreign reader is dropped on construction.
- Reading memory wrote it back. The result of a `memory_read` was eligible for
  `store_as` like any other, so looking at a stored document copied it under a
  second key — measured on a research run as 9 of 13 reads going to the agent's
  own bookkeeping rather than to sources. A memory view is no longer stored.

**Repetition detector** — `has_progress()` catches a run that is failing; it
cannot catch one that is succeeding pointlessly. `Journal.is_repeating(k)`
reports a window of attempts that were all the same move, and goal mode writes
that into the next action prompt. Surfaced, never enforced: an agent circling a
hard sub-problem must not be cut off, so the model decides. Only exact
repetition counts — a similarity-based rule was measured against real runs and
did not separate a stuck agent from a healthy search.

**Fix — `Journal.has_progress()` judged on a partial window,** reporting "no
progress" after a single early failure. It now requires a full window, so one
bad attempt cannot end a run that was about to recover.

See [`examples/22_goal_mode.py`](examples/22_goal_mode.py) and
[docs/agents/overview.md](docs/agents/overview.md#goal-mode).

### 1.5.2

**The attempt journal — an append-only record of what the agent did.** Separate
from `AgentMemory` (which holds the data the agent works *with*), the journal
records every attempt with a typed outcome and the evidence behind it, and is
preserved across suspend/resume.

```python
res = agent.run("…")
for e in res.journal:
    print(e["outcome"], e["evidence"]["kind"], e["intent"])
# failed  check        call the API      ← we KNOW it failed: the tool raised
# done    model_claim  fallback path     ← the model says so; nothing verified it
```

- **Outcomes:** `done` · `failed` · `refuted` · `skipped`.
- **Evidence is typed** — `check` (a programmatic fact) vs `model_claim` (asserted,
  not verified). A run whose `done` entries are all `model_claim` has proven
  nothing, and the journal shows that instead of hiding it.
- **Do not redo** — `refuted` entries are fed back into the next action prompt as
  an `ALREADY RULED OUT` block, so a long run stops re-attempting dead ends.
- **Stuck detection** — `Journal.has_progress(k)` answers "did anything actually
  move lately", the stop rule an open-ended loop needs (a budget alone lets an
  agent spin until the tokens run out).

**Crash recovery.** The run is now checkpointed to the `Store` after **every
committed step**, not only when it suspends — so an unplanned death (crash, OOM,
function timeout) is recoverable: a fresh `Agent` sharing the store picks the run
up with `resume(run_id)`, restoring memory, cursor and journal. Suspend/resume
becomes the special case of the same mechanism. (Recovery retries the step that
was in flight — gate side-effecting tools with a `PermissionPolicy` or make them
idempotent; committed steps are never re-run.)

**Fix — a run that suspended twice became unresumable.** Since 1.4.4 the parked
`run_id` is kept aligned with the event-stream id, but `resume()` still deleted
that id unconditionally afterwards — so a second suspend parked a document and
immediately dropped it. Any approval → resume → approval flow lost the run.
`resume()` now keeps the document when the loop re-suspended under the same id.

**Fix — honest success could be bypassed.** The 1.3.4 guarantee ("a step that
ended with an execution error fails the run") was only enforced on the
run-to-completion path: an orchestrator emitting `final_answer` skipped the check
and the run reported `success=True` despite an earlier failed step. The same rule
now applies on every exit path.

### 1.5.1

**Recraft raster → vector (vectorize).** `Model("recraft-vectorize")` traces an
existing raster image into an SVG — no prompt, no generation, just a format
conversion (distinct from `imageToImage`, which is a content *variation*). Same
`Model` + `Skill` ergonomics; the model name selects the `/v1/images/vectorize`
endpoint. Verified live (returns `{"mime_type": "image/svg+xml", ...}`).

```python
Skill(model=Model("recraft-vectorize"),
      input={"messages": [{"role": "user", "parts": [
          {"type": "image", "source": {"kind": "file", "path": "logo.png"}}]}]},
      output={"modalities": ["image"], "format": {"type": "image"}}).run()
```

**11 providers / 82 models.**

- Anthropic json-mode robustness: when the model wraps its JSON in prose (leading
  or trailing commentary), the first balanced top-level object/array is now
  recovered instead of failing to parse (`_extract_first_json`).

### 1.5.0

**Directed multi-turn reasoning in one `Skill`.** Some tasks are a guided
sequence of refinements — ask, refine, refine, finish — where *you* know the
steps (unlike an `Agent`, which decides them). Express it inside a single Skill:
put several `user` turns in the message template, separated by **"generate here"
markers** — an `assistant` turn with no `parts`. Each marker is one model call;
its reply is appended to the running context, so later turns see what earlier
ones produced. No `Chain`, no manual output threading.

```python
Skill(model=Model("claude-sonnet-4-6"), input={"messages": [
    {"role": "system",    "parts": ["Be concise."]},
    {"role": "user",      "parts": ["Write 10 quotes about {topic}."]},
    {"role": "assistant"},                                   # ← generate, keep in context
    {"role": "user",      "parts": ["Drop the 5 most clichéd, write 5 fresh."]},
    {"role": "assistant"},                                   # ← generate, keep in context
    {"role": "user",      "parts": ["Translate the result into {language}."]},
]}).run(variables={"topic": "perseverance", "language": "French"})
```

- An `assistant` turn **with** `parts` is a fixed/seed turn (few-shot, a default
  answer) — no model call. **Without** `parts` it's the generate marker.
- The final turn uses the skill's `output` format (text / json); intermediate
  turns are plain text (they only feed the context).
- `skill.history` holds each generated turn (`history[-1]` is the return);
  `skill.last_usage` is the sum across turns.
- **Backward compatible:** no markers → one call, exactly as before.
- Structure is validated at construction (`ValueError`): at least one `user`
  turn, no leading `assistant`, no two `assistant` turns in a row.

### 1.4.6

**Honest billing errors + registry hygiene.**

- New **`InsufficientCreditsError`** (subclass of `APIError`). An out-of-credits
  state is a *billing* problem, not a credentials problem, but providers signal
  it inconsistently — xAI as `403 permission-denied "used all available credits"`,
  Reve as `402 PARTNER_API_BUDGET_EXHAUSTED`, OpenAI as `429 insufficient_quota`,
  Anthropic as `400 credit balance is too low`. These used to surface as
  `AuthenticationError` / `RateLimitError` / `InvalidRequestError`, sending you to
  re-check the key. Now they are detected from the response body (checked before
  the status-code mapping) and raised as `InsufficientCreditsError`, which is
  **never retried** (retrying the same account won't add funds).
- Registry hygiene (model names verified live against each API): removed the
  phantom `grok-imagine-image-pro` (xAI rejects it) and added the live-verified
  Recraft **V4.1 variants** — `recraftv4_1_utility` (controlled raster for
  product/icon/e-commerce), `recraftv4_1_utility_pro`, `recraftv4_1_pro`,
  `recraftv4_1_utility_vector`. **11 providers / 81 models.**
- `registry.refresh("perplexity")` now works: Perplexity exposes a public
  `/v1/models` router catalog, so `list_models()` reads it live (falling back to
  the curated static list only if the call fails) instead of always returning a
  hard-coded list.

### 1.4.5

**+1 image provider: Reve** (`api.reve.com`). `Model("reve-image")` generates,
edits, and remixes images behind the same `Skill`. Reve has its own (non-OpenAI)
wire shape, so it is its own client family: the operation is chosen by the input —
no image → `POST /v1/image/create`; one input image → `/v1/image/edit`
(`edit_instruction` + `reference_image`); two or more → `/v1/image/remix`
(`reference_images[]`, with `<img>0</img>` tokens in the prompt). Bearer auth
(`REVE_API_KEY`), `aspect_ratio` and `version` (default `latest`) forwarded.
**Now 11 providers / 78 models.**

```python
Skill(model=Model("reve-image"),
      input={"messages": [{"role": "user", "parts": ["A stack of golden pancakes"]}]},
      output={"modalities": ["image"], "format": {"type": "image", "aspect_ratio": "16:9"}}).run()
```

### 1.4.4

**The step boundary — observability & control.** Make every step of an Agent,
Chain, or Skill legible and governable from *outside* the model, without
touching `run()`.

```python
from yait_aichain import Tracer, PermissionPolicy
from yait_aichain.tools import Tool, FINANCIAL

class IssueRefund(Tool):
    name = "issue_refund"; risk = FINANCIAL          # ← risk class as data
    ...

tracer = Tracer()
agent = Agent(
    orchestrator = Model("gpt-4o-mini"),
    tools        = [IssueRefund()],
    hooks        = [tracer],                          # structured event stream
    permissions  = PermissionPolicy({"financial": "approve"}),
)
result = agent.run("Refund order #123")              # pauses for approval
result = agent.resume(result.run_id, signal={"approved": True})
for e in tracer.events: print(e)                     # run/step/llm_call/tool_call
```

- **Logging instead of `print()`.** The library emits through named
  `logging` loggers (`yait_aichain.*`) with a `NullHandler`; the application
  decides the sink. `verbose=` still prints to the console (now routed through
  the logger). Nothing is printed by default.
- **Lifecycle hooks + events.** `Agent`, `Chain`, and `Skill` accept
  `hooks=[...]` — callables that receive a structured `Event` at every boundary
  (`run.*`, `step.*`, `llm_call.*`, `tool_call.*`) carrying `run_id`, `step`,
  token `usage`, `cost`, `duration`, `error`. Ships `Hook`, `Tracer`, and
  `LoggingTracer` conveniences. A buggy hook can never crash the run.
- **Permission matrix.** A tool declares a `risk` class (`read`/`draft`/`write`/
  `external`/`financial`/`destructive`/`privileged`); a `PermissionPolicy` maps
  it to `allow` / `approve` / `deny` enforced *before* the tool runs, outside the
  model. `approve` pauses for an external approval (reusing suspend/resume —
  no manual `Gate`); `deny` still returns a result. Opt-in: no policy → no gating.
- **Tool-call repair.** Tool arguments are validated locally against the schema
  before execution; a malformed call returns a model-readable remediation so the
  agent corrects it within the step's attempt budget.
- **Invariant:** every tool call returns a result — even on denial, validation
  failure, or error.

### 1.4.1

Version is now a **single source of truth** in `yait_aichain/__init__.py`
(`__version__`); `pyproject.toml` reads it dynamically (`[tool.setuptools.dynamic]`)
and the publish workflow checks the same file — no more drift between the two
(which silently blocked the 1.4.0 publish until both were bumped).

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
