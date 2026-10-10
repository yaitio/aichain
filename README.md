# aichain

**The provider-agnostic AI toolkit for Python.**

Let your coding agent write your agents. Ship the code you tested. Leave any
provider with one word. Pipelines and agents, small enough for Claude Code,
Codex or Cursor to learn from one file and build on.

```python
from yait_aichain import Model, Skill

skill = Skill(
    model  = Model("claude-haiku-5-5"),   # change this one word to switch providers
    input  = {"messages": [{"role": "user", "parts": ["Summarise: {text}"]}]},
)

result = skill.run(variables={"text": "..."})
```

Change `"claude-haiku-5-5"` to `"gpt-6-luna"`, `"gemini-3-flash-preview"` or
`"grok-4.3"` — nothing else changes.

---

## Three promises

**Your coding agent writes the agents.** The whole contract is one 25 KB
[`llms.txt`](llms.txt), and the [plugin for Claude Code, Codex and Cursor](#use-it-from-claude-code-codex-or-cursor)
carries it with twenty worked recipes. Measured on tasks the docs were not
written against: on twenty, a small model with the docs passed 15 in all three
tries and Claude Code passed 19; on eleven for what 3.x added, both passed all
eleven; and without the docs the same small model passed none of an earlier
twenty ([the measurements](docs/with-your-agent.md#how-well-it-works)).

**The code you tested is the code you ship.** Only what you inject changes
between a laptop and production: where state is stored, whose keys are used,
what starts the run. A chain or an agent that waits for a person or a webhook
parks, and resumes later in another process. An agent's state is its
conversation, so one turn fits one serverless call. Spending stops at a ceiling
in dollars, and keys can be resolved per request, per tenant.

**No provider owns your code.** Switching provider is changing the model name.
When the new one cannot do what you asked — no `top_k`, no `seed`, a different
image size — the result says what was adapted or dropped, instead of leaving
you to find out from the output.

---

## Highlights

- **Small enough to read in one go.** `llms.txt` is 25 KB, the skill 4 KB, the
  whole library about 29,000 lines of Python.
- **One required dependency**, `urllib3`. YAML, MCP and local page fetching
  are extras you install when you use them.
- **The price of every call.** Usage and cost come from the registry on every
  result, and `max_cost` is one budget in dollars shared by a Skill, a Chain,
  a Pool and an Agent.
- **A pipeline is data.** Skills and Chains save to YAML and load back as they
  were; API keys are never written.
- **A skill for your coding agent** — one package for Claude Code, Codex and
  Cursor.

---

## What you build with it

| | |
|---|---|
| **Deterministic pipelines** | `Skill` — one model call with a template and an output format. `Chain` — steps in order, each output passed on by name. `Pool` — one Skill or Chain over many inputs at once. |
| **Agents, by how hard the task is** | `mode="agile"` decides at every turn. `mode="waterfall"` writes a plan first — from a stronger `planner_model` if you like — and holds to it. `team=` lets it delegate to workers with conversations of their own. On τ²-bench a plain loop did best with a weak model and a planned run with a strong one. |
| **Tools** | Search, fetch, decide, speech, transcription, images, embeddings, reranking, chunking, wait and gate — each made the same way, `search("exa")`. Any function becomes a tool with `@tool`; any REST API or MCP server goes into an agent in two lines. |
| **Utilities** | `Vectors` for a vector collection, `Shell` for one folder to browse, read, write and run commands in, `MCP` and `REST` for connections. Your code calls their methods; the agent gets only the methods that are tools. |
| **Control and measurement** | Usage and cost on every result. `max_cost` — one budget in dollars across every step. Permissions by risk class, and `approve=` for a person to decide. An event for every step. `eval/` runs arms × cases × trials and reports accuracy, `Pass^k` and cost side by side. |
| **Models** | **<!-- g:count-models -->130 models from 10 cloud providers<!-- /g:count-models -->**, plus any open-weight model on your own server, where the data never leaves your perimeter. Text and vision are models; image generation is a tool. |

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

<!-- g:exports -->
```bash
export ANTHROPIC_API_KEY="…"
export BFL_API_KEY="…"
export DEEPSEEK_API_KEY="…"
export GOOGLE_AI_API_KEY="…"
export MOONSHOT_API_KEY="…"
export OPENAI_API_KEY="…"
export PERPLEXITY_API_KEY="…"
export DASHSCOPE_API_KEY="…"
export RECRAFT_API_TOKEN="…"
export XAI_API_KEY="…"
export ="…"
export BRAVE_SEARCH_API_KEY="…"
export COHERE_API_KEY="…"
export EXA_API_KEY="…"
export FIRECRAWL_API_KEY="…"
export GOOGLE_API_KEY="…"
export PARALLEL_API_KEY="…"
export TAVILY_API_KEY="…"
export TYPESAFE_API_KEY="…"
export VOYAGE_API_KEY="…"
```
<!-- /g:exports -->

---

## Use it from Claude Code, Codex or Cursor

One plugin carries the skill that teaches a coding agent to build on this
library: `SKILL.md`, `llms.txt` and twenty worked recipes, generated by
`scripts/skill.py` from the files this repository tests.

**Claude Code**

```text
/plugin marketplace add yaitio/aichain
/plugin install yait-aichain@yait
```

**Codex** — add the repository as a marketplace, then install `yait-aichain`
from it:

```bash
codex plugin marketplace add yaitio/aichain
```

**Cursor** — Customize → *From GitHub Repository* → `yaitio/aichain`, then
install `yait-aichain`.

The skill alone works too: copy `plugin/skills/yait-aichain/` into
`~/.agents/skills/`, which Codex and Cursor both read, or into
`~/.claude/skills/` for Claude Code.

## Every provider, one syntax

```python
from yait_aichain.models import Model

Model("claude-sonnet-4-6")   # Anthropic
Model("gpt-4o")              # OpenAI
Model("gemini-2.5-flash")    # Google
Model("grok-3")              # xAI
Model("sonar-pro")           # Perplexity
Model("kimi-k2.6")           # Kimi
Model("deepseek-chat")       # DeepSeek
Model("qwen-max")            # Qwen
Model("private/llama3.3")    # your own server — vLLM, Ollama, LM Studio, …
```

Full list: [model registry →](docs/reference/model-registry.md).
The `private` provider needs no key and accepts whatever your server serves —
the prompts and the answers stay inside your perimeter
([private models →](docs/models/private-models.md)).

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
print(result.output, result["summary"])   # the last step, and any step by key
```

### Pool — parallel execution

```python
pool    = Pool(summarise_skill, items=[{"text": t} for t in documents], max_flows=10)
results = pool.run()                    # all documents processed simultaneously
print(results[0], results.success, results.cost)

print(pool.status)   # {PENDING: 0, RUNNING: 0, DONE: 50, FAILED: 0}
print(pool.history)  # per-item: status, output, error, duration
```

### Agent — autonomous reasoning

```python
from yait_aichain.agent import Agent, step_count

agent  = Agent(
    model        = Model("claude-opus-4-8"),
    tools        = [search("perplexity"), fetch("local")],
    mode         = "agile",
    stop_when    = [step_count(10)],
)
result = agent.run("Compare the top 3 vector databases.")
print(result.output)
print(f"steps={result.steps_taken}  tokens={result.tokens_used:,}")
```

### Full RAG pipeline

```python
from yait_aichain.tools          import chunk, embed, rerank
from yait_aichain.tools.vectordb import VectorDB, vectorUpsert, vectorQuery

store    = VectorDB("chroma", "docs", embedder=embed("cohere", model="embed-v4.0"))
reranker = rerank("cohere", model="rerank-v3.5")

# Ingest
chunks = chunk(max_chars=800).run(my_document)
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
`search(service)` — `perplexity` · `brave` · `exa` · `firecrawl` · `tavily` · `parallel`

### Fetch
`fetch(service)` — `firecrawl` · `tavily` · `exa` · `parallel` · `local` (free; `pip install "yait-aichain[fetch]"`)

### Decide
`decide(service, questions=…)` — `openai` · `typesafe`

```python
from yait_aichain.tools import search, fetch

found = search("exa").run("AI and leadership", {"domains": ["hbr.org"]})
pages = fetch("firecrawl").run([r["url"] for r in found["results"]])
Agent(model, tools=[search("perplexity"), fetch("firecrawl")])
```

### Speech
`speech(service)` — `openai` · `google` · `xai` · `qwen` (text → audio) ·
`transcribe(service)` — the same four (audio → text)

### Embeddings, reranking, chunking
`embed(service, model=…)` — `openai` · `cohere` · `voyage` · `google` · `qwen` ·
`rerank(service, model=…)` — `cohere` · `voyage` · `qwen` · `chunk(max_chars=…)`

### Images
`image(service, model=…)` — `openai` · `google` · `bfl` · `xai` · `qwen` · `recraft`;
`.run(prompt, images=[…], options={"aspect_ratio": "16:9"})` generates, or edits the images given

### Pause
`wait(reason=…)` · `gate(tool, reason=…)`

### Utilities — `from yait_aichain.utilities import …`
`Vectors` (a vector database; `kb.query` its tool) · `MCP` (a server's tools) ·
`REST` (an API; each endpoint a tool) · `Shell` (a folder; browse, read, write, run)


---

## Examples

→ **[examples/](examples/README.md)** — one concept per file

<!-- examples:start -->
| # | File | What it shows |
|---|---|---|
| 01 | [`01_skill.py`](examples/01_skill.py) | Run a prompt against a model. |
| 02 | [`02_skill_models.py`](examples/02_skill_models.py) | Same prompt, three providers. |
| 03 | [`03_skill_multimodal.py`](examples/03_skill_multimodal.py) | Text → Image → Text, three providers. |
| 04 | [`04_skill_save_load.py`](examples/04_skill_save_load.py) | Save a skill to YAML, reload and run it. |
| 05 | [`05_tool_fetch.py`](examples/05_tool_fetch.py) | A web page's text from its URL. |
| 06 | [`06_tool_mcp.py`](examples/06_tool_mcp.py) | An MCP server as a utility: discover its tools, call them. |
| 07 | [`07_tool_custom.py`](examples/07_tool_custom.py) | Make any function a tool, and use it in a Chain. |
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
| 19 | [`19_image_edit.py`](examples/19_image_edit.py) | Image → image (editing) across providers, one tool. |
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
| **Anthropic** | Claude Fable 5.1, Opus / Sonnet / Haiku 5.5 | ✓ | — | `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT-6.1, GPT-6, GPT-5.6, GPT-5.5 | ✓ | GPT-Image-2.5, GPT-Image-2 | `OPENAI_API_KEY` |
| **Google** | Gemini 3.8 / 3.6 Flash, 3.1 Pro, 3.1 Flash-Lite | ✓ | Gemini image models | `GOOGLE_AI_API_KEY` |
| **xAI** | Grok 4.7, 4.3, Grok Build | ✓ | Grok-Imagine | `XAI_API_KEY` |
| **Perplexity** | Sonar Pro, Sonar, Deep Research | — | — | `PERPLEXITY_API_KEY` |
| **Kimi** | K3, K2.7 Code, K2.6 | ✓ | — | `MOONSHOT_API_KEY` |
| **DeepSeek** | DeepSeek Flash, V4 Pro | — | — | `DEEPSEEK_API_KEY` |
| **Qwen** | Qwen 3.8 Max / Flash, 3.7 Plus | ✓ | Wan 2.2, Qwen-Image-Edit | `DASHSCOPE_API_KEY` |
| **Recraft** | — | — | Recraft V4.1, V3 (raster + vector) | `RECRAFT_API_TOKEN` |
| **BFL (FLUX)** | — | — | FLUX.3, FLUX.2, FLUX Kontext | `BFL_API_KEY` |

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
