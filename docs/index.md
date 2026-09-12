# aichain 2.0

**The AI Gateway For Developers**

Text, image, and video — route to any AI model through a single, centralized interface. Write your logic once; swap providers without changing a line of code.

---

## What it is

aichain is a pure-Python library that sits between your application and the AI providers you use. It normalises the differences between the providers listed below — and a private server of your own (vLLM, Ollama, LM Studio), where nothing leaves your perimeter — into one universal interface, then gives you programmable building blocks to compose those models into pipelines, tools, and autonomous agents.

```
Your code
    │
    ▼
┌──────────────────── aichain gateway ────────────────────────┐
│                                                              │
│   Model("gpt-4o")          Model("claude-opus-4-6")         │
│   Model("gemini-2.0-flash") Model("grok-3")                 │
│   Model("sonar-pro")        Model("gpt-image-1")            │
│                                                              │
│   Skills · Tools · Chains · Agents                          │
└──────────────────────────────────────────────────────────────┘
    │           │           │           │
    ▼           ▼           ▼           ▼
 OpenAI    Anthropic     Google       xAI    Perplexity
```

---

## One interface. Every provider.

```python
from yait_aichain.models import Model
from yait_aichain.skills import Skill

skill = Skill(
    model  = Model("claude-sonnet-4-6"),   # ← swap this for any model below
    input  = {"messages": [
        {"role": "user", "parts": [{"type": "text", "text": "Summarise: {text}"}]}
    ]},
    output = {"modalities": ["text"], "format": {"type": "text"}},
)

result = skill.run(variables={"text": "Your content here..."})
```

Change the model name — nothing else changes:

```python
Model("gpt-4o")              # OpenAI
Model("claude-sonnet-4-6")   # Anthropic
Model("gemini-2.0-flash")    # Google
Model("grok-3")              # xAI
Model("sonar-pro")           # Perplexity (with live web search)
Model("gpt-image-1")         # OpenAI image generation
Model("kimi-k2.5")           # Kimi
Model("deepseek-chat")       # DeepSeek
```

---

## What the gateway routes

<!-- g:modalities -->
| Modality | Providers |
|---|---|
| **Text → Text** | Anthropic, DeepSeek, Google AI, Kimi (Moonshot AI), OpenAI, Perplexity, Qwen (DashScope), xAI, [your own private server](getting-started/private-models.md) |
| **Text → Image** | Black Forest Labs (FLUX), Google AI, OpenAI, Qwen (DashScope), Recraft, Reve, xAI |
| **Image → Text** | Anthropic, Google AI, Kimi (Moonshot AI), OpenAI, Qwen (DashScope), xAI |
| **Image → Image** | Black Forest Labs (FLUX), Google AI, OpenAI, Qwen (DashScope), Recraft, Reve, xAI |
| **Web search** (a tool) | `searchBrave`, `searchOpenAI`, `searchPerplexity`, `searchSerp` |
<!-- /g:modalities -->

---

## Why a gateway instead of calling providers directly?

| Without aichain | With aichain |
|---|---|
| Different request format per provider | One universal message format for all |
| Different response parsing per provider | One `skill.run()` returns clean Python |
| Switching models requires rewriting code | Change the model name, nothing else |
| Building pipelines is manual glue code | Chain wires steps together automatically |
| Tool-using loops are hand-rolled per project | Agent runs one loop: tools or an answer each turn, `stop_when` decides the end |

---

## The programmable layer

The gateway becomes useful through five building blocks you compose in plain Python:

**[Model](primitives/models.md)** — the routing layer. Auto-detects the provider from the model name and handles all serialisation differences.

**[Skill](primitives/skills.md)** — a reusable task: one model + one prompt template + one output format. Supports text, JSON, and validated JSON Schema outputs.

**[Tool](primitives/tools.md)** — a Python function with a declared interface. Connects the gateway to the real world: web search, file conversion, external APIs.

**[Chain](primitives/chain.md)** — a sequential pipeline. Each step's output flows forward as named variables. Mixes Skills, Tools, and Agents freely.

**[Pool](primitives/pool.md)** — run one Skill or Chain across many inputs in parallel. Total time is the slowest item, not the sum.

**[Agent](agents/overview.md)** — one loop: each turn the model calls tools or answers, and `stop_when` decides when the run ends. Runs inside a Chain or standalone.

**[State](primitives/state.md)** — park a Chain until an external signal (human, webhook, cron) and resume it later, even in another process. An agent's state is its conversation, so it needs no suspend.

**[Eval](primitives/eval.md)** — run the same cases through several models, prompts or modes, several times each, and get accuracy, reliability and cost side by side. Reports on other benchmarks' data too.

---

## What's in the box

<!-- g:provider-list -->**11 cloud providers + your own** — Anthropic, Black Forest Labs (FLUX), DeepSeek, Google AI, Kimi (Moonshot AI), OpenAI, Perplexity, Qwen (DashScope), Recraft, Reve, xAI<!-- /g:provider-list -->, and any [private OpenAI-compatible server](getting-started/private-models.md) you run yourself (vLLM, Ollama, LM Studio, …)

**Built-in tools** — web search (Perplexity, Brave, SerpAPI, OpenAI), file conversion (Markdown / HTML / PDF / text), speech (TTS / STT), embeddings, vector DB, REST API, and `Wait` / `Gate` suspend tools

**Full persistence** — Skills and Chains serialise to YAML. API keys are never stored.

---

## Requirements

- Python 3.10+
- At least one provider API key
- `pip install pyyaml` for YAML save/load

See [Installation](getting-started/installation.md) for optional dependencies and setup.

---

## Where to go next

- **Set up and verify** → [Installation](getting-started/installation.md)
- **First working example** → [Quickstart](getting-started/quickstart.md)
- **Understand the design** → [Concepts](getting-started/concepts.md)
- **All model options** → [Models](primitives/models.md)
- **Why it is built this way** → design notes: [the default agent](design/default-agent.md) · [streaming to a UI](design/streaming-to-a-ui.md) · [image editing](design/image-edit.md) · [the 2026-09-13 audit](design/audit-2026-09-13.md) and its [cleanup plan](design/cleanup-plan-2026-09-13.md)
