# aichain 2.0

**The AI Gateway For Developers**

Text, image, and video — route to any AI model through a single, centralized interface. Write your logic once; swap providers without changing a line of code.

---

## What it is

aichain is a pure-Python library that sits between your application and the AI providers you use. It normalises the differences between OpenAI, Anthropic, Google, xAI, Perplexity, Kimi, and DeepSeek into one universal interface, then gives you programmable building blocks to compose those models into pipelines, tools, and autonomous agents.

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
from models import Model
from skills import Skill

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

| Modality | Supported providers |
|---|---|
| **Text → Text** | OpenAI, Anthropic, Google, xAI, Perplexity, Kimi, DeepSeek |
| **Text → Image** | OpenAI (gpt-image-1), Google (Imagen via Gemini), xAI (Aurora) |
| **Image → Text** | OpenAI, Anthropic, Google, xAI, Kimi (k2.5) |
| **Text → Search** | Perplexity (sonar), OpenAI web search, Brave, SerpAPI |

---

## Why a gateway instead of calling providers directly?

| Without aichain | With aichain |
|---|---|
| Different request format per provider | One universal message format for all |
| Different response parsing per provider | One `skill.run()` returns clean Python |
| Switching models requires rewriting code | Change the model name, nothing else |
| Building pipelines is manual glue code | Chain wires steps together automatically |
| LLMs truncate long document outputs | Sectional generation — no length limits |
| Complex research tasks need custom agents | Agent handles planning, tools, reflection |

---

## The programmable layer

The gateway becomes useful through five building blocks you compose in plain Python:

**[Model](primitives/models.md)** — the routing layer. Auto-detects the provider from the model name and handles all serialisation differences.

**[Skill](primitives/skills.md)** — a reusable task: one model + one prompt template + one output format. Supports text, JSON, and validated JSON Schema outputs.

**[Tool](primitives/tools.md)** — a Python function with a declared interface. Connects the gateway to the real world: web search, file conversion, external APIs.

**[Chain](primitives/chain.md)** — a sequential pipeline. Each step's output flows forward as named variables. Mixes Skills, Tools, and Agents freely.

**[Agent](agents/overview.md)** — an autonomous engine that plans, acts with tools, reflects, and replans. Runs inside a Chain or standalone.

---

## What's in the box

**7 providers** — OpenAI, Anthropic, Google AI, xAI, Perplexity, Kimi, DeepSeek

**12 built-in tools** — Perplexity search, Brave search, SerpAPI, OpenAI web search, MarkItDown (URL/file → Markdown), Mistletoe (→ HTML), WeasyPrint (→ PDF), DeepL translate, DeepL rephrase, section context, social scheduling (Late)

**Sectional document generation** — produce documents of any length without hitting output token limits. Each section is an independent model call; sections are assembled in order at the end.

**Multilingual search** — Professional and Expert pipelines automatically search in the local language(s) of the target geography alongside English.

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
