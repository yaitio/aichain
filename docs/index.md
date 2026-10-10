# aichain

**Let your coding agent write your agents. Ship the code you tested. Leave any provider with one word.**

A Python library for AI pipelines and agents, small enough for Claude Code or
Codex to learn from one file and build on.

```python
from yait_aichain import Model, Skill

skill = Skill(
    model  = Model("claude-haiku-5-5"),   # change this one word to switch providers
    input  = {"messages": [{"role": "user", "parts": ["Summarise: {text}"]}]},
)

result = skill.run(variables={"text": "Your content here..."})
```

---

## Three promises

**Your coding agent writes the agents.** The whole contract is one 25 KB
`llms.txt`, and the Claude Code plugin carries it with twenty worked recipes.
On twenty tasks the docs were not written against, a small model with the docs
passed 15 in all three tries, Claude Code passed 19, and the same small model
without the docs passed none.

**The code you tested is the code you ship.** Only what you inject changes
between a laptop and production: where state is stored, whose keys are used,
what starts the run. A run that waits for a person or a webhook
[parks and resumes](primitives/state.md) in another process; an agent's state
is its conversation, so one turn fits one serverless call; spending stops at a
ceiling in dollars; keys can be resolved per request, per tenant.

**No provider owns your code.** Switching provider is changing the model name.
When the new one cannot do what you asked, the result says what was adapted or
dropped — [what each provider does with each option](reference/parameters.md).

---

## What you build with it

**Deterministic pipelines.** [Skill](primitives/skills.md) — one model call
with a template and an output format. [Chain](primitives/chain.md) — steps in
order, each output passed on by name; Skills, tools, agents and your own
functions mix freely. [Pool](primitives/pool.md) — one Skill or Chain over many
inputs at once; total time is the slowest item, not the sum.

**Agents, by how hard the task is.** [Agent](agents/overview.md) runs one loop:
each turn the model calls tools or answers, and `stop_when` decides when it
ends. `mode="agile"` decides at every turn; `mode="waterfall"` writes a plan
first, from a stronger `planner_model` if you like, and holds to it; `team=`
lets it delegate to workers with conversations of their own. On τ²-bench a
plain loop did best with a weak model and a planned run with a strong one.

**Tools.** Search, fetch, decide, speech, transcription, images, embeddings,
reranking, chunking, wait and gate — each made the same way,
`search("exa").run(query)`. Any function becomes a tool with `@tool`; any REST
API or MCP server goes into an agent in two lines.
[All tools →](tools-reference/index.md)

**Utilities.** `Vectors` for a vector collection, `Shell` for one folder to
browse, read, write and run commands in, `MCP` and `REST` for connections. Your
code calls their methods; the agent gets only the methods that are tools.

**Control and measurement.** Usage and cost on every result; `max_cost`, one
budget in dollars across every step; permissions by risk class, and `approve=`
for a person to decide; an [event for every step](agents/observability.md);
[Eval](primitives/eval.md) runs arms × cases × trials and reports accuracy,
`Pass^k` and cost side by side.

**Models.** <!-- g:provider-list -->**10 cloud providers + your own** — Anthropic, Black Forest Labs (FLUX), DeepSeek, Google AI, Kimi (Moonshot AI), OpenAI, Perplexity, Qwen (DashScope), Recraft, xAI<!-- /g:provider-list -->, and any [private OpenAI-compatible server](getting-started/private-models.md)
you run yourself (vLLM, Ollama, LM Studio, …), where nothing leaves your
perimeter. Text and vision are [models](primitives/models.md); image generation
is a [tool](tools-reference/image.md).

<!-- g:modalities -->
| Modality | Providers |
|---|---|
| **Text → Text** | Anthropic, DeepSeek, Google AI, Kimi (Moonshot AI), OpenAI, Perplexity, Qwen (DashScope), xAI, [your own private server](getting-started/private-models.md) |
| **Text → Image** | Black Forest Labs (FLUX), Google AI, OpenAI, Qwen (DashScope), Recraft, xAI |
| **Image → Text** | Anthropic, Google AI, Kimi (Moonshot AI), OpenAI, Qwen (DashScope), xAI |
| **Image → Image** | Black Forest Labs (FLUX), Google AI, OpenAI, Qwen (DashScope), Recraft, xAI |
| **Web search** (a tool) | `searchBrave`, `searchExa`, `searchFirecrawl`, `searchParallel`, `searchPerplexity`, `searchTavily` |
<!-- /g:modalities -->

---

## Requirements

- Python 3.10+
- A provider's API key — or none, for a model on your own server
- `pip install pyyaml` for YAML save/load

See [Installation](getting-started/installation.md) for optional dependencies and setup.

---

## Where to go next

- **Set up and verify** → [Installation](getting-started/installation.md)
- **First working example** → [Quickstart](getting-started/quickstart.md)
- **Understand the design** → [Concepts](getting-started/concepts.md)
- **All model options** → [Models](primitives/models.md)
- **Why it is built this way** → design notes: [the default agent](design/default-agent.md) · [streaming to a UI](design/streaming-to-a-ui.md) · [image editing](design/image-edit.md) · [the 2026-09-13 audit](design/audit-2026-09-13.md) and its [cleanup plan](design/cleanup-plan-2026-09-13.md)
