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
without the docs passed none. On eleven more for what 3.x added, both passed
all eleven.

**The code you tested is the code you ship.** Only what you inject changes
between a laptop and production: where state is stored, whose keys are used,
what starts the run. A run that waits for a person or a webhook
[parks and resumes](pipelines/pause-and-resume.md) in another process; an agent's state
is its conversation, so one turn fits one serverless call; spending stops at a
ceiling in dollars; keys can be resolved per request, per tenant.

**No provider owns your code.** Switching provider is changing the model name.
When the new one cannot do what you asked, the result says what was adapted or
dropped — [what each provider does with each option](reference/parameters.md).

---

## What you build with it

**Deterministic pipelines.** [Skill](pipelines/skills.md) — one model call
with a template and an output format. [Chain](pipelines/chain.md) — steps in
order, each output passed on by name; Skills, tools, agents and your own
functions mix freely. [Pool](pipelines/pool.md) — one Skill or Chain over many
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
[All tools →](reference/tools.md)

**Utilities.** `Vectors` for a vector collection, `Shell` for one folder to
browse, read, write and run commands in, `MCP` and `REST` for connections. Your
code calls their methods; the agent gets only the methods that are tools.

**Control and measurement.** Usage and cost on every result; `max_cost`, one
budget in dollars across every step; permissions by risk class, and `approve=`
for a person to decide; an [event for every step](control/observability.md);
[Eval](control/eval.md) runs arms × cases × trials and reports accuracy,
`Pass^k` and cost side by side.

**Models.** <!-- g:provider-list -->**10 cloud providers + your own** — Anthropic, Black Forest Labs (FLUX), DeepSeek, Google AI, Kimi (Moonshot AI), OpenAI, Perplexity, Qwen (DashScope), Recraft, xAI<!-- /g:provider-list -->, and any [private OpenAI-compatible server](models/private-models.md)
you run yourself (vLLM, Ollama, LM Studio, …), where nothing leaves your
perimeter. Text and vision are [models](models/models.md); image generation
is a [tool](tools/image.md).

<!-- g:modalities -->
| Modality | Providers |
|---|---|
| **Text → Text** | Anthropic, DeepSeek, Google AI, Kimi (Moonshot AI), OpenAI, Perplexity, Qwen (DashScope), xAI, [your own private server](models/private-models.md) |
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

See [Installation](start/installation.md) for optional dependencies and setup.

---

## Where to go next

- **Set up and verify** → [Installation](start/installation.md)
- **First working example** → [Quickstart](start/quickstart.md)
- **Understand the design** → [Concepts](start/concepts.md)
- **All model options** → [Models](models/models.md)
- **Why it is built this way** → design notes: [the default agent](../design/default-agent.md) · [streaming to a UI](../design/streaming-to-a-ui.md) · [image editing](../design/image-edit.md) · [the 2026-09-13 audit](../design/audit-2026-09-13.md) and its [cleanup plan](../design/cleanup-plan-2026-09-13.md)
