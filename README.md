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

**Your coding agent writes the agents.** The whole contract is one 26 KB
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

- **Small enough to read in one go.** `llms.txt` is 26 KB, the skill 4 KB, the
  whole library about 29,000 lines of Python.
- **One required dependency**, `urllib3` — and `tomli` on Python 3.10, which
  has no TOML reader of its own. YAML, MCP and local page fetching are extras
  you install when you use them.
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
| **[Deterministic pipelines](docs/pipelines/chain.md)** | `Skill` — one model call with a template and an output format. `Chain` — steps in order, each output passed on by name. `Pool` — one Skill or Chain over many inputs at once. |
| **[Agents, by how hard the task is](docs/agents/overview.md)** | `mode="agile"` decides at every turn. `mode="waterfall"` writes a plan first — from a stronger `planner_model` if you like — and holds to it. `team=` lets it delegate to workers with conversations of their own. On τ²-bench a plain loop did best with a weak model and a planned run with a strong one. |
| **[Tools](docs/tools/index.md)** | Search, fetch, decide, speech, transcription, images, video, embeddings, reranking, chunking, wait and gate — each made the same way, `search("exa")`. Any function becomes a tool with `@tool`; any REST API or MCP server goes into an agent in two lines. |
| **[Utilities](docs/utilities/index.md)** | `Vectors` for a vector collection, `Shell` for one folder to browse, read, write and run commands in, `MCP` and `REST` for connections. Your code calls their methods; the agent gets only the methods that are tools. |
| **[Control and measurement](docs/control/cost-and-budget.md)** | Usage and cost on every result. `max_cost` — one budget in dollars across every step. Permissions by risk class in one `policy`, and a person to decide — now or later. An event for every step. `eval/` runs arms × cases × trials and reports accuracy, `Pass^k` and cost side by side. |
| **[Models](docs/models/models.md)** | **<!-- g:count-models -->181 models from 18 cloud providers<!-- /g:count-models -->**, plus any open-weight model on your own server, where the data never leaves your perimeter. Text and vision are models; image generation is a tool. |

---

## Install

```bash
pip install yait-aichain
```

One required dependency, `urllib3` (and `tomli` on Python 3.10). Extras when you use what needs them:
`[yaml]` to save and load, `[mcp]` for MCP servers, `[fetch]` to read pages
locally — [Installation](docs/start/installation.md). Keys, only for the
providers you use:

<!-- g:exports -->
```bash
export ANTHROPIC_API_KEY="…"
export BFL_API_KEY="…"
export DEEPSEEK_API_KEY="…"
export GOOGLE_AI_API_KEY="…"
export MOONSHOT_API_KEY="…"
export KLING_API_KEY="…"
export KREA_API_KEY="…"
export META_API_KEY="…"
export MINIMAX_API_KEY="…"
export MISTRAL_API_KEY="…"
export OPENAI_API_KEY="…"
export PERPLEXITY_API_KEY="…"
export DASHSCOPE_API_KEY="…"
export RECRAFT_API_TOKEN="…"
export STEPFUN_API_KEY="…"
export XAI_API_KEY="…"
export MIMO_API_KEY="…"
export ZAI_API_KEY="…"
export OPENROUTER_API_KEY="…"
export BRAVE_SEARCH_API_KEY="…"
export COHERE_API_KEY="…"
export EXA_API_KEY="…"
export FIRECRAWL_API_KEY="…"
export GOOGLE_API_KEY="…"
export GOOGLE_CLOUD_API_KEY="…"
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
`.github/scripts/skill.py` from the files this repository tests.

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

## Learn more

| | |
|---|---|
| **Start** | [Quickstart](docs/start/quickstart.md) — five minutes, four steps · [Concepts](docs/start/concepts.md) — six ideas, one screen each · [Migrating to 4.0](docs/start/migrating-to-4.md) |
| **Pipelines** | [Skill](docs/pipelines/skills.md) · [Chain](docs/pipelines/chain.md) · [Pool](docs/pipelines/pool.md) · [Pause and resume](docs/pipelines/pause-and-resume.md) |
| **Agents** | [The loop and its modes](docs/agents/overview.md) · [Driving the loop](docs/agents/driving-the-loop.md) · [An agent as a chain step](docs/agents/agent-as-chain-step.md) |
| **Tools** | [How tools work](docs/tools/index.md) · [Search](docs/tools/search.md) · [Fetch](docs/tools/fetch.md) · [Decide](docs/tools/decide.md) · [Chunk, embed, rerank](docs/tools/retrieval.md) · [Speech and images](docs/tools/media.md) |
| **Utilities** | [Vectors, Shell, REST, MCP](docs/utilities/index.md) |
| **Control** | [Cost and budget](docs/control/cost-and-budget.md) · [Permissions](docs/control/permissions.md) · [Observability](docs/control/observability.md) · [Eval](docs/control/eval.md) |
| **Models** | [Models and options](docs/models/models.md) · [Private models](docs/models/private-models.md) · [the registry](docs/reference/model-registry.md) |
| **Production** | [From laptop to production](docs/production.md) · [Build with your coding agent](docs/with-your-agent.md) |
| **[API](docs/api/index.md)** | every class, function and parameter, generated from the code |

## Examples

→ **[examples/](examples/README.md)** — one concept per file

<!-- examples:start -->
| # | File | What it shows |
|---|---|---|
| 01 | [`01_skill.py`](examples/01_skill.py) | One prompt, one model, one answer. |
| 02 | [`02_switch_provider.py`](examples/02_switch_provider.py) | The provider changes with one word, and nothing else does. |
| 03 | [`03_structured_output.py`](examples/03_structured_output.py) | Data from a model, not prose: an answer that fits a JSON schema. |
| 04 | [`04_fallback_models.py`](examples/04_fallback_models.py) | Survive a provider outage with one parameter: a list of models. |
| 05 | [`05_multi_turn.py`](examples/05_multi_turn.py) | A directed sequence of refinements in one Skill, without a Chain. |
| 06 | [`06_media_input.py`](examples/06_media_input.py) | An image, a PDF and a video to one model: the type follows the file. |
| 07 | [`07_stream.py`](examples/07_stream.py) | The answer printed as it is written: the same Skill, streamed. |
| 08 | [`08_skill_save_load.py`](examples/08_skill_save_load.py) | A prompt kept in a file apart from the code; the key never reaches it. |
| 09 | [`09_private_model.py`](examples/09_private_model.py) | A model on your own server: the data stays with you, the code is the same. |
| 10 | [`10_what_was_sent.py`](examples/10_what_was_sent.py) | The library never changes a request in silence. |
| 11 | [`11_custom_tool.py`](examples/11_custom_tool.py) | Any function becomes a tool: its docstring is the schema a model sees. |
| 12 | [`12_image.py`](examples/12_image.py) | Make an image, then edit it on every provider whose key is set: one tool, eight APIs. |
| 13 | [`13_speech.py`](examples/13_speech.py) | Voice both ways through one interface: text to an mp3, and back to text. |
| 14 | [`14_search_fetch.py`](examples/14_search_fetch.py) | Search the web, then read the first result as a model would. |
| 15 | [`15_retrieval.py`](examples/15_retrieval.py) | Answer from your own texts: chunk, embed, find, rerank, answer with the source. |
| 16 | [`16_decide.py`](examples/16_decide.py) | A closed set of answers with a visible price: classify support messages. |
| 17 | [`17_mcp.py`](examples/17_mcp.py) | An MCP server is a set of agent tools: discover them, call one. |
| 18 | [`18_rest.py`](examples/18_rest.py) | Your own API as tools, without code per method; the risk class follows the method. |
| 19 | [`19_shell.py`](examples/19_shell.py) | Shell under a policy that reads the command, not only the tool. |
| 20 | [`20_chain.py`](examples/20_chain.py) | Steps in order, each step's result under its own key: GPT writes, Claude reviews. |
| 21 | [`21_chain_steps.py`](examples/21_chain_steps.py) | Connect steps whose names differ, and let the chain say what it needs. |
| 22 | [`22_chain_save_load.py`](examples/22_chain_save_load.py) | A pipeline as a file, with its policy: stored, versioned, run elsewhere. |
| 23 | [`23_chain_human.py`](examples/23_chain_human.py) | A person in the loop: the chain parks, the draft is not re-made. |
| 24 | [`24_chain_webhook.py`](examples/24_chain_webhook.py) | A run parks until an outside event resumes it. |
| 25 | [`25_pool.py`](examples/25_pool.py) | A list processed in parallel with one parameter, no threads in your code. |
| 26 | [`26_pool_chain.py`](examples/26_pool_chain.py) | The primitives nest: a Chain is a Pool's runner, one per URL, all at once. |
| 27 | [`27_agent.py`](examples/27_agent.py) | An agent picks its own steps between search and fetch; the result says how it ended. |
| 28 | [`28_agent_stop.py`](examples/28_agent_stop.py) | "Done" on the model's word and "done" on a check are different ends. |
| 29 | [`29_agent_stream.py`](examples/29_agent_stream.py) | One ordered stream of events: what an interface is built on. |
| 30 | [`30_agent_team.py`](examples/30_agent_team.py) | A lead hands work to its team: an agent that searches, a chain that reads. |
| 32 | [`32_agent_pause.py`](examples/32_agent_pause.py) | An agent waits for an outside event, and another process resumes it. |
| 33 | [`33_agent_step.py`](examples/33_agent_step.py) | Outside code drives the loop, one step per invocation. |
| 34 | [`34_budget.py`](examples/34_budget.py) | One ceiling in money across a run: the next step is refused before it begins. |
| 35 | [`35_policy.py`](examples/35_policy.py) | Who allows a risky call, and how a call is changed before it runs. |
| 36 | [`36_observe.py`](examples/36_observe.py) | What happened inside a run, without print() in your code or a tracing SDK. |
| 37 | [`37_tenant_keys.py`](examples/37_tenant_keys.py) | One Model serves every tenant: each request carries its customer's key. |
| 38 | [`38_eval.py`](examples/38_eval.py) | Two prompts compared on repeated trials, not on one lucky run. |
| 39 | [`39_video.py`](examples/39_video.py) | A video takes minutes: submit returns a job at once, collect waits for it. |
<!-- examples:end -->

---

## License

MIT

---

## Changelog

Every release is in **[CHANGELOG.md](CHANGELOG.md)** — what changed, and why it
changed. Releases before `2.0` are recorded incompletely; the file says which
ones and why they were not reconstructed.
