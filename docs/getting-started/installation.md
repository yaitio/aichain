# Installation

## Library

```bash
pip install yait-aichain
```

Optional extras — install only what you need:

```bash
pip install yait-aichain[convert]   # file/URL → Markdown (markitdown)
pip install yait-aichain[yaml]      # save & load Skills and Chains (pyyaml)
pip install yait-aichain[mcp]       # MCP server integration (fastmcp)
pip install yait-aichain[all]       # everything above
```

Or install extras individually:

```bash
pip install markitdown   # file/URL → Markdown
pip install pyyaml       # Skill.save() / Chain.save() / load()
pip install fastmcp      # MCP server integration
```

---

## Dependencies

All HTTP calls use `urllib3`. It ships with most Python environments — if missing:

```bash
pip install urllib3
```

### Optional — by feature

| Feature | Package | Install |
|---|---|---|
| URL / file → Markdown | markitdown | `pip install markitdown` |
| Save & load Skills/Chains | pyyaml | `pip install pyyaml` |
| MCP server tools | fastmcp | `pip install fastmcp` |
| HTML export | mistletoe | `pip install mistletoe` |
| PDF export | weasyprint | `pip install weasyprint` |

#### WeasyPrint system libraries (macOS)

WeasyPrint needs Pango and GLib. Install with Homebrew:

```bash
brew install pango glib
export DYLD_LIBRARY_PATH="$(brew --prefix)/lib:$DYLD_LIBRARY_PATH"
```

Add the `export` line to your `~/.zshrc` to make it permanent.

---

## API keys

Set keys only for the providers you use — or none at all: a model you host
yourself needs no key and sends nothing outside your network. See
[Private models](private-models.md).

<!-- g:install-keys -->
| Service | Environment variable | Get a key |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | <https://console.anthropic.com/settings/keys> |
| Black Forest Labs (FLUX) | `BFL_API_KEY` | — |
| DeepSeek | `DEEPSEEK_API_KEY` | <https://platform.deepseek.com/api_keys> |
| Google AI | `GOOGLE_AI_API_KEY` | <https://aistudio.google.com/app/apikey> |
| Kimi (Moonshot AI) | `MOONSHOT_API_KEY` | <https://platform.kimi.ai/> |
| OpenAI | `OPENAI_API_KEY` | <https://platform.openai.com/api-keys> |
| Perplexity | `PERPLEXITY_API_KEY` | <https://www.perplexity.ai/settings/api> |
| Qwen (DashScope) | `DASHSCOPE_API_KEY` | <https://dashscope.aliyuncs.com> |
| Recraft | `RECRAFT_API_TOKEN` | — |
| Reve | `REVE_API_KEY` | — |
| xAI | `XAI_API_KEY` | <https://console.x.ai/> |
| Brave Search | `BRAVE_SEARCH_API_KEY` | <https://brave.com/search/api> |
| SerpAPI | `SERPAPI_API_KEY` | <https://serpapi.com> |
| Cohere | `COHERE_API_KEY` | <https://dashboard.cohere.com> |
| Voyage | `VOYAGE_API_KEY` | <https://dash.voyageai.com> |
<!-- /g:install-keys -->

Set for the current session:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

Or put them all in a `.env` file:

```bash
# .env
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_AI_API_KEY="AIza..."
```

```bash
source .env
```

> Keys are **never** written to YAML files by `save()`. They are always resolved from environment variables at load time.

---

## Verify installation

```bash
python -c "from models import Model; from skills import Skill; print('OK')"
```

Or run a live example (requires `ANTHROPIC_API_KEY`):

```bash
python examples/simple/01_skill.py
```

Expected output:
```
Machine learning is a branch of artificial intelligence that enables computers
to learn from data and improve their performance without being explicitly programmed.
```
