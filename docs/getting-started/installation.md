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

Set keys only for the providers you use:

| Provider | Environment variable | Get a key |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| Google AI | `GOOGLE_AI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |
| xAI | `XAI_API_KEY` | [console.x.ai](https://console.x.ai) |
| Perplexity | `PERPLEXITY_API_KEY` | [perplexity.ai/settings/api](https://perplexity.ai/settings/api) |
| Kimi | `MOONSHOT_API_KEY` | [platform.moonshot.cn](https://platform.moonshot.cn) |
| DeepSeek | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |
| Qwen | `DASHSCOPE_API_KEY` | [dashscope.aliyuncs.com](https://dashscope.aliyuncs.com) |
| Cohere | `COHERE_API_KEY` | [dashboard.cohere.com](https://dashboard.cohere.com) |
| Voyage | `VOYAGE_API_KEY` | [dash.voyageai.com](https://dash.voyageai.com) |
| Brave Search | `BRAVE_SEARCH_API_KEY` | [brave.com/search/api](https://brave.com/search/api) |
| SerpAPI | `SERP_API_KEY` | [serpapi.com](https://serpapi.com) |
| DeepL | `DEEPL_API_KEY` | [deepl.com/pro-api](https://www.deepl.com/pro-api) |

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
