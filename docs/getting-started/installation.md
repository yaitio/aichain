# Installation

## Library

aichain 2.0 is not yet published to PyPI. Clone the repository and add the root directory to your Python path.

```bash
git clone <repo-url> aichain_2.0
cd aichain_2.0
```

From your own script, add the library root to `sys.path` before importing:

```python
import sys
sys.path.insert(0, "/path/to/aichain_2.0")

from models import Model
from skills import Skill
```

---

## Dependencies

### Required

```bash
pip install pyyaml          # Skill.save() / Skill.load() / Chain.save() / Chain.load()
```

All HTTP calls use `urllib3`, which is installed by default in most Python environments. If it is missing:

```bash
pip install urllib3
```

### Optional — by feature

| Feature | Package | Install |
|---|---|---|
| HTML export | mistletoe | `pip install mistletoe` |
| PDF export | weasyprint | `pip install weasyprint` |
| URL / file → Markdown | markitdown | `pip install markitdown` |
| DeepL translation | *(none — uses urllib3)* | — |
| All search tools | *(none — uses urllib3)* | — |

#### WeasyPrint system libraries (macOS)

WeasyPrint needs Pango and GLib, which are not bundled in the Python package. Install them with Homebrew:

```bash
brew install pango glib
```

Then tell the dynamic linker where to find them. Add this to your `~/.zshrc` (or `~/.bashrc`) to make it permanent:

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix)/lib:$DYLD_LIBRARY_PATH"
```

Apply immediately without restarting the shell:

```bash
source ~/.zshrc
```

---

## API keys

Each provider requires its own key, set as an environment variable. You only need keys for the providers you actually use.

| Provider | Environment variable | Get a key |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| Google AI | `GOOGLE_AI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |
| xAI | `XAI_API_KEY` | [console.x.ai](https://console.x.ai) |
| Perplexity | `PERPLEXITY_API_KEY` | [perplexity.ai/settings/api](https://perplexity.ai/settings/api) |
| Brave Search | `BRAVE_SEARCH_API_KEY` | [brave.com/search/api](https://brave.com/search/api) |
| SerpAPI | `SERP_API_KEY` | [serpapi.com](https://serpapi.com) |
| DeepL | `DEEPL_API_KEY` | [deepl.com/pro-api](https://www.deepl.com/pro-api) |

Set them for the current shell session:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export PERPLEXITY_API_KEY="pplx-..."
```

Or put them in a `.env` file and source it:

```bash
# .env
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

```bash
source .env
```

> Keys are **never** written to YAML files by `save()`. They are always resolved from environment variables at load time.

---

## Verify installation

Run the built-in check script. It tests every provider whose key is set and skips the rest:

```bash
python examples/text_to_text.py
```

Expected output for a single provider:

```
  ── ANTHROPIC  (5 model(s)) ──────────────────────────────────────────

  ▸ claude-opus-4-6
    ✓  3.2s
      1. Readability: code should be easy to read and understand...
      2. ...
```
