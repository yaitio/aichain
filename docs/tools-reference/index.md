# Built-in tools

Tools are the library's bridge to the outside world — every action an Agent takes that isn't an LLM call, and every deterministic step in a Chain that isn't text generation, goes through a Tool.

All built-in tools follow the same contract described in [primitives/tools.md](../primitives/tools.md): a `name`, a `description`, a JSON-Schema `parameters` block, and a `run(**kwargs) -> str | dict` method. They plug into Chains and Agents identically.

---

## At a glance

| Tool | Class | Purpose | Requires |
|---|---|---|---|
| [`perplexity_search`](perplexity-search.md) | `PerplexitySearchTool` | Live web search + citations via Perplexity | `PERPLEXITY_API_KEY` |
| [`brave_search`](brave-search.md) | `BraveSearchTool` | Web search via Brave's index | `BRAVE_SEARCH_API_KEY` |
| [`serp_api_search`](serp-api.md) | `SerpApiTool` | 50+ search engines via SerpAPI | `SERPAPI_API_KEY` |
| [`openai_web_search`](openai-web-search.md) | `OpenAIWebSearchTool` | Web search via OpenAI's Responses API | `OPENAI_API_KEY` |
| [`markitdown`](markitdown.md) | `MarkItDownTool` | Convert files/URLs to Markdown | — |
| [`mistletoe`](mistletoe.md) | `MistletoeTool` | Convert Markdown → HTML / LaTeX | — |
| [`weasyprint`](weasyprint.md) | `WeasyprintTool` | Render HTML → PDF | — |
| [`deepl_translate`](deepl.md) | `DeepLTranslateTool` | Translate text via DeepL | `DEEPL_API_KEY` |
| [`deepl_rephrase`](deepl.md) | `DeepLRephraseTool` | Rephrase / improve text via DeepL | `DEEPL_API_KEY` |
| [`section_context`](section-context.md) | `SectionContextTool` | Rolling section context for long-document generation | — |
| [`late_accounts`](late.md) | `LateAccountsTool` | List connected social accounts (Late) | `LATE_API_KEY` |
| [`late_publish`](late.md) | `LatePublishTool` | Publish posts to social platforms (Late) | `LATE_API_KEY` |

---

## Grouping

**Search / retrieval**
- Perplexity (citations-first, best for research agents)
- Brave (raw web index, privacy-preserving)
- SerpAPI (any search engine, any locale)
- OpenAI web search (Responses API, US-centric)

**Document conversion**
- MarkItDown (anything → Markdown)
- Mistletoe (Markdown → HTML / LaTeX)
- Weasyprint (HTML → PDF)

**Language / rewriting**
- DeepL translate
- DeepL rephrase

**Long-document helpers**
- Section context (rolling window over completed sections)

**Social publishing**
- Late accounts
- Late publish

---

## See also

- **Tool contract & custom tools** → [primitives/tools.md](../primitives/tools.md)
- **Using tools in Chains** → [primitives/chain.md](../primitives/chain.md)
- **Using tools in Agents** → [agents/configuration.md](../agents/configuration.md)
