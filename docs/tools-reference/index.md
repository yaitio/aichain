# Built-in tools

Tools are the library's bridge to the outside world — every action an Agent
takes that isn't an LLM call, and every deterministic step in a Chain that isn't
text generation, goes through a Tool.

All built-in tools follow the same contract described in
[primitives/tools.md](../primitives/tools.md): a `name`, a `description`, a
JSON-Schema `parameters` block, and a `run(input, options=None) -> str | dict`
method (multi-parameter tools take their fields as kwargs). They plug into
Chains and Agents identically.

---

## At a glance

| Tool | Class | Purpose | Requires |
|---|---|---|---|
| [`searchPerplexity`](perplexity-search.md) | `PerplexitySearchTool` | Live web search + snippets via Perplexity | `PERPLEXITY_API_KEY` |
| [`searchBrave`](brave-search.md) | `BraveSearchTool` | Web search via Brave's index | `BRAVE_SEARCH_API_KEY` |
| [`searchSerp`](serp-api.md) | `SerpApiTool` | 50+ search engines via SerpAPI | `SERPAPI_API_KEY` |
| [`searchOpenAI`](openai-web-search.md) | `OpenAIWebSearchTool` | Web search via OpenAI's Responses API | `OPENAI_API_KEY` |
| [`convertToMD`](markitdown.md) | `MarkItDownTool` | Convert files/URLs to Markdown | — |
| [`convertToHTML`](mistletoe.md) | `MistletoeTool` | Convert Markdown → HTML | — |
| [`convertToPDF`](weasyprint.md) | `WeasyprintTool` | Render HTML → PDF | — |
| `convertToText` | — | Extract plain text from files/URLs | — |
| `TTS` / `STT` | `ttsOpenAI/Google/XAI/Qwen`, `sttOpenAI/Google/XAI/Qwen` | Text↔speech | provider key |
| `Embedding` | `EmbeddingOpenAI/Cohere/Voyage/Google/Qwen` | Text embeddings for RAG | provider key |
| `RestApiTool` | `RestApiTool` | Call any REST endpoint as a tool | per-API |
| `Wait` / `Gate` | `Wait`, `Gate` | Pause a run for an external signal | — |

---

## Grouping

**Search / retrieval**
- Perplexity (snippets-first, best for research agents)
- Brave (raw web index, privacy-preserving)
- SerpAPI (any search engine, any locale)
- OpenAI web search (Responses API)

**Document conversion**
- MarkItDown (anything → Markdown)
- Mistletoe (Markdown → HTML)
- Weasyprint (HTML → PDF)
- `convertToText` (files/URLs → plain text)

**Speech**
- `TTS` — text → audio (OpenAI / Google / xAI / Qwen)
- `STT` — audio → text (OpenAI / Google / xAI / Qwen)

**Embeddings & vectors**
- `Embedding` — OpenAI / Cohere / Voyage / Google / Qwen
- `VectorDB` / `VectorStore` — Chroma / Qdrant / Pinecone

**HTTP**
- `RestApiTool` — wrap any REST endpoint

**Suspend / resume**
- `Wait`, `Gate` — pause until an external signal ([State](../primitives/state.md))

---

## See also

- **Tool contract & custom tools** → [primitives/tools.md](../primitives/tools.md)
- **Using tools in Chains** → [primitives/chain.md](../primitives/chain.md)
- **Using tools in Agents** → [agents/configuration.md](../agents/configuration.md)
