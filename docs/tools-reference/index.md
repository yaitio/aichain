# Built-in tools

Tools are the library's bridge to the outside world — every action an Agent
takes that isn't an LLM call, and every deterministic step in a Chain that isn't
text generation, goes through a Tool.

All built-in tools follow the same contract described in
[primitives/tools.md](../primitives/tools.md): a `name`, a `description`, a
JSON-Schema `parameters` block, and a `run(input, options=None)`
method. Every table below is generated from the tool classes themselves. They plug into
Chains and Agents identically.

---

## At a glance

<!-- g:tools-index -->
| Group | Tool | Class | What it does | Key | Risk |
|---|---|---|---|---|---|
| Search | [`searchPerplexity`](perplexity-search.md) | `PerplexitySearchTool` | Web search powered by the Perplexity Search API. | `PERPLEXITY_API_KEY` | `write` |
| Search | [`searchBrave`](brave-search.md) | `BraveSearchTool` | Web search powered by the Brave Search API. | `BRAVE_SEARCH_API_KEY` | `write` |
| Search | [`searchSerp`](serp-api.md) | `SerpApiTool` | Multi-engine web search powered by SerpAPI. | `SERPAPI_API_KEY` | `write` |
| Search | [`searchOpenAI`](openai-web-search.md) | `OpenAIWebSearchTool` | Web search powered by OpenAI's built-in `web_search_preview` tool. | `OPENAI_API_KEY` | `write` |
| Conversion | [`convertToMD`](markitdown.md) | `MarkItDownTool` | Convert a file or URL to Markdown text. | — | `write` |
| Conversion | [`convertToHTML`](mistletoe.md) | `MistletoeTool` | Convert Markdown text to HTML, LaTeX, or normalised Markdown. | — | `write` |
| Conversion | [`convertToPDF`](weasyprint.md) | `WeasyprintTool` | Render an HTML document to PDF. | — | `write` |
| Conversion | [`ttsOpenAI`](speech.md#ttsopenai) | `ttsOpenAI` | Text-to-speech via the OpenAI TTS API. | `OPENAI_API_KEY` | `write` |
| Conversion | [`ttsGoogle`](speech.md#ttsgoogle) | `ttsGoogle` | Text-to-speech via Google Cloud Text-to-Speech. | `GOOGLE_AI_API_KEY`, `GOOGLE_API_KEY` | `write` |
| Conversion | [`ttsXAI`](speech.md#ttsxai) | `ttsXAI` | Text-to-speech via the xAI TTS API. | `XAI_API_KEY` | `write` |
| Conversion | [`ttsQwen`](speech.md#ttsqwen) | `ttsQwen` | Text-to-speech via the Alibaba DashScope CosyVoice / Qwen3-TTS API. | `DASHSCOPE_API_KEY` | `write` |
| Conversion | [`sttOpenAI`](speech.md#sttopenai) | `sttOpenAI` | Speech-to-text via the OpenAI Whisper API. | `OPENAI_API_KEY` | `write` |
| Conversion | [`sttGoogle`](speech.md#sttgoogle) | `sttGoogle` | Speech-to-text via Google Cloud Speech-to-Text. | `GOOGLE_AI_API_KEY`, `GOOGLE_API_KEY` | `write` |
| Conversion | [`sttXAI`](speech.md#sttxai) | `sttXAI` | Speech-to-text via the xAI STT API. | `XAI_API_KEY` | `write` |
| Conversion | [`sttQwen`](speech.md#sttqwen) | `sttQwen` | Speech-to-text via the Alibaba DashScope ASR API. | `DASHSCOPE_API_KEY` | `write` |
| Local | [`local_browse`](local.md#localbrowsetool) | `LocalBrowseTool` | List directory contents as a tree. | — | `write` |
| Local | [`local_read`](local.md#localreadtool) | `LocalReadTool` | Read a file's contents. | — | `write` |
| Local | [`local_write`](local.md#localwritetool) | `LocalWriteTool` | Write content to a file. | — | `write` |
| Local | [`local_run`](local.md#localruntool) | `LocalRunTool` | Execute a shell command or Python file. | — | `write` |
| Vector store | [`vector_chunk`](vectordb.md#vectorchunktool) | `VectorChunkTool` | Split text into overlapping chunks for embedding and vector-store ingestion. | — | `write` |
| Vector store | [`vector_upsert`](vectordb.md#vectorupserttool) | `VectorUpsertTool` | Insert or update records in a `VectorStore`. | — | `write` |
| Vector store | [`vector_query`](vectordb.md#vectorquerytool) | `VectorQueryTool` | Semantic similarity search over a `VectorStore`. | — | `write` |
| Vector store | [`vector_fetch`](vectordb.md#vectorfetchtool) | `VectorFetchTool` | Retrieve records from a `VectorStore` by exact ID. | — | `write` |
| Vector store | [`vector_delete`](vectordb.md#vectordeletetool) | `VectorDeleteTool` | Delete records from a `VectorStore`. | — | `write` |
| HTTP | [`RestApiTool`](rest-api.md) | `RestApiTool` | A single REST endpoint exposed as a library Tool. | — | `write` |
| MCP | [`MCPTool`](mcp.md) | `MCPTool` | A single tool from an MCP server, exposed as a standard aichain Tool. | — | `write` |
| Pause | [`wait`](wait-gate.md#wait) | `Wait` | Pause the run until an external signal arrives. | — | `write` |
| Pause | [`Gate`](wait-gate.md#gate) | `Gate` | Gate any Tool behind an external signal. | — | `write` |
<!-- /g:tools-index -->

---

## See also

- **Tool contract & custom tools** → [primitives/tools.md](../primitives/tools.md)
- **Using tools in Chains** → [primitives/chain.md](../primitives/chain.md)
- **Using tools in Agents** → [agents/configuration.md](../agents/configuration.md)
