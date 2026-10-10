# Built-in tools

Tools are the library's bridge to the outside world — every action an Agent
takes that isn't an LLM call, and every deterministic step in a Chain that isn't
text generation, goes through a Tool.

All built-in tools follow the same contract described in
[Tools](../tools/index.md): a `name`, a `description`, a
JSON-Schema `parameters` block, and a `run(input, options=None)`
method. Every table below is generated from the tool classes themselves. They plug into
Chains and Agents identically.

---

## At a glance

<!-- g:tools-index -->
| Group | Tool | Class | What it does | Key | Risk |
|---|---|---|---|---|---|
| Search | [`searchPerplexity`](../tools/search.md#searchperplexity) | `PerplexitySearchTool` | Web search powered by the Perplexity Search API — each result with a substantive passage from the page. | `PERPLEXITY_API_KEY` | `read` |
| Search | [`searchBrave`](../tools/search.md#searchbrave) | `BraveSearchTool` | Web search powered by the Brave Search API — titles, URLs, the page's description and, with `extra_snippets`, more passages from it. | `BRAVE_SEARCH_API_KEY` | `read` |
| Search | [`searchExa`](../tools/search.md#searchexa) | `searchExa` | Web search powered by the Exa Search API — results with the passages that answer the query. | `EXA_API_KEY` | `read` |
| Search | [`searchFirecrawl`](../tools/search.md#searchfirecrawl) | `searchFirecrawl` | Web search powered by Firecrawl — results with a description and, on request, each page's content as Markdown. | `FIRECRAWL_API_KEY` | `read` |
| Search | [`searchTavily`](../tools/search.md#searchtavily) | `searchTavily` | Web search powered by Tavily — each page with the passages chosen for the query, and on request the page itself. | `TAVILY_API_KEY` | `read` |
| Search | [`searchParallel`](../tools/search.md#searchparallel) | `searchParallel` | Web search powered by Parallel — the excerpts of each page that serve the objective of the search. | `PARALLEL_API_KEY` | `read` |
| Embeddings | [`embedOpenAI`](../tools/embed.md#embeddingopenai) | `EmbeddingOpenAI` | Embedder for the OpenAI Embeddings API. | `OPENAI_API_KEY` | `read` |
| Embeddings | [`embedCohere`](../tools/embed.md#embeddingcohere) | `CohereEmbedder` | Embedder for the Cohere Embed API (v2). | `COHERE_API_KEY` | `read` |
| Embeddings | [`embedVoyage`](../tools/embed.md#embeddingvoyage) | `EmbeddingVoyage` | Embedder for the Voyage AI Embeddings API. | `VOYAGE_API_KEY` | `read` |
| Embeddings | [`embedGoogle`](../tools/embed.md#embeddinggoogle) | `EmbeddingGoogle` | Embedder for the Google AI (Gemini) Embeddings API. | `GOOGLE_AI_API_KEY`, `GOOGLE_API_KEY` | `read` |
| Embeddings | [`embedQwen`](../tools/embed.md#embeddingqwen) | `EmbeddingQwen` | Embedder for the Alibaba DashScope Embeddings API. | `DASHSCOPE_API_KEY` | `read` |
| Reranking | [`rerankCohere`](../tools/rerank.md#rerankcohere) | `RerankCohere` | Reranker for the Cohere Rerank API (v2). | `COHERE_API_KEY` | `read` |
| Reranking | [`rerankVoyage`](../tools/rerank.md#rerankvoyage) | `RerankVoyage` | Reranker for the Voyage AI Rerank API. | `VOYAGE_API_KEY` | `read` |
| Reranking | [`rerankQwen`](../tools/rerank.md#rerankqwen) | `RerankQwen` | Reranker for the Alibaba DashScope Text-Rerank API. | `DASHSCOPE_API_KEY` | `read` |
| Image | [`imageOpenai`](../tools/image.md) | `Image` | An image model as a tool — generation from a prompt, and editing when the call carries images. Made by `tools.image` — `image("openai", model="gpt-image-1.5")`. | — | `draft` |
| Fetch | [`fetchFirecrawl`](../tools/fetch.md#fetchfirecrawl) | `fetchFirecrawl` | A web page's text by Firecrawl Scrape — the main content, as Markdown, with the page's metadata. | `FIRECRAWL_API_KEY` | `read` |
| Fetch | [`fetchTavily`](../tools/fetch.md#fetchtavily) | `fetchTavily` | Web pages' text by Tavily Extract — up to 20 URLs a call, as Markdown. | `TAVILY_API_KEY` | `read` |
| Fetch | [`fetchExa`](../tools/fetch.md#fetchexa) | `fetchExa` | Web pages' text by Exa Contents — up to 100 URLs a call, with title, date and author. | `EXA_API_KEY` | `read` |
| Fetch | [`fetchParallel`](../tools/fetch.md#fetchparallel) | `fetchParallel` | Web pages' full content by Parallel Extract — up to 20 URLs a call, with title and date. | `PARALLEL_API_KEY` | `read` |
| Fetch | [`fetchLocal`](../tools/fetch.md#fetchlocal) | `fetchLocal` | A web page's text fetched from this machine and converted by MarkItDown — free. | — | `read` |
| Speech | [`ttsOpenAI`](../tools/speech.md#ttsopenai) | `ttsOpenAI` | Text-to-speech via the OpenAI TTS API. | `OPENAI_API_KEY` | `write` |
| Speech | [`ttsGoogle`](../tools/speech.md#ttsgoogle) | `ttsGoogle` | Text-to-speech via Google Cloud Text-to-Speech. | `GOOGLE_AI_API_KEY`, `GOOGLE_API_KEY` | `write` |
| Speech | [`ttsXAI`](../tools/speech.md#ttsxai) | `ttsXAI` | Text-to-speech via the xAI TTS API. | `XAI_API_KEY` | `write` |
| Speech | [`ttsQwen`](../tools/speech.md#ttsqwen) | `ttsQwen` | Text-to-speech via the Alibaba DashScope CosyVoice / Qwen3-TTS API. | `DASHSCOPE_API_KEY` | `write` |
| Speech | [`sttOpenAI`](../tools/speech.md#sttopenai) | `sttOpenAI` | Speech-to-text via the OpenAI Whisper API. | `OPENAI_API_KEY` | `write` |
| Speech | [`sttGoogle`](../tools/speech.md#sttgoogle) | `sttGoogle` | Speech-to-text via Google Cloud Speech-to-Text. | `GOOGLE_AI_API_KEY`, `GOOGLE_API_KEY` | `write` |
| Speech | [`sttXAI`](../tools/speech.md#sttxai) | `sttXAI` | Speech-to-text via the xAI STT API. | `XAI_API_KEY` | `write` |
| Speech | [`sttQwen`](../tools/speech.md#sttqwen) | `sttQwen` | Speech-to-text via the Alibaba DashScope ASR API. | `DASHSCOPE_API_KEY` | `write` |
| Local | [`local_browse`](../utilities/shell.md#localbrowsetool) | `LocalBrowseTool` | List directory contents as a tree. | — | `write` |
| Local | [`local_read`](../utilities/shell.md#localreadtool) | `LocalReadTool` | Read a file's contents. | — | `write` |
| Local | [`local_write`](../utilities/shell.md#localwritetool) | `LocalWriteTool` | Write content to a file. | — | `write` |
| Local | [`local_run`](../utilities/shell.md#localruntool) | `LocalRunTool` | Execute a shell command or Python file. | — | `write` |
| Vector store | [`vector_chunk`](../utilities/vectors.md#vectorchunktool) | `VectorChunkTool` | Split text into overlapping chunks for embedding and vector-store ingestion. | — | `write` |
| Vector store | [`vector_upsert`](../utilities/vectors.md#vectorupserttool) | `VectorUpsertTool` | Insert or update records in a `VectorStore`. | — | `write` |
| Vector store | [`vector_query`](../utilities/vectors.md#vectorquerytool) | `VectorQueryTool` | Semantic similarity search over a `VectorStore`. | — | `write` |
| Vector store | [`vector_fetch`](../utilities/vectors.md#vectorfetchtool) | `VectorFetchTool` | Retrieve records from a `VectorStore` by exact ID. | — | `write` |
| Vector store | [`vector_delete`](../utilities/vectors.md#vectordeletetool) | `VectorDeleteTool` | Delete records from a `VectorStore`. | — | `write` |
| HTTP | [`RestApiTool`](../utilities/rest.md) | `RestApiTool` | A single REST endpoint exposed as a library Tool. | — | `write` |
| MCP | [`MCPTool`](../utilities/mcp.md) | `MCPTool` | A single tool from an MCP server, exposed as a standard aichain Tool. | — | `write` |
| Pause | [`wait`](../tools/wait-gate.md#wait) | `Wait` | Pause the run until an external signal arrives. | — | `write` |
| Pause | [`Gate`](../tools/wait-gate.md#gate) | `Gate` | Gate any Tool behind an external signal. | — | `write` |
| Decision | [`decisionOpenAI`](../tools/decide.md#decisionopenai) | `decisionOpenAI` | Typed questions about a text — and images — answered with probabilities, by OpenAI's Decisions API. | `OPENAI_API_KEY` | `read` |
| Decision | [`decisionTypeSafe`](../tools/decide.md#decisiontypesafe) | `decisionTypeSafe` | Typed questions about a text, answered with probabilities, by TypeSafe's Jev. | `TYPESAFE_API_KEY` | `read` |
<!-- /g:tools-index -->

---

## See also

- **Tool contract & custom tools** → [Tools](../tools/index.md)
- **Using tools in Chains** → [Chain](../pipelines/chain.md)
- **Using tools in Agents** → [agents/configuration.md](../agents/configuration.md)
