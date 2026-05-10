"""
tools
=====

Public API for the aichain 2.0 tool layer.

A ``Tool`` wraps any callable unit of work behind a uniform interface:

    tool.run(input, options=None)   — execute and return the raw result
    tool(input, options=None)       — safe wrapper; always returns ToolResult

Tool groups
-----------

search/          Web search — all return plain-text result strings.
  searchPerplexity, searchBrave, searchOpenAI, searchSerp

embedding/       Text embeddings — returns EmbeddingResult with float vectors.
  Embedding(model)  factory → EmbeddingOpenAI | EmbeddingCohere |
                              EmbeddingVoyage | EmbeddingGoogle | EmbeddingQwen
  run(input, options)    — embed one text or a short list
  batch(input, options)  — embed a list explicitly

convert/         Format conversion — all return str (path or content).
  convertToMD       file/URL → Markdown
  convertToHTML     Markdown → HTML (or LaTeX / normalised Markdown)
  convertToPDF      HTML → PDF (returns saved file path)
  convertToSpeech   text → audio file (TTS, returns saved file path)
  convertToText     audio file → transcript string (STT)

  TTS(provider)     factory → ttsOpenAI | ttsGoogle | ttsXAI | ttsQwen
  STT(provider)     factory → sttOpenAI | sttGoogle | sttXAI | sttQwen

rest_api         Universal REST endpoint tool.
  RestApiTool       one instance = one configured endpoint; drop any REST
                    call into a Chain, Agent, or script with no boilerplate.
                    Supports path / query / body params, Bearer / Cookie /
                    Basic / API-key auth, URL templating, JSON auto-parse.
                    See examples/booking_api.py for a full Chain example.

vectordb/        Provider-agnostic vector database interface.
  VectorDB(provider, collection, *, embedder, ...)
                    Factory → VectorStore (Chroma, Pinecone, Qdrant).
  vectorQuery(store)   — semantic RAG retrieval           [main feature]
  vectorUpsert(store)  — insert / update documents
  vectorFetch(store)   — retrieve by ID
  vectorDelete(store)  — delete by ID or filter
  VectorRecord         — uniform data model (id, text, score, metadata)

reranking/       Provider-agnostic reranking for RAG pipelines.
  Reranker(model)   factory → RerankCohere | RerankVoyage | RerankQwen
  .rerank(query, docs, top_n)  → RerankResult
  .run(docs, options)          → list[dict]  (Tool interface for Chains)

mcp/             Model Context Protocol integration.
  MCPTool           one instance = one tool on an MCP server; works in
                    Chains, Agents, and direct calls with no async code.
  MCPTools(server)  factory → list[MCPTool], one per tool on the server.
                    Supports STDIO (subprocess), Streamable HTTP, SSE.
                    Requires: pip install fastmcp

Infrastructure
--------------
  SectionContextTool   rolling-context queue manager for sectional reports
  Tool, ToolResult     base class and return type

Backward-compatible aliases
---------------------------
All previous class names still work:
  PerplexitySearchTool, BraveSearchTool, OpenAIWebSearchTool, SerpApiTool
  MarkItDownTool, MistletoeTool, WeasyprintTool
  DeepLTranslateTool, DeepLRephraseTool
  LatePublishTool, LateAccountsTool
  OpenAIEmbedder, CohereEmbedder, VoyageEmbedder, GoogleEmbedder
"""

from ._base           import Tool, ToolResult

# ── Search ────────────────────────────────────────────────────────────────────
from .search import (
    Search,
    searchPerplexity,
    searchBrave,
    searchOpenAI,
    searchSerp,
    # aliases
    PerplexitySearchTool,
    BraveSearchTool,
    OpenAIWebSearchTool,
    SerpApiTool,
)

# ── Embedding ─────────────────────────────────────────────────────────────────
from .embedding import (
    Embedding,
    Embedder,
    EmbeddingResult,
    EmbeddingOpenAI,
    EmbeddingCohere,
    EmbeddingVoyage,
    EmbeddingGoogle,
    EmbeddingQwen,
    # aliases
    OpenAIEmbedder,
    CohereEmbedder,
    VoyageEmbedder,
    GoogleEmbedder,
)

# ── Convert ───────────────────────────────────────────────────────────────────
from .convert import (
    convertToMD,
    convertToHTML,
    convertToPDF,
    convertToSpeech,
    convertToText,
    # TTS providers
    ttsOpenAI,
    ttsGoogle,
    ttsXAI,
    ttsQwen,
    # STT providers
    sttOpenAI,
    sttGoogle,
    sttXAI,
    sttQwen,
    # factories
    TTS,
    STT,
    # aliases
    MarkItDownTool,
    MistletoeTool,
    WeasyprintTool,
)


# ── Universal REST API ────────────────────────────────────────────────────────
from .rest_api import RestApiTool

# ── Vector DB ────────────────────────────────────────────────────────────────
from .vectordb import (
    VectorDB,
    VectorStore,
    VectorRecord,
    VectorBackend,
    ChromaBackend,
    PineconeBackend,
    QdrantBackend,
    vectorChunk,
    vectorQuery,
    vectorUpsert,
    vectorFetch,
    vectorDelete,
    VectorChunkTool,
    VectorQueryTool,
    VectorUpsertTool,
    VectorFetchTool,
    VectorDeleteTool,
)

# ── Reranking ────────────────────────────────────────────────────────────────
from .reranking import (
    Reranker,
    RerankBase,
    RerankResult,
    RerankCohere,
    RerankVoyage,
    RerankQwen,
)

# ── MCP (Model Context Protocol) ─────────────────────────────────────────────
from .mcp import MCPTool, MCPTools


__all__ = [
    # ── Base ──────────────────────────────────────────────────────────────
    "Tool",
    "ToolResult",
    # ── Infrastructure ────────────────────────────────────────────────────
    "SectionContextTool",
    # ── Search ────────────────────────────────────────────────────────────
    "Search",
    "searchPerplexity",
    "searchBrave",
    "searchOpenAI",
    "searchSerp",
    # search aliases
    "PerplexitySearchTool",
    "BraveSearchTool",
    "OpenAIWebSearchTool",
    "SerpApiTool",
    # ── Embedding ─────────────────────────────────────────────────────────
    "Embedding",
    "Embedder",
    "EmbeddingResult",
    "EmbeddingOpenAI",
    "EmbeddingCohere",
    "EmbeddingVoyage",
    "EmbeddingGoogle",
    "EmbeddingQwen",
    # embedding aliases
    "OpenAIEmbedder",
    "CohereEmbedder",
    "VoyageEmbedder",
    "GoogleEmbedder",
    # ── Convert ───────────────────────────────────────────────────────────
    "convertToMD",
    "convertToHTML",
    "convertToPDF",
    "convertToSpeech",
    "convertToText",
    # TTS providers
    "ttsOpenAI",
    "ttsGoogle",
    "ttsXAI",
    "ttsQwen",
    # STT providers
    "sttOpenAI",
    "sttGoogle",
    "sttXAI",
    "sttQwen",
    # factories
    "TTS",
    "STT",
    # convert aliases
    "MarkItDownTool",
    "MistletoeTool",
    "WeasyprintTool",
    # ── Universal REST API ────────────────────────────────────────────────
    "RestApiTool",
    # ── MCP (Model Context Protocol) ─────────────────────────────────────
    # ── Vector DB ───────────────────────────────────────────────────────
    "VectorDB",
    "VectorStore",
    "VectorRecord",
    "VectorBackend",
    "ChromaBackend",
    "PineconeBackend",
    "QdrantBackend",
    "vectorChunk",
    "vectorQuery",
    "vectorUpsert",
    "vectorFetch",
    "vectorDelete",
    "VectorChunkTool",
    "VectorQueryTool",
    "VectorUpsertTool",
    "VectorFetchTool",
    "VectorDeleteTool",
    # ── Reranking ─────────────────────────────────────────────────────────────
    "Reranker",
    "RerankBase",
    "RerankResult",
    "RerankCohere",
    "RerankVoyage",
    "RerankQwen",
    "MCPTool",
    "MCPTools",
]
