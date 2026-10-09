"""
tools
=====

Public API for the aichain 2.0 tool layer.

A ``Tool`` wraps any callable unit of work behind a uniform interface:

    tool.run(input, options=None)   — execute and return the raw result
    tool(input, options=None)       — safe wrapper; always returns ToolResult

Tool groups
-----------

search/          Web search — search("exa").run(query, options) → a Record.
  searchPerplexity, searchBrave, searchExa, searchFirecrawl, searchTavily,
  searchParallel — one format of options and results

fetch/           A web page's text from its URL — fetch("firecrawl").run(urls) → a Record.
  fetchFirecrawl, fetchTavily, fetchExa, fetchParallel, and fetchLocal
  (MarkItDown, free; needs pip install "yait-aichain[fetch]")

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

decision/        Typed questions about a text, answered with probabilities —
                 decide("openai", questions={...}).run(text) → a Record.
  decisionOpenAI, decisionTypeSafe — one format of questions and answers

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

local/           Local filesystem tools — sandboxed to a root directory.
  LocalTools(root_dir) — returns all 4 tools pre-configured for an Agent
  localBrowse(root_dir)  — list directory tree
  localRead(root_dir)    — read file contents (with line range)
  localWrite(root_dir)   — create or overwrite a file
  localRun(root_dir)     — execute a shell command or Python file

mcp/             Model Context Protocol integration.
  MCPTool           one instance = one tool on an MCP server; works in
                    Chains, Agents, and direct calls with no async code.
  MCPTools(server)  factory → list[MCPTool], one per tool on the server.
                    Supports STDIO (subprocess), Streamable HTTP, SSE.
                    Requires: pip install fastmcp

Infrastructure
--------------
  Tool, ToolResult     base class and return type

Backward-compatible aliases
---------------------------
All previous class names still work:
  PerplexitySearchTool, BraveSearchTool
  MarkItDownTool, MistletoeTool, WeasyprintTool
  OpenAIEmbedder, CohereEmbedder, VoyageEmbedder, GoogleEmbedder
"""

from ._base           import Tool, ToolResult, Record, FunctionTool, tool
from .general         import Wait, Gate, wait, gate
from ._permissions    import (
    PermissionPolicy, ApprovalRequest, ApprovalDecision,
    RISK_CLASSES, DECISIONS,
    READ, DRAFT, WRITE, EXTERNAL, FINANCIAL, DESTRUCTIVE, PRIVILEGED,
    ALLOW, APPROVE, DENY,
)

# ── Search ────────────────────────────────────────────────────────────────────
from .search import (
    Search,
    searchPerplexity,
    searchBrave,
    searchExa,
    searchFirecrawl,
    searchTavily,
    searchParallel,
    # aliases
    PerplexitySearchTool,
    BraveSearchTool,
)

# ── Fetch ─────────────────────────────────────────────────────────────────────
from .fetch import (
    Fetch,
    fetchFirecrawl,
    fetchTavily,
    fetchExa,
    fetchParallel,
    fetchLocal,
)

# ── Embedding ─────────────────────────────────────────────────────────────────
from .embed import (
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

# ── Speech and transcription ─────────────────────────────────────────────────
# Imported from tools.convert until 3.8.0, which removed the convert tools.
from .speech import TTS, convertToSpeech, ttsGoogle, ttsOpenAI, ttsQwen, ttsXAI
from .transcribe import STT, convertToText, sttGoogle, sttOpenAI, sttQwen, sttXAI

# ── Decision ──────────────────────────────────────────────────────────────────
from .decision import Decision, decisionOpenAI, decisionTypeSafe

# ── Reranking ────────────────────────────────────────────────────────────────
from .rerank import (
    Reranker,
    RerankBase,
    RerankResult,
    RerankCohere,
    RerankVoyage,
    RerankQwen,
)

# ── Utilities, under their tools/ names ──────────────────────────────────────
# The vector store, REST, MCP and local files are utilities since 3.6.0 and live
# in yait_aichain.utilities. Their old names still import from here, lazily: a
# utility imports tools._base, so importing it eagerly here would be a cycle.
_FROM_UTILITIES = {
    "yait_aichain.utilities.vectors": (
        "VectorDB", "VectorStore", "VectorRecord", "VectorBackend", "ChromaBackend",
        "PineconeBackend", "QdrantBackend", "vectorChunk", "vectorQuery", "vectorUpsert",
        "vectorFetch", "vectorDelete", "VectorChunkTool", "VectorQueryTool", "VectorUpsertTool",
        "VectorFetchTool", "VectorDeleteTool"),
    "yait_aichain.utilities.rest._tool": ("RestApiTool",),
    "yait_aichain.utilities.shell": (
        "LocalTools", "localBrowse", "LocalBrowseTool", "localRead", "LocalReadTool",
        "localWrite", "LocalWriteTool", "localRun", "LocalRunTool"),
    "yait_aichain.utilities.mcp._tools": ("MCPTool", "MCPTools"),
}
_LAZY = {name: module for module, names in _FROM_UTILITIES.items() for name in names}


def __getattr__(name):
    import importlib
    # importlib, not `from . import`: that form calls this __getattr__ again.
    removed = importlib.import_module(f"{__name__}._convert_removed")
    if name in removed.REMOVED:                          # removed in 3.8.0: say what replaced it
        raise removed.removed(name)
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    # A module that lived here before it moved — tools.vectordb, tools.embedding —
    # is still an attribute, as it was when this package imported it.
    _moved = importlib.import_module("yait_aichain._moved")
    if _moved.new_path(f"{__name__}.{name}") is not None:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


# The factories: each package is callable — `search("exa")` — and stays the
# package, so every import form and `mock.patch` path into it still resolves.
# Type checkers see the functions.
from typing import TYPE_CHECKING as _TC
if _TC:
    from .search     import search
    from .fetch      import fetch
    from .decision   import decide
    from .embed      import embed
    from .rerank     import rerank
    from .speech     import speech
    from .transcribe import transcribe
    from .chunk      import chunk
else:
    from . import search, fetch, embed, rerank, speech, transcribe, chunk
    from . import decision as decide

__all__ = [
    "Decision",
    "decide",
    "decisionOpenAI",
    "decisionTypeSafe",
    "searchExa",
    "searchFirecrawl",
    "searchTavily",
    "searchParallel",
    "search",
    "Record",
    "tool",
    "FunctionTool",
    "embed",
    "rerank",
    "speech",
    "transcribe",
    "chunk",
    "wait",
    "gate",
    "fetch",
    "Fetch",
    "fetchFirecrawl",
    "fetchTavily",
    "fetchExa",
    "fetchParallel",
    "fetchLocal",
    # ── Base ──────────────────────────────────────────────────────────────
    "Tool",
    "ToolResult",
    "Wait",
    "Gate",
    # ── Permissions (1.4.4) ───────────────────────────────────────────────
    "PermissionPolicy", "ApprovalRequest", "ApprovalDecision",
    "RISK_CLASSES", "DECISIONS",
    "READ", "DRAFT", "WRITE", "EXTERNAL", "FINANCIAL", "DESTRUCTIVE", "PRIVILEGED",
    "ALLOW", "APPROVE", "DENY",
    # ── Infrastructure ────────────────────────────────────────────────────
    # ── Search ────────────────────────────────────────────────────────────
    "Search",
    "searchPerplexity",
    "searchBrave",
    # search aliases
    "PerplexitySearchTool",
    "BraveSearchTool",
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
    # ── Speech and transcription ──────────────────────────────────────────
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
    # ── Local filesystem ─────────────────────────────────────────────────────
    "LocalTools",
    "localBrowse", "LocalBrowseTool",
    "localRead",   "LocalReadTool",
    "localWrite",  "LocalWriteTool",
    "localRun",    "LocalRunTool",
    "MCPTool",
    "MCPTools",
]
