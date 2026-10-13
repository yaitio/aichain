"""
tools
=====

What a model, a chain and your code can all call. A tool is made by its
service's name and called with ``.run()``; what it *is* is set when it is made,
what one call asks goes in ``.run()``:

    search("exa").run("vector databases", {"max_results": 5})
    fetch("local").run("https://example.com")
    decide("openai", questions={...}).run(text)
    chunk(max_chars=800).run(text)
    embed("openai", model="text-embedding-3-small").run(texts)
    rerank("cohere", model="rerank-v3.5").run(docs, {"query": q})
    speech("openai").run(text, {"voice": "nova"})
    transcribe("openai").run("call.mp3")
    image("openai", model="gpt-image-1.5").run(prompt)
    wait(reason="…") · gate(tool, reason="…")

Any function becomes one with ``@tool``; a ``Tool`` subclass is the way for a
tool with state. ``tool(...)`` called on a tool returns a ``ToolResult`` and
never raises; ``.run()`` returns the data. Utilities — a vector store, a
folder, a REST API, an MCP server — are in ``yait_aichain.utilities``.

The classes behind each factory — ``searchExa``, ``fetchLocal``,
``EmbeddingOpenAI`` and the rest — are importable by their own names.
"""

from ._base           import Tool, ToolResult, Record, FunctionTool, tool
from ..models._calls  import media_part
from .general         import Wait, Gate, wait, gate
from ._permissions    import (
    PermissionPolicy, ApprovalRequest, ApprovalDecision,
    RISK_CLASSES, DECISIONS,
    READ, DRAFT, WRITE, EXTERNAL, FINANCIAL, DESTRUCTIVE, PRIVILEGED,
    ALLOW, APPROVE, DENY,
)

from .search     import Search, searchPerplexity, searchBrave, searchExa, searchFirecrawl, searchTavily, searchParallel
from .fetch      import Fetch, fetchFirecrawl, fetchTavily, fetchExa, fetchParallel, fetchLocal
from .decision   import Decision, decisionOpenAI, decisionTypeSafe
from .embed      import Embedder, EmbeddingResult, EmbeddingOpenAI, EmbeddingCohere, EmbeddingVoyage, EmbeddingGoogle, EmbeddingQwen
from .rerank     import RerankBase, RerankResult, RerankCohere, RerankVoyage, RerankQwen
from .speech     import Speech, ttsGoogle, ttsOpenAI, ttsQwen, ttsXAI
from .transcribe import Transcription, sttGoogle, sttOpenAI
from .image._tool import Image
from .chunk._chunk import Chunk

def __getattr__(name):
    # A name from an older tutorial raises with the one to write instead.
    from .._hints import TOOL_NAMES, tool_name
    if name in TOOL_NAMES:
        raise tool_name(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The factories: each package is callable — `search("exa")` — and stays the
# package, so `import yait_aichain.tools.search.exa` and `mock.patch` paths into
# it resolve. Type checkers see the functions.
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
    from .image      import image
else:
    from . import search, fetch, embed, rerank, speech, transcribe, chunk, image
    from . import decision as decide

__all__ = [
    # making and calling a tool
    "tool", "media_part", "Tool", "FunctionTool", "ToolResult", "Record",
    # the factories
    "search", "fetch", "decide", "chunk", "embed", "rerank", "speech", "transcribe", "image",
    "wait", "gate",
    # the classes behind them
    "Search", "searchPerplexity", "searchBrave", "searchExa", "searchFirecrawl", "searchTavily",
    "searchParallel",
    "Fetch", "fetchFirecrawl", "fetchTavily", "fetchExa", "fetchParallel", "fetchLocal",
    "Decision", "decisionOpenAI", "decisionTypeSafe",
    "Chunk",
    "Embedder", "EmbeddingResult", "EmbeddingOpenAI", "EmbeddingCohere", "EmbeddingVoyage",
    "EmbeddingGoogle", "EmbeddingQwen",
    "RerankBase", "RerankResult", "RerankCohere", "RerankVoyage", "RerankQwen",
    "Speech", "ttsOpenAI", "ttsGoogle", "ttsXAI", "ttsQwen",
    "Transcription", "sttOpenAI", "sttGoogle",
    "Image", "Wait", "Gate",
    # permissions
    "PermissionPolicy", "ApprovalRequest", "ApprovalDecision",
    "RISK_CLASSES", "DECISIONS",
    "READ", "DRAFT", "WRITE", "EXTERNAL", "FINANCIAL", "DESTRUCTIVE", "PRIVILEGED",
    "ALLOW", "APPROVE", "DENY",
]
