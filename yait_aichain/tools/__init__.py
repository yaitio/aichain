"""What a model, a chain and your code can all call: the tool factories
(``search``, ``fetch``, ``decide``, ``chunk``, ``embed``, ``rerank``,
``speech``, ``transcribe``, ``image``, ``video``, ``wait``, ``gate``), ``@tool``, the
``Tool`` base, the permission vocabulary, and the classes behind each factory
by their own names. See docs/tools/index.md.
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
from .decision   import Decision, decisionInception, decisionOpenAI, decisionTypeSafe
from .embed      import Embedder, EmbeddingResult, EmbeddingOpenAI, EmbeddingCohere, EmbeddingVoyage, EmbeddingGoogle, EmbeddingQwen
from .rerank     import RerankBase, RerankResult, RerankCohere, RerankVoyage, RerankQwen
from .speech     import Speech, ttsGoogle, ttsOpenAI, ttsQwen, ttsXAI
from .transcribe import Transcription, sttGoogle, sttOpenAI
from .image._tool import Image
from .video._tool import Video
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
    from .video      import video
else:
    from . import search, fetch, embed, rerank, speech, transcribe, chunk, image, video
    from . import decision as decide

__all__ = [
    # making and calling a tool
    "tool", "media_part", "Tool", "FunctionTool", "ToolResult", "Record",
    # the factories
    "search", "fetch", "decide", "chunk", "embed", "rerank", "speech", "transcribe", "image",
    "video", "wait", "gate",
    # the classes behind them
    "Search", "searchPerplexity", "searchBrave", "searchExa", "searchFirecrawl", "searchTavily",
    "searchParallel",
    "Fetch", "fetchFirecrawl", "fetchTavily", "fetchExa", "fetchParallel", "fetchLocal",
    "Decision", "decisionInception", "decisionOpenAI", "decisionTypeSafe",
    "Chunk",
    "Embedder", "EmbeddingResult", "EmbeddingOpenAI", "EmbeddingCohere", "EmbeddingVoyage",
    "EmbeddingGoogle", "EmbeddingQwen",
    "RerankBase", "RerankResult", "RerankCohere", "RerankVoyage", "RerankQwen",
    "Speech", "ttsOpenAI", "ttsGoogle", "ttsXAI", "ttsQwen",
    "Transcription", "sttOpenAI", "sttGoogle",
    "Image", "Video", "Wait", "Gate",
    # permissions
    "PermissionPolicy", "ApprovalRequest", "ApprovalDecision",
    "RISK_CLASSES", "DECISIONS",
    "READ", "DRAFT", "WRITE", "EXTERNAL", "FINANCIAL", "DESTRUCTIVE", "PRIVILEGED",
    "ALLOW", "APPROVE", "DENY",
]
