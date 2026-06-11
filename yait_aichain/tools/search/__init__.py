"""
tools.search
============

Web search tools — all share the ``run(input, options=None) -> str`` interface.

Tools
-----
searchPerplexity   Perplexity Search API — rich snippets from page content.
searchBrave        Brave Search — ranked links with titles + description snippets.
searchOpenAI       OpenAI web_search_preview — synthesized answer with citations.
searchSerp         SerpAPI — 50+ engines (Google, Bing, DuckDuckGo, Baidu, …).

Backward-compatible aliases
---------------------------
PerplexitySearchTool, BraveSearchTool, OpenAIWebSearchTool, SerpApiTool
"""

from ._base      import Search
from .perplexity import searchPerplexity
from .brave      import searchBrave
from .openai     import searchOpenAI
from .serp       import searchSerp

# ── Backward-compatible aliases ──────────────────────────────────────────────
PerplexitySearchTool = searchPerplexity
BraveSearchTool      = searchBrave
OpenAIWebSearchTool  = searchOpenAI
SerpApiTool          = searchSerp

__all__ = [
    "Search",
    "searchPerplexity",
    "searchBrave",
    "searchOpenAI",
    "searchSerp",
    # aliases
    "PerplexitySearchTool",
    "BraveSearchTool",
    "OpenAIWebSearchTool",
    "SerpApiTool",
]
