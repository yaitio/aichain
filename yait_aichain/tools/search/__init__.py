"""
tools.search
============

Web search — one vocabulary of options and one record of results for every
provider, the way ``tools.decision`` works. ``run()`` returns numbered text for
a model; ``search()`` returns the records for code.

Tools
-----
searchPerplexity   Perplexity Search API — a substantive passage from each page.
searchBrave        Brave Search — ranked links with descriptions.
searchSerp         SerpAPI — Google, Bing, DuckDuckGo, Baidu, Yandex and more.
searchExa          Exa — the passages that answer the query; page text on request.
searchFirecrawl    Firecrawl — descriptions; each page as Markdown on request.
searchTavily       Tavily — passages chosen for the query; each page as Markdown on request.

``Search`` is the base; a new provider is a subclass that implements ``_search``.

Backward-compatible aliases
---------------------------
PerplexitySearchTool, BraveSearchTool, SerpApiTool
"""

from ._base      import Search
from .perplexity import searchPerplexity
from .brave      import searchBrave
from .serp       import searchSerp
from .exa        import searchExa
from .firecrawl  import searchFirecrawl
from .tavily     import searchTavily

# ── Backward-compatible aliases ──────────────────────────────────────────────
PerplexitySearchTool = searchPerplexity
BraveSearchTool      = searchBrave
SerpApiTool          = searchSerp

__all__ = [
    "Search",
    "searchPerplexity",
    "searchBrave",
    "searchSerp",
    "searchExa",
    "searchFirecrawl",
    "searchTavily",
    # aliases
    "PerplexitySearchTool",
    "BraveSearchTool",
    "SerpApiTool",
]
