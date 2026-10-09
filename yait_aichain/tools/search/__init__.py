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
searchExa          Exa — the passages that answer the query; page text on request.
searchFirecrawl    Firecrawl — descriptions; each page as Markdown on request.
searchTavily       Tavily — passages chosen for the query; each page as Markdown on request.
searchParallel     Parallel — excerpts that serve an objective; built for agents.

``search(service)`` makes one by the provider's name — ``search("exa")`` — and
is the way the documentation shows. ``Search`` is the base; a new provider is a
subclass that implements ``_search``, and an entry in ``SERVICES``.

Backward-compatible aliases
---------------------------
PerplexitySearchTool, BraveSearchTool
"""

from ._base      import Search
from .perplexity import searchPerplexity
from .brave      import searchBrave
from .exa        import searchExa
from .firecrawl  import searchFirecrawl
from .tavily     import searchTavily
from .parallel   import searchParallel

#: Provider name → tool class, for :func:`search`.
SERVICES = {
    "perplexity": searchPerplexity,
    "brave":      searchBrave,
    "exa":        searchExa,
    "firecrawl":  searchFirecrawl,
    "tavily":     searchTavily,
    "parallel":   searchParallel,
}


def search(service: str, *, api_key: "str | None" = None, name: "str | None" = None,
           description: "str | None" = None) -> Search:
    """
    A web search tool for *service* — one of ``SERVICES``.

    ``name`` and ``description`` default to the provider's own and override
    them; ``api_key`` defaults to the provider's environment variable. Every
    other setting of a search is an option of the call, so nothing else is
    taken here.

    ::

        search("exa").run("AI and leadership", {"domains": ["hbr.org"]})["results"]
        Agent(model, tools=[search("perplexity")])
    """
    cls = SERVICES.get(service.lower()) if isinstance(service, str) else None
    if cls is None:
        raise ValueError(f"unknown search service {service!r}; one of: {', '.join(SERVICES)}")
    return cls(api_key, name=name, description=description)


# ── Backward-compatible aliases ──────────────────────────────────────────────
PerplexitySearchTool = searchPerplexity
BraveSearchTool      = searchBrave

__all__ = [
    "search",
    "SERVICES",
    "Search",
    "searchPerplexity",
    "searchBrave",
    "searchExa",
    "searchFirecrawl",
    "searchTavily",
    "searchParallel",
    # aliases
    "PerplexitySearchTool",
    "BraveSearchTool",
]


# ── The package is callable ──────────────────────────────────────────────────
# `search("…")` is called on this package, not on a function that takes its name
# in `tools`: a function there hides the package, and `import
# yait_aichain.tools.search.x as m`, attribute access and — on Python 3.10 and
# 3.11 — `mock.patch("yait_aichain.tools.search.x.…")` resolve through that name
# and fail. Measured 2026-10-09; see docs/design/tool-convention.md.
import inspect as _inspect
import sys as _sys
import types as _types


class _CallableModule(_types.ModuleType):
    def __call__(self, *args, **kwargs):
        return search(*args, **kwargs)


_sys.modules[__name__].__class__ = _CallableModule
_sys.modules[__name__].__signature__ = _inspect.signature(search)
