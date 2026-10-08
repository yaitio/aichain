"""
tools.fetch
===========

A web page's text from its URL — one record of a page for every provider, the
way ``tools.search`` works. A search finds the link; a fetch brings the page.
``run()`` returns the page as text for a model; ``fetch()`` takes one URL or a
list and returns the records for code.

Tools
-----
fetchFirecrawl   Firecrawl Scrape — main content and metadata; reached the most in our measurement.
fetchTavily      Tavily Extract — up to 20 URLs a call; text and title only.
fetchExa         Exa Contents — up to 100 URLs a call, from Exa's cache; title, date, author.
fetchParallel    Parallel Extract — up to 20 URLs a call; full content, title, date.
fetchLocal       MarkItDown on this machine — free, respects robots.txt;
                 needs ``pip install "yait-aichain[fetch]"``.

``Fetch`` is the base; a new provider is a subclass that implements ``_fetch``.
"""

from ._base     import Fetch
from .firecrawl import fetchFirecrawl
from .tavily    import fetchTavily
from .exa       import fetchExa
from .parallel  import fetchParallel
from .local     import fetchLocal

__all__ = [
    "Fetch",
    "fetchFirecrawl",
    "fetchTavily",
    "fetchExa",
    "fetchParallel",
    "fetchLocal",
]
