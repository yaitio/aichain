"""A web page's text from its URL, one record of a page for every provider:
``fetchFirecrawl``, ``fetchTavily``, ``fetchExa``, ``fetchParallel`` and
``fetchLocal``, made with ``fetch(service)``. A new provider subclasses
``Fetch``, implements ``_fetch`` and joins ``SERVICES``. See docs/tools/fetch.md.
"""

from ._base     import Fetch
from .firecrawl import fetchFirecrawl
from .tavily    import fetchTavily
from .exa       import fetchExa
from .parallel  import fetchParallel
from .local     import fetchLocal

#: Provider name → tool class, for :func:`fetch`.
SERVICES = {
    "firecrawl": fetchFirecrawl,
    "tavily":    fetchTavily,
    "exa":       fetchExa,
    "parallel":  fetchParallel,
    "local":     fetchLocal,
}

#: Settings only the local fetch has — it runs here, so how it connects is ours.
_LOCAL = ("timeout", "headers", "respect_robots", "robots_agents")


def fetch(service: str, *, api_key: "str | None" = None, name: "str | None" = None,
          description: "str | None" = None, **settings) -> Fetch:
    """
    A fetch tool for *service* — one of ``SERVICES``.

    ``name`` and ``description`` default to the provider's own and override
    them; ``api_key`` defaults to the provider's environment variable.
    ``local`` takes ``timeout``, ``headers``, ``respect_robots`` and
    ``robots_agents`` as well, and no key. Everything else is an option of the
    call.

    ::

        fetch("firecrawl").run(["https://example.com/a", "https://example.com/b"])["pages"]
        fetch("local", respect_robots=True).run("https://example.com")
    """
    cls = SERVICES.get(service.lower()) if isinstance(service, str) else None
    if cls is None:
        raise ValueError(f"unknown fetch service {service!r}; one of: {', '.join(SERVICES)}")
    if cls is fetchLocal:
        if unknown := sorted(set(settings) - set(_LOCAL)):
            raise TypeError(f"fetch('local') takes {', '.join(_LOCAL)}; got {', '.join(unknown)}")
        if api_key:
            raise TypeError("fetch('local') runs on this machine and takes no api_key")
        return cls(**settings, name=name, description=description)
    if settings:
        raise TypeError(f"fetch({service!r}) takes api_key, name and description; got "
                        f"{', '.join(sorted(settings))} — "
                        + (f"{', '.join(_LOCAL)} are fetch('local')'s"
                           if set(settings) <= set(_LOCAL) else "every other setting is an option of the call"))
    return cls(api_key, name=name, description=description)


__all__ = [
    "fetch",
    "SERVICES",
    "Fetch",
    "fetchFirecrawl",
    "fetchTavily",
    "fetchExa",
    "fetchParallel",
    "fetchLocal",
]



# The package is callable — `fetch("…")` — and stays the package; see tools._callable.
from .._callable import callable_package as _callable_package
_callable_package(__name__, fetch)
