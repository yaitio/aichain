"""
Reranking as a tool: ``rerank(service, model=…)`` over three providers
(``SERVICES``), with ``RerankResult`` and the ``RerankBase`` base class.

The model is a setting of the tool; ``query`` and ``top_n`` belong to a call.
Documents are texts or records with a ``"text"`` key, whose other fields are
kept on the results. See docs/tools/retrieval.md.
"""

from ._base   import RerankBase, RerankResult
from ._cohere import RerankCohere
from ._voyage import RerankVoyage
from ._qwen   import RerankQwen

__all__ = ["rerank", "SERVICES", "RerankBase", "RerankResult", "RerankCohere", "RerankVoyage", "RerankQwen"]


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PREFIX_MAP: dict[str, type] = {
    "cohere": RerankCohere,
    "voyage": RerankVoyage,
    "qwen":   RerankQwen,
}



# ---------------------------------------------------------------------------
# rerank(service, model=…) — the factory
# ---------------------------------------------------------------------------

#: Provider name → tool class, for :func:`rerank`.
SERVICES = _PREFIX_MAP


def rerank(service: str, *, model: str, api_key: "str | None" = None, name: "str | None" = None,
           description: "str | None" = None) -> RerankBase:
    """
    A reranking tool for *service* — one of ``SERVICES`` — and *model*.

    ::

        top = rerank("cohere", model="rerank-v3.5").run(docs, {"query": "how does caching work?", "top_n": 5})
    """
    from .._callable import pick
    return pick("rerank", SERVICES, service)(model, api_key=api_key, name=name, description=description)


from .._callable import callable_package as _callable_package
_callable_package(__name__, rerank)
