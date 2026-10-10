"""
tools.rerank
============

Reranking as a tool, for RAG pipelines: one interface over three providers.

Public API
----------
``rerank(service, *, model, api_key=None)``
    Makes the reranking tool for *service* — one of ``SERVICES``:
    ``"cohere"`` (:class:`RerankCohere`), ``"voyage"`` (:class:`RerankVoyage`),
    ``"qwen"`` (:class:`RerankQwen`).

``RerankResult``
    Uniform return type: ``results``, ``model``, ``provider``,
    ``total_tokens``, ``metadata``.
    Iterable, indexable, and has helper methods ``texts()`` / ``scores()``.

``RerankBase``
    Abstract base class.  Subclass to add a new provider.

Interface
---------
Every reranker exposes two call styles:

``reranker.rerank(query, documents, *, top_n=None)`` → ``RerankResult``
    Core method; returns the full result object with scores and metadata.

``reranker.run(documents, options)`` → ``list[dict]``
    Standard Tool interface for Chains and Agents.
    ``options = {"query": str, "top_n": int}``

The model is a setting of the tool; ``query`` and ``top_n`` belong to a call.

Document input
--------------
Both methods accept documents as:
  • ``list[str]``   — plain text
  • ``list[dict]``  — dicts with a ``"text"`` key, such as the output of
                       ``Vectors(...).query``

Original dict fields (``id``, ``metadata``, …) are preserved in the results.

Examples
--------
Standalone::

    from yait_aichain.tools import rerank

    reranker = rerank("cohere", model="rerank-v3.5")
    result   = reranker.rerank(
        "how does KV caching work?",
        ["LLMs cache key-value pairs …", "Python is a language …"],
        top_n=1,
    )
    print(result[0]["score"], result[0]["text"][:60])

After a vector search (dicts preserved)::

    from yait_aichain.tools import embed, rerank
    from yait_aichain.utilities import Vectors

    kb = Vectors("qdrant", collection="docs", description="the product documentation",
                 embedder=embed("openai", model="text-embedding-3-small"))
    candidates = kb.query.run("KV caching", {"n": 20})

    reranker = rerank("voyage", model="rerank-2")
    top5     = reranker.run(candidates, {"query": "KV caching", "top_n": 5})
    # top5[0] still has "id" and "metadata" from the search

In a Chain::

    from yait_aichain.chain import Chain

    pipeline = Chain([
        (kb.query, "candidates", {"input": "{question}"}),
        (reranker, "top5",       {"input": "{candidates}",
                                  "options": {"query": "{question}", "top_n": 5}}),
        answer_skill,
    ])
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
