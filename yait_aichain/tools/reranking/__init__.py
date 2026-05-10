"""
tools.reranking
===============

Provider-agnostic reranking interface for RAG pipelines.

Public API
----------
``Reranker(model, *, api_key=None, **kwargs)``
    Factory function — returns the right :class:`RerankBase` subclass.
    Mirrors the ``Embedding(model)`` factory pattern.

``RerankResult``
    Uniform return type: ``results``, ``model``, ``provider``,
    ``total_tokens``, ``metadata``.
    Iterable, indexable, and has helper methods ``texts()`` / ``scores()``.

``RerankBase``
    Abstract base class.  Subclass to add a new provider.

Provider routing
----------------
The ``model`` argument follows the ``"provider/model"`` convention:

  ``"cohere/rerank-v3.5"``       → :class:`RerankCohere`
  ``"cohere/rerank-english-v3.0"``
  ``"cohere/rerank-multilingual-v3.0"``

  ``"voyage/rerank-2"``          → :class:`RerankVoyage`
  ``"voyage/rerank-2-lite"``

  ``"qwen/gte-rerank"``          → :class:`RerankQwen`
  ``"qwen/gte-rerank-v2"``

Bare model names are resolved by prefix heuristics:
  starts with ``"rerank-"``      → Cohere
  starts with ``"gte-rerank"``   → Qwen

Interface
---------
Every reranker exposes two call styles:

``reranker.rerank(query, documents, *, top_n=None)`` → ``RerankResult``
    Core method; returns the full result object with scores and metadata.

``reranker.run(documents, options)`` → ``list[dict]``
    Standard Tool interface for Chains and Agents.
    ``options = {"query": str, "top_n": int}``

Document input
--------------
Both methods accept documents as:
  • ``list[str]``   — plain text
  • ``list[dict]``  — dicts with ``"text"`` key (direct output from
                       :class:`~tools.vectordb.VectorQueryTool`)

Original dict fields (``id``, ``metadata``, …) are preserved in the results.

Examples
--------
Standalone::

    from tools.reranking import Reranker

    reranker = Reranker("cohere/rerank-v3.5")
    result   = reranker.rerank(
        "how does KV caching work?",
        ["LLMs cache key-value pairs …", "Python is a language …"],
        top_n=1,
    )
    print(result[0]["score"], result[0]["text"][:60])

With vectorQuery output (dicts preserved)::

    from tools.vectordb  import VectorDB, vectorQuery
    from tools.reranking import Reranker

    store     = VectorDB("chroma", "docs", embedder=...)
    candidates = vectorQuery(store).run("KV caching", {"n": 20})

    reranker  = Reranker("voyage/rerank-2")
    top5      = reranker.run(candidates, {"query": "KV caching", "top_n": 5})
    # top5[0] still has "id" and "metadata" from the original vectorQuery result

In a Chain::

    from chain import Chain

    pipeline = Chain(steps=[
        (vectorQuery(store),  "candidates", {"input": "{question}"}),
        (reranker,            "top5",       {"input": "{candidates}",
                                             "options": {"query": "{question}",
                                                         "top_n": 5}}),
        answer_skill,
    ])
"""

from ._base   import RerankBase, RerankResult
from ._cohere import RerankCohere
from ._voyage import RerankVoyage
from ._qwen   import RerankQwen

__all__ = [
    "Reranker",
    "RerankBase",
    "RerankResult",
    "RerankCohere",
    "RerankVoyage",
    "RerankQwen",
]


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PREFIX_MAP: dict[str, type] = {
    "cohere": RerankCohere,
    "voyage": RerankVoyage,
    "qwen":   RerankQwen,
}

_HEURISTICS: list[tuple[str, type]] = [
    ("rerank-",     RerankCohere),   # rerank-v3.5, rerank-english-v3.0
    ("gte-rerank",  RerankQwen),     # gte-rerank, gte-rerank-v2
]


# ---------------------------------------------------------------------------
# Reranker factory
# ---------------------------------------------------------------------------

def Reranker(
    model:   str,
    *,
    api_key: str | None = None,
    **kwargs,
) -> RerankBase:
    """
    Return a :class:`RerankBase` instance for *model*.

    Parameters
    ----------
    model : str
        Model identifier.  Accepts ``"provider/model"`` (unambiguous) or
        a bare model name resolved by heuristic.

        Supported provider prefixes:
          ``"cohere/"``  — Cohere Rerank
          ``"voyage/"``  — Voyage AI Rerank
          ``"qwen/"``    — Alibaba DashScope GTE-Rerank

    api_key : str | None, optional
        Override the provider's environment variable.
    **kwargs
        Extra arguments forwarded to the backend constructor.
        Common extras:
          ``region`` (str) — DashScope region for Qwen (``"ap"``, ``"us"``, ``"cn"``, ``"hk"``)

    Returns
    -------
    RerankBase
        A concrete reranker ready to call :meth:`~RerankBase.rerank`.

    Raises
    ------
    ValueError
        When the provider cannot be inferred from *model*.

    Examples
    --------
    ::

        from tools.reranking import Reranker

        r1 = Reranker("cohere/rerank-v3.5")
        r2 = Reranker("voyage/rerank-2")
        r3 = Reranker("qwen/gte-rerank")

        # Bare model name
        r4 = Reranker("rerank-v3.5")         # → Cohere
        r5 = Reranker("gte-rerank")           # → Qwen
    """
    # ── Explicit provider/model prefix ────────────────────────────────────
    if "/" in model:
        prefix, bare = model.split("/", 1)
        cls = _PREFIX_MAP.get(prefix.lower())
        if cls is None:
            supported = ", ".join(sorted(_PREFIX_MAP))
            raise ValueError(
                f"Unknown reranking provider prefix {prefix!r}.  "
                f"Supported: {supported}."
            )
        return cls(bare, api_key=api_key, **kwargs)

    # ── Heuristic routing for bare model names ────────────────────────────
    lower = model.lower()
    for prefix, cls in _HEURISTICS:
        if lower.startswith(prefix):
            return cls(model, api_key=api_key, **kwargs)

    raise ValueError(
        f"Cannot infer reranking provider from model name {model!r}.  "
        f"Use a provider-prefixed name such as 'cohere/{model}', "
        f"'voyage/{model}', or 'qwen/{model}'."
    )
