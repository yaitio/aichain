"""
tools.embed
===========

Embedding as a tool: one interface over five providers.

Public API
----------
``embed(service, *, model, dimensions=None, batch_size=None, api_key=None)``
    Makes the embedding tool for *service* — one of ``SERVICES``:
    ``"openai"``, ``"cohere"``, ``"voyage"``, ``"google"``, ``"qwen"``.

``EmbeddingResult``
    Uniform return type for every embedding call.

``Embedder``
    The base class of :class:`EmbeddingOpenAI`, :class:`EmbeddingCohere`,
    :class:`EmbeddingVoyage`, :class:`EmbeddingGoogle` and
    :class:`EmbeddingQwen`.

The model, ``dimensions`` and ``batch_size`` are settings of the tool; a call
takes only ``input_type``. Vectors from two models, or of two lengths, cannot
share a store, so a tool is fixed to one.

Examples
--------
::

    from yait_aichain.tools import embed

    e = embed("openai", model="text-embedding-3-small")

    # One text
    result = e.run("What is machine learning?", {"input_type": "query"})
    vector = result.embeddings[0]                # list[float]

    # A list — split into requests of at most batch_size
    result = e.run(["text A", "text B"], {"input_type": "document"})
    print(result.dimensions, result.total_tokens)

    # A smaller batch, for rate-limiting — a setting of the tool
    e = embed("cohere", model="embed-v4.0", batch_size=32)
"""

from ._base   import Embedder, EmbeddingResult
from ._openai import EmbeddingOpenAI
from ._cohere import EmbeddingCohere
from ._voyage import EmbeddingVoyage
from ._google import EmbeddingGoogle
from ._qwen   import EmbeddingQwen

__all__ = ["embed", "SERVICES", "Embedder", "EmbeddingResult", "EmbeddingOpenAI", "EmbeddingCohere",
           "EmbeddingVoyage", "EmbeddingGoogle", "EmbeddingQwen"]


# ---------------------------------------------------------------------------
# Provider prefix → class registry
# ---------------------------------------------------------------------------

_PREFIX_MAP: dict[str, type] = {
    "openai": EmbeddingOpenAI,
    "cohere": EmbeddingCohere,
    "voyage": EmbeddingVoyage,
    "google": EmbeddingGoogle,
    "qwen":   EmbeddingQwen,
}





# ---------------------------------------------------------------------------
# embed(service, model=…) — the factory
# ---------------------------------------------------------------------------

#: Provider name → tool class, for :func:`embed`.
SERVICES = _PREFIX_MAP


def embed(service: str, *, model: str, dimensions: "int | None" = None, batch_size: "int | None" = None,
          api_key: "str | None" = None, name: "str | None" = None,
          description: "str | None" = None) -> Embedder:
    """
    An embedding tool for *service* — one of ``SERVICES`` — and *model*.

    The model, the vectors' ``dimensions`` and the ``batch_size`` are
    settings, not options: vectors from two models, or of two lengths, cannot
    share a store, so the tool is fixed to one. ``name`` and ``description``
    default to the provider's; ``api_key`` to its environment variable.

    ::

        vectors = embed("openai", model="text-embedding-3-small").run(["a text", "another"])
        vectors.embeddings[0][:3], vectors.dimensions
        str(vectors)          # what a model reads: "2 vector(s) of 1536 dimensions …"
    """
    from .._callable import pick
    return pick("embed", SERVICES, service)(model, api_key=api_key, name=name, description=description,
                                            dimensions=dimensions, batch_size=batch_size)


from .._callable import callable_package as _callable_package
_callable_package(__name__, embed)
