"""
Embedding as a tool: ``embed(service, model=…)`` over five providers
(``SERVICES``), with ``EmbeddingResult`` and the ``Embedder`` base class.

The model, ``dimensions`` and ``batch_size`` are settings of the tool; a call
takes only ``input_type``. See docs/tools/retrieval.md.
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
