"""
tools.embedding
===============

Provider-agnostic embedding interface.

Public API
----------
``Embedding(model, *, api_key=None, **defaults)``
    Factory function that inspects the ``model`` string and returns the
    right :class:`~tools.embedding._base.Embedder` subclass instance.

``EmbeddingResult``
    Uniform return type for every embedding call.

Provider routing
----------------
The ``model`` argument may be either a bare model name or a
``"provider/model"`` string:

  Routing rules (applied in order):
    1. ``"openai/..."``   or known OpenAI prefix → :class:`EmbeddingOpenAI`
    2. ``"cohere/..."``   or known Cohere prefix → :class:`EmbeddingCohere`
    3. ``"voyage/..."``   or known Voyage prefix → :class:`EmbeddingVoyage`
    4. ``"google/..."``   or known Google prefix → :class:`EmbeddingGoogle`
    5. ``"qwen/..."``     or known Qwen prefix   → :class:`EmbeddingQwen`

  Bare-name heuristics (when no ``provider/`` prefix is given):
    • Starts with ``"text-embedding-3"`` or ``"text-embedding-ada"`` → OpenAI
    • Starts with ``"embed-"``                                        → Cohere
    • Starts with ``"voyage-"``                                       → Voyage
    • Starts with ``"gemini-embedding-"``
      or ``"text-embedding-004"``                                     → Google
    • Starts with ``"text-embedding-v"``                              → Qwen

Examples
--------
::

    from tools.embedding import Embedding

    # Provider-prefixed (unambiguous)
    e = Embedding("openai/text-embedding-3-large")
    e = Embedding("cohere/embed-v4.0")
    e = Embedding("voyage/voyage-3-large")
    e = Embedding("google/gemini-embedding-001")
    e = Embedding("qwen/text-embedding-v4")

    # Bare model name (heuristic routing)
    e = Embedding("text-embedding-3-small")      # → OpenAI
    e = Embedding("embed-v4.0")                  # → Cohere
    e = Embedding("voyage-3")                    # → Voyage
    e = Embedding("gemini-embedding-001")        # → Google
    e = Embedding("text-embedding-v4")           # → Qwen

    # Embed a single string
    result = e.embed("What is machine learning?", input_type="query")
    vector = result.embeddings[0]                # list[float]

    # Standard run() interface
    result = e.run("What is machine learning?", {"input_type": "query"})

    # Batch interface (explicit list)
    result = e.batch(["text A", "text B"], {"input_type": "document"})

    # Embed a batch via embed()
    result = e.embed(["text A", "text B"], input_type="document")
    print(result.dimensions, result.total_tokens)

    # Override batch chunk size (useful for rate-limiting)
    result = e.embed(my_large_list, batch_size=32)
"""

from ._base   import Embedder, EmbeddingResult
from ._openai import EmbeddingOpenAI, OpenAIEmbedder
from ._cohere import EmbeddingCohere, CohereEmbedder
from ._voyage import EmbeddingVoyage, VoyageEmbedder
from ._google import EmbeddingGoogle, GoogleEmbedder
from ._qwen   import EmbeddingQwen

__all__ = [
    "Embedding",
    "Embedder",
    "EmbeddingResult",
    # New canonical names
    "EmbeddingOpenAI",
    "EmbeddingCohere",
    "EmbeddingVoyage",
    "EmbeddingGoogle",
    "EmbeddingQwen",
    # Backward-compatible aliases
    "OpenAIEmbedder",
    "CohereEmbedder",
    "VoyageEmbedder",
    "GoogleEmbedder",
]


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

# Bare-name prefix heuristics (checked with str.startswith)
_HEURISTICS: list[tuple[str, type]] = [
    ("text-embedding-3",       EmbeddingOpenAI),
    ("text-embedding-ada",     EmbeddingOpenAI),
    ("embed-",                 EmbeddingCohere),
    ("voyage-",                EmbeddingVoyage),
    ("gemini-embedding-",      EmbeddingGoogle),
    ("text-embedding-004",     EmbeddingGoogle),
    ("text-embedding-v",       EmbeddingQwen),
]


def Embedding(
    model:   str,
    *,
    api_key: "str | None" = None,
    **defaults,
) -> Embedder:
    """
    Return an :class:`Embedder` instance for *model*.

    Parameters
    ----------
    model : str
        Model identifier.  May be a bare name (``"voyage-3-large"``) or a
        provider-prefixed string (``"voyage/voyage-3-large"``).
    api_key : str | None, optional
        Override the provider's environment variable.
    **defaults
        Default call-time options forwarded to every :meth:`~Embedder.embed`
        call, e.g. ``input_type="document"``, ``dimensions=512``.

    Returns
    -------
    Embedder
        A concrete :class:`Embedder` subclass ready to call :meth:`~Embedder.embed`.

    Raises
    ------
    ValueError
        When the provider cannot be inferred from *model*.
    """
    # ── Explicit provider/model prefix ───────────────────────────────────
    if "/" in model:
        prefix, bare = model.split("/", 1)
        cls = _PREFIX_MAP.get(prefix.lower())
        if cls is None:
            raise ValueError(
                f"Unknown embedding provider prefix {prefix!r}.  "
                f"Supported: {', '.join(_PREFIX_MAP)}."
            )
        return cls(bare, api_key=api_key, **defaults)

    # ── Heuristic routing for bare model names ────────────────────────────
    lower = model.lower()
    for prefix, cls in _HEURISTICS:
        if lower.startswith(prefix):
            return cls(model, api_key=api_key, **defaults)

    raise ValueError(
        f"Cannot infer embedding provider from model name {model!r}.  "
        f"Use a provider-prefixed name such as 'openai/{model}', "
        f"'cohere/{model}', 'voyage/{model}', or 'google/{model}'."
    )
