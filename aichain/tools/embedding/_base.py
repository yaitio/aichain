"""
tools.embedding._base
=====================

``EmbeddingResult`` — uniform return type for every embedding call.
``Embedder``        — abstract base class for all provider implementations.

Design
------
Every concrete embedder:

  1. Sets three class attributes:
       provider   : str   — canonical provider name ("openai", "cohere", …)
       _ENV_KEY   : str   — environment variable holding the API key
       _MAX_BATCH : int   — maximum texts per single API request

  2. Implements ``_embed_chunk(texts, *, input_type, dimensions, **options)``
     which sends one request and returns ``(embeddings, total_tokens)``.

The base ``embed()`` method handles everything above that layer:
normalisation, chunking, fan-out across batches, result assembly.

The convenience methods ``run()`` and ``batch()`` wrap ``embed()`` with the
standard ``run(input, options=None)`` interface shared by all Tool/Embedder
implementations.

Universal input_type values
---------------------------
  ``"query"``     — short retrieval query; providers map to their native term
  ``"document"``  — document / passage; providers map to their native term
  ``None``        — no hint; provider default is used

Batch behaviour
---------------
``embed()`` auto-chunks the input list into slices of at most
``batch_size`` (default: the provider's ``_MAX_BATCH``).  Each slice becomes
one API request.  Results are concatenated in order so the output
``embeddings[i]`` always corresponds to ``texts[i]``.

To process a list larger than any single provider limit:

    result = embedder.embed(my_100k_list)   # sends ceil(100k/limit) requests

To cap the slice size for rate-limiting purposes:

    result = embedder.embed(texts, batch_size=32)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import urllib3

from clients._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# EmbeddingResult
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    """
    Uniform return type for every :class:`Embedder` call.

    Attributes
    ----------
    embeddings : list[list[float]]
        One float vector per input text, in the same order as the input.
    model : str
        The model name actually used (may differ from the requested name when
        a provider applies an alias).
    provider : str
        Canonical provider identifier: ``"openai"``, ``"cohere"``,
        ``"voyage"``, or ``"google"``.
    dimensions : int
        Length of each embedding vector.  ``0`` when *embeddings* is empty.
    total_tokens : int | None
        Total input tokens consumed, as reported by the provider.
        ``None`` when the provider does not expose token counts.
    metadata : dict
        Provider-specific extras (e.g. Cohere's billed units, Voyage's usage
        object).  Empty dict when nothing extra is returned.

    Examples
    --------
    ::

        result = embedder.embed(["hello world", "foo bar"])
        print(len(result.embeddings))      # 2
        print(len(result.embeddings[0]))   # e.g. 1536
        print(result.dimensions)           # 1536
        print(result.total_tokens)         # e.g. 4
    """

    embeddings:   list[list[float]]
    model:        str
    provider:     str
    dimensions:   int
    total_tokens: int | None = None
    metadata:     dict       = field(default_factory=dict)

    def __repr__(self) -> str:
        n = len(self.embeddings)
        return (
            f"EmbeddingResult("
            f"n={n}, dimensions={self.dimensions}, "
            f"model={self.model!r}, provider={self.provider!r}, "
            f"total_tokens={self.total_tokens})"
        )

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> list[float]:
        return self.embeddings[idx]


# ---------------------------------------------------------------------------
# Embedder base class
# ---------------------------------------------------------------------------

class Embedder:
    """
    Abstract base class for all embedding provider clients.

    Subclasses must set the three class-level attributes and implement
    :meth:`_embed_chunk`.  Everything else — normalisation, chunking, result
    assembly — is provided here.

    Class attributes
    ----------------
    provider : str
        Canonical provider name (``"openai"``, ``"cohere"``, …).
    _ENV_KEY : str
        Name of the environment variable that holds the API key.
    _MAX_BATCH : int
        Maximum number of texts the provider accepts in a single request.
        Used as the default ``batch_size`` in :meth:`embed`.

    Constructor
    -----------
    All subclasses accept ``(model, *, api_key=None, **defaults)``:

    * ``model``    — model identifier string (e.g. ``"text-embedding-3-large"``)
    * ``api_key``  — override the environment variable
    * ``**defaults`` — default values for ``input_type``, ``dimensions``, etc.

    Public methods
    --------------
    :meth:`embed`   — core low-level method; keyword-argument interface.
    :meth:`run`     — standard ``run(input, options=None)`` wrapper around embed.
    :meth:`batch`   — like run() but asserts the input is always a list.

    Examples
    --------
    Subclassing::

        class MyEmbedder(Embedder):
            provider   = "myprovider"
            _ENV_KEY   = "MY_API_KEY"
            _MAX_BATCH = 500

            def _embed_chunk(self, texts, *, input_type, dimensions, **opts):
                # ... call API, return (embeddings, total_tokens)
                return [[0.0] * 128] * len(texts), len(texts)
    """

    provider:   str = ""
    _ENV_KEY:   str = ""
    _MAX_BATCH: int = 100

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model:   str,
        *,
        api_key: str | None = None,
        **defaults: Any,
    ) -> None:
        key = api_key or os.environ.get(self._ENV_KEY, "")
        if not key:
            raise ValueError(
                f"No {self.provider} API key found.  "
                f"Pass api_key= or set the {self._ENV_KEY!r} environment variable."
            )
        self.model    = model
        self._api_key = key
        self._defaults = defaults
        self._http = urllib3.PoolManager(
            timeout=DEFAULT_TIMEOUT,
            retries=DEFAULT_RETRIES,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: "str | list[str]",
        *,
        input_type:  "str | None"  = None,
        dimensions:  "int | None"  = None,
        batch_size:  "int | None"  = None,
        **options: Any,
    ) -> EmbeddingResult:
        """
        Embed one or more texts and return an :class:`EmbeddingResult`.

        The input is automatically chunked into batches of at most
        ``batch_size`` (default: the provider's ``_MAX_BATCH``).  Results are
        assembled in the original order, so ``result.embeddings[i]``
        always corresponds to ``texts[i]``.

        Parameters
        ----------
        texts : str | list[str]
            A single string or a list of strings to embed.
        input_type : ``"query"`` | ``"document"`` | None, optional
            Semantic hint passed to the provider.  Improves retrieval quality
            for providers that distinguish query and document embeddings
            (Cohere, Voyage, Google v1).  OpenAI ignores it.
        dimensions : int | None, optional
            Output vector size.  When omitted the provider default is used.
            Supported by: OpenAI (``dimensions``), Cohere v4
            (``output_dimension``), Voyage (``output_dimension``),
            Google (``output_dimensionality``).
        batch_size : int | None, optional
            Override the provider's default maximum batch size.  Useful for
            rate-limiting.  Clamped to ``[1, _MAX_BATCH]``.
        **options
            Provider-specific keyword arguments forwarded directly to
            :meth:`_embed_chunk`.  Unknown keys are silently ignored by
            each provider implementation.

        Returns
        -------
        EmbeddingResult
            ``embeddings[i]`` is a ``list[float]`` for ``texts[i]``.

        Raises
        ------
        ValueError
            On invalid input (empty list, non-string items).
        RuntimeError
            On non-2xx HTTP responses from the provider API.

        Examples
        --------
        On-demand (single text)::

            result = embedder.embed("What is gradient descent?")
            vector = result.embeddings[0]       # list[float]

        Batch (list of texts — auto-chunked if needed)::

            result = embedder.embed(
                ["query A", "query B", "query C"],
                input_type="query",
            )
            vectors = result.embeddings         # list[list[float]]

        Large list with custom chunk size (useful for rate-limiting)::

            result = embedder.embed(my_10k_list, batch_size=32)
        """
        # ── Normalise input ───────────────────────────────────────────
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("texts must not be empty")

        # ── Resolve effective options (defaults → call-time overrides) ─
        merged_opts = {**self._defaults, **options}
        eff_input_type = input_type  if input_type  is not None else merged_opts.pop("input_type", None)
        eff_dimensions = dimensions  if dimensions  is not None else merged_opts.pop("dimensions", None)

        # ── Determine chunk size ──────────────────────────────────────
        limit = self._MAX_BATCH
        if batch_size is not None:
            limit = max(1, min(batch_size, self._MAX_BATCH))

        # ── Fan-out across batches ────────────────────────────────────
        all_embeddings: list[list[float]] = []
        total_tokens_sum: int = 0
        has_tokens = False
        last_metadata: dict = {}

        n_chunks = math.ceil(len(texts) / limit)
        for i in range(n_chunks):
            chunk = texts[i * limit : (i + 1) * limit]
            vecs, tok, meta = self._embed_chunk(
                chunk,
                input_type = eff_input_type,
                dimensions = eff_dimensions,
                **merged_opts,
            )
            all_embeddings.extend(vecs)
            if tok is not None:
                total_tokens_sum += tok
                has_tokens = True
            last_metadata = meta

        dims = len(all_embeddings[0]) if all_embeddings else 0
        return EmbeddingResult(
            embeddings   = all_embeddings,
            model        = self.model,
            provider     = self.provider,
            dimensions   = dims,
            total_tokens = total_tokens_sum if has_tokens else None,
            metadata     = last_metadata,
        )

    def run(self, input: "str | list[str]", options: "dict | None" = None) -> "EmbeddingResult":
        """
        Embed one text (or a short list) and return an EmbeddingResult.

        This is the standard interface shared by all Embedder implementations.
        For large batches, prefer :meth:`batch` which makes the intent explicit.

        Parameters
        ----------
        input : str | list[str]
            A single string or a list of strings to embed.
        options : dict | None, optional
            Embedding options:
              ``input_type``  — ``"query"`` | ``"document"`` | ``None``
              ``dimensions``  — output vector size override
              ``batch_size``  — chunk size cap for large lists

        Returns
        -------
        EmbeddingResult
        """
        opts       = options or {}
        input_type = opts.get("input_type")
        dimensions = opts.get("dimensions")
        batch_size = opts.get("batch_size")
        return self.embed(
            input,
            input_type = input_type,
            dimensions = dimensions,
            batch_size = batch_size,
        )

    def batch(self, input: "list[str]", options: "dict | None" = None) -> "EmbeddingResult":
        """
        Embed a list of texts explicitly as a batch.

        Identical to :meth:`run` but communicates intent clearly when the
        caller knows the input is always a list.

        Parameters
        ----------
        input : list[str]
            Texts to embed.
        options : dict | None, optional
            Same keys as :meth:`run`.

        Returns
        -------
        EmbeddingResult
        """
        if not isinstance(input, list):
            raise TypeError(
                f"batch() expects a list[str]; got {type(input).__name__}.  "
                f"Use run() for a single string."
            )
        return self.run(input, options)

    # ------------------------------------------------------------------
    # Abstract method — must be implemented by subclasses
    # ------------------------------------------------------------------

    def _embed_chunk(
        self,
        texts:      list[str],
        *,
        input_type: "str | None",
        dimensions: "int | None",
        **options:  Any,
    ) -> "tuple[list[list[float]], int | None, dict]":
        """
        Embed a single chunk of texts.

        Must be implemented by every subclass.  The chunk is guaranteed to
        have at most ``_MAX_BATCH`` items.

        Parameters
        ----------
        texts : list[str]
            The texts to embed (guaranteed non-empty, length ≤ _MAX_BATCH).
        input_type : str | None
            Universal input type hint (``"query"`` / ``"document"`` / ``None``).
        dimensions : int | None
            Desired output dimensionality, or ``None`` for provider default.
        **options
            Any provider-specific keyword arguments.

        Returns
        -------
        tuple[list[list[float]], int | None, dict]
            * ``embeddings``   — list of float vectors, one per text
            * ``total_tokens`` — token count reported by provider, or ``None``
            * ``metadata``     — provider-specific extras dict
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _embed_chunk()"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _request(self, url: str, body: dict, extra_headers: dict | None = None) -> dict:
        """
        POST *body* as JSON to *url* and return the parsed response dict.

        Raises
        ------
        RuntimeError
            On any non-2xx HTTP status.
        """
        import json
        headers = {
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            **(extra_headers or {}),
        }
        response = self._http.request(
            "POST",
            url,
            body    = json.dumps(body).encode("utf-8"),
            headers = headers,
        )
        raw = response.data.decode("utf-8", errors="replace")
        if not (200 <= response.status < 300):
            raise RuntimeError(
                f"{self.provider} Embeddings API error [{response.status}]: {raw[:500]}"
            )
        import json as _json
        return _json.loads(raw)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, "
            f"provider={self.provider!r})"
        )
