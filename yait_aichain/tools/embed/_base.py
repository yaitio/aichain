"""
``EmbeddingResult``, the return type of every embedding call, and
``Embedder``, the base class of every provider.

A provider sets ``provider``, ``_ENV_KEY`` and ``_MAX_BATCH`` and implements
``_embed_chunk``; the base normalises input, splits it into requests the
provider accepts and reassembles the vectors in order.
See docs/tools/retrieval.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from yait_aichain.clients._base import make_http

from ...clients._constants import TOOL_TIMEOUT, DEFAULT_RETRIES
from ...clients._http import request_json
from .._base import Tool, text_of


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

    def __str__(self) -> str:
        # What an agent shows the model, and what a Skill prompt substitutes:
        # a summary. The vectors are numbers a model can do nothing with —
        # 1 536 floats a text in its context.
        tokens = f", {self.total_tokens} tokens" if self.total_tokens is not None else ""
        return (f"{len(self.embeddings)} vector(s) of {self.dimensions} dimensions from "
                f"{self.provider} {self.model}{tokens}")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> list[float]:
        return self.embeddings[idx]


# ---------------------------------------------------------------------------
# Embedder base class
# ---------------------------------------------------------------------------

def _refuse_settings(who: str, given: dict) -> None:
    """A setting passed with a call is an error that says where it belongs."""
    if not given:
        return
    settings = [k for k in given if k in ("dimensions", "batch_size")]
    if settings:
        raise ValueError(f"{who}: {', '.join(settings)} is a setting of the embedder, not of a call — "
                         f"embed(service, model=…, {settings[0]}=…). Vectors of two lengths cannot share a store.")
    raise ValueError(f"{who}: unknown option(s) {sorted(given)}; a call takes input_type")


class Embedder(Tool):
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
    All subclasses accept ``(model, *, api_key=None, dimensions=None, batch_size=None)``:

    * ``model``    — model identifier string (e.g. ``"text-embedding-3-large"``)
    * ``api_key``  — override the environment variable
    * ``dimensions``, ``batch_size`` — settings of the embedder; a call takes only ``input_type``

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
    #: Tokens one request may carry, estimated from the text. A count of rows
    #: is not a size: 2,048 chunks of 400 tokens is 819k tokens, and OpenAI
    #: refuses a request over 300k.
    _MAX_TOKENS: int | None = None
    #: Bytes of text one request may carry.
    _MAX_BYTES: int = 1_500_000
    #: Models that take a ``dimensions``; ``None`` means every model does.
    _DIMENSION_MODELS: frozenset | None = None

    # A tool: code, a chain and an agent call it alike. An agent
    # reads a summary, never the vectors (see EmbeddingResult.__str__).
    risk        = "read"
    priced      = True
    _safe_for_untrusted_load = True      # see Tool._safe_for_untrusted_load
    parameters  = {
        "type": "object",
        "properties": {
            "input":   {"type": "array", "items": {"type": "string"},
                        "description": "The texts to embed — or one text."},
            "options": {"type": "object", "description": "Optional settings.", "properties": {
                "input_type": {"type": "string", "enum": ["query", "document"],
                               "description": "What the texts are, for providers that embed the two differently."},
            }},
        },
        "required": ["input"],
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model:   str,
        *,
        api_key: str | None = None,
        name:        str | None = None,
        description: str | None = None,
        dimensions:  int | None = None,
        batch_size:  int | None = None,
    ) -> None:
        # _ENV_KEY may be a single name or several accepted names (the first
        # set one wins) — e.g. Google embeddings accept GOOGLE_AI_API_KEY or
        # GOOGLE_API_KEY, matching the rest of the library.
        env_keys = (self._ENV_KEY,) if isinstance(self._ENV_KEY, str) else tuple(self._ENV_KEY)
        key = api_key or next((os.environ[k] for k in env_keys if os.environ.get(k)), "")
        if not key:
            raise ValueError(
                f"No {self.provider} API key found.  Pass api_key= or set "
                f"the {' / '.join(env_keys)} environment variable."
            )
        self.model    = model
        self._api_key = key
        # Settings, in one place, and never options of a call: one store must
        # not be filled with vectors of two lengths.
        if (dimensions is not None and self._DIMENSION_MODELS is not None
                and model not in self._DIMENSION_MODELS):
            # A fixed-length model would ignore it, and a store built for 512
            # would get the model's own length. Refused where it is set.
            raise ValueError(f"{type(self).__name__}: {model} has a fixed length and takes no "
                             f"dimensions; the models that do: {', '.join(sorted(self._DIMENSION_MODELS))}")
        self.dimensions = dimensions
        self.batch_size = batch_size
        label = type(self).__name__.replace("Embedding", "", 1) or self.provider
        self.name = name or f"embed{label}"
        self.description = description or (
            f"Embed texts as vectors with {label}'s {model}. Returns a summary — the "
            "vectors go to the code that called it, not into the conversation.")
        self._http = make_http(
            timeout=TOOL_TIMEOUT,
            retries=DEFAULT_RETRIES,
        )

    def _serialise_init_args(self) -> dict:
        """What `Chain.save` writes: the model and its settings, never the key."""
        out = {"model": self.model}
        if self.dimensions is not None:
            out["dimensions"] = self.dimensions
        if self.batch_size is not None:
            out["batch_size"] = self.batch_size
        if getattr(self, "_region", None):
            out["region"] = self._region
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: str | list[str],
        *,
        input_type:  str | None  = None,
        **refused: Any,
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
        **refused
            ``dimensions`` and ``batch_size`` here are an error that says where
            they belong — the embedder — and any other key is an error too.

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

            embedder = embed("openai", model="text-embedding-3-small", batch_size=32)
            result   = embedder.embed(my_10k_list)
        """
        # ── Normalise input ───────────────────────────────────────────
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("texts must not be empty")
        # A chunk() record gives its text; None and "" are refused, not sent.
        texts = [text_of(t, type(self).__name__) for t in texts]
        if blank := [i for i, t in enumerate(texts) if not t.strip()]:
            raise ValueError(f"{type(self).__name__}: text {blank[:5]} is empty; "
                             "an embedding of nothing is not a vector of anything")

        _refuse_settings(type(self).__name__, refused)
        vectors: list[list[float]] = []
        tokens: list[int] = []
        metadata: dict = {}
        for chunk in self._batches(texts):
            vecs, tok, metadata = self._embed_chunk(
                chunk, input_type=input_type, dimensions=self.dimensions)
            # A reply short of vectors (Google can omit `embeddings`) is an
            # error, not a result that pairs nothing with anything.
            if len(vecs) != len(chunk) or any(not v for v in vecs):
                raise RuntimeError(f"{type(self).__name__}: {len(chunk)} texts sent, "
                                   f"{sum(1 for v in vecs if v)} vectors returned")
            vectors.extend(vecs)
            if tok is not None:
                tokens.append(tok)
        return EmbeddingResult(
            embeddings   = vectors,
            model        = self.model,
            provider     = self.provider,
            dimensions   = len(vectors[0]) if vectors else 0,
            total_tokens = sum(tokens) if tokens else None,
            metadata     = metadata,
        )

    def _max_rows(self) -> int:
        return self._MAX_BATCH

    def _batches(self, texts: list[str]):
        """Slices of *texts* no request refuses: at most the rows, the
        estimated tokens and the bytes the provider takes, whichever binds."""
        rows = max(1, min(self.batch_size or self._max_rows(), self._max_rows()))
        chunk: list[str] = []
        size = toks = 0
        for text in texts:
            n_bytes = len(text.encode("utf-8"))
            n_toks = n_bytes // 3 + 1             # over, not under: most text is 4 bytes a token
            if chunk and (len(chunk) >= rows or size + n_bytes > self._MAX_BYTES
                          or (self._MAX_TOKENS and toks + n_toks > self._MAX_TOKENS)):
                yield chunk
                chunk, size, toks = [], 0, 0
            chunk.append(text)
            size += n_bytes
            toks += n_toks
        if chunk:
            yield chunk

    def run(self, input: str | list[str], options: dict | None = None) -> EmbeddingResult:
        """
        Embed one text (or a short list) and return an EmbeddingResult.

        This is the standard tool interface; a long list is split into
        requests of at most the provider's batch size.

        Parameters
        ----------
        input : str | list[str]
            A single string or a list of strings to embed.
        options : dict | None, optional
            Takes one key, ``input_type`` — ``"query"`` | ``"document"`` |
            ``None``. ``dimensions`` and ``batch_size`` are settings of the
            embedder and raise here, as does any other key.

        Returns
        -------
        EmbeddingResult
        """
        opts = dict(options or {})
        input_type = opts.pop("input_type", None)
        _refuse_settings(self.name, opts)
        return self.embed(input, input_type=input_type)

    def batch(self, input: list[str], options: dict | None = None) -> EmbeddingResult:
        """Deprecated: the same as :meth:`run`, which takes a list. Removed in 5.0."""
        import warnings
        warnings.warn("Embedder.batch() is run() — call run(); batch() goes in 5.0",
                      DeprecationWarning, stacklevel=2)
        return self.run(input, options)

    # ------------------------------------------------------------------
    # Abstract method — must be implemented by subclasses
    # ------------------------------------------------------------------

    def _embed_chunk(
        self,
        texts:      list[str],
        *,
        input_type: str | None,
        dimensions: int | None,
        **options:  Any,
    ) -> tuple[list[list[float]], int | None, dict]:
        """One request for a non-empty slice no larger than the provider takes.
        Returns ``(vectors, total_tokens or None, metadata)``, one vector per text."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _embed_chunk()"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _request(self, url: str, body: dict, extra_headers: dict | None = None) -> dict:
        """POST *body* as JSON and return the parsed reply; RuntimeError on non-2xx."""
        return request_json(self._http, "POST", url, json=body,
                            headers={"Accept": "application/json", **(extra_headers or {})},
                            what=f"{self.provider} Embeddings API")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, "
            f"provider={self.provider!r})"
        )


class _OpenAIShaped(Embedder):
    """The OpenAI embeddings wire, which Voyage and DashScope speak too. A provider
    on it is data: ``_URL``, its dimension field, floats or not, ``input_type`` or not.
    ``dimensions`` is sent whenever set: a fixed-length model refused it at construction."""

    _URL: str = ""
    _DIMS_FIELD: str = "dimensions"
    _FLOAT: bool = True
    _INPUT_TYPE: bool = False

    def _url(self) -> str:
        return self._URL

    def _embed_chunk(self, texts, *, input_type, dimensions, **options):
        body: dict = {"model": self.model, "input": texts}
        if self._FLOAT:
            body["encoding_format"] = "float"
        if dimensions is not None:
            body[self._DIMS_FIELD] = dimensions
        if self._INPUT_TYPE and input_type in ("query", "document"):
            body["input_type"] = input_type
        data = self._request(self._url(), body, {"Authorization": f"Bearer {self._api_key}"})
        items = sorted(data["data"], key=lambda x: x["index"])
        return [i["embedding"] for i in items], (data.get("usage") or {}).get("total_tokens"), {}
