"""
models._base
============

``Model`` is both the **factory** and the **base class** for every
provider-specific model.

Factory usage (recommended)
---------------------------
::

    model = Model("gpt-4o")
    model = Model("claude-sonnet-4-5", options={"temperature": 0.3})
    model = Model("gemini-2.0-flash", api_key="AIza...")
    model = Model("grok-3", client_options={"proxy": {"url": "http://proxy:3128"}})

Direct subclass usage (also valid)
-----------------------------------
::

    model = OpenAIModel("gpt-4o", options={"max_tokens": 4096})

Provider detection is based on well-known model-name prefixes:

  ============  ============================================
  Provider      Prefixes / patterns
  ============  ============================================
  OpenAI        ``gpt-``, ``o<digit>``, ``dall-e-``,
                ``text-embedding-``, ``whisper-``, ``tts-``
  Anthropic     ``claude-``
  Google AI     ``gemini-``
  xAI           ``grok-``
  Perplexity    ``sonar``, ``r1-1776``
  Kimi          ``kimi-``
  ============  ============================================
"""

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clients._base import BaseClient


# ---------------------------------------------------------------------------
# Internal: provider prefix → provider key
# ---------------------------------------------------------------------------

_PROVIDER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^claude-"),                                          "anthropic"),
    (re.compile(r"^gemini-"),                                          "google"),
    (re.compile(r"^grok-"),                                            "xai"),
    (re.compile(r"^(sonar|r1-1776)"),                                  "perplexity"),
    (re.compile(r"^kimi-"),                                            "kimi"),
    (re.compile(r"^deepseek-"),                                        "deepseek"),
    (re.compile(r"^(qwen|qwq|wanx)"),                                  "qwen"),
    (re.compile(r"^(gpt-|dall-e-|text-embedding-|whisper-|tts-|o\d)"), "openai"),
]


def _resolve_provider(name: str) -> str:
    """Return the provider key for *name*, or raise ``ValueError``."""
    lower = name.lower()
    for pattern, provider in _PROVIDER_PATTERNS:
        if pattern.match(lower):
            return provider
    raise ValueError(
        f"Cannot detect provider for model {name!r}.\n"
        "Supported prefixes: claude-, gemini-, grok-, sonar, r1-1776, "
        "kimi-, deepseek-, gpt-, dall-e-, text-embedding-, whisper-, tts-, o<digit>.\n"
        "Use a provider subclass directly if you need a custom model name."
    )


# ---------------------------------------------------------------------------
# Model — factory + base
# ---------------------------------------------------------------------------

class Model:
    """
    Factory and base class for all provider-specific model instances.

    ``Model(name, options, client_options, api_key)`` inspects *name*,
    constructs the correct provider subclass, and returns a fully-
    configured instance — including an attached, ready-to-use client.

    Parameters
    ----------
    name : str
        Model identifier, e.g. ``"gpt-4o"``, ``"claude-sonnet-4-5"``,
        ``"gemini-2.0-flash"``, ``"grok-3"``, ``"sonar-pro"``.

    options : dict | None, optional
        Override any subset of the model's default generation parameters.
        Supported keys (all optional):

        ================  ========  =======================================
        Key               Type      Description
        ================  ========  =======================================
        ``temperature``   float     Sampling temperature.
        ``max_tokens``    int       Maximum output tokens.
        ``top_p``         float     Nucleus-sampling probability mass.
        ``top_k``         int       Top-K sampling (provider-dependent).
        ``cache_control`` bool      Enable provider-level prompt caching.
        ``reasoning``     str|None  Universal reasoning depth (see below).
        ================  ========  =======================================

        **reasoning** accepts ``None``, ``"low"``, ``"medium"``, or
        ``"high"``.  Each provider class translates the universal level to
        its own native format via ``_REASONING_MAP``:

        * **Anthropic**  — maps to ``{"type": "enabled", "budget_tokens": N}``
          (``low`` → 4 000, ``medium`` → 10 000, ``high`` → 20 000 tokens).
          Temperature is automatically forced to 1.0 when reasoning is active.

        * **Google AI**  — maps to ``{"thinkingBudget": N}`` inside
          ``generationConfig.thinkingConfig``
          (``low`` → 2 048, ``medium`` → 8 192, ``high`` → 24 576 tokens).

        * **OpenAI**     — maps to ``reasoning_effort`` on o-series models
          (``"low"`` / ``"medium"`` / ``"high"``).  GPT models ignore it.

        * **xAI**        — maps to ``reasoning_effort`` for grok-3-mini /
          grok-3-mini-fast (``low`` → ``"low"``, ``medium`` / ``high`` →
          ``"high"``).  Other grok models ignore it.

        * **Perplexity** — no reasoning parameter; value is silently ignored.

        * **Kimi**       — maps to ``thinking: {"type": "enabled"}`` when any
          level is set.  Kimi has no token-budget granularity; all three levels
          activate thinking mode.  Temperature is automatically forced to 1.0
          when thinking is active (API constraint).

        * **DeepSeek**   — maps to a model-name switch rather than an API
          parameter: ``"high"`` routes to ``deepseek-reasoner`` (always-on
          CoT); ``"low"`` / ``"medium"`` route to ``deepseek-chat`` (standard).
          For the reasoner, ``temperature`` / ``top_p`` are omitted from the
          request (ignored by the API).

    client_options : dict | None, optional
        Override settings for the underlying HTTP client.
        Supported keys (all optional):

        ===========  ========================  ==========================
        Key          Type                      Description
        ===========  ========================  ==========================
        ``url``      str                       Base URL override.
        ``timeout``  ``urllib3.Timeout``       Custom connect/read timeout.
        ``retries``  ``urllib3.Retry``         Custom retry policy.
        ``proxy``    dict                      Proxy config (see BaseClient).
        ===========  ========================  ==========================

    api_key : str | None, optional
        Provider API key.  When omitted the class reads from the
        environment variable named in ``_ENV_KEY`` (e.g.
        ``OPENAI_API_KEY``).  Raises ``ValueError`` if neither is found.

    Attributes
    ----------
    name          : str
    temperature   : float
    max_tokens    : int
    top_p         : float | None
    top_k         : int | None
    cache_control : bool
    reasoning     : str | None  (None | "low" | "medium" | "high")
    client        : BaseClient subclass (ready to use)

    Examples
    --------
    >>> m = Model("gpt-4o")
    >>> type(m)
    <class 'models._openai.OpenAIModel'>

    >>> m = Model("claude-sonnet-4-5", options={"temperature": 0.5, "reasoning": "high"})
    >>> m.temperature
    0.5

    >>> m = Model("gemini-2.0-flash", client_options={"proxy": {"url": "http://corp-proxy:3128"}})
    >>> m.client._base_url
    'https://generativelanguage.googleapis.com/v1beta'
    """

    # ------------------------------------------------------------------
    # Class-level defaults — overridden in every provider subclass.
    # These serve as the last-resort fallback; they should never be used
    # directly because BaseModel is never instantiated on its own.
    # ------------------------------------------------------------------

    #: Environment variable name for the provider API key.
    _ENV_KEY: str = ""

    _DEFAULT_TEMPERATURE:  float        = 1.0
    _DEFAULT_MAX_TOKENS:   int          = 4096
    _DEFAULT_TOP_P:        float | None = None
    _DEFAULT_TOP_K:        int   | None = None
    _DEFAULT_CACHE_CONTROL: bool        = False
    _DEFAULT_REASONING:    str   | None = None

    #: Maps universal reasoning levels to provider-native values.
    #: Overridden in each provider subclass; empty dict = no reasoning support.
    _REASONING_MAP: dict = {}

    # ------------------------------------------------------------------
    # Factory — __new__
    # ------------------------------------------------------------------

    def __new__(
        cls,
        name: str,
        options:        dict | None = None,
        client_options: dict | None = None,
        api_key:        str  | None = None,
    ):
        if cls is not Model:
            # Called as a subclass constructor (e.g. OpenAIModel("gpt-4o")).
            return object.__new__(cls)

        # Lazy imports avoid circular-reference issues at module load time.
        from ._openai     import OpenAIModel
        from ._anthropic  import AnthropicModel
        from ._google     import GoogleAIModel
        from ._xai        import XAIModel
        from ._perplexity import PerplexityModel
        from ._kimi       import KimiModel
        from ._deepseek   import DeepSeekModel
        from ._qwen       import QwenModel

        _MAP = {
            "openai":     OpenAIModel,
            "anthropic":  AnthropicModel,
            "google":     GoogleAIModel,
            "xai":        XAIModel,
            "perplexity": PerplexityModel,
            "kimi":       KimiModel,
            "deepseek":   DeepSeekModel,
            "qwen":       QwenModel,
        }

        provider = _resolve_provider(name)
        return object.__new__(_MAP[provider])

    # ------------------------------------------------------------------
    # Shared initialiser — inherited by all provider subclasses
    # ------------------------------------------------------------------

    def __init__(
        self,
        name: str,
        options:        dict | None = None,
        client_options: dict | None = None,
        api_key:        str  | None = None,
    ) -> None:
        self.name = name

        # ── resolve API key ──────────────────────────────────────────
        resolved_key = api_key or os.getenv(self._ENV_KEY)
        if not resolved_key:
            raise ValueError(
                f"No API key found for {type(self).__name__}. "
                f"Pass api_key= or set the {self._ENV_KEY!r} environment variable."
            )
        # Do not store the raw key as a public attribute; keep it private.
        self._api_key = resolved_key

        # ── merge options with provider defaults ──────────────────────
        opts = options or {}
        self.temperature   = opts.get("temperature",   self._DEFAULT_TEMPERATURE)
        self.max_tokens    = opts.get("max_tokens",    self._DEFAULT_MAX_TOKENS)
        self.top_p         = opts.get("top_p",         self._DEFAULT_TOP_P)
        self.top_k         = opts.get("top_k",         self._DEFAULT_TOP_K)
        self.cache_control = opts.get("cache_control", self._DEFAULT_CACHE_CONTROL)

        reasoning = opts.get("reasoning", self._DEFAULT_REASONING)
        if reasoning not in (None, "low", "medium", "high"):
            raise ValueError(
                f"reasoning must be None, 'low', 'medium', or 'high'; "
                f"got {reasoning!r}"
            )
        self.reasoning = reasoning

        # ── build the provider HTTP client ────────────────────────────
        self.client = self._build_client(resolved_key, client_options or {})

    # ------------------------------------------------------------------
    # Abstract — each provider subclass must implement these
    # ------------------------------------------------------------------

    def _build_client(self, api_key: str, client_options: dict) -> "BaseClient":
        """
        Construct and return the provider-specific HTTP client.

        Parameters
        ----------
        api_key        : Resolved provider API key.
        client_options : Caller-supplied overrides (url, timeout, retries,
                         proxy).

        Must be implemented by every subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_client()"
        )

    def to_request(self, messages: list, output: dict) -> "tuple[str, dict]":
        """
        Translate substituted universal *messages* and *output* spec into the
        provider's native ``(path, body)`` pair.

        Parameters
        ----------
        messages : list
            Substituted universal message list (variables already filled in).
        output : dict
            Universal output spec, e.g.
            ``{"modalities": ["text"], "format": {"type": "text"}}``.

        Returns
        -------
        tuple[str, dict]
            ``(path, body)`` ready to pass to ``client._post()``.

        Must be implemented by every subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement to_request()"
        )

    def from_response(self, response: dict, output: dict) -> "str | dict":
        """
        Extract the clean result from a raw provider API response.

        Parameters
        ----------
        response : dict
            Parsed JSON response body from the provider.
        output : dict
            Universal output spec used when building the request.

        Returns
        -------
        str
            When ``output["format"]["type"] == "text"``.
        dict
            When ``output["format"]["type"]`` is ``"json"`` or
            ``"json_schema"``.

        Must be implemented by every subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement from_response()"
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [
            f"name={self.name!r}",
            f"temperature={self.temperature}",
            f"max_tokens={self.max_tokens}",
        ]
        if self.top_p is not None:
            parts.append(f"top_p={self.top_p}")
        if self.top_k is not None:
            parts.append(f"top_k={self.top_k}")
        if self.cache_control:
            parts.append("cache_control=True")
        if self.reasoning is not None:
            parts.append(f"reasoning={self.reasoning!r}")
        return f"{type(self).__name__}({', '.join(parts)})"
