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

from ._data import PROVIDERS

if TYPE_CHECKING:
    from ..clients._base import BaseClient


# ---------------------------------------------------------------------------
# Internal: build the API-family client for a provider (data-driven)
# ---------------------------------------------------------------------------

def _build_client(provider: str, api_key: str, client_options: dict) -> "BaseClient":
    """
    Construct the family client that owns *provider*'s wire format + transport.

    The provider's data file names its client family (``[provider].client``):
    one of ``openai`` / ``anthropic`` / ``google`` / ``perplexity`` / ``qwen``.
    The client receives the whole provider data dict, so it knows its
    endpoints, base URL and quirk branches; per-call model settings arrive in
    ``params`` later.
    """
    data  = PROVIDERS[provider]
    ctype = data["provider"]["client"]

    # Lazy imports keep module load cheap and avoid import cycles.
    from ..clients._families.openai     import OpenAIClient
    from ..clients._families.anthropic  import AnthropicClient
    from ..clients._families.google     import GoogleClient
    from ..clients._families.perplexity import PerplexityClient
    from ..clients._families.qwen       import QwenClient
    from ..clients._families.recraft    import RecraftClient
    from ..clients._families.bfl        import BFLClient

    family = {
        "openai":     OpenAIClient,
        "anthropic":  AnthropicClient,
        "google":     GoogleClient,
        "perplexity": PerplexityClient,
        "qwen":       QwenClient,
        "recraft":    RecraftClient,
        "bfl":        BFLClient,
    }[ctype]
    return family(api_key, data=data, **client_options)


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
    (re.compile(r"^(qwen|qwq|wanx|wan\d)"),                            "qwen"),
    (re.compile(r"^recraftv\d"),                                       "recraft"),
    (re.compile(r"^flux"),                                             "bfl"),
    (re.compile(r"^(gpt-|dall-e-|chatgpt-image-|text-embedding-|whisper-|tts-|o\d)"), "openai"),
]


# Provider keys accepted as an explicit ``"provider/model"`` prefix.
_PROVIDER_KEYS = frozenset({
    "openai", "anthropic", "google", "xai",
    "perplexity", "kimi", "deepseek", "qwen",
    "recraft", "bfl",
})


def _split_provider_prefix(name: str) -> "tuple[str | None, str]":
    """
    Split an explicit ``"provider/model"`` identifier.

    Returns ``(provider_key, model_name)`` when *name* starts with a known
    provider prefix (e.g. ``"openai/gpt-4o"`` → ``("openai", "gpt-4o")``),
    otherwise ``(None, name)`` — leaving auto-detection to ``_resolve_provider``.

    An explicit prefix also lets you use a custom model name the regex can't
    recognise, e.g. ``Model("openai/ft:gpt-4o:org:abc")``.
    """
    if "/" in name:
        head, _, tail = name.partition("/")
        if head.lower() in _PROVIDER_KEYS and tail:
            return head.lower(), tail
    return None, name


def _resolve_provider(name: str) -> str:
    """Return the provider key for *name*, or raise ``ValueError``."""
    explicit, model_name = _split_provider_prefix(name)
    if explicit:
        return explicit
    lower = model_name.lower()
    for pattern, provider in _PROVIDER_PATTERNS:
        if pattern.match(lower):
            return provider
    raise ValueError(
        f"Cannot detect provider for model {name!r}.\n"
        "Supported prefixes: claude-, gemini-, grok-, sonar, r1-1776, "
        "kimi-, deepseek-, gpt-, dall-e-, text-embedding-, whisper-, tts-, o<digit>.\n"
        "Use an explicit \"provider/model\" prefix (e.g. \"openai/my-custom\") "
        "or a provider subclass directly if you need a custom model name."
    )


# ---------------------------------------------------------------------------
# Model — factory + base
# ---------------------------------------------------------------------------

class Model:
    """
    A configured model: provider resolved from the name, settings from data,
    format + transport delegated to the matching family client.

    ``Model(name, options, client_options, api_key)`` resolves *name* to a
    provider (via the ``providers/`` data), merges that provider's default
    generation parameters, and attaches a ready-to-use family client.  There
    is a single ``Model`` class — no per-provider subclasses; the provider is
    available as ``model._provider``.

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
        ``"high"``.  Each provider's family client translates the universal
        level to its own native format via the provider's ``reasoning_map``
        data:

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
        Provider API key.  When omitted it is read from the provider's
        environment variable (e.g. ``OPENAI_API_KEY``), named in the provider
        data.  Raises ``ValueError`` if neither is found.

    Attributes
    ----------
    name          : str
    temperature   : float
    max_tokens    : int
    top_p         : float | None
    top_k         : int | None
    cache_control : bool
    reasoning     : str | None  (None | "low" | "medium" | "high")
    client        : family client (ready to use)

    Examples
    --------
    >>> m = Model("gpt-4o")
    >>> m._provider
    'openai'

    >>> m = Model("claude-sonnet-4-5", options={"temperature": 0.5, "reasoning": "high"})
    >>> m.temperature
    0.5

    >>> m = Model("gemini-2.0-flash", client_options={"proxy": {"url": "http://corp-proxy:3128"}})
    >>> m.client._base_url
    'https://generativelanguage.googleapis.com/v1beta'
    """

    # ------------------------------------------------------------------
    # Data-driven initialiser
    # ------------------------------------------------------------------

    def __init__(
        self,
        name: str,
        options:        dict | None = None,
        client_options: dict | None = None,
        api_key:        str  | None = None,
    ) -> None:
        # The provider is resolved from the (possibly prefixed) name; the wire
        # name has any "provider/" prefix stripped (it only steers selection).
        self._provider = _resolve_provider(name)
        self.name      = _split_provider_prefix(name)[1]

        prov     = PROVIDERS[self._provider]["provider"]
        defaults = prov["defaults"]

        # ── resolve API key (env var named in the provider data) ──────
        resolved_key = api_key or os.getenv(prov["env_key"])
        if not resolved_key:
            raise ValueError(
                f"No API key found for the {self._provider!r} provider. "
                f"Pass api_key= or set the {prov['env_key']!r} environment variable."
            )
        # Do not store the raw key as a public attribute; keep it private.
        self._api_key = resolved_key

        # ── merge options with provider defaults (from data) ──────────
        opts = options or {}
        self.temperature   = opts.get("temperature",   defaults.get("temperature"))
        self.max_tokens    = opts.get("max_tokens",    defaults.get("max_tokens"))
        self.top_p         = opts.get("top_p",         defaults.get("top_p"))
        self.top_k         = opts.get("top_k",         defaults.get("top_k"))
        self.cache_control = opts.get("cache_control", False)

        reasoning = opts.get("reasoning", None)
        if reasoning not in (None, "low", "medium", "high"):
            raise ValueError(
                f"reasoning must be None, 'low', 'medium', or 'high'; "
                f"got {reasoning!r}"
            )
        self.reasoning = reasoning

        # ── build the family client (format + transport) ──────────────
        self.client = _build_client(self._provider, resolved_key, client_options or {})

    # ------------------------------------------------------------------
    # Format — thin delegation to the family client
    # ------------------------------------------------------------------

    def _params(self) -> dict:
        """Per-call model settings handed to the client's ``build_request``."""
        return {
            "name":        self.name,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
            "top_p":       self.top_p,
            "top_k":       self.top_k,
            "reasoning":   self.reasoning,
        }

    def to_request(self, messages: list, output: dict) -> "tuple[str, dict]":
        """
        Translate substituted universal *messages* + *output* spec into the
        provider's native ``(path, body)`` pair, by delegating to the family
        client that owns this provider's wire format.
        """
        return self.client.build_request(messages, output, self._params())

    def from_response(self, response: dict, output: dict) -> "str | dict":
        """
        Extract the clean result (str for text, dict for json / image) from a
        raw provider response, by delegating to the family client.
        """
        return self.client.parse_response(response, output)

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


# ---------------------------------------------------------------------------
# Registry query — data-driven (capabilities live in providers/*.toml)
# ---------------------------------------------------------------------------
#
# Each model entry in the provider data carries a ``caps`` list (the tasks it
# supports).  These functions read that data; there is no separate registry.

#: Canonical task vocabulary.
TASKS: tuple[str, ...] = (
    "text-to-text", "text-to-image", "image-to-text", "image-to-image",
)


def _check_task(task: "str | None") -> None:
    if task is not None and task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Valid tasks: {list(TASKS)}")


def _check_provider(provider: "str | None") -> None:
    if provider is not None and provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Valid providers: {list(PROVIDERS)}"
        )


def models(provider: "str | None" = None, task: "str | None" = None) -> list[str]:
    """Return model names, optionally filtered by *provider* and/or *task*."""
    _check_provider(provider)
    _check_task(task)
    result: set[str] = set()
    for prov, data in PROVIDERS.items():
        if provider is not None and prov != provider:
            continue
        for name, mdl in data.get("models", {}).items():
            if task is None or task in mdl.get("caps", []):
                result.add(name)
    return sorted(result)


def providers(task: "str | None" = None) -> list[str]:
    """Return providers with at least one model (optionally supporting *task*)."""
    _check_task(task)
    out: list[str] = []
    for prov, data in PROVIDERS.items():
        ms = data.get("models", {})
        if not ms:
            continue
        if task is None or any(task in m.get("caps", []) for m in ms.values()):
            out.append(prov)
    return out


def tasks(model_name: str) -> list[str]:
    """Return the tasks supported by *model_name* (empty list if unknown)."""
    found: set[str] = set()
    for data in PROVIDERS.values():
        mdl = data.get("models", {}).get(model_name)
        if mdl is not None:
            found.update(mdl.get("caps", []))
    return sorted(found)


def is_supported(model_name: str, task: "str | None" = None) -> bool:
    """Return True when *model_name* is known (and supports *task* if given)."""
    _check_task(task)
    for data in PROVIDERS.values():
        mdl = data.get("models", {}).get(model_name)
        if mdl is not None and (task is None or task in mdl.get("caps", [])):
            return True
    return False


def refresh(provider: str, api_key: "str | None" = None, client=None) -> dict:
    """
    Diff the data registry against the provider's *live* model list.

    Calls the provider's ``list_models()`` and compares it to what the data
    knows, surfacing drift.  The data is **not** mutated.
    """
    _check_provider(provider)
    if provider is None:
        raise ValueError("provider is required")
    if client is None:
        env_key = PROVIDERS[provider]["provider"]["env_key"]
        key = api_key or os.getenv(env_key)
        if not key:
            raise ValueError(
                f"No API key for provider {provider!r}. "
                f"Set {env_key} or pass api_key=/client=."
            )
        client = _build_client(provider, key, {})

    live       = set(client.list_models())
    registered = set(models(provider=provider))
    return {
        "provider":   provider,
        "live":       sorted(live),
        "registered": sorted(registered),
        "new":        sorted(live - registered),
        "removed":    sorted(registered - live),
    }
