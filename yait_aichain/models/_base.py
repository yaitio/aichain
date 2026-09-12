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
    from ..clients._families.reve       import ReveClient

    family = {
        "openai":     OpenAIClient,
        "anthropic":  AnthropicClient,
        "google":     GoogleClient,
        "perplexity": PerplexityClient,
        "qwen":       QwenClient,
        "recraft":    RecraftClient,
        "bfl":        BFLClient,
        "reve":       ReveClient,
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
    (re.compile(r"^recraft"),                                          "recraft"),
    (re.compile(r"^flux"),                                             "bfl"),
    (re.compile(r"^reve-"),                                            "reve"),
    (re.compile(r"^(gpt-|dall-e-|chatgpt-image-|text-embedding-|whisper-|tts-|o\d)"), "openai"),
]


# Provider keys accepted as an explicit ``"provider/model"`` prefix.
_PROVIDER_KEYS = frozenset({
    "openai", "anthropic", "google", "xai",
    "perplexity", "kimi", "deepseek", "qwen",
    "recraft", "bfl", "reve", "private",
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


def _merge_usage(into: "dict | None", incoming: dict) -> dict:
    """Combine two usage envelopes of the same shape, field by field.

    Two levels is all the shape has — an envelope key (`usage` /
    `usageMetadata`) over a flat block of counts — and a provider that
    reports a field twice means the later value, not the sum: Anthropic's
    `message_delta` restates `output_tokens` as it grows.
    """
    merged = {k: dict(v) for k, v in (into or {}).items()}
    for key, block in incoming.items():
        if isinstance(block, dict):
            merged.setdefault(key, {}).update(block)
        else:
            merged[key] = block
    return merged


class _ToolCallAssembly:
    """Collects tool-call fragments from a stream into whole calls.

    Two rules, and both were learned from what breaks without them:

    * **Slot, not arrival order.** Providers interleave the fragments of
      several calls in one turn, and the index is the only thing that says
      which is which.
    * **Arguments are concatenated, then parsed once.** Parsing on the way
      sees truncated JSON on every fragment but the last. Parsing is left to
      the same builder the buffered path uses, so a malformed argument string
      fails identically whether it was streamed or not.
    """

    def __init__(self) -> None:
        self._slots: dict = {}

    def add(self, fragments: list) -> None:
        for f in fragments or []:
            slot = self._slots.setdefault(
                f.get("slot", len(self._slots)),
                {"id": "", "name": "", "arguments": ""})
            # An id and a name arrive once, usually with the first fragment;
            # later ones carry arguments alone. Overwriting with the empty
            # string that follows is how a call loses its name and becomes
            # unroutable.
            if f.get("id"):
                slot["id"] = f["id"]
            if f.get("name"):
                slot["name"] = f["name"]
            if f.get("arguments"):
                slot["arguments"] += f["arguments"]

    def request(self, text: str = ""):
        """The assembled :class:`ToolCallRequest`, or None if nothing came."""
        if not self._slots:
            return None
        from ..clients._families._openai_compat import _tool_call_request
        return _tool_call_request(
            [(slot["name"], slot["arguments"], slot["id"])
             for _, slot in sorted(self._slots.items(),
                                   key=lambda kv: str(kv[0]))],
            text)

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
        ``cache_control`` bool|int  Prompt caching. True marks the
                                    second-to-last message; an int names
                                    the message the stable prefix ends at.
        ``cache_ttl``     str       How long the mark lives: ``"5m"``
                                    (default) or ``"1h"``.
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
        on_unsupported: str = "warn",
    ) -> None:
        # The provider is resolved from the (possibly prefixed) name; the wire
        # name has any "provider/" prefix stripped (it only steers selection).
        self._provider = _resolve_provider(name)
        self.name      = _split_provider_prefix(name)[1]

        prov     = PROVIDERS[self._provider]["provider"]
        defaults = prov["defaults"]

        # ── resolve API key (env var named in the provider data) ──────
        # auth = "none" marks a provider whose key is optional rather than
        # required — a local server usually runs open, but accepts a Bearer
        # token when started with --api-key. So the key is still *resolved*
        # (passed or from env) and still sent when present; only the "you
        # must have one" gate is lifted.
        # A callable key is resolved per request rather than here, and that
        # is the whole of multi-tenant support. One `Model` serves every
        # tenant; the key belongs to whichever run is in flight, so resolving
        # at construction would bind the first tenant's credential to every
        # later one. The callable is handed the current `RunContext` — which
        # is emphatically *not* where the secret lives: that object is
        # serialised into the run document and `FileStore` writes it to disk,
        # so it carries the tenant's **name** and the caller looks the key up
        # from it.
        self._resolve_key = api_key if callable(api_key) else None
        resolved_key = "" if self._resolve_key else (
            api_key or os.getenv(prov["env_key"]))
        if not self._resolve_key and not resolved_key:
            if prov.get("auth") == "none":
                resolved_key = ""
            else:
                raise ValueError(
                    f"No API key found for the {self._provider!r} provider. "
                    f"Pass api_key= or set the {prov['env_key']!r} environment variable."
                )
        # Do not store the raw key as a public attribute; keep it private.
        self._api_key = resolved_key

        # ── merge options with provider defaults (from data) ──────────
        opts = options or {}
        # What the caller asked for, apart from what the defaults supply. Only
        # these are reported when an option does not survive to the wire: a
        # default that a provider ignores is not something anybody chose.
        self._asked = dict(opts)

        # An option this library has no word for is a mistake, not a request,
        # and the place to say so is here — where it was written — rather than
        # at the wire or, as before, nowhere at all. A misspelt `temperatur`
        # used to be accepted, ignored, and cost a whole run to notice.
        from ._options import UNIVERSAL_OPTIONS
        unknown = [k for k in opts if k not in UNIVERSAL_OPTIONS]
        if unknown:
            raise ValueError(
                f"unknown model option(s): {', '.join(map(repr, sorted(unknown)))}. "
                f"Known options are: {', '.join(sorted(UNIVERSAL_OPTIONS))}. "
                "Per-call settings such as image size or quality belong in "
                "output={'format': {...}}, not here."
            )
        self.temperature   = opts.get("temperature",   defaults.get("temperature"))
        self.max_tokens    = opts.get("max_tokens",    defaults.get("max_tokens"))
        self.top_p         = opts.get("top_p",         defaults.get("top_p"))
        self.top_k         = opts.get("top_k",         defaults.get("top_k"))
        # Prompt caching. The option has been accepted and documented since
        # before 1.6.1 but reached nothing but ``__repr__``; it is wired to the
        # family clients here. Off by default: a cache breakpoint costs 1.25x
        # to store, so it only pays back when the same prefix is sent again.
        # True marks the second-to-last message; an int names the message the
        # stable prefix ends at, for callers that know where that is.
        cc = opts.get("cache_control", defaults.get("cache_control", False))
        self.cache_control = cc if isinstance(cc, int) and not isinstance(cc, bool) else bool(cc)
        # How long the stored prefix lives. Which one is right is a property of
        # the caller's cadence, not of the library: a write costs 1.25x at five
        # minutes and 2x at an hour, against 0.1x per read, so 5m pays back on
        # the second read and 1h on the third. A caller invoking a skill every
        # few minutes misses the 5m window every time and pays the write
        # premium forever at a hit rate of zero — worse than not caching — so
        # the choice has to be reachable even though the default stays 5m.
        ttl = opts.get("cache_ttl", defaults.get("cache_ttl", "5m"))
        if ttl not in ("5m", "1h"):
            raise ValueError(
                f"cache_ttl must be '5m' or '1h'; got {ttl!r}"
            )
        self.cache_ttl = ttl

        # Valid levels come from the provider's own reasoning_map — that map is
        # the single source of truth for what this provider can express. The
        # previous hardcoded (low, medium, high) was a second copy of it, and
        # the day openai gained "none" the copy rejected a value the provider
        # itself accepts.
        reasoning = opts.get("reasoning", None)
        rmap      = prov.get("reasoning_map", {})
        if reasoning is not None and reasoning not in rmap:
            allowed = ", ".join(repr(k) for k in rmap) or "(none for this provider)"
            raise ValueError(
                f"reasoning for the {self._provider!r} provider must be None "
                f"or one of: {allowed}; got {reasoning!r}"
            )
        self.reasoning = reasoning

        # What to do when an option cannot be honoured here. The library
        # will not decide this: dropping `temperature` yields a different
        # answer, dropping `seed` or `size` yields one that looks right and
        # is not, and whether that is fatal depends on the caller and not on
        # the option. 'warn' sends the request without it, 'requirements'
        # raises for the second class only, 'raise' for any loss. See
        # models._adaptation.STRICTNESS.
        from ._adaptation import STRICTNESS
        if on_unsupported not in STRICTNESS:
            raise ValueError(
                f"on_unsupported must be one of {', '.join(map(repr, STRICTNESS))}; "
                f"got {on_unsupported!r}")
        self.on_unsupported = on_unsupported

        #: What the last :meth:`to_request` had to change to fit the provider.
        #: Empty when the request went out as asked.
        self.last_adaptations: list = []

        #: Token counts from the last :meth:`stream`, when the provider sent
        #: them. None rather than a guess: counting the text we happened to
        #: see would be a number with no provider behind it.
        self.last_stream_usage: "dict | None" = None

        #: The last :meth:`stream`'s answer as an object, when the provider
        #: could not stream and it came back whole. None on a real stream,
        #: where the pieces are the answer.
        self.last_stream_result = None

        # ── build the family client (format + transport) ──────────────
        self.client = _build_client(self._provider, resolved_key, client_options or {})
        if self._resolve_key is not None:
            # The client asks at header time, once per request.
            self.client._resolve_key = self._resolve_key

    # ------------------------------------------------------------------
    # Format — thin delegation to the family client
    # ------------------------------------------------------------------

    @property
    def effective_options(self) -> dict:
        """What sampling this model actually runs with, asked for or not.

        Provider defaults are not a shared baseline and never were:
        ``temperature`` starts at 0.0 on DeepSeek, 0.2 on Perplexity, 0.7 on
        Qwen and 1.0 on OpenAI, Google, Kimi and xAI; ``max_tokens`` spans
        2048 to 32768; ``top_k`` has a value on Google alone. Two models
        compared with no options set are therefore not one variable apart,
        which is exactly the mistake this library tells its users not to make.

        Converging them was considered and rejected: a library-wide default
        overrides a number each provider chose for its own model, and it
        would change every existing caller's output to fix a comparison only
        some of them are making. Publishing the fact is the honest half —
        ``defaults`` in the provider data, this property at runtime, and a
        table in ``docs/reference/parameters.md``. A measurement run records
        this beside its results and its arms are comparable or visibly not.
        """
        return {
            "temperature":   self.temperature,
            "max_tokens":    self.max_tokens,
            "top_p":         self.top_p,
            "top_k":         self.top_k,
            "reasoning":     self.reasoning,
            "cache_control": self.cache_control,
            "cache_ttl":     self.cache_ttl,
            "_asked":        dict(self._asked),
        }

    def _params(self) -> dict:
        """Per-call model settings handed to the client's ``build_request``."""
        return {
            "name":        self.name,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
            "top_p":       self.top_p,
            "top_k":       self.top_k,
            "reasoning":   self.reasoning,
            "cache_control": self.cache_control,
            "cache_ttl":     self.cache_ttl,
        }

    def to_request(self, messages: list, output: dict,
                   tools: "list | None" = None) -> "tuple[str, dict]":
        """
        Translate substituted universal *messages* + *output* spec into the
        provider's native ``(path, body)`` pair, by delegating to the family
        client that owns this provider's wire format.

        *tools* is a list of function-calling schemas (what ``Tool.schema()``
        returns) declared to the provider natively. Families that have not
        implemented the tool wire format refuse loudly here — silently sending
        a request without the declared tools would produce a model that simply
        never calls anything, with no error anywhere.
        """
        if tools:
            if not getattr(self.client, "supports_tools", False):
                raise ValueError(
                    f"provider family {type(self.client).__name__!r} does not "
                    f"support native tool calling yet; model {self.name!r} "
                    "cannot be given tools"
                )
            path, body = self._built(messages, output, tools=tools)
        else:
            path, body = self._built(messages, output)
        return path, body

    def _canonical_format(self, output: dict) -> dict:
        """Rename an option that has been renamed, and say so once.

        Removing the old name would break working code; accepting it in
        silence would leave a caller writing a name that no longer exists in
        the documentation. So it is translated and reported, which is what
        the channel is for."""
        from ._adaptation import Adaptation, TRANSLATED, record
        from ._options import ALIASES, canonical_value

        fmt = output.get("format") or {}
        renamed = {k: v for k, v in fmt.items() if k in ALIASES}
        # A shorthand is resolved to the canonical type before anything else
        # looks at it, so a range declared for one provider does not reject a
        # word another provider needs.
        shorthand = {}
        for key, value in fmt.items():
            fixed_value, note = canonical_value(ALIASES.get(key, key), value)
            if note:
                shorthand[key] = (fixed_value, note)
        if not renamed and not shorthand:
            return output
        fixed = {k: v for k, v in fmt.items() if k not in ALIASES}
        # Not recorded as an adaptation: resolving 'high' to 0.9 is
        # normalising the caller's own input to the vocabulary's one type,
        # not something a provider did. Recording it made every provider that
        # then drops the option report "translated" for an option that never
        # left.
        for key, (value, _note) in shorthand.items():
            if key not in ALIASES:
                fixed[key] = value
        for old, value in renamed.items():
            new = ALIASES[old]
            value = shorthand.get(old, (value, None))[0]
            fixed.setdefault(new, value)
            record(Adaptation(
                kind=TRANSLATED, option=old, asked=value, sent=new,
                model=self.name,
                why=f"{old!r} is now called {new!r}; both are accepted"))
        return {**output, "format": fixed}

    def _check_values(self, output: dict) -> dict:
        """Refuse a value this model does not take, before the wire.

        Otherwise the provider answers for us, after the round trip is spent:
        "Invalid value: 'ultra-max-supreme'. Supported values are: …". The
        library knows the same thing and knows it earlier.
        """
        from ._adaptation import Adaptation, ADAPTED, record
        from ._options import check_applies, check_value

        fmt = (output.get("format") or {})
        kind = fmt.get("type", "text")
        fixed = None
        for key, value in fmt.items():
            if key in ("type", "schema", "name", "strict") or value is None:
                continue
            # Wrong everywhere, so it stops here rather than being dropped
            # and reported: a caller cannot fix this by changing provider.
            check_applies(key, kind)
            sent, note = check_value(key, value, self._provider, self.name)
            if note is not None:
                fixed = {**(fixed or fmt), key: sent}
                record(Adaptation(kind=ADAPTED, option=key, asked=value,
                                  sent=sent, model=self.name, why=note))
        return {**output, "format": fixed} if fixed else output

    def _built(self, messages: list, output: dict, tools=None):
        """Build, then say what building changed.

        Everything a provider cannot take unchanged is adapted here or in the
        family client; this is the one place all of it passes through, so it
        is where the adaptations are gathered and announced. See
        ``models._adaptation``: the library may adapt a request, it may not do
        so in silence.
        """
        from ._adaptation import announce, collect, enforce, note_absent

        with collect() as made:
            output = self._check_values(self._canonical_format(output))
            if tools is not None:
                path, body = self.client.build_request(
                    messages, output, self._params(), tools=tools)
            else:
                path, body = self.client.build_request(
                    messages, output, self._params())

        # Options set in the output format travel beside the model's own, and
        # are asked for per call rather than per model.
        asked = {**self._asked,
                 **{k: v for k, v in (output.get("format") or {}).items()
                    if k not in ("type", "schema", "name", "strict")}}
        note_absent(made, asked, body, self.name, provider=self._provider)
        announce(made)
        self.last_adaptations = list(made)
        # After the record is stored, not before: a caller catching this needs
        # `last_adaptations` to say what happened, and a raise that leaves the
        # evidence unwritten is the silence this whole channel exists against.
        enforce(made, self.on_unsupported)
        return path, body

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(self, messages: list, output: dict,
               tools: "list | None" = None):
        """
        Yield the answer in pieces as the provider produces it.

        Not every provider can. An image endpoint has nothing to deliver
        progressively, and two of OpenAI's own paths speak a different event
        vocabulary that is not wired yet. The question that decides the
        design is what to do then, and it is the same question 2.3.0 answered
        for options: **the request still goes, and the caller is told.** So a
        provider that cannot stream returns the whole answer as one piece and
        records a ``declined`` — because a caller who asked to stream and got
        one chunk at the end has something that looks like it worked, and
        that is precisely the class of loss `on_unsupported="requirements"`
        exists to raise on.

        Usage lands on ``last_stream_usage`` when the provider reports it.
        """
        from ._adaptation import (Adaptation, DECLINED, announce, collect,
                                  enforce, note_absent)

        self.last_stream_usage = None
        self.last_stream_result = None
        with collect() as made:
            output = self._check_values(self._canonical_format(output))
            try:
                path, body = self.client.build_stream_request(
                    messages, output, self._params(), tools=tools)
            except NotImplementedError as why:
                made.append(Adaptation(
                    kind=DECLINED, option="stream", asked=True, sent=None,
                    model=self.name,
                    why=f"{why}; the answer arrives in one piece at the end"))
                path = body = None

        asked = {**self._asked,
                 **{k: v for k, v in (output.get("format") or {}).items()
                    if k not in ("type", "schema", "name", "strict")}}
        if body is not None:
            note_absent(made, asked, body, self.name, provider=self._provider)
        announce(made)
        self.last_adaptations = list(made)
        enforce(made, self.on_unsupported)

        if body is None:
            # `_built` reports on its own build and overwrites the record,
            # which would drop the `declined` just made — the notice about
            # streaming would be warned about and then vanish from the
            # machine-readable half, which is the half that matters.
            spoken = list(made)
            path, body = self._built(messages, output, tools=tools)
            self.last_adaptations = spoken + list(self.last_adaptations)
            import json as _json
            raw = self.client.send(path, body, self.client._auth_headers())
            response = _json.loads(raw)
            # The whole response is already the envelope `extract_usage`
            # reads, per provider — taking `response["usage"]` out of it
            # worked for two families and read as zero for Google.
            self.last_stream_usage = response
            result = self.from_response(response, output)
            # Kept whole beside the text. A tool call is neither a string nor
            # a dict, and rendering one to text and throwing the object away
            # is how a streamed tool-calling turn came to look like a model
            # that answered nothing.
            self.last_stream_result = result
            if isinstance(result, str):
                yield result
            elif isinstance(result, dict):
                yield _json.dumps(result)
            else:
                # A tool call's own text, which is usually empty — and an
                # empty stream is right here. What must not go out is its
                # repr: a caller printing the pieces would print
                # "ToolCallRequest(calls=(...))" to a user's screen.
                text = getattr(result, "text", "") or ""
                if text:
                    yield text
            return

        assembly = _ToolCallAssembly()
        spoken: list = []
        for event in self.client._post_sse(
                path, body, self.client._auth_headers()):
            assembly.add(self.client.stream_tool_fragments(event))
            usage = self.client.stream_usage(event)
            if usage:
                # Merged, not replaced. Anthropic reports the input tokens in
                # `message_start` and the output tokens in `message_delta`, so
                # keeping the last report drops the entire prompt from the
                # bill — and the number that comes back is still plausible,
                # which is what makes it worth a comment rather than a line.
                self.last_stream_usage = _merge_usage(
                    self.last_stream_usage, usage)
            piece = self.client.parse_stream_event(event, output)
            if piece:
                spoken.append(piece)
                yield piece

        # A turn that asked for tools is a decision, not prose, so it is left
        # as an object for the caller to act on. Any text the model sent
        # beside the calls has already been yielded and rides along in
        # `text`, so nothing is shown twice and nothing is lost.
        self.last_stream_result = assembly.request("".join(spoken))

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
