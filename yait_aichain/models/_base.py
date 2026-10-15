"""``Model``: a name resolved to a provider, settings from the provider data,
format and transport delegated to that provider's family client.

Also the data-driven registry queries (``models``, ``providers``, ``tasks``,
``is_supported``). See docs/models/models.md.
"""

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .. import _hints
from ._data import PROVIDERS

if TYPE_CHECKING:
    from ..clients._base import BaseClient


# ---------------------------------------------------------------------------
# Internal: build the API-family client for a provider (data-driven)
# ---------------------------------------------------------------------------

def _build_client(provider: str, api_key: str, client_options: dict) -> "BaseClient":
    """The family client named by ``[provider].client``, given the whole
    provider data; per-call model settings arrive later in ``params``."""
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
    _hints.check_client_options(ctype, client_options)
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
    (re.compile(r"^glm-"),                                             "zai"),
    (re.compile(r"^muse-"),                                            "meta"),
    (re.compile(r"^mimo-"),                                            "xiaomi"),
    (re.compile(r"^minimax-"),                                         "minimax"),
    (re.compile(r"^step-"),                                            "stepfun"),
    (re.compile(r"^(mistral-|ministral-|codestral-|magistral-|devstral-)"), "mistral"),
    (re.compile(r"^(gpt-|dall-e-|chatgpt-image-|text-embedding-|whisper-|tts-|o\d)"), "openai"),
]


#: The prefixes above as a person reads them, for the error that lists them.
_PREFIXES = ", ".join(
    alt.replace("\\d", "<digit>")
    for pattern, _ in _PROVIDER_PATTERNS
    for alt in pattern.pattern.lstrip("^").strip("()").split("|"))

# Provider keys accepted as an explicit ``"provider/model"`` prefix.
_PROVIDER_KEYS = frozenset(PROVIDERS)


def _split_provider_prefix(name: str) -> "tuple[str | None, str]":
    """``"openai/gpt-4o"`` → ``("openai", "gpt-4o")``; ``(None, name)`` when the
    head is not a known provider. The prefix is how a custom name the patterns
    cannot recognise is routed."""
    if "/" in name:
        head, _, tail = name.partition("/")
        if head.lower() in _PROVIDER_KEYS and tail:
            return head.lower(), tail
    return None, name


#: Names already reported as unlisted, so a loop that builds one model a call
#: says it once.
_UNLISTED_SAID: set = set()


def _note_unlisted(name: str, provider: str) -> None:
    """Say once that *name* was routed by its prefix alone, with the closest
    listed names: a retired name and a typo look exactly like this. An explicit
    `provider/name` means it, and is not reported."""
    if (provider, name) in _UNLISTED_SAID:
        return
    _UNLISTED_SAID.add((provider, name))
    import difflib
    import warnings
    known = list(PROVIDERS[provider].get("models", {}))
    close = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(map(repr, close))}?" if close else ""
    warnings.warn(
        f"{name!r} is not in the {provider} registry; it was routed to {provider} by its "
        f"prefix and is sent as written. Its price is unknown (cost is None) and its "
        f"capabilities are the provider's defaults.{hint} Write {provider + '/' + name!r} "
        "to say the name is meant.", stacklevel=3)


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
        f"Supported prefixes: {_PREFIXES}.\n"
        "Use an explicit \"provider/model\" prefix (e.g. \"openai/my-custom\", "
        "\"private/llama3.3\") for a name the prefixes do not cover."
    )


# ---------------------------------------------------------------------------
# Model — factory + base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamOutcome:
    """What a finished `Model.stream` returns: the whole result — text, a
    parsed object, or a `ToolCallRequest` — the usage envelope the provider
    reported (None when it reported none), the adaptations made, and whether
    the output budget cut the answer short."""

    result: object
    usage: dict | None
    adaptations: list
    truncated: bool = False


def _merge_usage(into: "dict | None", incoming: dict) -> dict:
    """Combine two usage envelopes (envelope key over a flat block of counts).
    A field reported twice means the later value, not the sum: Anthropic's
    `message_delta` restates `output_tokens` as it grows."""
    merged = {k: dict(v) for k, v in (into or {}).items()}
    for key, block in incoming.items():
        if isinstance(block, dict):
            merged.setdefault(key, {}).update(block)
        else:
            merged[key] = block
    return merged


class _ToolCallAssembly:
    """Tool-call fragments from a stream, assembled by slot (providers interleave
    calls), arguments concatenated and parsed once by the buffered path's own
    builder, so a malformed one fails the same either way."""

    def __init__(self) -> None:
        self._slots: dict = {}

    def add(self, fragments: list) -> None:
        for f in fragments or []:
            # A fragment with no slot is a whole call of its own: it gets a
            # new one, never a slot an earlier call already holds.
            key = f["slot"] if "slot" in f else ("whole", len(self._slots))
            slot = self._slots.setdefault(key, {"id": "", "name": "", "arguments": ""})
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
        # In the order the calls began, as the buffered reply lists them:
        # sorted as strings, slot 10 ran before slot 2.
        return _tool_call_request(
            [(slot["name"], slot["arguments"], slot["id"]) for slot in self._slots.values()],
            text)


#: The sampling options a Model checks at construction: (type, lowest, highest).
_SAMPLING = {
    "temperature": ((int, float), 0, None),
    "top_p":       ((int, float), 0, 1),
    "top_k":       (int, 1, None),
    "max_tokens":  (int, 1, None),
}


def _check_name(name) -> None:
    """A model name is part of a URL path on some providers — Google's
    `models/{name}:generateContent` — so one that could walk the path
    (`gemini-x/../..`) or end it (`?`, `#`, a space) is refused."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"a model name is a non-empty string, not {name!r}")
    if any(seg in ("..", ".") for seg in name.split("/")) or any(
            c in name for c in "?#\\ \t\r\n%"):
        raise ValueError(f"model name {name!r} cannot be part of a URL path")

class Model:
    """
    A configured model: provider resolved from the name, settings from data,
    format and transport delegated to the matching family client.

    One ``Model`` class serves every provider; the provider is
    ``model._provider``.

    Parameters
    ----------
    name : str
        Model identifier, e.g. ``"gpt-4o"``, ``"claude-sonnet-4-5"``.
    options : dict | None, optional
        Overrides of the defaults: ``temperature``, ``max_tokens``, ``top_p``,
        ``top_k``, ``reasoning`` (``"none"``/``"low"``/``"medium"``/``"high"``),
        ``cache_control`` (True, a message index, or ``"last"`` for an agent
        loop), ``cache_ttl`` (``"5m"``/``"1h"``), ``extra`` (fields sent
        verbatim). Per provider: ``docs/reference/parameters.md``.
    client_options : dict | None, optional
        HTTP client settings: ``url`` (base URL), ``timeout``
        (``urllib3.Timeout``), ``retries`` (``urllib3.Retry``), ``proxy`` (dict).
    api_key : str | None, optional
        Provider key; read from the provider's environment variable when
        omitted. ``ValueError`` if neither is found.
    on_unsupported : str, optional
        ``"warn"`` (default) adapts and records in ``last_adaptations``;
        ``"requirements"`` raises only for losses that leave a result looking
        right (``seed``, ``size``, ``aspect_ratio``, ...); ``"raise"`` for any.

    Attributes
    ----------
    name, temperature, max_tokens, top_p, top_k, cache_control, reasoning
        The resolved settings.
    client : family client, ready to use.

    Examples
    --------
    >>> Model("claude-sonnet-4-5", options={"temperature": 0.5}).temperature
    0.5
    """

    # ------------------------------------------------------------------
    # Data-driven initialiser
    # ------------------------------------------------------------------

    @_hints.refuse("Model")
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
        _check_name(name)
        self._provider = _resolve_provider(name)
        explicit, self.name = _split_provider_prefix(name)
        #: Whether the registry knows this model. An unknown one still runs —
        #: custom and new models have to — but its price is unknown and its
        #: capabilities are the provider's defaults.
        self.listed = self.name in PROVIDERS[self._provider].get("models", {})
        if not self.listed and not explicit:
            _note_unlisted(self.name, self._provider)

        prov     = PROVIDERS[self._provider]["provider"]
        defaults = prov["defaults"]

        # ── resolve API key (env var named in the provider data) ──────
        # auth = "none": the key is optional, not absent — a local server
        # usually runs open but accepts a Bearer token, so it is still
        # resolved and sent when present.
        # A callable key is resolved per request: one `Model` serves every
        # tenant, and resolving here would bind the first tenant's key to all.
        # It is handed the `RunContext`, which is written to disk with the run
        # document, so it carries the tenant's name and never the secret.
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
        # A copy: popping `extra` from the caller's dict would strip it from
        # the next Model built from the same options.
        opts = dict(options or {})
        # What the caller asked for, apart from what the defaults supply. Only
        # these are reported when an option does not survive to the wire: a
        # default that a provider ignores is not something anybody chose.
        # `extra` is excluded: its fields are outside the vocabulary
        # `note_absent` reports on.
        self._asked = {k: v for k, v in opts.items() if k != "extra"}

        # An unknown option is a mistake, refused where it was written: a
        # misspelt `temperatur` ignored would cost a whole run to notice.
        from ._options import UNIVERSAL_OPTIONS
        # `extra` is the escape hatch a closed vocabulary needs (Perplexity's
        # `search_domain_filter` has no universal meaning). It goes on the
        # wire verbatim and unchecked: nothing translates, declines or
        # reports it.
        extra = opts.pop("extra", None)
        if extra is not None and not isinstance(extra, dict):
            raise ValueError(
                f"extra must be a dict of provider fields; got "
                f"{type(extra).__name__}")
        self.extra = dict(extra or {})

        unknown = [k for k in opts if k not in UNIVERSAL_OPTIONS]
        if unknown:
            raise ValueError(
                f"unknown model option(s): {', '.join(map(repr, sorted(unknown)))}. "
                f"Known options are: {', '.join(sorted(UNIVERSAL_OPTIONS))}. "
                "Per-call settings such as image size or quality belong in "
                "output={'format': {...}}, not here. A field only one "
                "provider has goes in options={'extra': {...}}, which travels "
                "verbatim and unchecked."
            )
        # Checked before any request, so a wrong type or range fails here
        # rather than as a 400 after the round trip.
        self._clamped: list = []
        for name, (kind, low, high) in _SAMPLING.items():
            if name not in opts:
                continue
            value = opts[name]
            if value is None or isinstance(value, bool) or not isinstance(value, kind):
                raise ValueError(f"{name} is a {'whole number' if kind is int else 'number'}, "
                                 f"not {value!r}; leave it out for the provider's default")
            if value < low or (high is not None and value > high):
                raise ValueError(f"{name} is between {low} and {high if high is not None else 'any'}; "
                                 f"got {value}")
            from ._options import check_value
            sent, note = check_value(name, value, self._provider, self.name)
            if note:
                opts[name] = sent
                self._clamped.append((name, value, sent, note))
        self.temperature   = opts.get("temperature",   defaults.get("temperature"))
        self.max_tokens    = opts.get("max_tokens",    defaults.get("max_tokens"))
        self.top_p         = opts.get("top_p",         defaults.get("top_p"))
        self.top_k         = opts.get("top_k",         defaults.get("top_k"))
        # Prompt caching, off by default: a cache breakpoint costs 1.25x to
        # store, so it only pays back when the same prefix is sent again.
        # True marks the second-to-last message; an int names the message the
        # stable prefix ends at, for callers that know where that is.
        # ``"last"`` marks the newest block, for agent loops (see the
        # Anthropic client). Any other string is refused: read as ``bool()`` it
        # would become ``True``, and a misspelt placement would choose another.
        cc = opts.get("cache_control", defaults.get("cache_control", False))
        if isinstance(cc, str):
            if cc != "last":
                raise ValueError(
                    f"cache_control={cc!r} is not a placement. It takes True "
                    "(the second-to-last message), an int (the message the "
                    "stable prefix ends at), or \"last\" (the newest block, "
                    "for an agent loop).")
            self.cache_control = cc
        else:
            self.cache_control = cc if isinstance(cc, int) and not isinstance(cc, bool) else bool(cc)
        # How long the stored prefix lives depends on the caller's cadence: a
        # write costs 1.25x at 5m and 2x at 1h against 0.1x per read, and a
        # caller every few minutes misses a 5m window every time — worse than
        # not caching. So 1h is reachable though 5m stays the default.
        ttl = opts.get("cache_ttl", defaults.get("cache_ttl", "5m"))
        if ttl not in ("5m", "1h"):
            raise ValueError(
                f"cache_ttl must be '5m' or '1h'; got {ttl!r}"
            )
        self.cache_ttl = ttl

        # Valid levels come from the provider's own reasoning_map, the single
        # source of truth: a second copy here would reject a level the
        # provider gains.
        reasoning = opts.get("reasoning", None)
        rmap      = dict(prov.get("reasoning_map", {}))
        # A provider that switches thinking off rather than naming a level for
        # it says so as `[provider.reasoning] off`; "none" is then a level too.
        if (prov.get("reasoning") or {}).get("off"):
            rmap.setdefault("none", None)
        if reasoning is not None and reasoning not in rmap:
            allowed = ", ".join(repr(k) for k in rmap) or "(none for this provider)"
            raise ValueError(
                f"reasoning for the {self._provider!r} provider must be None "
                f"or one of: {allowed}; got {reasoning!r}"
            )
        self.reasoning = reasoning

        # Whether a lost option is fatal is the caller's call, not the
        # library's; see models._adaptation.STRICTNESS.
        from ._adaptation import STRICTNESS
        if on_unsupported not in STRICTNESS:
            raise ValueError(
                f"on_unsupported must be one of {', '.join(map(repr, STRICTNESS))}; "
                f"got {on_unsupported!r}")
        self.on_unsupported = on_unsupported

        #: What the last :meth:`to_request` had to change to fit the provider.
        #: Empty when the request went out as asked.
        self.last_adaptations: list = []
        self._sent_tools = False

        #: Token counts from the last :meth:`stream`, when the provider sent
        #: them. None rather than a guess: counting the text we happened to
        #: see would be a number with no provider behind it.
        self.last_stream_usage: dict | None = None

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

        Provider defaults differ (``temperature`` from 0.0 to 1.0,
        ``max_tokens`` from 2048 to 32768), so two models compared with no
        options set are not one variable apart: record this beside a
        measurement. The defaults per provider: ``docs/reference/parameters.md``.
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
            # `extra` is reported nowhere else, but it goes on the wire: left
            # out, two arms differing only by it would record as identical.
            "extra":         dict(self.extra),
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
        Translate universal *messages* + *output* into the provider's native
        ``(path, body)`` pair, through the family client.

        Parameters
        ----------
        messages : list
            Universal messages, placeholders filled: ``{"role", "parts"}``.
        output : dict
            The output spec: ``{"modalities": [...], "format": {"type": …}}``.
        tools : list | None, optional
            Function-calling schemas (``Tool.schema()``); a family without a
            tool wire format refuses them rather than send a request without.
        """
        path, body, _ = self.build(messages, output, tools)
        return path, body

    def build(self, messages: list, output: dict, tools: list | None = None):
        """`to_request`, and what building changed: ``(path, body, adaptations)``.

        The adaptations belong to this request; ``last_adaptations`` is only a
        mirror, unsafe on a model shared between threads.

        Parameters
        ----------
        messages : list
            Universal messages, placeholders already filled.
        output : dict
            The output spec.
        tools : list | None, optional
            Function-calling schemas, as ``Tool.schema()`` returns them.
        """
        # A mirror for a caller reading it afterwards; the library passes
        # `tools_sent` to `from_response` instead.
        self._gate_tools(tools)
        return self._built(messages, output, tools=tools or None)

    def _gate_tools(self, tools) -> None:
        """One tools gate for the buffered and the streamed path."""
        self._sent_tools = bool(tools)
        if tools and not self.client.supports_tools:
            raise ValueError(
                f"provider family {type(self.client).__name__!r} does not "
                f"support native tool calling yet; model {self.name!r} "
                "cannot be given tools"
            )

    def _finish(self, made: list, output: dict, body) -> None:
        """The tail of every request, buffered or streamed: `extra` on top,
        absent options noted, adaptations stored, announced, then enforced."""
        from ._adaptation import announce, enforce, note_absent
        asked = {**self._asked,
                 **{k: v for k, v in (output.get("format") or {}).items()
                    if k not in ("type", "schema", "name", "strict")}}
        if self.extra and isinstance(body, dict):
            # Last, and on top: `extra` overrides the library's own fields.
            # Multipart bodies are a list of pairs and are left alone.
            body.update(self.extra)
        if body is not None:
            note_absent(made, asked, body, self.name, provider=self._provider)
        # Stored before it is announced: under `-W error` the warning raises,
        # and the record has to be there to say what happened.
        self.last_adaptations = list(made)
        announce(made)
        # After the record is stored: a caller catching this needs
        # `last_adaptations` to say what happened.
        enforce(made, self.on_unsupported)

    def _canonical_format(self, output: dict) -> dict:
        """Resolve shorthand (``fidelity="high"`` → ``0.9``) before any range check
        sees it. Not an adaptation: it normalises the caller's own input."""
        from ._options import canonical_value

        fmt = output.get("format") or {}
        fixed, changed = dict(fmt), False
        for key, value in fmt.items():
            new_value, note = canonical_value(key, value)
            if note:
                fixed[key], changed = new_value, True
        return {**output, "format": fixed} if changed else output

    def _check_values(self, output: dict) -> dict:
        """Refuse a value this model does not take before the wire, rather
        than let the provider refuse it after the round trip."""
        from ._adaptation import Adaptation, ADAPTED, record
        from ._options import check_applies, check_value

        fmt = (output.get("format") or {})
        # An image asked for by modalities alone is an image, not "text".
        kind = "image" if "image" in (output.get("modalities") or []) else fmt.get("type", "text")
        fixed = None
        for name, asked, sent, note in getattr(self, "_clamped", ()):
            record(Adaptation(kind=ADAPTED, option=name, asked=asked, sent=sent,
                              model=self.name, why=note))
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
        """Build, then say what building changed: the one place every adaptation
        passes through, so it is where they are gathered and announced."""
        from ._adaptation import collect

        # Emptied first, so a build that raises leaves no stale record.
        self.last_adaptations = []
        with collect() as made:
            output = self._check_values(self._canonical_format(output))
            path, body = self.client.build_request(messages, output, self._params(), tools=tools)

        self._finish(made, output, body)
        return path, body, list(made)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(self, messages: list, output: dict,
               tools: "list | None" = None):
        """
        Yield the answer in pieces as the provider produces it.

        A provider that cannot stream returns the whole answer as one piece
        and records a ``declined`` adaptation (``on_unsupported="raise"``
        raises on it).

        Parameters
        ----------
        messages : list
            Universal messages, placeholders filled: ``{"role", "parts"}``.
        output : dict
            The output spec: ``{"modalities": [...], "format": {"type": …}}``.
        tools : list | None, optional
            Function-calling schemas, as ``Tool.schema()`` returns them.

        Returns
        -------
        StreamOutcome
            On exhaustion, for ``yield from``: result, usage, adaptations.
            ``last_stream_usage``/``last_stream_result`` mirror it.
        """
        from ._adaptation import Adaptation, DECLINED, collect

        self._gate_tools(tools)
        usage_seen: dict | None = None
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

        self._finish(made, output, body)

        if body is None:
            # `_built` reports on its own build only; the `declined` just made
            # is added back so the record keeps it.
            path, body, built = self._built(messages, output, tools=tools)
            notes = list(made) + built
            self.last_adaptations = notes
            import json as _json
            raw = self.client.send(path, body, self.client._auth_headers())
            response = _json.loads(raw)
            # Only the envelopes `extract_usage` reads: the whole response
            # would hold every image's base64 on a long-lived object.
            usage_seen = {k: response[k] for k in ("usage", "usageMetadata") if k in response}
            result = self.from_response(response, output, tools_sent=bool(tools))
            # Kept whole beside the text: a tool call rendered to text and
            # thrown away looks like a model that answered nothing.
            self.last_stream_usage, self.last_stream_result = usage_seen, result
            from ..clients._families._structured import hit_ceiling
            cut = hit_ceiling(response)
            if isinstance(result, str):
                yield result
            elif isinstance(result, dict):
                yield _json.dumps(result)
            else:
                # A tool call's own text, usually empty; never its repr, which
                # a caller printing the pieces would show to a user.
                text = getattr(result, "text", "") or ""
                if text:
                    yield text
            return StreamOutcome(result, usage_seen, notes, cut)

        from ..clients._families._structured import hit_ceiling
        assembly = _ToolCallAssembly()
        spoken: list = []
        cut = False
        for event in self.client._post_sse(
                path, body, self.client._auth_headers()):
            cut = cut or hit_ceiling(event)
            assembly.add(self.client.stream_tool_fragments(event))
            usage = self.client.stream_usage(event)
            if usage:
                # Merged, not replaced: Anthropic reports input tokens in
                # `message_start` and output in `message_delta`, and keeping
                # the last report drops the prompt from the bill, plausibly.
                usage_seen = _merge_usage(usage_seen, usage)
            piece = self.client.parse_stream_event(event, output)
            if piece:
                spoken.append(piece)
                yield piece

        # A turn that asked for tools is a decision, not prose, so it is left
        # as an object for the caller to act on. Any text the model sent
        # beside the calls has already been yielded and rides along in
        # `text`, so nothing is shown twice and nothing is lost.
        result = assembly.request("".join(spoken))
        self.last_stream_usage, self.last_stream_result = usage_seen, result
        return StreamOutcome(result, usage_seen, list(made), cut)

    def from_response(self, response: dict, output: dict, *,
                      tools_sent: bool | None = None) -> "str | dict":
        """
        Extract the clean result (str for text, dict for json / image) from a
        raw provider response, by delegating to the family client.

        Parameters
        ----------
        response : dict
            The provider's raw response body.
        output : dict
            The output spec the request was built with.
        tools_sent : bool | None, optional
            Whether that request offered tools — only then can a reply be a
            request to act. Passed by the caller that built the request; the
            model's own record of its last request is the fallback.
        """
        from ..clients._families._openai_compat import speaks_harmony
        if speaks_harmony(self.name):
            sent = self._sent_tools if tools_sent is None else tools_sent
            mode = "calls" if sent else "text"
            return self.client.parse_response(response, output, harmony=mode)
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
        if self.cache_control is not False:
            parts.append(f"cache_control={self.cache_control!r}")
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
        key = api_key or os.getenv(env_key) or ""
        # A keyless provider (a local server) is asked without one.
        if not key and PROVIDERS[provider]["provider"].get("auth") != "none":
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
