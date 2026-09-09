"""
models._options
===============

The universal vocabulary: what a caller may ask for, independent of provider.

This is the list the library's promise rests on — one way to say a thing, and
the library translating it to whatever each provider calls it. It is defined
here rather than inferred from the clients because a vocabulary inferred from
implementations is not a contract; it is a description of whatever happened to
be wired.

Which of these a provider has a control for at all is declared in its data
(``[provider.options] accepts``). That answers a question the request diff
cannot: whether an option that did not arrive is one this provider has no
control for, or a name the library has never heard of. Both used to be
reported identically — ``top_k`` on OpenAI and a mistyped ``temperatur``
both came back "this provider has no such control" — and they call for
opposite reactions from the reader.

Narrowing *within* a provider stays a runtime matter and is not declared:
OpenAI has a temperature control, and a reasoning model of theirs refuses it
on a particular call. That is an adaptation with a reason, reported when it
happens, not a capability that is absent.
"""

from __future__ import annotations

#: Every option a caller may set, with what it means and — where the provider
#: has no such control — what to reach for instead. The hint is the difference
#: between "you cannot have that" and "here is how you get the same effect".
UNIVERSAL_OPTIONS: dict = {
    "temperature": {
        "what": "how much randomness the model is allowed",
        "instead": None,
    },
    "top_p": {
        "what": "nucleus sampling: the probability mass to sample from",
        "instead": None,
    },
    "top_k": {
        "what": "sample only from the k most likely tokens",
        "instead": "top_p narrows the sampling in a way every provider takes",
    },
    "max_tokens": {
        "what": "ceiling on the answer's length",
        "instead": None,
    },
    "reasoning": {
        "what": "how much deliberation before answering: low / medium / high",
        "instead": None,
    },
    "cache_control": {
        "what": "cache the stable prefix so a repeated one is not re-billed",
        "instead": "this provider caches automatically, with nothing to turn "
                   "on — the saving is in the usage report, not in a setting",
    },
    "cache_ttl": {
        "what": "how long a cached prefix lives: '5m' or '1h'",
        "instead": None,
    },
}


#: Keys of ``output["format"]``. A second vocabulary because it is asked for
#: per call rather than per model — and an **open** one, unlike the options
#: above: providers legitimately extend it with names of their own (Recraft's
#: ``style``, Reve's ``version`` and ``postprocessing``), which their clients
#: read. So an unrecognised key here is reported and the request still goes,
#: while an unrecognised model option raises at construction. The difference
#: is not tidiness: a closed vocabulary can refuse a typo outright, an open
#: one cannot tell a typo from a provider's own word — the same Model renders one picture square
#: and the next one wide. Several of these names are still one provider's own
#: word (`output_format` and `background` are OpenAI's, `aspect_ratio` is
#: BFL's); choosing a neutral set is its own decision, and the matrix shows
#: which word each provider currently reads.
UNIVERSAL_FORMAT: dict = {
    "size": {
        "what": "pixel dimensions, as 'WIDTHxHEIGHT'",
        "instead": "this provider takes aspect_ratio and picks the pixels",
    },
    "aspect_ratio": {
        "what": "shape without committing to a pixel count, as '16:9'",
        "instead": "this provider takes size, so give the pixels directly",
    },
    "quality": {
        "what": "how much work to spend on the render",
        "instead": None,
    },
    "background": {
        "what": "transparent, opaque, or let the model decide",
        "instead": None,
    },
    "output_format": {
        "what": "the file format to return: png, jpeg, webp",
        "instead": None,
    },
    "compression": {
        "what": "compression level for jpeg and webp, 0-100",
        "instead": None,
    },
    "seed": {
        "what": "fix the randomness so the same prompt renders the same way",
        "instead": None,
    },
    "fidelity": {
        "what": "how much of the original survives an edit: 0 to 1, or "
                "'low'/'high'. One axis, and providers run it both ways — "
                "OpenAI asks how much to preserve, Recraft how much to "
                "change, so the number is inverted for the latter",
        "instead": None,
    },
}


#: Names that used to be ours, and what they are called now. Kept rather than
#: removed: renaming a public option breaks working code, and the library's
#: own rule is that a caller is told what changed, not left to find out.
#:
#: Only two names were replaced, and the measurement is why. Most of the
#: vocabulary turned out to be the field's shared lexicon rather than one
#: vendor's: `aspect_ratio` goes on the wire under that name at four
#: providers, `size` at three, `output_format` at two. `output_compression`
#: and `input_fidelity` were OpenAI's alone — the first carrying a redundant
#: "output" inside a dict already called `output["format"]`, the second
#: ambiguous about what the input is fidelity *to*.
ALIASES: dict = {
    "output_compression": "compression",
    # `input_fidelity` and `strength` named the same axis from opposite ends:
    # measured on Recraft 2026-09-09, strength=0.05 left a blue square blue
    # and strength=0.95 replaced it with the prompt, while OpenAI's
    # input_fidelity asks how much of the original to keep. One intent, one
    # name — the direction a caller cares about is what survives.
    "input_fidelity":     "fidelity",
    "reference_fidelity": "fidelity",
}


def canonical(option: str) -> str:
    """The current name for *option*, which may be its own."""
    return ALIASES.get(option, option)


#: Words a caller may write instead of a number. The canonical type is the
#: number — one type inside the library, whatever the caller wrote and
#: whatever each provider takes — and each client converts from it. Without a
#: single internal form, a range declared for one provider rejects the word
#: another provider requires.
SHORTHANDS: dict = {
    "fidelity": {"low": 0.3, "high": 0.9, "auto": 0.5},
}


def canonical_value(option: str, value):
    """Return ``(value, note)`` with a shorthand resolved to its number."""
    table = SHORTHANDS.get(option) or {}
    if isinstance(value, str) and value in table:
        return table[value], f"{value!r} is {table[value]} on this scale"
    return value, None


#: Which kinds of output each format key means anything for. Everything here
#: describes a picture, so the set is the same today; the field exists because
#: the rule it enforces is not about images but about *when to be strict*.
#:
#: The test is whether a request is wrong everywhere or only here. `top_k` on
#: Perplexity is wrong only there — it works on Anthropic and Google — and
#: raising would force a caller to branch per provider, which is the promise
#: this library exists to keep. `background` on a text request is wrong at
#: every provider that will ever exist, because text has no background. The
#: first is declined with a notice; the second is a mistake and stops here.
APPLIES_TO: dict = {key: frozenset({"image"}) for key in UNIVERSAL_FORMAT}


def check_applies(option: str, output_type: str) -> None:
    """Raise when *option* cannot mean anything for this kind of output."""
    kinds = APPLIES_TO.get(option)
    if kinds and output_type not in kinds:
        raise ValueError(
            f"{option!r} has no meaning for a {output_type!r} result; it "
            f"applies to: {', '.join(sorted(kinds))}. Nothing was sent — this "
            "is wrong for every provider, not just this one."
        )


def is_universal(option: str) -> bool:
    """True when *option* is a name the library knows, of either vocabulary."""
    return option in UNIVERSAL_OPTIONS or option in UNIVERSAL_FORMAT


def accepted_by(provider: str) -> "frozenset[str] | None":
    """
    The options *provider* has a control for, or ``None`` when it declares
    nothing.

    ``None`` is not "accepts nothing": a provider whose data predates the
    declaration should keep behaving as it did rather than have every option
    reported as unsupported. Absence of a claim is not a claim.
    """
    from ._data import PROVIDERS
    data = (PROVIDERS.get(provider) or {}).get("provider") or {}
    opts = data.get("options") or {}
    if "accepts" not in opts and "format_accepts" not in opts:
        return None
    return frozenset(opts.get("accepts") or ()) | frozenset(
        opts.get("format_accepts") or ())


def _renders_images(provider: str, model: str) -> bool:
    from ._data import PROVIDERS
    caps = ((PROVIDERS.get(provider) or {}).get("models", {})
            .get(model, {}).get("caps") or ())
    return any(c.split("-to-")[-1] == "image" for c in caps)


def why_absent(option: str, provider: str, model: str = "") -> str:
    """One sentence explaining why *option* left no trace on the request."""
    # A format key asked of a model that does not draw is a category
    # mistake, not a capability gap, and the two read alike unless said
    # apart: "openai has a background control but it did not reach this
    # request" is true of gpt-4o and tells the reader nothing they can act
    # on.
    if (option in UNIVERSAL_FORMAT and model
            and not _renders_images(provider, model)):
        return (f"{model} does not produce images, and {option!r} only means "
                "something for one that does")

    if not is_universal(option):
        known = ", ".join(sorted({**UNIVERSAL_OPTIONS, **UNIVERSAL_FORMAT}))
        return (f"{option!r} is not an option this library knows — a typo, or "
                f"a provider's own name for something. Known options: {known}")

    accepts = accepted_by(provider)
    if accepts is not None and option not in accepts:
        hint = {**UNIVERSAL_OPTIONS, **UNIVERSAL_FORMAT}[option]["instead"]
        return (f"the {provider} API has no {option}"
                + (f"; {hint}" if hint else "; the request was sent without it"))

    # Declared as available, yet nothing arrived: either this particular model
    # narrows it, or it is wired for some paths and not others. Say what is
    # certain rather than guessing which.
    return (f"{provider} has a {option} control, but it did not reach this "
            "request — this model or this call does not take it")


# ── What a value may be ──────────────────────────────────────────────────────
#
# Three different things can be wrong with an option, and they want three
# different answers:
#
#   the name is not one we know        → refuse where it was written
#   the provider has no such control   → decline, say so, name the way round
#   the value is not one this model
#   accepts                            → refuse before the wire, list what is
#
# The third is the one a provider answers for us otherwise, and badly: OpenAI
# says "Invalid value: 'ultra-max-supreme'" after the request has been paid
# for and the round trip spent. The allowed set belongs to the **model**, not
# the provider — `quality="xhigh"` is fine on gpt-image-2.5-flare and refused
# by gpt-image-1.5, measured 2026-09-09.


def allowed_values(option: str, provider: str, model: str) -> "list | None":
    """The values *model* accepts for *option*, or ``None`` when unconstrained.

    A per-model entry wins over the provider-wide one, because within one
    provider the newer models take values the older ones refuse.
    """
    from ._data import PROVIDERS
    data = PROVIDERS.get(provider) or {}
    per_model = (data.get("models", {}).get(model, {}).get("values") or {})
    if option in per_model:
        return list(per_model[option])
    provider_wide = ((data.get("provider") or {}).get("options") or {}
                     ).get("values") or {}
    return list(provider_wide[option]) if option in provider_wide else None


def check_value(option: str, value, provider: str, model: str):
    """
    Return ``(value_to_send, note)``.

    A value outside a declared set raises: substituting a guess for what the
    caller asked is worse than stopping, because "high" is not what somebody
    who wrote "xhigh" wanted. A number outside a declared range is clamped
    instead — there the intent is unambiguous, and *note* says it happened.
    """
    allowed = allowed_values(option, provider, model)
    if allowed is None:
        return value, None

    # A two-number range, written [min, max], means clamp rather than refuse.
    if (len(allowed) == 2 and all(isinstance(v, (int, float)) for v in allowed)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)):
        low, high = allowed
        if value < low or value > high:
            fixed = min(max(value, low), high)
            return fixed, (f"{value} is outside {low}-{high} for {model}; "
                           f"sent {fixed}")
        return value, None

    if value not in allowed:
        raise ValueError(
            f"{model} does not accept {option}={value!r}. "
            f"Allowed: {', '.join(map(repr, allowed))}."
        )
    return value, None


# ── Bringing a value to the provider's own scale ─────────────────────────────
#
# A universal name is only half the promise. `reasoning="medium"` has to
# become 10000 budget tokens on Anthropic, 8192 on Google, the string
# "medium" on OpenAI, `true` on Qwen and a different model on DeepSeek — one
# word, seven shapes, and the caller writes it once.
#
# That mapping existed for `reasoning` alone, as `[provider.reasoning_map]`.
# It is the same need everywhere: Reve expresses render quality as
# `test_time_scaling`, a number from 1 to 15, and a transparent background as
# an entry in a `postprocessing` list. Declaring those in data rather than
# writing them into a client keeps the translation where it can be read.


def value_map(option: str, provider: str) -> "dict | None":
    """
    How *provider* spells each universal value of *option*.

    Declared as ``[provider.options.map.<option>]``. ``reasoning`` also reads
    the older ``[provider.reasoning_map]``, which predates this and stays
    valid — a provider file written before the generalisation must keep
    working unchanged.
    """
    from ._data import PROVIDERS
    data = (PROVIDERS.get(provider) or {}).get("provider") or {}
    maps = (data.get("options") or {}).get("map") or {}
    if option in maps:
        return dict(maps[option])
    if option == "reasoning" and data.get("reasoning_map"):
        return dict(data["reasoning_map"])
    return None


def to_provider_value(option: str, value, provider: str):
    """
    Return ``(sent, note)`` — *value* in this provider's own scale.

    ``note`` is a sentence for the adaptation record when the value changed
    shape, and ``None`` when it went out as written. An unmapped value passes
    through: the map names the levels a caller may ask for, not everything a
    provider will take.
    """
    table = value_map(option, provider)
    if not table or value not in table:
        return value, None
    sent = table[value]
    if sent == value:
        return value, None
    return sent, (f"{value!r} is {sent!r} on {provider} — the same level in "
                  "this provider's own scale")
