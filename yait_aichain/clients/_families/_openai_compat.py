"""
clients._families._openai_compat
================================

Wire-format helpers for the OpenAI Chat Completions / Responses / Images
APIs — shared by every OpenAI-compatible provider (openai, xai, perplexity,
kimi, deepseek, qwen).  Pure functions: universal format ↔ provider JSON.
"""

from __future__ import annotations

from ..._multipart import Multipart

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...models._calls import ToolCallRequest

from ._shape import ratio_of, size_from_ratio
import base64
import json
import re


def _drop_rejected(model, fmt: dict) -> dict:
    """
    Remove parameters the model's API refuses, warning once for each.

    The list lives in the provider data (``rejects``) rather than in a set of
    prefixes here, because it was established by asking the API and not by
    reading the guide: the guide names only ``gpt-image-2``, while both GPT
    Image 2.5 models and ``gpt-image-1-mini`` refuse ``input_fidelity`` too.

    Dropping rather than forwarding turns a hard 400 into a working call. It
    is not silent: a parameter the caller set and did not get is worth a line,
    since for some models the setting is a no-op and for others it is a
    request the API will not honour.
    """
    rejects = getattr(model, "_REJECTS", ())
    if not rejects:
        return fmt
    dropped = [k for k in rejects if fmt.get(k) is not None]
    if not dropped:
        return fmt
    # Reported through the adaptation channel, not warned here: a warning of
    # its own would announce the same dropped parameter twice, and only the
    # channel reaches the run record.
    from ...models._adaptation import Adaptation, DECLINED, record
    for key in dropped:
        record(Adaptation(
            kind=DECLINED, option=key, asked=fmt[key], sent=None,
            model=model.name,
            why=f"{model.name} does not accept this parameter, so it was "
                "dropped and the request sent; the model applies its own "
                "handling for the setting"))
    return {k: v for k, v in fmt.items() if k not in dropped}


def _part_to_openai(part: dict) -> dict | None:
    """
    Convert one universal part dict to an OpenAI content item.

    Returns ``None`` for part types that OpenAI does not support (video, URL
    audio) so callers can filter them out.
    """
    ptype = part["type"]

    if ptype == "text":
        return {"type": "text", "text": part["text"]}

    if ptype == "image":
        src    = part["source"]
        detail = part.get("meta", {}).get("detail", "auto")
        kind   = src["kind"]
        if kind == "url":
            return {
                "type":      "image_url",
                "image_url": {"url": src["url"], "detail": detail},
            }
        if kind in ("base64", "file"):
            mime = src.get("mime", "image/png")
            return {
                "type":      "image_url",
                "image_url": {
                    "url":    f"data:{mime};base64,{src['data']}",
                    "detail": detail,
                },
            }

    if ptype == "audio":
        src  = part["source"]
        kind = src["kind"]
        if kind == "base64":
            mime = src.get("mime", "audio/wav")
            fmt  = mime.split("/")[-1]   # "wav", "mp3", …
            return {
                "type":        "input_audio",
                "input_audio": {"data": src["data"], "format": fmt},
            }
        # URL audio is not supported by OpenAI input_audio
        return None

    # ptype == "video" — not supported by OpenAI chat completions
    return None


def _part_to_responses(part: dict, role: str = "user") -> dict | None:
    """
    Convert one universal part to a **Responses API** content item.

    The two OpenAI wire formats name their content types differently, and the
    difference is not cosmetic: chat completions take ``text``/``image_url``
    with the image as an object, the Responses API takes
    ``input_text``/``input_image`` with the image URL as a bare string, and
    refuses the other spelling outright —

        Invalid value: 'text'. Supported values are: 'input_text',
        'input_image', 'input_audio', 'output_text', 'refusal', 'input_file'

    Reusing the chat encoder here meant vision never worked on any model that
    routes through the Responses API. It stayed invisible because a message
    with exactly one text part was collapsed to a plain string, which is the
    overwhelmingly common case; the moment a second part appeared — an image,
    or a tool handing one back — the list went out in the wrong names.

    An assistant turn spells its text ``output_text``: the same content in the
    other direction has its own name here.
    """
    ptype = part["type"]

    if ptype == "text":
        kind = "output_text" if role == "assistant" else "input_text"
        return {"type": kind, "text": part["text"]}

    if ptype == "image":
        src    = part["source"]
        detail = part.get("meta", {}).get("detail", "auto")
        kind   = src["kind"]
        if kind == "url":
            url = src["url"]
        elif kind in ("base64", "file"):
            url = f"data:{src.get('mime', 'image/png')};base64,{src['data']}"
        else:
            return None
        # A string, not an object — the object form is the chat spelling.
        return {"type": "input_image", "image_url": url, "detail": detail}

    if ptype == "document":
        src = part["source"]
        if src.get("kind") in ("base64", "file"):
            mime = src.get("mime", "application/pdf")
            return {"type": "input_file",
                    "filename": src.get("filename", "document"),
                    "file_data": f"data:{mime};base64,{src['data']}"}
        return None

    if ptype == "audio":
        src = part["source"]
        if src.get("kind") == "base64":
            mime = src.get("mime", "audio/wav")
            return {"type": "input_audio",
                    "input_audio": {"data": src["data"],
                                    "format": mime.split("/")[-1]}}
        return None

    return None                                   # video: unsupported


def _text_of(msg: dict) -> str:
    """The concatenated text parts of a universal message ("" when none)."""
    return "\n".join(p.get("text", "") for p in msg.get("parts") or []
                     if p.get("type") == "text")


def _build_openai_compat_request(
    model,
    messages: list,
    output:   dict,
    path:     str,
    max_tokens_field: str = "max_completion_tokens",
    tools:    list | None = None,
) -> tuple[str, dict]:
    """
    Build an OpenAI-compatible ``(path, body)`` pair.

    Parameters
    ----------
    model    : Provider model instance (OpenAI / xAI / Perplexity / …).
    messages : Substituted universal messages list.
    output   : Universal output spec.
    path     : Provider-specific endpoint path.
    max_tokens_field : Request key for the output-token limit
                       (``"max_completion_tokens"`` default, ``"max_tokens"``
                       for Kimi/DeepSeek).
    """
    openai_messages: list[dict] = []
    # Media the tools of one turn produced, sent in one user message after the
    # last of that turn's results: the results of parallel calls must follow
    # their assistant turn back to back, and a user message between two of
    # them is a 400.
    held: list = []

    def release():
        if held:
            openai_messages.append({"role": "user", "content": [_part_to_openai(p) for p in held]})
            held.clear()

    for msg in messages:
        role = msg["role"]

        # A tool turn is one call's result — chat format keys it by id, and
        # that content must be text. Media a tool produced is moved into a
        # user message after the turn's results, so the model sees it now.
        if role == "tool":
            from ...models._calls import split_media_result
            msg, follow_up = split_media_result(msg)
            openai_messages.append({
                "role":         "tool",
                "tool_call_id": msg.get("call_id", ""),
                "content":      _text_of(msg),
            })
            if follow_up:
                held.extend(follow_up["parts"])
            continue
        release()

        # An assistant turn that asked for calls. ``arguments`` goes out as a
        # JSON string — that is the chat wire format, not a convenience.
        if role == "assistant" and msg.get("tool_calls"):
            entry: dict = {
                "role": "assistant",
                "content": _text_of(msg) or None,
                "tool_calls": [
                    {"id": c.get("id", ""), "type": "function",
                     "function": {"name": c["name"],
                                  "arguments": json.dumps(c.get("arguments", {}),
                                                          ensure_ascii=False)}}
                    for c in msg["tool_calls"]
                ],
            }
            openai_messages.append(entry)
            continue

        items = [_part_to_openai(p) for p in msg["parts"]]
        items = [it for it in items if it is not None]
        if not items:
            continue
        # Use a plain string when the message is a single text block
        if len(items) == 1 and items[0]["type"] == "text":
            openai_messages.append({"role": role, "content": items[0]["text"]})
        else:
            openai_messages.append({"role": role, "content": items})

    release()
    # A system message means the same thing wherever it sits: it applies to
    # the whole conversation. That is not a preference — it is the only
    # meaning every provider can express. Anthropic, Google and the Responses
    # API each carry the system prompt in a field of their own, so a system
    # message written in the middle of a conversation is hoisted out of the
    # sequence and becomes global; this family left it in place, where it
    # reads as an instruction that begins to apply at that point. The same
    # conversation therefore meant two different things depending on the
    # model, with nothing said — measured 2026-09-12 across every provider.
    #
    # Hoisted here too, so the meaning is one. Reported, because for this
    # family it is a real change to what the model sees.
    from ...models._adaptation import Adaptation, ADAPTED, record
    if any(m["role"] == "system" for m in openai_messages[1:]):
        hoisted = [m for m in openai_messages if m["role"] == "system"]
        rest    = [m for m in openai_messages if m["role"] != "system"]
        joined  = "\n".join(
            m["content"] if isinstance(m["content"], str)
            else " ".join(i.get("text", "") for i in m["content"])
            for m in hoisted)
        openai_messages = [{"role": "system", "content": joined}] + rest
        record(Adaptation(
            kind=ADAPTED, option="system message position",
            asked="mid-conversation", sent="hoisted to the front",
            model=model.name,
            why="a system message applies to the whole conversation on every "
                "provider; three of them cannot express one that begins "
                "mid-way, so it is made global here as well"))

    body: dict = {
        "model":          model.name,
        "messages":       openai_messages,
        max_tokens_field: model.max_tokens,
        "temperature":    model.temperature,
    }

    if tools:
        # Canonical input is what Tool.schema() returns —
        # {"type": "function", "function": {...}} — passed through verbatim.
        body["tools"] = list(tools)

    if model.top_p is not None:
        body["top_p"] = model.top_p

    # Output format
    fmt   = output.get("format", {})
    ftype = fmt.get("type", "text")
    if ftype == "json":
        body["response_format"] = {"type": "json_object"}
    elif ftype == "json_schema":
        body["response_format"] = {"type": "json_schema", "json_schema": _json_schema(fmt)}

    return path, body


def _json_schema(fmt: dict) -> dict:
    """The schema block both OpenAI wires take: name, schema, strict."""
    strict = fmt.get("strict", True)
    return {"name": fmt.get("name", "response"),
            "schema": _sanitize_openai_schema(fmt["schema"]) if strict else fmt["schema"],
            "strict": strict}


class SchemaNotExpressible(ValueError):
    """A schema that strict mode cannot represent, named at the point of use.

    Raised instead of letting the provider answer, because the provider's
    message points at the wire format rather than at the schema the caller
    wrote.
    """


def _sanitize_openai_schema(schema: object, path: str = "$") -> object:
    """
    Convert a JSON Schema dict to the form OpenAI's ``strict`` mode accepts.

    The mirror of ``_sanitize_google_schema``. Google forbids
    ``additionalProperties`` and union types; strict mode requires the exact
    opposite — every object closed with ``additionalProperties: false`` and
    every property listed in ``required``. A schema written for one family
    therefore fails on the other, and the caller ends up deriving a portable
    form by hand. The library already normalises for one family; leaving the
    other alone is what makes swapping providers break working code.

    Fixed recursively:

    * every object with ``properties`` gets ``additionalProperties: false``;
    * every property is moved into ``required`` — an optional one keeps its
      optionality as ``"type": ["X", "null"]``, which is how strict mode
      spells it;
    * ``$defs``/``definitions``, ``items`` and ``anyOf``/``oneOf``/``allOf``
      are followed.

    One shape has no strict equivalent and is reported rather than mangled:
    a map with arbitrary keys (``additionalProperties`` given a schema, no
    ``properties``). Strict mode cannot express it at all; the value has to
    be modelled as an array of pairs.
    """
    if not isinstance(schema, dict):
        return schema

    ap = schema.get("additionalProperties")
    if isinstance(ap, dict) and not schema.get("properties"):
        raise SchemaNotExpressible(
            f"{path}: a map with arbitrary keys cannot be expressed in "
            "OpenAI strict mode. Model it as an array of {key, value} "
            "objects, or pass strict=False."
        )

    result: dict = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            result[key] = {k: _sanitize_openai_schema(v, f"{path}.{k}")
                           for k, v in value.items()}
        elif key == "items":
            result[key] = _sanitize_openai_schema(value, f"{path}[]")
        elif key in ("$defs", "definitions") and isinstance(value, dict):
            result[key] = {k: _sanitize_openai_schema(v, f"{path}.{k}")
                           for k, v in value.items()}
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            result[key] = [_sanitize_openai_schema(v, f"{path}<{i}>")
                           for i, v in enumerate(value)]
        else:
            result[key] = value

    props = result.get("properties")
    if isinstance(props, dict):
        result["additionalProperties"] = False
        was_required = set(result.get("required") or [])
        for name, sub in props.items():
            if name in was_required or not isinstance(sub, dict):
                continue
            t = sub.get("type")
            if isinstance(t, str) and t != "null":
                sub["type"] = [t, "null"]        # optional, strict spelling
            elif isinstance(t, list) and "null" not in t:
                sub["type"] = [*t, "null"]
        result["required"] = list(props.keys())
    return result


def _tool_call_request(triples, text: str = "") -> ToolCallRequest:
    """
    Build a :class:`ToolCallRequest` from ``(name, arguments, id)`` triples.

    ``arguments`` arrives as a JSON **string** on the wire; a dict is accepted
    too (some compatible servers send one), and unparseable arguments become an
    empty dict rather than an exception — the executor's schema check is the
    right place to complain, with the tool named.
    """
    from ...models._calls import ToolCall, ToolCallRequest
    calls = []
    for name, args, call_id in triples:
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        calls.append(ToolCall(id=call_id or "", name=name,
                              arguments=args if isinstance(args, dict) else {}))
    return ToolCallRequest(calls=tuple(calls), text=text)


#: Marker that a reply is written in OpenAI's *harmony* format rather than
#: plain text. gpt-oss models emit it, and servers that do not implement the
#: format (mlx_lm, llama.cpp, plain vLLM without a tool-call parser) pass it
#: through verbatim — the tool call arrives as prose in ``content`` and is
#: lost, which is the exact silent failure the native channel exists to remove.
_HARMONY_MARK = "<|channel|>"

#: Where a harmony segment's body ends.
_HARMONY_STOPS = ("<|end|>", "<|call|>", "<|return|>", "<|start|>")

#: The recipient of a tool call inside a commentary channel:
#: ``to=functions.get_weather``.
_HARMONY_RECIPIENT = re.compile(r"to=functions\.(?P<name>[\w-]+)")


def _is_harmony(text: str) -> bool:
    return isinstance(text, str) and _HARMONY_MARK in text


def speaks_harmony(model_name: str) -> bool:
    """Whether a model writes harmony: the gpt-oss family, by name — the one
    family that does, and the only one whose replies are parsed for it. Any
    other model that quotes a channel marker it read in a file or a page is
    quoting, and its text stays text."""
    return "gpt-oss" in (model_name or "").lower()


def _parse_harmony(text: str, *, calls_allowed: bool = True):
    """
    Split a harmony reply into its channels and return either a
    :class:`ToolCallRequest` (when it asks to act) or the user-facing text.

    Three channels matter. ``analysis`` is the model thinking aloud and is
    **dropped** — surfacing it as the answer is how a reasoning model ends up
    "replying" with its own notes. ``commentary`` carries tool calls, one per
    segment, addressed with ``to=functions.NAME``. ``final`` is what the user
    should see.

    *calls_allowed* is False when no tools were sent: a call then is text the
    model wrote, never a request to act, and the reply is its final text.

    A truncated reply (max_tokens cut mid-call) yields a segment whose body is
    not valid JSON; those become empty arguments rather than an exception, so
    the executor can complain by name instead of the turn dying here. One pass
    over the text: each segment is cut at its first stop token, where a
    backtracking pattern took seconds on a long reply.
    """
    calls, final_parts = [], []
    for segment in text.split(_HARMONY_MARK)[1:]:
        head, sep, rest = segment.partition("<|message|>")
        if not sep:
            continue
        channel = re.match(r"\w*", head).group(0)
        cut = min((i for i in (rest.find(t) for t in _HARMONY_STOPS) if i >= 0), default=len(rest))
        body = rest[:cut].strip()
        if channel == "analysis":
            continue                       # deliberation, not an answer
        if channel == "commentary":
            recipient = _HARMONY_RECIPIENT.search(head)
            if recipient:
                if calls_allowed:
                    try:
                        args = json.loads(body) if body else {}
                    except json.JSONDecodeError:
                        args = {}
                    calls.append((recipient.group("name"),
                                  args if isinstance(args, dict) else {},
                                  f"call_{len(calls)}"))
            elif body:
                final_parts.append(body)   # commentary addressed to nobody
        elif channel == "final":
            final_parts.append(body)

    text_out = "\n".join(p for p in final_parts if p)
    if calls:
        # this wire carries no ids; one per call keeps two calls to one tool apart
        return _tool_call_request(iter(calls), text=text_out)
    return text_out


def _parse_openai_compat_response(response: dict, output: dict, *,
                                  strip_fences: bool = False,
                                  harmony: str | None = None) -> str | dict:
    """
    Extract the clean result from an OpenAI-compatible chat completion
    response.

    Shared by the openai, xai and perplexity providers (and, via their own
    request builders, the kimi/deepseek/qwen providers).

    Raises
    ------
    ValueError
        With a descriptive message when the response carries no usable
        content: empty ``choices`` (blocked/failed upstream), an explicit
        model ``refusal``, or invalid/truncated JSON in a JSON mode.
    """
    choices = response.get("choices") or []
    if not choices:
        detail = (response.get("error") or {}).get("message")
        raise ValueError(
            "Provider response contains no choices — the request was "
            "blocked or failed upstream"
            + (f": {detail}" if detail else ".")
        )

    choice  = choices[0] or {}
    message = choice.get("message") or {}

    refusal = message.get("refusal")
    if refusal:
        raise ValueError(f"Model refused to answer: {refusal}")

    # The model asked to act. This outranks any accompanying text: a reply
    # that both says something and calls a tool is a call, with the text
    # carried alongside — collapsing it to the text would silently drop the
    # action, which is the exact failure this channel exists to remove.
    from ._structured import refuse_cut_off
    finish = choice.get("finish_reason") or ""
    spent = (response.get("usage") or {}).get("completion_tokens")
    refuse_cut_off(finish, spent, calls=bool(message.get("tool_calls")),
                   text=content_text(message.get("content")))
    if message.get("tool_calls"):
        return _tool_call_request(
            ((c.get("function", {}).get("name", ""),
              c.get("function", {}).get("arguments"),
              c.get("id", ""))
             for c in message["tool_calls"]),
            text=content_text(message.get("content")),
        )

    text  = content_text(message.get("content"))

    # A server that does not implement harmony passes it through as prose.
    # Parsing it here is what turns a lost tool call into a real one — and
    # what keeps a reasoning model's notes out of the user-facing answer.
    # Only for a model that writes it (*harmony* is "calls" when tools were
    # sent, "text" when not): any other model's text that quotes a channel
    # marker — read from a file, a page — was being executed as a tool call.
    if harmony and _is_harmony(text):
        parsed = _parse_harmony(text, calls_allowed=harmony == "calls")
        if not isinstance(parsed, str):
            return parsed
        text = parsed

    ftype = output.get("format", {}).get("type", "text")
    if ftype in ("json", "json_schema"):
        from ._structured import parse_structured
        usage = response.get("usage") or {}
        return parse_structured(
            text, output.get("format", {}).get("schema"),
            finish_reason=choice.get("finish_reason") or "",
            output_tokens=usage.get("completion_tokens"),
            strip_fences=strip_fences,
        )
    return text


def content_text(content) -> str:
    """The answer's text from a message's ``content``.

    Most servers send a string. Mistral's reasoning models send a list of
    parts — ``{"type": "thinking", ...}`` then ``{"type": "text", "text": ...}``
    — and only the text parts are the answer: the thinking is the model's
    notes, kept out of what the caller reads, as harmony's are.
    """
    if isinstance(content, str):
        return content
    if not content:
        return ""
    return "".join(part.get("text", "") for part in content
                   if isinstance(part, dict) and part.get("type", "text") == "text"
                   and isinstance(part.get("text"), str))


# ---------------------------------------------------------------------------
# Image generation helpers  (for dall-e-* and gpt-image-* models)
# ---------------------------------------------------------------------------

_GPT_IMAGE = ("gpt-image-", "chatgpt-image-")
_OPENAI_IMAGE_MODEL_PREFIXES = ("dall-e-", *_GPT_IMAGE)


def _is_openai_image_model(name: str) -> bool:
    """Return True when *name* identifies an OpenAI image-generation model."""
    return name.startswith(_OPENAI_IMAGE_MODEL_PREFIXES)


# o-series reasoning models reject the sampling parameters that ordinary
# GPT models accept.
_OSERIES_FAMILIES = ("o1", "o3", "o4")


def _is_o_series_model(name: str) -> bool:
    """True for o-series reasoning models (o1, o1-mini, o3, o4-mini, …)."""
    return any(
        name == fam or name.startswith(f"{fam}-") for fam in _OSERIES_FAMILIES
    )


def _build_image_generations_request(
    model,
    messages: list,
    output:   dict,
    path:     str = "/v1/images/generations",
) -> tuple[str, dict]:
    """
    Build an OpenAI ``/v1/images/generations`` ``(path, body)`` pair.

    Shared by the openai and xai providers.

    The text prompt is extracted from the last user message.  Optional
    ``size`` and ``quality`` values are read from ``output["format"]``.

    ``response_format`` handling
    ----------------------------
    All models that accept the parameter receive ``"b64_json"`` so that
    the output is always an inline base64 string — consistent across every
    provider (OpenAI DALL-E, xAI Grok-Imagine, Google).

    ``gpt-image-*`` models are the single exception: they always return
    ``b64_json`` natively and explicitly reject the parameter, so it is
    omitted for them.

    +-----------------------+--------------------------------------+
    | Model prefix          | response_format sent?                |
    +=======================+======================================+
    | ``dall-e-*``          | Yes → ``"b64_json"``                 |
    | ``gpt-image-*``       | No  → native ``b64_json``            |
    | ``grok-imagine-*``    | Yes → ``"b64_json"``                 |
    +-----------------------+--------------------------------------+
    """
    prompt = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            texts = [p["text"] for p in msg["parts"] if p["type"] == "text"]
            if texts:
                prompt = "\n".join(texts)
                break

    fmt: dict  = _drop_rejected(model, output.get("format", {}))
    body: dict = {"model": model.name, "prompt": prompt, "n": 1}

    # This API takes pixels; a ratio has no field. Every other image provider
    # in the library now understands both, so the ratio is turned into a size
    # rather than dropped. GPT Image 2.5 accepts any WIDTHxHEIGHT whose edges
    # are multiples of 16 and whose ratio is between 1:3 and 3:1, which a
    # 1024 short edge satisfies for every shape worth asking for.
    if fmt.get("aspect_ratio") and not fmt.get("size"):
        fmt = {**fmt, "size": size_from_ratio(fmt["aspect_ratio"], multiple=16)}
        if fmt["size"]:
            from ...models._adaptation import Adaptation, ADAPTED, record
            record(Adaptation(
                kind=ADAPTED, option="aspect_ratio",
                asked=output["format"]["aspect_ratio"],
                sent=f"size={fmt['size']}", model=model.name,
                why="this provider takes pixels, not a ratio; the shape was "
                    "kept at a 1024 short edge"))

    if fmt.get("size"):
        body["size"] = fmt["size"]
    if fmt.get("quality"):
        body["quality"] = fmt["quality"]
    # gpt-image-* accepts a transparent background and an explicit file format
    # (``png``/``webp`` keep the alpha channel; ``jpeg`` would flatten it).
    if fmt.get("background"):
        body["background"] = fmt["background"]
    if fmt.get("output_format"):
        body["output_format"] = fmt["output_format"]
    # Only meaningful with jpeg/webp; 0 is a legal value, so the presence of
    # the key decides, not its truth.
    if fmt.get("compression") is not None:
        body["output_compression"] = fmt["compression"]

    # Request base64 output on every model that accepts the parameter.
    # gpt-image-* / chatgpt-image-* always return b64_json natively and reject
    # the param — omit it only for those families (dall-e-* still needs it).
    if not model.name.startswith(_GPT_IMAGE):
        body["response_format"] = "b64_json"

    return path, body


def _detect_image_mime(b64_data: str | None) -> str:
    """
    Return the MIME type of a base64-encoded image by inspecting its magic bytes.

    Handles JPEG (``FF D8 FF``), PNG (``89 50 4E 47``), WebP (``RIFF…WEBP``),
    and GIF (``GIF8``).  Falls back to ``"image/png"`` for unknown formats.
    """
    if not b64_data:
        return "image/png"
    try:
        import base64 as _b64
        # Decode just enough bytes to read the magic header (12 bytes → 16 b64 chars)
        header = _b64.b64decode(b64_data[:16] + "==")
        if header[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if header[:4] == b"\x89PNG":
            return "image/png"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image/webp"
        if header[:4] == b"GIF8":
            return "image/gif"
        # Vector output (e.g. Recraft's *_vector models) returns raw SVG markup.
        if header[:4] == b"<svg" or header[:5] == b"<?xml":
            return "image/svg+xml"
    except Exception:
        pass
    return "image/png"


def _parse_image_generations_response(response: dict) -> dict:
    """
    Extract the image result from an OpenAI ``/v1/images/generations``
    response.

    Shared by the openai and xai providers.

    Returns a dict with keys ``url``, ``base64``, ``mime_type``, and
    ``revised_prompt`` — consistent with the Google image response format.

    ``mime_type`` is detected from the image's magic bytes so it is always
    accurate, regardless of which provider generated the image.

    Raises
    ------
    ValueError
        When ``data`` is empty (e.g. the prompt was rejected by moderation).
    """
    data = response.get("data") or []
    if not data:
        detail = (response.get("error") or {}).get("message")
        raise ValueError(
            "Provider returned no image data — the prompt may have been "
            "rejected by moderation"
            + (f": {detail}" if detail else ".")
        )
    item = data[0]
    b64  = item.get("b64_json")
    return {
        "url":            item.get("url"),
        "base64":         b64,
        "mime_type":      item.get("mime_type") or _detect_image_mime(b64),
        "revised_prompt": item.get("revised_prompt", ""),
    }


# ---------------------------------------------------------------------------
# Image editing helpers  (image-to-image — input image(s) + prompt → image)
# ---------------------------------------------------------------------------

def _messages_have_image(messages: list) -> bool:
    """True when any message carries an image input part (→ an edit, not a gen)."""
    return any(
        p.get("type") == "image"
        for msg in messages
        for p in msg.get("parts", [])
    )


def _image_sources(messages: list) -> list:
    """Every image part's ``source`` dict, in message/part order."""
    return [
        p["source"]
        for msg in messages
        for p in msg.get("parts", [])
        if p.get("type") == "image" and isinstance(p.get("source"), dict)
    ]


def _image_source_to_data_uri(src: dict) -> str:
    """A URL source → its URL; a base64/file source → a ``data:`` URI."""
    if src.get("kind") == "url":
        return src["url"]
    mime = src.get("mime", "image/png")
    return f"data:{mime};base64,{src['data']}"


def _last_user_text(messages: list) -> str:
    """The last user message's concatenated text — the edit instruction."""
    for msg in reversed(messages):
        if msg["role"] == "user":
            texts = [p["text"] for p in msg["parts"] if p["type"] == "text"]
            if texts:
                return "\n".join(texts)
    return ""


def _is_openai_editable_image_model(name: str) -> bool:
    """True for OpenAI image models that accept the ``/v1/images/edits`` endpoint."""
    return name.startswith(_GPT_IMAGE) or name == "dall-e-2"


def _build_image_edits_request(
    model,
    messages: list,
    output:   dict,
    path:     str = "/v1/images/edits",
) -> tuple[str, dict]:
    """
    Build an OpenAI ``/v1/images/edits`` **multipart** ``(path, body)`` pair.

    The body is a sentinel the client's ``send()`` seam turns into a
    ``multipart/form-data`` POST: a :class:`Multipart` of the fields —
    a list of ``(name, value)`` tuples so multiple input images can share the
    ``image[]`` key. ``gpt-image-*`` accepts multiple references via ``image[]``;
    ``dall-e-2`` takes a single ``image``. ``response_format`` is omitted
    (gpt-image returns b64 natively).
    """
    sources = _image_sources(messages)
    if not sources:
        raise ValueError("image edit requires at least one input image part")

    field_name = "image[]" if model.name.startswith(_GPT_IMAGE) else "image"

    fields: list = [("model", model.name), ("prompt", _last_user_text(messages))]
    fmt = _drop_rejected(model, output.get("format", {}))
    # Our name on the left, the provider's on the wire.
    for ours, theirs in (("size", "size"), ("quality", "quality"),
                         ("background", "background"),
                         ("output_format", "output_format"),
                         ("fidelity", "input_fidelity")):
        if fmt.get(ours) is None:
            continue
        value = fmt[ours]
        if ours == "fidelity" and isinstance(value, (int, float)):
            from ...models._adaptation import Adaptation, ADAPTED, record
            # This API takes two words where the vocabulary takes a number.
            value = "high" if value >= 0.5 else "low"
            record(Adaptation(
                kind=ADAPTED, option="fidelity", asked=fmt[ours], sent=value,
                model=model.name,
                why="this API has two levels, not a scale"))
        fields.append((theirs, str(value)))
    if fmt.get("compression") is not None:
        fields.append(("output_compression", str(fmt["compression"])))

    for i, src in enumerate(sources):
        if src.get("kind") == "url":
            raise ValueError(
                "OpenAI image edits need binary image data, not a URL — "
                "pass a base64 or file source."
            )
        raw  = base64.b64decode(src["data"])
        mime = src.get("mime", "image/png")
        ext  = mime.split("/")[-1]
        fields.append((field_name, (f"image_{i}.{ext}", raw, mime)))

    return path, Multipart(fields)


def _build_xai_image_edit_request(
    model,
    messages: list,
    output:   dict,
    path:     str = "/v1/images/edits",
) -> tuple[str, dict]:
    """
    Build an xAI ``/v1/images/edits`` **JSON** ``(path, body)`` pair.

    Confirmed by live probe: xAI edits is JSON (not multipart); the source image
    rides a nested ``image: {"url": <data-uri|url>, "type": "image_url"}`` and
    ``response_format:"b64_json"`` yields base64 directly — so
    ``_parse_image_generations_response`` consumes it unchanged.
    """
    sources = _image_sources(messages)
    if not sources:
        raise ValueError("image edit requires at least one input image part")
    imgs = [{"url": _image_source_to_data_uri(s), "type": "image_url"}
            for s in sources[:3]]                      # xAI Imagine: up to 3 source images
    body = {"model": model.name, "prompt": _last_user_text(messages),
            "response_format": "b64_json"}
    # Shape control works on this path too — measured 2026-09-09, an edit with
    # aspect_ratio "16:9" came back 1280x720 — and was read by neither the
    # generation nor the edit branch until now.
    fmt = output.get("format", {})
    _ratio = fmt.get("aspect_ratio") or (
        ratio_of(fmt["size"]) if fmt.get("size") else None)
    if _ratio:
        from ...models._adaptation import Adaptation, ADAPTED, TRANSLATED, record
        body["aspect_ratio"] = _ratio
        record(Adaptation(
            kind=TRANSLATED if fmt.get("aspect_ratio") else ADAPTED,
            option="aspect_ratio" if fmt.get("aspect_ratio") else "size",
            asked=fmt.get("aspect_ratio") or fmt.get("size"),
            sent=f"aspect_ratio={_ratio}", model=model.name,
            why="this API takes a ratio, not a pixel size"))
    # Single image → `image`; multiple → `images` array (both confirmed by live probe).
    if len(imgs) == 1:
        body["image"] = imgs[0]
    else:
        body["images"] = imgs
    return path, body


# ---------------------------------------------------------------------------
# Responses API helpers  (every GPT model newer than gpt-4)
# ---------------------------------------------------------------------------

#: GPT models that stay on Chat Completions. Everything else named ``gpt-<N>``
#: goes to the Responses API.
#:
#: The list is of the exceptions, not of the Responses models, and that is the
#: whole change. Until 2026-10-01 it was the other way round —
#: ``_RESPONSES_API_PREFIXES = ("gpt-5",)`` — so every new generation arrived on
#: the old endpoint by default, and gpt-6 did: Chat Completions refuses function
#: tools together with reasoning on it, HTTP 400, and the agent could not call a
#: tool at all until someone noticed. New models ship on the new endpoint, so
#: the new endpoint is the default and the old one is what has to be named.
#: The gpt-4 and gpt-3.5 generations are a closed set; nothing will be added
#: to them.
_CHAT_COMPLETIONS_MODELS = ("gpt-4", "gpt-3.5")

#: ``gpt-<digit>`` — the generation number names a chat model. It deliberately
#: does not match ``gpt-image-*``, ``gpt-oss-*`` or ``gpt-realtime-*``, which
#: are other APIs that only share the prefix.
_GPT_GENERATION = re.compile(r"^gpt-\d")


def _should_use_responses_api(model_name: str) -> bool:
    """Return True when *model_name* must be called via /v1/responses."""
    return (bool(_GPT_GENERATION.match(model_name))
            and not model_name.startswith(_CHAT_COMPLETIONS_MODELS))


def _build_responses_api_request(
    model,
    messages: list,
    output:   dict,
    tools:    list | None = None,
) -> tuple[str, dict]:
    """
    Build an OpenAI ``/v1/responses`` ``(path, body)`` pair.

    Key differences from Chat Completions:
    - ``input``        instead of ``messages``
    - ``instructions`` for the system message (separate top-level field)
    - ``max_output_tokens`` instead of ``max_completion_tokens``
    - ``text.format``  instead of ``response_format`` for structured output
    """
    system_texts: list = []
    input_messages: list[dict] = []
    held: list = []                      # as in the chat builder: after the turn's results

    def release():
        if held:
            input_messages.append({"role": "user", "content": [_part_to_responses(p) for p in held]})
            held.clear()

    for msg in messages:
        role = msg["role"]

        # Tool traffic uses top-level input items, not role messages — the
        # Responses API's own shape, different from chat completions.
        if role == "tool":
            from ...models._calls import split_media_result
            msg, follow_up = split_media_result(msg)
            input_messages.append({
                "type":    "function_call_output",
                "call_id": msg.get("call_id", ""),
                "output":  _text_of(msg),
            })
            if follow_up:
                held.extend(follow_up["parts"])
            continue
        release()
        if role == "assistant" and msg.get("tool_calls"):
            for c in msg["tool_calls"]:
                input_messages.append({
                    "type":      "function_call",
                    "call_id":   c.get("id", ""),
                    "name":      c["name"],
                    "arguments": json.dumps(c.get("arguments", {}),
                                            ensure_ascii=False),
                })
            continue

        items = [_part_to_responses(p, role) for p in msg["parts"]]
        items = [it for it in items if it is not None]
        if not items:
            continue

        if role == "system":
            # Responses API takes the system prompt as a top-level field —
            # one string, so several system messages are joined here, as every
            # other family joins them. Assigning instead would keep only the
            # last, and the same conversation would carry different
            # instructions depending on the model it went to.
            text_parts = [it["text"] for it in items
                          if it["type"] in ("input_text", "output_text")]
            system_texts.extend(text_parts)
        else:
            # No collapsing a lone text part to a bare string: one shape for
            # one part and many, so the common single-part case cannot hide
            # a defect in the multi-part one.
            input_messages.append({"role": role, "content": items})

    release()
    # gpt-5-family models are reasoning models: the Responses API rejects
    # ``temperature``/``top_p`` for them ("Unsupported parameter"), so the
    # sampling knobs are deliberately omitted here.
    body: dict = {
        "model":             model.name,
        "input":             input_messages,
        "max_output_tokens": model.max_tokens,
    }

    instructions = "\n".join(system_texts)
    if instructions:
        body["instructions"] = instructions

    if tools:
        # The Responses API takes the FLAT schema — name/description/parameters
        # at top level — where chat completions nests them under "function".
        # Canonical input here is the chat form (Tool.schema()); unwrap it.
        body["tools"] = [
            {"type": "function", **t["function"]} if "function" in t else t
            for t in tools
        ]

    if model.reasoning:
        effort = model._REASONING_MAP.get(model.reasoning)
        if effort:
            body["reasoning"] = {"effort": effort}

    # Structured output — lives under ``text.format`` in the Responses API
    fmt   = output.get("format", {})
    ftype = fmt.get("type", "text")
    if ftype == "json":
        body["text"] = {"format": {"type": "json_object"}}
    elif ftype == "json_schema":
        body["text"] = {"format": {"type": "json_schema", **_json_schema(fmt)}}

    return "/v1/responses", body


def _parse_responses_api_response(response: dict, output: dict) -> str | dict:
    """
    Extract the result from an OpenAI Responses API response.

    Response structure::

        {
          "output": [
            {
              "type": "message",
              "role": "assistant",
              "content": [{"type": "output_text", "text": "…"}]
            }
          ]
        }
    """
    # Calls first: a reply that both says something and asks to act is a call,
    # with the text carried alongside. The output array mixes item kinds
    # (reasoning, message, function_call) — collect across all of them.
    calls = [it for it in response.get("output", [])
             if it.get("type") == "function_call"]
    text  = ""
    for item in response.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text = part["text"]
    if response.get("status") == "failed":
        detail = (response.get("error") or {}).get("message", "no reason given")
        raise ValueError(f"The Responses API reports the response failed: {detail}")
    if response.get("status") == "incomplete":
        from ._structured import refuse_cut_off
        reason = (response.get("incomplete_details") or {}).get("reason") or ""
        refuse_cut_off("max_tokens" if reason == "max_output_tokens" else reason,
                       (response.get("usage") or {}).get("output_tokens"),
                       calls=bool(calls), text=text)
    if calls:
        return _tool_call_request(
            ((c.get("name", ""), c.get("arguments"), c.get("call_id", ""))
             for c in calls),
            text=text,
        )
    if text:
        ftype = output.get("format", {}).get("type", "text")
        if ftype in ("json", "json_schema"):
            # The Responses API reports the ceiling as `incomplete_details`
            # rather than a finish_reason, and this path had no truncation
            # handling at all — a cut-off plan surfaced as a JSONDecodeError
            # with nothing to attribute it to.
            from ._structured import parse_structured
            reason = ((response.get("incomplete_details") or {}).get("reason")
                      or response.get("status_details", {}).get("reason", ""))
            usage = response.get("usage") or {}
            return parse_structured(
                text, output.get("format", {}).get("schema"),
                finish_reason=("max_tokens" if reason == "max_output_tokens"
                               else reason or ""),
                output_tokens=usage.get("output_tokens"),
            )
        return text
    return ""


# ---------------------------------------------------------------------------
# openai provider
# ---------------------------------------------------------------------------

