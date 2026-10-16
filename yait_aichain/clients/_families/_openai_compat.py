"""Wire-format helpers for the OpenAI Chat Completions, Responses and Images
APIs, shared by every OpenAI-compatible provider. Pure functions: universal
format to provider JSON and back. See docs/models/models.md.
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
    """Drop the parameters the model's API refuses (``rejects`` in the provider
    data, established by asking the API, not the guide), each recorded as
    declined: a hard 400 becomes a working call, and is still said."""
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


def _dropped(part: dict, why: str) -> None:
    """A part this wire cannot carry: left out, and said, or the model answers
    about an attachment it never saw."""
    from ...models._adaptation import Adaptation, DECLINED, record
    src = part.get("source") or {}
    record(Adaptation(kind=DECLINED, option=f"{part.get('type')} input",
                      asked=src.get("mime") or src.get("kind"), sent=None,
                      why=f"{why}; the part was left out of the request"))
    return None


#: The audio formats chat completions names, by MIME type: "mpeg", the MIME
#: subtype, is not a format any endpoint takes.
_AUDIO_FORMATS = {"audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav",
                  "audio/x-wav": "wav", "audio/wave": "wav"}


def _audio_format(mime: str) -> str | None:
    return _AUDIO_FORMATS.get((mime or "").lower())


def _part_to_openai(part: dict) -> dict | None:
    """One universal part as a chat content item, or ``None`` (recorded as
    declined) for a part this wire cannot carry."""
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
        if kind in ("base64", "file"):
            fmt = _audio_format(src.get("mime", "audio/wav"))
            if fmt is None:
                return _dropped(part, f"audio input takes wav or mp3, not {src.get('mime')}")
            return {
                "type":        "input_audio",
                "input_audio": {"data": src["data"], "format": fmt},
            }
        return _dropped(part, "audio input takes the bytes, not a URL")

    if ptype == "document":
        src = part["source"]
        if src.get("kind") in ("base64", "file"):
            mime = src.get("mime", "application/pdf")
            # A PDF goes as a file part, as the Responses API sends it.
            return {"type": "file",
                    "file": {"filename": src.get("filename", "document"),
                             "file_data": f"data:{mime};base64,{src['data']}"}}
        return _dropped(part, "a document is sent as its bytes, not a URL")

    if ptype == "image":
        return _dropped(part, f"an image source of kind {part['source'].get('kind')!r} cannot be sent")
    return _dropped(part, f"chat completions take no {ptype} input")


def _part_to_responses(part: dict, role: str = "user") -> dict | None:
    """One universal part as a Responses API item: ``input_text``/``input_image``
    with the URL a bare string (the chat spelling is refused outright), and
    ``output_text`` for an assistant's text."""
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
            return _dropped(part, f"an image source of kind {kind!r} cannot be sent")
        # A string, not an object — the object form is the chat spelling.
        return {"type": "input_image", "image_url": url, "detail": detail}

    if ptype == "document":
        src = part["source"]
        if src.get("kind") in ("base64", "file"):
            mime = src.get("mime", "application/pdf")
            return {"type": "input_file",
                    "filename": src.get("filename", "document"),
                    "file_data": f"data:{mime};base64,{src['data']}"}
        return _dropped(part, "a document is sent as its bytes, not a URL")

    if ptype == "audio":
        src = part["source"]
        if src.get("kind") in ("base64", "file"):
            fmt = _audio_format(src.get("mime", "audio/wav"))
            if fmt is None:
                return _dropped(part, f"audio input takes wav or mp3, not {src.get('mime')}")
            return {"type": "input_audio",
                    "input_audio": {"data": src["data"], "format": fmt}}
        return _dropped(part, "audio input takes the bytes, not a URL")

    return _dropped(part, f"the Responses API takes no {ptype} input")


def _text_of(msg: dict) -> str:
    """The concatenated text parts of a universal message ("" when none)."""
    return "\n".join(p.get("text", "") for p in msg.get("parts") or []
                     if p.get("type") == "text")


def _chat_messages(model, messages: list) -> list:
    """Universal messages as Chat Completions messages, system ones first."""
    openai_messages: list[dict] = []
    # Media the tools of one turn produced, sent in one user message after the
    # last of that turn's results: the results of parallel calls must follow
    # their assistant turn back to back, and a user message between two of
    # them is a 400.
    held: list = []

    def release():
        if held:
            # Without the parts this wire cannot carry: a None among them
            # would go out as `null`, a 400.
            items = [it for it in (_part_to_openai(p) for p in held) if it is not None]
            if items:
                openai_messages.append({"role": "user", "content": items})
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
            kept = (msg.get("opaque") or {}).get("reasoning_content")
            if kept and getattr(model, "_KEEP_REASONING", False):
                entry["reasoning_content"] = kept
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
    # A system message applies to the whole conversation wherever it sits:
    # the only meaning every provider can express, since Anthropic, Google and
    # the Responses API carry it in a field of their own. Hoisted here too, and
    # reported, because for this family it changes what the model sees.
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
    return openai_messages


def _build_openai_compat_request(
    model,
    messages: list,
    output:   dict,
    path:     str,
    max_tokens_field: str = "max_completion_tokens",
    tools:    list | None = None,
) -> tuple[str, dict]:
    """An OpenAI-compatible chat ``(path, body)``. *max_tokens_field* is the
    output-limit key: ``max_completion_tokens``, or ``max_tokens`` on
    providers that still name it so."""
    openai_messages = _chat_messages(model, messages)
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
    """A JSON Schema in OpenAI strict form, recursively: objects closed, every
    property required (an optional one as ``[X, "null"]``). A map with arbitrary
    keys has no strict form and raises. Mirror of ``_sanitize_google_schema``."""
    if not isinstance(schema, dict):
        return schema

    ap = schema.get("additionalProperties")
    if isinstance(ap, dict) and not schema.get("properties"):
        raise SchemaNotExpressible(
            f"{path}: a map with arbitrary keys cannot be expressed in "
            "OpenAI strict mode. Model it as an array of {key, value} "
            "objects, or pass strict=False."
        )
    if schema.get("type") == "object" and not schema.get("properties") and ap is not False:
        # A free object: strict mode closes every object, so it would accept
        # only `{}`, and the provider answers 400.
        raise SchemaNotExpressible(
            f"{path}: an object with no properties cannot be expressed in OpenAI "
            "strict mode; give it properties, or pass strict=False.")

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
        for name, sub in list(props.items()):
            if name in was_required or not isinstance(sub, dict):
                continue
            t = sub.get("type")
            if "enum" in sub and None not in sub["enum"]:
                sub["enum"] = [*sub["enum"], None]   # null must be a value it may take
            if isinstance(t, str) and t != "null":
                sub["type"] = [t, "null"]        # optional, strict spelling
            elif isinstance(t, list) and "null" not in t:
                sub["type"] = [*t, "null"]
            elif t is None:
                # A $ref, an anyOf, an enum with no type: required and not
                # nullable would force the model to invent a value. Optional
                # is "it, or null".
                props[name] = {"anyOf": [sub, {"type": "null"}]}
        result["required"] = list(props.keys())
    return result


def _tool_call_request(triples, text: str = "", opaque: dict | None = None) -> ToolCallRequest:
    """A :class:`ToolCallRequest` from ``(name, arguments, id)`` triples; arguments
    as a JSON string or a dict, unparseable ones as ``{}`` so the executor's
    schema check complains with the tool named."""
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
    return ToolCallRequest(calls=tuple(calls), text=text, opaque=opaque or {})


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
    """A harmony reply as a :class:`ToolCallRequest` (``commentary`` addressed
    ``to=functions.NAME``) or the ``final`` text; ``analysis`` is dropped. One
    pass, cut at stop tokens: a backtracking pattern takes seconds on a long reply."""
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
    """The clean result of a chat completion: text, a parsed object, or a
    :class:`ToolCallRequest`. Raises ``ValueError`` on no choices, a refusal,
    or JSON that does not parse."""
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
    # carried alongside.
    from ._structured import refuse_cut_off
    finish = choice.get("finish_reason") or ""
    spent = (response.get("usage") or {}).get("completion_tokens")
    refuse_cut_off(finish, spent, calls=bool(message.get("tool_calls")),
                   text=content_text(message.get("content")))
    if message.get("tool_calls"):
        # vLLM renamed the field to `reasoning`, as OpenRouter has it.
        thought = message.get("reasoning_content")
        if not isinstance(thought, str):
            thought = message.get("reasoning") if isinstance(message.get("reasoning"), str) else None
        return _tool_call_request(
            ((c.get("function", {}).get("name", ""),
              c.get("function", {}).get("arguments"),
              c.get("id", ""))
             for c in message["tool_calls"]),
            text=content_text(message.get("content")),
            opaque={"reasoning_content": thought} if thought else None,
        )

    text  = content_text(message.get("content"))

    # A server that does not implement harmony passes it through as prose;
    # parsed here, a lost tool call becomes a real one. Only for a model that
    # writes it (*harmony* is "calls" when tools were sent, "text" when not):
    # another model quoting a channel marker from a file is only quoting.
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


def _ratio_as_size(fmt: dict, model) -> dict:
    """A ratio as pixels for an API that takes only pixels — WxH in multiples of
    16 between 1:3 and 3:1, which a 1024 short edge meets. Every other image
    provider in the library takes both."""
    if not fmt.get("aspect_ratio") or fmt.get("size"):
        return fmt
    size = size_from_ratio(fmt["aspect_ratio"], multiple=16)
    if not size:
        return fmt
    from ...models._adaptation import Adaptation, ADAPTED, record
    record(Adaptation(
        kind=ADAPTED, option="aspect_ratio", asked=fmt["aspect_ratio"],
        sent=f"size={size}", model=model.name,
        why="this provider takes pixels, not a ratio; the shape was "
            "kept at a 1024 short edge"))
    return {**fmt, "size": size}


def _build_image_generations_request(
    model,
    messages: list,
    output:   dict,
    path:     str = "/v1/images/generations",
) -> tuple[str, dict]:
    """An ``/v1/images/generations`` ``(path, body)`` for openai and xai: the
    prompt from the last user message, ``response_format: b64_json`` on every
    model but ``gpt-image-*``, which returns base64 natively and refuses it."""
    prompt = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            texts = [p["text"] for p in msg["parts"] if p["type"] == "text"]
            if texts:
                prompt = "\n".join(texts)
                break

    fmt: dict  = _drop_rejected(model, output.get("format", {}))
    body: dict = {"model": model.name, "prompt": prompt, "n": 1}

    fmt = _ratio_as_size(fmt, model)

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
    """The MIME type of a base64 image from its magic bytes (JPEG, PNG, WebP,
    GIF, SVG); ``image/png`` when unknown."""
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
    """``{url, base64, mime_type, revised_prompt}`` from an images response, the
    MIME type read from the bytes. ``ValueError`` on no data (moderation)."""
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


def _edit_sources(messages: list, provider: str, model_name: str, limit: int | None = None) -> list:
    """The images an edit sends: the newest *limit* (the provider's
    `max_input_images` unless given), and a `declined` for the rest."""
    from ...models._adaptation import Adaptation, DECLINED, record
    from ...models._data import PROVIDERS
    sources = _image_sources(messages)
    if limit is None:
        limit = ((PROVIDERS.get(provider) or {}).get("provider", {}).get("options", {})
                 .get("max_input_images"))
    if limit is not None and len(sources) > limit:
        record(Adaptation(kind=DECLINED, option="images", asked=len(sources), sent=limit,
                          model=model_name,
                          why=f"this edit takes at most {limit} image(s); the newest were sent and "
                              f"{len(sources) - limit} earlier one(s) left out"))
        sources = sources[-limit:]
    return sources


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
    """An ``/v1/images/edits`` body as a :class:`Multipart` of ``(name, value)``
    pairs, so several references share ``image[]`` (``gpt-image-*``);
    ``dall-e-2`` takes one ``image``."""
    sources = _edit_sources(messages, "openai", model.name,
                            None if model.name.startswith(_GPT_IMAGE) else 1)
    if not sources:
        raise ValueError("image edit requires at least one input image part")

    field_name = "image[]" if model.name.startswith(_GPT_IMAGE) else "image"

    fields: list = [("model", model.name), ("prompt", _last_user_text(messages))]
    fmt = _ratio_as_size(_drop_rejected(model, output.get("format", {})), model)
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
    """An xAI ``/v1/images/edits`` body: JSON, not multipart, the image as
    ``{"url": …, "type": "image_url"}``, and base64 back, which
    ``_parse_image_generations_response`` reads unchanged."""
    sources = _edit_sources(messages, "xai", model.name)
    if not sources:
        raise ValueError("image edit requires at least one input image part")
    imgs = [{"url": _image_source_to_data_uri(s), "type": "image_url"} for s in sources]
    body = {"model": model.name, "prompt": _last_user_text(messages),
            "response_format": "b64_json"}
    # Shape control works on this path too: aspect_ratio "16:9" returns 1280x720.
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
#: The list is of the exceptions, not of the Responses models: new models ship
#: on the new endpoint, so it is the default and the old one has to be named.
#: The gpt-4 and gpt-3.5 generations are a closed set.
_CHAT_COMPLETIONS_MODELS = ("gpt-4", "gpt-3.5")

#: ``gpt-<digit>`` — the generation number names a chat model. It deliberately
#: does not match ``gpt-image-*``, ``gpt-oss-*`` or ``gpt-realtime-*``, which
#: are other APIs that only share the prefix.
_GPT_GENERATION = re.compile(r"^gpt-\d")


def _should_use_responses_api(model_name: str) -> bool:
    """Return True when *model_name* must be called via /v1/responses."""
    return (bool(_GPT_GENERATION.match(model_name))
            and not model_name.startswith(_CHAT_COMPLETIONS_MODELS))


def _responses_input(messages: list) -> tuple:
    """Universal messages as the Responses API's system texts and input items."""
    system_texts: list = []
    input_messages: list[dict] = []
    held: list = []                      # as in the chat builder: after the turn's results

    def release():
        if held:
            items = [it for it in (_part_to_responses(p) for p in held) if it is not None]
            if items:
                input_messages.append({"role": "user", "content": items})
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
            # other family joins them.
            text_parts = [it["text"] for it in items
                          if it["type"] in ("input_text", "output_text")]
            system_texts.extend(text_parts)
        else:
            # No collapsing a lone text part to a bare string: one shape for
            # one part and many, so the common single-part case cannot hide
            # a defect in the multi-part one.
            input_messages.append({"role": role, "content": items})

    release()
    return system_texts, input_messages


def _build_responses_api_request(
    model,
    messages: list,
    output:   dict,
    tools:    list | None = None,
) -> tuple[str, dict]:
    """An ``/v1/responses`` ``(path, body)``: ``input`` for messages,
    ``instructions`` for the system prompt, ``max_output_tokens``, and
    ``text.format`` for structured output."""
    system_texts, input_messages = _responses_input(messages)
    # gpt-5-family models are reasoning models: the Responses API rejects
    # ``temperature``/``top_p`` for them ("Unsupported parameter"), so the
    # sampling knobs are deliberately omitted here.
    body: dict = {
        "model":             model.name,
        "input":             input_messages,
        "max_output_tokens": model.max_tokens,
        # Not kept by OpenAI: nothing here reads `previous_response_id`, and
        # the API stores every conversation by default. `extra={"store": True}`
        # keeps it.
        "store":             False,
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
    """The result of a Responses API reply: its ``function_call`` items as a
    :class:`ToolCallRequest`, else the ``output_text`` of its message items."""
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
                    text += part["text"]           # accumulated, not the last one
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
            # rather than a finish_reason; read here, a cut-off answer is not
            # an unattributed JSONDecodeError.
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

