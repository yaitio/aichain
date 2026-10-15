"""The universal input and output format on the Skill side: normalising the
shorthand forms, validating messages and output specs, substituting
``{placeholder}`` tokens, and reading ``kind: "file"`` media on send. The
provider-specific translation lives in the family clients.
See docs/models/models.md, "The message format".
"""

import base64
import copy
import pathlib

from .._template import substitute_placeholders


# ---------------------------------------------------------------------------
# Local-file media sources
# ---------------------------------------------------------------------------

#: File-extension → MIME, used when a ``kind:"file"`` source omits ``mime``.
_EXT_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".flac": "audio/flac",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
}


def _guess_mime(path: pathlib.Path, head: bytes) -> str:
    """Best-effort MIME for a local file: extension first, then magic bytes."""
    by_ext = _EXT_MIME.get(path.suffix.lower())
    if by_ext:
        return by_ext
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:4] == b"\x89PNG":
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:4] == b"GIF8":
        return "image/gif"
    return "application/octet-stream"


def _resolve_file_source(part: dict) -> None:
    """Read a ``kind: "file"`` source with no ``data`` into base64, in place, with
    ``mime`` guessed when absent. The family clients read ``source["data"]`` for
    both ``base64`` and ``file`` kinds."""
    src = part.get("source")
    if not isinstance(src, dict) or src.get("kind") != "file":
        return
    if src.get("data") or "path" not in src:
        return
    from ..tools._security import check_input_file
    path = pathlib.Path(check_input_file(src["path"], part.get("type", "image")))
    raw = path.read_bytes()
    src["data"] = base64.b64encode(raw).decode("ascii")
    if not src.get("mime"):
        src["mime"] = _guess_mime(path, raw[:16])


def resolve_media(messages: list) -> list:
    """
    Return *messages* with every ``kind:"file"`` source read into base64.

    Called on send, after :func:`substitute`, so a media path may contain a
    ``{placeholder}`` and be pointed at a different file per call. Parts that
    already carry ``data`` are untouched, and a message list with no file
    sources is returned unchanged rather than copied.
    """
    if not any(isinstance(p, dict) and isinstance(p.get("source"), dict)
               and p["source"].get("kind") == "file" and not p["source"].get("data")
               for m in messages for p in m.get("parts", [])):
        return messages
    out = copy.deepcopy(messages)
    for msg in out:
        for part in msg.get("parts", []):
            if isinstance(part, dict):
                _resolve_file_source(part)
    return out


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_input(input_: dict) -> dict:
    """
    Return a normalised deep copy of *input_*, filling in implicit defaults.

    A plain ``str`` part, or a dict with ``"text"`` and no ``"type"``, becomes
    ``{"type": "text", "text": ...}``; media and typed parts pass unchanged.

    Parameters
    ----------
    input_ : dict
        Raw input dict, possibly using shorthand forms.

    Returns
    -------
    dict
        Normalised copy ready for :func:`validate_input`.
    """
    if not isinstance(input_, dict):
        validate_input(input_)                  # raises, naming the shape expected
    result = copy.deepcopy(input_)
    for msg in result.get("messages", []):
        if not isinstance(msg, dict):
            continue
        parts = msg.get("parts")
        if not isinstance(parts, list):
            continue
        normalised_parts = []
        for part in parts:
            if isinstance(part, str):
                # Plain string → text part
                normalised_parts.append({"type": "text", "text": part})
            elif isinstance(part, dict) and "type" not in part and "text" in part:
                # Dict with "text" but no "type" → default type to "text"
                normalised_parts.append({"type": "text", **part})
            else:
                # A media part is left as written and read on send, so a path
                # may carry a {placeholder} and one Skill can serve a whole Pool.
                normalised_parts.append(part)
        msg["parts"] = normalised_parts
    return result


def normalize_output(output: "dict | None") -> dict:
    """
    Return a normalised output dict, filling in implicit defaults.

    ``None``, ``{}`` and any missing part default to plain text:
    ``{"modalities": ["text"], "format": {"type": "text"}}``.

    Parameters
    ----------
    output : dict | None
        Raw output spec, possibly empty or partial.

    Returns
    -------
    dict
        Normalised dict ready for :func:`validate_output`.
    """
    if not output:
        return {"modalities": ["text"], "format": {"type": "text"}}

    result = dict(output)

    if "modalities" not in result:
        result["modalities"] = ["text"]

    if "format" not in result:
        result["format"] = {"type": "text"}
    elif isinstance(result["format"], dict) and "type" not in result["format"]:
        result["format"] = {"type": "text", **result["format"]}

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_input(input_: dict) -> None:
    """
    Raise ``ValueError`` if *input_* does not conform to the universal format.

    Checks:
    - Top-level ``messages`` key is a non-empty list.
    - Each message has a valid ``role`` and a non-empty ``parts`` list.
    - Each part has a valid ``type``; ``text`` parts have a string ``text``
      field; media parts have a ``source`` dict with a valid ``kind``.
    """
    if not isinstance(input_, dict):
        if isinstance(input_, list):
            raise ValueError("input must be a dict with a 'messages' list: "
                             "input={'messages': [...]}; got a bare list")
        raise ValueError("input must be a dict")
    messages = input_.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("input['messages'] must be a non-empty list")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{i}] must be a dict")
        # `content` is every other API's word. Accepted, an assistant turn
        # written that way would read as the "generate here" marker.
        if "content" in msg and "parts" not in msg:
            raise ValueError(
                f"messages[{i}] has 'content'; messages here carry 'parts': "
                f"{{'role': {msg.get('role', 'user')!r}, 'parts': [<text or part>, ...]}}"
            )
        if msg.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError(
                f"messages[{i}]['role'] must be 'system', 'user', 'assistant', "
                "or 'tool'"
            )
        # A ``tool`` turn is one call's result, keyed to the call id. The id is
        # mandatory: providers reject results that reference nothing, and a
        # result without one is indistinguishable from ordinary conversation.
        if msg["role"] == "tool" and not isinstance(msg.get("call_id"), str):
            raise ValueError(
                f"messages[{i}]: a 'tool' turn must carry a string 'call_id' "
                "referencing the assistant tool call it answers"
            )
        # An ``assistant`` turn may carry ``tool_calls`` — the model asked to
        # act. Each call needs a name and dict arguments; a malformed one here
        # would only surface later as a provider 400 with no pointer back.
        calls = msg.get("tool_calls")
        if calls is not None:
            if msg["role"] != "assistant":
                raise ValueError(
                    f"messages[{i}]: only 'assistant' turns may carry 'tool_calls'"
                )
            if not isinstance(calls, list) or not calls:
                raise ValueError(
                    f"messages[{i}]['tool_calls'] must be a non-empty list"
                )
            for k, call in enumerate(calls):
                if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                    raise ValueError(
                        f"messages[{i}]['tool_calls'][{k}] must be a dict "
                        "with a string 'name'"
                    )
                if not isinstance(call.get("arguments", {}), dict):
                    raise ValueError(
                        f"messages[{i}]['tool_calls'][{k}]['arguments'] "
                        "must be a dict"
                    )
        parts = msg.get("parts")
        # An ``assistant`` turn with no parts and no tool_calls is a "generate
        # here" marker (multi-turn directed reasoning): the model fills it in at
        # run time, so it legitimately carries no content. An assistant turn
        # WITH tool_calls needs no parts either — the calls are its content.
        if msg["role"] == "assistant" and not parts:
            continue
        if not isinstance(parts, list) or not parts:
            raise ValueError(
                f"messages[{i}]['parts'] must be a non-empty list"
            )
        for j, part in enumerate(parts):
            if not isinstance(part, dict):
                raise ValueError(f"messages[{i}]['parts'][{j}] must be a dict")
            ptype = part.get("type")
            if ptype not in ("text", "image", "video", "audio", "document"):
                raise ValueError(
                    f"messages[{i}]['parts'][{j}]['type'] must be "
                    "'text', 'image', 'video', 'audio' or 'document'"
                )
            if ptype == "text":
                if not isinstance(part.get("text"), str):
                    raise ValueError(
                        f"messages[{i}]['parts'][{j}]['text'] must be a string"
                    )
            else:
                src = part.get("source")
                if not isinstance(src, dict):
                    raise ValueError(
                        f"messages[{i}]['parts'][{j}]['source'] must be a dict"
                    )
                if src.get("kind") not in ("url", "base64", "file"):
                    raise ValueError(
                        f"messages[{i}]['parts'][{j}]['source']['kind'] must be "
                        "'url', 'base64', or 'file'"
                    )

    _validate_turn_structure(messages)


def is_generate_marker(msg: dict) -> bool:
    """
    True for an ``assistant`` turn with no ``parts`` — the "generate here"
    marker used by multi-turn directed reasoning (the model produces this turn
    at run time and the reply is appended to the running context).
    """
    return (isinstance(msg, dict) and msg.get("role") == "assistant"
            and not msg.get("parts") and not msg.get("tool_calls"))


def _validate_turn_structure(messages: list) -> None:
    """Raise ``ValueError`` at construction unless the turns read
    ``[system?] user (assistant user)*`` (consecutive users allowed), and every
    ``tool`` turn follows the assistant turn whose call it answers."""
    content_idx = [i for i, m in enumerate(messages) if m.get("role") != "system"]
    if not any(messages[i].get("role") == "user" for i in content_idx):
        raise ValueError("input['messages'] must contain at least one 'user' turn")

    if content_idx and messages[content_idx[0]].get("role") == "assistant":
        raise ValueError(
            f"messages[{content_idx[0]}]: an 'assistant' turn cannot come first "
            f"— it must follow a 'user' turn"
        )

    prev_assistant  = False
    open_call_turn  = False   # last turn was assistant-with-tool_calls or tool
    for i in content_idx:
        role = messages[i].get("role")
        if role == "tool":
            # A result must answer a call: it may only follow the assistant
            # turn that made the call, or a sibling result of the same turn.
            if not open_call_turn:
                raise ValueError(
                    f"messages[{i}]: a 'tool' turn must directly follow an "
                    f"'assistant' turn carrying tool_calls (or another 'tool' "
                    f"turn answering the same calls)"
                )
            prev_assistant = False
            continue
        is_assistant = role == "assistant"
        if is_assistant and prev_assistant:
            raise ValueError(
                f"messages[{i}]: two 'assistant' turns in a row — every "
                f"'assistant' turn must be preceded by a 'user' turn"
            )
        prev_assistant = is_assistant
        open_call_turn = is_assistant and bool(messages[i].get("tool_calls"))


def validate_output(output: dict) -> None:
    """
    Raise ``ValueError`` if *output* does not conform to the universal format.

    Checks:
    - Top-level ``modalities`` is a non-empty list.
    - ``format`` is a dict with a valid ``type`` field.
    - ``json_schema`` format has a ``schema`` dict.
    """
    if not isinstance(output, dict):
        raise ValueError("output must be a dict")
    modalities = output.get("modalities")
    if not isinstance(modalities, list) or not modalities:
        raise ValueError("output['modalities'] must be a non-empty list")
    fmt = output.get("format")
    if not isinstance(fmt, dict):
        raise ValueError("output['format'] must be a dict")
    ftype = fmt.get("type")
    if ftype not in ("text", "json", "json_schema", "image"):
        raise ValueError(
            "output['format']['type'] must be 'text', 'json', 'json_schema', or 'image'"
        )
    if ftype == "json" and "schema" in fmt:
        # A schema under `json` is never sent, yet the answer would be checked
        # against it — a failure for a rule the model never saw.
        raise ValueError("a schema goes with format type 'json_schema', not 'json': "
                         "{'type': 'json_schema', 'schema': …} sends it to the model")
    if ftype == "json_schema" and not isinstance(fmt.get("schema"), dict):
        raise ValueError(
            "output['format']['schema'] must be a dict for json_schema format"
        )


# ---------------------------------------------------------------------------
# Variable substitution
# ---------------------------------------------------------------------------

def substitute(messages: list, variables: dict) -> list:
    """
    Return *messages* with ``{placeholders}`` in text parts replaced by the
    matching value from *variables* — a new list where anything is replaced,
    and *messages* itself where there are no variables. Nothing downstream
    writes into a conversation (pinned for every provider), so the copy an
    agent turn would make of its whole history buys nothing.

    Substitution is performed by
    :func:`yait_aichain._template.substitute_placeholders`: only bare
    ``{identifier}`` placeholders with a matching key are replaced.  Unknown
    placeholders, format specs and literal braces (e.g. JSON examples like
    ``{"a": 1}``) are left intact — they never raise.

    Non-text parts are deep-copied unchanged.
    """
    if not variables:
        return messages
    result = []
    for msg in messages:
        new_parts = []
        for part in msg.get("parts", []):
            if part.get("type") == "text":
                new_part = dict(part)
                new_part["text"] = substitute_placeholders(
                    part["text"], variables, warn=True
                )
                new_parts.append(new_part)
            else:
                new_part = copy.deepcopy(part)
                src = new_part.get("source")
                if isinstance(src, dict):
                    # The path is as much a template as the prompt is; without
                    # this a media Skill can only ever point at one file.
                    for field in ("path", "url"):
                        if isinstance(src.get(field), str):
                            src[field] = substitute_placeholders(
                                src[field], variables)
                new_parts.append(new_part)
        new_msg = dict(msg)
        new_msg["parts"] = new_parts
        result.append(new_msg)
    return result
