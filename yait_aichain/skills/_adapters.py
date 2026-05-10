"""
skills._adapters
================

Provider-agnostic format utilities used by the Skill layer.

The actual provider-specific serialisation / deserialisation logic lives
inside each model class as ``to_request()`` and ``from_response()``
methods.  This module only contains the pieces that are not tied to any
single provider:

* :func:`normalize_input`  — fill in implicit defaults in an input dict
* :func:`normalize_output` — fill in implicit defaults in an output dict
* :func:`validate_input`   — check a normalised input dict
* :func:`validate_output`  — check a normalised output dict
* :func:`substitute`       — replace ``{placeholder}`` tokens in message text

Universal input format
----------------------
Full (explicit) form::

    {
      "messages": [
        {
          "role": "system" | "user" | "assistant",
          "parts": [
            {"type": "text",  "text": "..."},
            {
              "type": "image",
              "source": {"kind": "url",    "url": "https://…"},
              "meta":   {"detail": "high"}
            },
            {
              "type": "image",
              "source": {"kind": "base64", "mime": "image/png", "data": "…"},
              "meta":   {}
            },
            {
              "type": "video",
              "source": {"kind": "url", "url": "https://…"},
              "meta":   {"sampling": {"strategy": "fps", "fps": 1, "max_frames": 60},
                         "include_audio_track": false}
            },
            {
              "type": "audio",
              "source": {"kind": "base64", "mime": "audio/wav", "data": "…"},
              "meta":   {"purpose": "stt"}
            }
          ]
        }
      ]
    }

Shorthand form (normalised automatically):

* A part that is a plain string → ``{"type": "text", "text": string}``
* A part dict that has ``"text"`` but no ``"type"`` → ``"type"`` defaults to
  ``"text"``

So the minimal input for a single user message is::

    {"messages": [{"role": "user", "parts": ["Hello!"]}]}

    # or, equivalently:
    {"messages": [{"role": "user", "parts": [{"text": "Hello!"}]}]}

Universal output format
-----------------------
Default (omit entirely or pass ``{}`` / ``None`` for plain text output)::

    {}   →   {"modalities": ["text"], "format": {"type": "text"}}

Explicit forms::

    {"modalities": ["text"], "format": {"type": "text"}}

    {"modalities": ["text"], "format": {"type": "json"}}

    {"modalities": ["text"], "format": {
        "type":   "json_schema",
        "name":   "result",
        "schema": {...},
        "strict": true
    }}

    {"modalities": ["image"], "format": {"type": "image"}}

    {"modalities": ["image", "text"], "format": {"type": "image"}}

Partial forms (each missing key is filled in with its default)::

    {"format": {"type": "json"}}
    →  {"modalities": ["text"], "format": {"type": "json"}}

    {"modalities": ["image"], "format": {"type": "image"}}
    →  (already complete; unchanged)
"""

import copy


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_input(input_: dict) -> dict:
    """
    Return a normalised deep copy of *input_*, filling in implicit defaults.

    Part shortcuts (applied per message):

    * A plain ``str`` part → ``{"type": "text", "text": <string>}``
    * A dict part with a ``"text"`` key but no ``"type"`` key → ``"type"``
      is set to ``"text"`` automatically.

    Media parts (those with a ``"source"`` key) and parts that already have
    a ``"type"`` field are passed through unchanged.

    Parameters
    ----------
    input_ : dict
        Raw input dict, possibly using shorthand forms.

    Returns
    -------
    dict
        Normalised copy ready for :func:`validate_input`.

    Examples
    --------
    ::

        normalize_input({
            "messages": [
                {"role": "system", "parts": ["Be concise."]},
                {"role": "user",   "parts": [{"text": "Explain {topic}."}]},
            ]
        })
        # →
        {
            "messages": [
                {"role": "system", "parts": [{"type": "text", "text": "Be concise."}]},
                {"role": "user",   "parts": [{"type": "text", "text": "Explain {topic}."}]},
            ]
        }
    """
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
                normalised_parts.append(part)
        msg["parts"] = normalised_parts
    return result


def normalize_output(output: "dict | None") -> dict:
    """
    Return a normalised output dict, filling in implicit defaults.

    Passing ``None`` or ``{}`` produces the plain-text default::

        {"modalities": ["text"], "format": {"type": "text"}}

    Missing ``"modalities"`` defaults to ``["text"]``.
    Missing ``"format"`` defaults to ``{"type": "text"}``.
    A ``"format"`` dict that is present but lacks a ``"type"`` key has
    ``"type"`` set to ``"text"``.

    Parameters
    ----------
    output : dict | None
        Raw output spec, possibly empty or partial.

    Returns
    -------
    dict
        Normalised dict ready for :func:`validate_output`.

    Examples
    --------
    ::

        normalize_output(None)
        # → {"modalities": ["text"], "format": {"type": "text"}}

        normalize_output({})
        # → {"modalities": ["text"], "format": {"type": "text"}}

        normalize_output({"format": {"type": "json"}})
        # → {"modalities": ["text"], "format": {"type": "json"}}

        normalize_output({"modalities": ["image"], "format": {"type": "image"}})
        # → {"modalities": ["image"], "format": {"type": "image"}}
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
        raise ValueError("input must be a dict")
    messages = input_.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("input['messages'] must be a non-empty list")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{i}] must be a dict")
        if msg.get("role") not in ("system", "user", "assistant"):
            raise ValueError(
                f"messages[{i}]['role'] must be 'system', 'user', or 'assistant'"
            )
        parts = msg.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError(
                f"messages[{i}]['parts'] must be a non-empty list"
            )
        for j, part in enumerate(parts):
            if not isinstance(part, dict):
                raise ValueError(f"messages[{i}]['parts'][{j}] must be a dict")
            ptype = part.get("type")
            if ptype not in ("text", "image", "video", "audio"):
                raise ValueError(
                    f"messages[{i}]['parts'][{j}]['type'] must be "
                    "'text', 'image', 'video', or 'audio'"
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
    if ftype == "json_schema" and not isinstance(fmt.get("schema"), dict):
        raise ValueError(
            "output['format']['schema'] must be a dict for json_schema format"
        )


# ---------------------------------------------------------------------------
# Variable substitution
# ---------------------------------------------------------------------------

class _SafeFormatMap(dict):
    """
    A dict subclass that returns ``"{key}"`` for missing keys so that
    :py:meth:`str.format_map` leaves unknown placeholders intact instead
    of raising ``KeyError``.
    """
    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


def substitute(messages: list, variables: dict) -> list:
    """
    Return a deep copy of *messages* with ``{placeholders}`` in text parts
    replaced by the matching value from *variables*.

    Unknown placeholders that have no corresponding key in *variables* are
    left intact (they are not removed or replaced with an empty string).

    Non-text parts are deep-copied unchanged.
    """
    if not variables:
        return copy.deepcopy(messages)
    mapping = _SafeFormatMap(variables)
    result = []
    for msg in messages:
        new_parts = []
        for part in msg.get("parts", []):
            if part.get("type") == "text":
                new_part = dict(part)
                new_part["text"] = part["text"].format_map(mapping)
                new_parts.append(new_part)
            else:
                new_parts.append(copy.deepcopy(part))
        new_msg = dict(msg)
        new_msg["parts"] = new_parts
        result.append(new_msg)
    return result
