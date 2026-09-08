"""
models._schema
==============

Schemas travel badly between providers, and each family breaks them in its own
direction: Google forbids ``additionalProperties`` and union types, while
OpenAI's ``strict`` mode requires the exact opposite — every object closed and
every property listed in ``required``. A schema written for one family is
rejected by the other, and until this module existed every caller derived a
portable form by hand.

Two public helpers:

* :func:`portable_schema` — normalise a JSON Schema for a provider family.
* :func:`check_structure` — a dependency-free structural check of a value
  against a schema, used to tell "the provider ignored the schema" from
  "the response was cut off".

The check is deliberately shallow. A full JSON Schema validator is a
dependency, and this package has exactly one; what it catches is the failure
that actually happens in practice — a provider that silently dropped
``additionalProperties`` and returned fields nobody asked for, or omitted a
required one.
"""

from __future__ import annotations

_JSON_TYPES = {
    "object":  dict,
    "array":   list,
    "string":  str,
    "integer": int,
    "number":  (int, float),
    "boolean": bool,
    "null":    type(None),
}


def portable_schema(schema: object, target: str) -> object:
    """
    Return *schema* in the form *target* accepts.

    ``target`` is a provider family: ``"openai"`` (strict mode) or
    ``"google"``. The original is never modified.

    Neither direction is lossless, and one shape has no strict equivalent at
    all — a map with arbitrary keys. For ``"openai"`` that raises rather than
    being mangled, because the provider's own message points at the wire
    format instead of at the schema the caller wrote.
    """
    if target == "openai":
        from ..clients._families._openai_compat import _sanitize_openai_schema
        return _sanitize_openai_schema(schema)
    if target == "google":
        from ..clients._families.google import _sanitize_google_schema
        return _sanitize_google_schema(schema)
    raise ValueError(f"unknown target {target!r}; expected 'openai' or 'google'")


def check_structure(value: object, schema: object, path: str = "$") -> "list[str]":
    """
    Compare *value* against *schema*, returning one message per violation.

    An empty list means nothing was found — not that the value is valid under
    the full specification. Checked: declared types, required properties,
    unexpected properties when ``additionalProperties`` is false, enum
    membership, and array element types. Recurses through ``properties`` and
    ``items``.
    """
    problems: list[str] = []
    if not isinstance(schema, dict):
        return problems

    types = schema.get("type")
    if isinstance(types, str):
        types = [types]
    if isinstance(types, list):
        allowed = tuple(_JSON_TYPES[t] for t in types if t in _JSON_TYPES)
        # bool is a subclass of int in Python; a boolean where an integer was
        # declared would otherwise pass unnoticed.
        if allowed and (not isinstance(value, allowed)
                        or (isinstance(value, bool) and "boolean" not in types)):
            problems.append(
                f"{path}: expected {'/'.join(types)}, got "
                f"{type(value).__name__}")
            return problems

    if (enum := schema.get("enum")) is not None and value not in enum:
        problems.append(f"{path}: {value!r} is not one of {enum!r}")

    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                problems.append(f"{path}: required property {name!r} is missing")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in props:
                    problems.append(
                        f"{path}: unexpected property {name!r} — the schema "
                        "closes this object")
        for name, sub in props.items():
            if name in value:
                problems.extend(check_structure(value[name], sub,
                                                f"{path}.{name}"))

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, element in enumerate(value):
            problems.extend(check_structure(element, schema["items"],
                                            f"{path}[{i}]"))
    return problems
