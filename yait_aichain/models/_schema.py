"""Schemas across provider families: :func:`portable_schema` rewrites a JSON
Schema for OpenAI strict mode or Google, and :func:`check_structure` is a
shallow, dependency-free check of an answer against its schema.
See docs/models/models.md, "A schema that travels".
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
    Return *schema* in the form *target* accepts; the original is untouched.

    Not lossless. A map with arbitrary keys has no strict equivalent, and
    for ``"openai"`` it raises instead of being mangled.

    Parameters
    ----------
    schema : dict
        A JSON Schema.
    target : str
        The provider family: ``"openai"`` (strict mode) or ``"google"``.
    """
    if target == "openai":
        from ..clients._families._openai_compat import _sanitize_openai_schema
        return _sanitize_openai_schema(schema)
    if target == "google":
        from ..clients._families.google import _sanitize_google_schema
        return _sanitize_google_schema(schema)
    raise ValueError(f"unknown target {target!r}; expected 'openai' or 'google'")


def check_structure(value: object, schema: object, path: str = "$") -> list[str]:
    """
    Compare *value* against *schema*, returning one message per violation.

    Empty means nothing found, not full validity: types, required and
    unexpected properties, enums and array items are checked, recursively.

    Parameters
    ----------
    value : any
        What a model returned.
    schema : dict
        The JSON Schema it was asked for.
    path : str, optional
        Where *value* sits, for the messages; ``"$"`` at the top.
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
