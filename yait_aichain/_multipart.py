"""
A request body that goes out as ``multipart/form-data``.

Image edits and transcriptions upload files, so their body is a list of form
fields rather than a JSON object. It is its own type so that every layer which
treats a body as a mapping — the ``extra`` merge, the adaptation check, the
JSON send — can tell it apart with ``isinstance`` rather than by a magic key
inside a dict that any of them might also read as a field.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Multipart:
    """Form fields as ``(name, value)`` pairs; a file's value is ``(filename, bytes, mime)``."""

    fields: list
