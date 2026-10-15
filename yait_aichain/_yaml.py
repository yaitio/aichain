"""
Saved files: written whole or not at all, read back only if they are the shape
they claim.

Tuples are written as lists so ``safe_load`` reads them, a write goes through a
temporary file so an interruption leaves no half file, and a file that holds no
mapping is refused at read rather than failing later.
"""

from __future__ import annotations

import os
import tempfile

from ._require import require


def _plain(value):
    """Tuples as lists, so the file is plain YAML any reader takes."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def write(path: str, data: dict, who: str) -> None:
    """Write *data* to *path* as plain YAML, atomically."""
    yaml = require("yaml", package="pyyaml", who=who, extra="yaml")
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(_plain(data), fh, allow_unicode=True, sort_keys=False,
                           default_flow_style=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read(path: str, who: str, required: tuple) -> dict:
    """Read *path*, and refuse it unless it is a mapping with *required* keys."""
    yaml = require("yaml", package="pyyaml", who=who, extra="yaml")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path!r} is not a saved {who.split('.')[0]} — it holds no mapping")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path!r} is not a saved {who.split('.')[0]} — no {missing}")
    return data
