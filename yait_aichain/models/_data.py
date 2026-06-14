"""
models._data
============

Loader for the provider data files (``models/providers/*.toml``) — the data
tier of the two-tier model layer.

One file per provider; each holds the provider's transport/format settings
(`[provider]`) and its models (`[models."name"]`).  Loaded once at import into
``PROVIDERS`` (provider key → data).  ``Model`` and the registry query
functions read from here.

TOML parsing uses stdlib ``tomllib`` (Python 3.11+); on 3.10 it falls back to
the optional ``tomli`` package.
"""

from __future__ import annotations

import os

try:
    import tomllib as _toml          # Python 3.11+
except ModuleNotFoundError:          # pragma: no cover
    try:
        import tomli as _toml        # Python 3.10
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "Reading provider data needs tomllib (Python 3.11+) or 'tomli' "
            "on 3.10. Install it with: pip install tomli"
        ) from exc

_DIR = os.path.join(os.path.dirname(__file__), "providers")


def _load() -> dict:
    providers: dict = {}
    for fname in sorted(os.listdir(_DIR)):
        if not fname.endswith(".toml"):
            continue
        with open(os.path.join(_DIR, fname), "rb") as fh:
            data = _toml.load(fh)
        key = data["provider"]["key"]
        providers[key] = data
    return providers


#: provider key → {"provider": {...}, "models": {name: {...}}}
PROVIDERS: dict[str, dict] = _load()


def provider_of(model_name: str) -> "str | None":
    """Return the provider key that owns *model_name*, or None."""
    for key, data in PROVIDERS.items():
        if model_name in data.get("models", {}):
            return key
    return None
