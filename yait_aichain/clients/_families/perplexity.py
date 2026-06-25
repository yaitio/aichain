"""
clients._families.perplexity
=============================

Perplexity rides the OpenAI Chat Completions *format* (inherited from
OpenAIClient) but has its own *transport*: a custom auth check (a minimal
1-token completion, since there is no dedicated auth endpoint). It exposes a
public ``/v1/models`` router catalog, so ``list_models()`` reads it live and
falls back to a curated static list only if the call fails.
"""

from __future__ import annotations

from .openai import OpenAIClient
from .._errors import APIError

# Fallback only — used when the live /v1/models call fails (e.g. offline).
_KNOWN_MODELS = [
    "sonar", "sonar-pro", "sonar-reasoning",
    "sonar-reasoning-pro", "sonar-deep-research", "r1-1776",
]


class PerplexityClient(OpenAIClient):

    def list_models(self) -> list[str]:
        # Perplexity's public /v1/models is a *router catalog* — it also lists
        # third-party models it can proxy (``anthropic/…``, ``openai/…``, …).
        # Keep only Perplexity's OWN models (the ``perplexity/`` namespace) and
        # strip the prefix so the names match how the registry calls them
        # (``sonar`` etc.). Fall back to the curated static list on any failure.
        try:
            live = super().list_models()
        except Exception:
            return list(_KNOWN_MODELS)
        own = [m.split("/", 1)[1] for m in live if m.startswith("perplexity/")]
        return own or list(_KNOWN_MODELS)

    def check_auth(self) -> bool:
        # No auth-check endpoint; make a minimal 1-token completion.
        body = {"model": _KNOWN_MODELS[0],
                "messages": [{"role": "user", "content": "1"}],
                "max_tokens": 1}
        try:
            self._post(self._chat_path, body, self._auth_headers())
            return True
        except APIError as exc:
            if exc.status in (401, 403):
                return False
            raise
