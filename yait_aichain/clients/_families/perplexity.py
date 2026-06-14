"""
clients._families.perplexity
=============================

Perplexity rides the OpenAI Chat Completions *format* (inherited from
OpenAIClient) but has its own *transport*: no /models discovery endpoint, and
a custom auth check (a minimal 1-token completion). So it overrides only the
transport methods.
"""

from __future__ import annotations

from .openai import OpenAIClient
from .._errors import APIError

_KNOWN_MODELS = [
    "sonar", "sonar-pro", "sonar-reasoning",
    "sonar-reasoning-pro", "sonar-deep-research", "r1-1776",
]


class PerplexityClient(OpenAIClient):

    def list_models(self) -> list[str]:
        # Perplexity has no /models discovery endpoint — curated static list.
        return list(_KNOWN_MODELS)

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
