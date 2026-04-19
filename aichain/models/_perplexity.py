"""
models._perplexity — PerplexityModel
======================================

Covers the Perplexity Sonar model family:

  sonar-pro             Advanced search model, higher context
  sonar                 Lightweight search model
  sonar-reasoning-pro   Advanced chain-of-thought search
  sonar-deep-research   Multi-step deep research agent

Default generation parameters
------------------------------
temperature   0.2   Perplexity recommends a low temperature for search
                    and factual tasks; 0.2 keeps answers focused without
                    being completely deterministic.
max_tokens    8192  Practical default across the Sonar family.
                    sonar-deep-research can return much longer outputs;
                    raise in options when using that model.
top_p         0.9   Perplexity's recommended value for nucleus sampling.
top_k         None  Not in the Perplexity API parameter set.
cache_control False Perplexity does not support prompt caching.
reasoning     None  Not applicable; sonar-reasoning / sonar-reasoning-pro
                    use built-in chain-of-thought automatically.
                    Any value set here is silently ignored.

Recommended overrides by model
--------------------------------
sonar-pro            temperature=0.2,  max_tokens=8192
sonar                temperature=0.2,  max_tokens=8192
sonar-reasoning-pro  temperature=0.2   (CoT is automatic)
sonar-deep-research  temperature=0.1,  max_tokens=32000

Note
----
Perplexity exposes an OpenAI-compatible ``/chat/completions`` endpoint.
It also supports extra search-control parameters (``search_domain_filter``,
``search_recency_filter``, ``return_related_questions``) which are
accepted by the task layer, not stored on the model object.

Environment variable
---------------------
PERPLEXITY_API_KEY
"""

import urllib3

from ._base import Model
from ._openai import _build_openai_compat_request, _parse_openai_compat_response
from clients._perplexity import PerplexityClient
from clients._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


class PerplexityModel(Model):
    """Provider-specific model for the Perplexity AI API."""

    _ENV_KEY = "PERPLEXITY_API_KEY"

    # ── generation defaults ──────────────────────────────────────────
    _DEFAULT_TEMPERATURE:   float        = 0.2   # low = more factual / search-oriented
    _DEFAULT_MAX_TOKENS:    int          = 8192
    _DEFAULT_TOP_P:         float | None = 0.9
    _DEFAULT_TOP_K:         int   | None = None   # unsupported
    _DEFAULT_CACHE_CONTROL: bool         = False  # unsupported
    _DEFAULT_REASONING:     str   | None = None   # built-in for reasoning models

    # Perplexity has no external reasoning parameter; map is empty.
    _REASONING_MAP: dict = {}

    # ------------------------------------------------------------------

    def _build_client(self, api_key: str, client_options: dict) -> PerplexityClient:
        """
        Construct a :class:`~clients.PerplexityClient`.

        Supported *client_options* keys: ``url``, ``timeout``,
        ``retries``, ``proxy``.
        """
        return PerplexityClient(
            api_key = api_key,
            url     = client_options.get("url"),
            timeout = client_options.get("timeout", DEFAULT_TIMEOUT),
            retries = client_options.get("retries", DEFAULT_RETRIES),
            proxy   = client_options.get("proxy"),
        )

    def to_request(self, messages: list, output: dict) -> "tuple[str, dict]":
        """
        Translate universal messages → Perplexity ``POST /chat/completions``
        ``(path, body)`` pair.

        Perplexity exposes an OpenAI-compatible endpoint at a different base
        path (``/chat/completions`` without the ``/v1`` prefix); this reuses
        the shared OpenAI-compatible request builder.
        """
        return _build_openai_compat_request(
            self, messages, output, "/chat/completions"
        )

    def from_response(self, response: dict, output: dict) -> "str | dict":
        """Extract the clean result from a Perplexity chat completion response."""
        return _parse_openai_compat_response(response, output)
