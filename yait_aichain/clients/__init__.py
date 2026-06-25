"""
clients
=======

The protocol layer: each family client owns both the wire format
(``build_request`` / ``parse_response``) and the HTTP transport (auth +
requests) for one API family.  Five family classes cover all eight providers:

  ``BaseClient``        — transport primitives + abstract format hooks
  ``OpenAIClient``      — OpenAI Chat Completions family
                          (openai, xai, kimi, deepseek)
  ``PerplexityClient``  — OpenAI-compatible; live /v1/models catalog  (perplexity)
  ``QwenClient``        — OpenAI-compatible; region-resolved URL (qwen)
  ``AnthropicClient``   — Anthropic Messages family            (anthropic)
  ``GoogleClient``      — Google Generative AI family           (google)

A client is created per provider with that provider's data dict; the model
layer (``models.Model``) picks and builds the right one.
"""

from ._base import BaseClient, APIError
from ._errors import (
    NetworkError,
    RateLimitError,
    AuthenticationError,
    InsufficientCreditsError,
    InvalidRequestError,
    NotFoundError,
    ServerError,
    TaskFailedError,
)
from ._families.openai     import OpenAIClient
from ._families.perplexity import PerplexityClient
from ._families.qwen       import QwenClient
from ._families.anthropic  import AnthropicClient
from ._families.google     import GoogleClient

__all__ = [
    "BaseClient",
    "APIError",
    "NetworkError",
    "RateLimitError",
    "AuthenticationError",
    "InsufficientCreditsError",
    "InvalidRequestError",
    "NotFoundError",
    "ServerError",
    "TaskFailedError",
    "OpenAIClient",
    "PerplexityClient",
    "QwenClient",
    "AnthropicClient",
    "GoogleClient",
]
