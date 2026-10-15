"""The protocol layer: a family client per API dialect owns both the wire format
(``build_request`` / ``parse_response``) and the HTTP transport. ``Model``
builds one per provider from that provider's data. The ``APIError`` hierarchy
is exported here too. See docs/models/models.md.
"""

from ._base import BaseClient, APIError
from ._errors import (
    NetworkError,
    RateLimitError,
    AuthenticationError,
    InsufficientCreditsError,
    InvalidRequestError,
    ContextWindowExceededError,
    NotFoundError,
    ServerError,
    TaskFailedError,
    TruncatedResponseError,
    InvalidStructuredOutputError,
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
    "TruncatedResponseError",
    "InvalidStructuredOutputError",
    "InvalidRequestError",
    "ContextWindowExceededError",
    "NotFoundError",
    "ServerError",
    "TaskFailedError",
    "OpenAIClient",
    "PerplexityClient",
    "QwenClient",
    "AnthropicClient",
    "GoogleClient",
]
