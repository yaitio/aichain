"""
clients._errors
===============

Exception hierarchy for provider API failures.

``APIError`` is the base class — catching it still catches every provider
failure, so existing ``except APIError`` code keeps working unchanged.  The
subclasses let callers react to a *specific* failure mode (retry on rate
limits, re-auth on 401, surface a bad request, etc.) and are the foundation
for the model fallback chain (block 1.2-F).

Mapping (HTTP status → class):

    429              → RateLimitError       (carries ``retry_after`` seconds)
    401, 403         → AuthenticationError
    400, 422         → InvalidRequestError
    404              → NotFoundError
    5xx              → ServerError
    0 (no response)  → NetworkError
    other            → APIError
"""

from __future__ import annotations


class APIError(Exception):
    """
    Base class for all provider API failures.

    Raised when a provider returns a non-2xx HTTP status or a network-level
    error occurs.  Concrete failures use the subclasses below; catching
    ``APIError`` catches them all.

    Attributes
    ----------
    status  : HTTP status code (0 for network/connection errors).
    message : Error detail from the response body or exception message.
    retry_after : Seconds to wait before retrying, if the provider sent a
                  ``Retry-After`` header; otherwise ``None``.
    """

    def __init__(
        self,
        status: int,
        message: str,
        retry_after: "float | None" = None,
    ) -> None:
        self.status = status
        self.message = message
        self.retry_after = retry_after
        super().__init__(f"[HTTP {status}] {message}")


class NetworkError(APIError):
    """No HTTP response at all — DNS failure, connection refused, timeout."""


class RateLimitError(APIError):
    """HTTP 429 — too many requests.  ``retry_after`` may carry the delay."""


class AuthenticationError(APIError):
    """HTTP 401 / 403 — missing, invalid, or unauthorized API key."""


class InvalidRequestError(APIError):
    """HTTP 400 / 422 — malformed request (bad params, schema, etc.)."""


class NotFoundError(APIError):
    """HTTP 404 — model or endpoint does not exist."""


class ServerError(APIError):
    """HTTP 5xx — provider-side outage or transient failure."""


class TaskFailedError(APIError):
    """
    A provider's asynchronous job reached a terminal failure.

    Raised when an async task API (e.g. DashScope image-synthesis) reports the
    job FAILED / CANCELED / returned no usable result, or did not finish within
    the poll budget.  Unlike a transport ``ServerError``, this is **terminal**:
    re-issuing the same request would submit a brand-new (billable) job, so it
    is never retried and never triggers model fallback.
    """


def _parse_retry_after(headers) -> "float | None":
    """Extract a numeric ``Retry-After`` (seconds) from response headers."""
    if not headers:
        return None
    try:
        value = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        # HTTP-date form is not handled — callers fall back to backoff.
        return None


def error_from_status(
    status: int,
    message: str,
    headers=None,
) -> APIError:
    """
    Build the most specific ``APIError`` subclass for *status*.

    ``headers`` (optional) is the response header mapping, used to read
    ``Retry-After`` for rate-limit errors.
    """
    if status == 0:
        return NetworkError(status, message)
    if status == 429:
        return RateLimitError(status, message, _parse_retry_after(headers))
    if status in (401, 403):
        return AuthenticationError(status, message)
    if status in (400, 422):
        return InvalidRequestError(status, message)
    if status == 404:
        return NotFoundError(status, message)
    if 500 <= status < 600:
        return ServerError(status, message)
    return APIError(status, message)
