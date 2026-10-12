# API: Errors

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../models/models.md).

<!-- g:api -->
[`APIError`](#apierror) · [`AuthenticationError`](#authenticationerror) · [`RateLimitError`](#ratelimiterror) · [`InvalidRequestError`](#invalidrequesterror) · [`NotFoundError`](#notfounderror) · [`ServerError`](#servererror) · [`NetworkError`](#networkerror) · [`InsufficientCreditsError`](#insufficientcreditserror) · [`TruncatedResponseError`](#truncatedresponseerror) · [`InvalidStructuredOutputError`](#invalidstructuredoutputerror) · [`TaskFailedError`](#taskfailederror)

---

## `APIError`

```python
APIError(status: int, message: str, retry_after: float | None = None)
```

Base class for all provider API failures. A `RuntimeError`, so code that
caught what a tool raised before tools raised typed errors still does.

Raised when a provider returns a non-2xx HTTP status or a network-level
error occurs.  Concrete failures use the subclasses below; catching
`APIError` catches them all.

**Attributes**

- `status` (*HTTP status code (0 for network/connection errors).*)
- `message` (*Error detail from the response body or exception message.*)
- `retry_after` (*Seconds to wait before retrying, if the provider sent a*, default `None`) — `Retry-After` header; otherwise `None`.

---

## `AuthenticationError`

HTTP 401 / 403 — missing, invalid, or unauthorized API key.

---

## `RateLimitError`

HTTP 429 — too many requests.  `retry_after` may carry the delay.

---

## `InvalidRequestError`

HTTP 400 / 422 — malformed request (bad params, schema, etc.).

---

## `NotFoundError`

HTTP 404 — model or endpoint does not exist.

---

## `ServerError`

HTTP 5xx — provider-side outage or transient failure.

---

## `NetworkError`

No HTTP response at all — DNS failure, connection refused, timeout.

---

## `InsufficientCreditsError`

The account is out of credits / funds / quota — a **billing** problem, not
a credentials problem.

Providers signal this inconsistently (HTTP 402, or a 403 / 429 / 400 whose
body mentions credits / quota / budget), which is why it is detected from the
response body and surfaced as its own class: the fix is to top up the account,
not to re-check the API key. Non-transient — retrying the same account will
not add funds, so it is never retried.

---

## `TruncatedResponseError`

```python
TruncatedResponseError(
    message: str,
    finish_reason: str = '',
    output_tokens: int | None = None,
)
```

A structured response was cut off at the output-token ceiling.

It arrives as HTTP 200 carrying JSON that simply stops, so without this
class the caller sees a bare `JSONDecodeError` at some column and has to
read the client source to learn it was a length limit. The tell that it is
a ceiling and not a fluctuation: **three attempts break at the same
position**, because a constant is not noise.

Also a `ValueError`: before these classes existed both failures were
raised as one, and code catching `ValueError` must keep working.

**Attributes**

- `finish_reason` (*the provider's own word for why it stopped*, default `''`) — (`"length"`, `"MAX_TOKENS"`, `"max_tokens"`).
- `output_tokens` (*how many output tokens were produced, when the response*, default `None`) — reported it. That number *is* the ceiling that was hit,
  so it is what `max_tokens` has to be raised above.

**Parameters**

- `message` (*str*) — What happened.
- `finish_reason` (*str, optional*, default `''`) — The provider's own word for why it stopped.
- `output_tokens` (*int | None, optional*, default `None`) — How many tokens it wrote before the ceiling.

---

## `InvalidStructuredOutputError`

```python
InvalidStructuredOutputError(
    message: str,
    problems: list[str] | None = None,
    value: object = None,
)
```

A structured response came back whole and does not match the schema.

Separate from `TruncatedResponseError` on purpose: one is fixed by
raising `max_tokens`, the other by changing the schema or the prompt,
and a caller that cannot tell them apart retries the wrong one.

Also a `ValueError`, for the same backward-compatibility reason.

**Attributes**

- `problems` (*one string per violation found, in the order they were found.*, default `None`)
- `value` (*what the provider actually returned, so the caller can log it*, default `None`) — without re-parsing.

**Parameters**

- `message` (*str*) — What happened.
- `problems` (*list[str] | None, optional*, default `None`) — Where the value departs from the schema, one path each.
- `value` (*any, optional*, default `None`) — What came back.

---

## `TaskFailedError`

A provider's asynchronous job reached a terminal failure.

Raised when an async task API (e.g. DashScope image-synthesis) reports the
job FAILED / CANCELED / returned no usable result, or did not finish within
the poll budget.  Unlike a transport `ServerError`, this is **terminal**:
re-issuing the same request would submit a brand-new (billable) job, so it
is never retried and never triggers model fallback.
<!-- /g:api -->
