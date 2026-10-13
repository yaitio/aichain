import urllib3
from urllib3.exceptions import MaxRetryError


class Retry(urllib3.Retry):
    """urllib3's retry policy, except that an empty balance is not retried.

    A 429 means two things. Usually it is a rate limit, and waiting helps.
    Some providers — Z.ai among them — also answer 429 when the account has
    no money, and waiting does not help: the same request fails five times,
    with backoff, before the library reads the body and sees billing. That
    cost a minute a call. On a 429 the body is read here first (an error body
    is small, and urllib3 keeps it for the caller), and a billing body ends
    the retries at once, so the caller gets `InsufficientCreditsError` on
    the first answer.
    """

    #: The longest ``Retry-After`` slept through here. A server asking for
    #: longer — an hour is not unusual on a daily quota — is answered at once
    #: with its 429, so the caller gets `RateLimitError.retry_after` and
    #: decides, instead of a thread that sleeps for six hours.
    MAX_RETRY_AFTER = 60.0

    def increment(self, method=None, url=None, response=None, error=None,
                  _pool=None, _stacktrace=None):
        if response is not None and self.respect_retry_after_header:
            wait = self.get_retry_after(response)
            if wait is not None and wait > self.MAX_RETRY_AFTER:
                raise MaxRetryError(_pool, url, f"Retry-After {wait:.0f}s is past the ceiling")
        if response is not None and response.status == 429:
            from ._errors import _looks_like_billing
            try:
                body = response.data.decode("utf-8", errors="replace")
            except Exception:
                body = ""
            if _looks_like_billing(body):
                raise MaxRetryError(_pool, url, "the account has no balance; retrying cannot help")
        return super().increment(method, url, response=response, error=error,
                                 _pool=_pool, _stacktrace=_stacktrace)


# ---------------------------------------------------------------------------
# Network defaults — can be overridden per client instance
# ---------------------------------------------------------------------------

# 10 s to establish a connection; up to 10 min to read a response.
# Long read timeout covers slow model inference (e.g. large-context requests).
DEFAULT_TIMEOUT = urllib3.Timeout(connect=10.0, read=600.0)

# For a tool's call that does not generate text — embeddings, reranking, a
# vector store: a minute is generous, and ten would hold a Lambda for nothing.
TOOL_TIMEOUT = urllib3.Timeout(connect=10.0, read=60.0)

# The statuses a provider returns *before* it runs anything: a rate limit, a
# service briefly unavailable, Anthropic's 529 overloaded. Retrying them cannot
# bill a generation twice. 500/502/504 can arrive after the work started, and
# are left to the caller's own retry (`Skill(max_retries=…)`), so a POST has
# one layer that may repeat it, not two that multiply.
PRE_INFERENCE = frozenset({429, 503, 529})

# Three attempts. `read=False`: a read that failed — a balancer cutting a long
# generation, a 600 s timeout — means the request reached the model and may
# have been paid for, so it is never sent again from here; connection errors,
# which fail before anything is sent, still are. Backoff 1 s then 2 s, with up
# to a second of jitter so a fleet of Lambdas does not retry in step.
# allowed_methods=None so POST — every generative call — is covered for the
# statuses above, which urllib3's default method list would exclude.
# redirect=False: a provider's API does not redirect, and urllib3 following one
# to another host drops Authorization but sends `x-api-key`,
# `x-goog-api-key` and the body on — so a 3xx is returned, and raises.
DEFAULT_RETRIES = Retry(
    total=3,
    read=False,
    redirect=False,
    backoff_factor=1.0,
    backoff_jitter=1.0,
    status_forcelist=PRE_INFERENCE,
    allowed_methods=None,
    raise_on_status=False,
)

# Idempotent requests (GET model lists, fetch/count) can additionally retry
# the 5xx codes that may follow the start of the work, without double-charge
# concerns.
DEFAULT_IDEMPOTENT_RETRIES = Retry(
    total=3,
    redirect=False,
    backoff_factor=1.0,
    backoff_jitter=1.0,
    status_forcelist=PRE_INFERENCE | {500, 502, 504},
    raise_on_status=False,
)
