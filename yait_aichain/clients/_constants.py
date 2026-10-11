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

    def increment(self, method=None, url=None, response=None, error=None,
                  _pool=None, _stacktrace=None):
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

# 5 total attempts (4 retries after the first failure), up to 2 redirects.
# backoff_factor=2 → sleeps 0 s, 2 s, 4 s, 8 s, 16 s between attempts
# (≈30 s total exposure) — enough to survive a brief provider rolling-restart
# or transient 503 storm without holding up short interactive calls too long.
#
# allowed_methods includes POST: every generative/embedding call is a POST,
# and urllib3's default method list would silently exclude them from status
# retries. POST is only retried on 429/503 — statuses providers return
# *before* running inference — so a retry cannot double-bill a generation.
# 500/502/504 may arrive after the request started processing and are
# therefore retried only for idempotent methods.
DEFAULT_RETRIES = Retry(
    total=5,
    redirect=2,
    backoff_factor=2.0,
    status_forcelist={429, 503},  # safe pre-inference statuses, incl. POST
    allowed_methods=None,         # retry all methods on the codes above
    raise_on_status=False,
)

# Idempotent requests (GET model lists, fetch/count) can additionally retry
# the riskier 5xx codes without double-charge concerns.
DEFAULT_IDEMPOTENT_RETRIES = Retry(
    total=5,
    redirect=2,
    backoff_factor=2.0,
    status_forcelist={429, 500, 502, 503, 504},
    raise_on_status=False,
)
