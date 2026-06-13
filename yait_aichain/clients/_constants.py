import urllib3

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
DEFAULT_RETRIES = urllib3.Retry(
    total=5,
    redirect=2,
    backoff_factor=2.0,
    status_forcelist={429, 503},  # safe pre-inference statuses, incl. POST
    allowed_methods=None,         # retry all methods on the codes above
    raise_on_status=False,
)

# Idempotent requests (GET model lists, fetch/count) can additionally retry
# the riskier 5xx codes without double-charge concerns.
DEFAULT_IDEMPOTENT_RETRIES = urllib3.Retry(
    total=5,
    redirect=2,
    backoff_factor=2.0,
    status_forcelist={429, 500, 502, 503, 504},
    raise_on_status=False,
)
